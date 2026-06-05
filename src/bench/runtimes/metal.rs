use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use metal::{
    CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize,
};

use crate::bench::input::{BenchInput, ModelFormat};
use crate::bench::runtime::{Runtime, RuntimeDescriptor};

// ── MSL kernel ────────────────────────────────────────────────────────────────

/// Single-dispatch GEMV kernel: each thread computes one output element.
///
/// `dims` packs {M, K}: `M` rows in the weight matrix, `K` inner dimension.
/// Input is a flat f32 vector of length K; output is f32 of length M.
const MSL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void bench_matmul(
    device const float* weights [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    constant uint2&     dims    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dims.x) return;
    float sum = 0.0f;
    uint offset = gid * dims.y;
    for (uint k = 0; k < dims.y; ++k)
        sum += weights[offset + k] * input[k];
    output[gid] = sum;
}
"#;

// ── MetalRuntime ──────────────────────────────────────────────────────────────

/// Inference runtime that dispatches a GEMV (matrix × vector) operation via
/// Apple's Metal GPU compute API.
///
/// Mirrors `CandleRuntime` in purpose but exercises the Apple GPU rather than
/// the CPU.  Uses the weight data from a SafeTensors checkpoint as the matrix
/// and the bench harness input as the vector, running a single dispatch per
/// `infer()` call.
///
/// `vram_used_mib` is tracked by the bench harness via IOKit (see
/// `bench::memory`); this struct only manages Metal objects.
pub struct MetalRuntime {
    device: Option<Device>,
    queue: Option<metal::CommandQueue>,
    pipeline: Option<ComputePipelineState>,
    /// Weight buffer: the first tensor from the SafeTensors file, re-cast to
    /// f32.  Shape [M × K] — M rows, K columns.
    weight_buf: Option<metal::Buffer>,
    /// Input vector buffer (K elements).  Allocated on first `infer()` call
    /// and reused on subsequent calls if the size is unchanged.
    input_buf: Option<metal::Buffer>,
    /// Output buffer (M elements).
    output_buf: Option<metal::Buffer>,
    /// Weight matrix dimensions cached from `load()`.
    dims: [u32; 2],
    /// Anti-dead-code-elimination sink: sum of last output.
    _sink: f32,
}

impl MetalRuntime {
    pub fn new() -> Self {
        Self {
            device: None,
            queue: None,
            pipeline: None,
            weight_buf: None,
            input_buf: None,
            output_buf: None,
            dims: [0, 0],
            _sink: 0.0,
        }
    }

    pub fn descriptor() -> RuntimeDescriptor {
        RuntimeDescriptor {
            name: "metal",
            is_available: || Device::system_default().is_some(),
            is_compatible: |fmt| matches!(fmt, ModelFormat::SafeTensors),
            create: || Box::new(MetalRuntime::new()),
        }
    }

    /// Compile the MSL kernel and create the pipeline once.
    fn build_pipeline(device: &Device) -> Result<ComputePipelineState> {
        let library = device
            .new_library_with_source(MSL_SRC, &CompileOptions::new())
            .map_err(|e| anyhow::anyhow!("MSL compile error: {e}"))?;
        let function = library
            .get_function("bench_matmul", None)
            .map_err(|e| anyhow::anyhow!("Metal function not found: {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow::anyhow!("Pipeline state creation failed: {e}"))
    }

    /// Encode and dispatch one GEMV call on the command buffer.
    fn encode_dispatch(
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        weight_buf: &metal::Buffer,
        input_buf: &metal::Buffer,
        output_buf: &metal::Buffer,
        dims: [u32; 2],
    ) {
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(weight_buf), 0);
        encoder.set_buffer(1, Some(input_buf), 0);
        encoder.set_buffer(2, Some(output_buf), 0);

        let dims_bytes: [u8; 8] = unsafe { std::mem::transmute(dims) };
        encoder.set_bytes(3, 8, dims_bytes.as_ptr() as *const _);

        let m = dims[0] as u64;
        let max_tpg = pipeline.max_total_threads_per_threadgroup();
        let tpg = max_tpg.min(256).min(m);

        let threads_per_grid = MTLSize {
            width: m,
            height: 1,
            depth: 1,
        };
        let threads_per_group = MTLSize {
            width: tpg,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_group);
        encoder.end_encoding();
    }
}

impl Runtime for MetalRuntime {
    fn name(&self) -> &str {
        "metal"
    }

    fn load(&mut self, model_path: &Path) -> Result<Duration> {
        let start = Instant::now();

        // ── Validate SafeTensors header ──────────────────────────────────
        let bytes = std::fs::read(model_path)
            .with_context(|| format!("cannot read {}", model_path.display()))?;
        if bytes.len() < 8 {
            bail!("file too small to be a valid SafeTensors checkpoint");
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
        if bytes.len() < 8 + header_len {
            bail!("SafeTensors header is truncated");
        }

        // ── Parse the first tensor shape to determine M, K ───────────────
        let header_json = &bytes[8..8 + header_len];
        let (m, k) = parse_first_tensor_shape(header_json)?;
        self.dims = [m, k];

        // ── Initialise Metal ─────────────────────────────────────────────
        let device = Device::system_default()
            .context("No Metal device found — is this macOS with Apple/AMD GPU?")?;

        let pipeline = Self::build_pipeline(&device)?;
        let queue = device.new_command_queue();

        // ── Allocate weight buffer from SafeTensors data ─────────────────
        // The data section starts at byte offset 8 + header_len.
        // We take at most M*K*4 bytes; if the file has fewer, we zero-pad.
        let data_offset = 8 + header_len;
        let weight_byte_len = (m as usize) * (k as usize) * 4;
        let weight_data = if bytes.len() >= data_offset + weight_byte_len {
            bytes[data_offset..data_offset + weight_byte_len].to_vec()
        } else {
            // Pad to the required size with zeros.
            let mut padded = bytes[data_offset..].to_vec();
            padded.resize(weight_byte_len, 0);
            padded
        };

        let weight_buf = device.new_buffer_with_data(
            weight_data.as_ptr() as *const _,
            weight_byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Output buffer: M f32 values.
        let output_buf = device.new_buffer(
            (m as usize * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        self.device = Some(device);
        self.queue = Some(queue);
        self.pipeline = Some(pipeline);
        self.weight_buf = Some(weight_buf);
        self.output_buf = Some(output_buf);

        Ok(start.elapsed())
    }

    fn infer(&mut self, input: &BenchInput) -> Result<()> {
        let device = self.device.as_ref().context("MetalRuntime: not loaded")?;
        let queue = self.queue.as_ref().context("MetalRuntime: not loaded")?;
        let pipeline = self.pipeline.as_ref().context("MetalRuntime: not loaded")?;
        let weight_buf = self
            .weight_buf
            .as_ref()
            .context("MetalRuntime: not loaded")?;
        let output_buf = self
            .output_buf
            .as_ref()
            .context("MetalRuntime: not loaded")?;

        let k = self.dims[1] as usize;

        // Build or refresh input buffer.
        let input_data: Vec<f32> = if input.data.len() >= k {
            input.data[..k].to_vec()
        } else {
            let mut padded = input.data.clone();
            padded.resize(k, 0.0);
            padded
        };

        let input_byte_len = (k * 4) as u64;

        match &self.input_buf {
            Some(b) if b.length() == input_byte_len => {
                // Reuse existing buffer — just overwrite the contents.
                let ptr = b.contents() as *mut f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(input_data.as_ptr(), ptr, k);
                }
            }
            _ => {
                self.input_buf = Some(device.new_buffer_with_data(
                    input_data.as_ptr() as *const _,
                    input_byte_len,
                    MTLResourceOptions::StorageModeShared,
                ));
            }
        }

        let input_buf = self.input_buf.as_ref().unwrap();

        // Encode and dispatch.
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        Self::encode_dispatch(
            encoder, pipeline, weight_buf, input_buf, output_buf, self.dims,
        );
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read one value from output to prevent dead-code elimination.
        let out_ptr = output_buf.contents() as *const f32;
        self._sink = unsafe { *out_ptr };

        Ok(())
    }

    fn unload(&mut self) {
        self.weight_buf = None;
        self.input_buf = None;
        self.output_buf = None;
        self.pipeline = None;
        self.queue = None;
        self.device = None;
        self.dims = [0, 0];
        self._sink = 0.0;
    }
}

// ── SafeTensors header parsing ────────────────────────────────────────────────

/// Extract `[M, K]` from the first tensor entry in the SafeTensors header JSON.
///
/// The header is a JSON object: `{"tensor_name": {"dtype": "F32", "shape": [M, K], ...}, ...}`.
/// We find the first non-`__metadata__` key and read its `shape` array.
///
/// Falls back to `[512, 512]` if parsing fails so the runtime can still run
/// (the weight data is treated as a 512×512 f32 matrix).
fn parse_first_tensor_shape(header_json: &[u8]) -> Result<(u32, u32)> {
    let text = std::str::from_utf8(header_json)?;
    let value: serde_json::Value = serde_json::from_str(text)?;

    let obj = value
        .as_object()
        .context("SafeTensors header is not a JSON object")?;

    for (key, entry) in obj.iter() {
        if key == "__metadata__" {
            continue;
        }
        if let Some(shape_arr) = entry.get("shape").and_then(|s| s.as_array()) {
            if shape_arr.len() >= 2 {
                let m = shape_arr[0].as_u64().unwrap_or(512) as u32;
                let k = shape_arr[shape_arr.len() - 1].as_u64().unwrap_or(512) as u32;
                let m = m.max(1);
                let k = k.max(1);
                return Ok((m, k));
            }
            if shape_arr.len() == 1 {
                let n = shape_arr[0].as_u64().unwrap_or(512) as u32;
                let side = ((n as f64).sqrt() as u32).max(1);
                return Ok((side, side));
            }
        }
    }

    // Fallback: treat the weight data as a 512×512 matrix.
    Ok((512, 512))
}
