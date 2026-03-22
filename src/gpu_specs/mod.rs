pub mod client;
pub mod fallback;

/// Specification for a single GPU model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuSpec {
    /// Canonical GPU name as reported by NVML (e.g. "NVIDIA GeForce RTX 3090").
    pub name: String,
    /// Peak BF16 TFLOPS (used for MFU denominator).
    pub bf16_tflops: f64,
    /// Peak FP32 TFLOPS.
    pub fp32_tflops: f64,
    /// VRAM in GiB.
    pub vram_gib: u32,
    /// Boost clock in MHz (for normalization reference).
    pub boost_clock_mhz: u32,
}

/// Port: anything that can resolve a GPU name to its spec.
pub trait SpecsRepository: Send + Sync {
    /// Look up a GPU spec by its NVML device name.
    ///
    /// Returns `None` when the name is not in the database so callers can
    /// decide whether to degrade gracefully or return an error.
    fn get_by_name(&self, name: &str) -> Option<GpuSpec>;
}

/// Resolve a spec using the remote → cache → baked-in fallback chain.
///
/// This is the production resolver used by `WatchCommand`.  It is called
/// once at startup (blocking) on a `spawn_blocking` thread.
pub fn resolve(name: &str) -> GpuSpec {
    // 1. Try remote/cached repository.
    if let Some(spec) = client::HttpSpecsClient::load().get_by_name(name) {
        return spec;
    }
    // 2. Fall back to baked-in table.
    if let Some(spec) = fallback::FallbackRepository.get_by_name(name) {
        tracing::debug!("Using fallback spec for '{name}'");
        return spec;
    }
    // 3. Unknown GPU — synthesize a placeholder so the tool still runs.
    tracing::warn!("GPU '{name}' not in spec database; MFU will be approximate");
    GpuSpec {
        name: name.to_string(),
        bf16_tflops: 20.0, // conservative placeholder
        fp32_tflops: 20.0,
        vram_gib: 16,
        boost_clock_mhz: 1800,
    }
}
