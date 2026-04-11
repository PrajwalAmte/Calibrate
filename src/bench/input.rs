use std::path::Path;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng}; // SeedableRng is used by SmallRng::seed_from_u64

/// Model format inferred from the file extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Onnx,
    Gguf,
    TorchScript,
    SafeTensors,
    Unknown,
}

impl ModelFormat {
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("onnx") => Self::Onnx,
            Some("gguf") => Self::Gguf,
            Some("pt") | Some("pth") | Some("torchscript") => Self::TorchScript,
            Some("safetensors") => Self::SafeTensors,
            _ => Self::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Onnx => "ONNX",
            Self::Gguf => "GGUF",
            Self::TorchScript => "TorchScript",
            Self::SafeTensors => "SafeTensors",
            Self::Unknown => "Unknown",
        }
    }
}

/// A fixed random input reused across all runtimes and all iterations.
///
/// Using the same input eliminates variability introduced by different input
/// values, ensuring that runtime comparisons are apples-to-apples.
#[derive(Debug, Clone)]
pub struct BenchInput {
    /// Flat f32 data in row-major order.
    pub data: Vec<f32>,
    /// Shape of the input tensor (e.g. [batch, seq_len] or [batch, C, H, W]).
    pub shape: Vec<usize>,
}

impl BenchInput {
    /// Generate a deterministic random input for the given shape.
    ///
    /// The seed is fixed so the same input is reproduced across CLI
    /// invocations and across all runtimes within a single run.
    pub fn generate(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        let mut rng = SmallRng::seed_from_u64(0xCAFE_BABE_DEAD_BEEF);
        let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    /// Return a default input shape for the given format and batch size.
    ///
    /// These are conservative fallback shapes used when a runtime cannot
    /// extract the real input shape from the model file. Actual runtimes
    /// should replace these with shapes parsed from the model.
    pub fn default_shape_for_format(fmt: ModelFormat, batch_size: u32) -> Vec<usize> {
        let b = batch_size as usize;
        match fmt {
            // Language models: [batch, sequence_length]
            ModelFormat::Gguf => vec![b, 512],
            // Vision models: [batch, channels, height, width]
            ModelFormat::Onnx => vec![b, 3, 224, 224],
            ModelFormat::TorchScript => vec![b, 3, 224, 224],
            // General embeddings: [batch, hidden_size]
            ModelFormat::SafeTensors => vec![b, 512],
            ModelFormat::Unknown => vec![b, 512],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_is_deterministic() {
        let a = BenchInput::generate(&[2, 512]);
        let b = BenchInput::generate(&[2, 512]);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn generate_correct_length() {
        let input = BenchInput::generate(&[4, 3, 224, 224]);
        assert_eq!(input.data.len(), 4 * 3 * 224 * 224);
    }

    #[test]
    fn format_from_path() {
        let cases = [
            ("model.onnx", ModelFormat::Onnx),
            ("weights.gguf", ModelFormat::Gguf),
            ("model.pt", ModelFormat::TorchScript),
            ("model.safetensors", ModelFormat::SafeTensors),
            ("model.bin", ModelFormat::Unknown),
        ];
        for (name, expected) in cases {
            assert_eq!(ModelFormat::from_path(Path::new(name)), expected, "{name}");
        }
    }
}
