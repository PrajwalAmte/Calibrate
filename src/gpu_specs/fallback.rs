//! Built-in GPU spec table, compiled into the binary.

use crate::gpu_specs::GpuSpec;

pub(crate) static FALLBACK_DATA: &str = include_str!("../../assets/fallback_specs.json");

/// Parse the embedded table. The JSON is validated by tests, so a parse
/// failure here is a build defect.
pub fn builtin_specs() -> Vec<GpuSpec> {
    serde_json::from_str(FALLBACK_DATA).expect("assets/fallback_specs.json is valid")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_entry_is_valid_and_unique() {
        let specs = builtin_specs();
        assert!(specs.len() >= 50, "got {}", specs.len());
        let mut names = HashSet::new();
        for s in &specs {
            assert!(s.is_valid(), "invalid entry {s:?}");
            assert!(names.insert(s.name.to_lowercase()), "duplicate {}", s.name);
            assert!(
                s.bf16_tflops >= s.fp32_tflops * 0.99 || s.name.starts_with("GTX"),
                "{}: tensor rate below FP32 rate looks like a data error",
                s.name
            );
        }
    }

    #[test]
    fn geforce_ampere_uses_tensor_rate() {
        let specs = builtin_specs();
        let r3090 = specs.iter().find(|s| s.name == "RTX 3090").unwrap();
        assert!(
            (r3090.bf16_tflops - 71.2).abs() < 0.5,
            "RTX 3090 dense BF16 (FP32 acc) is ~71 TFLOPS, got {}",
            r3090.bf16_tflops
        );
    }

    #[test]
    fn datacenter_reference_values() {
        let specs = builtin_specs();
        let get = |n: &str| specs.iter().find(|s| s.name == n).unwrap().bf16_tflops;
        assert_eq!(get("A100 SXM"), 312.0);
        assert_eq!(get("H100 SXM"), 989.0);
        assert_eq!(get("L40S"), 362.0);
    }
}
