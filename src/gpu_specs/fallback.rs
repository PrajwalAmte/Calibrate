/// Baked-in GPU spec table, compiled into the binary via `include_str!`.
///
/// This is the last-resort fallback when the network is unavailable and the
/// local cache is missing or corrupt.  It covers the 30 most common GPUs
/// seen in ML training workloads.
pub struct FallbackRepository;

/// JSON is embedded at compile time — zero runtime I/O, zero allocation until
/// the first lookup.
pub(crate) static FALLBACK_DATA: &str = include_str!("../../assets/fallback_specs.json");

impl crate::gpu_specs::SpecsRepository for FallbackRepository {
    fn get_by_name(&self, name: &str) -> Option<crate::gpu_specs::GpuSpec> {
        // Parse once per lookup.  Because the binary is small (< 4 KB of JSON)
        // and lookups happen only at startup, the cost is negligible.
        let specs: Vec<crate::gpu_specs::GpuSpec> = serde_json::from_str(FALLBACK_DATA).ok()?;
        crate::gpu_specs::find_best_match(&specs, name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_specs::SpecsRepository;

    #[test]
    fn json_parses_without_error() {
        let specs: Vec<crate::gpu_specs::GpuSpec> =
            serde_json::from_str(FALLBACK_DATA).expect("fallback_specs.json must be valid JSON");
        assert!(
            specs.len() >= 40,
            "expected at least 40 GPU specs, got {}",
            specs.len()
        );
    }

    #[test]
    fn all_specs_have_positive_tflops() {
        let specs: Vec<crate::gpu_specs::GpuSpec> = serde_json::from_str(FALLBACK_DATA).unwrap();
        for spec in &specs {
            assert!(
                spec.bf16_tflops > 0.0,
                "spec '{}' has non-positive bf16_tflops",
                spec.name
            );
        }
    }

    #[test]
    fn resolves_rtx_3090_exact_name() {
        let spec = FallbackRepository.get_by_name("RTX 3090").unwrap();
        assert!((spec.bf16_tflops - 35.6).abs() < 0.1);
        assert_eq!(spec.vram_gib, 24);
    }

    #[test]
    fn network_unavailable_falls_back_cleanly() {
        // Simulate what crate::gpu_specs::resolve does when the network is
        // down: HttpSpecsClient.get_by_name returns None, so FallbackRepository
        // is the only source.  Verify it resolves known names independently.
        let result = FallbackRepository.get_by_name("NVIDIA GeForce RTX 4090");
        assert!(
            result.is_some(),
            "should resolve RTX 4090 from baked-in table"
        );
    }

    #[test]
    fn v100_sxm2_matches_hyphenated_nvml_name() {
        let spec = FallbackRepository
            .get_by_name("NVIDIA Tesla V100-SXM2-32GB")
            .unwrap();
        assert!(
            spec.name.contains("V100"),
            "expected V100, got {}",
            spec.name
        );
        assert!(
            spec.name.contains("SXM"),
            "expected SXM variant, got {}",
            spec.name
        );
    }
}
