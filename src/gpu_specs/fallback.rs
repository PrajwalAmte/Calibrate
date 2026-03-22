/// Baked-in GPU spec table, compiled into the binary via `include_str!`.
///
/// This is the last-resort fallback when the network is unavailable and the
/// local cache is missing or corrupt.  It covers the 30 most common GPUs
/// seen in ML training workloads.
pub struct FallbackRepository;

static FALLBACK_DATA: &str = include_str!("../../assets/fallback_specs.json");

impl crate::gpu_specs::SpecsRepository for FallbackRepository {
    fn get_by_name(&self, name: &str) -> Option<crate::gpu_specs::GpuSpec> {
        let specs: Vec<crate::gpu_specs::GpuSpec> =
            serde_json::from_str(FALLBACK_DATA).ok()?;

        let lower = name.to_lowercase();
        // Fuzzy match: check if the spec name is a substring of the NVML name
        // (NVML often prepends "NVIDIA " which may not be in our table).
        specs
            .into_iter()
            .find(|s| lower.contains(&s.name.to_lowercase()) || s.name.to_lowercase().contains(&lower))
    }
}
