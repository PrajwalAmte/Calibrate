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
    fn get_by_name(&self, name: &str) -> Option<GpuSpec>;
}

// ── Name matching ────────────────────────────────────────────────────────────

/// Normalise an NVML device name ready for matching against spec-DB keys.
///
/// Strips vendor and product-line prefixes, replaces hyphens, and drops
/// memory-size tokens so the remaining tokens can be matched against short
/// canonical keys such as "RTX 3090" or "A100 SXM".
///
/// # Examples
/// ```text
/// "NVIDIA GeForce RTX 3090"  →  "rtx 3090"
/// "NVIDIA A100-SXM4-80GB"    →  "a100 sxm4"
/// "NVIDIA A100-PCIE-40GB"    →  "a100 pcie"
/// "Tesla T4"                 →  "t4"
/// "NVIDIA Tesla V100-SXM2-32GB" → "v100 sxm2"
/// ```
pub fn normalize_for_match(name: &str) -> String {
    let lower = name.to_lowercase();
    // Strip vendor prefixes in order of most-to-least specific so
    // "NVIDIA Tesla T4" → strip "nvidia " → "tesla t4" → strip "tesla " → "t4".
    let s = lower.strip_prefix("nvidia ").unwrap_or(&lower);
    let s = s.strip_prefix("geforce ").unwrap_or(s);
    let s = s.strip_prefix("tesla ").unwrap_or(s);
    // Normalise separators, then filter memory-size tokens like "80gb", "40gb".
    s.replace('-', " ")
        .split_whitespace()
        .filter(|t| !is_memory_size_token(t))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Score how well `spec_name` matches `normalized_query`.
///
/// All tokens in the normalised spec name must prefix-match at least one token
/// in `normalized_query`.  A score of 0 means no match.  Higher scores indicate
/// more-specific matches, allowing "A10G" to beat "A10" when the query is "A10G".
pub fn match_score(spec_name: &str, normalized_query: &str) -> usize {
    let spec_lower = spec_name.to_lowercase();
    let query_tokens: Vec<&str> = normalized_query.split_whitespace().collect();

    let mut total = 0usize;
    for spec_token in spec_lower.split_whitespace() {
        // Find the query token that prefix-matches this spec token; add the
        // number of characters matched so longer (more-specific) specs win.
        let best = query_tokens
            .iter()
            .filter(|qt| qt.starts_with(spec_token))
            .map(|_| spec_token.len())
            .max();
        match best {
            Some(chars) => total += chars,
            None => return 0, // all spec tokens must match; any miss → no match
        }
    }
    total
}

/// Return the best-matching spec from `specs` for the given NVML device name.
///
/// Returns `None` when no spec scores above zero — i.e. the GPU is entirely
/// unknown to the database.
pub fn find_best_match<'a>(specs: &'a [GpuSpec], query: &str) -> Option<&'a GpuSpec> {
    let normalised = normalize_for_match(query);
    specs
        .iter()
        .filter_map(|spec| {
            let score = match_score(&spec.name, &normalised);
            if score > 0 { Some((spec, score)) } else { None }
        })
        .max_by_key(|(_, score)| *score)
        .map(|(spec, _)| spec)
}

fn is_memory_size_token(token: &str) -> bool {
    token.ends_with("gb") && token[..token.len() - 2].parse::<u32>().is_ok()
}

// ── Production resolver ──────────────────────────────────────────────────────

/// Resolve a spec using the remote → cache → baked-in fallback chain.
///
/// Called once at startup via `tokio::task::spawn_blocking`.
pub fn resolve(name: &str) -> GpuSpec {
    if let Some(spec) = client::HttpSpecsClient::load().get_by_name(name) {
        return spec;
    }
    if let Some(spec) = fallback::FallbackRepository.get_by_name(name) {
        tracing::debug!("Using fallback spec for '{name}'");
        return spec;
    }
    tracing::warn!("GPU '{name}' not in spec database; MFU will be approximate");
    GpuSpec {
        name: name.to_string(),
        bf16_tflops: 20.0,
        fp32_tflops: 20.0,
        vram_gib: 16,
        boost_clock_mhz: 1800,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_specs::fallback::FallbackRepository;

    // ── normalize_for_match ──────────────────────────────────────────────

    #[test]
    fn normalize_geforce_prefix() {
        assert_eq!(normalize_for_match("NVIDIA GeForce RTX 3090"), "rtx 3090");
    }

    #[test]
    fn normalize_tesla_prefix() {
        assert_eq!(normalize_for_match("Tesla T4"), "t4");
        assert_eq!(normalize_for_match("NVIDIA Tesla T4"), "t4");
    }

    #[test]
    fn normalize_hyphenated_with_memory() {
        assert_eq!(normalize_for_match("NVIDIA A100-SXM4-80GB"), "a100 sxm4");
        assert_eq!(normalize_for_match("NVIDIA A100-PCIE-40GB"), "a100 pcie");
        assert_eq!(normalize_for_match("NVIDIA Tesla V100-SXM2-32GB"), "v100 sxm2");
    }

    #[test]
    fn normalize_h100_sxm() {
        assert_eq!(normalize_for_match("NVIDIA H100 SXM5 80GB"), "h100 sxm5");
        // "80GB" is a memory-size token and gets filtered out.
        assert_eq!(normalize_for_match("NVIDIA H100 PCIe 80GB"), "h100 pcie");
        assert_eq!(normalize_for_match("NVIDIA H100 PCIe"), "h100 pcie");
    }

    // ── match_score ──────────────────────────────────────────────────────

    #[test]
    fn match_score_exact() {
        assert!(match_score("RTX 3090", "rtx 3090") > 0);
    }

    #[test]
    fn match_score_prefix_sxm() {
        // "sxm" token in spec should prefix-match "sxm4" in query.
        assert!(match_score("A100 SXM", "a100 sxm4") > 0);
    }

    #[test]
    fn match_score_no_match_wrong_variant() {
        // A100 PCIe tokens should not match an SXM query.
        assert_eq!(match_score("A100 PCIe", "a100 sxm4"), 0);
    }

    #[test]
    fn match_score_a10g_beats_a10_for_a10g_query() {
        let score_a10g = match_score("A10G", "a10g");
        let score_a10 = match_score("A10", "a10g");
        assert!(score_a10g > score_a10, "A10G ({score_a10g}) should score higher than A10 ({score_a10}) for 'A10G' query");
    }

    #[test]
    fn match_score_a10_matches_a10_not_a10g() {
        // "a10g".starts_with("a10g") only; "a10".starts_with("a10g") is false.
        assert!(match_score("A10", "a10") > 0);
        assert_eq!(match_score("A10G", "a10"), 0, "A10G should not match plain 'a10' query");
    }

    // ── find_best_match / FallbackRepository integration ─────────────────

    #[test]
    fn fallback_resolves_geforce_rtx_3090() {
        let spec = FallbackRepository.get_by_name("NVIDIA GeForce RTX 3090").unwrap();
        assert!((spec.bf16_tflops - 35.6).abs() < 0.1, "unexpected tflops: {}", spec.bf16_tflops);
        assert_eq!(spec.vram_gib, 24);
    }

    #[test]
    fn fallback_resolves_tesla_t4() {
        let spec = FallbackRepository.get_by_name("Tesla T4").unwrap();
        assert!((spec.bf16_tflops - 65.0).abs() < 0.1);
    }

    #[test]
    fn fallback_resolves_a100_sxm_from_nvml_name() {
        let spec = FallbackRepository.get_by_name("NVIDIA A100-SXM4-80GB").unwrap();
        // Should match "A100 SXM", not PCIe.
        assert!(spec.name.contains("SXM"), "expected SXM variant, got: {}", spec.name);
    }

    #[test]
    fn fallback_resolves_a100_pcie_from_nvml_name() {
        let spec = FallbackRepository.get_by_name("NVIDIA A100-PCIE-40GB").unwrap();
        assert!(spec.name.to_lowercase().contains("pcie"), "expected PCIe variant, got: {}", spec.name);
    }

    #[test]
    fn fallback_a10g_disambiguated_from_a10() {
        let a10g = FallbackRepository.get_by_name("NVIDIA A10G").unwrap();
        let a10 = FallbackRepository.get_by_name("NVIDIA A10").unwrap();
        assert!(a10g.name.to_uppercase().contains("A10G"), "got: {}", a10g.name);
        assert_eq!(a10.name.to_uppercase().trim(), "A10", "got: {}", a10.name);
    }

    #[test]
    fn fallback_unknown_gpu_returns_none() {
        assert!(FallbackRepository.get_by_name("NVIDIA RTX Unknown 9999X").is_none());
    }
}

