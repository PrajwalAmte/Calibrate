//! GPU peak-throughput database and name matching.
//!
//! `bf16_tflops` is the **dense** (no sparsity) BF16/FP16 tensor-core rate with
//! FP32 accumulation at boost clock — the denominator used for MFU in mixed
//! precision training. GeForce cards run FP32-accumulate tensor math at half
//! the FP16-accumulate rate, and the table reflects that.

pub mod client;
pub mod fallback;

use serde::{Deserialize, Serialize};

/// Specification for a single GPU model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuSpec {
    /// Canonical short name (e.g. "RTX 3090", "H100 SXM").
    pub name: String,
    /// Other names the same GPU is reported under (NVML, cloud providers).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    /// Peak dense BF16 tensor TFLOPS (MFU denominator).
    pub bf16_tflops: f64,
    /// Peak FP32 (non-tensor) TFLOPS.
    pub fp32_tflops: f64,
    /// VRAM in GiB (typical SKU).
    pub vram_gib: u32,
    /// Boost clock in MHz — the clock at which `bf16_tflops` is specified.
    pub boost_clock_mhz: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl GpuSpec {
    fn is_valid(&self) -> bool {
        !self.name.trim().is_empty()
            && self.bf16_tflops.is_finite()
            && self.bf16_tflops > 0.0
            && self.bf16_tflops < 100_000.0
            && self.fp32_tflops.is_finite()
            && self.fp32_tflops > 0.0
            && self.boost_clock_mhz > 0
    }
}

/// Where a resolved spec came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecSource {
    Remote,
    Builtin,
    /// GPU not in the database — peak throughput unknown.
    Unknown,
}

/// Result of looking a GPU name up.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedSpec {
    pub spec: GpuSpec,
    pub source: SpecSource,
}

impl ResolvedSpec {
    pub fn known(&self) -> bool {
        self.source != SpecSource::Unknown
    }
}

/// The in-memory spec database used for all lookups.
#[derive(Debug, Clone)]
pub struct SpecDb {
    entries: Vec<(GpuSpec, SpecSource)>,
}

impl SpecDb {
    /// Built-in table only (no I/O).
    pub fn builtin() -> Self {
        Self {
            entries: fallback::builtin_specs()
                .into_iter()
                .map(|s| (s, SpecSource::Builtin))
                .collect(),
        }
    }

    /// Built-in table overlaid with the remote/cached database when available.
    /// Never fails; never blocks longer than the configured HTTP timeout.
    pub fn load() -> Self {
        let mut db = Self::builtin();
        if let Some(remote) = client::load_remote_specs() {
            db.merge(remote);
        }
        db
    }

    /// Remote entries replace built-in ones with the same name; new names are added.
    pub fn merge(&mut self, remote: Vec<GpuSpec>) {
        for spec in remote.into_iter().filter(GpuSpec::is_valid) {
            if let Some(slot) = self
                .entries
                .iter_mut()
                .find(|(s, _)| s.name.eq_ignore_ascii_case(&spec.name))
            {
                // Remote values win, but never lose aliases this build knows
                // about (an older remote file may predate them).
                let mut spec = spec;
                for a in &slot.0.aliases {
                    if !spec.aliases.iter().any(|x| x.eq_ignore_ascii_case(a)) {
                        spec.aliases.push(a.clone());
                    }
                }
                *slot = (spec, SpecSource::Remote);
            } else {
                self.entries.push((spec, SpecSource::Remote));
            }
        }
    }

    /// Look a device name up. Unknown GPUs get a placeholder flagged `Unknown`
    /// so callers can show "peak unknown" instead of a made-up number.
    pub fn resolve(&self, name: &str) -> ResolvedSpec {
        match self.find(name) {
            Some((spec, source)) => ResolvedSpec {
                spec: spec.clone(),
                source,
            },
            None => ResolvedSpec {
                spec: GpuSpec {
                    name: name.to_string(),
                    aliases: vec![],
                    bf16_tflops: 0.0,
                    fp32_tflops: 0.0,
                    vram_gib: 0,
                    boost_clock_mhz: 0,
                    notes: None,
                },
                source: SpecSource::Unknown,
            },
        }
    }

    pub fn find(&self, name: &str) -> Option<(&GpuSpec, SpecSource)> {
        let query = normalize_for_match(name);
        if query.is_empty() {
            return None;
        }
        self.entries
            .iter()
            .filter_map(|(spec, src)| {
                std::iter::once(spec.name.as_str())
                    .chain(spec.aliases.iter().map(String::as_str))
                    .map(|n| match_score(&normalize_for_match(n), &query))
                    .max()
                    .filter(|&s| s > 0)
                    .map(|s| (spec, *src, s))
            })
            .max_by_key(|&(_, _, s)| s)
            .map(|(spec, src, _)| (spec, src))
    }
}

//  Name matching

/// Tokens that distinguish a different (usually faster) SKU. If the query has
/// one, the matching spec must have it too — an "RTX 3090 Ti" must never
/// silently resolve to "RTX 3090".
const VARIANT_TOKENS: &[&str] = &[
    "ti", "super", "max", "pro", "ultra", "nvl", "mobile", "laptop", "sff",
];

/// Normalise a device / listing name for matching.
///
/// ```text
/// "NVIDIA GeForce RTX 3090"        → "rtx 3090"
/// "NVIDIA A100-SXM4-80GB"          → "a100 sxm4"
/// "NVIDIA H100 80GB HBM3"          → "h100 hbm3"
/// "RTX 4090 (spot)"                → "rtx 4090"
/// "RTX 4070S"                      → "rtx 4070 super"
/// "NVIDIA RTX 6000 Ada Generation" → "rtx 6000 ada"
/// "Apple M3 Max GPU"               → "m3 max"
/// ```
pub fn normalize_for_match(name: &str) -> String {
    let mut lower = name.to_lowercase();
    // Drop parenthesised annotations such as "(spot)" or "(PCIe)".
    while let (Some(a), Some(b)) = (lower.find('('), lower.find(')')) {
        if b > a {
            lower.replace_range(a..=b, " ");
        } else {
            break;
        }
    }
    let lower = lower.replace(['-', '_', '/'], " ");
    let mut tokens: Vec<String> = Vec::new();
    for t in lower.split_whitespace() {
        if matches!(
            t,
            "nvidia" | "geforce" | "tesla" | "quadro" | "apple" | "gpu" | "generation"
        ) || is_memory_size_token(t)
        {
            continue;
        }
        // "4070s" → "4070 super"
        if t.len() > 2 && t.ends_with('s') && t[..t.len() - 1].chars().all(|c| c.is_ascii_digit()) {
            tokens.push(t[..t.len() - 1].to_string());
            tokens.push("super".into());
            continue;
        }
        tokens.push(t.to_string());
    }
    tokens.join(" ")
}

fn token_matches(spec_tok: &str, query_tok: &str) -> bool {
    if spec_tok == query_tok {
        return true;
    }
    // Purely alphabetic spec tokens may match a numbered variant
    // ("sxm" ↔ "sxm4", "pcie" ↔ "pcie4") but model numbers must be exact
    // ("a10" ≠ "a100", "l4" ≠ "l40s").
    spec_tok.chars().all(|c| c.is_ascii_alphabetic())
        && query_tok.starts_with(spec_tok)
        && query_tok[spec_tok.len()..]
            .chars()
            .all(|c| c.is_ascii_digit())
}

/// Score how well normalised `spec_name` matches normalised `query`.
/// 0 = no match. Every spec token must match, and every variant token in the
/// query must be matched. Longer (more specific) specs score higher.
pub fn match_score(spec_name: &str, query: &str) -> usize {
    let spec_tokens: Vec<&str> = spec_name.split_whitespace().collect();
    let query_tokens: Vec<&str> = query.split_whitespace().collect();
    if spec_tokens.is_empty() {
        return 0;
    }
    let mut used = vec![false; query_tokens.len()];
    let mut score = 0;
    for st in &spec_tokens {
        let hit = query_tokens
            .iter()
            .enumerate()
            .find(|(i, qt)| !used[*i] && token_matches(st, qt));
        match hit {
            Some((i, _)) => {
                used[i] = true;
                score += st.len();
            }
            None => return 0,
        }
    }
    for (i, qt) in query_tokens.iter().enumerate() {
        if !used[i] && VARIANT_TOKENS.contains(qt) {
            return 0;
        }
    }
    score
}

fn is_memory_size_token(token: &str) -> bool {
    let digits = token
        .strip_suffix("gb")
        .or_else(|| token.strip_suffix('g'))
        .unwrap_or("");
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> SpecDb {
        SpecDb::builtin()
    }

    fn resolves_to(query: &str, expected: &str) {
        let r = db().resolve(query);
        assert_eq!(r.spec.name, expected, "query {query:?}");
        assert!(r.known());
    }

    #[test]
    fn normalization_examples() {
        assert_eq!(normalize_for_match("NVIDIA GeForce RTX 3090"), "rtx 3090");
        assert_eq!(normalize_for_match("NVIDIA A100-SXM4-80GB"), "a100 sxm4");
        assert_eq!(normalize_for_match("NVIDIA H100 80GB HBM3"), "h100 hbm3");
        assert_eq!(normalize_for_match("RTX 4090 (spot)"), "rtx 4090");
        assert_eq!(normalize_for_match("RTX 4070S"), "rtx 4070 super");
        assert_eq!(normalize_for_match("Tesla V100-SXM2-16GB"), "v100 sxm2");
        assert_eq!(normalize_for_match("Apple M3 Max GPU"), "m3 max");
        assert_eq!(normalize_for_match("A100 SXM 40G"), "a100 sxm");
    }

    #[test]
    fn real_nvml_device_names_resolve() {
        resolves_to("NVIDIA H100 80GB HBM3", "H100 SXM");
        resolves_to("NVIDIA H100 PCIe", "H100 PCIe");
        resolves_to("NVIDIA H100 NVL", "H100 NVL");
        resolves_to("NVIDIA H200", "H200");
        resolves_to("NVIDIA GH200 480GB", "GH200");
        resolves_to("NVIDIA B200", "B200");
        resolves_to("NVIDIA A100-SXM4-80GB", "A100 SXM");
        resolves_to("NVIDIA A100 80GB PCIe", "A100 PCIe");
        resolves_to("NVIDIA A10G", "A10G");
        resolves_to("NVIDIA A10", "A10");
        resolves_to("NVIDIA L4", "L4");
        resolves_to("NVIDIA L40S", "L40S");
        resolves_to("NVIDIA L40", "L40");
        resolves_to("Tesla T4", "T4");
        resolves_to("Tesla V100-SXM2-16GB", "V100 SXM2");
        resolves_to("NVIDIA GeForce RTX 4090", "RTX 4090");
        resolves_to("NVIDIA GeForce RTX 5090", "RTX 5090");
        resolves_to("NVIDIA GeForce RTX 3090 Ti", "RTX 3090 Ti");
        resolves_to("NVIDIA RTX A6000", "RTX A6000");
        resolves_to("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada");
        resolves_to("Apple M3 Max GPU", "M3 Max");
        resolves_to("Apple M2 GPU", "M2");
        resolves_to("NVIDIA RTX 4000 Ada Generation", "RTX 4000 Ada");
        resolves_to("RTX 4000 Ada SFF", "RTX 4000 Ada SFF");
        resolves_to("NVIDIA RTX A4500", "RTX A4500");
    }

    #[test]
    fn cloud_listing_names_resolve() {
        resolves_to("RTX 4090 (spot)", "RTX 4090");
        resolves_to("H100 SXM", "H100 SXM");
        resolves_to("A100 SXM 40GB", "A100 SXM");
        resolves_to("RTX 4080 SUPER", "RTX 4080 Super");
        resolves_to("RTX 4070S", "RTX 4070 Super");
        resolves_to("Tesla V100", "V100 PCIe");
        resolves_to("H200 SXM", "H200");
    }

    #[test]
    fn model_numbers_must_match_exactly() {
        assert_eq!(match_score("a10", "a100"), 0);
        assert_eq!(match_score("l4", "l40s"), 0);
        assert!(match_score("a100 sxm", "a100 sxm4") > 0);
    }

    #[test]
    fn variant_suffix_never_falls_back_to_base_model() {
        assert_eq!(match_score("rtx 3090", "rtx 3090 ti"), 0);
        assert_eq!(match_score("m3", "m3 max"), 0);
        assert!(!db().resolve("NVIDIA GeForce RTX 4060 Super").known());
    }

    #[test]
    fn unknown_gpu_is_flagged_not_invented() {
        let r = db().resolve("NVIDIA Quantum Flux 9000");
        assert_eq!(r.source, SpecSource::Unknown);
        assert_eq!(r.spec.bf16_tflops, 0.0);
        assert!(!db().resolve("").known());
    }

    #[test]
    fn remote_entries_override_and_extend() {
        let mut db = db();
        let base = db.resolve("RTX 4090").spec.bf16_tflops;
        db.merge(vec![
            GpuSpec {
                name: "RTX 4090".into(),
                aliases: vec![],
                bf16_tflops: base + 1.0,
                fp32_tflops: 82.6,
                vram_gib: 24,
                boost_clock_mhz: 2520,
                notes: None,
            },
            GpuSpec {
                name: "Bogus".into(),
                aliases: vec![],
                bf16_tflops: -5.0,
                fp32_tflops: 1.0,
                vram_gib: 1,
                boost_clock_mhz: 1,
                notes: None,
            },
        ]);
        let r = db.resolve("NVIDIA GeForce RTX 4090");
        assert_eq!(r.source, SpecSource::Remote);
        // An older remote entry without aliases must not drop built-in ones.
        let mut db2 = SpecDb::builtin();
        let mut v100 = db2.resolve("V100 PCIe").spec;
        v100.aliases.clear();
        db2.merge(vec![v100]);
        assert!(db2.resolve("Tesla V100").known());
        assert_eq!(r.spec.bf16_tflops, base + 1.0);
        assert!(
            !db.resolve("Bogus").known(),
            "invalid remote specs are dropped"
        );
    }
}
