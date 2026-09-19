//! Model registry: the static catalogue behind `/v1/models` and tier routing.
//!
//! Two things live here. The first is metadata — context limits, modalities, and
//! which quota pool a model draws from — used to shape the model list and to
//! validate requests. The second is tier routing: the upstream exposes most
//! Gemini models as separate wire models per thinking tier, so
//! `gemini-3.8-flash` with `reasoning_effort: high` has to become
//! `gemini-3.8-flash-high` before it reaches the wire.
//!
//! The wire names and budgets below come from a captured `agy` CLI 1.1.24 model
//! catalogue (`reference/antigravity-auth/test-fixtures/agy-cli-1.1.24-model-metadata.json`)
//! and the reference resolver. Where the resolver and the capture disagree, the
//! capture wins, because it is a recording of real traffic rather than an
//! interpretation of it.
//!
//! The live catalogue is authoritative at runtime — `/v1/models` merges this
//! table with `fetchAvailableModels` and treats the upstream as the source of
//! truth for what exists. This table supplies what the upstream does not report:
//! limits, modalities, and tier routes.

use serde::{Deserialize, Serialize};

/// Thinking tier. Ordered, so `max`/`min` express "at least this much thinking".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingTier {
    Minimal,
    Low,
    #[default]
    Medium,
    High,
}

impl ThinkingTier {
    /// Parse a tier name. Accepts the suffix forms clients use in model names
    /// and the values `reasoning_effort` takes.
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            other => match other {
                "none" | "off" | "disabled" => None,
                _ => None,
            },
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// Map an OpenAI `reasoning_effort` value onto a tier.
    ///
    /// OpenAI defines low/medium/high; `minimal` and `none` also appear in the
    /// wild, so both are accepted rather than rejected.
    pub fn from_reasoning_effort(effort: &str) -> Option<Self> {
        match effort.trim().to_ascii_lowercase().as_str() {
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }
}

/// Which upstream model family a model belongs to. Drives quota pool selection
/// and the request transform branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelFamily {
    Gemini,
    Claude,
    GptOss,
}

impl ModelFamily {
    /// Infer the family from a wire model name.
    pub fn infer(model: &str) -> Self {
        let lowered = model.to_ascii_lowercase();
        if lowered.contains("claude") {
            Self::Claude
        } else if lowered.starts_with("gpt-") {
            Self::GptOss
        } else {
            Self::Gemini
        }
    }

    /// Whether this family counts as non-Gemini for quota accounting.
    ///
    /// The upstream splits quota into a Gemini pool and a third-party pool;
    /// Claude and GPT-OSS draw from the latter.
    pub fn is_non_gemini(self) -> bool {
        !matches!(self, Self::Gemini)
    }

    /// Quota pool key used in rate-limit bookkeeping.
    pub fn quota_pool(self) -> &'static str {
        match self {
            Self::Gemini => "gemini",
            Self::Claude => "claude",
            Self::GptOss => "non-gemini",
        }
    }
}

/// Input or output modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Text,
    Image,
    Pdf,
}

const TEXT_ONLY: &[Modality] = &[Modality::Text];
const TEXT_IMAGE_PDF: &[Modality] = &[Modality::Text, Modality::Image, Modality::Pdf];
const TEXT_IMAGE: &[Modality] = &[Modality::Text, Modality::Image];

/// One tier's route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierRoute {
    pub wire_model: &'static str,
    /// Numeric thinking budget. `-1` means the upstream should choose
    /// dynamically, which is what the captured catalogue records for the
    /// highest tier of the 3.7 and 3.8 Flash models.
    pub thinking_budget: i64,
}

/// A model as the gateway knows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    /// Public base name, e.g. `gemini-3.8-flash`.
    pub id: &'static str,
    pub display_name: &'static str,
    pub family: ModelFamily,
    pub context_limit: u32,
    pub output_limit: u32,
    pub supports_thinking: bool,
    pub supports_tools: bool,
    pub input_modalities: &'static [Modality],
    pub output_modalities: &'static [Modality],
    /// Tier routes, when the model is exposed with per-tier wire models.
    routes: Option<TierRoutes>,
}

/// Tier routes for a tiered model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TierRoutes {
    /// Every tier maps to a distinct wire model.
    Explicit {
        minimal: Option<TierRoute>,
        low: TierRoute,
        medium: TierRoute,
        high: TierRoute,
    },
    /// The model has one wire name; tiers only vary the thinking budget.
    BudgetOnly {
        wire_model: &'static str,
        minimal: i64,
        low: i64,
        medium: i64,
        high: i64,
    },
}

impl ModelSpec {
    /// Whether this model is exposed as distinct wire models per tier.
    pub fn is_tiered(&self) -> bool {
        self.routes.is_some()
    }

    /// The wire route for a tier, when the model has routes.
    pub fn route(&self, tier: ThinkingTier) -> Option<TierRoute> {
        self.routes.map(|routes| routes.route(tier).0)
    }
}

impl TierRoutes {
    fn route(self, tier: ThinkingTier) -> (TierRoute, Option<TierRoute>) {
        let tier = if !matches!(tier, ThinkingTier::Minimal) {
            tier
        } else {
            ThinkingTier::Low
        };
        let chosen = match self {
            TierRoutes::Explicit {
                minimal,
                low,
                medium,
                high,
            } => match tier {
                ThinkingTier::Minimal | ThinkingTier::Low => minimal.unwrap_or(low),
                ThinkingTier::Medium => medium,
                ThinkingTier::High => high,
            },
            TierRoutes::BudgetOnly {
                wire_model,
                minimal,
                low,
                medium,
                high,
            } => {
                let budget = match tier {
                    ThinkingTier::Minimal => minimal,
                    ThinkingTier::Low => low,
                    ThinkingTier::Medium => medium,
                    ThinkingTier::High => high,
                };
                TierRoute {
                    wire_model,
                    thinking_budget: budget,
                }
            }
        };
        (chosen, None)
    }
}

const fn route(wire_model: &'static str, thinking_budget: i64) -> TierRoute {
    TierRoute {
        wire_model,
        thinking_budget,
    }
}

/// The static catalogue.
///
/// Only models observed in captured CLI traffic are listed. Anything else is
/// passed through to the upstream unmodified, which is the right default: the
/// upstream rejects what it does not know, and inventing routes for unknown
/// models would produce plausible-looking failures.
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "gemini-3.1-pro",
        display_name: "Gemini 3.1 Pro",
        family: ModelFamily::Gemini,
        context_limit: 1_048_576,
        output_limit: 65_535,
        supports_thinking: true,
        supports_tools: true,
        input_modalities: TEXT_IMAGE_PDF,
        output_modalities: TEXT_ONLY,
        // Only two tiers exist upstream; `low` covers everything below `high`.
        routes: Some(TierRoutes::Explicit {
            minimal: None,
            low: route("gemini-3.1-pro-low", 1001),
            medium: route("gemini-3.1-pro-low", 1001),
            high: route("gemini-pro-agent", 10001),
        }),
    },
    ModelSpec {
        id: "gemini-3.5-flash",
        display_name: "Gemini 3.5 Flash",
        family: ModelFamily::Gemini,
        context_limit: 1_048_576,
        output_limit: 65_536,
        supports_thinking: true,
        supports_tools: true,
        input_modalities: TEXT_IMAGE_PDF,
        output_modalities: TEXT_ONLY,
        routes: Some(TierRoutes::Explicit {
            minimal: None,
            low: route("gemini-3.5-flash-extra-low", 1000),
            medium: route("gemini-3.5-flash-low", 4000),
            high: route("gemini-3-flash-agent", 10000),
        }),
    },
    ModelSpec {
        id: "gemini-3.6-flash",
        display_name: "Gemini 3.6 Flash",
        family: ModelFamily::Gemini,
        context_limit: 1_048_576,
        output_limit: 65_536,
        supports_thinking: true,
        supports_tools: true,
        input_modalities: TEXT_IMAGE_PDF,
        output_modalities: TEXT_ONLY,
        routes: Some(TierRoutes::Explicit {
            minimal: None,
            low: route("gemini-3.6-flash-low", 1000),
            medium: route("gemini-3.6-flash-medium", 4000),
            high: route("gemini-3.6-flash-high", 10000),
        }),
    },
    ModelSpec {
        id: "gemini-3.7-flash",
        display_name: "Gemini 3.7 Flash",
        family: ModelFamily::Gemini,
        context_limit: 1_048_576,
        output_limit: 65_536,
        supports_thinking: true,
        supports_tools: true,
        input_modalities: TEXT_IMAGE_PDF,
        output_modalities: TEXT_ONLY,
        // The captured catalogue records -1 for the high tier of 3.7 and 3.8
        // Flash, meaning the model picks its own budget.
        routes: Some(TierRoutes::Explicit {
            minimal: None,
            low: route("gemini-3.7-flash-low", 1000),
            medium: route("gemini-3.7-flash-medium", 4000),
            high: route("gemini-3.7-flash-high", -1),
        }),
    },
    ModelSpec {
        id: "gemini-3.8-flash",
        display_name: "Gemini 3.8 Flash",
        family: ModelFamily::Gemini,
        context_limit: 1_048_576,
        output_limit: 65_536,
        supports_thinking: true,
        supports_tools: true,
        input_modalities: TEXT_IMAGE_PDF,
        output_modalities: TEXT_ONLY,
        routes: Some(TierRoutes::Explicit {
            minimal: None,
            low: route("gemini-3.8-flash-low", 1000),
            medium: route("gemini-3.8-flash-medium", 4000),
            high: route("gemini-3.8-flash-high", -1),
        }),
    },
    ModelSpec {
        id: "claude-sonnet-4-6",
        display_name: "Claude Sonnet 4.6",
        family: ModelFamily::Claude,
        context_limit: 250_000,
        output_limit: 64_000,
        supports_thinking: true,
        supports_tools: true,
        input_modalities: TEXT_IMAGE_PDF,
        output_modalities: TEXT_ONLY,
        // Claude models use numeric budgets for the same wire model.
        routes: Some(TierRoutes::BudgetOnly {
            wire_model: "claude-sonnet-4-6",
            minimal: 8192,
            low: 8192,
            medium: 16384,
            high: 32768,
        }),
    },
    ModelSpec {
        id: "claude-opus-4-6-thinking",
        display_name: "Claude Opus 4.6 Thinking",
        family: ModelFamily::Claude,
        context_limit: 250_000,
        output_limit: 64_000,
        supports_thinking: true,
        supports_tools: true,
        input_modalities: TEXT_IMAGE_PDF,
        output_modalities: TEXT_ONLY,
        routes: Some(TierRoutes::BudgetOnly {
            wire_model: "claude-opus-4-6-thinking",
            minimal: 8192,
            low: 8192,
            medium: 16384,
            high: 32768,
        }),
    },
    ModelSpec {
        id: "gpt-oss-120b-medium",
        display_name: "GPT-OSS 120B",
        family: ModelFamily::GptOss,
        context_limit: 131_072,
        output_limit: 32_768,
        supports_thinking: true,
        supports_tools: true,
        input_modalities: TEXT_IMAGE_PDF,
        output_modalities: TEXT_ONLY,
        // The `-medium` suffix is part of the name, not a thinking tier, so
        // tiers cannot be expressed by renaming.
        routes: Some(TierRoutes::BudgetOnly {
            wire_model: "gpt-oss-120b-medium",
            minimal: 4096,
            low: 4096,
            medium: 8192,
            high: 16384,
        }),
    },
    ModelSpec {
        id: "gemini-3.1-flash-image",
        display_name: "Gemini 3.1 Flash Image",
        family: ModelFamily::Gemini,
        context_limit: 66_000,
        output_limit: 33_000,
        supports_thinking: false,
        supports_tools: false,
        input_modalities: TEXT_IMAGE,
        output_modalities: TEXT_IMAGE,
        routes: None,
    },
];

/// Look up a model by its public base name.
pub fn lookup(id: &str) -> Option<&'static ModelSpec> {
    let lowered = id.to_ascii_lowercase();
    MODELS.iter().find(|spec| spec.id == lowered)
}

/// The catalogue entry a *wire* model name belongs to.
///
/// Wire models are what actually reaches the upstream — `gemini-3.8-flash-medium`
/// rather than `gemini-3.8-flash` — so a lookup by base name misses nearly every
/// real request. Mapping back to the base id collapses tier variants onto one
/// entry, which is also what a dashboard wants: traffic by model, not by model
/// and tier.
pub fn base_for_wire(wire_model: &str) -> Option<&'static str> {
    let lowered = wire_model.to_ascii_lowercase();
    MODELS.iter().find_map(|spec| {
        if spec.id == lowered {
            return Some(spec.id);
        }
        let routes = spec.routes?;
        [ThinkingTier::Minimal, ThinkingTier::Low, ThinkingTier::Medium, ThinkingTier::High]
            .into_iter()
            .any(|tier| routes.route(tier).0.wire_model == lowered)
            .then_some(spec.id)
    })
}

/// A resolved model ready for the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    /// Exact name to send upstream.
    pub wire_model: String,
    /// Thinking budget to request, when the model supports thinking.
    pub thinking_budget: Option<i64>,
    /// Tier that was applied.
    pub tier: ThinkingTier,
    pub family: ModelFamily,
    /// Whether the request should ask for reasoning content at all.
    pub thinking_enabled: bool,
    /// Static metadata, when the model is known.
    pub spec: Option<&'static ModelSpec>,
}

impl ResolvedModel {
    pub fn context_limit(&self) -> Option<u32> {
        self.spec.map(|spec| spec.context_limit)
    }

    pub fn output_limit(&self) -> u32 {
        match self.spec {
            Some(spec) => spec.output_limit,
            // A conservative default for unknown models: better to cap than to
            // have the upstream reject the request outright.
            None => 32_768,
        }
    }
}

/// Error from model resolution.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResolveError {
    #[error("model name is empty")]
    Empty,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_parses_every_supported_spelling() {
        assert_eq!(ThinkingTier::parse("low"), Some(ThinkingTier::Low));
        assert_eq!(ThinkingTier::parse("HIGH"), Some(ThinkingTier::High));
        assert_eq!(ThinkingTier::parse(" medium "), Some(ThinkingTier::Medium));
        assert_eq!(ThinkingTier::parse("minimal"), Some(ThinkingTier::Minimal));
        assert_eq!(ThinkingTier::parse("none"), None);
        assert_eq!(ThinkingTier::parse("banana"), None);
    }

    #[test]
    fn reasoning_effort_maps_onto_tiers() {
        assert_eq!(
            ThinkingTier::from_reasoning_effort("high"),
            Some(ThinkingTier::High)
        );
        assert_eq!(
            ThinkingTier::from_reasoning_effort("minimal"),
            Some(ThinkingTier::Minimal)
        );
        assert_eq!(ThinkingTier::from_reasoning_effort("nonsense"), None);
    }

    #[test]
    fn default_tier_is_medium() {
        assert_eq!(ThinkingTier::default(), ThinkingTier::Medium);
    }

    #[test]
    fn family_inference_handles_each_prefix() {
        assert_eq!(ModelFamily::infer("claude-opus-4-6-thinking"), ModelFamily::Claude);
        assert_eq!(ModelFamily::infer("gpt-oss-120b-medium"), ModelFamily::GptOss);
        assert_eq!(ModelFamily::infer("gemini-3.8-flash"), ModelFamily::Gemini);
        assert_eq!(
            ModelFamily::infer("Gemini-3.8-Flash-High"),
            ModelFamily::Gemini,
            "family inference must be case-insensitive"
        );
    }

    #[test]
    fn only_gemini_models_are_in_the_gemini_pool() {
        assert!(!ModelFamily::Gemini.is_non_gemini());
        assert!(ModelFamily::Claude.is_non_gemini());
        assert!(ModelFamily::GptOss.is_non_gemini());
        assert_eq!(ModelFamily::Gemini.quota_pool(), "gemini");
        assert_eq!(ModelFamily::Claude.quota_pool(), "claude");
    }

    #[test]
    fn catalogue_lookup_is_case_insensitive() {
        assert!(lookup("gemini-3.8-flash").is_some());
        assert!(lookup("GEMINI-3.8-FLASH").is_some());
        assert!(lookup("nonexistent-model").is_none());
    }

    #[test]
    fn a_wire_model_maps_back_to_its_catalogue_entry() {
        assert_eq!(base_for_wire("gemini-3.8-flash-medium"), Some("gemini-3.8-flash"));
        assert_eq!(base_for_wire("gemini-3.8-flash-high"), Some("gemini-3.8-flash"));
        assert_eq!(base_for_wire("gemini-3.1-pro-low"), Some("gemini-3.1-pro"));
        assert_eq!(base_for_wire("gemini-pro-agent"), Some("gemini-3.1-pro"));
    }

    #[test]
    fn a_base_name_maps_to_itself() {
        assert_eq!(base_for_wire("gemini-3.8-flash"), Some("gemini-3.8-flash"));
    }

    #[test]
    fn an_unknown_wire_model_maps_to_nothing() {
        assert_eq!(base_for_wire("gemini-9.9-ultra"), None);
        assert_eq!(base_for_wire(""), None);
    }

    #[test]
    fn every_route_maps_back_to_its_own_entry() {
        // Guards against two catalogue entries claiming the same wire model.
        for spec in MODELS.iter().filter(|spec| spec.is_tiered()) {
            for tier in [ThinkingTier::Low, ThinkingTier::Medium, ThinkingTier::High] {
                let Some(route) = spec.route(tier) else {
                    continue;
                };
                assert_eq!(
                    base_for_wire(route.wire_model),
                    Some(spec.id),
                    "{} route {tier:?} -> {} mapped elsewhere",
                    spec.id,
                    route.wire_model
                );
            }
        }
    }

    #[test]
    fn catalogue_ids_are_unique() {
        let mut ids: Vec<&str> = MODELS.iter().map(|spec| spec.id).collect();
        ids.sort_unstable();
        let count = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), count, "duplicate model id in the catalogue");
    }

    #[test]
    fn every_tiered_model_produces_a_low_and_high_wire_name() {
        // Guards against a tier route accidentally collapsing to one model.
        for spec in MODELS.iter().filter(|spec| spec.routes.is_some()) {
            let routes = spec.routes.unwrap();
            let (low, _) = routes.route(ThinkingTier::Low);
            let (high, _) = routes.route(ThinkingTier::High);
            assert!(
                !low.wire_model.is_empty() && !high.wire_model.is_empty(),
                "{} produced an empty wire model",
                spec.id
            );
        }
    }

    #[test]
    fn minimal_tier_falls_back_to_low_rather_than_failing() {
        // Not every model has a minimal route; the closest cheaper tier is a
        // better answer than an error.
        let spec = lookup("gemini-3.8-flash").unwrap();
        let routes = spec.routes.unwrap();
        let (minimal, _) = routes.route(ThinkingTier::Minimal);
        let (low, _) = routes.route(ThinkingTier::Low);
        assert_eq!(minimal, low);
    }

    #[test]
    fn gemini_tier_routes_match_the_captured_catalogue() {
        let spec = lookup("gemini-3.8-flash").unwrap();
        let routes = spec.routes.unwrap();

        assert_eq!(routes.route(ThinkingTier::Low).0.wire_model, "gemini-3.8-flash-low");
        assert_eq!(routes.route(ThinkingTier::Low).0.thinking_budget, 1000);
        assert_eq!(
            routes.route(ThinkingTier::Medium).0.wire_model,
            "gemini-3.8-flash-medium"
        );
        assert_eq!(routes.route(ThinkingTier::Medium).0.thinking_budget, 4000);
        assert_eq!(routes.route(ThinkingTier::High).0.wire_model, "gemini-3.8-flash-high");
        assert_eq!(
            routes.route(ThinkingTier::High).0.thinking_budget,
            -1,
            "the capture records -1 (model chooses) for the high tier"
        );
    }

    #[test]
    fn budget_only_models_keep_one_wire_name() {
        let spec = lookup("claude-opus-4-6-thinking").unwrap();
        let routes = spec.routes.unwrap();
        let (low, _) = routes.route(ThinkingTier::Low);
        let (high, _) = routes.route(ThinkingTier::High);
        assert_eq!(low.wire_model, "claude-opus-4-6-thinking");
        assert_eq!(high.wire_model, "claude-opus-4-6-thinking");
        assert!(high.thinking_budget > low.thinking_budget);
    }

    #[test]
    fn gpt_oss_suffix_is_not_treated_as_a_tier() {
        // `gpt-oss-120b-medium` ends in `-medium`, but that is part of the model
        // name. Stripping it would produce a model that does not exist.
        let spec = lookup("gpt-oss-120b-medium").unwrap();
        assert_eq!(spec.family, ModelFamily::GptOss);
        let routes = spec.routes.unwrap();
        assert_eq!(
            routes.route(ThinkingTier::High).0.wire_model,
            "gpt-oss-120b-medium"
        );
    }

    #[test]
    fn image_model_has_no_tier_routes_and_no_tools() {
        let spec = lookup("gemini-3.1-flash-image").unwrap();
        assert!(spec.routes.is_none());
        assert!(!spec.supports_tools);
        assert!(!spec.supports_thinking);
        assert!(spec.output_modalities.contains(&Modality::Image));
    }

    #[test]
    fn limits_are_ordered_sensibly() {
        for spec in MODELS {
            assert!(
                spec.output_limit <= spec.context_limit,
                "{} has output_limit > context_limit",
                spec.id
            );
            assert!(spec.context_limit > 0 && spec.output_limit > 0);
        }
    }
}
