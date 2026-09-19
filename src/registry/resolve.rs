//! Model name resolution.
//!
//! Turns what a client asked for into an exact wire model name plus a thinking
//! budget. The precedence is deliberate:
//!
//! 1. **A tier suffix in the model name** — `gemini-3.8-flash-high`. Explicit and
//!    unambiguous, so nothing may override it.
//! 2. **An explicit thinking budget** — `thinking.budget_tokens`, a widely
//!    implemented OpenAI extension. Converted to the nearest tier.
//! 3. **`reasoning_effort`** — the OpenAI-native signal.
//! 4. **The configured default tier.**
//!
//! Suffix detection is restricted to models that actually have tier routes. This
//! matters for `gpt-oss-120b-medium`, whose name ends in `-medium`; treating that
//! as a tier would strip it and produce a model that does not exist.

use crate::registry::models::{
    ModelFamily, ModelSpec, ResolvedModel, ThinkingTier, lookup,
};

/// Tier suffixes, checked longest-first so `minimal` is not shadowed.
const TIER_SUFFIXES: &[(&str, ThinkingTier)] = &[
    ("-minimal", ThinkingTier::Minimal),
    ("-medium", ThinkingTier::Medium),
    ("-high", ThinkingTier::High),
    ("-low", ThinkingTier::Low),
];

/// Quota marker some clients prepend to force a specific quota pool. Accepted
/// and stripped; this gateway has a single quota pool, so it carries no further
/// meaning.
const ANTIGRAVITY_PREFIX: &str = "antigravity-";

/// What the client asked for.
#[derive(Debug, Clone, Default)]
pub struct ResolveInput<'a> {
    pub requested: &'a str,
    /// OpenAI `reasoning_effort`.
    pub reasoning_effort: Option<&'a str>,
    /// Explicit budget from an extension field, if the client sent one.
    pub thinking_budget: Option<i64>,
    /// Explicit enable/disable from an extension field.
    pub thinking_enabled: Option<bool>,
    /// Tier to use when the client expressed no preference.
    pub default_tier: ThinkingTier,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResolveError {
    #[error("model name is empty")]
    Empty,
}

/// Resolve a requested model name.
pub fn resolve(input: ResolveInput<'_>) -> Result<ResolvedModel, ResolveError> {
    let requested = input.requested.trim();
    if requested.is_empty() {
        return Err(ResolveError::Empty);
    }

    // Normalise case and drop the quota marker before any matching.
    let lowered = requested.to_ascii_lowercase();
    let name = lowered
        .strip_prefix(ANTIGRAVITY_PREFIX)
        .unwrap_or(&lowered)
        .to_string();

    // Exact catalogue match first. This ordering matters: `gpt-oss-120b-medium`
    // is a catalogue entry whose name *ends* in a tier suffix, so splitting
    // before looking up would strip `-medium`, miss the catalogue, and lose the
    // model's thinking budget.
    if let Some(spec) = lookup(&name) {
        let tier = resolve_tier_from_signals(&input);
        return Ok(apply_spec(spec, tier, &input));
    }

    // Otherwise the name may be a tier-suffixed form of a catalogue entry, e.g.
    // `gemini-3.8-flash-high`. The suffix only counts if the base is known —
    // which is what keeps `-medium` in `gpt-oss-120b-medium` from being read as
    // a tier on a model we do not recognise.
    let (base, suffix_tier) = split_tier_suffix(&name);
    if let Some(spec) = lookup(&base) {
        let tier = suffix_tier.unwrap_or_else(|| resolve_tier_from_signals(&input));
        return Ok(apply_spec(spec, tier, &input));
    }

    // Unknown model: pass the name through untouched. Guessing a tier suffix for
    // a model we have no route table for risks naming a model that does not
    // exist, and a hard 404 is worse than a client's reasoning preference being
    // ignored.
    Ok(ResolvedModel {
        wire_model: name,
        thinking_budget: None,
        tier: suffix_tier.unwrap_or_else(|| resolve_tier_from_signals(&input)),
        family: ModelFamily::infer(requested),
        thinking_enabled: input.thinking_enabled.unwrap_or(true),
        spec: None,
    })
}

/// Split a trailing tier suffix, returning the base name and the tier.
fn split_tier_suffix(name: &str) -> (String, Option<ThinkingTier>) {
    for (suffix, tier) in TIER_SUFFIXES {
        if let Some(base) = name.strip_suffix(suffix) {
            // A bare suffix with nothing before it is not a tier; nor is a name
            // that would become empty or end in a separator.
            if !base.is_empty() && !base.ends_with('-') {
                return (base.to_string(), Some(*tier));
            }
        }
    }
    (name.to_string(), None)
}

/// Derive a tier from everything except the model name.
///
/// An explicit numeric budget outranks a coarse effort hint: the budget is a
/// precise request, the effort a guideline.
fn resolve_tier_from_signals(input: &ResolveInput<'_>) -> ThinkingTier {
    if let Some(budget) = input.thinking_budget
        && budget > 0
        && let Some(tier) = nearest_tier_for_budget(budget)
    {
        return tier;
    }

    if let Some(effort) = input.reasoning_effort
        && let Some(tier) = ThinkingTier::from_reasoning_effort(effort)
    {
        return tier;
    }

    input.default_tier
}

/// Pick the cheapest tier whose budget covers `budget`.
///
/// Cheapest-that-covers rather than nearest avoids over-provisioning thinking:
/// a client asking for 5000 tokens should land on the 10000 tier, not be pushed
/// to a higher one because it happened to be numerically closer.
fn nearest_tier_for_budget(budget: i64) -> Option<ThinkingTier> {
    // Tier budgets cluster at 1000/4000/10000 for Gemini and 8192/16384/32768
    // for Claude. Rather than hard-code either scale, walk the tiers in
    // increasing order and let the caller's model decide the actual budget.
    const SCALE: &[(ThinkingTier, i64)] = &[
        (ThinkingTier::Low, 4_000),
        (ThinkingTier::Medium, 16_384),
        (ThinkingTier::High, i64::MAX),
    ];
    SCALE
        .iter()
        .find(|(_, ceiling)| budget <= *ceiling)
        .map(|(tier, _)| *tier)
}

/// Apply a spec's routes to produce the wire model and budget.
fn apply_spec(
    spec: &'static ModelSpec,
    tier: ThinkingTier,
    input: &ResolveInput<'_>,
) -> ResolvedModel {
    let thinking_enabled = input.thinking_enabled.unwrap_or(spec.supports_thinking);

    if !spec.is_tiered() {
        // No thinking support at all — the name is the wire name.
        return ResolvedModel {
            wire_model: spec.id.to_string(),
            thinking_budget: None,
            tier,
            family: spec.family,
            thinking_enabled: false,
            spec: Some(spec),
        };
    }

    let chosen = spec
        .route(tier)
        .expect("a tiered model always has a route for every tier");
    ResolvedModel {
        wire_model: chosen.wire_model.to_string(),
        thinking_budget: thinking_enabled.then_some(chosen.thinking_budget),
        tier,
        family: spec.family,
        thinking_enabled,
        spec: Some(spec),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve_simple(requested: &str) -> ResolvedModel {
        resolve(ResolveInput {
            requested,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn empty_model_name_is_rejected() {
        assert_eq!(
            resolve(ResolveInput {
                requested: "  ",
                ..Default::default()
            }),
            Err(ResolveError::Empty)
        );
    }

    #[test]
    fn base_name_defaults_to_the_medium_tier() {
        let resolved = resolve_simple("gemini-3.8-flash");
        assert_eq!(resolved.wire_model, "gemini-3.8-flash-medium");
        assert_eq!(resolved.tier, ThinkingTier::Medium);
        assert_eq!(resolved.thinking_budget, Some(4000));
    }

    #[test]
    fn explicit_suffix_selects_the_tier() {
        let resolved = resolve_simple("gemini-3.8-flash-high");
        assert_eq!(resolved.wire_model, "gemini-3.8-flash-high");
        assert_eq!(resolved.tier, ThinkingTier::High);
        assert_eq!(resolved.thinking_budget, Some(-1));
    }

    #[test]
    fn suffix_outranks_reasoning_effort() {
        let resolved = resolve(ResolveInput {
            requested: "gemini-3.8-flash-low",
            reasoning_effort: Some("high"),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(resolved.tier, ThinkingTier::Low);
        assert_eq!(resolved.wire_model, "gemini-3.8-flash-low");
    }

    #[test]
    fn reasoning_effort_applies_when_no_suffix_is_present() {
        let resolved = resolve(ResolveInput {
            requested: "gemini-3.8-flash",
            reasoning_effort: Some("high"),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(resolved.tier, ThinkingTier::High);
        assert_eq!(resolved.wire_model, "gemini-3.8-flash-high");
    }

    #[test]
    fn explicit_budget_outranks_reasoning_effort() {
        let resolved = resolve(ResolveInput {
            requested: "gemini-3.8-flash",
            reasoning_effort: Some("low"),
            thinking_budget: Some(20_000),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(resolved.tier, ThinkingTier::High);
    }

    #[test]
    fn unknown_reasoning_effort_falls_through_to_the_default() {
        let resolved = resolve(ResolveInput {
            requested: "gemini-3.8-flash",
            reasoning_effort: Some("extreme"),
            default_tier: ThinkingTier::Low,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(resolved.tier, ThinkingTier::Low);
    }

    #[test]
    fn configured_default_tier_is_honoured() {
        let resolved = resolve(ResolveInput {
            requested: "gemini-3.8-flash",
            default_tier: ThinkingTier::Low,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(resolved.wire_model, "gemini-3.8-flash-low");
    }

    #[test]
    fn case_and_whitespace_are_normalised() {
        let resolved = resolve_simple("  GEMINI-3.8-FLASH-HIGH  ");
        assert_eq!(resolved.wire_model, "gemini-3.8-flash-high");
    }

    #[test]
    fn quota_prefix_is_stripped() {
        let resolved = resolve_simple("antigravity-gemini-3.8-flash");
        assert_eq!(resolved.wire_model, "gemini-3.8-flash-medium");
    }

    #[test]
    fn gpt_oss_medium_suffix_is_not_a_tier() {
        // The critical case: stripping `-medium` here would name a model that
        // does not exist.
        let resolved = resolve_simple("gpt-oss-120b-medium");
        assert_eq!(resolved.wire_model, "gpt-oss-120b-medium");
        assert_eq!(resolved.family, ModelFamily::GptOss);
    }

    #[test]
    fn gpt_oss_ignores_reasoning_effort_in_its_name() {
        let resolved = resolve(ResolveInput {
            requested: "gpt-oss-120b-medium",
            reasoning_effort: Some("high"),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(resolved.wire_model, "gpt-oss-120b-medium");
        assert_eq!(resolved.thinking_budget, Some(16_384));
    }

    #[test]
    fn claude_keeps_one_wire_name_across_tiers() {
        let low = resolve(ResolveInput {
            requested: "claude-opus-4-6-thinking",
            reasoning_effort: Some("low"),
            ..Default::default()
        })
        .unwrap();
        let high = resolve(ResolveInput {
            requested: "claude-opus-4-6-thinking",
            reasoning_effort: Some("high"),
            ..Default::default()
        })
        .unwrap();

        assert_eq!(low.wire_model, high.wire_model);
        assert!(low.thinking_budget.unwrap() < high.thinking_budget.unwrap());
        assert_eq!(low.family, ModelFamily::Claude);
    }

    #[test]
    fn unknown_models_pass_through_untouched() {
        let resolved = resolve_simple("gemini-9.9-ultra");
        assert_eq!(resolved.wire_model, "gemini-9.9-ultra");
        assert!(resolved.spec.is_none());
        assert!(resolved.thinking_budget.is_none());
    }

    #[test]
    fn unknown_model_does_not_gain_a_tier_suffix() {
        // Guessing here would risk naming a model that does not exist.
        let resolved = resolve(ResolveInput {
            requested: "gemini-9.9-ultra",
            reasoning_effort: Some("high"),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(resolved.wire_model, "gemini-9.9-ultra");
        assert_eq!(resolved.tier, ThinkingTier::High);
    }

    #[test]
    fn unknown_model_family_is_inferred_for_quota_accounting() {
        assert_eq!(
            resolve_simple("claude-future-9").family,
            ModelFamily::Claude
        );
        assert_eq!(resolve_simple("gemini-future-9").family, ModelFamily::Gemini);
    }

    #[test]
    fn thinking_can_be_disabled_explicitly() {
        let resolved = resolve(ResolveInput {
            requested: "gemini-3.8-flash",
            thinking_enabled: Some(false),
            ..Default::default()
        })
        .unwrap();
        assert!(!resolved.thinking_enabled);
        assert!(
            resolved.thinking_budget.is_none(),
            "a disabled budget must not be sent"
        );
        // The wire model still reflects the tier, which is correct: the tier is
        // the model variant, not the reasoning request.
        assert_eq!(resolved.wire_model, "gemini-3.8-flash-medium");
    }

    #[test]
    fn image_model_never_reports_thinking() {
        let resolved = resolve_simple("gemini-3.1-flash-image");
        assert!(!resolved.thinking_enabled);
        assert!(resolved.thinking_budget.is_none());
        assert!(!resolved.spec.unwrap().supports_tools);
    }

    #[test]
    fn output_limit_defaults_for_unknown_models() {
        let resolved = resolve_simple("mystery-model");
        assert_eq!(resolved.output_limit(), 32_768);
        assert_eq!(resolved.context_limit(), None);
    }

    #[test]
    fn known_models_report_their_real_limits() {
        let resolved = resolve_simple("gemini-3.8-flash");
        assert_eq!(resolved.context_limit(), Some(1_048_576));
        assert_eq!(resolved.output_limit(), 65_536);
    }

    #[test]
    fn split_tier_suffix_leaves_non_tier_names_alone() {
        assert_eq!(
            split_tier_suffix("gemini-3.8-flash"),
            ("gemini-3.8-flash".to_string(), None)
        );
        assert_eq!(
            split_tier_suffix("gemini-3.8-flash-high"),
            ("gemini-3.8-flash".to_string(), Some(ThinkingTier::High))
        );
        // A name that is only a suffix must not be reduced to nothing.
        assert_eq!(split_tier_suffix("-high"), ("-high".to_string(), None));
    }
}
