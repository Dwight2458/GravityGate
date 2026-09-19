//! `GET /v1/models`.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::registry::live::LiveModel;
use crate::registry::models::{MODELS, ModelSpec, Modality};
use crate::registry::{ThinkingTier, resolve};

use super::SharedState;

#[derive(Debug, Serialize)]
pub struct ModelList {
    object: &'static str,
    data: Vec<ModelEntry>,
}

#[derive(Debug, Serialize)]
pub struct ModelEntry {
    id: String,
    object: &'static str,
    created: i64,
    owned_by: String,
    /// Context window and output ceiling, which clients use for budgeting.
    /// Absent for a model the static catalogue does not describe.
    #[serde(skip_serializing_if = "Option::is_none")]
    context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    supports_reasoning: bool,
    supports_tools: bool,
    input_modalities: Vec<&'static str>,
    output_modalities: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    display_name: Option<String>,
}

/// List the models this gateway can serve.
///
/// Built from the static catalogue rather than by querying the upstream, for two
/// reasons. The catalogue carries what the upstream does not report — context
/// limits, modalities, tier routes — and a model list that disappears when the
/// network is down is worse than one that is slightly optimistic, since clients
/// routinely fetch it once at startup and cache it.
///
/// Tier variants are listed alongside their base name because both are accepted:
/// `gemini-3.8-flash` resolves by `reasoning_effort`, and
/// `gemini-3.8-flash-high` resolves by name.
pub async fn list(State(state): State<SharedState>) -> Response {
    let live = state.engine.live_models().await;

    // Merge order matters. The static table supplies the metadata the upstream
    // does not report, so it wins on anything it knows. The live list is the
    // authority on membership, so it adds anything the table has never heard of
    // — a model the account can reach that this build predates.
    //
    // A model the *upstream* says is exhausted is still listed. Clients use this
    // endpoint to populate a picker once at startup and cache it for the
    // session; removing a model that clears in an hour would mean a restart to
    // get it back. Exhaustion belongs on the quota endpoints, not here.
    let known: std::collections::BTreeSet<&str> =
        MODELS.iter().map(|spec| spec.id).collect();

    let mut data: Vec<ModelEntry> = MODELS.iter().flat_map(entries_for).collect();

    // Live models whose wire name is not already represented. A wire model maps
    // back to its catalogue entry, so `gemini-3.8-flash-medium` arriving live
    // does not add a duplicate of `gemini-3.8-flash`.
    let mut seen: std::collections::BTreeSet<String> =
        data.iter().map(|entry| entry.id.clone()).collect();
    let mut live_only: Vec<ModelEntry> = Vec::new();

    for model in &live {
        if crate::registry::models::base_for_wire(&model.id).is_some() {
            continue;
        }
        if known.contains(model.id.as_str()) || !seen.insert(model.id.clone()) {
            continue;
        }
        live_only.push(entry_for_live(model));
    }
    data.extend(live_only);

    (StatusCode::OK, Json(ModelList { object: "list", data })).into_response()
}

/// Describe a model the static catalogue does not know.
///
/// The limits are deliberately absent rather than guessed: an entry with a
/// wrong context window is worse than one with no context window, because a
/// client will budget against it. `None` here means "ask the model".
fn entry_for_live(model: &LiveModel) -> ModelEntry {
    ModelEntry {
        id: model.id.clone(),
        object: "model",
        created: 0,
        owned_by: match crate::registry::ModelFamily::infer(&model.id) {
            crate::registry::ModelFamily::Gemini => "google".into(),
            crate::registry::ModelFamily::Claude => "anthropic".into(),
            crate::registry::ModelFamily::GptOss => "openai".into(),
        },
        context_length: None,
        max_output_tokens: None,
        supports_reasoning: !matches!(
            crate::registry::ModelFamily::infer(&model.id),
            crate::registry::ModelFamily::GptOss
        ),
        supports_tools: true,
        input_modalities: vec!["text"],
        output_modalities: vec!["text"],
        display_name: model.display_name.clone(),
    }
}

/// One catalogue entry, plus one per tier variant where the model has them.
///
/// A variant is listed only when it reaches a *different* wire model than the
/// base name does. Some models have tiers that collapse — `gemini-3.1-pro` and
/// `gemini-3.1-pro-medium` both resolve to `gemini-3.1-pro-low` — and listing
/// every accepted spelling would pad the list with names that behave
/// identically, which is noise for a client building a model picker.
fn entries_for(spec: &'static ModelSpec) -> Vec<ModelEntry> {
    // The base name's own destination, which is what variants are compared
    // against.
    let base_wire = resolve::resolve(resolve::ResolveInput {
        requested: spec.id,
        ..Default::default()
    })
    .map(|resolved| resolved.wire_model)
    .unwrap_or_default();

    let mut entries = vec![entry_for(spec.id, spec)];

    for tier in [ThinkingTier::Low, ThinkingTier::Medium, ThinkingTier::High] {
        let Some(route) = spec.route(tier) else {
            continue;
        };
        if route.wire_model == spec.id || route.wire_model == base_wire {
            continue;
        }

        let id = format!("{}-{}", spec.id, tier.as_str());
        // Only list a variant the name actually resolves to, so the list cannot
        // advertise something resolution would reject.
        if resolve::resolve(resolve::ResolveInput {
            requested: &id,
            ..Default::default()
        })
        .map(|resolved| resolved.wire_model == route.wire_model)
        .unwrap_or(false)
        {
            entries.push(entry_for(&id, spec));
        }
    }

    entries
}

fn entry_for(id: &str, spec: &ModelSpec) -> ModelEntry {
    ModelEntry {
        id: id.to_string(),
        object: "model",
        // The catalogue is static, so there is no real creation date. Zero is
        // what other gateways send and clients do not act on it.
        created: 0,
        owned_by: match spec.family {
            crate::registry::ModelFamily::Gemini => "google".into(),
            crate::registry::ModelFamily::Claude => "anthropic".into(),
            crate::registry::ModelFamily::GptOss => "openai".into(),
        },
        context_length: Some(spec.context_limit),
        max_output_tokens: Some(spec.output_limit),
        supports_reasoning: spec.supports_thinking,
        supports_tools: spec.supports_tools,
        input_modalities: spec.input_modalities.iter().map(modality_name).collect(),
        output_modalities: spec.output_modalities.iter().map(modality_name).collect(),
        display_name: Some(spec.display_name.to_string()),
    }
}

fn modality_name(modality: &Modality) -> &'static str {
    match modality {
        Modality::Text => "text",
        Modality::Image => "image",
        Modality::Pdf => "pdf",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::models::base_for_wire;

    fn entries() -> Vec<ModelEntry> {
        MODELS.iter().flat_map(entries_for).collect()
    }

    #[test]
    fn every_catalogue_model_is_listed() {
        let listed = entries();
        for spec in MODELS {
            assert!(
                listed.iter().any(|entry| entry.id == spec.id),
                "{} is missing",
                spec.id
            );
        }
    }

    #[test]
    fn listed_ids_are_unique() {
        let entries = entries();
        let mut ids: Vec<&str> = entries.iter().map(|entry| entry.id.as_str()).collect();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count, "duplicate id in the model list");
    }

    #[test]
    fn every_listed_id_resolves_to_a_real_wire_model() {
        // The list is a promise: anything in it must be usable.
        for entry in entries() {
            let resolved = resolve::resolve(resolve::ResolveInput {
                requested: &entry.id,
                ..Default::default()
            })
            .unwrap_or_else(|_| panic!("{} failed to resolve", entry.id));
            assert!(
                !resolved.wire_model.is_empty(),
                "{} resolved to nothing",
                entry.id
            );
        }
    }

    #[test]
    fn tier_variants_are_listed_for_tiered_models() {
        let ids: Vec<String> = entries().iter().map(|entry| entry.id.clone()).collect();
        assert!(ids.contains(&"gemini-3.8-flash-low".to_string()));
        assert!(ids.contains(&"gemini-3.8-flash-high".to_string()));
    }

    #[test]
    fn variants_that_reach_the_same_wire_model_are_not_listed() {
        // `gemini-3.1-pro` resolves to `gemini-3.1-pro-low`, so listing
        // `gemini-3.1-pro-low` as well would advertise the same behaviour twice.
        let ids: Vec<String> = entries().iter().map(|entry| entry.id.clone()).collect();
        assert!(!ids.contains(&"gemini-3.1-pro-low".to_string()));
        assert!(!ids.contains(&"gemini-3.1-pro-medium".to_string()));
        // `-high` reaches `gemini-pro-agent`, so it is genuinely distinct.
        assert!(ids.contains(&"gemini-3.1-pro-high".to_string()));
    }

    #[test]
    fn every_listed_id_reaches_a_distinct_wire_model_per_family() {
        // Two entries may share a wire model only if they are the same model;
        // this guards the comparison above from over-pruning.
        let listed = entries();
        let ids: Vec<&str> = listed.iter().map(|entry| entry.id.as_str()).collect();
        assert!(ids.contains(&"gemini-3.1-pro"), "the base name must survive");
    }

    #[test]
    fn models_that_do_not_rename_per_tier_have_no_variants() {
        // Listing `claude-opus-4-6-thinking-high` would advertise a name that
        // resolves to the same wire model, which is noise.
        let ids: Vec<String> = entries().iter().map(|entry| entry.id.clone()).collect();
        assert!(!ids.iter().any(|id| id.starts_with("claude-opus-4-6-thinking-")));
        assert!(!ids.iter().any(|id| id.starts_with("gpt-oss-120b-medium-")));
    }

    #[test]
    fn gpt_oss_keeps_its_suffix() {
        // The regression that once lost this model's thinking budget.
        let ids: Vec<String> = entries().iter().map(|entry| entry.id.clone()).collect();
        assert!(ids.contains(&"gpt-oss-120b-medium".to_string()));
    }

    #[test]
    fn limits_are_reported() {
        let entry = entries()
            .into_iter()
            .find(|entry| entry.id == "gemini-3.8-flash")
            .unwrap();
        assert_eq!(entry.context_length, Some(1_048_576));
        assert_eq!(entry.max_output_tokens, Some(65_536));
        assert!(entry.supports_reasoning);
        assert!(entry.supports_tools);
        assert!(entry.input_modalities.contains(&"image"));
    }

    #[test]
    fn ownership_reflects_the_model_family() {
        let all = entries();
        let find = |id: &str| all.iter().find(|entry| entry.id == id).unwrap();
        assert_eq!(find("gemini-3.8-flash").owned_by, "google");
        assert_eq!(find("claude-opus-4-6-thinking").owned_by, "anthropic");
        assert_eq!(find("gpt-oss-120b-medium").owned_by, "openai");
    }

    #[test]
    fn a_live_only_model_omits_limits_rather_than_guessing() {
        // A wrong context window is worse than none: a client budgets against
        // whatever it is told.
        let entry = entry_for_live(&LiveModel {
            id: "gemini-9.9-ultra".into(),
            display_name: Some("Gemini 9.9".into()),
            remaining_fraction: None,
            reset_time: None,
        });
        assert!(entry.context_length.is_none());
        assert!(entry.max_output_tokens.is_none());
        assert_eq!(entry.display_name.as_deref(), Some("Gemini 9.9"));
        assert_eq!(entry.owned_by, "google");
    }

    #[test]
    fn a_live_model_that_the_catalogue_knows_is_not_duplicated() {
        // `gemini-3.8-flash-medium` arriving live maps back to the catalogue's
        // `gemini-3.8-flash`, which is already listed with better metadata.
        assert!(base_for_wire("gemini-3.8-flash-medium").is_some());
        assert!(base_for_wire("gemini-3.8-flash").is_some());
    }

    #[test]
    fn an_unknown_live_model_is_recognised_as_live_only() {
        assert!(base_for_wire("gemini-9.9-ultra").is_none());
    }

    #[test]
    fn the_image_model_reports_image_output() {
        let entry = entries()
            .into_iter()
            .find(|entry| entry.id == "gemini-3.1-flash-image")
            .unwrap();
        assert_eq!(entry.output_modalities, vec!["text", "image"]);
        assert!(!entry.supports_tools);
    }
}
