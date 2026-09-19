//! The live model catalogue.
//!
//! The static [`super::models`] table says what models *exist* and carries the
//! metadata the upstream does not report: context limits, modalities, tier
//! routes. It cannot say what a given account can actually reach, because that
//! varies by account and changes without warning.
//!
//! `fetchAvailableModels` answers that, and also returns each model's remaining
//! quota, which is what makes `/account-limits` mean something. It is the
//! upstream's view, so it is the authority on membership.
//!
//! Two properties shape the caching:
//!
//! - **Fail open.** A model list that empties when the network hiccups is worse
//!   than one that is slightly stale, because clients routinely fetch it once at
//!   startup and cache it for the session. The static table is the floor.
//! - **Cached briefly and globally.** Membership changes on the scale of
//!   deployments, not seconds. One fetch is shared across all accounts: the
//!   differences between accounts are in quota, not in which models exist.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Deserialize;

/// How long a fetched catalogue is trusted.
pub const CACHE_TTL: Duration = Duration::from_secs(5 * 60);

/// One model as the upstream describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveModel {
    /// Wire model id, which is also what clients send.
    pub id: String,
    pub display_name: Option<String>,
    /// Remaining quota as a fraction, when the upstream reports it.
    pub remaining_fraction: Option<f64>,
    /// When that quota refreshes, as an RFC 3339 timestamp.
    pub reset_time: Option<String>,
}

impl LiveModel {
    /// Whether the upstream reports this model as exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.remaining_fraction.is_some_and(|fraction| fraction <= 0.0)
    }
}

/// The upstream's response to `fetchAvailableModels`.
#[derive(Debug, Deserialize)]
struct Response {
    #[serde(default)]
    models: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Whether an id the upstream reports is a model a client can actually call.
///
/// The response is not a clean model list. Observed entries that are not models
/// at all:
///
/// ```text
/// chat_20706                       an internal routing identifier
/// chat_23310                       likewise
/// tab_flash_lite_preview           a feature flag
/// tab_jump_flash_lite_preview      likewise
/// ```
///
/// Advertising those is worse than omitting a real model: a client that picks
/// one from a picker gets a failure it cannot interpret.
///
/// The test is an explicit family prefix. The reference implementation instead
/// asks "is this Claude or Gemini", which removes the internal entries but also
/// discards `gpt-oss-120b-medium` as collateral — a real model that works.
pub fn is_model_shaped(id: &str) -> bool {
    let lowered = id.to_ascii_lowercase();
    lowered.starts_with("gemini-") || lowered.starts_with("claude-") || lowered.starts_with("gpt-")
}

/// Parse the upstream response into a model list.
///
/// Kept as a free function so the parsing is testable against a captured body
/// without a network.
pub fn parse(body: &str) -> Result<Vec<LiveModel>, serde_json::Error> {
    let response: Response = serde_json::from_str(body)?;
    let Some(models) = response.models else {
        return Ok(Vec::new());
    };

    let mut parsed: Vec<LiveModel> = models
        .into_iter()
        .filter(|(id, _)| is_model_shaped(id))
        .map(|(id, entry)| LiveModel {
            display_name: entry
                .get("displayName")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            remaining_fraction: entry
                .pointer("/quotaInfo/remainingFraction")
                .and_then(serde_json::Value::as_f64),
            reset_time: entry
                .pointer("/quotaInfo/resetTime")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            id,
        })
        .collect();

    // Deterministic order, so the endpoint does not shuffle on every fetch.
    parsed.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(parsed)
}

#[derive(Debug, Clone)]
struct Cached {
    models: Vec<LiveModel>,
    fetched_at: Instant,
}

/// A TTL'd cache of the upstream's model list.
#[derive(Debug, Default)]
pub struct LiveCatalogue {
    cached: Mutex<Option<Cached>>,
}

impl LiveCatalogue {
    pub fn new() -> Self {
        Self::default()
    }

    /// The cached list, if it is still fresh.
    pub fn fresh(&self) -> Option<Vec<LiveModel>> {
        let cached = self.cached.lock().ok()?;
        cached
            .as_ref()
            .filter(|cached| cached.fetched_at.elapsed() < CACHE_TTL)
            .map(|cached| cached.models.clone())
    }

    /// The cached list regardless of age.
    ///
    /// Used as the fallback when a refresh fails: stale membership beats no
    /// membership, and a client that fetched the list at startup will not ask
    /// again.
    pub fn stale(&self) -> Option<Vec<LiveModel>> {
        let cached = self.cached.lock().ok()?;
        cached.as_ref().map(|cached| cached.models.clone())
    }

    pub fn store(&self, models: Vec<LiveModel>) {
        if let Ok(mut cached) = self.cached.lock() {
            *cached = Some(Cached {
                models,
                fetched_at: Instant::now(),
            });
        }
    }

    pub fn clear(&self) {
        if let Ok(mut cached) = self.cached.lock() {
            *cached = None;
        }
    }

    /// How old the cached list is, for `/health` and diagnostics.
    pub fn age(&self) -> Option<Duration> {
        let cached = self.cached.lock().ok()?;
        cached.as_ref().map(|cached| cached.fetched_at.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed from the real response shape observed from the upstream.
    const SAMPLE: &str = r#"{
      "models": {
        "gemini-3.8-flash-medium": {
          "displayName": "Gemini 3.8 Flash (Medium)",
          "quotaInfo": { "remainingFraction": 0.8875432, "resetTime": "2026-07-31T15:54:18Z" }
        },
        "claude-opus-4-6-thinking": {
          "displayName": "Claude Opus 4.6 Thinking",
          "quotaInfo": { "remainingFraction": 0.0 }
        },
        "gpt-oss-120b-medium": {
          "displayName": "GPT-OSS 120B"
        }
      }
    }"#;

    fn sample() -> Vec<LiveModel> {
        parse(SAMPLE).unwrap()
    }

    fn find<'a>(models: &'a [LiveModel], id: &str) -> &'a LiveModel {
        models.iter().find(|model| model.id == id).unwrap()
    }

    #[test]
    fn every_model_in_the_response_is_parsed() {
        assert_eq!(sample().len(), 3);
    }

    #[test]
    fn quota_and_reset_are_read() {
        let models = sample();
        let flash = find(&models, "gemini-3.8-flash-medium");
        assert_eq!(
            flash.display_name.as_deref(),
            Some("Gemini 3.8 Flash (Medium)")
        );
        assert!((flash.remaining_fraction.unwrap() - 0.8875432).abs() < 1e-6);
        assert_eq!(flash.reset_time.as_deref(), Some("2026-07-31T15:54:18Z"));
        assert!(!flash.is_exhausted());
    }

    #[test]
    fn an_exhausted_model_is_recognised() {
        let models = sample();
        assert!(find(&models, "claude-opus-4-6-thinking").is_exhausted());
    }

    #[test]
    fn a_model_without_quota_info_is_not_exhausted() {
        // Absent is not the same as zero: a model the upstream reports without
        // quota data is not a model that has run out.
        let models = sample();
        let model = find(&models, "gpt-oss-120b-medium");
        assert!(model.remaining_fraction.is_none());
        assert!(!model.is_exhausted());
    }

    #[test]
    fn models_are_ordered_deterministically() {
        // The endpoint must not shuffle between fetches, or a client diffing the
        // list sees phantom changes.
        let first = sample();
        let second = sample();
        assert_eq!(first, second);
        let ids: Vec<&str> = first.iter().map(|model| model.id.as_str()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn an_empty_models_map_yields_an_empty_list() {
        assert!(parse(r#"{"models": {}}"#).unwrap().is_empty());
    }

    #[test]
    fn a_response_without_a_models_key_yields_an_empty_list() {
        // A 200 with no models is a valid answer, not a parse error.
        assert!(parse("{}").unwrap().is_empty());
    }

    #[test]
    fn internal_identifiers_are_not_advertised_as_models() {
        // Observed live, and none of them is a model a client can call.
        let body = r#"{"models":{
            "gemini-3.8-flash-medium":{"displayName":"real"},
            "chat_20706":{},"chat_23310":{},
            "tab_flash_lite_preview":{},
            "tab_jump_flash_lite_preview":{}
        }}"#;
        let models = parse(body).unwrap();
        assert_eq!(models.len(), 1, "got {:?}", models.iter().map(|m| &m.id).collect::<Vec<_>>());
        assert_eq!(models[0].id, "gemini-3.8-flash-medium");
    }

    #[test]
    fn every_model_family_survives_the_filter() {
        // The reference's filter drops GPT-OSS as collateral; this one must not.
        assert!(is_model_shaped("gemini-3.8-flash"));
        assert!(is_model_shaped("claude-opus-4-6-thinking"));
        assert!(is_model_shaped("gpt-oss-120b-medium"));
    }

    #[test]
    fn the_filter_is_case_insensitive() {
        assert!(is_model_shaped("Gemini-3.8-Flash"));
        assert!(!is_model_shaped("Chat_20706"));
    }

    #[test]
    fn a_malformed_body_is_an_error() {
        assert!(parse("not json").is_err());
    }

    #[test]
    fn an_entry_with_unexpected_fields_is_still_parsed() {
        // Schema drift must not lose a model.
        let body = r#"{"models":{"gemini-9.9":{"displayName":"M","quotaInfo":{"remainingFraction":0.5},"somethingNew":true}}}"#;
        let models = parse(body).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gemini-9.9");
        assert_eq!(models[0].remaining_fraction, Some(0.5));
    }

    #[test]
    fn freshly_stored_models_are_returned_as_fresh() {
        let catalogue = LiveCatalogue::new();
        assert!(catalogue.fresh().is_none());

        catalogue.store(sample());
        assert_eq!(catalogue.fresh().unwrap().len(), 3);
        assert_eq!(catalogue.stale().unwrap().len(), 3);
        assert!(catalogue.age().is_some());
    }

    #[test]
    fn an_expired_entry_is_stale_but_not_fresh() {
        let catalogue = LiveCatalogue::new();
        catalogue.store(sample());
        // Backdate past the TTL.
        {
            let mut cached = catalogue.cached.lock().unwrap();
            cached.as_mut().unwrap().fetched_at =
                Instant::now() - CACHE_TTL - Duration::from_secs(1);
        }

        assert!(catalogue.fresh().is_none(), "expired entries are not fresh");
        assert!(
            catalogue.stale().is_some(),
            "but they remain usable as a fallback"
        );
    }

    #[test]
    fn clearing_empties_the_cache() {
        let catalogue = LiveCatalogue::new();
        catalogue.store(sample());
        catalogue.clear();
        assert!(catalogue.fresh().is_none());
        assert!(catalogue.stale().is_none());
        assert!(catalogue.age().is_none());
    }

    #[test]
    fn storing_replaces_what_was_there() {
        let catalogue = LiveCatalogue::new();
        catalogue.store(sample());
        catalogue.store(vec![LiveModel {
            id: "only".into(),
            display_name: None,
            remaining_fraction: None,
            reset_time: None,
        }]);
        assert_eq!(catalogue.fresh().unwrap().len(), 1);
    }
}
