//! Project context discovery.
//!
//! Every generation call must name a GCP project. Consumer accounts do not have
//! one until the backend provisions it, so the sequence is:
//!
//! 1. `loadCodeAssist` — asks the backend what project and tier this account
//!    has. Usually answers immediately for accounts that have already been
//!    provisioned.
//! 2. `onboardUser` — provisions a project when step 1 came back empty. This is
//!    an LRO, so it is polled.
//! 3. Cached, keyed by the refresh token, for [`CACHE_TTL`].
//!
//! The one thing not to do here is invent a project. `opencode-antigravity-auth`
//! generates a random synthetic id per request, which fragments server-side
//! prompt caching and quota accounting — the outcome is a slower, more
//! rate-limited account for no benefit. The fallback below is a fixed id the CLI
//! also uses, never a random one.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::time::sleep;

use crate::upstream::constants;
use crate::upstream::transport::{TransportError, UpstreamClient};

/// How long a resolved project context is trusted.
pub const CACHE_TTL: Duration = Duration::from_secs(30 * 60);

/// Poll attempts for the onboarding long-running operation.
const ONBOARD_POLL_ATTEMPTS: u32 = 10;

/// Delay between onboarding polls. Matches the reference implementations; the
/// operation typically completes in one or two polls.
const ONBOARD_POLL_INTERVAL: Duration = Duration::from_secs(5);

/// The project context a set of credentials resolves to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectContext {
    /// Project used for generation. Falls back to [`constants::DEFAULT_PROJECT_ID`].
    pub project_id: String,
    /// Project the backend provisioned for us, when it did.
    pub managed_project_id: Option<String>,
    /// Tier from the most recent `loadCodeAssist`, e.g. `free-tier`.
    pub tier_id: Option<String>,
    /// Paid tier, when the account has one.
    pub paid_tier_id: Option<String>,
    /// True when no real project was found and the shared fallback is in use.
    pub used_fallback: bool,
}

impl ProjectContext {
    fn fallback() -> Self {
        Self {
            project_id: constants::DEFAULT_PROJECT_ID.into(),
            managed_project_id: None,
            tier_id: None,
            paid_tier_id: None,
            used_fallback: true,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    #[error("project discovery failed: {0}")]
    Transport(#[from] TransportError),

    #[error("all project discovery endpoints failed; last error: {0}")]
    AllEndpointsFailed(String),

    #[error("onboarding did not complete within {} attempts", ONBOARD_POLL_ATTEMPTS)]
    OnboardingTimedOut,
}

/// Cached project resolution.
struct CacheEntry {
    context: ProjectContext,
    resolved_at: Instant,
}

/// Resolves and caches project contexts.
#[derive(Default)]
pub struct ProjectResolver {
    cache: Mutex<HashMap<String, CacheEntry>>,
}

impl ProjectResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve the project for an account, using the cache when it is still warm.
    ///
    /// `cache_key` should be something stable and not the access token, which
    /// rotates. A hash of the refresh token is the natural choice.
    pub async fn resolve(
        &self,
        client: &UpstreamClient,
        access_token: &str,
        cache_key: &str,
        endpoints: &[String],
    ) -> Result<ProjectContext, ProjectError> {
        if let Some(context) = self.cached(cache_key) {
            return Ok(context);
        }

        let context = self.discover(client, access_token, endpoints).await?;
        self.store(cache_key, context.clone());
        Ok(context)
    }

    fn cached(&self, key: &str) -> Option<ProjectContext> {
        let cache = self.cache.lock().expect("project cache poisoned");
        cache
            .get(key)
            .filter(|entry| entry.resolved_at.elapsed() < CACHE_TTL)
            .map(|entry| entry.context.clone())
    }

    fn store(&self, key: &str, context: ProjectContext) {
        let mut cache = self.cache.lock().expect("project cache poisoned");
        cache.insert(
            key.to_string(),
            CacheEntry {
                context,
                resolved_at: Instant::now(),
            },
        );
    }

    /// Drop a cached entry, forcing re-resolution. Called when the upstream
    /// rejects the project we have been using.
    pub fn invalidate(&self, key: &str) {
        self.cache.lock().expect("project cache poisoned").remove(key);
    }

    /// Walk the endpoint list, returning the first successful resolution.
    async fn discover(
        &self,
        client: &UpstreamClient,
        access_token: &str,
        endpoints: &[String],
    ) -> Result<ProjectContext, ProjectError> {
        let mut last_error = String::from("no endpoints configured");

        for endpoint in endpoints {
            match self.load_code_assist(client, access_token, endpoint).await {
                Ok(Some(context)) => return Ok(context),
                Ok(None) => {
                    // Reached the backend, but the account has no project yet.
                    // Provisioning is worth attempting once, against this endpoint.
                    match self.onboard_user(client, access_token, endpoint).await {
                        Ok(context) => return Ok(context),
                        Err(error) => last_error = error.to_string(),
                    }
                }
                Err(error) => last_error = error.to_string(),
            }
        }

        // Every endpoint failed or refused to provision. Fall through to the
        // shared project rather than failing the account outright: it works for
        // some accounts, and the account layer surfaces the real error if not.
        tracing::warn!(
            error = %last_error,
            fallback_project = constants::DEFAULT_PROJECT_ID,
            "project discovery failed; using fallback project"
        );
        Ok(ProjectContext::fallback())
    }

    /// Ask the backend what this account already has.
    ///
    /// Returns `Ok(None)` when the backend answered but reported no project,
    /// which is the signal to attempt onboarding.
    async fn load_code_assist(
        &self,
        client: &UpstreamClient,
        access_token: &str,
        endpoint: &str,
    ) -> Result<Option<ProjectContext>, ProjectError> {
        let url = format!("{endpoint}{}", constants::API_LOAD_CODE_ASSIST);
        let body = serde_json::json!({
            "metadata": { "ideType": "ANTIGRAVITY" }
        });

        let payload = client
            .post_json_buffered(&url, access_token, body.to_string().as_bytes())
            .await?
            .into_success()?
            .json()?;

        Ok(interpret_load_code_assist(&payload))
    }

    /// Provision a project for an account that does not have one.
    async fn onboard_user(
        &self,
        client: &UpstreamClient,
        access_token: &str,
        endpoint: &str,
    ) -> Result<ProjectContext, ProjectError> {
        let url = format!("{endpoint}{}", constants::API_ONBOARD_USER);
        let tier = self.select_tier(client, access_token, endpoint).await;

        let body = serde_json::json!({ "tierId": tier });
        // The first response sometimes already carries the finished operation.
        let payload = client
            .post_json_buffered(&url, access_token, body.to_string().as_bytes())
            .await?
            .into_success()?
            .json()?;

        if let Some(project_id) = extract_managed_project(&payload) {
            return Ok(ProjectContext {
                project_id: project_id.clone(),
                managed_project_id: Some(project_id),
                tier_id: Some(tier),
                paid_tier_id: None,
                used_fallback: false,
            });
        }

        // Otherwise poll until it completes or we give up.
        for attempt in 1..=ONBOARD_POLL_ATTEMPTS {
            sleep(ONBOARD_POLL_INTERVAL).await;
            // A transient failure mid-poll is not fatal; the operation may still
            // complete, so keep polling until the attempt budget runs out.
            let Ok(response) = client
                .post_json_buffered(&url, access_token, body.to_string().as_bytes())
                .await
            else {
                continue;
            };
            let Ok(payload) = response.json() else {
                continue;
            };

            if payload.get("done").and_then(Value::as_bool) != Some(true) {
                tracing::debug!(attempt, "onboarding still in progress");
                continue;
            }
            if let Some(project_id) = extract_managed_project(&payload) {
                return Ok(ProjectContext {
                    project_id: project_id.clone(),
                    managed_project_id: Some(project_id),
                    tier_id: Some(tier),
                    paid_tier_id: None,
                    used_fallback: false,
                });
            }
        }

        Err(ProjectError::OnboardingTimedOut)
    }

    /// Pick the tier to onboard into: the default entry from `allowedTiers`,
    /// else `free-tier`.
    async fn select_tier(
        &self,
        client: &UpstreamClient,
        access_token: &str,
        endpoint: &str,
    ) -> String {
        const FREE_TIER: &str = "free-tier";

        let url = format!("{endpoint}{}", constants::API_LOAD_CODE_ASSIST);
        let body = serde_json::json!({ "metadata": { "ideType": "ANTIGRAVITY" } });

        let Ok(response) = client
            .post_json_buffered(&url, access_token, body.to_string().as_bytes())
            .await
        else {
            return FREE_TIER.into();
        };
        let Ok(payload) = response.json() else {
            return FREE_TIER.into();
        };

        payload
            .get("allowedTiers")
            .and_then(Value::as_array)
            .and_then(|tiers| {
                tiers
                    .iter()
                    .find(|tier| tier.get("isDefault").and_then(Value::as_bool) == Some(true))
                    .or_else(|| tiers.first())
            })
            .and_then(|tier| tier.get("id").and_then(Value::as_str))
            .map(str::to_string)
            .unwrap_or_else(|| FREE_TIER.into())
    }
}

/// Interpret a `loadCodeAssist` payload.
fn interpret_load_code_assist(payload: &Value) -> Option<ProjectContext> {
    let project_id = payload
        .get("cloudaicompanionProject")
        .and_then(extract_id)?;

    let tier_id = payload.get("currentTier").and_then(extract_id);
    let paid_tier_id = payload.get("paidTier").and_then(extract_id);

    Some(ProjectContext {
        project_id,
        managed_project_id: None,
        tier_id,
        paid_tier_id,
        used_fallback: false,
    })
}

/// Extract a project id from the onboarding operation's `response` object.
fn extract_managed_project(payload: &Value) -> Option<String> {
    payload
        .get("response")
        .and_then(|response| response.get("cloudaicompanionProject"))
        .and_then(extract_id)
        .or_else(|| {
            payload
                .get("cloudaicompanionProject")
                .and_then(extract_id)
        })
}

/// Read an id that the backend may render either as a bare string or as an
/// object with an `id` field. Both shapes appear in the wild — the reference
/// implementations handle each in a different place, so we handle both here.
fn extract_id(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        Value::Object(map) => map
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string),
        _ => None,
    }
}

/// Typed wrapper so callers can log tier information without re-parsing.
pub fn describe_tier(context: &ProjectContext) -> String {
    match (&context.tier_id, &context.paid_tier_id) {
        (Some(tier), Some(paid)) => format!("{tier} (paid: {paid})"),
        (Some(tier), None) => tier.clone(),
        (None, Some(paid)) => format!("unknown (paid: {paid})"),
        (None, None) => "unknown".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn project_as_bare_string_is_understood() {
        let payload = json!({ "cloudaicompanionProject": "my-project-123" });
        let context = interpret_load_code_assist(&payload).unwrap();
        assert_eq!(context.project_id, "my-project-123");
        assert!(!context.used_fallback);
    }

    #[test]
    fn project_as_object_is_understood() {
        let payload = json!({ "cloudaicompanionProject": { "id": "nested-project" } });
        let context = interpret_load_code_assist(&payload).unwrap();
        assert_eq!(context.project_id, "nested-project");
    }

    #[test]
    fn tier_is_parsed_from_both_shapes() {
        let as_string = json!({
            "cloudaicompanionProject": "p",
            "currentTier": "free-tier"
        });
        assert_eq!(
            interpret_load_code_assist(&as_string).unwrap().tier_id,
            Some("free-tier".into())
        );

        let as_object = json!({
            "cloudaicompanionProject": "p",
            "currentTier": { "id": "pro-tier" },
            "paidTier": { "id": "pro" }
        });
        let context = interpret_load_code_assist(&as_object).unwrap();
        assert_eq!(context.tier_id, Some("pro-tier".into()));
        assert_eq!(context.paid_tier_id, Some("pro".into()));
    }

    #[test]
    fn missing_project_signals_onboarding_is_needed() {
        let payload = json!({ "allowedTiers": [{ "id": "free-tier", "isDefault": true }] });
        assert!(interpret_load_code_assist(&payload).is_none());
    }

    #[test]
    fn empty_project_string_is_not_a_project() {
        let payload = json!({ "cloudaicompanionProject": "" });
        assert!(interpret_load_code_assist(&payload).is_none());
    }

    #[test]
    fn empty_object_id_is_not_a_project() {
        let payload = json!({ "cloudaicompanionProject": { "id": "" } });
        assert!(interpret_load_code_assist(&payload).is_none());
    }

    #[test]
    fn managed_project_is_read_from_operation_response() {
        let payload = json!({
            "done": true,
            "response": { "cloudaicompanionProject": { "id": "onboarded-project" } }
        });
        assert_eq!(
            extract_managed_project(&payload),
            Some("onboarded-project".into())
        );
    }

    #[test]
    fn managed_project_is_read_from_bare_string_too() {
        let payload = json!({
            "done": true,
            "response": { "cloudaicompanionProject": "bare-project" }
        });
        assert_eq!(extract_managed_project(&payload), Some("bare-project".into()));
    }

    #[test]
    fn unfinished_operation_yields_no_project() {
        let payload = json!({ "name": "operations/abc", "done": false });
        assert!(extract_managed_project(&payload).is_none());
    }

    #[test]
    fn fallback_context_is_marked_and_uses_the_shared_project() {
        let context = ProjectContext::fallback();
        assert!(context.used_fallback);
        assert_eq!(context.project_id, constants::DEFAULT_PROJECT_ID);
        assert!(context.managed_project_id.is_none());
    }

    #[test]
    fn cache_returns_stored_context_and_honours_invalidation() {
        let resolver = ProjectResolver::new();
        assert!(resolver.cached("k").is_none());

        resolver.store(
            "k",
            ProjectContext {
                project_id: "cached".into(),
                managed_project_id: None,
                tier_id: None,
                paid_tier_id: None,
                used_fallback: false,
            },
        );
        assert_eq!(resolver.cached("k").unwrap().project_id, "cached");

        resolver.invalidate("k");
        assert!(resolver.cached("k").is_none());
    }

    #[test]
    fn expired_cache_entries_are_not_returned() {
        let resolver = ProjectResolver::new();
        {
            let mut cache = resolver.cache.lock().unwrap();
            cache.insert(
                "k".into(),
                CacheEntry {
                    context: ProjectContext::fallback(),
                    resolved_at: Instant::now() - CACHE_TTL - Duration::from_secs(1),
                },
            );
        }
        assert!(resolver.cached("k").is_none());
    }

    #[test]
    fn tier_description_covers_all_combinations() {
        let base = ProjectContext::fallback();
        assert_eq!(describe_tier(&base), "unknown");

        let free = ProjectContext {
            tier_id: Some("free-tier".into()),
            ..base.clone()
        };
        assert_eq!(describe_tier(&free), "free-tier");

        let paid = ProjectContext {
            tier_id: Some("free-tier".into()),
            paid_tier_id: Some("pro".into()),
            ..base.clone()
        };
        assert_eq!(describe_tier(&paid), "free-tier (paid: pro)");
    }
}
