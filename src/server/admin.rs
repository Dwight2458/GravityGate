//! Operator endpoints.
//!
//! These exist because the interesting failure modes of this gateway are
//! account-shaped: an account awaiting verification, one that has been banned,
//! one that is rate limited until a specific time. None of that is visible from
//! the request path, and without surfacing it an operator's only tool is reading
//! logs.
//!
//! `/health` deliberately reports each account's *state vocabulary* rather than a
//! boolean: "rate limited for 12 more minutes" and "banned" call for completely
//! different responses, and collapsing them into `isInvalid` is what makes the
//! reference implementations hard to operate.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::accounts::account::{Account, CooldownReason};

use super::SharedState;
use super::error;

/// Per-account status, with an explicit condition rather than a flag.
#[derive(Debug, Serialize)]
pub struct AccountStatus {
    label: String,
    credential_id: String,
    enabled: bool,
    /// `ready`, `disabled`, `ineligible`, `verify`, `cooling`, or `limited`.
    status: &'static str,
    detail: Option<String>,
    /// Where to send an account holder who has to clear a verification demand.
    #[serde(skip_serializing_if = "Option::is_none")]
    verification_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    project_id: Option<String>,
    /// Seconds until the account becomes usable, when a limit is what blocks it.
    #[serde(skip_serializing_if = "Option::is_none")]
    available_in_seconds: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct Health {
    status: &'static str,
    accounts_configured: usize,
    accounts_available: usize,
    accounts: Vec<AccountStatus>,
    /// Whether any account at all can serve a request right now.
    serving: bool,
    signatures_cached: usize,
}

/// `GET /health`.
pub async fn health(State(state): State<SharedState>) -> Response {
    let now = crate::accounts::account::now_ms();
    let snapshot = state.engine.accounts.snapshot();

    let accounts: Vec<AccountStatus> = snapshot
        .accounts
        .iter()
        .map(|account| describe(account, now))
        .collect();
    let available = snapshot
        .accounts
        .iter()
        .filter(|account| account.is_available(now))
        .count();

    let stats = state.engine.signatures.stats();
    let health = Health {
        // `degraded` rather than `unhealthy`: some accounts working is a working
        // gateway, and a healthcheck that fails on a partial outage would take
        // the last working accounts out of service too.
        status: if available == 0 {
            "unhealthy"
        } else if available < snapshot.accounts.len() {
            "degraded"
        } else {
            "healthy"
        },
        accounts_configured: snapshot.accounts.len(),
        accounts_available: available,
        accounts,
        serving: available > 0,
        signatures_cached: stats.tool_signatures + stats.session_signatures,
    };

    // 503 only when nothing can serve, so a load balancer stops sending traffic
    // to a gateway that cannot do anything with it.
    let status = if health.serving {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(health)).into_response()
}

/// Describe one account's condition.
fn describe(account: &Account, now: i64) -> AccountStatus {
    let (status, detail, available_in) = condition(account, now);
    AccountStatus {
        label: account.label(),
        credential_id: account.credential_id().chars().take(8).collect(),
        enabled: account.enabled,
        status,
        detail,
        verification_url: account.verification_url.clone(),
        tier: account.captured_tier_id.clone(),
        project_id: account.project_id.clone(),
        available_in_seconds: available_in,
    }
}

/// Reduce an account to one condition, most severe first.
///
/// Ordered deliberately: a ban outranks a verification demand, which outranks a
/// rate limit. An operator scanning this should see the worst thing.
fn condition(account: &Account, now: i64) -> (&'static str, Option<String>, Option<u64>) {
    if !account.enabled {
        return (
            "disabled",
            Some("removed from rotation by an operator".into()),
            None,
        );
    }
    if account.account_ineligible {
        return ("ineligible", account.account_ineligible_reason.clone(), None);
    }
    if account.verification_required {
        return (
            "verify",
            account
                .verification_required_reason
                .clone()
                .or(Some("account holder verification required".into())),
            None,
        );
    }
    if let Some(until) = account.cooling_down_until.filter(|until| *until > now) {
        let reason = account
            .cooldown_reason
            .map(describe_cooldown)
            .unwrap_or("cooling down");
        return (
            "cooling",
            Some(reason.to_string()),
            Some(((until - now) / 1000).max(0) as u64),
        );
    }
    if let Some(reset) = account.next_reset_at(now) {
        let pools: Vec<&str> = account
            .rate_limit_reset_times
            .iter()
            .filter(|(_, value)| **value > now)
            .map(|(key, _)| key.as_str())
            .collect();
        return (
            "limited",
            Some(format!("rate limited on {}", pools.join(", "))),
            Some(((reset - now) / 1000).max(0) as u64),
        );
    }
    ("ready", None, None)
}

fn describe_cooldown(reason: CooldownReason) -> &'static str {
    match reason {
        CooldownReason::AuthFailure => "paused after authentication failures",
        CooldownReason::NetworkError => "paused after network failures",
        CooldownReason::ProjectError => "paused after project errors",
        CooldownReason::ValidationRequired => "paused pending verification",
    }
}

/// `GET /account-limits`.
pub async fn account_limits(State(state): State<SharedState>) -> Response {
    let now = crate::accounts::account::now_ms();
    let snapshot = state.engine.accounts.snapshot();

    let limits: Vec<serde_json::Value> = snapshot
        .accounts
        .iter()
        .map(|account| {
            let pools: serde_json::Map<String, serde_json::Value> = account
                .rate_limit_reset_times
                .iter()
                .map(|(pool, reset)| {
                    let remaining = (reset - now).max(0) / 1000;
                    (
                        pool.clone(),
                        serde_json::json!({ "resets_in_seconds": remaining }),
                    )
                })
                .collect();
            serde_json::json!({
                "account": account.label(),
                "credential_id": account.credential_id().chars().take(8).collect::<String>(),
                "tier": account.captured_tier_id,
                "pools": pools,
            })
        })
        .collect();

    (StatusCode::OK, Json(serde_json::json!({ "accounts": limits }))).into_response()
}

/// `POST /refresh-token`.
///
/// Drops the in-memory caches. Used after fixing something out of band — an
/// account removed, a verification cleared — where waiting for expiry would be
/// the only alternative.
pub async fn refresh_token(State(state): State<SharedState>) -> Response {
    state.engine.credentials.clear();
    state.engine.signatures.clear();
    state.engine.sessions.clear();

    tracing::info!("caches cleared by /refresh-token");
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "detail": "access tokens, project contexts, and signatures will be refetched",
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accounts::account::{Account, CooldownReason};

    const NOW: i64 = 1_800_000_000_000;

    #[test]
    fn a_fresh_account_is_ready() {
        let (status, detail, wait) = condition(&Account::new("t"), NOW);
        assert_eq!(status, "ready");
        assert!(detail.is_none());
        assert!(wait.is_none());
    }

    #[test]
    fn a_ban_outranks_everything_else() {
        let mut account = Account::new("t");
        account.mark_rate_limited("gemini", NOW + 60_000);
        account.mark_verification_required(None, "check");
        account.mark_ineligible("banned for terms");

        let (status, detail, _) = condition(&account, NOW);
        assert_eq!(status, "ineligible");
        assert!(detail.unwrap().contains("banned"));
    }

    #[test]
    fn a_verification_demand_outranks_a_rate_limit() {
        let mut account = Account::new("t");
        account.mark_rate_limited("gemini", NOW + 60_000);
        account.mark_verification_required(Some("https://example.com".into()), "verify");

        assert_eq!(condition(&account, NOW).0, "verify");
    }

    #[test]
    fn a_rate_limit_reports_how_long_it_lasts() {
        let mut account = Account::new("t");
        account.mark_rate_limited("claude", NOW + 90_000);

        let (status, detail, wait) = condition(&account, NOW);
        assert_eq!(status, "limited");
        assert_eq!(wait, Some(90));
        assert!(detail.unwrap().contains("claude"));
    }

    #[test]
    fn a_cooldown_reports_its_reason() {
        let mut account = Account::new("t");
        account.mark_cooling_down(NOW + 30_000, CooldownReason::NetworkError);

        let (status, detail, wait) = condition(&account, NOW);
        assert_eq!(status, "cooling");
        assert_eq!(wait, Some(30));
        assert!(detail.unwrap().contains("network"));
    }

    #[test]
    fn an_expired_cooldown_is_ready() {
        let mut account = Account::new("t");
        account.mark_cooling_down(NOW - 1, CooldownReason::AuthFailure);
        assert_eq!(condition(&account, NOW).0, "ready");
    }

    #[test]
    fn a_disabled_account_is_reported_as_such() {
        let mut account = Account::new("t");
        account.enabled = false;
        assert_eq!(condition(&account, NOW).0, "disabled");
    }

    #[test]
    fn a_disabled_account_outranks_its_other_conditions() {
        let mut account = Account::new("t");
        account.enabled = false;
        account.mark_ineligible("banned");
        // Disabled is what an operator did, so it is what they should see.
        assert_eq!(condition(&account, NOW).0, "disabled");
    }

    #[test]
    fn the_verification_url_is_surfaced_for_an_operator() {
        let mut account = Account::new("t");
        account.mark_verification_required(
            Some("https://accounts.google.com/signin/continue?plt=x".into()),
            "verify",
        );
        let status = describe(&account, NOW);
        assert_eq!(
            status.verification_url.as_deref(),
            Some("https://accounts.google.com/signin/continue?plt=x")
        );
    }

    #[test]
    fn the_credential_id_is_truncated_for_display() {
        let status = describe(&Account::new("t"), NOW);
        assert_eq!(status.credential_id.len(), 8);
    }

    #[test]
    fn the_label_is_used_rather_than_the_token() {
        let mut account = Account::new("1//secret-refresh-token");
        account.email = Some("someone@example.com".into());
        let status = describe(&account, NOW);
        assert_eq!(status.label, "someone@example.com");
        assert!(!format!("{status:?}").contains("secret"));
    }
}

// ---------------------------------------------------------------------------
// Observability surface
// ---------------------------------------------------------------------------

/// `GET /metrics`.
///
/// Prometheus text format. Returns 404 when metrics are disabled, so a scraper
/// learns the endpoint is off rather than scraping an empty body forever.
pub async fn metrics(State(state): State<SharedState>) -> Response {
    // Publish the account conditions on read rather than on a timer: they change
    // rarely, and computing them here means the gauge is current whenever anyone
    // looks, with no background task to keep alive.
    publish_account_counts(&state);

    match state.observability.render_metrics() {
        Some(body) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            body,
        )
            .into_response(),
        None => error::ApiError::new(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "metrics are disabled",
        )
        .into_response(),
    }
}

/// Refresh the account-condition gauge from the live pool.
fn publish_account_counts(state: &SharedState) {
    let now = crate::accounts::account::now_ms();
    let snapshot = state.engine.accounts.snapshot();

    let mut counts = std::collections::BTreeMap::new();
    for account in &snapshot.accounts {
        let condition = condition(account, now).0;
        *counts.entry(condition).or_insert(0u64) += 1;
    }
    // Publish zeroes for conditions that are currently empty, so a graph shows a
    // line dropping to zero rather than a gap.
    for condition in ["ready", "limited", "cooling", "verify", "ineligible", "disabled"] {
        counts.entry(condition).or_insert(0);
    }

    let pairs: Vec<(&str, u64)> = counts.into_iter().collect();
    crate::observ::metrics::set_account_counts(&pairs);
    crate::observ::metrics::set_signatures_cached(
        state.engine.signatures.stats().tool_signatures
            + state.engine.signatures.stats().session_signatures,
    );
}

/// `GET /api/stats` — aggregate traffic over a window.
pub async fn stats(State(state): State<SharedState>) -> Response {
    let window = stats_window();
    match state.observability.audit() {
        Some(log) => match log.totals(window) {
            Ok(totals) => (
                StatusCode::OK,
                Json(serde_json::json!({
                    "window_seconds": window.as_secs(),
                    "requests": totals.requests,
                    "ok": totals.ok,
                    "errors": totals.errors,
                    "prompt_tokens": totals.prompt_tokens,
                    "completion_tokens": totals.completion_tokens,
                    "cached_tokens": totals.cached_tokens,
                    "reasoning_tokens": totals.reasoning_tokens,
                    "mean_latency_ms": totals.mean_latency_ms,
                })),
            )
                .into_response(),
            Err(error) => unavailable(error),
        },
        None => audit_disabled(),
    }
}

/// `GET /api/stats/accounts` — the same aggregates per account.
pub async fn stats_by_account(State(state): State<SharedState>) -> Response {
    let window = stats_window();
    match state.observability.audit() {
        Some(log) => match log.per_account(window) {
            Ok(summaries) => {
                let rows: Vec<serde_json::Value> = summaries
                    .into_iter()
                    .map(|summary| {
                        serde_json::json!({
                            "account_id": summary.account_id,
                            "requests": summary.requests,
                            "errors": summary.errors,
                            "mean_latency_ms": summary.mean_latency_ms,
                        })
                    })
                    .collect();
                (StatusCode::OK, Json(serde_json::json!({ "accounts": rows }))).into_response()
            }
            Err(error) => unavailable(error),
        },
        None => audit_disabled(),
    }
}

/// `GET /api/requests` — the most recent requests.
pub async fn requests(
    State(state): State<SharedState>,
    axum::extract::Query(query): axum::extract::Query<RecentQuery>,
) -> Response {
    match state.observability.audit() {
        Some(log) => match log.recent(query.limit()) {
            Ok(records) => (StatusCode::OK, Json(serde_json::json!({ "requests": records })))
                .into_response(),
            Err(error) => unavailable(error),
        },
        None => audit_disabled(),
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct RecentQuery {
    limit: Option<usize>,
}

impl RecentQuery {
    /// Clamp the page size so a dashboard typo cannot ask for everything.
    fn limit(&self) -> usize {
        self.limit.unwrap_or(50).clamp(1, 500)
    }
}

/// How far back the aggregates look.
fn stats_window() -> std::time::Duration {
    std::time::Duration::from_secs(60 * 60)
}

fn unavailable(error: crate::observ::audit::AuditError) -> Response {
    error::ApiError::unavailable(format!("the audit log is unavailable: {error}")).into_response()
}

fn audit_disabled() -> Response {
    error::ApiError::new(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "the audit log is disabled",
    )
    .into_response()
}

/// `GET /` — the dashboard.
///
/// Embedded rather than served from disk so a deployment is a single binary with
/// no working directory to get wrong.
pub async fn dashboard() -> Response {
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("dashboard.html"),
    )
        .into_response()
}
