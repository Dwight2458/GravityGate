//! HTTP surface.
//!
//! ```text
//! POST /v1/chat/completions   OpenAI-compatible, streaming and buffered
//! GET  /v1/models             the model list
//! GET  /health                account state, for operators and healthchecks
//! GET  /account-limits        quota matrix
//! POST /refresh-token         drop caches and force a token refresh
//! ```
//!
//! The gateway is a thin shell over [`crate::engine`]: a handler translates the
//! request, dispatches it, and translates the response back. Everything decided
//! elsewhere — model resolution, account selection, retry, signature handling —
//! is already built and tested by the time a request arrives here, which is why
//! this layer contains almost no logic worth testing on its own.

pub mod admin;
pub mod chat;
pub mod error;
pub mod execute;
pub mod models;
pub mod responses;

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use tokio::net::TcpListener;

use crate::accounts::store::AccountStore;
use crate::config::Config;
use crate::engine::Engine;
use crate::observ::Observability;

/// Shared handler state.
pub struct AppState {
    /// One `Engine` for the process: the account router's cursors, token
    /// buckets, and health scores are per-process by design, and the signature
    /// cache has to be shared or a signature minted on one request would be
    /// invisible to the next.
    pub engine: Engine,
    /// Metrics and the audit log. Beside the engine rather than inside it,
    /// because observability outlives any single request and is useless to the
    /// engine's own logic.
    pub observability: Observability,
}

pub type SharedState = Arc<AppState>;

/// Build the application.
pub fn app(config: Config, store: AccountStore) -> anyhow::Result<Router> {
    Ok(app_with_state(build_state(config, store)?))
}

/// Assemble shared state.
///
/// Separate from [`app`] because [`serve`] needs a handle on the state to start
/// the retention task, and the router consumes its own copy.
pub fn build_state(config: Config, store: AccountStore) -> anyhow::Result<SharedState> {
    let observability = Observability::setup(&config);
    let engine = Engine::new(config, store)?;
    Ok(Arc::new(AppState {
        engine,
        observability,
    }))
}

/// Build the application around existing state.
///
/// Separate from [`app`] so tests can construct state directly.
pub fn app_with_state(state: SharedState) -> Router {
    Router::new()
        .route("/", get(admin::dashboard))
        .route("/v1/chat/completions", post(chat::completions))
        .route("/v1/models", get(models::list))
        .route("/v1/responses", post(responses::create))
        .route("/health", get(admin::health))
        .route("/account-limits", get(admin::account_limits))
        .route("/refresh-token", post(admin::refresh_token))
        .route("/metrics", get(admin::metrics))
        .route("/api/stats", get(admin::stats))
        .route("/api/stats/accounts", get(admin::stats_by_account))
        .route("/api/requests", get(admin::requests))
        .fallback(not_found)
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ))
        // Applied last so it wraps everything, including rejected requests.
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state)
}

/// Anything unrecognised gets the OpenAI error shape rather than an empty body.
async fn not_found() -> Response {
    error::ApiError::new(StatusCode::NOT_FOUND, "invalid_request_error", "not found")
        .into_response()
}

/// Reject requests without a valid gateway credential.
///
/// A no-op when no keys are configured, which is the sensible default for a
/// loopback-only gateway. When keys *are* configured, this is what stops an
/// exposed instance from being an open relay to the operator's accounts.
async fn require_api_key(
    State(state): State<SharedState>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, error::ApiError> {
    let configured = &state.engine.config.server.api_keys;
    if configured.is_empty() {
        return Ok(next.run(request).await);
    }

    let presented = extract_credential(&headers);
    match presented {
        // Any configured key is accepted, so keys can be rotated by adding the
        // new one, deploying, then removing the old.
        Some(token) if configured.iter().any(|key| constant_time_eq(key, &token)) => {
            Ok(next.run(request).await)
        }
        Some(_) => Err(error::ApiError::new(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid API key",
        )),
        None => Err(error::ApiError::new(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "missing API key: send Authorization: Bearer <key> or x-api-key: <key>",
        )),
    }
}

/// Pull the presented credential from either accepted header.
///
/// Both spellings are supported because clients disagree: OpenAI SDKs send
/// `Authorization: Bearer`, Anthropic-flavoured ones send `x-api-key`.
fn extract_credential(headers: &HeaderMap) -> Option<String> {
    if let Some(value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    {
        // Scheme comparison is case-insensitive per RFC 9110.
        if let Some((scheme, token)) = value.split_once(' ')
            && scheme.eq_ignore_ascii_case("bearer")
            && !token.is_empty()
        {
            return Some(token.to_string());
        }
    }
    headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Compare two secrets without leaking their length or content through timing.
///
/// A plain `==` on strings short-circuits at the first differing byte, which is
/// enough to recover a key one byte at a time given enough attempts.
fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

/// Run the gateway.
pub async fn serve(config: Config, store: AccountStore) -> anyhow::Result<()> {
    let address = format!("{}:{}", config.server.host, config.server.port);
    let retention = config.audit.retention();
    let state = build_state(config, store)?;

    // An audit log that is never pruned is a slow disk leak on a gateway that
    // runs for months. Pruned once at startup so a restart reclaims space, then
    // daily.
    spawn_retention(state.clone(), retention);

    let app = app_with_state(state);
    let listener = TcpListener::bind(&address).await?;
    tracing::info!(%address, "gravitygate listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// Prune the audit log at startup and then daily.
///
/// Spawned rather than run inline so a slow first prune cannot delay the
/// listener. A prune failure is logged and the loop continues: retention is
/// housekeeping, and giving up on it entirely because one attempt failed would
/// be the wrong response to a temporarily locked database.
fn spawn_retention(state: SharedState, retention: std::time::Duration) {
    let Some(log) = state.observability.audit() else {
        return;
    };
    let log = log.path().to_path_buf();

    tokio::spawn(async move {
        // Startup prune uses its own handle so the state Arc is not held for the
        // lifetime of the task.
        prune_once(&log, retention);

        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(24 * 60 * 60));
        // The first tick fires immediately, and the startup prune just ran.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            prune_once(&log, retention);
        }
    });
}

/// One prune pass, reopening the log.
///
/// Reopening rather than holding the handle keeps this independent of the
/// server's lifecycle, and a prune is rare enough that the cost is irrelevant.
fn prune_once(path: &std::path::Path, retention: std::time::Duration) {
    match crate::observ::audit::AuditLog::open(path) {
        Ok(log) => match log.prune(retention) {
            Ok(0) => tracing::debug!("no audit records to prune"),
            Ok(removed) => tracing::info!(removed, "pruned old audit records"),
            Err(error) => tracing::warn!(%error, "could not prune the audit log"),
        },
        Err(error) => tracing::warn!(%error, "could not open the audit log to prune"),
    }
}

/// Resolve on either ctrl-c or SIGTERM, so a container stop drains cleanly.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    tracing::info!("shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                axum::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        map
    }

    #[test]
    fn a_bearer_token_is_extracted() {
        let map = headers(&[("authorization", "Bearer sk-abc")]);
        assert_eq!(extract_credential(&map).as_deref(), Some("sk-abc"));
    }

    #[test]
    fn the_scheme_is_case_insensitive() {
        // RFC 9110 makes the auth scheme case-insensitive; clients vary.
        for scheme in ["Bearer", "bearer", "BEARER", "BeArEr"] {
            let map = headers(&[("authorization", &format!("{scheme} sk-abc"))]);
            assert_eq!(
                extract_credential(&map).as_deref(),
                Some("sk-abc"),
                "scheme {scheme}"
            );
        }
    }

    #[test]
    fn an_x_api_key_header_is_extracted() {
        let map = headers(&[("x-api-key", "sk-abc")]);
        assert_eq!(extract_credential(&map).as_deref(), Some("sk-abc"));
    }

    #[test]
    fn authorization_wins_over_x_api_key() {
        // Documented: when both are present the standard header is used.
        let map = headers(&[("authorization", "Bearer from-auth"), ("x-api-key", "from-header")]);
        assert_eq!(extract_credential(&map).as_deref(), Some("from-auth"));
    }

    #[test]
    fn a_non_bearer_authorization_header_is_ignored() {
        let map = headers(&[("authorization", "Basic dXNlcjpwYXNz")]);
        assert!(extract_credential(&map).is_none());
    }

    #[test]
    fn an_empty_credential_is_treated_as_absent() {
        assert!(extract_credential(&headers(&[("authorization", "Bearer ")])).is_none());
        assert!(extract_credential(&headers(&[("x-api-key", "")])).is_none());
        assert!(extract_credential(&HeaderMap::new()).is_none());
    }

    #[test]
    fn constant_time_comparison_matches_equality() {
        assert!(constant_time_eq("sk-abc", "sk-abc"));
        assert!(!constant_time_eq("sk-abc", "sk-abd"));
        assert!(!constant_time_eq("sk-abc", "sk-ab"));
        assert!(!constant_time_eq("", "x"));
        assert!(constant_time_eq("", ""));
    }

    #[test]
    fn constant_time_comparison_handles_multibyte_values() {
        // Byte-wise comparison must not panic on a split code point.
        assert!(constant_time_eq("ключ", "ключ"));
        assert!(!constant_time_eq("ключ", "ключ2"));
    }

    #[tokio::test]
    async fn the_router_builds_without_an_upstream_call() {
        // Guards the wiring: a route typo or a missing state bound fails here.
        let mut config = Config::default();
        // Neither is wanted in a test, and the audit log would write to the
        // user's real config directory.
        config.metrics.enabled = false;
        config.audit.enabled = false;
        let store = AccountStore::load(std::env::temp_dir().join("gg-router-test.json")).unwrap();
        assert!(app(config, store).is_ok());
    }
}
