use super::*;
use crate::{
    accounts::store::AccountStore, config::Config, server::SharedState, upstream::constants,
};
use axum::{Json, Router, body::Body, http::StatusCode, response::IntoResponse, routing::post};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

const GOOD: &str = "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]}}]}}\n\ndata: {\"response\":{\"candidates\":[{\"finishReason\":\"STOP\"}]}}\n\n";

struct Fixture {
    state: SharedState,
    path: std::path::PathBuf,
    mode: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

impl Fixture {
    async fn new(mode: usize) -> Self {
        let setting = Arc::new(AtomicUsize::new(mode));
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = calls.clone();
        let response_mode = setting.clone();
        let app = Router::new()
            .route(constants::API_LOAD_CODE_ASSIST, post(|| async {
                Json(json!({"cloudaicompanionProject": "test-project", "currentTier": {"id": "free-tier"}}))
            }))
            .route(constants::API_STREAM_GENERATE, post(move || {
                let seen = seen.clone();
                let mode = response_mode.load(Ordering::SeqCst);
                async move {
                    seen.fetch_add(1, Ordering::SeqCst);
                    match mode {
                        1 => (StatusCode::FORBIDDEN, Json(json!({"error":{"message":"VALIDATION_REQUIRED", "details":[{"metadata":{"validation_url":"https://accounts.google.com/new-check"}}]}}))).into_response(),
                        2 => "".into_response(),
                        3 => "data: broken-json\n\n".into_response(),
                        4 => {
                            let body = futures::stream::pending::<Result<bytes::Bytes, std::io::Error>>();
                            axum::response::Response::new(Body::from_stream(body))
                        }
                        5 => (StatusCode::TOO_MANY_REQUESTS, [("retry-after", "600")], Json(json!({"error":{"message":"rate limited", "details":[{"reason":"RATE_LIMIT_EXCEEDED"}]}}))).into_response(),
                        6 => GOOD.replace("\"STOP\"", "\"MAX_TOKENS\"").into_response(),
                        7 => {
                            let chunks = futures::stream::iter([
                                Ok(bytes::Bytes::from_static(GOOD.as_bytes())),
                                Err(std::io::Error::other("broken response stream")),
                            ]);
                            axum::response::Response::new(Body::from_stream(chunks))
                        }
                        8 => format!("{GOOD}data: {{\"error\":{{\"message\":\"failed\"}}}}\n\n").into_response(),
                        _ => GOOD.into_response(),
                    }
                }
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let path =
            std::env::temp_dir().join(format!("gg-verification-{}.json", uuid::Uuid::new_v4()));
        let store = AccountStore::load(&path).unwrap();
        store
            .mutate(|s| {
                let mut account = Account::new("test-token-held");
                account.email = Some("held@example.com".into());
                account.mark_verification_required(
                    Some("https://accounts.google.com/old-check".into()),
                    "asked",
                );
                account.verification_recheck_at = Some(now_ms() - 1);
                s.accounts.push(account);
                s.accounts.push(Account::new("test-token-healthy"));
                true
            })
            .unwrap();
        let mut config = Config::default();
        config.upstream.endpoints = vec![endpoint];
        config.audit.enabled = false;
        config.metrics.enabled = false;
        config.verification.timeout_secs = if mode == 4 { 1 } else { 10 };
        config.routing.max_wait_before_error_secs = 0;
        let state = crate::server::build_state(config, store).unwrap();
        for account in &state.engine.accounts.snapshot().accounts {
            state.engine.credentials.seed_for_test(account);
        }
        Self {
            state,
            path,
            mode: setting,
            calls,
            task,
        }
    }

    fn held(&self) -> Account {
        self.state.engine.accounts.snapshot().accounts[0].clone()
    }

    fn allow_another_check(&self) {
        self.state
            .engine
            .accounts
            .mutate(|s| {
                s.accounts[0].verification_check.started_at = Some(now_ms() - 16_000);
                s.accounts[0].verification_recheck_at = Some(now_ms() - 1);
                true
            })
            .unwrap();
    }

    async fn wait_for_call(&self) {
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while self.calls.load(Ordering::SeqCst) == 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
        let _ = std::fs::remove_file(&self.path);
    }
}

#[tokio::test]
async fn automatic_check_recovers_bypasses_old_health_and_stops_when_ready() {
    let f = Fixture::new(0).await;
    let id = f.held().credential_id();
    for _ in 0..12 {
        f.state.engine.router.record_failure(&id, now_ms());
    }
    assert!(f.state.engine.router.health(&id, now_ms()) < 50.0);
    f.state.engine.recheck_due_verifications().await.unwrap();
    let account = f.held();
    assert!(!account.verification_required, "check={:?}, calls={}", account.verification_check, f.calls.load(Ordering::SeqCst));
    assert!(account.verification_url.is_none());
    assert_eq!(view(&account, now_ms()).state, "recovered");
    assert!(f.state.engine.router.health(&id, now_ms()) >= 50.0);
    assert!(account.is_available(now_ms()));
    let external = AccountStore::load(&f.path).unwrap();
    assert!(!external.snapshot().accounts[0].verification_required);
    f.state.engine.recheck_due_verifications().await.unwrap();
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
    assert_eq!(f.state.engine.accounts.snapshot().accounts[1].last_used, 0);
}

#[tokio::test]
async fn rejection_uses_bounded_backoff_and_new_link_without_rotating() {
    let f = Fixture::new(1).await;
    for seconds in [60, 120, 300, 300] {
        f.allow_another_check();
        let result = f
            .state
            .engine
            .recheck_verification("held@example.com", true)
            .await
            .unwrap();
        assert_eq!(result.state, "waiting");
        let wait = result.next_check_at.unwrap() - result.last_checked_at.unwrap();
        assert_eq!(wait, seconds * 1000);
        assert!(
            f.held().verification_url.unwrap().contains("new-check"),
            "{result:?}"
        );
        // Timer scans must not bypass the newly established delay.
        let calls = f.calls.load(Ordering::SeqCst);
        f.state.engine.recheck_due_verifications().await.unwrap();
        assert_eq!(f.calls.load(Ordering::SeqCst), calls);
    }
    assert_eq!(f.calls.load(Ordering::SeqCst), 4);
    assert!(!f.state.engine.accounts.snapshot().accounts[1].verification_required);
}

#[tokio::test]
async fn successful_headers_empty_malformed_or_unfinished_streams_do_not_recover() {
    let f = Fixture::new(2).await;
    for mode in [2, 3, 6, 7, 8] {
        f.mode.store(mode, Ordering::SeqCst);
        f.allow_another_check();
        let result = f
            .state
            .engine
            .recheck_verification("held@example.com", true)
            .await
            .unwrap();
        assert_eq!(result.state, "waiting");
        assert!(f.held().verification_required);
        assert!(f.held().verification_check.id.is_none());
        assert!(result.recovered_at.is_none());
    }
}

#[tokio::test]
async fn duplicate_clicks_do_not_send_more_probes_and_other_account_can_serve() {
    let f = Fixture::new(4).await;
    let state = f.state.clone();
    let task = tokio::spawn(async move {
        state
            .engine
            .recheck_verification("held@example.com", true)
            .await
    });
    f.wait_for_call().await;
    assert_eq!(view(&f.held(), now_ms()).state, "checking");
    assert!(!f.held().is_eligible(now_ms()));
    let other_store = AccountStore::load(&f.path).unwrap();
    let other = Engine::new(f.state.engine.config.clone(), other_store).unwrap();
    assert!(matches!(
        other.recheck_verification("held@example.com", true).await,
        Err(CheckError::Busy)
    ));
    assert!(matches!(
        f.state
            .engine
            .recheck_verification("held@example.com", true)
            .await,
        Err(CheckError::Busy)
    ));
    f.mode.store(0, Ordering::SeqCst);
    let request = serde_json::from_value(
        json!({"model":"gemini-3.8-flash-low", "messages":[{"role":"user","content":"ok"}]}),
    )
    .unwrap();
    let prepared = f.state.engine.prepare(&request).unwrap();
    let call = f.state.engine.dispatch(&prepared).await.unwrap();
    assert_ne!(call.credential_id, f.held().credential_id());
    task.abort();
    let _ = task.await;
    assert_eq!(
        f.held().verification_check.last_result.as_deref(),
        Some("cancelled")
    );
    assert!(f.held().verification_check.id.is_none());
}

#[tokio::test]
async fn timeout_cancels_underlying_read_and_releases_lease() {
    let f = Fixture::new(4).await;
    let result = f
        .state
        .engine
        .recheck_verification("held@example.com", true)
        .await
        .unwrap();
    assert_eq!(result.last_result.as_deref(), Some("timeout"));
    assert_eq!(result.state, "waiting");
    assert!(f.held().verification_check.id.is_none());
    f.mode.store(0, Ordering::SeqCst);
    f.allow_another_check();
    assert_eq!(
        f.state
            .engine
            .recheck_verification("held@example.com", true)
            .await
            .unwrap()
            .state,
        "recovered"
    );
}

#[tokio::test]
async fn a_late_result_cannot_overwrite_operator_changes_or_resurrect_removed_accounts() {
    let f = Fixture::new(4).await;
    let state = f.state.clone();
    let task = tokio::spawn(async move {
        state
            .engine
            .recheck_verification("held@example.com", true)
            .await
    });
    f.wait_for_call().await;
    let external = AccountStore::load(&f.path).unwrap();
    external
        .mutate(|s| {
            s.accounts.remove(0);
            s.accounts[0].enabled = false;
            true
        })
        .unwrap();
    let result = task.await.unwrap().unwrap();
    assert_eq!(result.state, "removed");
    assert_eq!(f.state.engine.accounts.snapshot().accounts.len(), 1);
    assert!(!f.state.engine.accounts.snapshot().accounts[0].enabled);
}

#[tokio::test]
async fn operator_reset_keeps_an_active_probe_reserved_and_discards_its_result() {
    let f = Fixture::new(4).await;
    let state = f.state.clone();
    let task = tokio::spawn(async move {
        state
            .engine
            .recheck_verification("held@example.com", true)
            .await
    });
    f.wait_for_call().await;
    AccountStore::load(&f.path)
        .unwrap()
        .mutate(|s| {
            s.accounts[0].clear_holds();
            true
        })
        .unwrap();
    assert!(matches!(
        f.state
            .engine
            .recheck_verification("held@example.com", true)
            .await,
        Err(CheckError::Busy)
    ));
    let result = task.await.unwrap().unwrap();
    assert_eq!(result.state, "unchecked");
    assert_eq!(result.last_result.as_deref(), Some("operator_reset"));
    assert!(result.recovered_at.is_none());
    assert!(!f.held().verification_required);
    assert!(f.held().verification_check.id.is_none());
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_new_verification_demand_keeps_the_lease_and_supersedes_old_checks() {
    let f = Fixture::new(4).await;
    let state = f.state.clone();
    let task = tokio::spawn(async move {
        state
            .engine
            .recheck_verification("held@example.com", true)
            .await
    });
    f.wait_for_call().await;
    AccountStore::load(&f.path)
        .unwrap()
        .mutate(|s| {
            s.accounts[0].mark_verification_required(
                Some("https://accounts.google.com/newer-challenge".into()),
                "newer",
            );
            true
        })
        .unwrap();
    assert!(matches!(
        f.state
            .engine
            .recheck_verification("held@example.com", true)
            .await,
        Err(CheckError::Busy)
    ));
    let result = task.await.unwrap().unwrap();
    assert_eq!(result.state, "waiting");
    assert_eq!(result.last_result.as_deref(), Some("verification_required"));
    assert!(
        f.held()
            .verification_url
            .unwrap()
            .contains("newer-challenge")
    );
    assert!(f.held().verification_check.id.is_none());
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn manual_check_respects_quota_bans_and_expired_leases_are_recoverable() {
    let f = Fixture::new(0).await;
    f.state
        .engine
        .accounts
        .mutate(|s| {
            s.accounts[0].mark_rate_limited("gemini:gemini-3.8-flash", now_ms() + 600_000);
            true
        })
        .unwrap();
    assert!(matches!(
        f.state
            .engine
            .recheck_verification("held@example.com", true)
            .await,
        Err(CheckError::Blocked)
    ));
    assert_eq!(f.calls.load(Ordering::SeqCst), 0);
    f.state
        .engine
        .accounts
        .mutate(|s| {
            s.accounts[0].rate_limit_reset_times.clear();
            s.accounts[0].enabled = false;
            true
        })
        .unwrap();
    assert!(matches!(
        f.state
            .engine
            .recheck_verification("held@example.com", true)
            .await,
        Err(CheckError::Blocked)
    ));
    f.state
        .engine
        .accounts
        .mutate(|s| {
            let a = &mut s.accounts[0];
            a.enabled = true;
            a.verification_check.id = Some("abandoned".into());
            a.verification_check.started_at = Some(now_ms() - 120_000);
            a.verification_check.lease_until = Some(now_ms() - 1);
            // An unrelated Pro quota must survive a successful Flash check.
            a.mark_rate_limited("gemini:gemini-3.1-pro", now_ms() + 600_000);
            true
        })
        .unwrap();
    let result = f
        .state
        .engine
        .recheck_verification("held@example.com", true)
        .await
        .unwrap();
    assert_eq!(result.state, "recovered");
    assert!(
        f.held()
            .rate_limit_reset_times
            .contains_key("gemini:gemini-3.1-pro")
    );
}

#[tokio::test]
async fn upstream_retry_after_is_preserved_and_no_immediate_reprobe_occurs() {
    let f = Fixture::new(5).await;
    let result = f
        .state
        .engine
        .recheck_verification("held@example.com", true)
        .await
        .unwrap();
    assert_eq!(result.state, "waiting");
    assert!(result.next_check_at.unwrap() - result.last_checked_at.unwrap() >= 600_000);
    assert!(matches!(
        f.state
            .engine
            .recheck_verification("held@example.com", true)
            .await,
        Err(CheckError::Busy | CheckError::Blocked)
    ));
    assert_eq!(f.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn immediate_http_endpoint_uses_auth_and_health_exposes_progress() {
    let f = Fixture::new(4).await;
    // Build another state with the same store and configured gateway auth.
    let mut config = f.state.engine.config.clone();
    config.server.api_keys = vec!["test-admin-key".into()];
    let state = crate::server::build_state(config, AccountStore::load(&f.path).unwrap()).unwrap();
    state.engine.credentials.seed_for_test(&f.held());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = crate::server::app_with_state(state);
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let client = reqwest::Client::new();
    let endpoint = format!("{url}/api/accounts/held@example.com/recheck");
    assert_eq!(
        client.post(&endpoint).send().await.unwrap().status(),
        StatusCode::UNAUTHORIZED
    );
    let request = client.post(&endpoint).bearer_auth("test-admin-key").send();
    let task = tokio::spawn(request);
    f.wait_for_call().await;
    let health: serde_json::Value = client
        .get(format!("{url}/health"))
        .bearer_auth("test-admin-key")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["accounts"][0]["status"], "checking");
    assert_eq!(health["accounts"][0]["verification"]["state"], "checking");
    assert!(health["accounts"][0].get("refresh_token").is_none());
    assert_eq!(
        client
            .post(&endpoint)
            .bearer_auth("test-admin-key")
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::CONFLICT
    );
    let response = task.await.unwrap().unwrap();
    let result: serde_json::Value = response.json().await.unwrap();
    assert_eq!(result["verification"]["last_result"], "timeout");
    server.abort();
}
