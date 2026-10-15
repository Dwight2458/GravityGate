//! Shared JSON extraction for generation requests. Reject oversized bodies
//! before dispatch, while retaining the OpenAI error shape and ingress evidence.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

use axum::Json;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, FromRequest, Request};
use axum::http::{StatusCode, header};
use futures::TryStreamExt;
use serde::de::DeserializeOwned;

use super::{SharedState, error::ApiError, execute};

pub struct ApiJson<T>(pub T);

impl<T: DeserializeOwned> FromRequest<SharedState> for ApiJson<T> {
    type Rejection = ApiError;

    async fn from_request(mut request: Request, state: &SharedState) -> Result<Self, ApiError> {
        let started = Instant::now();
        let limit = state.engine.config.server.max_request_body_bytes;
        let route = request.uri().path().to_string();
        let declared_bytes = request
            .headers()
            .get(header::CONTENT_LENGTH)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        DefaultBodyLimit::max(limit).apply(&mut request);

        // Count bytes actually read even when Content-Length is absent. On a
        // rejection this is an observed prefix, not necessarily the full body.
        let observed = Arc::new(AtomicU64::new(0));
        let counter = observed.clone();
        let (parts, body) = request.into_parts();
        let body = body.into_data_stream().inspect_ok(move |bytes| {
            counter.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        });
        let request = Request::from_parts(parts, Body::from_stream(body));
        match Json::<T>::from_request(request, state).await {
            Ok(Json(value)) => {
                tracing::debug!(%route, request_body_bytes = observed.load(Ordering::Relaxed),
                    limit_bytes = limit, "generation request body read");
                Ok(Self(value))
            }
            Err(rejection) => {
                let status = rejection.status();
                let mut error = if status == StatusCode::PAYLOAD_TOO_LARGE {
                    ApiError::new(
                        status,
                        "invalid_request_error",
                        format!(
                            "Request body exceeds the configured limit of {limit} bytes. Reduce conversation history or image size, or increase server.max_request_body_bytes."
                        ),
                    )
                } else {
                    ApiError::new(status, "invalid_request_error", rejection.body_text())
                };
                error.code = Some(
                    if status == StatusCode::PAYLOAD_TOO_LARGE {
                        "request_body_too_large"
                    } else {
                        "invalid_request_body"
                    }
                    .into(),
                );
                tracing::warn!(%route, status = status.as_u16(), limit_bytes = limit,
                    ?declared_bytes, observed_body_bytes = observed.load(Ordering::Relaxed),
                    "generation request body rejected before dispatch");
                // JSON was not parsed, so model/stream preference are unknown.
                // Keep the existing audit schema's empty/default identifiers.
                execute::record(
                    state,
                    "",
                    "",
                    "",
                    false,
                    status.as_u16(),
                    0,
                    None,
                    started.elapsed(),
                );
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        accounts::store::AccountStore,
        config::{Config, DEFAULT_MAX_REQUEST_BODY_BYTES},
    };
    use serde_json::Value;

    const ROUTES: [&str; 2] = ["/v1/chat/completions", "/v1/responses"];

    struct TestServer {
        url: String,
        task: tokio::task::JoinHandle<()>,
    }

    impl TestServer {
        async fn new(limit: usize) -> Self {
            let mut config = Config::default();
            config.server.max_request_body_bytes = limit;
            config.audit.enabled = false;
            config.metrics.enabled = false;
            // An empty private store guarantees no request can reach Google.
            let store = AccountStore::load(
                std::env::temp_dir().join(format!("gg-body-{}.json", uuid::Uuid::new_v4())),
            )
            .unwrap();
            let app = crate::server::app(config, store).unwrap();
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let task = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            Self { url, task }
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    fn body_of_size(route: &str, bytes: usize) -> String {
        let (prefix, suffix) = if route == ROUTES[0] {
            (
                r#"{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":""#,
                r#""}]}"#,
            )
        } else {
            (r#"{"model":"gemini-3.8-flash-high","input":""#, r#""}"#)
        };
        format!(
            "{prefix}{}{suffix}",
            "x".repeat(bytes - prefix.len() - suffix.len())
        )
    }

    #[tokio::test]
    async fn zcode_sized_payload_passes_both_generation_extractors() {
        let server = TestServer::new(DEFAULT_MAX_REQUEST_BODY_BYTES).await;
        let client = reqwest::Client::new();
        for route in ROUTES {
            let body = body_of_size(route, 2_098_816);
            let response = client
                .post(format!("{}{route}", server.url))
                .header(header::CONTENT_TYPE, "application/json")
                .body(body)
                .send()
                .await
                .unwrap();
            // Extraction and translation succeeded; dispatch stops at the
            // empty test account pool instead of contacting an upstream.
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            let error: Value = response.json().await.unwrap();
            assert!(
                error["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("no accounts are configured")
            );
        }
    }

    #[tokio::test]
    async fn exact_configured_limit_is_accepted_and_one_extra_byte_returns_json_413() {
        let limit = 1024;
        let server = TestServer::new(limit).await;
        let client = reqwest::Client::new();
        for route in ROUTES {
            for bytes in [limit, limit + 1] {
                let body = body_of_size(route, bytes);
                assert_eq!(body.len(), bytes);
                let response = client
                    .post(format!("{}{route}", server.url))
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(body)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    if bytes == limit {
                        StatusCode::SERVICE_UNAVAILABLE
                    } else {
                        StatusCode::PAYLOAD_TOO_LARGE
                    }
                );
                assert!(
                    response.headers()[header::CONTENT_TYPE]
                        .to_str()
                        .unwrap()
                        .starts_with("application/json")
                );
                assert!(response.headers().get(header::RETRY_AFTER).is_none());
                let error: Value = response.json().await.unwrap();
                if bytes > limit {
                    assert_eq!(error["error"]["type"], "invalid_request_error");
                    assert_eq!(error["error"]["code"], "request_body_too_large");
                    assert!(
                        error["error"]["message"]
                            .as_str()
                            .unwrap()
                            .contains("1024 bytes")
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn chunked_requests_cannot_bypass_the_limit() {
        let server = TestServer::new(1024).await;
        let client = reqwest::Client::new();
        for route in ROUTES {
            let body = body_of_size(route, 1025).into_bytes();
            let chunks: Vec<Result<Vec<u8>, std::io::Error>> =
                body.chunks(200).map(|b| Ok(b.to_vec())).collect();
            let request = client
                .post(format!("{}{route}", server.url))
                .header(header::CONTENT_TYPE, "application/json")
                .body(reqwest::Body::wrap_stream(futures::stream::iter(chunks)))
                .build()
                .unwrap();
            assert!(request.headers().get(header::CONTENT_LENGTH).is_none());
            let response = client.execute(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
            let error: Value = response.json().await.unwrap();
            assert_eq!(error["error"]["code"], "request_body_too_large");
        }
    }

    #[tokio::test]
    async fn malformed_json_schema_and_content_type_errors_keep_their_status_in_json() {
        let server = TestServer::new(1024).await;
        let client = reqwest::Client::new();
        for route in ROUTES {
            for (body, content_type, status) in [
                ("{", "application/json", StatusCode::BAD_REQUEST),
                ("{}", "application/json", StatusCode::UNPROCESSABLE_ENTITY),
                ("{}", "text/plain", StatusCode::UNSUPPORTED_MEDIA_TYPE),
            ] {
                let response = client
                    .post(format!("{}{route}", server.url))
                    .header(header::CONTENT_TYPE, content_type)
                    .body(body)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), status);
                let error: Value = response.json().await.unwrap();
                assert_eq!(error["error"]["type"], "invalid_request_error");
                assert_eq!(error["error"]["code"], "invalid_request_body");
            }
        }
    }

    #[tokio::test]
    async fn body_limit_counts_utf8_bytes_rather_than_characters() {
        let server = TestServer::new(1024).await;
        let client = reqwest::Client::new();
        for route in ROUTES {
            let body = body_of_size(route, 600).replace('x', "界");
            assert!(body.chars().count() < 1024 && body.len() > 1024);
            let response = client
                .post(format!("{}{route}", server.url))
                .header(header::CONTENT_TYPE, "application/json")
                .body(body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        }
    }

    #[tokio::test]
    async fn extraction_rejections_are_audited_without_an_upstream_attempt() {
        let path = std::env::temp_dir().join(format!("gg-ingress-{}.db", uuid::Uuid::new_v4()));
        let mut config = Config::default();
        config.server.max_request_body_bytes = 1024;
        config.metrics.enabled = false;
        config.audit.path = Some(path.to_string_lossy().into_owned());
        let store = AccountStore::load(path.with_extension("json")).unwrap();
        let state = crate::server::build_state(config, store).unwrap();
        let request = Request::builder()
            .method("POST")
            .uri(ROUTES[0])
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body_of_size(ROUTES[0], 1025)))
            .unwrap();
        let error = ApiJson::<Value>::from_request(request, &state)
            .await
            .err()
            .unwrap();
        assert_eq!(error.status, StatusCode::PAYLOAD_TOO_LARGE);
        let deadline = Instant::now() + std::time::Duration::from_secs(2);
        let records = loop {
            let records = state.observability.audit().unwrap().recent(10).unwrap();
            if !records.is_empty() {
                break records;
            }
            assert!(
                Instant::now() < deadline,
                "audit writer did not persist the rejection"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].status, 413);
        assert_eq!(records[0].attempts, 0);
        assert!(records[0].account_id.is_empty());
        drop(state);
        for file in [
            path.clone(),
            path.with_file_name(format!(
                "{}-wal",
                path.file_name().unwrap().to_string_lossy()
            )),
            path.with_file_name(format!(
                "{}-shm",
                path.file_name().unwrap().to_string_lossy()
            )),
        ] {
            let _ = std::fs::remove_file(file);
        }
    }
}
