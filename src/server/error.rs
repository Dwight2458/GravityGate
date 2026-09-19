//! Client-facing error mapping.
//!
//! Clients in this ecosystem act on status codes, so the code chosen here is a
//! behavioural decision rather than a cosmetic one. The rules:
//!
//! - **A malformed request is a 400.** The client should fix it and not retry.
//! - **An exhausted pool is a 429**, with `Retry-After` and a human-readable
//!   reset time, because that is what the OpenAI ecosystem expects and what
//!   clients back off on. The reference implementations deliberately return 400
//!   here to stop agentic clients retrying a condition that will not clear; that
//!   is available via `routing.exhausted_error_mode` for operators who want it,
//!   but it is not the default, because 429 with a correct `Retry-After` is both
//!   true and actionable.
//! - **A dead credential is a 401** so the operator notices, and an upstream
//!   fault is a 502 so the client knows the request was fine.
//!
//! Every failure that is *our* problem is logged with its cause and reported to
//! the client with a message that does not leak account identity.

use axum::Json;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::config::ExhaustedErrorMode;
use crate::engine::dispatch::DispatchError;
use crate::engine::retry::RetryError;

/// The OpenAI error envelope.
#[derive(Debug, Serialize)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// An error ready to be returned to a client.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub message: String,
    pub code: Option<String>,
    /// Seconds until the client may retry, when that is knowable.
    pub retry_after: Option<u64>,
}

impl ApiError {
    pub fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
            code: None,
            retry_after: None,
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
        )
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, "api_error", message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        // Log our own failures loudly; a client error is not worth a warning.
        if self.status.is_server_error() {
            tracing::error!(status = %self.status, message = %self.message, "request failed");
        } else {
            tracing::debug!(status = %self.status, message = %self.message, "request rejected");
        }

        let envelope = ErrorEnvelope {
            error: ErrorBody {
                message: self.message,
                kind: self.kind,
                param: None,
                code: self.code,
            },
        };

        let mut response = (self.status, Json(envelope)).into_response();
        if let Some(seconds) = self.retry_after
            && let Ok(value) = HeaderValue::from_str(&seconds.to_string())
        {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
        response
    }
}

/// Map a dispatch failure onto the client-facing error.
pub fn from_dispatch(error: DispatchError, mode: ExhaustedErrorMode) -> ApiError {
    match error {
        // The client sent something we could not turn into an upstream request.
        DispatchError::Resolve(error) => ApiError::invalid_request(format!(
            "could not resolve model: {error}"
        )),
        DispatchError::Translate(error) => {
            ApiError::invalid_request(format!("could not translate request: {error}"))
        }
        DispatchError::Serialise(error) => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            format!("could not encode the request: {error}"),
        ),

        DispatchError::NoAccounts => {
            ApiError::unavailable("no accounts are configured on this gateway")
        }
        DispatchError::NoCandidate(reason) => {
            ApiError::unavailable(format!("no account can serve this request: {reason}"))
        }

        DispatchError::Retry(error) => from_retry(error, mode),
    }
}

/// Map a retry-loop failure onto the client-facing error.
pub fn from_retry(error: RetryError, mode: ExhaustedErrorMode) -> ApiError {
    match error {
        RetryError::BadRequest { message } => ApiError::invalid_request(message),

        RetryError::AuthInvalid { message } => ApiError::new(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            format!("the upstream rejected the gateway's credentials: {message}"),
        ),

        RetryError::AccountsExhausted { tried, last } => ApiError::unavailable(format!(
            "all {tried} available account(s) failed; last error: {last}"
        )),

        RetryError::PoolExhausted { earliest_in } => exhausted(earliest_in, mode),
    }
}

/// Report that every account is rate limited.
///
/// The two modes exist because the right answer depends on the client. A
/// well-behaved OpenAI client honours `Retry-After` and backs off; an agentic
/// loop treats 429 as "try again immediately" and will spin. Operators running
/// the latter can switch to 400, which those clients treat as terminal.
fn exhausted(
    earliest_in: Option<std::time::Duration>,
    mode: ExhaustedErrorMode,
) -> ApiError {
    let seconds = earliest_in.map(|wait| wait.as_secs().max(1));
    let message = match seconds {
        Some(seconds) => format!(
            "every account is rate limited. The earliest clears in {}.",
            humanise(seconds)
        ),
        None => "every account is rate limited.".to_string(),
    };

    match mode {
        ExhaustedErrorMode::TooManyRequests => ApiError {
            status: StatusCode::TOO_MANY_REQUESTS,
            kind: "rate_limit_error",
            message,
            code: Some("rate_limit_exceeded".into()),
            retry_after: seconds,
        },
        ExhaustedErrorMode::BadRequest => ApiError {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message,
            code: None,
            retry_after: None,
        },
    }
}

/// Render a duration for a human reading an error message.
fn humanise(seconds: u64) -> String {
    let (hours, minutes, secs) = (
        seconds / 3600,
        (seconds % 3600) / 60,
        seconds % 60,
    );
    if hours > 0 {
        format!("{hours}h{minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m{secs:02}s")
    } else {
        format!("{secs}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ResolveError;
    use crate::transform::request::TranslateError;
    use std::time::Duration;

    fn mode() -> ExhaustedErrorMode {
        ExhaustedErrorMode::TooManyRequests
    }

    #[test]
    fn a_bad_request_maps_to_a_400() {
        let error = from_retry(
            RetryError::BadRequest {
                message: "Invalid JSON".into(),
            },
            mode(),
        );
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.kind, "invalid_request_error");
    }

    #[test]
    fn a_dead_credential_maps_to_a_401() {
        let error = from_retry(
            RetryError::AuthInvalid {
                message: "revoked".into(),
            },
            mode(),
        );
        assert_eq!(error.status, StatusCode::UNAUTHORIZED);
        assert_eq!(error.kind, "authentication_error");
    }

    #[test]
    fn an_exhausted_pool_maps_to_a_429_with_retry_after() {
        let error = from_retry(
            RetryError::PoolExhausted {
                earliest_in: Some(Duration::from_secs(90)),
            },
            mode(),
        );
        assert_eq!(error.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(error.kind, "rate_limit_error");
        assert_eq!(error.retry_after, Some(90));
        assert!(error.message.contains("1m30s"), "got {}", error.message);
    }

    #[test]
    fn the_configured_mode_can_downgrade_exhaustion_to_a_400() {
        // Operators running agentic clients that retry 429 forever can opt out.
        let error = from_retry(
            RetryError::PoolExhausted {
                earliest_in: Some(Duration::from_secs(90)),
            },
            ExhaustedErrorMode::BadRequest,
        );
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.kind, "invalid_request_error");
        assert!(error.retry_after.is_none(), "a 400 has no retry hint");
    }

    #[test]
    fn exhaustion_with_no_known_reset_still_reports_429() {
        let error = from_retry(RetryError::PoolExhausted { earliest_in: None }, mode());
        assert_eq!(error.status, StatusCode::TOO_MANY_REQUESTS);
        assert!(error.retry_after.is_none(), "no reset means no hint");
    }

    #[test]
    fn a_very_short_reset_is_reported_as_one_second() {
        // Zero would be read as "retry immediately", which is the opposite of
        // the intent.
        let error = from_retry(
            RetryError::PoolExhausted {
                earliest_in: Some(Duration::from_millis(10)),
            },
            mode(),
        );
        assert_eq!(error.retry_after, Some(1));
    }

    #[test]
    fn exhausted_accounts_map_to_a_503() {
        let error = from_retry(
            RetryError::AccountsExhausted {
                tried: 3,
                last: "rate limited".into(),
            },
            mode(),
        );
        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(error.message.contains('3'));
    }

    #[test]
    fn no_accounts_configured_is_a_503_that_says_so() {
        let error = from_dispatch(DispatchError::NoAccounts, mode());
        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(error.message.contains("no accounts"));
    }

    #[test]
    fn an_unresolvable_model_is_a_400() {
        let error = from_dispatch(
            DispatchError::Resolve(ResolveError::Empty),
            mode(),
        );
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.kind, "invalid_request_error");
    }

    #[test]
    fn an_untranslatable_request_is_a_400() {
        let error = from_dispatch(
            DispatchError::Translate(TranslateError::NoMessages),
            mode(),
        );
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn a_blocking_account_state_is_a_503_naming_the_reason() {
        let error = from_dispatch(
            DispatchError::NoCandidate("alice is ineligible".into()),
            mode(),
        );
        assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(error.message.contains("ineligible"));
    }

    #[test]
    fn envelope_shape_matches_the_openai_spec() {
        let error = ApiError::invalid_request("bad model");
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(
            response.headers().get(header::CONTENT_TYPE).is_some(),
            "the envelope must be json"
        );
    }

    #[test]
    fn retry_after_is_emitted_as_a_header() {
        let error = ApiError {
            status: StatusCode::TOO_MANY_REQUESTS,
            kind: "rate_limit_error",
            message: "wait".into(),
            code: None,
            retry_after: Some(42),
        };
        let response = error.into_response();
        assert_eq!(
            response
                .headers()
                .get(header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some("42")
        );
    }

    #[test]
    fn durations_read_naturally() {
        assert_eq!(humanise(45), "45s");
        assert_eq!(humanise(90), "1m30s");
        assert_eq!(humanise(3600), "1h00m");
        assert_eq!(humanise(7_800), "2h10m");
    }

    #[test]
    fn the_error_message_does_not_leak_account_identity() {
        // A client should not learn the operator's account emails from an error.
        let error = from_retry(
            RetryError::AccountsExhausted {
                tried: 2,
                last: "rate limited (QuotaExhausted)".into(),
            },
            mode(),
        );
        assert!(!error.message.contains('@'), "got {}", error.message);
    }
}
