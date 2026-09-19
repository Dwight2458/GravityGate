//! OAuth 2.0 token operations against Google's identity endpoints.
//!
//! Two flows matter: refreshing an access token from a stored refresh token
//! (used on every request cycle), and the authorization-code exchange used once
//! per account when the user completes the browser flow.
//!
//! Failure classification is the point of this module. `invalid_grant` is
//! terminal — the refresh token has been revoked or the account has been
//! disabled — and must disable the account rather than trigger a retry loop.
//! Everything else is transient.

use std::time::{Duration, SystemTime};

use serde::Deserialize;

use crate::upstream::constants;
use crate::upstream::transport::{TransportError, UpstreamClient};

/// User-Agent presented on OAuth calls. The CLI reaches Google through the
/// Node API client, so a server-side flow that presents as a raw Rust binary is
/// an unnecessary difference.
const OAUTH_USER_AGENT: &str = "google-api-nodejs-client/9.15.1";

/// Fallback lifetime when the token endpoint omits or mangles `expires_in`.
pub const DEFAULT_EXPIRES_IN_SECS: i64 = 3600;

/// Tokens returned by the token endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    #[serde(default)]
    pub expires_in: Option<i64>,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub token_type: Option<String>,
}

impl TokenResponse {
    /// Absolute expiry, tolerating a missing or nonsensical `expires_in`.
    ///
    /// A non-positive value is treated as absent rather than trusted: a zero
    /// lifetime would make every request look expired and spin the refresh path.
    pub fn expires_at(&self, now: SystemTime) -> SystemTime {
        let lifetime = match self.expires_in {
            Some(secs) if secs > 0 => secs,
            _ => DEFAULT_EXPIRES_IN_SECS,
        };
        now + Duration::from_secs(lifetime as u64)
    }
}

/// Identity pulled from the userinfo endpoint, used to label an account.
#[derive(Debug, Clone, Deserialize)]
pub struct UserInfo {
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
}

/// Google's OAuth error envelope.
#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    #[serde(default)]
    error: String,
    #[serde(default)]
    error_description: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum OAuthError {
    #[error("oauth request failed: {0}")]
    Transport(#[from] TransportError),

    /// The token endpoint rejected the request. `code` is Google's machine
    /// readable value such as `invalid_grant` or `invalid_client`.
    #[error("oauth endpoint rejected the request ({code}): {message}")]
    Rejected {
        status: reqwest::StatusCode,
        code: String,
        message: String,
    },

    #[error("network error during oauth: {0}")]
    Network(String),
}

impl OAuthError {
    /// Whether the refresh token is permanently unusable.
    ///
    /// These mean the account has to be re-authorised or removed; retrying will
    /// never succeed and repeatedly hammering the endpoint risks flagging the
    /// client.
    pub fn is_terminal(&self) -> bool {
        match self {
            OAuthError::Rejected { code, .. } => matches!(
                code.as_str(),
                "invalid_grant" | "invalid_client" | "unauthorized_client"
            ),
            _ => false,
        }
    }

    /// Whether this was a transport-level problem rather than a credential
    /// problem, and therefore safe to retry later without touching the account.
    pub fn is_transient(&self) -> bool {
        matches!(self, OAuthError::Network(_) | OAuthError::Transport(_))
    }
}

pub struct OAuthClient {
    client: UpstreamClient,
}

impl OAuthClient {
    pub fn new() -> Result<Self, OAuthError> {
        Ok(Self {
            client: UpstreamClient::new()?,
        })
    }

    /// Exchange a stored refresh token for a fresh access token.
    pub async fn refresh(&self, refresh_token: &str) -> Result<TokenResponse, OAuthError> {
        let form = [
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", constants::OAUTH_CLIENT_ID),
            ("client_secret", constants::OAUTH_CLIENT_SECRET),
        ];
        self.post_form(constants::OAUTH_TOKEN_URL, &form).await
    }

    /// Exchange an authorization code for tokens. Used once per account, at the
    /// end of the browser flow.
    pub async fn exchange_code(
        &self,
        code: &str,
        code_verifier: &str,
        redirect_uri: &str,
    ) -> Result<TokenResponse, OAuthError> {
        let form = [
            ("grant_type", "authorization_code"),
            ("code", code),
            ("client_id", constants::OAUTH_CLIENT_ID),
            ("client_secret", constants::OAUTH_CLIENT_SECRET),
            ("redirect_uri", redirect_uri),
            ("code_verifier", code_verifier),
        ];
        self.post_form(constants::OAUTH_TOKEN_URL, &form).await
    }

    /// Fetch the account's email address, used as its display label.
    pub async fn userinfo(&self, access_token: &str) -> Result<UserInfo, OAuthError> {
        let url = format!("{}?alt=json", constants::OAUTH_USERINFO_URL);
        let response = self
            .client
            .post_json_buffered_as_get(&url, access_token, "application/json")
            .await
            .map_err(|e| match e {
                TransportError::Request(source) if source.is_connect() => {
                    OAuthError::Network(source.to_string())
                }
                other => OAuthError::Transport(other),
            })?;

        if !response.is_success() {
            return Err(self.classify(&response));
        }
        serde_json::from_slice(&response.body).map_err(|source| {
            OAuthError::Transport(TransportError::Malformed {
                status: response.status,
                body: response.body_text(),
                source,
            })
        })
    }

    async fn post_form(
        &self,
        url: &str,
        form: &[(&str, &str)],
    ) -> Result<TokenResponse, OAuthError> {
        let body = encode_form(form);
        let response = self
            .client
            .post_form(url, OAUTH_USER_AGENT, body.as_bytes())
            .await
            .map_err(|e| match e {
                TransportError::Request(source) if source.is_connect() => {
                    OAuthError::Network(source.to_string())
                }
                other => OAuthError::Transport(other),
            })?;

        if !response.is_success() {
            return Err(self.classify(&response));
        }

        serde_json::from_slice(&response.body).map_err(|source| {
            OAuthError::Transport(TransportError::Malformed {
                status: response.status,
                body: response.body_text(),
                source,
            })
        })
    }

    /// Turn a non-2xx response into a classified error, preserving Google's code
    /// when the body parses and falling back to the raw text when it does not.
    fn classify(&self, response: &crate::upstream::transport::BufferedResponse) -> OAuthError {
        match serde_json::from_slice::<ErrorEnvelope>(&response.body) {
            Ok(envelope) if !envelope.error.is_empty() => OAuthError::Rejected {
                status: response.status,
                code: envelope.error,
                message: envelope
                    .error_description
                    .unwrap_or_else(|| "no description provided".into()),
            },
            _ => OAuthError::Rejected {
                status: response.status,
                code: format!("http_{}", response.status.as_u16()),
                message: response.body_text(),
            },
        }
    }
}

/// Percent-encode an `application/x-www-form-urlencoded` body.
///
/// Implemented locally rather than pulling in a query-string crate: the input set
/// is small, known, and fixed.
fn encode_form(form: &[(&str, &str)]) -> String {
    fn encode(value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        for byte in value.as_bytes() {
            match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    out.push(*byte as char)
                }
                b' ' => out.push('+'),
                other => out.push_str(&format!("%{other:02X}")),
            }
        }
        out
    }

    form.iter()
        .map(|(k, v)| format!("{}={}", encode(k), encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expires_at_uses_reported_lifetime() {
        let now = SystemTime::UNIX_EPOCH;
        let token = TokenResponse {
            access_token: "t".into(),
            expires_in: Some(1200),
            refresh_token: None,
            scope: None,
            token_type: None,
        };
        assert_eq!(
            token.expires_at(now),
            SystemTime::UNIX_EPOCH + Duration::from_secs(1200)
        );
    }

    #[test]
    fn expires_at_falls_back_when_lifetime_is_absent() {
        let token = TokenResponse {
            access_token: "t".into(),
            expires_in: None,
            refresh_token: None,
            scope: None,
            token_type: None,
        };
        assert_eq!(
            token.expires_at(SystemTime::UNIX_EPOCH),
            SystemTime::UNIX_EPOCH + Duration::from_secs(DEFAULT_EXPIRES_IN_SECS as u64)
        );
    }

    #[test]
    fn expires_at_rejects_nonpositive_lifetimes() {
        // A zero or negative lifetime would make every request look expired.
        for bad in [Some(0), Some(-1)] {
            let token = TokenResponse {
                access_token: "t".into(),
                expires_in: bad,
                refresh_token: None,
                scope: None,
                token_type: None,
            };
            assert_eq!(
                token.expires_at(SystemTime::UNIX_EPOCH),
                SystemTime::UNIX_EPOCH + Duration::from_secs(DEFAULT_EXPIRES_IN_SECS as u64),
                "expires_in = {bad:?}"
            );
        }
    }

    #[test]
    fn invalid_grant_is_terminal() {
        let error = OAuthError::Rejected {
            status: reqwest::StatusCode::BAD_REQUEST,
            code: "invalid_grant".into(),
            message: "Token has been expired or revoked.".into(),
        };
        assert!(error.is_terminal());
        assert!(!error.is_transient());
    }

    #[test]
    fn server_error_is_not_terminal() {
        let error = OAuthError::Rejected {
            status: reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal_failure".into(),
            message: "backend error".into(),
        };
        assert!(!error.is_terminal());
    }

    #[test]
    fn network_errors_are_transient_not_terminal() {
        let error = OAuthError::Network("connection reset".into());
        assert!(error.is_transient());
        assert!(!error.is_terminal());
    }

    #[test]
    fn form_encoding_escapes_reserved_characters() {
        let encoded = encode_form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", "1//0g ABC+def/ghi="),
            ("client_id", "x.apps.googleusercontent.com"),
        ]);
        assert!(encoded.starts_with("grant_type=refresh_token&"));
        assert!(encoded.contains("refresh_token=1%2F%2F0g+ABC%2Bdef%2Fghi%3D"));
        // Dots and hyphens must survive unescaped.
        assert!(encoded.contains("x.apps.googleusercontent.com"));
    }

    #[test]
    fn form_encoding_handles_every_hex_digit() {
        // Guards the uppercase %XX path against a lowercase-formatting slip.
        let encoded = encode_form(&[("k", "\u{1}")]);
        assert_eq!(encoded, "k=%01");
    }

    #[test]
    fn malformed_error_body_still_produces_an_error() {
        let client = OAuthClient::new().unwrap();
        let response = crate::upstream::transport::BufferedResponse {
            status: reqwest::StatusCode::TOO_MANY_REQUESTS,
            headers: reqwest::header::HeaderMap::new(),
            body: bytes::Bytes::from_static(b"<html>rate limited</html>"),
        };
        let error = client.classify(&response);
        match error {
            OAuthError::Rejected { code, message, .. } => {
                assert_eq!(code, "http_429");
                assert!(message.contains("rate limited"));
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[test]
    fn error_envelope_is_parsed_into_a_code() {
        let client = OAuthClient::new().unwrap();
        let response = crate::upstream::transport::BufferedResponse {
            status: reqwest::StatusCode::BAD_REQUEST,
            headers: reqwest::header::HeaderMap::new(),
            body: bytes::Bytes::from_static(
                br#"{"error":"invalid_grant","error_description":"Token has been expired or revoked."}"#,
            ),
        };
        match client.classify(&response) {
            OAuthError::Rejected { code, message, .. } => {
                assert_eq!(code, "invalid_grant");
                assert!(message.contains("revoked"));
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }
}
