//! Upstream HTTP transport.
//!
//! Everything that touches the network on the way to Cloud Code Assist goes
//! through [`UpstreamClient`]. The module is deliberately self-contained: the M0
//! spike proved `reqwest`/`hyper` can reproduce the captured wire format, but
//! that conclusion is only validated against a live upstream once credentials
//! exist. If validation fails, the fallback is a hand-rolled HTTP/1.1 writer over
//! `rustls` — and this module is the only place that changes.
//!
//! Header order follows the capture. `http::HeaderMap` iterates in insertion
//! order and hyper serialises in iteration order, so the sequence of `.header()`
//! calls below is the sequence on the wire. Do not reorder them.

use std::pin::Pin;
use std::time::Duration;

use bytes::Bytes;
use futures::Stream;
use reqwest::header::{
    ACCEPT, ACCEPT_ENCODING, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT,
};
use reqwest::{Client, StatusCode};

use crate::upstream::constants::agy_cli_user_agent;

/// Connect phase budget.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Total budget for a buffered (non-streaming) call.
const BUFFERED_TIMEOUT: Duration = Duration::from_secs(180);

/// Maximum gap between reads on a streaming body. The reference transport uses a
/// 180s idle watchdog; matching it means a stalled upstream is detected on the
/// same schedule rather than hanging a client connection indefinitely.
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(180);

/// Body stream type handed back for streaming responses.
pub type BodyStream = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

/// A buffered upstream response, kept whole so the caller can classify errors.
#[derive(Debug)]
pub struct BufferedResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl BufferedResponse {
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }

    /// Body as UTF-8, lossily — error bodies are diagnostics, not data.
    pub fn body_text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// Parse the body as JSON, surfacing the raw text on failure so callers can
    /// report what the upstream actually said.
    pub fn json(&self) -> Result<serde_json::Value, TransportError> {
        serde_json::from_slice(&self.body).map_err(|source| TransportError::Malformed {
            status: self.status,
            body: self.body_text(),
            source,
        })
    }

    /// Convert a non-2xx response into an error, preserving status and body.
    ///
    /// Lets callers write `response.into_success()?.json()?` and get a single
    /// error type that still carries everything the classifier needs.
    pub fn into_success(self) -> Result<Self, TransportError> {
        if self.is_success() {
            Ok(self)
        } else {
            Err(TransportError::HttpStatus {
                status: self.status,
                body: self.body_text(),
            })
        }
    }
}

/// A streaming upstream response. The head is available immediately; the body is
/// consumed lazily.
pub struct StreamingResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: BodyStream,
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("upstream request failed: {0}")]
    Request(#[from] reqwest::Error),

    /// The upstream answered with a non-2xx status. The body is kept verbatim
    /// because classification — rate limit vs ban vs bad request — reads it.
    #[error("upstream returned {status}")]
    HttpStatus {
        status: StatusCode,
        body: String,
    },

    #[error("upstream returned {status} with an unparseable body: {body}")]
    Malformed {
        status: StatusCode,
        body: String,
        #[source]
        source: serde_json::Error,
    },
}

/// Which transport style a call uses, which decides its framing and timeouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallKind {
    /// Buffered JSON body, `Content-Length` framing, whole-request deadline.
    Buffered,
    /// Streamed body, chunked framing, idle-only deadline.
    Streaming,
}

pub struct UpstreamClient {
    http: Client,
}

impl UpstreamClient {
    pub fn new() -> Result<Self, TransportError> {
        let http = Client::builder()
            // The captured CLI speaks HTTP/1.1. Not enabling h2 keeps ALPN from
            // silently upgrading us onto a different code path.
            .http1_only()
            .connect_timeout(CONNECT_TIMEOUT)
            // Applies to the gap between reads, so a long stream is fine but a
            // stalled one is not. This is the idle watchdog.
            .read_timeout(STREAM_IDLE_TIMEOUT)
            // No whole-request deadline here: streams must be allowed to run long.
            // Buffered calls apply their own budget per request.
            .build()?;
        Ok(Self { http })
    }

    /// Assemble the header block in captured order.
    ///
    /// `Accept-Encoding: gzip` is set explicitly. Left to itself reqwest would
    /// send `gzip,deflate,br`, which is a deviation from the capture; setting it
    /// also tells reqwest not to install its own value.
    fn headers(access_token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            USER_AGENT,
            HeaderValue::from_str(&agy_cli_user_agent())
                .expect("generated user agent is always a valid header value"),
        );
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {access_token}"))
                .unwrap_or_else(|_| HeaderValue::from_static("Bearer ")),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT_ENCODING, HeaderValue::from_static("gzip"));
        headers
    }

    /// Issue a POST carrying a JSON body.
    pub async fn post_json(
        &self,
        url: &str,
        access_token: &str,
        body: &[u8],
        kind: CallKind,
    ) -> Result<StreamingResponse, TransportError> {
        let mut request = self
            .http
            .post(url)
            .headers(Self::headers(access_token));

        request = match kind {
            CallKind::Buffered => {
                request.timeout(BUFFERED_TIMEOUT).body(body.to_vec())
            }
            CallKind::Streaming => {
                // A single-chunk stream body makes hyper use chunked framing,
                // matching the captured CLI request rather than a buffered
                // Content-Length body. The idle watchdog comes from the client.
                let chunk = Bytes::copy_from_slice(body);
                let stream = futures::stream::once(async move {
                    Ok::<Bytes, std::io::Error>(chunk)
                });
                request.body(reqwest::Body::wrap_stream(stream))
            }
        };

        let response = request.send().await?;
        let status = response.status();
        let headers = response.headers().clone();
        let body: BodyStream = Box::pin(response.bytes_stream());

        Ok(StreamingResponse {
            status,
            headers,
            body,
        })
    }

    /// Issue a POST and buffer the whole response.
    pub async fn post_json_buffered(
        &self,
        url: &str,
        access_token: &str,
        body: &[u8],
    ) -> Result<BufferedResponse, TransportError> {
        let response = self.post_json(url, access_token, body, CallKind::Buffered).await?;
        Self::drain(response).await
    }

    /// POST an `application/x-www-form-urlencoded` body.
    ///
    /// Used for OAuth token operations, which are form-encoded rather than JSON
    /// and present a different User-Agent from generation traffic.
    pub async fn post_form(
        &self,
        url: &str,
        user_agent: &str,
        body: &[u8],
    ) -> Result<BufferedResponse, TransportError> {
        let response = self
            .http
            .post(url)
            .header(
                USER_AGENT,
                HeaderValue::from_str(user_agent)
                    .unwrap_or_else(|_| HeaderValue::from_static("gravitygate")),
            )
            .header(
                CONTENT_TYPE,
                HeaderValue::from_static("application/x-www-form-urlencoded;charset=UTF-8"),
            )
            .header(ACCEPT_ENCODING, HeaderValue::from_static("gzip"))
            .timeout(BUFFERED_TIMEOUT)
            .body(body.to_vec())
            .send()
            .await?;

        Self::drain(Self::into_streaming(response)).await
    }

    /// GET a JSON document with bearer authentication.
    ///
    /// Used for the userinfo endpoint, which is a read rather than a generation
    /// call and therefore carries no CLI identity headers.
    pub async fn post_json_buffered_as_get(
        &self,
        url: &str,
        access_token: &str,
        accept: &str,
    ) -> Result<BufferedResponse, TransportError> {
        let response = self
            .http
            .get(url)
            .header(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {access_token}"))
                    .unwrap_or_else(|_| HeaderValue::from_static("Bearer ")),
            )
            .header(
                ACCEPT,
                HeaderValue::from_str(accept).unwrap_or_else(|_| HeaderValue::from_static("*/*")),
            )
            .header(ACCEPT_ENCODING, HeaderValue::from_static("gzip"))
            .timeout(BUFFERED_TIMEOUT)
            .send()
            .await?;

        Self::drain(Self::into_streaming(response)).await
    }

    /// Split a `reqwest` response into the head-plus-lazy-body shape used
    /// throughout this module.
    fn into_streaming(response: reqwest::Response) -> StreamingResponse {
        let status = response.status();
        let headers = response.headers().clone();
        StreamingResponse {
            status,
            headers,
            body: Box::pin(response.bytes_stream()),
        }
    }

    /// Drain a streaming response into a buffered one.
    async fn drain(response: StreamingResponse) -> Result<BufferedResponse, TransportError> {
        let StreamingResponse {
            status,
            headers,
            mut body,
        } = response;

        let mut collected = Vec::new();
        use futures::StreamExt;
        while let Some(chunk) = body.next().await {
            collected.extend_from_slice(&chunk?);
        }

        Ok(BufferedResponse {
            status,
            headers,
            body: Bytes::from(collected),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_block_matches_captured_order_and_content() {
        let headers = UpstreamClient::headers("TOKEN");
        let names: Vec<String> = headers
            .keys()
            .map(|k| k.as_str().to_ascii_lowercase())
            .collect();
        // Insertion order is iteration order is wire order.
        assert_eq!(
            names,
            vec!["user-agent", "authorization", "content-type", "accept-encoding"]
        );
        assert_eq!(headers[ACCEPT_ENCODING], "gzip");
        assert_eq!(headers[CONTENT_TYPE], "application/json");
        assert_eq!(headers[AUTHORIZATION], "Bearer TOKEN");
    }

    #[test]
    fn user_agent_header_is_the_cli_identity() {
        let headers = UpstreamClient::headers("t");
        let ua = headers[USER_AGENT].to_str().unwrap();
        assert!(ua.starts_with("antigravity/cli/"));
        assert!(ua.contains("aidev_client"));
    }

    #[test]
    fn no_legacy_goog_headers_are_present() {
        // The 2.0 reference deliberately dropped these from content requests;
        // `x-goog-user-project` in particular causes a 403.
        let headers = UpstreamClient::headers("t");
        for name in ["x-goog-api-client", "client-metadata", "x-goog-user-project"] {
            assert!(
                !headers.contains_key(name),
                "{name} must not be sent on content requests"
            );
        }
    }

    #[test]
    fn buffered_response_reports_body_text_on_parse_failure() {
        let response = BufferedResponse {
            status: StatusCode::FORBIDDEN,
            headers: HeaderMap::new(),
            body: Bytes::from_static(b"not json at all"),
        };
        let error = response.json().unwrap_err();
        let message = error.to_string();
        assert!(message.contains("not json at all"), "got: {message}");
    }

    #[test]
    fn buffered_response_parses_valid_json() {
        let response = BufferedResponse {
            status: StatusCode::OK,
            headers: HeaderMap::new(),
            body: Bytes::from_static(br#"{"ok":true}"#),
        };
        assert_eq!(response.json().unwrap()["ok"], true);
    }

    #[test]
    fn client_builds() {
        assert!(UpstreamClient::new().is_ok());
    }
}
