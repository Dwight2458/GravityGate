//! The local redirect listener for the authorization-code flow.
//!
//! Google redirects the browser to `http://localhost:<port>/oauth-callback`,
//! which means the port has to be settled **before** the authorization URL is
//! built — the URI is part of the request, not a detail of the exchange. So the
//! listener binds first and the caller asks it for its redirect URI, rather than
//! the caller assuming 51121 and hoping.
//!
//! Two things make this more than a few lines of socket code:
//!
//! - **Both loopback families are bound.** On Windows `localhost` frequently
//!   resolves to `::1` first. A listener on `127.0.0.1` alone leaves the browser
//!   connecting to a closed port, which presents to the user as an infinite hang
//!   rather than an error.
//! - **Browsers make unrelated requests.** A stray `/favicon.ico` must not
//!   consume the one callback we are waiting for, so non-callback requests are
//!   answered and the wait continues.

use std::net::{Ipv4Addr, Ipv6Addr};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Path Google redirects to. Part of the registered redirect URI.
pub const CALLBACK_PATH: &str = "/oauth-callback";

/// Ports to try, primary first.
///
/// The fallbacks exist because 51121 is not always available: on Windows it can
/// sit inside a port range reserved by Hyper-V, WSL2, or Docker. Google's
/// loopback redirect handling permits a varying port, so falling back is viable
/// rather than merely optimistic.
pub const DEFAULT_PORTS: &[u16] = &[51121, 51122, 51123, 51124, 51125, 51126];

/// Cap on how much of a request head we will read before giving up on it.
const MAX_REQUEST_HEAD: usize = 16 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum CallbackError {
    #[error("could not bind any callback port; tried: {}", .failures.join(", "))]
    NoPort { failures: Vec<String> },

    #[error("timed out after {}s waiting for the browser redirect", .0.as_secs())]
    Timeout(Duration),

    #[error("the authorization server returned an error: {0}")]
    AuthorizationFailed(String),

    #[error("state mismatch: the callback did not match this login attempt")]
    StateMismatch,

    #[error("the callback carried no authorization code")]
    MissingCode,

    #[error("could not read the authorization code: {0}")]
    Input(String),
}

/// A bound listener, plus the port it actually got.
pub struct CallbackListener {
    listeners: Vec<TcpListener>,
    port: u16,
}

impl std::fmt::Debug for CallbackListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackListener")
            .field("port", &self.port)
            .field("sockets", &self.listeners.len())
            .finish()
    }
}

impl CallbackListener {
    /// Bind the first available port from `ports`.
    pub async fn bind(ports: &[u16]) -> Result<Self, CallbackError> {
        let mut failures = Vec::new();

        for port in ports {
            match Self::bind_port(*port).await {
                Ok(listeners) => return Ok(Self { listeners, port: *port }),
                Err(error) => failures.push(format!("{port} ({error})")),
            }
        }

        Err(CallbackError::NoPort { failures })
    }

    /// Bind both loopback families on one port.
    ///
    /// Succeeds if at least one family binds: a machine with IPv6 disabled still
    /// works, and so does one where `::1` is unavailable for another reason.
    async fn bind_port(port: u16) -> std::io::Result<Vec<TcpListener>> {
        let mut listeners = Vec::new();
        let mut last_error = None;

        for address in [
            std::net::SocketAddr::from((Ipv4Addr::LOCALHOST, port)),
            std::net::SocketAddr::from((Ipv6Addr::LOCALHOST, port)),
        ] {
            match TcpListener::bind(address).await {
                Ok(listener) => listeners.push(listener),
                Err(error) => last_error = Some(error),
            }
        }

        if listeners.is_empty() {
            return Err(last_error.unwrap_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::AddrNotAvailable, "no loopback address")
            }));
        }
        Ok(listeners)
    }

    /// The port that was actually bound.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// The redirect URI to send to the authorization server.
    ///
    /// Always spells the host `localhost` rather than `127.0.0.1`: that is the
    /// form the client is registered with.
    pub fn redirect_uri(&self) -> String {
        format!("http://localhost:{}{}", self.port, CALLBACK_PATH)
    }

    /// Wait for the browser to deliver an authorization code.
    ///
    /// `expected_state` is compared against the callback's `state` parameter; a
    /// mismatch aborts rather than being ignored, since it means the code came
    /// from a flow we did not start.
    pub async fn wait(
        self,
        expected_state: &str,
        timeout: Duration,
    ) -> Result<String, CallbackError> {
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<Result<String, CallbackError>>(4);
        let mut handles = Vec::new();

        for listener in self.listeners {
            let sender = sender.clone();
            let expected = expected_state.to_string();
            handles.push(tokio::spawn(async move {
                loop {
                    let Ok((mut stream, _)) = listener.accept().await else {
                        // Accept errors on a loopback listener are transient.
                        continue;
                    };
                    match serve(&mut stream, &expected).await {
                        // Unrelated request (favicon and friends): already
                        // answered, keep waiting for the real callback.
                        None => continue,
                        Some(outcome) => {
                            let _ = sender.send(outcome).await;
                            return;
                        }
                    }
                }
            }));
        }
        // Drop our own sender so the channel closes if every task exits.
        drop(sender);

        let outcome = tokio::time::timeout(timeout, receiver.recv()).await;

        for handle in handles {
            handle.abort();
        }

        match outcome {
            Ok(Some(result)) => result,
            // Every listener task ended without producing a code.
            Ok(None) => Err(CallbackError::MissingCode),
            Err(_) => Err(CallbackError::Timeout(timeout)),
        }
    }
}

/// Handle one connection.
///
/// Returns `None` when the request was not the callback, in which case a
/// response has still been written.
async fn serve(stream: &mut TcpStream, expected_state: &str) -> Option<Result<String, CallbackError>> {
    let head = match read_request_head(stream).await {
        Ok(head) => head,
        Err(_) => return None,
    };

    let target = match request_target(&head) {
        Some(target) => target,
        None => return None,
    };

    let params = match parse_callback_query(target) {
        Some(params) => params,
        None => {
            // Not our path. Answer politely so the browser does not show a
            // connection error for a favicon request.
            let _ = respond(stream, 404, &page("Not found", "This is not the OAuth callback.")).await;
            return None;
        }
    };

    if let Some(error) = params.error {
        let _ = respond(
            stream,
            400,
            &page("Authentication failed", &format!("Google returned: {error}")),
        )
        .await;
        return Some(Err(CallbackError::AuthorizationFailed(error)));
    }

    if let Some(state) = &params.state
        && state != expected_state
    {
        let _ = respond(
            stream,
            400,
            &page(
                "Authentication failed",
                "The callback did not match this login attempt.",
            ),
        )
        .await;
        return Some(Err(CallbackError::StateMismatch));
    }

    let Some(code) = params.code else {
        let _ = respond(
            stream,
            400,
            &page("Authentication failed", "No authorization code was returned."),
        )
        .await;
        return Some(Err(CallbackError::MissingCode));
    };

    let _ = respond(
        stream,
        200,
        &page(
            "Authentication successful",
            "You can close this window and return to the terminal.",
        ),
    )
    .await;
    Some(Ok(code))
}

/// Read a request head, stopping at the blank line that ends it.
async fn read_request_head(stream: &mut TcpStream) -> std::io::Result<String> {
    let mut buffer = Vec::with_capacity(2048);
    let mut chunk = [0u8; 1024];

    loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
        if buffer.len() >= MAX_REQUEST_HEAD {
            break;
        }
    }

    Ok(String::from_utf8_lossy(&buffer).into_owned())
}

/// Extract the request target from a request head, e.g. `/oauth-callback?code=x`.
fn request_target(head: &str) -> Option<&str> {
    let line = head.lines().next()?;
    let mut parts = line.split_whitespace();
    let _method = parts.next()?;
    parts.next()
}

async fn respond(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        _ => "Not Found",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await?;
    stream.flush().await
}

/// Minimal result page. Self-contained, no external assets.
fn page(title: &str, message: &str) -> String {
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>{title}</title></head>\
         <body style=\"font-family: system-ui, sans-serif; padding: 3rem; text-align: center;\">\
         <h1>{title}</h1><p>{message}</p></body></html>"
    )
}

/// Query parameters of a callback request.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CallbackParams {
    pub code: Option<String>,
    pub state: Option<String>,
    pub error: Option<String>,
}

/// Parse a request target, returning `None` when the path is not the callback.
pub fn parse_callback_query(target: &str) -> Option<CallbackParams> {
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path, query),
        None => (target, ""),
    };

    if path != CALLBACK_PATH {
        return None;
    }

    let mut params = CallbackParams::default();
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (key, value) = match pair.split_once('=') {
            Some((key, value)) => (key, value),
            None => (pair, ""),
        };
        let value = percent_decode(value);
        match key {
            "code" => params.code = Some(value),
            "state" => params.state = Some(value),
            "error" => params.error = Some(value),
            _ => {}
        }
    }

    Some(params)
}

/// Decode `%XX` escapes and `+` as space.
pub fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).ok();
                match hex.and_then(|hex| u8::from_str_radix(hex, 16).ok()) {
                    Some(byte) => {
                        out.push(byte);
                        index += 3;
                    }
                    // A malformed escape is passed through literally rather than
                    // dropping the byte, so the code stays recognizable.
                    None => {
                        out.push(bytes[index]);
                        index += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                index += 1;
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }

    String::from_utf8_lossy(&out).into_owned()
}

/// A code recovered from pasted input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedCode {
    pub code: String,
    pub state: Option<String>,
}

/// Pull an authorization code out of whatever the user pasted.
///
/// Accepts either the full redirect URL — which is what the address bar shows
/// when nothing is listening on the callback port — or the bare code.
pub fn extract_code_from_input(input: &str) -> Result<ExtractedCode, CallbackError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(CallbackError::Input("nothing was pasted".into()));
    }

    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        // Reuse the same parser that serves live callbacks, so a pasted URL and
        // a delivered one cannot diverge.
        let target = trimmed
            .split_once("://")
            .and_then(|(_, rest)| rest.split_once('/'))
            .map(|(_, path)| format!("/{path}"))
            .ok_or_else(|| CallbackError::Input("that URL has no path".into()))?;

        let params = parse_callback_query(&target)
            .ok_or_else(|| CallbackError::Input(format!("not an OAuth callback URL: {trimmed}")))?;

        if let Some(error) = params.error {
            return Err(CallbackError::AuthorizationFailed(error));
        }
        let code = params.code.ok_or(CallbackError::MissingCode)?;
        return Ok(ExtractedCode {
            code,
            state: params.state,
        });
    }

    // A bare code. Google's are long and start with `4/`, but the length check
    // alone catches the realistic mistake, which is pasting a truncated line.
    if trimmed.len() < 16 {
        return Err(CallbackError::Input(format!(
            "that is too short to be an authorization code ({} characters)",
            trimmed.len()
        )));
    }

    Ok(ExtractedCode {
        code: percent_decode(trimmed),
        state: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_path_params_are_parsed() {
        let params = parse_callback_query("/oauth-callback?code=abc123&state=deadbeef").unwrap();
        assert_eq!(params.code.as_deref(), Some("abc123"));
        assert_eq!(params.state.as_deref(), Some("deadbeef"));
        assert!(params.error.is_none());
    }

    #[test]
    fn a_different_path_is_not_the_callback() {
        assert!(parse_callback_query("/favicon.ico").is_none());
        assert!(parse_callback_query("/").is_none());
        assert!(parse_callback_query("/oauth-callback/extra").is_none());
    }

    #[test]
    fn query_order_does_not_matter() {
        let params = parse_callback_query("/oauth-callback?state=s&code=c").unwrap();
        assert_eq!(params.code.as_deref(), Some("c"));
        assert_eq!(params.state.as_deref(), Some("s"));
    }

    #[test]
    fn error_parameter_is_captured() {
        let params =
            parse_callback_query("/oauth-callback?error=access_denied&state=s").unwrap();
        assert_eq!(params.error.as_deref(), Some("access_denied"));
        assert!(params.code.is_none());
    }

    #[test]
    fn missing_query_yields_empty_params() {
        let params = parse_callback_query("/oauth-callback").unwrap();
        assert_eq!(params, CallbackParams::default());
    }

    #[test]
    fn percent_escapes_are_decoded() {
        // Authorization codes routinely contain `/`, which arrives escaped as
        // %2F. Failing to decode produces a code the token endpoint rejects.
        let params = parse_callback_query("/oauth-callback?code=4%2F0AeanS0b-x_y").unwrap();
        assert_eq!(params.code.as_deref(), Some("4/0AeanS0b-x_y"));
    }

    #[test]
    fn percent_decode_handles_the_full_range() {
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("a+b"), "a b");
        assert_eq!(percent_decode("%2F"), "/");
        assert_eq!(percent_decode("plain"), "plain");
        assert_eq!(percent_decode(""), "");
    }

    #[test]
    fn malformed_escapes_pass_through() {
        // Better a recognizable code than a silently truncated one.
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%ZZ"), "%ZZ");
    }

    #[test]
    fn multivariate_utf8_escapes_decode() {
        assert_eq!(percent_decode("%E4%B8%AD"), "中");
    }

    #[test]
    fn request_target_is_extracted_from_a_request_line() {
        let head = "GET /oauth-callback?code=x HTTP/1.1\r\nHost: localhost\r\n\r\n";
        assert_eq!(request_target(head), Some("/oauth-callback?code=x"));
    }

    #[test]
    fn malformed_request_line_yields_no_target() {
        assert_eq!(request_target(""), None);
        assert_eq!(request_target("GET"), None);
    }

    #[test]
    fn pasted_redirect_url_is_accepted() {
        let code = extract_code_from_input(
            "http://localhost:51121/oauth-callback?code=4/0AeanS0b-xyz&state=abc",
        )
        .unwrap();
        assert_eq!(code.code, "4/0AeanS0b-xyz");
        assert_eq!(code.state.as_deref(), Some("abc"));
    }

    #[test]
    fn pasted_redirect_url_with_escapes_is_decoded() {
        let code = extract_code_from_input(
            "http://localhost:51121/oauth-callback?code=4%2F0AeanS0b-xyz",
        )
        .unwrap();
        assert_eq!(code.code, "4/0AeanS0b-xyz");
    }

    #[test]
    fn pasted_bare_code_is_accepted() {
        let code = extract_code_from_input("4/0AeanS0b-abcdefghijklmnop").unwrap();
        assert_eq!(code.code, "4/0AeanS0b-abcdefghijklmnop");
        assert!(code.state.is_none());
    }

    #[test]
    fn pasted_error_url_is_rejected_with_the_reason() {
        let error =
            extract_code_from_input("http://localhost:51121/oauth-callback?error=access_denied")
                .unwrap_err();
        assert!(matches!(error, CallbackError::AuthorizationFailed(_)));
        assert!(error.to_string().contains("access_denied"));
    }

    #[test]
    fn pasted_url_without_a_code_is_rejected() {
        let error =
            extract_code_from_input("http://localhost:51121/oauth-callback?state=abc").unwrap_err();
        assert!(matches!(error, CallbackError::MissingCode));
    }

    #[test]
    fn unrelated_url_is_rejected() {
        let error = extract_code_from_input("https://example.com/page?code=abc").unwrap_err();
        assert!(matches!(error, CallbackError::Input(_)));
    }

    #[test]
    fn empty_and_truncated_input_is_rejected() {
        assert!(matches!(
            extract_code_from_input("   ").unwrap_err(),
            CallbackError::Input(_)
        ));
        assert!(matches!(
            extract_code_from_input("4/short").unwrap_err(),
            CallbackError::Input(_)
        ));
    }

    #[test]
    fn surrounding_whitespace_is_tolerated() {
        // Terminals love to include a trailing newline or space.
        let code = extract_code_from_input("  4/0AeanS0b-abcdefghijklmnop  \n").unwrap();
        assert_eq!(code.code, "4/0AeanS0b-abcdefghijklmnop");
    }

    #[tokio::test]
    async fn listener_binds_and_reports_a_redirect_uri() {
        // Port 0 is not in the candidate list, so bind an ephemeral port.
        let listener = CallbackListener::bind_ephemeral_for_test()
            .await
            .expect("should bind");
        let uri = listener.redirect_uri();
        assert!(uri.starts_with("http://localhost:"));
        assert!(uri.ends_with(CALLBACK_PATH));
        assert!(uri.contains(&listener.port().to_string()));
    }

    #[tokio::test]
    async fn a_real_http_callback_delivers_the_code() {
        let listener = CallbackListener::bind_ephemeral_for_test().await.unwrap();
        let port = listener.port();

        let waiting = tokio::spawn(async move {
            listener.wait("expected-state", Duration::from_secs(5)).await
        });

        // Give the accept loop a moment, then act like a browser.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let response = raw_request(
            port,
            "GET /oauth-callback?code=4%2Fabc&state=expected-state HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .await;
        assert!(response.starts_with("HTTP/1.1 200"), "got: {response}");

        let code = waiting.await.unwrap().unwrap();
        assert_eq!(code, "4/abc");
    }

    #[tokio::test]
    async fn unrelated_requests_do_not_consume_the_callback() {
        let listener = CallbackListener::bind_ephemeral_for_test().await.unwrap();
        let port = listener.port();

        let waiting = tokio::spawn(async move {
            listener.wait("s", Duration::from_secs(5)).await
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        // A favicon request must be answered and ignored, not treated as the
        // callback.
        let favicon = raw_request(port, "GET /favicon.ico HTTP/1.1\r\nHost: localhost\r\n\r\n").await;
        assert!(favicon.starts_with("HTTP/1.1 404"), "got: {favicon}");

        let real = raw_request(
            port,
            "GET /oauth-callback?code=real&state=s HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .await;
        assert!(real.starts_with("HTTP/1.1 200"), "got: {real}");

        assert_eq!(waiting.await.unwrap().unwrap(), "real");
    }

    #[tokio::test]
    async fn state_mismatch_aborts() {
        let listener = CallbackListener::bind_ephemeral_for_test().await.unwrap();
        let port = listener.port();

        let waiting = tokio::spawn(async move {
            listener.wait("expected", Duration::from_secs(5)).await
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        let response = raw_request(
            port,
            "GET /oauth-callback?code=x&state=wrong HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .await;
        assert!(response.starts_with("HTTP/1.1 400"), "got: {response}");

        let error = waiting.await.unwrap().unwrap_err();
        assert!(matches!(error, CallbackError::StateMismatch));
    }

    #[tokio::test]
    async fn authorization_error_aborts() {
        let listener = CallbackListener::bind_ephemeral_for_test().await.unwrap();
        let port = listener.port();

        let waiting = tokio::spawn(async move {
            listener.wait("s", Duration::from_secs(5)).await
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        raw_request(
            port,
            "GET /oauth-callback?error=access_denied HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .await;

        let error = waiting.await.unwrap().unwrap_err();
        assert!(matches!(error, CallbackError::AuthorizationFailed(_)));
    }

    #[tokio::test]
    async fn waiting_times_out_when_nothing_arrives() {
        let listener = CallbackListener::bind_ephemeral_for_test().await.unwrap();
        let error = listener
            .wait("s", Duration::from_millis(120))
            .await
            .unwrap_err();
        assert!(matches!(error, CallbackError::Timeout(_)));
    }

    /// Send a raw HTTP request and return the raw response head plus body.
    async fn raw_request(port: u16, request: &str) -> String {
        use tokio::io::AsyncWriteExt as _;

        let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port))
            .await
            .expect("should connect to the listener");
        stream.write_all(request.as_bytes()).await.unwrap();
        stream.flush().await.unwrap();

        let mut response = Vec::new();
        let _ = stream.read_to_end(&mut response).await;
        String::from_utf8_lossy(&response).into_owned()
    }

    impl CallbackListener {
        /// Bind an ephemeral port for tests.
        ///
        /// The production constructor takes a candidate list; tests do not want
        /// to race for a fixed port.
        async fn bind_ephemeral_for_test() -> std::io::Result<Self> {
            let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await?;
            let port = listener.local_addr()?.port();
            Ok(Self {
                listeners: vec![listener],
                port,
            })
        }
    }
}
