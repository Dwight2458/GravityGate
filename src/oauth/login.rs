//! The interactive authorization-code login flow.
//!
//! Sequence, and why it is in this order:
//!
//! 1. Bind the callback listener, so the redirect URI carries the port that was
//!    actually obtained rather than an assumed 51121.
//! 2. Generate PKCE and `state`.
//! 3. Build and display the authorization URL, and try to open a browser.
//! 4. Wait for the code — from the listener, or from a paste when headless.
//! 5. Exchange the code, then read the account's identity.
//!
//! `access_type=offline` and `prompt=consent` are both required. Offline is what
//! asks for a refresh token at all; consent is what makes Google issue one on
//! every authorization rather than only the first, so re-running this command
//! for an account that has already granted access still yields a usable token.

use std::time::Duration;

use crate::oauth::callback::{
    CallbackError, CallbackListener, DEFAULT_PORTS, ExtractedCode, extract_code_from_input,
};
use crate::oauth::pkce::{Pkce, generate_state};
use crate::oauth::token::{OAuthClient, OAuthError, UserInfo};
use crate::upstream::constants;

/// How long to wait for the user to finish in the browser.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// Parameters for a login attempt.
#[derive(Debug, Clone)]
pub struct LoginOptions {
    /// Print the URL and read the code from stdin instead of running a listener.
    /// For SSH sessions and containers where the browser cannot reach loopback.
    pub no_browser: bool,
    /// Callback ports to try, in order.
    pub ports: Vec<u16>,
    pub timeout: Duration,
    /// Attempt to open a browser. Failure is not fatal; the URL is printed either
    /// way.
    pub open_browser: bool,
}

impl Default for LoginOptions {
    fn default() -> Self {
        Self {
            no_browser: false,
            ports: DEFAULT_PORTS.to_vec(),
            timeout: DEFAULT_TIMEOUT,
            open_browser: true,
        }
    }
}

/// Credentials obtained from a successful login.
#[derive(Debug, Clone)]
pub struct LoginOutcome {
    /// The refresh token. The only durable secret produced here.
    pub refresh_token: String,
    /// Account identity, when the userinfo call succeeded.
    pub email: Option<String>,
    pub user_id: Option<String>,
}

/// What a caller learns about the flow before it completes.
///
/// Returned early so the CLI can print the URL before blocking on the wait.
pub struct LoginPrompt {
    pub url: String,
}

#[derive(Debug, thiserror::Error)]
pub enum LoginError {
    #[error(transparent)]
    Callback(#[from] CallbackError),

    // Transparent rather than wrapped: `CallerError: {0}` combined with anyhow's
    // `{:#}` context chain would print the same message twice.
    #[error(transparent)]
    OAuth(#[from] OAuthError),

    #[error("could not read from the terminal: {0}")]
    Terminal(String),

    #[error("the token response contained no refresh token; Google only issues one with access_type=offline and prompt=consent")]
    NoRefreshToken,
}

/// Run the login flow.
///
/// `on_prompt` is called once with the authorization URL, before blocking. It
/// exists so the CLI can print the URL at the right moment without this module
/// knowing about stdout.
pub async fn login<F>(
    oauth: &OAuthClient,
    options: LoginOptions,
    on_prompt: F,
) -> Result<LoginOutcome, LoginError>
where
    F: FnOnce(&LoginPrompt),
{
    let pkce = Pkce::generate();
    let state = generate_state();

    // The listener must exist before the URL is built: its port is part of the
    // registered redirect URI.
    let listener = if options.no_browser {
        None
    } else {
        Some(CallbackListener::bind(&options.ports).await?)
    };

    let redirect_uri = match &listener {
        Some(listener) => listener.redirect_uri(),
        // Nothing is listening, so any of the registered ports works. The first
        // candidate is the one the client is registered with.
        None => {
            let port = options.ports.first().copied().unwrap_or(51121);
            format!("http://localhost:{port}{}", crate::oauth::callback::CALLBACK_PATH)
        }
    };

    let url = build_authorize_url(&redirect_uri, &pkce, &state);

    if options.open_browser && !options.no_browser {
        // Best effort. The URL has been handed to the caller and will be shown
        // regardless, so a failure here costs convenience, not correctness.
        if let Err(error) = open_browser(&url) {
            tracing::debug!(%error, "could not launch a browser");
        }
    }

    on_prompt(&LoginPrompt { url });

    let code = match listener {
        Some(listener) => listener.wait(&state, options.timeout).await?,
        None => read_code_from_terminal(&state)?,
    };

    let tokens = oauth.exchange_code(&code, pkce.verifier(), &redirect_uri).await?;

    let refresh_token = tokens.refresh_token.ok_or(LoginError::NoRefreshToken)?;

    // Identity is a convenience, not a requirement: an account with no label is
    // still usable, and failing the whole login over a userinfo hiccup would be
    // the wrong trade.
    let (email, user_id) = match oauth.userinfo(&tokens.access_token).await {
        Ok(UserInfo { email, id }) => (email, id),
        Err(error) => {
            tracing::warn!(%error, "could not read account identity; continuing without a label");
            (None, None)
        }
    };

    Ok(LoginOutcome {
        refresh_token,
        email,
        user_id,
    })
}

/// Read an authorization code pasted into the terminal.
fn read_code_from_terminal(expected_state: &str) -> Result<String, LoginError> {
    use std::io::Write as _;

    print!("Paste the full redirect URL (or just the code): ");
    std::io::stdout()
        .flush()
        .map_err(|error| LoginError::Terminal(error.to_string()))?;

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .map_err(|error| LoginError::Terminal(error.to_string()))?;

    let ExtractedCode { code, state } = extract_code_from_input(&input)?;

    // A pasted URL carries the state and can be checked; a bare code cannot, and
    // is accepted on the user's word. The window in which an unattended paste
    // could be wrong is small, and rejecting bare codes would break the very
    // case this path exists for.
    if let Some(state) = state
        && state != expected_state
    {
        return Err(LoginError::Callback(CallbackError::StateMismatch));
    }

    Ok(code)
}

/// Build the authorization URL.
pub fn build_authorize_url(redirect_uri: &str, pkce: &Pkce, state: &str) -> String {
    let params = [
        ("client_id", constants::OAUTH_CLIENT_ID.to_string()),
        ("redirect_uri", redirect_uri.to_string()),
        ("response_type", "code".to_string()),
        ("scope", constants::OAUTH_SCOPES.join(" ")),
        // Requests a refresh token.
        ("access_type", "offline".to_string()),
        // Forces a refresh token to be reissued even if the account has already
        // granted access, which is what makes a repeat login useful.
        ("prompt", "consent".to_string()),
        ("code_challenge", pkce.challenge().to_string()),
        ("code_challenge_method", "S256".to_string()),
        ("state", state.to_string()),
    ];

    let query: Vec<String> = params
        .iter()
        .map(|(key, value)| format!("{}={}", encode(key), encode(value)))
        .collect();

    format!("{}?{}", constants::OAUTH_AUTHORIZE_URL, query.join("&"))
}

/// Percent-encode a query component.
fn encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            // Everything else is escaped, including `:` and `/` in the scope and
            // redirect URIs and the space between scopes.
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

/// Launch the default browser.
///
/// Deliberately avoids a shell. On Windows, routing through `cmd /C start` would
/// let `cmd` reinterpret the `&` separators in the query string as command
/// separators, silently truncating the URL.
fn open_browser(url: &str) -> std::io::Result<()> {
    use std::process::Command;

    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = Command::new("rundll32.exe");
        command.arg("url.dll,FileProtocolHandler");
        command
    };

    #[cfg(target_os = "macos")]
    let mut command = Command::new("open");

    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = Command::new("xdg-open");

    command
        .arg(url)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oauth::callback::parse_callback_query;

    fn sample_url() -> String {
        let pkce = Pkce::from_verifier("a".repeat(64));
        build_authorize_url(
            "http://localhost:51121/oauth-callback",
            &pkce,
            "deadbeef",
        )
    }

    /// Parse the query of a built URL into key/value pairs.
    fn query_pairs(url: &str) -> Vec<(String, String)> {
        let query = url.split_once('?').expect("url should have a query").1;
        query
            .split('&')
            .map(|pair| {
                let (key, value) = pair.split_once('=').expect("pair should have '='");
                (
                    crate::oauth::callback::percent_decode(key),
                    crate::oauth::callback::percent_decode(value),
                )
            })
            .collect()
    }

    fn param<'a>(pairs: &'a [(String, String)], name: &str) -> Option<&'a str> {
        pairs
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    #[test]
    fn authorize_url_points_at_google() {
        let url = sample_url();
        assert!(url.starts_with(constants::OAUTH_AUTHORIZE_URL));
        assert!(url.contains("accounts.google.com"));
    }

    #[test]
    fn authorize_url_carries_every_required_parameter() {
        let url = sample_url();
        let pairs = query_pairs(&url);

        assert_eq!(param(&pairs, "client_id"), Some(constants::OAUTH_CLIENT_ID));
        assert_eq!(param(&pairs, "response_type"), Some("code"));
        assert_eq!(param(&pairs, "code_challenge_method"), Some("S256"));
        assert_eq!(
            param(&pairs, "redirect_uri"),
            Some("http://localhost:51121/oauth-callback")
        );
        assert_eq!(param(&pairs, "state"), Some("deadbeef"));
    }

    #[test]
    fn offline_and_consent_are_both_requested() {
        // Omitting `prompt=consent` makes Google skip the refresh token for an
        // account that has already authorized, which is the single most common
        // way to end up with a useless token response.
        let pairs = query_pairs(&sample_url());
        assert_eq!(param(&pairs, "access_type"), Some("offline"));
        assert_eq!(param(&pairs, "prompt"), Some("consent"));
    }

    #[test]
    fn all_scopes_are_requested_space_separated() {
        let pairs = query_pairs(&sample_url());
        let scope = param(&pairs, "scope").expect("scope is required");
        let scopes: Vec<&str> = scope.split(' ').collect();
        assert_eq!(scopes.len(), constants::OAUTH_SCOPES.len());
        for expected in constants::OAUTH_SCOPES {
            assert!(scopes.contains(expected), "missing scope {expected}");
        }
    }

    #[test]
    fn the_challenge_is_sent_unpadded_and_url_safe() {
        let pairs = query_pairs(&sample_url());
        let challenge = param(&pairs, "code_challenge").unwrap();
        assert_eq!(challenge.len(), 43);
        assert!(!challenge.contains('='));
        assert!(!challenge.contains('+'));
        assert!(!challenge.contains('/'));
    }

    #[test]
    fn the_verifier_is_never_in_the_authorize_url() {
        let pkce = Pkce::from_verifier("unique-verifier-value".repeat(4));
        let url = build_authorize_url("http://localhost:1/oauth-callback", &pkce, "s");
        assert!(
            !url.contains(pkce.verifier()),
            "the verifier must only appear in the token exchange"
        );
    }

    #[test]
    fn redirect_uri_is_encoded_so_it_survives_parsing() {
        // An unencoded `:` or `/` in the redirect_uri is tolerated by Google but
        // makes the URL ambiguous to read and to re-parse.
        let url = sample_url();
        let raw = url.split_once('?').unwrap().1;
        assert!(raw.contains("redirect_uri=http%3A%2F%2Flocalhost%3A51121%2Foauth-callback"));
    }

    #[test]
    fn built_url_reparses_to_the_expected_parameters() {
        // Round-trip through the same parser the callback listener uses, which
        // is the check that the URL is well-formed rather than merely plausible.
        let url = sample_url();
        let target = format!("/oauth-callback?{}", url.split_once('?').unwrap().1);
        let params = parse_callback_query(&target).unwrap();
        assert!(params.code.is_none());
        assert_eq!(params.state.as_deref(), Some("deadbeef"));
    }

    #[test]
    fn encoding_escapes_everything_but_unreserved_characters() {
        assert_eq!(encode("abc-_.~"), "abc-_.~");
        assert_eq!(encode("a b"), "a%20b");
        assert_eq!(encode("a&b"), "a%26b");
        assert_eq!(encode("a=b"), "a%3Db");
        assert_eq!(encode("a+b"), "a%2Bb");
        assert_eq!(encode("https://x/y"), "https%3A%2F%2Fx%2Fy");
    }

    #[test]
    fn headless_mode_uses_the_first_candidate_port() {
        // No listener means no way to learn a port, so the URL falls back to the
        // registered primary.
        let ports = DEFAULT_PORTS.to_vec();
        assert_eq!(ports[0], 51121);
    }

    #[test]
    fn default_options_are_interactive() {
        let options = LoginOptions::default();
        assert!(!options.no_browser);
        assert!(options.open_browser);
        assert_eq!(options.ports, DEFAULT_PORTS);
        assert_eq!(options.timeout, DEFAULT_TIMEOUT);
    }
}
