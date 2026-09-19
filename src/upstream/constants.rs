//! Wire-level constants for the Cloud Code Assist upstream.
//!
//! Values here are calibrated against a captured `agy` CLI 1.1.24 request
//! (2026-09-02), preserved in
//! `reference/antigravity-auth/test-fixtures/agy-cli-1.1.24-stream-request.json`.
//! Where the capture and the reference source disagree, the capture wins.

/// Public OAuth client issued to the Antigravity CLI. This is not a secret —
/// it ships inside the distributed client and is identical across every
/// reference implementation.
pub const OAUTH_CLIENT_ID: &str =
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";

/// Companion secret for the client above, for the same reason.
pub const OAUTH_CLIENT_SECRET: &str = "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";

pub const OAUTH_AUTHORIZE_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
pub const OAUTH_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
pub const OAUTH_USERINFO_URL: &str = "https://www.googleapis.com/oauth2/v1/userinfo";

/// Local loopback listener for the authorization-code redirect. Port 51121 is
/// what the CLI itself binds.
pub const OAUTH_REDIRECT_PORT: u16 = 51121;
pub const OAUTH_REDIRECT_PATH: &str = "/oauth-callback";

pub const OAUTH_SCOPES: &[&str] = &[
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
];

/// Primary upstream host. Live CLI traffic goes here rather than to the
/// `.sandbox` variants the older reference projects use.
pub const ENDPOINT_DAILY: &str = "https://daily-cloudcode-pa.googleapis.com";
pub const ENDPOINT_PROD: &str = "https://cloudcode-pa.googleapis.com";

/// Generation endpoint fallback order, daily first.
pub const ENDPOINT_FALLBACKS: &[&str] = &[ENDPOINT_DAILY, ENDPOINT_PROD];

/// Project discovery hits the same hosts in the same order.
pub const LOAD_ENDPOINTS: &[&str] = ENDPOINT_FALLBACKS;

/// Used when the upstream declines to provision a project. Business and
/// Workspace accounts commonly land here. This project is owned by Google and
/// exists to keep the CLI usable; it 403s unless the Cloud AI Companion API is
/// enabled on it, which is why it is a fallback and never a first choice.
pub const DEFAULT_PROJECT_ID: &str = "rising-fact-p41fc";

/// `agy` CLI version we present. Pinned rather than discovered: the reference
/// project fetches a live version string at startup, but pinning keeps request
/// construction deterministic and offline-capable.
pub const AGY_CLI_VERSION: &str = "1.1.24";

/// Change-list number that ships alongside [`AGY_CLI_VERSION`] in the
/// User-Agent. The pair must stay consistent — a mismatched `cl` is a
/// fingerprint in its own right.
pub const AGY_CLI_CHANGE_LIST: &str = "974782877";

pub const API_GENERATE: &str = "/v1internal:generateContent";
pub const API_STREAM_GENERATE: &str = "/v1internal:streamGenerateContent";
pub const API_LOAD_CODE_ASSIST: &str = "/v1internal:loadCodeAssist";
pub const API_ONBOARD_USER: &str = "/v1internal:onboardUser";
pub const API_FETCH_AVAILABLE_MODELS: &str = "/v1internal:fetchAvailableModels";
pub const API_RETRIEVE_USER_QUOTA_SUMMARY: &str = "/v1internal:retrieveUserQuotaSummary";

/// Sentinel accepted by the upstream in place of a real `thoughtSignature` when
/// a signature is unavailable. Used for Gemini targets only.
pub const SKIP_THOUGHT_SIGNATURE: &str = "skip_thought_signature_validator";

/// Signatures shorter than this are treated as absent. Empirically derived by
/// the reference implementations; real signatures run to hundreds of bytes.
pub const MIN_SIGNATURE_LENGTH: usize = 50;

/// How long before stated expiry an access token is considered stale.
pub const TOKEN_EXPIRY_BUFFER_SECS: i64 = 60;

/// Normalised host OS token for the User-Agent. The CLI spells macOS `darwin`.
fn os_type() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        other => other,
    }
}

/// Normalised architecture token for the User-Agent. The CLI spells x86-64 `amd64`.
fn arch() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "amd64",
        other => other,
    }
}

/// Build the User-Agent that identifies us as the Antigravity CLI.
///
/// Shape, exactly:
/// `antigravity/cli/{version} (aidev_client; os_type={os}; arch={arch}; cl={cl}; auth_method={method})`
///
/// We report the *real* host platform rather than impersonating the captured
/// macOS host — a Windows CLI is an equally legitimate configuration, and a
/// fixed foreign platform would be its own tell.
pub fn agy_cli_user_agent() -> String {
    format!(
        "antigravity/cli/{version} (aidev_client; os_type={os}; arch={arch}; cl={cl}; auth_method=consumer)",
        version = AGY_CLI_VERSION,
        os = os_type(),
        arch = arch(),
        cl = AGY_CLI_CHANGE_LIST,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_agent_matches_captured_shape() {
        let ua = agy_cli_user_agent();
        assert!(ua.starts_with("antigravity/cli/1.1.24 (aidev_client; os_type="));
        assert!(ua.contains("; cl=974782877; auth_method=consumer)"));
        // The capture used darwin/arm64; we emit the real host, so only assert
        // that a normalised token was substituted rather than a raw one.
        assert!(!ua.contains("x86_64"), "arch should be normalised to amd64");
        assert!(!ua.contains("macos"), "os should be normalised to darwin");
    }

    #[test]
    fn endpoint_fallbacks_are_daily_then_prod() {
        assert_eq!(ENDPOINT_FALLBACKS, &[ENDPOINT_DAILY, ENDPOINT_PROD]);
    }

    #[test]
    fn stream_path_carries_sse_query() {
        // The capture uses `?alt=sse`; the query is appended by the caller, so
        // assert the base path stays bare.
        assert!(!API_STREAM_GENERATE.contains('?'));
    }
}
