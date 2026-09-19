//! Classifying upstream failures, and deciding what to do about each.
//!
//! This is pure logic on purpose. Every rule here was derived from an
//! unstructured upstream error body, which means every rule is a guess that has
//! to be re-checked whenever the upstream changes. Keeping it free of I/O and
//! state makes those re-checks cheap.
//!
//! The distinctions that matter, and why:
//!
//! - **A rate limit is not an account problem.** A quota exhaustion clears on its
//!   own; a verification demand or an ineligibility holds until a human acts.
//!   Conflating them either rotates away from a recoverable account or keeps
//!   retrying one that will never work.
//! - **Capacity exhaustion is not quota exhaustion.** A capacity error means the
//!   *model* is busy, so the account is fine and the same account should be
//!   retried shortly. Rotating accounts for it burns the pool for nothing.
//! - **Backoff has to be shaped per reason.** Quota exhaustions escalate over
//!   hours; rate limits clear in seconds. One ladder cannot serve both.

use std::time::Duration;

use reqwest::header::HeaderMap;
use reqwest::StatusCode;
use serde_json::Value;

/// Why the upstream refused a request.
///
/// Mirrors the reference implementation's taxonomy, which is the distilled
/// result of dealing with this API in production.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitReason {
    /// The account's quota for this pool is spent. Clears on a slow ladder.
    QuotaExhausted,
    /// Short-term request-rate limiting. Clears quickly.
    RateLimitExceeded,
    /// The model is busy. The account is fine; retrying the same one is correct.
    ModelCapacityExhausted,
    /// An upstream fault. Not the account's fault, not the request's fault.
    ServerError,
    /// Unrecognised. Treated conservatively.
    Unknown,
}

/// The quota-exhaustion ladder, indexed by how many times in a row this has
/// happened. Escalating rather than fixed: a second exhaustion immediately after
/// the first means the window is longer than the first estimate suggested.
const QUOTA_EXHAUSTED_BACKOFFS: &[Duration] = &[
    Duration::from_secs(60),
    Duration::from_secs(5 * 60),
    Duration::from_secs(30 * 60),
    Duration::from_secs(2 * 60 * 60),
];

const RATE_LIMIT_EXCEEDED_BACKOFF: Duration = Duration::from_secs(30);
const MODEL_CAPACITY_BASE_BACKOFF: Duration = Duration::from_secs(45);
const MODEL_CAPACITY_JITTER: Duration = Duration::from_secs(30);
const SERVER_ERROR_BACKOFF: Duration = Duration::from_secs(20);
const UNKNOWN_BACKOFF: Duration = Duration::from_secs(60);

/// Floor on any computed backoff, so a nonsensical upstream hint cannot turn
/// into a hot retry loop.
pub const MIN_BACKOFF: Duration = Duration::from_secs(2);

/// A permanent or operator-resolvable problem with the account itself.
///
/// Separate from a rate limit because the remedies are different: an operator
/// has to visit a URL, or remove the account.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccountProblem {
    /// The account holder must complete a verification step.
    VerificationRequired { url: Option<String> },
    /// The upstream will not serve this account at all.
    Ineligible { reason: String },
    /// Disabled for a terms-of-service violation. Not recoverable.
    Banned { reason: String },
    /// The credential is no longer accepted.
    AuthInvalid { reason: String },
}

impl AccountProblem {
    /// Whether an operator could plausibly fix this.
    pub fn is_recoverable(&self) -> bool {
        matches!(self, Self::VerificationRequired { .. })
    }

    /// Short label for logs and the status table.
    pub fn label(&self) -> &'static str {
        match self {
            Self::VerificationRequired { .. } => "verification-required",
            Self::Ineligible { .. } => "account-ineligible",
            Self::Banned { .. } => "banned",
            Self::AuthInvalid { .. } => "auth-invalid",
        }
    }
}

/// What an upstream failure means for the next attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpstreamFailure {
    /// Retryable after a delay. `reset_in` is the upstream's own estimate when
    /// it gave one, which is always more accurate than our ladder.
    RateLimited {
        reason: RateLimitReason,
        reset_in: Option<Duration>,
    },
    /// The account cannot serve this request until something changes.
    Account(AccountProblem),
    /// A server-side fault. Not attributable to the account.
    Server { status: u16 },
    /// A transport-level failure: connection, TLS, timeout.
    Network,
    /// The request itself was rejected. Retrying it unchanged cannot help.
    BadRequest { message: String },
    /// The endpoint or project refused. Worth trying another endpoint.
    Refused { status: u16, message: String },
}

impl UpstreamFailure {
    /// Whether retrying the same request against another account could help.
    pub fn worth_another_account(&self) -> bool {
        match self {
            Self::RateLimited { reason, .. } => {
                // Capacity exhaustion is a property of the model, not the
                // account, so moving accounts wastes the pool.
                *reason != RateLimitReason::ModelCapacityExhausted
            }
            Self::Account(problem) => {
                // A verification demand or a ban is per-account.
                !matches!(problem, AccountProblem::AuthInvalid { .. })
            }
            Self::Server { .. } | Self::Network => true,
            // The endpoint refused, not the account.
            Self::Refused { .. } => true,
            Self::BadRequest { .. } => false,
        }
    }

    /// Whether retrying the same account after a delay could help.
    ///
    /// Network faults are included deliberately. A transport failure is far more
    /// often local — DNS, TCP, TLS, a stalled read — than it is specific to one
    /// account, so rotating on it spends another account's goodwill for nothing.
    /// The retry loop bounds this with the same budget it uses for capacity.
    pub fn worth_retrying_same_account(&self) -> bool {
        matches!(
            self,
            Self::RateLimited {
                reason: RateLimitReason::ModelCapacityExhausted | RateLimitReason::ServerError,
                ..
            } | Self::Server { .. }
            | Self::Network
        )
    }
}

/// Classify a non-2xx upstream response.
pub fn classify(status: StatusCode, headers: &HeaderMap, body: &str) -> UpstreamFailure {
    let code = status.as_u16();

    // Status-based classification first: it is authoritative where it applies,
    // and body keywords are a fallback rather than a competitor.
    match code {
        401 => {
            return UpstreamFailure::Account(AccountProblem::AuthInvalid {
                reason: first_message(body).unwrap_or_else(|| "unauthenticated".into()),
            });
        }
        403 => return classify_forbidden(body),
        500 => return UpstreamFailure::Server { status: 500 },
        // 503 and 529 are the upstream's overload signals.
        503 | 529 => {
            return UpstreamFailure::RateLimited {
                reason: RateLimitReason::ModelCapacityExhausted,
                reset_in: parse_reset_hint(headers, body),
            };
        }
        404 => {
            return UpstreamFailure::Refused {
                status: 404,
                message: first_message(body).unwrap_or_else(|| "not found".into()),
            };
        }
        _ => {}
    }

    if code == 429 {
        return UpstreamFailure::RateLimited {
            reason: classify_rate_limit_reason(status, body),
            reset_in: parse_reset_hint(headers, body),
        };
    }

    // A 400 carrying a quota message is a disguised rate limit; the reference
    // implementations see this shape from the upstream's error mapper.
    if code == 400 {
        let reason = classify_rate_limit_reason(status, body);
        if reason != RateLimitReason::Unknown {
            return UpstreamFailure::RateLimited {
                reason,
                reset_in: parse_reset_hint(headers, body),
            };
        }
        return UpstreamFailure::BadRequest {
            message: first_message(body).unwrap_or_else(|| body.chars().take(400).collect()),
        };
    }

    if status.is_server_error() {
        return UpstreamFailure::Server { status: code };
    }

    UpstreamFailure::Refused {
        status: code,
        message: first_message(body).unwrap_or_else(|| body.chars().take(400).collect()),
    }
}

/// Classify a 403, which is where account problems surface.
///
/// A 403 is overloaded upstream: it carries bans, verification demands,
/// ineligibility, and plain permission failures. The discriminator is entirely
/// in the body.
fn classify_forbidden(body: &str) -> UpstreamFailure {
    let lowered = body.to_ascii_lowercase();

    // A terms-of-service ban. Both phrases are required: `has been disabled`
    // alone appears in milder messages.
    if lowered.contains("has been disabled") && lowered.contains("violation of terms of service") {
        return UpstreamFailure::Account(AccountProblem::Banned {
            reason: first_message(body).unwrap_or_else(|| "terms of service violation".into()),
        });
    }

    if contains_token(&lowered, "account_ineligible") {
        return UpstreamFailure::Account(AccountProblem::Ineligible {
            reason: first_message(body).unwrap_or_else(|| "account_ineligible".into()),
        });
    }

    // `validation_required` is the marker the account-access probe looks for; a
    // companion URL is what the account holder has to visit.
    if lowered.contains("validation_required") {
        return UpstreamFailure::Account(AccountProblem::VerificationRequired {
            url: extract_verification_url(body),
        });
    }

    // The related disable messages also require holder action, and the reference
    // implementations treat them the same way.
    if contains_token(&lowered, "account_disabled") || contains_token(&lowered, "user_disabled") {
        return UpstreamFailure::Account(AccountProblem::VerificationRequired {
            url: extract_verification_url(body),
        });
    }

    if lowered.contains("permission_denied") || lowered.contains("permission denied") {
        return UpstreamFailure::Refused {
            status: 403,
            message: first_message(body).unwrap_or_else(|| "permission denied".into()),
        };
    }

    UpstreamFailure::Refused {
        status: 403,
        message: first_message(body).unwrap_or_else(|| body.chars().take(400).collect()),
    }
}

/// Decide which of the four rate-limit shapes this is.
///
/// Checked in a fixed order because the phrases overlap: "resource exhausted" in
/// a message that also says "quota" is a capacity problem, not a quota problem,
/// and the order below encodes that precedence.
pub fn classify_rate_limit_reason(status: StatusCode, body: &str) -> RateLimitReason {
    match status.as_u16() {
        529 | 503 => return RateLimitReason::ModelCapacityExhausted,
        500 => return RateLimitReason::ServerError,
        _ => {}
    }

    // An explicit machine-readable reason wins where present.
    if let Some(reason) = explicit_reason(body) {
        return reason;
    }

    let lowered = body.to_ascii_lowercase();

    if lowered.contains("capacity")
        || lowered.contains("overloaded")
        || lowered.contains("resource exhausted")
        || lowered.contains("resource_exhausted")
    {
        return RateLimitReason::ModelCapacityExhausted;
    }
    if lowered.contains("per minute")
        || lowered.contains("rate limit")
        || lowered.contains("rate_limit")
        || lowered.contains("too many requests")
    {
        return RateLimitReason::RateLimitExceeded;
    }
    if lowered.contains("exhausted") || lowered.contains("quota") {
        return RateLimitReason::QuotaExhausted;
    }

    RateLimitReason::Unknown
}

/// Read a machine-readable reason from the error body when the upstream supplies
/// one, rather than inferring it from prose.
fn explicit_reason(body: &str) -> Option<RateLimitReason> {
    let value: Value = serde_json::from_str(body).ok()?;
    let reason = value
        .pointer("/error/status")
        .or_else(|| value.pointer("/error/details/0/reason"))
        .and_then(Value::as_str)?;

    match reason.to_ascii_uppercase().as_str() {
        "QUOTA_EXHAUSTED" => Some(RateLimitReason::QuotaExhausted),
        "RATE_LIMIT_EXCEEDED" => Some(RateLimitReason::RateLimitExceeded),
        "MODEL_CAPACITY_EXHAUSTED" => Some(RateLimitReason::ModelCapacityExhausted),
        "RESOURCE_EXHAUSTED" => Some(RateLimitReason::QuotaExhausted),
        "SERVER_ERROR" | "INTERNAL" => Some(RateLimitReason::ServerError),
        _ => None,
    }
}

/// Compute how long to wait before retrying.
///
/// An upstream-supplied reset always wins: it reflects server state that our
/// ladder is only guessing at. `consecutive_failures` escalates the quota ladder.
pub fn backoff_for(
    reason: RateLimitReason,
    consecutive_failures: u32,
    reset_in: Option<Duration>,
    jitter: f64,
) -> Duration {
    if let Some(reset) = reset_in.filter(|reset| !reset.is_zero()) {
        return reset.max(MIN_BACKOFF);
    }

    let base = match reason {
        RateLimitReason::QuotaExhausted => {
            let index = (consecutive_failures as usize).min(QUOTA_EXHAUSTED_BACKOFFS.len() - 1);
            QUOTA_EXHAUSTED_BACKOFFS[index]
        }
        RateLimitReason::RateLimitExceeded => RATE_LIMIT_EXCEEDED_BACKOFF,
        RateLimitReason::ModelCapacityExhausted => {
            // Jittered around the base: several clients hitting a busy model
            // should not synchronise their retries.
            let span = MODEL_CAPACITY_JITTER.as_secs_f64();
            let offset = (jitter.clamp(0.0, 1.0) * span) - span / 2.0;
            MODEL_CAPACITY_BASE_BACKOFF.saturating_add(Duration::from_secs_f64(offset.max(0.0)))
        }
        RateLimitReason::ServerError => SERVER_ERROR_BACKOFF,
        RateLimitReason::Unknown => UNKNOWN_BACKOFF,
    };

    base.max(MIN_BACKOFF)
}

/// Extract a reset estimate from headers or the body.
///
/// Tried in order of trustworthiness: `Retry-After` is a standard header, the
/// `retry-after-ms` extension is more precise when present, and the body's
/// `RetryInfo` is the upstream's own structured estimate.
pub fn parse_reset_hint(headers: &HeaderMap, body: &str) -> Option<Duration> {
    if let Some(value) = headers.get("retry-after-ms").and_then(|v| v.to_str().ok())
        && let Ok(ms) = value.trim().parse::<f64>()
        && ms > 0.0
    {
        return Some(Duration::from_millis(ms as u64));
    }

    if let Some(value) = headers.get("retry-after").and_then(|v| v.to_str().ok())
        && let Some(duration) = parse_retry_after(value)
    {
        return Some(duration);
    }

    let value: Value = serde_json::from_str(body).ok()?;

    // `error.details[]` with a RetryInfo type carries `retryDelay`, e.g.
    // `"3.957525076s"`.
    if let Some(details) = value.pointer("/error/details").and_then(Value::as_array) {
        for detail in details {
            let is_retry_info = detail
                .get("@type")
                .and_then(Value::as_str)
                .is_some_and(|t| t.ends_with("RetryInfo"));
            if is_retry_info
                && let Some(delay) = detail.get("retryDelay").and_then(Value::as_str)
                && let Some(duration) = parse_duration_string(delay)
            {
                return Some(duration);
            }
        }
    }

    // Unstructured fallbacks seen in the wild.
    for key in ["retryDelay", "quotaResetDelay"] {
        if let Some(delay) = value.pointer(&format!("/error/{key}")).and_then(Value::as_str)
            && let Some(duration) = parse_duration_string(delay)
        {
            return Some(duration);
        }
    }
    if let Some(text) = value.pointer("/error/message").and_then(Value::as_str)
        && let Some(duration) = parse_duration_in_prose(text)
    {
        return Some(duration);
    }

    None
}

/// Parse a `Retry-After` header value, which is either seconds or an HTTP date.
fn parse_retry_after(value: &str) -> Option<Duration> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(seconds) = value.parse::<f64>() {
        return (seconds > 0.0).then(|| Duration::from_secs_f64(seconds));
    }
    // An HTTP date is also legal. Converting one properly needs a date library;
    // the relative form is what this API actually sends, so an absolute date is
    // reported as absent rather than mis-parsed.
    None
}

/// Parse a Go-style duration such as `3.957525076s`, `754.43ms`, or `1h23m45s`.
pub fn parse_duration_string(value: &str) -> Option<Duration> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }

    // Plain integer means seconds.
    if let Ok(seconds) = value.parse::<f64>() {
        return (seconds > 0.0).then(|| Duration::from_secs_f64(seconds));
    }

    let mut total = Duration::ZERO;
    let mut number = String::new();
    let mut matched = false;
    let mut chars = value.chars().peekable();

    while let Some(character) = chars.next() {
        if character.is_ascii_digit() || character == '.' {
            number.push(character);
            continue;
        }
        let Ok(amount) = number.parse::<f64>() else {
            return None;
        };
        number.clear();

        // `ms` must be read before `m`, or milliseconds parse as minutes.
        let unit: String = if character == 'm' && chars.peek() == Some(&'s') {
            chars.next();
            "ms".into()
        } else {
            character.to_string()
        };

        let scale = match unit.as_str() {
            "ms" => 0.001,
            "s" => 1.0,
            "m" => 60.0,
            "h" => 3600.0,
            _ => return None,
        };
        total = total.saturating_add(Duration::from_secs_f64(amount * scale));
        matched = true;
    }

    (matched && !total.is_zero()).then_some(total)
}

/// Find a wait duration mentioned in prose, e.g. `retry after 30 seconds`.
fn parse_duration_in_prose(text: &str) -> Option<Duration> {
    let lowered = text.to_ascii_lowercase();
    let marker = lowered.find("retry after")?;
    let rest = &lowered[marker + "retry after".len()..];

    let digits: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    if digits.is_empty() {
        return None;
    }
    let amount: f64 = digits.parse().ok()?;

    let unit = rest.trim_start()[digits.len()..].trim_start();
    let scale = if unit.starts_with("second") || unit.starts_with('s') {
        1.0
    } else if unit.starts_with("minute") || unit.starts_with('m') {
        60.0
    } else if unit.starts_with("hour") || unit.starts_with('h') {
        3600.0
    } else {
        return None;
    };

    let seconds = amount * scale;
    (seconds > 0.0).then(|| Duration::from_secs_f64(seconds))
}

/// Whether a lowercase haystack contains `needle` as a standalone token.
///
/// Guards against matching `not_account_ineligible` or a substring of a longer
/// identifier, which the reference implementation handles with an equivalent
/// boundary check.
fn contains_token(haystack: &str, needle: &str) -> bool {
    let mut from = 0;
    while let Some(found) = haystack[from..].find(needle) {
        let start = from + found;
        let end = start + needle.len();

        let before_ok = start == 0
            || !haystack.as_bytes()[start - 1].is_ascii_alphanumeric()
                && haystack.as_bytes()[start - 1] != b'_';
        let after_ok = end == haystack.len()
            || !haystack.as_bytes()[end].is_ascii_alphanumeric()
                && haystack.as_bytes()[end] != b'_';

        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

/// Pull the human-readable message out of an error body.
///
/// The upstream sometimes nests an Anthropic-shaped envelope inside the Google
/// error's `message`, so a message that is itself JSON is unwrapped once. Without
/// this a client sees an escaped JSON blob instead of the sentence describing
/// what went wrong.
fn first_message(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let message = value
        .pointer("/error/message")
        .and_then(Value::as_str)
        .or_else(|| value.get("error_description").and_then(Value::as_str))
        .or_else(|| value.get("error").and_then(Value::as_str))?;

    let trimmed = message.trim();
    if trimmed.starts_with('{')
        && let Ok(nested) = serde_json::from_str::<Value>(trimmed)
        && let Some(inner) = nested.pointer("/error/message").and_then(Value::as_str)
    {
        return Some(inner.to_string());
    }
    Some(message.to_string())
}

/// Find the URL an account holder has to visit to clear a verification demand.
fn extract_verification_url(body: &str) -> Option<String> {
    // The structured location, when the upstream provides one.
    if let Ok(value) = serde_json::from_str::<Value>(body)
        && let Some(details) = value.pointer("/error/details").and_then(Value::as_array)
    {
        for detail in details {
            for pointer in ["/metadata/validation_url", "/metadata/validationUrl"] {
                if let Some(url) = detail.pointer(pointer).and_then(Value::as_str) {
                    return Some(url.to_string());
                }
            }
        }
    }

    // Otherwise scan for a sign-in URL. Prefer the `signin/continue` form, which
    // is the one that actually carries the challenge.
    const SIGNIN: &str = "https://accounts.google.com/signin/continue";
    if let Some(start) = body.find(SIGNIN) {
        return Some(take_url(&body[start..]));
    }
    const ACCOUNTS: &str = "https://accounts.google.com/";
    body.find(ACCOUNTS).map(|start| take_url(&body[start..]))
}

/// Take a URL from the start of `text`, stopping at a character that cannot
/// appear in one.
fn take_url(text: &str) -> String {
    text.chars()
        .take_while(|c| !c.is_whitespace() && !matches!(c, '"' | '\'' | '<' | '>' | '\\'))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        map
    }

    // -- reason classification ---------------------------------------------

    #[test]
    fn capacity_phrases_map_to_capacity() {
        for body in [
            r#"{"error":{"message":"The model is overloaded"}}"#,
            r#"{"error":{"message":"model capacity exceeded"}}"#,
            r#"{"error":{"message":"RESOURCE_EXHAUSTED: capacity"}}"#,
        ] {
            assert_eq!(
                classify_rate_limit_reason(StatusCode::TOO_MANY_REQUESTS, body),
                RateLimitReason::ModelCapacityExhausted,
                "for {body}"
            );
        }
    }

    #[test]
    fn rate_limit_phrases_map_to_rate_limit() {
        for body in [
            r#"{"error":{"message":"rate limit exceeded"}}"#,
            r#"{"error":{"message":"too many requests per minute"}}"#,
        ] {
            assert_eq!(
                classify_rate_limit_reason(StatusCode::TOO_MANY_REQUESTS, body),
                RateLimitReason::RateLimitExceeded,
                "for {body}"
            );
        }
    }

    #[test]
    fn quota_phrases_map_to_quota() {
        for body in [
            r#"{"error":{"message":"quota exhausted"}}"#,
            r#"{"error":{"message":"You have exceeded your quota for this model"}}"#,
        ] {
            assert_eq!(
                classify_rate_limit_reason(StatusCode::TOO_MANY_REQUESTS, body),
                RateLimitReason::QuotaExhausted,
                "for {body}"
            );
        }
    }

    #[test]
    fn capacity_outranks_quota_when_both_phrases_appear() {
        // "resource exhausted" plus "quota" is a capacity problem; the ordering
        // of the checks encodes that.
        let body = r#"{"error":{"message":"resource exhausted: quota for this model"}}"#;
        assert_eq!(
            classify_rate_limit_reason(StatusCode::TOO_MANY_REQUESTS, body),
            RateLimitReason::ModelCapacityExhausted
        );
    }

    #[test]
    fn the_word_capacity_wins_over_the_word_quota() {
        // Genuinely ambiguous, and documented as such. The upstream's own quota
        // message reads "you have exhausted your capacity on <model>", which
        // contains "capacity"; the reference implementation orders its checks
        // capacity-first and this matches it. The consequence of getting it
        // wrong is only which retry shape is chosen, and both eventually retry.
        let body = r#"{"error":{"message":"You have exhausted your capacity on this model. Quota will reset after 1h."}}"#;
        assert_eq!(
            classify_rate_limit_reason(StatusCode::TOO_MANY_REQUESTS, body),
            RateLimitReason::ModelCapacityExhausted
        );
    }

    #[test]
    fn status_outranks_body_phrases() {
        // A 503 is capacity regardless of what the body claims.
        assert_eq!(
            classify_rate_limit_reason(StatusCode::SERVICE_UNAVAILABLE, "quota exhausted"),
            RateLimitReason::ModelCapacityExhausted
        );
        assert_eq!(
            classify_rate_limit_reason(StatusCode::INTERNAL_SERVER_ERROR, "rate limit"),
            RateLimitReason::ServerError
        );
    }

    #[test]
    fn an_explicit_reason_field_is_trusted() {
        let body = r#"{"error":{"status":"QUOTA_EXHAUSTED","message":"anything at all"}}"#;
        assert_eq!(
            classify_rate_limit_reason(StatusCode::TOO_MANY_REQUESTS, body),
            RateLimitReason::QuotaExhausted
        );
    }

    #[test]
    fn an_unrecognised_body_is_unknown() {
        assert_eq!(
            classify_rate_limit_reason(StatusCode::TOO_MANY_REQUESTS, "something new"),
            RateLimitReason::Unknown
        );
    }

    // -- top-level classification ------------------------------------------

    #[test]
    fn a_401_is_an_invalid_credential() {
        let failure = classify(
            StatusCode::UNAUTHORIZED,
            &HeaderMap::new(),
            r#"{"error":{"message":"UNAUTHENTICATED"}}"#,
        );
        assert!(matches!(
            failure,
            UpstreamFailure::Account(AccountProblem::AuthInvalid { .. })
        ));
    }

    #[test]
    fn a_403_terms_violation_is_a_ban() {
        let body = r#"{"error":{"message":"This account has been disabled due to a violation of terms of service."}}"#;
        let failure = classify(StatusCode::FORBIDDEN, &HeaderMap::new(), body);
        match failure {
            UpstreamFailure::Account(problem) => {
                assert!(matches!(problem, AccountProblem::Banned { .. }));
                assert!(!problem.is_recoverable());
            }
            other => panic!("expected a ban, got {other:?}"),
        }
    }

    #[test]
    fn a_403_requires_both_ban_phrases() {
        // `has been disabled` alone appears in messages an operator can fix.
        let body = r#"{"error":{"message":"Your project has been disabled"}}"#;
        let failure = classify(StatusCode::FORBIDDEN, &HeaderMap::new(), body);
        assert!(
            !matches!(
                failure,
                UpstreamFailure::Account(AccountProblem::Banned { .. })
            ),
            "a lone disable phrase must not be read as a ban"
        );
    }

    #[test]
    fn a_403_validation_required_captures_the_url() {
        let body = r#"{"error":{"message":"validation_required","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","metadata":{"validation_url":"https://accounts.google.com/signin/continue?plt=abc"}}]}}"#;
        let failure = classify(StatusCode::FORBIDDEN, &HeaderMap::new(), body);
        match failure {
            UpstreamFailure::Account(problem @ AccountProblem::VerificationRequired { .. }) => {
                assert!(problem.is_recoverable());
                assert_eq!(
                    problem,
                    AccountProblem::VerificationRequired {
                        url: Some("https://accounts.google.com/signin/continue?plt=abc".into())
                    }
                );
            }
            other => panic!("expected a verification demand, got {other:?}"),
        }
    }

    #[test]
    fn a_403_validation_required_without_structured_url_still_finds_one() {
        let body = r#"{"error":{"message":"validation_required: visit https://accounts.google.com/signin/continue?plt=xyz to continue"}}"#;
        match classify(StatusCode::FORBIDDEN, &HeaderMap::new(), body) {
            UpstreamFailure::Account(AccountProblem::VerificationRequired { url }) => {
                let url = url.expect("a url");
                assert!(url.starts_with("https://accounts.google.com/signin/continue"));
                assert!(!url.contains(' '), "the url must stop at whitespace");
            }
            other => panic!("expected a verification demand, got {other:?}"),
        }
    }

    #[test]
    fn a_403_account_ineligible_is_detected() {
        let body = r#"{"error":{"message":"ACCOUNT_INELIGIBLE"}}"#;
        match classify(StatusCode::FORBIDDEN, &HeaderMap::new(), body) {
            UpstreamFailure::Account(problem @ AccountProblem::Ineligible { .. }) => {
                assert!(!problem.is_recoverable());
                assert_eq!(problem.label(), "account-ineligible");
            }
            other => panic!("expected ineligibility, got {other:?}"),
        }
    }

    #[test]
    fn ineligibility_detection_requires_a_whole_token() {
        // A substring inside a longer identifier must not trip the check.
        let body = r#"{"error":{"message":"my_account_ineligible_reason_field"}}"#;
        let failure = classify(StatusCode::FORBIDDEN, &HeaderMap::new(), body);
        assert!(!matches!(
            failure,
            UpstreamFailure::Account(AccountProblem::Ineligible { .. })
        ));
    }

    #[test]
    fn account_disabled_is_treated_as_a_verification_demand() {
        let body = r#"{"error":{"message":"ACCOUNT_DISABLED"}}"#;
        assert!(matches!(
            classify(StatusCode::FORBIDDEN, &HeaderMap::new(), body),
            UpstreamFailure::Account(AccountProblem::VerificationRequired { .. })
        ));
    }

    #[test]
    fn a_plain_permission_denied_is_not_an_account_problem() {
        let body = r#"{"error":{"message":"Permission 'cloudaicompanion.companions.generateChat' denied"}}"#;
        assert!(matches!(
            classify(StatusCode::FORBIDDEN, &HeaderMap::new(), body),
            UpstreamFailure::Refused { status: 403, .. }
        ));
    }

    #[test]
    fn a_400_with_quota_wording_is_a_rate_limit() {
        // The upstream's error mapper disguises quota failures as 400s.
        let body = r#"{"error":{"message":"You have exhausted your quota"}}"#;
        assert!(matches!(
            classify(StatusCode::BAD_REQUEST, &HeaderMap::new(), body),
            UpstreamFailure::RateLimited {
                reason: RateLimitReason::QuotaExhausted,
                ..
            }
        ));
    }

    #[test]
    fn a_genuine_400_is_a_bad_request() {
        let body = r#"{"error":{"message":"Invalid JSON payload received"}}"#;
        match classify(StatusCode::BAD_REQUEST, &HeaderMap::new(), body) {
            UpstreamFailure::BadRequest { message } => assert!(message.contains("Invalid JSON")),
            other => panic!("expected a bad request, got {other:?}"),
        }
    }

    #[test]
    fn a_404_is_a_refusal_naming_the_endpoint() {
        assert!(matches!(
            classify(StatusCode::NOT_FOUND, &HeaderMap::new(), "{}"),
            UpstreamFailure::Refused { status: 404, .. }
        ));
    }

    #[test]
    fn a_503_is_capacity() {
        assert!(matches!(
            classify(StatusCode::SERVICE_UNAVAILABLE, &HeaderMap::new(), ""),
            UpstreamFailure::RateLimited {
                reason: RateLimitReason::ModelCapacityExhausted,
                ..
            }
        ));
    }

    #[test]
    fn a_500_is_a_server_fault() {
        assert!(matches!(
            classify(StatusCode::INTERNAL_SERVER_ERROR, &HeaderMap::new(), ""),
            UpstreamFailure::Server { status: 500 }
        ));
    }

    // -- retry policy -------------------------------------------------------

    #[test]
    fn capacity_errors_do_not_justify_another_account() {
        // The model is busy, not the account; rotating wastes the pool.
        let failure = UpstreamFailure::RateLimited {
            reason: RateLimitReason::ModelCapacityExhausted,
            reset_in: None,
        };
        assert!(!failure.worth_another_account());
        assert!(failure.worth_retrying_same_account());
    }

    #[test]
    fn quota_exhaustion_justifies_another_account() {
        let failure = UpstreamFailure::RateLimited {
            reason: RateLimitReason::QuotaExhausted,
            reset_in: None,
        };
        assert!(failure.worth_another_account());
        assert!(!failure.worth_retrying_same_account());
    }

    #[test]
    fn a_verification_demand_justifies_another_account() {
        let failure = UpstreamFailure::Account(AccountProblem::VerificationRequired { url: None });
        assert!(failure.worth_another_account());
        assert!(!failure.worth_retrying_same_account());
    }

    #[test]
    fn an_invalid_credential_does_not_justify_retrying_anything() {
        let failure = UpstreamFailure::Account(AccountProblem::AuthInvalid {
            reason: "revoked".into(),
        });
        assert!(!failure.worth_another_account());
        assert!(!failure.worth_retrying_same_account());
    }

    #[test]
    fn a_bad_request_is_never_retried() {
        let failure = UpstreamFailure::BadRequest {
            message: "malformed".into(),
        };
        assert!(!failure.worth_another_account());
        assert!(!failure.worth_retrying_same_account());
    }

    #[test]
    fn network_and_server_faults_are_retried_everywhere() {
        for failure in [
            UpstreamFailure::Network,
            UpstreamFailure::Server { status: 502 },
        ] {
            assert!(failure.worth_another_account(), "{failure:?}");
            assert!(failure.worth_retrying_same_account(), "{failure:?}");
        }
    }

    // -- backoff ------------------------------------------------------------

    #[test]
    fn an_upstream_reset_always_wins() {
        let computed = backoff_for(
            RateLimitReason::QuotaExhausted,
            0,
            Some(Duration::from_secs(7)),
            0.5,
        );
        assert_eq!(computed, Duration::from_secs(7));
    }

    #[test]
    fn a_zero_reset_is_ignored_in_favour_of_the_ladder() {
        let computed = backoff_for(
            RateLimitReason::RateLimitExceeded,
            0,
            Some(Duration::ZERO),
            0.5,
        );
        assert_eq!(computed, RATE_LIMIT_EXCEEDED_BACKOFF);
    }

    #[test]
    fn the_quota_ladder_escalates_and_then_plateaus() {
        let rung = |failures: u32| {
            backoff_for(RateLimitReason::QuotaExhausted, failures, None, 0.5)
        };
        assert_eq!(rung(0), Duration::from_secs(60));
        assert_eq!(rung(1), Duration::from_secs(300));
        assert_eq!(rung(2), Duration::from_secs(1800));
        assert_eq!(rung(3), Duration::from_secs(7200));
        // Past the end, it stays at the longest rung rather than growing.
        assert_eq!(rung(99), Duration::from_secs(7200));
    }

    #[test]
    fn capacity_backoff_stays_within_its_band() {
        for jitter in [0.0, 0.5, 1.0] {
            let computed =
                backoff_for(RateLimitReason::ModelCapacityExhausted, 0, None, jitter);
            assert!(
                computed >= MODEL_CAPACITY_BASE_BACKOFF && computed <= MIN_BACKOFF.max(
                    MODEL_CAPACITY_BASE_BACKOFF + MODEL_CAPACITY_JITTER
                ),
                "jitter {jitter} produced {computed:?}"
            );
        }
    }

    #[test]
    fn every_backoff_respects_the_floor() {
        // A hot retry loop is worse than any amount of waiting.
        for reason in [
            RateLimitReason::QuotaExhausted,
            RateLimitReason::RateLimitExceeded,
            RateLimitReason::ModelCapacityExhausted,
            RateLimitReason::ServerError,
            RateLimitReason::Unknown,
        ] {
            assert!(
                backoff_for(reason, 0, None, 0.0) >= MIN_BACKOFF,
                "{reason:?} fell below the floor"
            );
            assert!(backoff_for(reason, 0, Some(Duration::from_millis(1)), 0.0) >= MIN_BACKOFF);
        }
    }

    // -- hint parsing -------------------------------------------------------

    #[test]
    fn retry_after_ms_header_is_preferred() {
        let map = headers(&[("retry-after-ms", "1500"), ("retry-after", "99")]);
        assert_eq!(
            parse_reset_hint(&map, "{}"),
            Some(Duration::from_millis(1500))
        );
    }

    #[test]
    fn retry_after_seconds_header_is_read() {
        let map = headers(&[("retry-after", "30")]);
        assert_eq!(parse_reset_hint(&map, "{}"), Some(Duration::from_secs(30)));
    }

    #[test]
    fn a_fractional_retry_after_is_read() {
        let map = headers(&[("retry-after", "0.5")]);
        assert_eq!(
            parse_reset_hint(&map, "{}"),
            Some(Duration::from_millis(500))
        );
    }

    #[test]
    fn an_http_date_retry_after_is_ignored_rather_than_mis_parsed() {
        // Reading it as seconds would produce a wildly wrong wait.
        let map = headers(&[("retry-after", "Wed, 21 Oct 2026 07:28:00 GMT")]);
        assert_eq!(parse_reset_hint(&map, "{}"), None);
    }

    #[test]
    fn retry_info_in_the_body_is_read() {
        let body = r#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"3.957525076s"}]}}"#;
        let parsed = parse_reset_hint(&HeaderMap::new(), body).unwrap();
        assert!(
            parsed >= Duration::from_secs(3) && parsed <= Duration::from_secs(4),
            "got {parsed:?}"
        );
    }

    #[test]
    fn a_retry_info_without_the_type_marker_is_ignored() {
        let body = r#"{"error":{"details":[{"retryDelay":"30s"}]}}"#;
        assert_eq!(parse_reset_hint(&HeaderMap::new(), body), None);
    }

    #[test]
    fn unstructured_reset_fields_are_read() {
        let body = r#"{"error":{"retryDelay":"45s"}}"#;
        assert_eq!(
            parse_reset_hint(&HeaderMap::new(), body),
            Some(Duration::from_secs(45))
        );
        let body = r#"{"error":{"quotaResetDelay":"2m"}}"#;
        assert_eq!(
            parse_reset_hint(&HeaderMap::new(), body),
            Some(Duration::from_secs(120))
        );
    }

    #[test]
    fn a_reset_mentioned_in_prose_is_found() {
        let body = r#"{"error":{"message":"Rate limited. Please retry after 45 seconds."}}"#;
        assert_eq!(
            parse_reset_hint(&HeaderMap::new(), body),
            Some(Duration::from_secs(45))
        );
    }

    #[test]
    fn no_hint_yields_nothing() {
        assert_eq!(parse_reset_hint(&HeaderMap::new(), "{}"), None);
        assert_eq!(parse_reset_hint(&HeaderMap::new(), "not json"), None);
    }

    #[test]
    fn duration_strings_of_each_shape_parse() {
        assert_eq!(parse_duration_string("30s"), Some(Duration::from_secs(30)));
        let millis = parse_duration_string("754.43ms").unwrap();
        assert!(
            millis >= Duration::from_micros(754_000) && millis <= Duration::from_micros(754_500),
            "fractional milliseconds must be preserved, got {millis:?}"
        );
        assert_eq!(parse_duration_string("2m"), Some(Duration::from_secs(120)));
        assert_eq!(parse_duration_string("1h"), Some(Duration::from_secs(3600)));
        assert_eq!(parse_duration_string("45"), Some(Duration::from_secs(45)));
        assert_eq!(
            parse_duration_string("1h23m45s"),
            Some(Duration::from_secs(3600 + 23 * 60 + 45))
        );
    }

    #[test]
    fn milliseconds_are_not_read_as_minutes() {
        // The `ms` check has to run before the `m` check or this yields 754
        // minutes instead of 754 milliseconds.
        let parsed = parse_duration_string("754ms").unwrap();
        assert!(parsed < Duration::from_secs(2), "got {parsed:?}");
    }

    #[test]
    fn malformed_duration_strings_are_rejected() {
        assert_eq!(parse_duration_string(""), None);
        assert_eq!(parse_duration_string("abc"), None);
        assert_eq!(parse_duration_string("10x"), None);
        assert_eq!(parse_duration_string("0s"), None);
    }

    #[test]
    fn a_nested_error_envelope_is_unwrapped() {
        // The live shape: the Anthropic-shaped envelope arrives inside the
        // Google error's message field.
        let body = r#"{"error":{"message":"{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"Thinking may not be enabled when tool_choice forces tool use.\"}}"}}"#;
        assert_eq!(
            first_message(body).as_deref(),
            Some("Thinking may not be enabled when tool_choice forces tool use.")
        );
    }

    #[test]
    fn a_plain_message_is_left_alone() {
        let body = r#"{"error":{"message":"Invalid JSON payload"}}"#;
        assert_eq!(first_message(body).as_deref(), Some("Invalid JSON payload"));
    }

    #[test]
    fn a_message_that_is_json_without_an_inner_message_is_kept() {
        let body = r#"{"error":{"message":"{\"unexpected\":true}"}}"#;
        assert!(first_message(body).unwrap().contains("unexpected"));
    }

    #[test]
    fn token_matching_respects_boundaries() {
        assert!(contains_token("has account_ineligible now", "account_ineligible"));
        assert!(contains_token("account_ineligible", "account_ineligible"));
        assert!(!contains_token("not_account_ineligible", "account_ineligible"));
        assert!(!contains_token("account_ineligible_x", "account_ineligible"));
    }

    #[test]
    fn a_url_is_taken_up_to_its_terminator() {
        assert_eq!(
            take_url("https://example.com/a?b=c\"}"),
            "https://example.com/a?b=c"
        );
        assert_eq!(take_url("https://example.com/x rest"), "https://example.com/x");
    }
}
