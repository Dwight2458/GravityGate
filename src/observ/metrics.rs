//! Prometheus metrics.
//!
//! Metric names and labels are a public interface: once a dashboard or alert
//! depends on them, renaming one is a breaking change. They are declared here in
//! one place so that stays visible, and recording goes through typed functions
//! rather than raw macro calls scattered across the request path.
//!
//! Cardinality is the thing to watch. Labels are limited to values with a small
//! fixed set — model, outcome, account — and never to anything client-supplied
//! such as a request id or a user field, which would grow the series count
//! without bound.

use std::time::Duration;

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Requests handled, by the model the client asked for and how they ended.
pub const REQUESTS_TOTAL: &str = "gravitygate_requests_total";
/// End-to-end duration of a client request.
pub const REQUEST_DURATION: &str = "gravitygate_request_duration_seconds";
/// Upstream attempts a request needed, including the successful one.
pub const UPSTREAM_ATTEMPTS: &str = "gravitygate_upstream_attempts";
/// Times an account was selected to serve a request.
pub const ACCOUNT_SELECTED: &str = "gravitygate_account_selected_total";
/// Rate limits and other account problems, by kind.
pub const ADVERSE_EVENTS: &str = "gravitygate_adverse_events_total";
/// Tokens reported by the upstream, by kind.
pub const TOKENS: &str = "gravitygate_tokens_total";
/// Accounts currently in each condition.
pub const ACCOUNTS: &str = "gravitygate_accounts";
/// Thinking signatures currently held.
pub const SIGNATURES_CACHED: &str = "gravitygate_signatures_cached";

/// Registers and renders metrics.
///
/// The handle renders the Prometheus text exposition format on demand; nothing
/// is pushed anywhere.
pub struct Metrics {
    handle: PrometheusHandle,
}

impl std::fmt::Debug for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The handle owns a registry of atomics with no useful rendering.
        f.debug_struct("Metrics").finish_non_exhaustive()
    }
}

impl Metrics {
    /// Install the global recorder.
    ///
    /// Fails if a recorder is already installed, which happens when two engines
    /// exist in one process — a test scenario, not a production one.
    pub fn install() -> Result<Self, metrics_exporter_prometheus::BuildError> {
        let handle = PrometheusBuilder::new().install_recorder()?;
        Self::describe();
        Ok(Self { handle })
    }

    /// Declare every metric's type and units.
    ///
    /// Prometheus requires the `# TYPE` and `# HELP` lines, and describing them
    /// centrally keeps the help text from drifting away from the metric.
    fn describe() {
        describe_counter!(
            REQUESTS_TOTAL,
            "Chat completion requests handled, by requested model and outcome"
        );
        describe_histogram!(
            REQUEST_DURATION,
            "End-to-end latency of a client request, in seconds"
        );
        describe_histogram!(
            UPSTREAM_ATTEMPTS,
            "Upstream attempts per request, including the successful one"
        );
        describe_counter!(
            ACCOUNT_SELECTED,
            "Requests dispatched per account, by credential id"
        );
        describe_counter!(
            ADVERSE_EVENTS,
            "Rate limits and account problems observed, by kind"
        );
        describe_counter!(
            TOKENS,
            "Tokens reported by the upstream, by kind (prompt, completion, cached, reasoning)"
        );
        describe_gauge!(
            ACCOUNTS,
            "Accounts currently in each condition (ready, limited, verify, ineligible, disabled)"
        );
        describe_gauge!(SIGNATURES_CACHED, "Thinking signatures currently cached");
    }

    /// Render the current values in the Prometheus text format.
    pub fn render(&self) -> String {
        self.handle.render()
    }
}

/// How a request ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    /// The client sent something we could not use.
    ClientError,
    /// The upstream refused in a way that is not the account's fault.
    UpstreamError,
    /// No account could serve it.
    Unavailable,
}

impl Outcome {
    fn label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::ClientError => "client_error",
            Self::UpstreamError => "upstream_error",
            Self::Unavailable => "unavailable",
        }
    }

    /// Which outcome a status code represents.
    pub fn of(status: u16) -> Self {
        match status {
            200..=299 => Self::Ok,
            500..=599 => Self::UpstreamError,
            _ => Self::ClientError,
        }
    }
}

/// Record one finished request.
pub fn record_request(model: &str, outcome: Outcome, elapsed: Duration) {
    counter!(REQUESTS_TOTAL, "model" => model.to_string(), "outcome" => outcome.label()).increment(1);
    histogram!(REQUEST_DURATION, "model" => model.to_string())
        .record(elapsed.as_secs_f64());
}

/// Record how many upstream attempts a successful request needed.
pub fn record_attempts(model: &str, attempts: u32) {
    histogram!(UPSTREAM_ATTEMPTS, "model" => model.to_string()).record(f64::from(attempts));
}

/// Record that an account served a request.
pub fn record_account_selected(credential_id: &str) {
    counter!(ACCOUNT_SELECTED, "account" => credential_id.to_string()).increment(1);
}

/// Record a rate limit or account problem.
///
/// `kind` is a small fixed vocabulary — quota, rate, capacity, verification,
/// ineligibility, ban, network — never a free-form message.
pub fn record_adverse(kind: &str) {
    counter!(ADVERSE_EVENTS, "kind" => kind.to_string()).increment(1);
}

/// Record token usage.
pub fn record_tokens(model: &str, kind: &str, count: i64) {
    if count > 0 {
        counter!(TOKENS, "model" => model.to_string(), "kind" => kind.to_string())
            .increment(count as u64);
    }
}

/// Publish the account condition counts.
///
/// A gauge of counts rather than one series per account: an operator watching a
/// dashboard cares that four accounts are rate limited, and per-account detail
/// belongs on `/health` where the reset timers and URLs are.
pub fn set_account_counts(counts: &[(&str, u64)]) {
    for (condition, count) in counts {
        gauge!(ACCOUNTS, "condition" => condition.to_string()).set(*count as f64);
    }
}

/// Publish the signature cache size.
pub fn set_signatures_cached(count: usize) {
    gauge!(SIGNATURES_CACHED).set(count as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcomes_are_classified_by_status() {
        assert_eq!(Outcome::of(200), Outcome::Ok);
        assert_eq!(Outcome::of(204), Outcome::Ok);
        assert_eq!(Outcome::of(400), Outcome::ClientError);
        assert_eq!(Outcome::of(401), Outcome::ClientError);
        assert_eq!(Outcome::of(429), Outcome::ClientError);
        assert_eq!(Outcome::of(500), Outcome::UpstreamError);
        assert_eq!(Outcome::of(503), Outcome::UpstreamError);
    }

    #[test]
    fn outcome_labels_are_stable() {
        // These strings are a public interface; a dashboard breaks if they move.
        assert_eq!(Outcome::Ok.label(), "ok");
        assert_eq!(Outcome::ClientError.label(), "client_error");
        assert_eq!(Outcome::UpstreamError.label(), "upstream_error");
        assert_eq!(Outcome::Unavailable.label(), "unavailable");
    }

    #[test]
    fn metric_names_are_prefixed_and_typed() {
        // A metric without the prefix is easy to mistake for a dependency's.
        for name in [
            REQUESTS_TOTAL,
            REQUEST_DURATION,
            UPSTREAM_ATTEMPTS,
            ACCOUNT_SELECTED,
            ADVERSE_EVENTS,
            TOKENS,
            ACCOUNTS,
            SIGNATURES_CACHED,
        ] {
            assert!(name.starts_with("gravitygate_"), "{name} is unprefixed");
        }
        // Prometheus convention: counters end in `_total`, durations in `_seconds`.
        assert!(REQUESTS_TOTAL.ends_with("_total"));
        assert!(ACCOUNT_SELECTED.ends_with("_total"));
        assert!(ADVERSE_EVENTS.ends_with("_total"));
        assert!(TOKENS.ends_with("_total"));
        assert!(REQUEST_DURATION.ends_with("_seconds"));
    }
}
