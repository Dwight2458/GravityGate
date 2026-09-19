//! The retry state machine.
//!
//! Every request walks two nested loops: endpoints within an account, and
//! accounts within the pool. The reference implementations express this as a
//! `for` loop with `continue`, a decremented counter, and a `switch` that falls
//! through — which works but cannot be read, let alone tested.
//!
//! Here the decision is a pure function, [`decide`], from a classified failure
//! plus the attempt history to a [`Step`]. The async driver that executes steps
//! is then thin enough to have no logic in it worth testing. Everything that is
//! easy to get wrong — which failure justifies spending another account, when to
//! prefer another endpoint, when to give up — lives in the pure half.
//!
//! The central distinction, because getting it wrong is expensive in both
//! directions:
//!
//! - **Capacity exhaustion is the model's problem, not the account's.** Retrying
//!   the same account shortly is correct; rotating burns another account for
//!   nothing and leaves less headroom for a genuinely account-specific failure.
//! - **Quota exhaustion is the account's problem.** Retrying the same account
//!   cannot succeed, so rotating is the only useful move.

use std::time::Duration;

use crate::accounts::ratelimit::{AccountProblem, RateLimitReason, UpstreamFailure, backoff_for};
use crate::config::RoutingConfig;

/// What to do after a failed attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Try the next endpoint for the same account, immediately.
    NextEndpoint,
    /// Wait, then repeat against the same account and endpoint.
    RetrySameEndpoint { delay: Duration },
    /// Move to another account. `delay` is a pause before dispatching there.
    RotateAccount { delay: Duration },
    /// Stop and report the failure.
    Fail(RetryError),
}

/// Why the loop gave up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryError {
    /// The request was rejected as malformed. It will be malformed everywhere.
    BadRequest { message: String },

    /// The credential is no longer accepted, so no account can serve this.
    AuthInvalid { message: String },

    /// Every account was tried and none worked.
    AccountsExhausted { tried: u32, last: String },

    /// Every account is rate-limited and the wait exceeds the configured budget.
    PoolExhausted { earliest_in: Option<Duration> },
}

impl std::fmt::Display for RetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadRequest { message } => write!(f, "upstream rejected the request: {message}"),
            Self::AuthInvalid { message } => write!(f, "credential rejected: {message}"),
            Self::AccountsExhausted { tried, last } => {
                write!(f, "tried {tried} account(s); last failure: {last}")
            }
            Self::PoolExhausted { earliest_in } => match earliest_in {
                Some(wait) => write!(
                    f,
                    "every account is rate limited; the earliest clears in {}s",
                    wait.as_secs()
                ),
                None => write!(f, "every account is rate limited"),
            },
        }
    }
}

impl std::error::Error for RetryError {}

/// How far the request has got, for the next decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttemptState {
    /// Endpoints configured for this request.
    pub endpoint_count: usize,
    /// Index of the endpoint that just failed.
    pub endpoint_index: usize,
    /// Accounts already dispatched to, this request.
    pub accounts_tried: u32,
    /// Consecutive same-account retries for the current account.
    pub same_account_retries: u32,
}

impl AttemptState {
    pub fn new(endpoint_count: usize) -> Self {
        Self {
            endpoint_count: endpoint_count.max(1),
            endpoint_index: 0,
            accounts_tried: 1,
            same_account_retries: 0,
        }
    }

    /// Whether an endpoint remains untried for this account.
    pub fn has_more_endpoints(&self) -> bool {
        self.endpoint_index + 1 < self.endpoint_count
    }

    /// Advance to the next endpoint, resetting the same-account budget because a
    /// different endpoint is a genuinely different attempt.
    pub fn advance_endpoint(&mut self) {
        self.endpoint_index = (self.endpoint_index + 1) % self.endpoint_count;
        self.same_account_retries = 0;
    }

    /// Note that a retry against the same account and endpoint is happening.
    pub fn note_same_account_retry(&mut self) {
        self.same_account_retries += 1;
    }

    /// Reset for a new account.
    pub fn begin_next_account(&mut self) {
        self.accounts_tried += 1;
        self.endpoint_index = 0;
        self.same_account_retries = 0;
    }
}

/// Decide what to do after a failure.
///
/// `jitter` is a value in `0.0..=1.0` used to spread capacity retries; passing a
/// fixed value makes the decision deterministic under test.
pub fn decide(
    failure: &UpstreamFailure,
    state: &AttemptState,
    config: &RoutingConfig,
    jitter: f64,
) -> Step {
    // Neither of these can be fixed by trying something else.
    match failure {
        UpstreamFailure::BadRequest { message } => {
            return Step::Fail(RetryError::BadRequest {
                message: message.clone(),
            });
        }
        UpstreamFailure::Account(AccountProblem::AuthInvalid { reason }) => {
            return Step::Fail(RetryError::AuthInvalid {
                message: reason.clone(),
            });
        }
        _ => {}
    }

    let can_retry_here = state.same_account_retries < config.max_capacity_retries;

    match failure {
        // The model is busy. The account is fine, so the same account with the
        // same endpoint is the right retry; another endpoint only if this one has
        // exhausted its budget.
        UpstreamFailure::RateLimited {
            reason: RateLimitReason::ModelCapacityExhausted | RateLimitReason::ServerError,
            reset_in,
        } => {
            if can_retry_here {
                return Step::RetrySameEndpoint {
                    delay: backoff_for(
                        RateLimitReason::ModelCapacityExhausted,
                        state.same_account_retries,
                        *reset_in,
                        jitter,
                    ),
                };
            }
            if state.has_more_endpoints() {
                return Step::NextEndpoint;
            }
        }

        // An upstream or transport fault. Try another endpoint first: the
        // endpoint may be the problem, and the account has not been implicated.
        UpstreamFailure::Server { .. } | UpstreamFailure::Network => {
            if state.has_more_endpoints() {
                return Step::NextEndpoint;
            }
            if can_retry_here {
                let (reason, reset_in) = match failure {
                    UpstreamFailure::Server { .. } => (RateLimitReason::ServerError, None),
                    _ => (RateLimitReason::Unknown, None),
                };
                return Step::RetrySameEndpoint {
                    delay: backoff_for(reason, state.same_account_retries, reset_in, jitter),
                };
            }
        }

        // The endpoint or the project refused. Another endpoint is the first
        // thing to try; the account is not necessarily at fault.
        UpstreamFailure::Refused { .. } => {
            if state.has_more_endpoints() {
                return Step::NextEndpoint;
            }
        }

        // Quota, rate limiting, verification, ineligibility, bans: all account
        // scoped. Rotating is the only move that can help.
        UpstreamFailure::RateLimited { .. } | UpstreamFailure::Account(_) => {}

        // Handled above.
        UpstreamFailure::BadRequest { .. } => unreachable!("returned early"),
    }

    if state.accounts_tried >= config.max_account_attempts {
        return Step::Fail(RetryError::AccountsExhausted {
            tried: state.accounts_tried,
            last: describe(failure),
        });
    }

    // A small pause before moving accounts. Without it a pool-wide failure
    // produces an immediate burst against the next account.
    Step::RotateAccount {
        delay: Duration::from_millis(250),
    }
}

/// A short description of a failure, for error messages.
fn describe(failure: &UpstreamFailure) -> String {
    match failure {
        UpstreamFailure::RateLimited { reason, reset_in } => match reset_in {
            Some(reset) => format!(
                "rate limited ({reason:?}), clears in {}s",
                reset.as_secs()
            ),
            None => format!("rate limited ({reason:?})"),
        },
        UpstreamFailure::Account(problem) => format!("account problem: {}", problem.label()),
        UpstreamFailure::Server { status } => format!("upstream error {status}"),
        UpstreamFailure::Network => "network error".into(),
        UpstreamFailure::BadRequest { message } => format!("bad request: {message}"),
        UpstreamFailure::Refused { status, message } => format!("refused ({status}): {message}"),
    }
}

/// Whether the pool is worth waiting for.
///
/// Returns the wait if it fits inside the configured budget, or `None` when
/// waiting would exceed it — in which case the client is better served by an
/// immediate error than by a request that hangs for minutes.
pub fn pool_wait(
    earliest_in: Option<Duration>,
    config: &RoutingConfig,
) -> Option<Duration> {
    let earliest = earliest_in?;
    let budget = Duration::from_secs(config.max_wait_before_error_secs);
    (earliest <= budget).then_some(earliest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RoutingConfig {
        RoutingConfig::default()
    }

    fn state() -> AttemptState {
        // One endpoint by default, so endpoint fallback does not mask the
        // account-level decision being tested.
        AttemptState::new(1)
    }

    fn rate_limited(reason: RateLimitReason) -> UpstreamFailure {
        UpstreamFailure::RateLimited {
            reason,
            reset_in: None,
        }
    }

    // -- terminal failures --------------------------------------------------

    #[test]
    fn a_bad_request_fails_immediately() {
        let failure = UpstreamFailure::BadRequest {
            message: "Invalid JSON payload".into(),
        };
        let step = decide(&failure, &state(), &config(), 0.5);
        match step {
            Step::Fail(RetryError::BadRequest { message }) => {
                assert!(message.contains("Invalid JSON"))
            }
            other => panic!("expected an immediate failure, got {other:?}"),
        }
    }

    #[test]
    fn an_invalid_credential_fails_immediately() {
        let failure = UpstreamFailure::Account(AccountProblem::AuthInvalid {
            reason: "revoked".into(),
        });
        assert!(matches!(
            decide(&failure, &state(), &config(), 0.5),
            Step::Fail(RetryError::AuthInvalid { .. })
        ));
    }

    // -- capacity -----------------------------------------------------------

    #[test]
    fn capacity_exhaustion_retries_the_same_account() {
        // The model is busy; the account is fine. Rotating would waste it.
        let step = decide(
            &rate_limited(RateLimitReason::ModelCapacityExhausted),
            &state(),
            &config(),
            0.5,
        );
        assert!(matches!(step, Step::RetrySameEndpoint { .. }), "got {step:?}");
    }

    #[test]
    fn capacity_retries_are_bounded() {
        let mut state = state();
        state.same_account_retries = config().max_capacity_retries;
        let step = decide(
            &rate_limited(RateLimitReason::ModelCapacityExhausted),
            &state,
            &config(),
            0.5,
        );
        // Budget spent with no other endpoint: the account has to go.
        assert!(matches!(step, Step::RotateAccount { .. }), "got {step:?}");
    }

    #[test]
    fn capacity_falls_back_to_another_endpoint_after_the_retry_budget() {
        let mut state = AttemptState::new(2);
        state.same_account_retries = config().max_capacity_retries;
        let step = decide(
            &rate_limited(RateLimitReason::ModelCapacityExhausted),
            &state,
            &config(),
            0.5,
        );
        assert_eq!(step, Step::NextEndpoint);
    }

    #[test]
    fn capacity_retries_carry_a_backoff() {
        let step = decide(
            &rate_limited(RateLimitReason::ModelCapacityExhausted),
            &state(),
            &config(),
            0.5,
        );
        match step {
            Step::RetrySameEndpoint { delay } => {
                assert!(delay >= crate::accounts::ratelimit::MIN_BACKOFF)
            }
            other => panic!("expected a delay, got {other:?}"),
        }
    }

    // -- quota and rate limits ----------------------------------------------

    #[test]
    fn quota_exhaustion_rotates_the_account() {
        let step = decide(
            &rate_limited(RateLimitReason::QuotaExhausted),
            &state(),
            &config(),
            0.5,
        );
        assert!(matches!(step, Step::RotateAccount { .. }), "got {step:?}");
    }

    #[test]
    fn a_short_rate_limit_also_rotates() {
        // Retrying the same account cannot help while it is limited.
        let step = decide(
            &rate_limited(RateLimitReason::RateLimitExceeded),
            &state(),
            &config(),
            0.5,
        );
        assert!(matches!(step, Step::RotateAccount { .. }), "got {step:?}");
    }

    #[test]
    fn rotation_stops_at_the_attempt_budget() {
        let mut state = state();
        state.accounts_tried = config().max_account_attempts;
        let step = decide(
            &rate_limited(RateLimitReason::QuotaExhausted),
            &state,
            &config(),
            0.5,
        );
        match step {
            Step::Fail(RetryError::AccountsExhausted { tried, .. }) => {
                assert_eq!(tried, config().max_account_attempts)
            }
            other => panic!("expected exhaustion, got {other:?}"),
        }
    }

    // -- account problems ---------------------------------------------------

    #[test]
    fn a_verification_demand_rotates_rather_than_failing() {
        // The account is held, but the others may be fine.
        let failure = UpstreamFailure::Account(AccountProblem::VerificationRequired { url: None });
        assert!(matches!(
            decide(&failure, &state(), &config(), 0.5),
            Step::RotateAccount { .. }
        ));
    }

    #[test]
    fn a_ban_rotates_away() {
        let failure = UpstreamFailure::Account(AccountProblem::Banned {
            reason: "terms".into(),
        });
        assert!(matches!(
            decide(&failure, &state(), &config(), 0.5),
            Step::RotateAccount { .. }
        ));
    }

    // -- endpoint fallback --------------------------------------------------

    #[test]
    fn a_refused_endpoint_falls_back_to_another_endpoint_first() {
        let failure = UpstreamFailure::Refused {
            status: 404,
            message: "not found".into(),
        };
        let state = AttemptState::new(2);
        assert_eq!(decide(&failure, &state, &config(), 0.5), Step::NextEndpoint);
    }

    #[test]
    fn a_refusal_cascades_to_another_account_once_endpoints_run_out() {
        let failure = UpstreamFailure::Refused {
            status: 403,
            message: "permission denied".into(),
        };
        let mut state = AttemptState::new(2);
        state.endpoint_index = 1;
        assert!(matches!(
            decide(&failure, &state, &config(), 0.5),
            Step::RotateAccount { .. }
        ));
    }

    #[test]
    fn a_server_error_tries_another_endpoint_before_retrying() {
        let failure = UpstreamFailure::Server { status: 502 };
        let state = AttemptState::new(2);
        assert_eq!(decide(&failure, &state, &config(), 0.5), Step::NextEndpoint);
    }

    #[test]
    fn a_network_error_with_one_endpoint_retries_in_place() {
        let state = AttemptState::new(1);
        assert!(matches!(
            decide(&UpstreamFailure::Network, &state, &config(), 0.5),
            Step::RetrySameEndpoint { .. }
        ));
    }

    #[test]
    fn a_refusal_does_not_consume_the_capacity_budget() {
        // Endpoint fallback and same-account retries are different budgets.
        let mut state = AttemptState::new(2);
        state.same_account_retries = 0;
        decide(
            &UpstreamFailure::Refused {
                status: 404,
                message: String::new(),
            },
            &state,
            &config(),
            0.5,
        );
        assert_eq!(state.same_account_retries, 0);
    }

    // -- attempt bookkeeping -----------------------------------------------

    #[test]
    fn advancing_an_endpoint_resets_the_same_account_budget() {
        let mut state = AttemptState::new(3);
        state.note_same_account_retry();
        state.note_same_account_retry();
        state.advance_endpoint();

        assert_eq!(state.endpoint_index, 1);
        assert_eq!(
            state.same_account_retries, 0,
            "a different endpoint is a different attempt"
        );
    }

    #[test]
    fn the_endpoint_index_wraps() {
        let mut state = AttemptState::new(2);
        state.advance_endpoint();
        assert_eq!(state.endpoint_index, 1);
        state.advance_endpoint();
        assert_eq!(state.endpoint_index, 0);
    }

    #[test]
    fn beginning_a_new_account_resets_the_per_account_counter() {
        let mut state = AttemptState::new(2);
        state.note_same_account_retry();
        state.advance_endpoint();
        state.begin_next_account();

        assert_eq!(state.accounts_tried, 2);
        assert_eq!(state.endpoint_index, 0);
        assert_eq!(state.same_account_retries, 0);
    }

    #[test]
    fn has_more_endpoints_reflects_the_index() {
        let mut state = AttemptState::new(2);
        assert!(state.has_more_endpoints());
        state.endpoint_index = 1;
        assert!(!state.has_more_endpoints());
    }

    #[test]
    fn a_single_endpoint_never_reports_more() {
        let state = AttemptState::new(1);
        assert!(!state.has_more_endpoints());
    }

    #[test]
    fn zero_endpoints_is_normalised_to_one() {
        // Guards against a config that would otherwise make the loop spin.
        let state = AttemptState::new(0);
        assert_eq!(state.endpoint_count, 1);
        assert!(!state.has_more_endpoints());
    }

    // -- pool waiting -------------------------------------------------------

    #[test]
    fn a_short_pool_wait_is_accepted() {
        let config = config();
        let wait = Duration::from_secs(config.max_wait_before_error_secs - 1);
        assert_eq!(pool_wait(Some(wait), &config), Some(wait));
    }

    #[test]
    fn a_long_pool_wait_is_refused() {
        // Better to fail now than to hang a client for minutes.
        let config = config();
        let wait = Duration::from_secs(config.max_wait_before_error_secs + 1);
        assert_eq!(pool_wait(Some(wait), &config), None);
    }

    #[test]
    fn a_wait_exactly_at_the_budget_is_accepted() {
        let config = config();
        let wait = Duration::from_secs(config.max_wait_before_error_secs);
        assert_eq!(pool_wait(Some(wait), &config), Some(wait));
    }

    #[test]
    fn no_reset_means_no_waiting() {
        assert_eq!(pool_wait(None, &config()), None);
    }

    // -- messages -----------------------------------------------------------

    #[test]
    fn errors_describe_themselves_usefully() {
        let exhausted = RetryError::PoolExhausted {
            earliest_in: Some(Duration::from_secs(90)),
        };
        assert!(exhausted.to_string().contains("90s"));

        let bad = RetryError::BadRequest {
            message: "Invalid JSON".into(),
        };
        assert!(bad.to_string().contains("Invalid JSON"));

        let tried = RetryError::AccountsExhausted {
            tried: 5,
            last: "rate limited".into(),
        };
        assert!(tried.to_string().contains('5'));
    }

    #[test]
    fn failure_descriptions_name_the_reason() {
        assert!(
            describe(&rate_limited(RateLimitReason::QuotaExhausted)).contains("QuotaExhausted")
        );
        assert!(describe(&UpstreamFailure::Network).contains("network"));
        assert!(describe(&UpstreamFailure::Server { status: 502 }).contains("502"));
    }
}
