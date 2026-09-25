//! Account selection.
//!
//! Two pieces of runtime state per account, neither of which belongs on disk:
//! a token bucket that limits how fast we spend an account's goodwill, and a
//! health score that decays on failure and recovers with rest. Both are keyed by
//! credential id rather than by position, so removing an account from the pool
//! cannot silently transfer another account's history to whatever slides into
//! that index.
//!
//! The scoring constants are taken from `antigravity-auth`, which arrived at
//! them by running this in production. They are not arbitrary:
//!
//! - The token bucket is the *client-side* throttle. Its job is to avoid
//!   reaching the server's 429 at all, which is strictly better than handling it.
//! - The stickiness bonus exists to protect the prompt cache. Switching accounts
//!   mid-conversation costs more than the load-balancing it buys, so a rival has
//!   to beat the incumbent by a clear margin before a switch happens. Without
//!   this the score jitters and accounts swap on nearly every request.
//! - Health recovery is what makes a benched account usable again. Without it a
//!   single bad minute would retire an account permanently.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::config::Strategy;

/// Token bucket ceiling, per account.
///
/// Deliberately small: the bucket's whole job is to stop a burst before the
/// upstream's rate limiter sees it, and a large ceiling defeats that.
pub const DEFAULT_TOKEN_MAX: f64 = 10.0;

/// Tokens returned per minute. Six per minute against a ceiling of ten means a
/// sustained rate of one request every ten seconds per account.
pub const DEFAULT_TOKEN_REFILL_PER_MIN: f64 = 6.0;

pub const DEFAULT_HEALTH_INITIAL: f64 = 70.0;
pub const DEFAULT_HEALTH_SUCCESS_REWARD: f64 = 1.0;
pub const DEFAULT_HEALTH_RATE_LIMIT_PENALTY: f64 = -10.0;
pub const DEFAULT_HEALTH_FAILURE_PENALTY: f64 = -20.0;
pub const DEFAULT_HEALTH_RECOVERY_PER_HOUR: f64 = 2.0;
pub const DEFAULT_HEALTH_MIN_USABLE: f64 = 50.0;
pub const DEFAULT_HEALTH_MAX: f64 = 100.0;

/// Added to the incumbent's score so it is not displaced by noise.
pub const DEFAULT_STICKINESS_BONUS: f64 = 150.0;

/// How far a rival must beat the incumbent's *base* score to justify switching.
pub const DEFAULT_SWITCH_THRESHOLD: f64 = 100.0;

/// The freshness component saturates at an hour; beyond that an account is
/// rested enough and idleness stops being a reason to prefer it.
const FRESHNESS_CAP_SECS: f64 = 3600.0;

/// Scoring weights, mirroring the reference implementation.
const HEALTH_WEIGHT: f64 = 2.0;
const TOKEN_WEIGHT: f64 = 5.0;
const FRESHNESS_WEIGHT: f64 = 0.1;

/// Tunable scoring constants.
#[derive(Debug, Clone, Copy)]
pub struct ScoringConfig {
    pub token_max: f64,
    pub token_refill_per_min: f64,
    pub health_initial: f64,
    pub health_success_reward: f64,
    pub health_rate_limit_penalty: f64,
    pub health_failure_penalty: f64,
    pub health_recovery_per_hour: f64,
    pub health_min_usable: f64,
    pub health_max: f64,
    pub stickiness_bonus: f64,
    pub switch_threshold: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            token_max: DEFAULT_TOKEN_MAX,
            token_refill_per_min: DEFAULT_TOKEN_REFILL_PER_MIN,
            health_initial: DEFAULT_HEALTH_INITIAL,
            health_success_reward: DEFAULT_HEALTH_SUCCESS_REWARD,
            health_rate_limit_penalty: DEFAULT_HEALTH_RATE_LIMIT_PENALTY,
            health_failure_penalty: DEFAULT_HEALTH_FAILURE_PENALTY,
            health_recovery_per_hour: DEFAULT_HEALTH_RECOVERY_PER_HOUR,
            health_min_usable: DEFAULT_HEALTH_MIN_USABLE,
            health_max: DEFAULT_HEALTH_MAX,
            stickiness_bonus: DEFAULT_STICKINESS_BONUS,
            switch_threshold: DEFAULT_SWITCH_THRESHOLD,
        }
    }
}

/// What the router needs to know about one account.
///
/// Deliberately a plain snapshot: the router never reads the account store, so
/// selection can be tested without one.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Position in the pool, which is what the caller acts on.
    pub index: usize,
    pub credential_id: String,
    /// Last use, in epoch milliseconds.
    pub last_used: i64,
    /// Whether the account's *persisted* state permits dispatch — enabled, not
    /// under a hold, not cooling down, not rate-limited.
    pub is_available: bool,
    /// When the account's earliest limit clears, for the all-limited case.
    pub next_reset_in: Option<std::time::Duration>,
}

/// The outcome of a selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// Dispatch to this account.
    Chosen(usize),
    /// Every candidate is limited. The earliest clearance is this far away, and
    /// waiting that long is the only way forward.
    AllLimited { earliest_in: Option<std::time::Duration> },
    /// Nothing can serve this request: every account is disabled, under a hold,
    /// or too unhealthy. No amount of waiting fixes it.
    NoCandidate,
}

/// Resolved per-account scores, useful for `/health` and for tests.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub credential_id: String,
    pub health: f64,
    pub tokens: f64,
    pub total: f64,
    pub is_available: bool,
}

#[derive(Debug, Clone, Copy)]
struct BucketState {
    tokens: f64,
    updated_ms: i64,
}

#[derive(Debug, Clone, Copy)]
struct HealthState {
    score: f64,
    updated_ms: i64,
    consecutive_failures: u32,
}

#[derive(Debug, Default)]
struct Inner {
    buckets: HashMap<String, BucketState>,
    health: HashMap<String, HealthState>,
    /// Sticky cursor per routing family, held as a credential id so pool edits
    /// cannot shift it onto a different account.
    cursors: HashMap<String, String>,
}

/// Selects accounts and tracks their runtime health.
#[derive(Debug)]
pub struct AccountRouter {
    config: ScoringConfig,
    strategy: Strategy,
    inner: Mutex<Inner>,
}

impl AccountRouter {
    pub fn new(strategy: Strategy, config: ScoringConfig) -> Self {
        Self {
            config,
            strategy,
            inner: Mutex::new(Inner::default()),
        }
    }

    pub fn strategy(&self) -> Strategy {
        self.strategy
    }

    /// The account currently preferred for a family, if any.
    pub fn current(&self, family: &str) -> Option<String> {
        let inner = self.inner.lock().expect("router poisoned");
        inner.cursors.get(family).cloned()
    }

    /// Record which account was actually used, so the next request prefers it.
    pub fn set_current(&self, family: &str, credential_id: &str) {
        let mut inner = self.inner.lock().expect("router poisoned");
        inner
            .cursors
            .insert(family.to_string(), credential_id.to_string());
    }

    /// Choose an account.
    ///
    /// `now_ms` is passed in rather than read from the clock so that selection
    /// is deterministic under test.
    pub fn select(&self, candidates: &[Candidate], family: &str, now_ms: i64) -> Selection {
        if candidates.is_empty() {
            return Selection::NoCandidate;
        }

        // One lock for the whole selection. Taking it per candidate would be
        // correct but pointlessly contended, and this runs on every request.
        let inner = self.inner.lock().expect("router poisoned");

        // An account out of tokens counts as unavailable: the bucket is the
        // client-side throttle whose whole job is to avoid reaching the
        // server's rate limit at all.
        let has_tokens = |candidate: &Candidate| {
            self.tokens_of(&inner, &candidate.credential_id, now_ms) >= 1.0
        };

        let available: Vec<&Candidate> = candidates
            .iter()
            .filter(|candidate| candidate.is_available && has_tokens(candidate))
            .collect();

        if available.is_empty() {
            let earliest = candidates
                .iter()
                .filter_map(|candidate| candidate.next_reset_in)
                .min();

            if let Some(earliest) = earliest {
                return Selection::AllLimited {
                    earliest_in: Some(earliest),
                };
            }

            // Nothing is rate limited, so the block is either a token deficit —
            // which refills, and is therefore a wait — or an account state that
            // no amount of waiting fixes.
            if candidates.iter().any(|candidate| candidate.is_available) {
                return Selection::AllLimited {
                    earliest_in: self.time_until_a_token(&inner, candidates, now_ms),
                };
            }
            return Selection::NoCandidate;
        }

        let current = inner.cursors.get(family).cloned();
        let scored = self.score_all(&inner, &available, now_ms);

        let chosen = match self.strategy {
            Strategy::Hybrid => self.select_hybrid(&scored, current.as_deref()),
            Strategy::Sticky => self.select_sticky(&scored, current.as_deref()),
            Strategy::RoundRobin => self.select_round_robin(&scored, candidates, current.as_deref()),
            Strategy::LeastRecentlyUsed => self.select_lru(&scored, &available),
        };

        match chosen {
            Some(credential_id) => {
                match candidates
                    .iter()
                    .find(|candidate| candidate.credential_id == credential_id)
                {
                    Some(candidate) => Selection::Chosen(candidate.index),
                    // A credential id that is not in the pool means the snapshot
                    // changed underneath us; fall back to the first candidate
                    // rather than reporting no candidate.
                    None => Selection::Chosen(candidates[0].index),
                }
            }
            None => Selection::NoCandidate,
        }
    }

    /// How long until the least-depleted unavailable account can serve again.
    fn time_until_a_token(
        &self,
        inner: &Inner,
        candidates: &[Candidate],
        now_ms: i64,
    ) -> Option<std::time::Duration> {
        let refill = self.config.token_refill_per_min;
        if refill <= 0.0 {
            return None;
        }
        candidates
            .iter()
            .filter(|candidate| candidate.is_available)
            .map(|candidate| {
                let tokens = self.tokens_of(inner, &candidate.credential_id, now_ms);
                let deficit = (1.0 - tokens).max(0.0);
                std::time::Duration::from_secs_f64((deficit / refill) * 60.0)
            })
            .min()
    }

    /// Score every candidate, applying time-based recovery as it goes.
    fn score_all(&self, inner: &Inner, candidates: &[&Candidate], now_ms: i64) -> Vec<Score> {
        candidates
            .iter()
            .map(|candidate| {
                let health = self
                    .health_of(inner, &candidate.credential_id, now_ms)
                    .score;
                let tokens = self.tokens_of(inner, &candidate.credential_id, now_ms);

                let seconds_idle = ((now_ms - candidate.last_used).max(0) as f64) / 1000.0;
                let freshness = seconds_idle.min(FRESHNESS_CAP_SECS);

                let total = health * HEALTH_WEIGHT
                    + (tokens / self.config.token_max) * 100.0 * TOKEN_WEIGHT
                    + freshness * FRESHNESS_WEIGHT;

                Score {
                    credential_id: candidate.credential_id.clone(),
                    health,
                    tokens,
                    total,
                    is_available: candidate.is_available,
                }
            })
            .collect()
    }

    /// Health score with passive recovery applied.
    fn health_of(&self, inner: &Inner, credential_id: &str, now_ms: i64) -> HealthState {
        let Some(state) = inner.health.get(credential_id) else {
            return HealthState {
                score: self.config.health_initial,
                updated_ms: now_ms,
                consecutive_failures: 0,
            };
        };

        let hours = ((now_ms - state.updated_ms).max(0) as f64) / 3_600_000.0;
        let recovered = (hours * self.config.health_recovery_per_hour).floor();

        HealthState {
            score: (state.score + recovered).min(self.config.health_max),
            updated_ms: now_ms,
            consecutive_failures: state.consecutive_failures,
        }
    }

    /// Token balance with refill applied.
    fn tokens_of(&self, inner: &Inner, credential_id: &str, now_ms: i64) -> f64 {
        let Some(state) = inner.buckets.get(credential_id) else {
            return self.config.token_max;
        };
        let minutes = ((now_ms - state.updated_ms).max(0) as f64) / 60_000.0;
        (state.tokens + minutes * self.config.token_refill_per_min).min(self.config.token_max)
    }

    /// Hybrid: highest total score, with the incumbent protected.
    fn select_hybrid(&self, scored: &[Score], current: Option<&str>) -> Option<String> {
        let usable: Vec<&Score> = scored
            .iter()
            .filter(|score| score.health >= self.config.health_min_usable)
            .collect();
        if usable.is_empty() {
            return None;
        }

        let best = usable.iter().max_by(|a, b| {
            a.total
                .partial_cmp(&b.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

        // Stickiness is applied only as a tie-break against the incumbent, and
        // the comparison uses base scores on both sides so the bonus cannot
        // compare against itself.
        let Some(current) = current else {
            return Some(best.credential_id.clone());
        };
        let Some(incumbent) = usable
            .iter()
            .find(|score| score.credential_id == current)
        else {
            return Some(best.credential_id.clone());
        };

        if best.credential_id == incumbent.credential_id {
            return Some(incumbent.credential_id.clone());
        }

        let advantage = best.total - incumbent.total;
        if advantage < self.config.switch_threshold {
            return Some(incumbent.credential_id.clone());
        }
        Some(best.credential_id.clone())
    }

    /// Sticky: stay put until the incumbent becomes unusable.
    fn select_sticky(&self, scored: &[Score], current: Option<&str>) -> Option<String> {
        if let Some(current) = current
            && let Some(incumbent) = scored
                .iter()
                .find(|score| score.credential_id == current)
            && incumbent.health >= self.config.health_min_usable
        {
            return Some(incumbent.credential_id.clone());
        }
        self.select_hybrid(scored, None)
    }

    /// Round robin: advance one position from the cursor, wrapping.
    fn select_round_robin(
        &self,
        scored: &[Score],
        candidates: &[Candidate],
        current: Option<&str>,
    ) -> Option<String> {
        if scored.is_empty() {
            return None;
        }
        let available: Vec<&Score> = scored
            .iter()
            .filter(|score| score.health >= self.config.health_min_usable)
            .collect();
        if available.is_empty() {
            return None;
        }

        // The cursor names a position in the full pool; step to the next
        // candidate that is actually available.
        let start = current
            .and_then(|current| {
                candidates
                    .iter()
                    .position(|candidate| candidate.credential_id == current)
            })
            .map(|position| position + 1)
            .unwrap_or(0);

        for offset in 0..candidates.len() {
            let candidate = &candidates[(start + offset) % candidates.len()];
            if candidate.is_available
                && let Some(score) = available
                    .iter()
                    .find(|score| score.credential_id == candidate.credential_id)
            {
                return Some(score.credential_id.clone());
            }
        }
        None
    }

    /// Least recently used, with health as the tie-break.
    fn select_lru(&self, scored: &[Score], candidates: &[&Candidate]) -> Option<String> {
        scored
            .iter()
            .filter(|score| score.health >= self.config.health_min_usable)
            .filter_map(|score| {
                candidates
                    .iter()
                    .find(|candidate| candidate.credential_id == score.credential_id)
                    .map(|candidate| (score, *candidate))
            })
            // Oldest `last_used` first. On a tie the healthier account wins, so
            // the comparison is inverted: a higher health must sort as smaller.
            .min_by(|(a_score, a), (b_score, b)| {
                a.last_used.cmp(&b.last_used).then_with(|| {
                    b_score
                        .health
                        .partial_cmp(&a_score.health)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .map(|(score, _)| score.credential_id.clone())
    }

    /// Spend a token. Returns false when the account is out of budget.
    pub fn consume_token(&self, credential_id: &str, now_ms: i64) -> bool {
        let mut inner = self.inner.lock().expect("router poisoned");
        let available = self.tokens_of(&inner, credential_id, now_ms);
        if available < 1.0 {
            return false;
        }
        inner.buckets.insert(
            credential_id.to_string(),
            BucketState {
                tokens: available - 1.0,
                updated_ms: now_ms,
            },
        );
        true
    }

    /// Give a token back, for a request that never reached the upstream.
    pub fn refund_token(&self, credential_id: &str, now_ms: i64) {
        let mut inner = self.inner.lock().expect("router poisoned");
        let available = self.tokens_of(&inner, credential_id, now_ms);
        inner.buckets.insert(
            credential_id.to_string(),
            BucketState {
                tokens: (available + 1.0).min(self.config.token_max),
                updated_ms: now_ms,
            },
        );
    }

    pub fn record_success(&self, credential_id: &str, now_ms: i64) {
        self.adjust_health(credential_id, now_ms, |score, _| {
            score + self.config.health_success_reward
        }, true);
    }

    pub fn record_rate_limit(&self, credential_id: &str, now_ms: i64) {
        self.adjust_health(credential_id, now_ms, |score, _| {
            score + self.config.health_rate_limit_penalty
        }, false);
    }

    pub fn record_failure(&self, credential_id: &str, now_ms: i64) {
        self.adjust_health(credential_id, now_ms, |score, _| {
            score + self.config.health_failure_penalty
        }, false);
    }

    fn adjust_health(
        &self,
        credential_id: &str,
        now_ms: i64,
        apply: impl FnOnce(f64, u32) -> f64,
        reset_failures: bool,
    ) {
        let mut inner = self.inner.lock().expect("router poisoned");
        let current = self.health_of(&inner, credential_id, now_ms);
        let next = apply(current.score, current.consecutive_failures).clamp(0.0, self.config.health_max);

        inner.health.insert(
            credential_id.to_string(),
            HealthState {
                score: next,
                updated_ms: now_ms,
                consecutive_failures: if reset_failures {
                    0
                } else {
                    current.consecutive_failures + 1
                },
            },
        );
    }

    pub fn consecutive_failures(&self, credential_id: &str) -> u32 {
        let inner = self.inner.lock().expect("router poisoned");
        inner
            .health
            .get(credential_id)
            .map(|state| state.consecutive_failures)
            .unwrap_or(0)
    }

    pub fn health(&self, credential_id: &str, now_ms: i64) -> f64 {
        let inner = self.inner.lock().expect("router poisoned");
        self.health_of(&inner, credential_id, now_ms).score
    }

    pub fn tokens(&self, credential_id: &str, now_ms: i64) -> f64 {
        let inner = self.inner.lock().expect("router poisoned");
        self.tokens_of(&inner, credential_id, now_ms)
    }

    /// Drop tracking state for credentials no longer in the pool.
    pub fn retain(&self, live: &[String]) {
        let mut inner = self.inner.lock().expect("router poisoned");
        inner.buckets.retain(|key, _| live.contains(key));
        inner.health.retain(|key, _| live.contains(key));
        inner.cursors.retain(|_, value| live.contains(value));
    }
}

/// Convenience for building candidates from an account snapshot.
///
/// `already_tried` marks accounts this request has already dispatched to. The
/// caller filters them out before selecting, because the router has no notion of
/// a request in flight and should not acquire one: the alternative is threading
/// per-request state through every strategy.
pub fn candidate(index: usize, account: &crate::accounts::account::Account, now_ms: i64) -> Candidate {
    Candidate {
        index,
        credential_id: account.credential_id(),
        last_used: account.last_used,
        is_available: account.is_available(now_ms),
        next_reset_in: account
            .next_reset_at(now_ms)
            .map(|reset| std::time::Duration::from_millis((reset - now_ms).max(0) as u64)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const NOW: i64 = 1_800_000_000_000;

    fn router(strategy: Strategy) -> AccountRouter {
        AccountRouter::new(strategy, ScoringConfig::default())
    }

    fn candidate_at(id: &str, index: usize, last_used: i64, available: bool) -> Candidate {
        Candidate {
            index,
            credential_id: id.to_string(),
            last_used,
            is_available: available,
            next_reset_in: None,
        }
    }

    fn available(id: &str, index: usize) -> Candidate {
        candidate_at(id, index, NOW, true)
    }

    // -- basic selection ----------------------------------------------------

    #[test]
    fn an_empty_pool_has_no_candidate() {
        assert_eq!(router(Strategy::Hybrid).select(&[], "gemini", NOW), Selection::NoCandidate);
    }

    #[test]
    fn a_single_available_account_is_chosen() {
        let router = router(Strategy::Hybrid);
        let candidates = vec![available("a", 0)];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(0)
        );
    }

    #[test]
    fn the_index_returned_is_the_pool_position_not_the_score_rank() {
        let router = router(Strategy::Hybrid);
        // Only the second entry is available, and it must be reported as index 1.
        let candidates = vec![
            candidate_at("a", 0, NOW, false),
            candidate_at("b", 1, NOW, true),
        ];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(1)
        );
    }

    #[test]
    fn all_limited_reports_the_earliest_clearance() {
        let router = router(Strategy::Hybrid);
        let candidates = vec![
            Candidate {
                next_reset_in: Some(Duration::from_secs(300)),
                ..candidate_at("a", 0, NOW, false)
            },
            Candidate {
                next_reset_in: Some(Duration::from_secs(60)),
                ..candidate_at("b", 1, NOW, false)
            },
        ];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::AllLimited {
                earliest_in: Some(Duration::from_secs(60))
            }
        );
    }

    #[test]
    fn nothing_available_and_no_reset_is_not_a_waiting_problem() {
        // Disabled, banned, or held accounts have no reset time, and waiting
        // will not fix them.
        let router = router(Strategy::Hybrid);
        let candidates = vec![candidate_at("a", 0, NOW, false)];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::NoCandidate
        );
    }

    // -- token bucket -------------------------------------------------------

    #[test]
    fn a_fresh_account_has_a_full_bucket() {
        let router = router(Strategy::Hybrid);
        assert_eq!(router.tokens("a", NOW), DEFAULT_TOKEN_MAX);
    }

    #[test]
    fn consuming_drains_the_bucket() {
        let router = router(Strategy::Hybrid);
        for _ in 0..(DEFAULT_TOKEN_MAX as u32) {
            assert!(router.consume_token("a", NOW));
        }
        assert_eq!(router.tokens("a", NOW), 0.0);
        assert!(!router.consume_token("a", NOW), "an empty bucket must refuse");
    }

    #[test]
    fn the_bucket_refills_over_time() {
        let router = router(Strategy::Hybrid);
        for _ in 0..10 {
            router.consume_token("a", NOW);
        }
        assert_eq!(router.tokens("a", NOW), DEFAULT_TOKEN_MAX - 10.0);

        // One minute later, six tokens have returned.
        let later = NOW + 60_000;
        assert!((router.tokens("a", later) - (DEFAULT_TOKEN_MAX - 4.0)).abs() < 0.001);
    }

    #[test]
    fn the_bucket_never_overfills() {
        let router = router(Strategy::Hybrid);
        router.consume_token("a", NOW);
        let long_later = NOW + 24 * 60 * 60 * 1000;
        assert_eq!(router.tokens("a", long_later), DEFAULT_TOKEN_MAX);
    }

    #[test]
    fn a_refund_returns_a_token() {
        let router = router(Strategy::Hybrid);
        router.consume_token("a", NOW);
        assert_eq!(router.tokens("a", NOW), DEFAULT_TOKEN_MAX - 1.0);
        router.refund_token("a", NOW);
        assert_eq!(router.tokens("a", NOW), DEFAULT_TOKEN_MAX);
    }

    #[test]
    fn a_refund_cannot_exceed_the_ceiling() {
        let router = router(Strategy::Hybrid);
        router.refund_token("a", NOW);
        assert_eq!(router.tokens("a", NOW), DEFAULT_TOKEN_MAX);
    }

    // -- health -------------------------------------------------------------

    #[test]
    fn a_new_account_starts_at_the_initial_score() {
        let router = router(Strategy::Hybrid);
        assert_eq!(router.health("a", NOW), DEFAULT_HEALTH_INITIAL);
    }

    #[test]
    fn success_raises_and_failure_lowers_the_score() {
        let router = router(Strategy::Hybrid);
        router.record_success("a", NOW);
        assert_eq!(router.health("a", NOW), DEFAULT_HEALTH_INITIAL + 1.0);

        router.record_failure("a", NOW);
        assert_eq!(router.health("a", NOW), DEFAULT_HEALTH_INITIAL + 1.0 - 20.0);
    }

    #[test]
    fn a_rate_limit_penalises_less_than_a_failure() {
        // A rate limit is often not the account's fault; a failure usually is.
        let rate_limited = router(Strategy::Hybrid);
        rate_limited.record_rate_limit("a", NOW);

        let failed = router(Strategy::Hybrid);
        failed.record_failure("a", NOW);

        assert!(rate_limited.health("a", NOW) > failed.health("a", NOW));
    }

    #[test]
    fn health_recovers_with_rest() {
        let router = router(Strategy::Hybrid);
        router.record_failure("a", NOW);
        let damaged = router.health("a", NOW);

        // Two hours of rest at two points per hour.
        let later = NOW + 2 * 60 * 60 * 1000;
        assert_eq!(router.health("a", later), damaged + 4.0);
    }

    #[test]
    fn health_does_not_exceed_the_cap() {
        // A brand-new account is already at its initial score and does not
        // "recover" upward; only a damaged one does, and it stops at the cap.
        let router = router(Strategy::Hybrid);
        assert_eq!(router.health("a", NOW + 1000 * 60 * 60 * 1000), DEFAULT_HEALTH_INITIAL);

        for _ in 0..5 {
            router.record_failure("a", NOW);
        }
        let damaged = router.health("a", NOW);
        assert!(damaged < DEFAULT_HEALTH_MAX);

        let much_later = NOW + 1000 * 60 * 60 * 1000;
        assert_eq!(router.health("a", much_later), DEFAULT_HEALTH_MAX);
    }

    #[test]
    fn health_does_not_go_below_zero() {
        let router = router(Strategy::Hybrid);
        for _ in 0..20 {
            router.record_failure("a", NOW);
        }
        assert_eq!(router.health("a", NOW), 0.0);
    }

    #[test]
    fn consecutive_failures_accumulate_and_reset_on_success() {
        let router = router(Strategy::Hybrid);
        router.record_failure("a", NOW);
        router.record_failure("a", NOW);
        assert_eq!(router.consecutive_failures("a"), 2);

        router.record_success("a", NOW);
        assert_eq!(router.consecutive_failures("a"), 0);
    }

    // -- strategy behaviour -------------------------------------------------

    #[test]
    fn a_sticky_router_stays_on_the_incumbent() {
        let router = router(Strategy::Sticky);
        router.set_current("gemini", "a");

        // Even after b has been idle far longer, stickiness wins.
        let candidates = vec![
            candidate_at("a", 0, NOW, true),
            candidate_at("b", 1, NOW - 10 * 60 * 1000, true),
        ];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(0)
        );
    }

    #[test]
    fn a_sticky_router_moves_when_the_incumbent_is_unavailable() {
        let router = router(Strategy::Sticky);
        router.set_current("gemini", "a");
        let candidates = vec![
            candidate_at("a", 0, NOW, false),
            candidate_at("b", 1, NOW, true),
        ];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(1)
        );
    }

    #[test]
    fn the_cursor_is_per_family() {
        // Claude and Gemini traffic must not drag each other's sticky account.
        let router = router(Strategy::Sticky);
        let candidates = vec![available("a", 0), available("b", 1)];
        router.set_current("gemini", "a");
        router.set_current("claude", "b");

        assert_eq!(router.current("gemini").as_deref(), Some("a"));
        assert_eq!(router.current("claude").as_deref(), Some("b"));
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(0)
        );
        assert_eq!(
            router.select(&candidates, "claude", NOW),
            Selection::Chosen(1)
        );
    }

    #[test]
    fn a_hybrid_router_will_not_switch_on_a_small_advantage() {
        // Switching costs the prompt cache, so noise must not trigger it.
        let router = router(Strategy::Hybrid);
        router.set_current("gemini", "a");
        let candidates = vec![
            candidate_at("a", 0, NOW, true),
            // Slightly fresher, but nowhere near the switch threshold.
            candidate_at("b", 1, NOW - 60_000, true),
        ];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(0)
        );
    }

    #[test]
    fn a_hybrid_router_switches_on_a_clear_advantage() {
        let router = router(Strategy::Hybrid);
        router.set_current("gemini", "a");
        // Damage the incumbent so the rival clearly wins.
        for _ in 0..5 {
            router.record_failure("a", NOW);
        }
        let candidates = vec![available("a", 0), available("b", 1)];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(1)
        );
    }

    #[test]
    fn an_unhealthy_account_is_not_a_candidate() {
        let router = router(Strategy::Hybrid);
        for _ in 0..5 {
            router.record_failure("a", NOW);
        }
        let candidates = vec![available("a", 0), available("b", 1)];
        // `a` scores below the usable floor and must be passed over entirely.
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(1)
        );
    }

    #[test]
    fn an_unhealthy_incumbent_is_abandoned_by_a_sticky_router() {
        let router = router(Strategy::Sticky);
        router.set_current("gemini", "a");
        for _ in 0..5 {
            router.record_failure("a", NOW);
        }
        let candidates = vec![available("a", 0), available("b", 1)];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(1)
        );
    }

    #[test]
    fn round_robin_advances_through_the_pool() {
        let router = router(Strategy::RoundRobin);
        let candidates = vec![
            available("a", 0),
            available("b", 1),
            available("c", 2),
        ];

        assert_eq!(router.select(&candidates, "gemini", NOW), Selection::Chosen(0));
        router.set_current("gemini", "a");
        assert_eq!(router.select(&candidates, "gemini", NOW), Selection::Chosen(1));
        router.set_current("gemini", "b");
        assert_eq!(router.select(&candidates, "gemini", NOW), Selection::Chosen(2));
        router.set_current("gemini", "c");
        // Wraps back to the start.
        assert_eq!(router.select(&candidates, "gemini", NOW), Selection::Chosen(0));
    }

    #[test]
    fn round_robin_skips_unavailable_accounts() {
        let router = router(Strategy::RoundRobin);
        router.set_current("gemini", "a");
        let candidates = vec![
            candidate_at("a", 0, NOW, true),
            candidate_at("b", 1, NOW, false),
            candidate_at("c", 2, NOW, true),
        ];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(2)
        );
    }

    #[test]
    fn lru_picks_the_least_recently_used() {
        let router = router(Strategy::LeastRecentlyUsed);
        let candidates = vec![
            candidate_at("a", 0, NOW - 10_000, true),
            candidate_at("b", 1, NOW - 500_000, true),
            candidate_at("c", 2, NOW - 1_000, true),
        ];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(1)
        );
    }

    #[test]
    fn lru_never_picks_an_unavailable_account() {
        let router = router(Strategy::LeastRecentlyUsed);
        let candidates = vec![
            candidate_at("a", 0, NOW - 900_000, false),
            candidate_at("b", 1, NOW - 10_000, true),
        ];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(1)
        );
    }

    // -- the token gate ------------------------------------------------------

    #[test]
    fn an_account_out_of_tokens_is_passed_over() {
        let router = router(Strategy::Hybrid);
        for _ in 0..(DEFAULT_TOKEN_MAX as u32) {
            router.consume_token("a", NOW);
        }

        let candidates = vec![available("a", 0), available("b", 1)];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(1),
            "the drained account must not be selected"
        );
    }

    #[test]
    fn a_pool_out_of_tokens_reports_a_wait_rather_than_no_candidate() {
        // Buckets refill, so this is a waiting problem, not a broken pool.
        let router = router(Strategy::Hybrid);
        for _ in 0..(DEFAULT_TOKEN_MAX as u32) {
            router.consume_token("a", NOW);
        }

        match router.select(&[available("a", 0)], "gemini", NOW) {
            Selection::AllLimited { earliest_in } => {
                let wait = earliest_in.expect("a refill estimate");
                // One token at six per minute is ten seconds.
                assert!(
                    wait >= Duration::from_secs(9) && wait <= Duration::from_secs(11),
                    "got {wait:?}"
                );
            }
            other => panic!("expected a wait, got {other:?}"),
        }
    }

    #[test]
    fn a_refilling_bucket_returns_to_service() {
        let router = router(Strategy::Hybrid);
        for _ in 0..(DEFAULT_TOKEN_MAX as u32) {
            router.consume_token("a", NOW);
        }
        assert!(matches!(
            router.select(&[available("a", 0)], "gemini", NOW),
            Selection::AllLimited { .. }
        ));

        // A minute of refill is six tokens, comfortably above the gate.
        let later = NOW + 60_000;
        assert_eq!(
            router.select(&[available("a", 0)], "gemini", later),
            Selection::Chosen(0)
        );
    }

    #[test]
    fn a_rate_limit_still_outranks_a_token_deficit_for_the_wait_estimate() {
        // When an account is genuinely limited, the upstream's reset is the
        // right number to report, not our refill estimate.
        let router = router(Strategy::Hybrid);
        for _ in 0..(DEFAULT_TOKEN_MAX as u32) {
            router.consume_token("a", NOW);
        }
        let candidates = vec![Candidate {
            next_reset_in: Some(Duration::from_secs(600)),
            ..candidate_at("a", 0, NOW, false)
        }];

        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::AllLimited {
                earliest_in: Some(Duration::from_secs(600))
            }
        );
    }

    #[test]
    fn a_depleted_bucket_does_not_mask_a_disabled_account() {
        // Nothing is available and nothing has a reset, so waiting is useless.
        let router = router(Strategy::Hybrid);
        for _ in 0..(DEFAULT_TOKEN_MAX as u32) {
            router.consume_token("a", NOW);
        }
        let candidates = vec![candidate_at("a", 0, NOW, false)];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::NoCandidate
        );
    }

    // -- housekeeping -------------------------------------------------------

    #[test]
    fn retain_drops_state_for_departed_accounts() {
        let router = router(Strategy::Hybrid);
        router.consume_token("a", NOW);
        router.consume_token("gone", NOW);
        router.set_current("gemini", "gone");

        router.retain(&["a".to_string()]);

        assert_eq!(router.tokens("a", NOW), DEFAULT_TOKEN_MAX - 1.0);
        // A departed account starts fresh rather than inheriting its history.
        assert_eq!(router.tokens("gone", NOW), DEFAULT_TOKEN_MAX);
        assert!(router.current("gemini").is_none());
    }

    #[test]
    fn state_is_keyed_by_credential_not_by_position() {
        // Drawing down one account must not affect another, even if the other
        // occupies the slice it used to.
        let router = router(Strategy::Hybrid);
        for _ in 0..10 {
            router.consume_token("first", NOW);
        }
        assert_eq!(router.tokens("first", NOW), DEFAULT_TOKEN_MAX - 10.0);
        assert_eq!(
            router.tokens("second", NOW),
            DEFAULT_TOKEN_MAX,
            "a different credential must start fresh"
        );

        // And once the first departs, its state goes with it rather than being
        // inherited by whoever takes its place.
        router.retain(&["second".to_string()]);
        assert_eq!(router.tokens("first", NOW), DEFAULT_TOKEN_MAX);
        assert_eq!(router.tokens("second", NOW), DEFAULT_TOKEN_MAX);
    }

    #[test]
    fn selection_is_safe_under_concurrency() {
        let router = std::sync::Arc::new(router(Strategy::Hybrid));
        let candidates = vec![available("a", 0), available("b", 1)];

        let threads: Vec<_> = (0..8)
            .map(|worker| {
                let router = router.clone();
                let candidates = candidates.clone();
                std::thread::spawn(move || {
                    for index in 0..100 {
                        let family = if worker % 2 == 0 { "gemini" } else { "claude" };
                        let _ = router.select(&candidates, family, NOW + index);
                        router.consume_token("a", NOW + index);
                        router.record_success("b", NOW + index);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    }

    #[test]
    fn a_cursor_naming_a_departed_account_is_ignored() {
        let router = router(Strategy::Sticky);
        router.set_current("gemini", "vanished");
        let candidates = vec![available("a", 0)];
        assert_eq!(
            router.select(&candidates, "gemini", NOW),
            Selection::Chosen(0)
        );
    }

    #[test]
    fn candidate_building_reads_account_state() {
        use crate::accounts::account::{Account, CooldownReason};

        let mut account = Account::new("token");
        let built = candidate(3, &account, NOW);
        assert_eq!(built.index, 3);
        assert!(built.is_available);
        assert!(built.next_reset_in.is_none());

        account.mark_rate_limited("gemini", NOW + 60_000);
        let built = candidate(0, &account, NOW);
        assert!(!built.is_available);
        assert_eq!(built.next_reset_in, Some(Duration::from_secs(60)));

        account.clear_holds();
        account.mark_cooling_down(NOW + 30_000, CooldownReason::NetworkError);
        let built = candidate(0, &account, NOW);
        assert!(!built.is_available);
    }
}
