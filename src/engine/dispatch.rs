//! The dispatch loop.
//!
//! Selection, endpoint fallback, retry, and the write-back of what was learned
//! about each account. The decision logic lives in [`super::retry`]; this file
//! only executes the steps it returns.
//!
//! Two properties are worth stating because they shape the code:
//!
//! **Retries happen on the response head, never mid-body.** A stream that has
//! already delivered bytes cannot be retried, so the loop does not hand control
//! back until it has a 2xx and headers. This is why the loop returns a
//! `StreamingResponse` rather than a decoded result.
//!
//! **Translation happens once; the envelope is rebuilt per account.** The IR is
//! account-independent, but the project id is not, so each attempt re-wraps the
//! same IR. Translating inside the loop would be wasteful and would also
//! re-cache signatures once per attempt, which is wrong for a retry that reached
//! no upstream.

use std::time::Duration;

use crate::accounts::account::{Account, now_ms};
use crate::accounts::ratelimit::{AccountProblem, RateLimitReason, UpstreamFailure, backoff_for};
use crate::accounts::router::candidate;
use crate::engine::retry::{AttemptState, RetryError, Step, decide, pool_wait};
use crate::registry::models::{ModelFamily, ResolvedModel};
use crate::transform::ir::GenerateContentRequest;
use crate::transform::openai::ChatCompletionRequest;
use crate::transform::request::{TranslateError, resolve_for, to_ir_with_signatures};
use crate::upstream::envelope::Envelope;
use crate::upstream::transport::{CallKind, StreamingResponse, TransportError};
use crate::upstream::constants;

use super::{Engine, ProbeError};

/// A request translated and resolved, ready to dispatch.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub ir: GenerateContentRequest,
    pub resolved: ResolvedModel,
    pub session_key: String,
    /// The model name the client sent, for metrics and the audit log. Distinct
    /// from the wire model, and the one an operator will recognise.
    pub requested_model: String,
    /// Whether the client asked for a stream, recorded so the audit log can
    /// distinguish the two without guessing from the outcome.
    pub stream: bool,
}

/// A successful upstream call.
pub struct UpstreamCall {
    pub response: StreamingResponse,
    /// Pool position of the account that served it.
    pub account_index: usize,
    pub credential_id: String,
    pub endpoint: String,
    /// How many upstream attempts this took, including the successful one.
    pub attempts: u32,
    pub wire_model: String,
    pub family: ModelFamily,
    pub session_key: String,
    /// Project the request was sent under, as resolved for this account.
    pub project_id: String,
    /// Whether that project is the shared fallback rather than a real one.
    pub used_fallback_project: bool,
    /// Subscription tier, as reported by the account's most recent discovery.
    pub tier: String,
}

/// What a successful attempt established.
struct AttemptSuccess {
    response: StreamingResponse,
    project_id: String,
    used_fallback_project: bool,
    tier: String,
    tier_id: Option<String>,
    paid_tier_id: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    #[error("{0}")]
    Retry(#[from] RetryError),

    #[error("no accounts are configured")]
    NoAccounts,

    #[error("no account can serve this request: {0}")]
    NoCandidate(String),

    #[error("could not resolve the model: {0}")]
    Resolve(#[from] crate::registry::ResolveError),

    #[error("could not translate the request: {0}")]
    Translate(#[from] TranslateError),

    #[error("could not serialise the request: {0}")]
    Serialise(#[source] serde_json::Error),
}

/// Why one attempt stopped.
enum AttemptError {
    /// A classified upstream or transport failure. Retryable per policy.
    Upstream(Box<UpstreamFailure>),
    /// Something that retrying cannot fix.
    Fatal(Box<DispatchError>),
}

impl From<UpstreamFailure> for AttemptError {
    fn from(failure: UpstreamFailure) -> Self {
        Self::Upstream(Box::new(failure))
    }
}

/// Jitter for capacity retries. A fixed value rather than a random one keeps
/// dispatch deterministic in tests; the spread that matters happens across
/// processes, not within one.
const CAPACITY_JITTER: f64 = 0.5;

impl Engine {
    /// Translate a client request and resolve the model it names.
    pub fn prepare(&self, request: &ChatCompletionRequest) -> Result<Prepared, DispatchError> {
        let resolved = resolve_for(request, &self.config)?;

        // The session key drives both the upstream session id and signature
        // storage, so it has to be stable across a conversation: the client's
        // `user` field when it sends one, otherwise the opening turn's text.
        let session_key = session_key_for(request);

        let (ir, notes) =
            to_ir_with_signatures(request, &resolved, &self.config, &self.signatures)?;
        for warning in &notes.warnings {
            tracing::debug!(warning, "request translation");
        }

        Ok(Prepared {
            ir,
            resolved,
            session_key,
            requested_model: request.model.clone(),
            stream: request.stream,
        })
    }

    /// Dispatch a prepared request, rotating accounts and endpoints as needed.
    pub async fn dispatch(&self, prepared: &Prepared) -> Result<UpstreamCall, DispatchError> {
        self.dispatch_inner(prepared, None).await
    }

    /// Dispatch to one named account only.
    ///
    /// Used by `account verify` and by a targeted probe, where the question is
    /// "does *this* account work" and falling back to another would answer a
    /// different question.
    pub async fn dispatch_to(
        &self,
        prepared: &Prepared,
        credential_id: &str,
    ) -> Result<UpstreamCall, DispatchError> {
        self.dispatch_inner(prepared, Some(credential_id.to_string()))
            .await
    }

    async fn dispatch_inner(
        &self,
        prepared: &Prepared,
        only: Option<String>,
    ) -> Result<UpstreamCall, DispatchError> {
        let endpoints = self.endpoints();
        if endpoints.is_empty() {
            return Err(DispatchError::NoAccounts);
        }

        let family = prepared.resolved.family;
        // Rate limits are enforced per quota pool, which is coarser than the
        // model and finer than the family.
        let pool = family.quota_pool();

        let mut state = AttemptState::new(endpoints.len());
        let mut tried: Vec<String> = Vec::new();
        let mut current: Option<(usize, String)> = None;
        let mut total_waited = Duration::ZERO;
        let mut last_failure: Option<UpstreamFailure> = None;

        loop {
            // Acquire an account when we do not already hold one. A same-account
            // retry keeps `current` and skips this entirely.
            if current.is_none() {
                match self
                    .acquire_account(&tried, family.quota_pool(), &mut total_waited, only.as_deref())
                    .await
                {
                    Ok(Some(account)) => current = Some(account),
                    Ok(None) => {
                        // Every account has been tried for this request.
                        return Err(DispatchError::Retry(RetryError::AccountsExhausted {
                            tried: state.accounts_tried,
                            last: last_failure
                                .as_ref()
                                .map(describe_failure)
                                .unwrap_or_else(|| "no attempt succeeded".into()),
                        }));
                    }
                    Err(error) => return Err(error),
                }
            }

            let (index, credential_id) = current.clone().expect("just ensured");
            let snapshot = self.accounts.snapshot();
            let Some(account) = snapshot.accounts.get(index).cloned() else {
                // The pool changed under us; start over with this one excluded.
                tried.push(credential_id);
                current = None;
                continue;
            };

            let endpoint = endpoints[state.endpoint_index % endpoints.len()].clone();
            tracing::debug!(
                account = %account.label(),
                endpoint = %endpoint,
                attempt = state.accounts_tried,
                "dispatching"
            );

            match self.attempt_once(prepared, &account, &endpoint).await {
                Ok(success) => {
                    self.record_success(&credential_id, family.quota_pool(), &success);
                    return Ok(UpstreamCall {
                        response: success.response,
                        account_index: index,
                        credential_id,
                        endpoint,
                        attempts: state.accounts_tried + state.same_account_retries,
                        wire_model: prepared.resolved.wire_model.clone(),
                        family,
                        session_key: prepared.session_key.clone(),
                        project_id: success.project_id,
                        used_fallback_project: success.used_fallback_project,
                        tier: success.tier,
                    });
                }

                Err(AttemptError::Fatal(error)) => return Err(*error),

                Err(AttemptError::Upstream(failure)) => {
                    let failure = *failure;
                    self.apply_failure(&credential_id, pool, &failure);
                    last_failure = Some(failure.clone());

                    let step = decide(&failure, &state, &self.config.routing, CAPACITY_JITTER);
                    tracing::debug!(
                        account = %account.label(),
                        failure = ?failure,
                        ?step,
                        "attempt failed"
                    );

                    match step {
                        Step::NextEndpoint => {
                            state.advance_endpoint();
                        }
                        Step::RetrySameEndpoint { delay } => {
                            state.note_same_account_retry();
                            tokio::time::sleep(delay).await;
                        }
                        Step::RotateAccount { delay } => {
                            if !delay.is_zero() {
                                tokio::time::sleep(delay).await;
                            }
                            tried.push(credential_id);
                            state.begin_next_account();
                            current = None;
                        }
                        Step::Fail(error) => {
                            return Err(DispatchError::Retry(error));
                        }
                    }
                }
            }
        }
    }

    /// Pick an account, waiting for a rate-limited pool when that is worthwhile.
    ///
    /// Returns `Ok(None)` when every account has already been tried for this
    /// request, which is the loop's signal to give up.
    async fn acquire_account(
        &self,
        tried: &[String],
        pool: &str,
        total_waited: &mut Duration,
        only: Option<&str>,
    ) -> Result<Option<(usize, String)>, DispatchError> {
        let budget = Duration::from_secs(self.config.routing.max_wait_before_error_secs);

        loop {
            let now = now_ms();
            let snapshot = self.accounts.snapshot();

            if snapshot.accounts.is_empty() {
                return Err(DispatchError::NoAccounts);
            }
            self.router.retain(
                &snapshot
                    .accounts
                    .iter()
                    .map(|account| account.credential_id())
                    .collect::<Vec<_>>(),
            );

            let remaining: Vec<_> = snapshot
                .accounts
                .iter()
                .enumerate()
                .filter(|(_, account)| !tried.contains(&account.credential_id()))
                .filter(|(_, account)| match only {
                    Some(only) => account.credential_id() == only,
                    None => true,
                })
                .map(|(index, account)| candidate(index, account, now))
                .collect();

            if remaining.is_empty() {
                return Ok(None);
            }

            match self.router.select(&remaining, pool, now) {
                crate::accounts::router::Selection::Chosen(index) => {
                    let credential_id = snapshot
                        .accounts
                        .get(index)
                        .map(Account::credential_id)
                        .unwrap_or_default();
                    // Spend here rather than per attempt: a same-account retry
                    // is not a new request as far as the bucket is concerned.
                    self.router.consume_token(&credential_id, now);
                    return Ok(Some((index, credential_id)));
                }

                crate::accounts::router::Selection::AllLimited { earliest_in } => {
                    match pool_wait(earliest_in, &self.config.routing) {
                        Some(wait) if *total_waited + wait <= budget => {
                            tracing::info!(
                                seconds = wait.as_secs(),
                                "every account is rate limited; waiting"
                            );
                            tokio::time::sleep(wait).await;
                            *total_waited += wait;
                            // Clear now-expired limits and look again.
                            self.accounts
                                .mutate(|storage| {
                                    let now = now_ms();
                                    // Every account is visited rather than
                                    // short-circuiting on the first change: a
                                    // fold that stopped early would leave later
                                    // accounts holding expired limits.
                                    let mut changed = false;
                                    for account in storage.accounts.iter_mut() {
                                        changed |= account.clear_expired_limits(now);
                                    }
                                    changed
                                })
                                .ok();
                        }
                        _ => {
                            return Err(DispatchError::Retry(RetryError::PoolExhausted {
                                earliest_in,
                            }));
                        }
                    }
                }

                crate::accounts::router::Selection::NoCandidate => {
                    return Err(DispatchError::NoCandidate(describe_pool(&snapshot, now)));
                }
            }
        }
    }

    /// One upstream attempt: resolve credentials, wrap, send.
    async fn attempt_once(
        &self,
        prepared: &Prepared,
        account: &Account,
        endpoint: &str,
    ) -> Result<AttemptSuccess, AttemptError> {
        let prepared_account = match self.prepare_account(account).await {
            Ok(prepared) => prepared,
            Err(ProbeError::OAuth(error)) => {
                // A terminal OAuth failure means the credential is dead, and no
                // amount of retrying will revive it.
                return Err(if error.is_terminal() {
                    AttemptError::Upstream(Box::new(UpstreamFailure::Account(
                        AccountProblem::AuthInvalid {
                            reason: error.to_string(),
                        },
                    )))
                } else {
                    AttemptError::Upstream(Box::new(UpstreamFailure::Network))
                });
            }
            Err(error) => {
                // Project discovery failed. Treat it as an endpoint-level
                // refusal so the loop tries elsewhere before blaming the account.
                tracing::warn!(%error, "account preparation failed");
                return Err(AttemptError::Upstream(Box::new(UpstreamFailure::Network)));
            }
        };

        let envelope = Envelope::build(
            prepared.ir.clone(),
            &prepared_account.project_id,
            &prepared.resolved.wire_model,
            &prepared.session_key,
            &self.sessions,
        );
        let body = serde_json::to_vec(&envelope).map_err(|error| {
            AttemptError::Fatal(Box::new(DispatchError::Serialise(error)))
        })?;

        let url = format!(
            "{endpoint}{}?alt=sse",
            constants::API_STREAM_GENERATE
        );

        match self
            .upstream
            .post_json(
                &url,
                &prepared_account.access_token,
                &body,
                CallKind::Streaming,
            )
            .await
        {
            Ok(response) if response.status.is_success() => Ok(AttemptSuccess {
                response,
                project_id: prepared_account.project_id.clone(),
                used_fallback_project: prepared_account.used_fallback_project,
                tier: prepared_account.tier.clone(),
                tier_id: prepared_account.tier_id.clone(),
                paid_tier_id: prepared_account.paid_tier_id.clone(),
            }),
            Ok(response) => {
                // A non-2xx body carries the classification, so it has to be
                // read before the response can be judged. This is the reason the
                // transport returns a head plus a lazy body rather than an
                // already-classified result.
                let status = response.status;
                let headers = response.headers.clone();
                let body = drain_to_string(response.body).await;
                Err(AttemptError::Upstream(Box::new(
                    crate::accounts::ratelimit::classify(status, &headers, &body),
                )))
            }
            Err(TransportError::HttpStatus { .. } | TransportError::Request(_)) => {
                Err(AttemptError::Upstream(Box::new(UpstreamFailure::Network)))
            }
            Err(TransportError::Malformed { .. }) => {
                Err(AttemptError::Upstream(Box::new(UpstreamFailure::Network)))
            }
        }
    }

    /// Record what a failure says about an account, in the router and on disk.
    fn apply_failure(&self, credential_id: &str, pool: &str, failure: &UpstreamFailure) {
        let now = now_ms();
        let max_failures = self.config.accounts.max_consecutive_failures;
        let cooldown_ms = (self.config.accounts.cooldown_secs as i64) * 1000;

        match failure {
            UpstreamFailure::RateLimited { reason, reset_in } => {
                self.router.record_rate_limit(credential_id, now);

                let consecutive = self.router.consecutive_failures(credential_id);
                let delay = backoff_for(*reason, consecutive, *reset_in, CAPACITY_JITTER);
                let reset_at = now + delay.as_millis() as i64;

                // A capacity error is the model's problem, so recording it
                // against the account would bench a perfectly good account.
                if *reason == RateLimitReason::ModelCapacityExhausted {
                    return;
                }

                let pool = pool.to_string();
                self.persist(credential_id, |account| {
                    account.mark_rate_limited(&pool, reset_at);
                });
            }

            UpstreamFailure::Account(problem) => {
                self.router.record_failure(credential_id, now);
                match problem {
                    AccountProblem::AuthInvalid { reason } => {
                        // A dead credential cannot be retried, and leaving it
                        // enabled means every future request pays for the
                        // discovery again.
                        let reason = reason.clone();
                        self.persist(credential_id, |account| {
                            account.enabled = false;
                            account.mark_ineligible(format!("credential rejected: {reason}"));
                        });
                    }
                    AccountProblem::VerificationRequired { url } => {
                        let url = url.clone();
                        let reason = problem.label().to_string();
                        self.persist(credential_id, |account| {
                            account.mark_verification_required(url, reason);
                        });
                    }
                    AccountProblem::Ineligible { reason } => {
                        let reason = reason.clone();
                        self.persist(credential_id, |account| {
                            account.mark_ineligible(reason);
                        });
                    }
                    AccountProblem::Banned { reason } => {
                        let reason = reason.clone();
                        self.persist(credential_id, |account| {
                            account.enabled = false;
                            account.mark_ineligible(reason);
                        });
                    }
                }
            }

            UpstreamFailure::Server { .. } | UpstreamFailure::Network => {
                self.router.record_failure(credential_id, now);
                // A transport failure probably means the request never reached
                // the upstream, so the spend was not really made.
                if matches!(failure, UpstreamFailure::Network) {
                    self.router.refund_token(credential_id, now);
                }
                if self.router.consecutive_failures(credential_id) >= max_failures {
                    let reason = if matches!(failure, UpstreamFailure::Network) {
                        crate::accounts::account::CooldownReason::NetworkError
                    } else {
                        crate::accounts::account::CooldownReason::AuthFailure
                    };
                    self.persist(credential_id, |account| {
                        account.mark_cooling_down(now + cooldown_ms, reason);
                    });
                }
            }

            UpstreamFailure::Refused { .. } => {
                // The endpoint refused; the account is not implicated.
            }

            UpstreamFailure::BadRequest { .. } => {
                // Cannot be reached: a bad request fails the loop immediately.
            }
        }
    }

    fn record_success(&self, credential_id: &str, pool: &str, success: &AttemptSuccess) {
        let now = now_ms();
        self.router.record_success(credential_id, now);
        self.router.set_current(pool, credential_id);

        let project_id = success.project_id.clone();
        let used_fallback = success.used_fallback_project;
        let tier_id = success.tier_id.clone();
        let paid_tier_id = success.paid_tier_id.clone();

        self.persist(credential_id, |account| {
            account.last_used = now;
            // A success clears any stale limit for this pool: the upstream just
            // served a request, so whatever we recorded is out of date.
            account.rate_limit_reset_times.remove(pool);

            // Write the resolved project back, so the next request skips
            // discovery. A project that only worked by falling back to the
            // shared one is deliberately *not* recorded: caching it would make
            // a wrong project permanent, and the fallback is what the account
            // layer retries away from.
            if !used_fallback {
                account.project_id = Some(project_id);
            }
            // Record the raw ids the upstream reported, not a formatted
            // description: this is data, and formatting belongs at the edges.
            //
            // Updated on every successful discovery rather than only when
            // absent, because tiers change: an account that upgrades from free
            // to paid would otherwise be described by its old tier forever.
            account.captured_tier_id = tier_id;
            account.captured_paid_tier_id = paid_tier_id;
            account.captured_tier_at = Some(now);
        });
    }

    /// Apply a change to one account, if it is still in the pool.
    ///
    /// Any change here is a state transition worth recording, so the closure
    /// returns nothing and the write always happens. An account that has since
    /// been removed is silently skipped.
    fn persist(&self, credential_id: &str, change: impl FnOnce(&mut Account)) {
        let result = self.accounts.mutate(|storage| {
            match storage
                .accounts
                .iter_mut()
                .find(|account| account.credential_id() == credential_id)
            {
                Some(account) => {
                    change(account);
                    true
                }
                None => false,
            }
        });
        if let Err(error) = result {
            // Losing a state update is not worth failing a client request over,
            // but it should be visible.
            tracing::error!(%error, credential_id, "could not persist account state");
        }
    }
}

/// Read a non-2xx body so it can be classified.
async fn drain_to_string(mut body: crate::upstream::transport::BodyStream) -> String {
    use futures::StreamExt;
    let mut collected = Vec::new();
    while let Some(chunk) = body.next().await {
        match chunk {
            Ok(chunk) => {
                collected.extend_from_slice(&chunk);
                // Error bodies are diagnostics; a few kilobytes is plenty.
                if collected.len() > 64 * 1024 {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    String::from_utf8_lossy(&collected).into_owned()
}

/// Derive the conversation key used for upstream session identity and signature
/// storage.
///
/// The client's `user` field is the best signal when present because it is
/// stable and explicit. Otherwise the opening user turn is hashed: it is the one
/// part of a conversation that does not change between turns, which is what
/// makes the upstream's prompt cache effective.
fn session_key_for(request: &ChatCompletionRequest) -> String {
    if let Some(user) = request.user.as_deref().filter(|user| !user.is_empty()) {
        return format!("user:{user}");
    }

    let opening = request
        .messages
        .iter()
        .find(|message| message.role == "user")
        .map(|message| match &message.content {
            Some(crate::transform::openai::MessageContent::Text(text)) => text.clone(),
            Some(crate::transform::openai::MessageContent::Parts(parts)) => parts
                .iter()
                .filter_map(|part| part.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
            None => String::new(),
        })
        .unwrap_or_default();

    if opening.is_empty() {
        // Nothing stable to key on. A per-request key is the honest answer:
        // pretending otherwise would alias unrelated conversations together.
        return format!("anon:{}", uuid::Uuid::new_v4());
    }

    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(opening.as_bytes());
    let hex: String = digest[..16].iter().map(|byte| format!("{byte:02x}")).collect();
    format!("conv:{hex}")
}

/// Describe why no account can serve a request.
fn describe_pool(storage: &crate::accounts::account::AccountStorage, now: i64) -> String {
    let mut reasons: Vec<String> = Vec::new();
    for account in &storage.accounts {
        if account.account_ineligible {
            reasons.push(format!("{} is ineligible", account.label()));
        } else if account.verification_required {
            reasons.push(format!("{} needs verification", account.label()));
        } else if !account.enabled {
            reasons.push(format!("{} is disabled", account.label()));
        } else if account
            .cooling_down_until
            .is_some_and(|until| until > now)
        {
            reasons.push(format!("{} is cooling down", account.label()));
        }
    }
    if reasons.is_empty() {
        "every account is below the usable health threshold".into()
    } else {
        reasons.join("; ")
    }
}

/// A short description of a failure, for the exhausted-pool message.
fn describe_failure(failure: &UpstreamFailure) -> String {
    match failure {
        UpstreamFailure::RateLimited { reason, .. } => format!("rate limited ({reason:?})"),
        UpstreamFailure::Account(problem) => format!("account problem ({})", problem.label()),
        UpstreamFailure::Server { status } => format!("upstream error {status}"),
        UpstreamFailure::Network => "network error".into(),
        UpstreamFailure::BadRequest { message } => format!("bad request: {message}"),
        UpstreamFailure::Refused { status, message } => format!("refused ({status}): {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::openai::ChatCompletionRequest;
    use serde_json::json;

    fn request(value: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn an_explicit_user_field_becomes_the_session_key() {
        let request = request(json!({
            "model": "m",
            "user": "alice",
            "messages": [{ "role": "user", "content": "hello" }]
        }));
        assert_eq!(session_key_for(&request), "user:alice");
    }

    #[test]
    fn an_empty_user_field_is_ignored() {
        let request = request(json!({
            "model": "m",
            "user": "",
            "messages": [{ "role": "user", "content": "hello" }]
        }));
        assert!(session_key_for(&request).starts_with("conv:"));
    }

    #[test]
    fn the_opening_turn_keys_the_session_when_no_user_is_given() {
        // The opening turn is the one part of a conversation that does not
        // change between turns, which is what makes prompt caching work.
        let first = request(json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "What is the weather?" },
                { "role": "assistant", "content": "Which city?" },
                { "role": "user", "content": "Paris" }
            ]
        }));
        let later = request(json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "What is the weather?" },
                { "role": "assistant", "content": "Which city?" },
                { "role": "user", "content": "Paris" },
                { "role": "assistant", "content": "Sunny." },
                { "role": "user", "content": "And tomorrow?" }
            ]
        }));

        assert_eq!(session_key_for(&first), session_key_for(&later));
        assert!(session_key_for(&first).starts_with("conv:"));
    }

    #[test]
    fn different_conversations_get_different_keys() {
        let a = request(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "first conversation" }]
        }));
        let b = request(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "a different one" }]
        }));
        assert_ne!(session_key_for(&a), session_key_for(&b));
    }

    #[test]
    fn a_multi_part_opening_turn_is_keyed_on_its_text() {
        let request = request(json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "describe this" },
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,AA" } }
                ]
            }]
        }));
        assert!(session_key_for(&request).starts_with("conv:"));
    }

    #[test]
    fn a_request_with_no_text_falls_back_to_a_unique_key() {
        // Aliasing unrelated conversations together would be worse than losing
        // the cache.
        let empty = request(json!({ "model": "m", "messages": [] }));
        let other = request(json!({ "model": "m", "messages": [] }));
        let first = session_key_for(&empty);
        assert!(first.starts_with("anon:"));
        assert_ne!(first, session_key_for(&other));
    }

    #[test]
    fn pool_descriptions_name_the_blocking_condition() {
        use crate::accounts::account::{Account, AccountStorage};

        let mut storage = AccountStorage::default();
        let mut ineligible = Account::new("a");
        ineligible.email = Some("bad@example.com".into());
        ineligible.mark_ineligible("banned");
        storage.accounts.push(ineligible);

        let description = describe_pool(&storage, 0);
        assert!(description.contains("ineligible"), "got: {description}");
        assert!(description.contains("bad@example.com"));
    }

    #[test]
    fn pool_description_falls_back_to_health_when_nothing_is_flagged() {
        use crate::accounts::account::{Account, AccountStorage};
        let mut storage = AccountStorage::default();
        storage.accounts.push(Account::new("a"));
        let description = describe_pool(&storage, 0);
        assert!(description.contains("health"), "got: {description}");
    }

    #[test]
    fn failure_descriptions_are_short_but_specific() {
        assert!(describe_failure(&UpstreamFailure::Network).contains("network"));
        assert!(
            describe_failure(&UpstreamFailure::Account(AccountProblem::Banned {
                reason: "x".into()
            }))
            .contains("banned")
        );
    }
}
