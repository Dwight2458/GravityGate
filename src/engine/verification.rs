//! Bounded account-holder verification checks. Discovery alone is not proof:
//! require visible output, a finish reason, and a clean end of the SSE stream.

use std::time::Duration;

use futures::StreamExt;
use serde::Serialize;

use crate::accounts::account::{Account, now_ms};
use crate::accounts::ratelimit::{AccountProblem, RateLimitReason, UpstreamFailure, backoff_for};
use crate::accounts::store::StoreError;
use crate::transform::openai::ChatCompletionRequest;
use crate::upstream::sse::SseDecoder;

use super::Engine;
use super::dispatch::{AttemptError, AttemptSuccess};

#[derive(Debug, Serialize)]
pub struct VerificationView {
    pub state: &'static str,
    pub next_check_at: Option<i64>,
    pub last_checked_at: Option<i64>,
    pub last_result: Option<String>,
    pub recovered_at: Option<i64>,
}

pub fn view(account: &Account, now: i64) -> VerificationView {
    VerificationView {
        state: if !account.enabled || account.account_ineligible {
            "blocked"
        } else if account.verification_check.running(now) {
            "checking"
        } else if account.verification_required {
            "waiting"
        } else if account.verification_check.recovered_at.is_some() {
            "recovered"
        } else {
            "unchecked"
        },
        next_check_at: (account.verification_required
            && account.enabled
            && !account.account_ineligible)
            .then(|| {
                account
                    .verification_recheck_at
                    .unwrap_or(now)
                    .max(account.cooling_down_until.unwrap_or(0))
                    .max(if account.verification_check.running(now) {
                        account.verification_check.lease_until.unwrap_or(now)
                    } else {
                        0
                    })
            }),
        last_checked_at: account.verification_check.last_checked_at,
        last_result: account.verification_check.last_result.clone(),
        recovered_at: account.verification_check.recovered_at,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CheckError {
    #[error("account not found or selector is ambiguous")]
    NotFound,
    #[error("a verification check is already running, or was started less than 15 seconds ago")]
    Busy,
    #[error(
        "account disabled, ineligible, cooling down, or rate limited; check /health for the next check time"
    )]
    Blocked,
    #[error("{0}")]
    Store(#[from] StoreError),
    #[error("could not prepare the configured verification model")]
    Model,
}

/// 1 minute, 2 minutes, then at most 5 minutes between inconclusive checks.
fn delay_ms(failures: u32) -> i64 {
    match failures {
        0 | 1 => 60_000,
        2 => 120_000,
        _ => 300_000,
    }
}

struct Lease<'a> {
    engine: &'a Engine,
    credential_id: String,
    id: String,
    revision: u64,
    active: bool,
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        // Cancellation releases only this generation's lease. A late task must
        // never overwrite a newer check, operator reset, or a removed account.
        let result = self.engine.accounts.mutate(|storage| {
            let Some(account) = storage.accounts.iter_mut().find(|a| {
                a.credential_id() == self.credential_id
                    && a.verification_check.id.as_deref() == Some(&self.id)
            }) else {
                return false;
            };
            account.verification_check.id = None;
            account.verification_check.lease_until = None;
            if account.verification_revision != self.revision {
                return true;
            }
            account.verification_check.last_checked_at = Some(now_ms());
            account.verification_check.last_result = Some("cancelled".into());
            account.verification_recheck_at = Some(now_ms() + 60_000);
            true
        });
        if let Err(error) = result {
            tracing::warn!(%error, "could not release verification check");
        }
    }
}

impl Engine {
    /// A manual check bypasses only the verification timer, never a real quota,
    /// disable, ban, or an already running check. No account fallback is allowed.
    pub async fn recheck_verification(
        &self,
        selector: &str,
        immediate: bool,
    ) -> Result<VerificationView, CheckError> {
        self.accounts.reload()?;
        let snapshot = self.accounts.snapshot();
        let matches: Vec<_> = snapshot
            .accounts
            .iter()
            .filter(|a| {
                a.email.as_deref() == Some(selector) || a.credential_id().starts_with(selector)
            })
            .collect();
        if matches.len() != 1 {
            return Err(CheckError::NotFound);
        }
        let credential_id = matches[0].credential_id();
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": self.config.verification.model,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            "max_tokens": 1024,
            "user": format!("verification:{}", uuid::Uuid::new_v4()),
        }))
        .map_err(|_| CheckError::Model)?;
        let prepared = self.prepare(&request).map_err(|_| CheckError::Model)?;
        let pool = prepared.resolved.quota_key();
        let id = uuid::Uuid::new_v4().to_string();
        let now = now_ms();
        let mut claimed = None;
        let mut rejection = CheckError::Blocked;
        self.accounts.mutate(|storage| {
            let Some(account) = storage
                .accounts
                .iter_mut()
                .find(|a| a.credential_id() == credential_id)
            else {
                return false;
            };
            if account.verification_check.running(now)
                || account
                    .verification_check
                    .started_at
                    .is_some_and(|at| at > now - 15_000)
            {
                rejection = CheckError::Busy;
                return false;
            }
            if !account.enabled
                || account.account_ineligible
                || account.cooling_down_until.is_some_and(|at| at > now)
                || (!immediate
                    && (!account.verification_required
                        || account.verification_recheck_at.is_some_and(|at| at > now)))
            {
                return false;
            }
            if let Some(until) = account.reset_at_for(&pool, now) {
                account.verification_recheck_at =
                    Some(until.max(account.verification_recheck_at.unwrap_or(0)));
                return true;
            }
            account.verification_check.id = Some(id.clone());
            account.verification_check.started_at = Some(now);
            account.verification_check.lease_until =
                Some(now + (self.config.verification.timeout_secs as i64 + 30) * 1000);
            claimed = Some(account.clone());
            true
        })?;
        let account = claimed.ok_or(rejection)?;
        let mut lease = Lease {
            engine: self,
            credential_id: credential_id.clone(),
            id,
            revision: account.verification_revision,
            active: true,
        };
        let outcome = if self.router.consume_token(&credential_id, now) {
            tracing::info!(account_id = %&credential_id[..8], immediate, "verification check started");
            match tokio::time::timeout(
                Duration::from_secs(self.config.verification.timeout_secs),
                self.check_generation(&prepared, &account),
            )
            .await
            {
                Ok(outcome) => outcome,
                Err(_) => Err(CheckFailure::Inconclusive("timeout")),
            }
        } else {
            Err(CheckFailure::Inconclusive("local_rate_limit"))
        };

        let mut updated = None;
        self.accounts.mutate(|storage| {
            let Some(stored) = storage.accounts.iter_mut().find(|a| {
                a.credential_id() == credential_id
                    && a.verification_check.id.as_deref() == Some(&lease.id)
            }) else {
                return false;
            };
            let now = now_ms();
            if stored.verification_revision != lease.revision {
                stored.verification_check.id = None;
                stored.verification_check.lease_until = None;
                updated = Some(view(stored, now));
                return true;
            }
            let failures = stored.verification_check.failures.saturating_add(1);
            let result = match &outcome {
                Ok(success) => {
                    stored.clear_verification();
                    stored.last_used = now;
                    stored.rate_limit_reset_times.remove(&pool);
                    if !success.used_fallback_project {
                        stored.project_id = Some(success.project_id.clone());
                    }
                    stored.captured_tier_id = success.tier_id.clone();
                    stored.captured_paid_tier_id = success.paid_tier_id.clone();
                    stored.captured_tier_at = Some(now);
                    stored.verification_check.recovered_at = Some(now);
                    stored.verification_check.failures = 0;
                    "recovered"
                }
                Err(CheckFailure::Upstream(failure)) => {
                    match failure {
                        UpstreamFailure::Account(AccountProblem::VerificationRequired { url }) => {
                            stored.mark_verification_required(url.clone(), "verification-required")
                        }
                        UpstreamFailure::Account(AccountProblem::AuthInvalid { .. }) => {
                            stored.enabled = false;
                            stored.mark_ineligible("credential rejected during verification check");
                        }
                        UpstreamFailure::Account(
                            AccountProblem::Ineligible { .. } | AccountProblem::Banned { .. },
                        ) => stored.mark_ineligible("upstream reports account ineligible"),
                        UpstreamFailure::RateLimited { reason, reset_in }
                            if !matches!(
                                reason,
                                RateLimitReason::ModelCapacityExhausted
                                    | RateLimitReason::ServerError
                            ) =>
                        {
                            let wait = backoff_for(*reason, failures, *reset_in, 0.5);
                            stored.mark_rate_limited(&pool, now + wait.as_millis() as i64);
                        }
                        _ => {}
                    }
                    match failure {
                        UpstreamFailure::Account(AccountProblem::VerificationRequired {
                            ..
                        }) => "verification_required",
                        UpstreamFailure::Account(_) => "account_rejected",
                        UpstreamFailure::RateLimited { .. } => "rate_limited",
                        UpstreamFailure::Network => "network_error",
                        UpstreamFailure::Server { .. } => "upstream_server_error",
                        UpstreamFailure::BadRequest { .. } => "invalid_request",
                        UpstreamFailure::Refused { .. } => "permission_denied",
                    }
                }
                Err(CheckFailure::Inconclusive(reason)) => reason,
            };
            stored.verification_check.id = None;
            stored.verification_check.started_at =
                Some(account.verification_check.started_at.unwrap_or(now));
            stored.verification_check.lease_until = None;
            stored.verification_check.last_checked_at = Some(now);
            stored.verification_check.last_result = Some(result.into());
            if outcome.is_err() {
                stored.verification_check.failures = failures;
                stored.verification_check.recovered_at = None;
                stored.verification_recheck_at = Some(
                    (now + delay_ms(failures)).max(stored.reset_at_for(&pool, now).unwrap_or(0)),
                );
            }
            updated = Some(view(stored, now));
            true
        })?;
        lease.active = false;
        if outcome.is_ok() && updated.as_ref().is_some_and(|v| v.state == "recovered") {
            self.router
                .record_verification_recovery(&credential_id, now_ms());
            self.router.clear_quota_failures(&credential_id, &pool);
        }
        let result = updated.unwrap_or_else(|| {
            self.accounts
                .snapshot()
                .accounts
                .iter()
                .find(|a| a.credential_id() == credential_id)
                .map(|a| view(a, now_ms()))
                .unwrap_or(VerificationView {
                    state: "removed",
                    next_check_at: None,
                    last_checked_at: None,
                    last_result: None,
                    recovered_at: None,
                })
        });
        tracing::info!(account_id = %&credential_id[..8], state = result.state, result = ?result.last_result,
            next_check_at = ?result.next_check_at, "verification check finished");
        Ok(result)
    }

    /// One scan; awaited checks cannot accumulate across timer ticks.
    pub async fn recheck_due_verifications(&self) -> Result<(), StoreError> {
        self.accounts.reload()?;
        let now = now_ms();
        // Migrate old hour-long verification windows to the new five-minute cap.
        self.accounts.mutate(|storage| {
            let mut changed = false;
            for account in &mut storage.accounts {
                if account.verification_required
                    && account.verification_check.failures == 0
                    && !account.verification_check.running(now)
                    && account
                        .verification_recheck_at
                        .is_some_and(|at| at > now + 300_000)
                {
                    account.verification_recheck_at = Some(now + 300_000);
                    changed = true;
                }
            }
            changed
        })?;
        for account in self.accounts.snapshot().accounts {
            if account.verification_required
                && account.enabled
                && !account.account_ineligible
                && account
                    .verification_recheck_at
                    .is_none_or(|at| at <= now_ms())
            {
                match self
                    .recheck_verification(&account.credential_id(), false)
                    .await
                {
                    Ok(_) | Err(CheckError::Busy | CheckError::Blocked | CheckError::NotFound) => {}
                    Err(error) => tracing::warn!(%error, "verification scan check failed"),
                }
            }
        }
        Ok(())
    }

    async fn check_generation(
        &self,
        prepared: &super::dispatch::Prepared,
        account: &Account,
    ) -> Result<AttemptSuccess, CheckFailure> {
        let mut last = CheckFailure::Inconclusive("no_endpoint");
        for endpoint in self.endpoints() {
            match self.attempt_once(prepared, account, &endpoint).await {
                Ok(mut success) => {
                    validate_stream(&mut success.response.body).await?;
                    return Ok(success);
                }
                Err(AttemptError::Upstream(failure)) => {
                    let try_next = matches!(
                        *failure,
                        UpstreamFailure::Network
                            | UpstreamFailure::Server { .. }
                            | UpstreamFailure::Refused { .. }
                    );
                    last = CheckFailure::Upstream(*failure);
                    if !try_next {
                        break;
                    }
                }
                Err(AttemptError::Fatal(_)) => {
                    return Err(CheckFailure::Inconclusive("invalid_request"));
                }
            }
        }
        Err(last)
    }
}

enum CheckFailure {
    Upstream(UpstreamFailure),
    Inconclusive(&'static str),
}

async fn validate_stream(
    body: &mut crate::upstream::transport::BodyStream,
) -> Result<(), CheckFailure> {
    let mut decoder = SseDecoder::new();
    let mut visible = false;
    let mut finished = false;
    let mut total = 0;
    let mut inspect = |decoded: crate::upstream::sse::Decoded| -> Result<(), CheckFailure> {
        if decoded.malformed > 0 {
            return Err(CheckFailure::Inconclusive("malformed_stream"));
        }
        for event in decoded.events {
            if event.payload.get("error").is_some() {
                return Err(CheckFailure::Inconclusive("stream_error"));
            }
            if let Some(candidates) = event.payload.get("candidates").and_then(|v| v.as_array()) {
                for candidate in candidates {
                    finished |=
                        candidate.get("finishReason").and_then(|v| v.as_str()) == Some("STOP");
                    if let Some(parts) = candidate
                        .pointer("/content/parts")
                        .and_then(|v| v.as_array())
                    {
                        visible |= parts.iter().any(|p| {
                            p.get("thought").and_then(|v| v.as_bool()) != Some(true)
                                && p.get("text")
                                    .and_then(|v| v.as_str())
                                    .is_some_and(|s| !s.trim().is_empty())
                        });
                    }
                }
            }
        }
        Ok(())
    };
    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|_| CheckFailure::Inconclusive("stream_read_error"))?;
        total += chunk.len();
        if total > 1024 * 1024 {
            return Err(CheckFailure::Inconclusive("response_too_large"));
        }
        inspect(
            decoder
                .push(&chunk)
                .map_err(|_| CheckFailure::Inconclusive("malformed_stream"))?,
        )?;
    }
    inspect(
        decoder
            .finish()
            .map_err(|_| CheckFailure::Inconclusive("malformed_stream"))?,
    )?;
    if visible && finished {
        Ok(())
    } else {
        Err(CheckFailure::Inconclusive("incomplete_stream"))
    }
}

#[cfg(test)]
mod tests;
