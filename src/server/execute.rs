//! Request execution shared by the protocol handlers.
//!
//! Both `/v1/chat/completions` and `/v1/responses` dispatch the same way and need
//! the same three things: read enough of the upstream response to know whether a
//! real answer is coming, retry when it is not, and record what happened. Keeping
//! that here rather than in either handler is what stops the two from drifting —
//! an empty-response retry that exists on one route and not the other is a bug
//! nobody would think to look for.

use serde::Deserialize as _;
use serde_json::Value;

use crate::engine::dispatch::Prepared;
use crate::observ::{audit, metrics};
use crate::transform::ir::{GenerateContentResponse, UsageMetadata};
use crate::transform::response::ResponseAccumulator;
use crate::upstream::sse::SseDecoder;
use crate::upstream::transport::BodyStream;

use super::SharedState;
use super::error::{self, ApiError};

/// Events read before committing to a response.
///
/// Reached in normal operation only by a model that emits many empty carrier
/// events before its first token.
pub(super) const PRELUDE_MAX_EVENTS: usize = 32;

/// Deadline for the prelude. Short: the first real event normally arrives in
/// well under a second, and this exists to bound the pathological case.
const PRELUDE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// What the prelude established.
pub struct Begun {
    /// Events read before committing, to be replayed into whichever consumer.
    pub events: Vec<Value>,
    /// The remaining body, or `None` when the stream ended during the prelude.
    pub tail: Option<BodyStream>,
    /// Upstream attempts this took, including the successful one.
    pub attempts: u32,
    /// Which account served it, for the audit log.
    pub account_id: String,
}

/// Dispatch and read until a real answer is known to be coming.
///
/// Retries an empty response up to the configured limit. Returning an error here
/// is safe: nothing has been written to the client yet.
pub async fn begin(
    state: &SharedState,
    prepared: &Prepared,
    mode: crate::config::ExhaustedErrorMode,
) -> Result<Begun, ApiError> {
    // At least one attempt, whatever the retry budget says.
    let max_attempts = state
        .engine
        .config
        .routing
        .max_empty_response_retries
        .saturating_add(1)
        .max(1);

    for attempt in 1..=max_attempts {
        let call = state
            .engine
            .dispatch(prepared)
            .await
            .map_err(|e| error::from_dispatch(e, mode))?;

        let (events, tail, has_content) = read_prelude(call.response.body).await;

        if has_content {
            return Ok(Begun {
                events,
                tail,
                attempts: call.attempts,
                account_id: call.credential_id,
            });
        }

        if attempt < max_attempts {
            tracing::warn!(
                attempt,
                max_attempts,
                model = %prepared.resolved.wire_model,
                "upstream returned no content; retrying"
            );
            continue;
        }

        // Out of retries. Hand back what we have rather than failing: an empty
        // completion truthfully describes an empty response, and a client can act
        // on it. An error here would be indistinguishable from a gateway fault.
        tracing::error!(
            attempts = max_attempts,
            model = %prepared.resolved.wire_model,
            "upstream returned no content after every retry"
        );
        return Ok(Begun {
            events,
            tail,
            attempts: call.attempts,
            account_id: call.credential_id,
        });
    }

    unreachable!("the loop returns on its final iteration")
}

/// Read initial events, stopping as soon as a real answer is visible.
///
/// Returns the events, the remaining body (`None` if the stream ended), and
/// whether client-visible content appeared.
pub async fn read_prelude(mut body: BodyStream) -> (Vec<Value>, Option<BodyStream>, bool) {
    let mut decoder = SseDecoder::new();
    let mut accumulator = ResponseAccumulator::new();
    let mut events: Vec<Value> = Vec::new();
    let deadline = tokio::time::Instant::now() + PRELUDE_TIMEOUT;

    loop {
        if tokio::time::Instant::now() >= deadline {
            // Out of time. Commit with what we have; a model this slow to start
            // will not be helped by waiting longer.
            return (events, Some(body), accumulator.has_client_content());
        }

        match tokio::time::timeout_at(deadline, futures::StreamExt::next(&mut body)).await {
            // A chunk arrived.
            Ok(Some(Ok(bytes))) => {
                let decoded = match decoder.push(&bytes) {
                    Ok(decoded) => decoded,
                    // A decoder that cannot make progress will not recover.
                    Err(_) => return (events, None, accumulator.has_client_content()),
                };

                for event in decoded.events {
                    absorb(&mut accumulator, &event.payload);
                    events.push(event.payload);
                }

                // Both bounds are checked after absorbing, not before reading,
                // because one chunk can carry many events: checking only at the
                // top of the loop would let a single large chunk blow the budget
                // arbitrarily far past its limit.
                if accumulator.has_client_content() {
                    return (events, Some(body), true);
                }
                if events.len() >= PRELUDE_MAX_EVENTS {
                    return (events, Some(body), false);
                }
            }

            // The stream ended, or delivered a transport error. Either way there
            // is nothing more to read, so there is no tail to hand back.
            Ok(Some(Err(_)) | None) => return (events, None, accumulator.has_client_content()),

            // Timed out waiting for the next chunk. The body is still live, so it
            // becomes the tail.
            Err(_) => return (events, Some(body), accumulator.has_client_content()),
        }
    }
}

/// Drain a remaining body into an accumulator.
pub async fn drain_into(accumulator: &mut ResponseAccumulator, mut body: BodyStream) {
    let mut decoder = SseDecoder::new();
    while let Some(chunk) = futures::StreamExt::next(&mut body).await {
        let Ok(bytes) = chunk else {
            break;
        };
        match decoder.push(&bytes) {
            Ok(decoded) => {
                for event in &decoded.events {
                    absorb(accumulator, &event.payload);
                }
            }
            Err(_) => break,
        }
    }
    if let Ok(decoded) = decoder.finish() {
        for event in &decoded.events {
            absorb(accumulator, &event.payload);
        }
    }
}

/// Deserialize one payload into the accumulator, ignoring anything unparseable.
pub fn absorb(accumulator: &mut ResponseAccumulator, payload: &Value) {
    if let Ok(response) = GenerateContentResponse::deserialize(payload) {
        accumulator.absorb(&response);
    }
}

/// Write one finished request to metrics and the audit log.
///
/// Takes loose arguments rather than a struct, because the call sites know
/// different amounts: an early error has no account and no usage to report.
#[allow(clippy::too_many_arguments)]
pub fn record(
    state: &SharedState,
    requested_model: &str,
    wire_model: &str,
    account_id: &str,
    stream: bool,
    status: u16,
    attempts: u32,
    usage: Option<&UsageMetadata>,
    elapsed: std::time::Duration,
) {
    let outcome = metrics::Outcome::of(status);
    // The label is the *wire* model, and only when it is one the catalogue
    // knows. Labelling with what the client asked for would let a client mint
    // unbounded series by sending arbitrary names, and a metric whose cardinality
    // is controlled by a caller is a way to take down the metrics endpoint.
    let label = bounded_model_label(wire_model);
    metrics::record_request(label, outcome, elapsed);
    if attempts > 0 {
        metrics::record_attempts(label, attempts);
    }
    if !account_id.is_empty() {
        metrics::record_account_selected(account_id);
    }
    if let Some(usage) = usage {
        metrics::record_tokens(label, "prompt", usage.prompt_tokens());
        metrics::record_tokens(label, "completion", usage.candidates_tokens());
        metrics::record_tokens(label, "cached", usage.cached_content_token_count);
        metrics::record_tokens(label, "reasoning", usage.thoughts_token_count);
    }

    let Some(log) = state.observability.audit() else {
        return;
    };
    log.record(audit::Record {
        status,
        outcome: outcome_name(outcome).to_string(),
        attempts,
        prompt_tokens: usage.map(UsageMetadata::prompt_tokens).unwrap_or(0),
        completion_tokens: usage.map(UsageMetadata::candidates_tokens).unwrap_or(0),
        cached_tokens: usage.map(|u| u.cached_content_token_count).unwrap_or(0),
        reasoning_tokens: usage.map(|u| u.thoughts_token_count).unwrap_or(0),
        latency_ms: elapsed.as_millis() as i64,
        stream,
        ..audit::Record::new(account_id, requested_model)
    });
}

/// Bound a model name to the catalogue, so metric cardinality is fixed.
///
/// An unknown model is reported as `other`. That loses the distinction between
/// two unknown models, which is the correct trade: the alternative is a metric
/// whose series count is chosen by whoever sends the requests.
pub fn bounded_model_label(wire_model: &str) -> &str {
    if wire_model.is_empty() {
        return "unknown";
    }
    // Collapse the wire model back to its catalogue entry, so the four tiered
    // spellings of one Flash model are one series rather than four.
    crate::registry::models::base_for_wire(wire_model).unwrap_or("other")
}

/// The audit log stores the outcome name rather than an enum, so the table stays
/// readable from a SQL prompt.
pub fn outcome_name(outcome: metrics::Outcome) -> &'static str {
    match outcome {
        metrics::Outcome::Ok => "ok",
        metrics::Outcome::ClientError => "client_error",
        metrics::Outcome::UpstreamError => "upstream_error",
        metrics::Outcome::Unavailable => "unavailable",
    }
}

/// The configured error mode, read once per request.
pub fn exhausted_mode(state: &SharedState) -> crate::config::ExhaustedErrorMode {
    state.engine.config.routing.exhausted_error_mode
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metric_labels_collapse_to_the_catalogue_entry() {
        // A client controls the model name, so labelling metrics with it would
        // let one client mint unbounded series.
        assert_eq!(bounded_model_label("gemini-3.8-flash-medium"), "gemini-3.8-flash");
        assert_eq!(bounded_model_label("claude-opus-4-6-thinking"), "claude-opus-4-6-thinking");
        assert_eq!(bounded_model_label("client-invented-model"), "other");
        assert_eq!(bounded_model_label(""), "unknown");
    }

    #[test]
    fn tier_variants_share_one_series() {
        // Traffic by model is what a dashboard wants, not by model and tier.
        let flash: std::collections::BTreeSet<&str> = [
            "gemini-3.8-flash-low",
            "gemini-3.8-flash-medium",
            "gemini-3.8-flash-high",
            "gemini-3.8-flash",
        ]
        .into_iter()
        .map(bounded_model_label)
        .collect();
        assert_eq!(flash.len(), 1);
    }

    #[test]
    fn an_unresolvable_label_collapses_to_one_series() {
        // Many distinct unknown names must not become many distinct series.
        let labels: std::collections::BTreeSet<String> = (0..100)
            .map(|index| bounded_model_label(&format!("made-up-{index}")).to_string())
            .collect();
        assert_eq!(labels.len(), 1);
    }

}
