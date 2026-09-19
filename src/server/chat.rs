//! `POST /v1/chat/completions`.
//!
//! Both the streaming and buffered forms share one dispatch and one prelude, and
//! differ only in how the events are consumed. That sharing is deliberate: it is
//! the only way the two can be guaranteed to describe the same thing.
//!
//! The prelude is the interesting part. Before committing to a response the
//! handler reads upstream events until it knows whether a real answer is coming:
//!
//! - **An empty response gets retried.** A stream that ends having produced only
//!   signatures and finish reasons is a failure, and retrying it is only
//!   possible before the response head is written. Committing first and then
//!   discovering the emptiness would mean emitting a 200 with nothing in it.
//! - **A transport failure is still a real HTTP status.** Because the head is
//!   inspected before any bytes are written, an upstream 429 becomes a client
//!   429 with `Retry-After`, not a broken stream.
//!
//! The cost is a short delay before the first byte reaches the client. It is
//! bounded by an event count and a deadline so an unusually chatty stream cannot
//! stall indefinitely.

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures::StreamExt;
use serde::Deserialize as _;
use serde_json::Value;

use crate::engine::dispatch::Prepared;
use crate::observ::{audit, metrics};
use crate::transform::ir::{GenerateContentResponse, UsageMetadata};
use crate::transform::openai::{ChatCompletion, ChatCompletionChunk, ChatCompletionRequest};
use crate::transform::response::{ResponseAccumulator, ResponseOptions, StreamTranslator};
use crate::upstream::sse::SseDecoder;
use crate::upstream::transport::BodyStream;

use super::SharedState;
use super::error::{self, ApiError};

/// Events read before committing to a response.
///
/// Reached in normal operation only by a model that emits many empty carrier
/// events before its first token.
const PRELUDE_MAX_EVENTS: usize = 32;

/// Deadline for the prelude. Short: the first real event normally arrives in
/// well under a second, and this exists to bound the pathological case.
const PRELUDE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

pub async fn completions(
    State(state): State<SharedState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let mode = state.engine.config.routing.exhausted_error_mode;
    let started = std::time::Instant::now();

    // A request that never reached dispatch still belongs in the audit log: a
    // spike in 400s is exactly what an operator goes looking for.
    let prepared = match state.engine.prepare(&request) {
        Ok(prepared) => prepared,
        Err(error) => {
            let api = error::from_dispatch(error, mode);
            record(
                &state,
                &request.model,
                "",
                "",
                request.stream,
                api.status.as_u16(),
                0,
                None,
                started.elapsed(),
            );
            return Err(api);
        }
    };
    let options = ResponseOptions {
        include_usage: request
            .stream_options
            .as_ref()
            .and_then(|options| options.include_usage)
            .unwrap_or(false),
        ..ResponseOptions::new(
            request.model.clone(),
            prepared.resolved.family,
            prepared.session_key.clone(),
        )
    };

    let outcome = if request.stream {
        stream(&state, &prepared, options, mode, started).await
    } else {
        buffered(&state, &prepared, options, mode, started).await
    };

    if let Err(api) = &outcome {
        record(
            &state,
            &prepared.requested_model,
            &prepared.resolved.wire_model,
            "",
            prepared.stream,
            api.status.as_u16(),
            0,
            None,
            started.elapsed(),
        );
    }
    outcome
}

/// Write one finished request to metrics and the audit log.
///
/// Takes loose arguments rather than a struct, because the call sites know
/// different amounts: an early error has no account and no usage to report.
#[allow(clippy::too_many_arguments)]
fn record(
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
fn bounded_model_label(wire_model: &str) -> &str {
    if wire_model.is_empty() {
        return "unknown";
    }
    // Collapse the wire model back to its catalogue entry, so the four tiered
    // spellings of one Flash model are one series rather than four.
    crate::registry::models::base_for_wire(wire_model).unwrap_or("other")
}

/// The audit log stores the outcome name rather than an enum, so the table stays
/// readable from a SQL prompt.
fn outcome_name(outcome: metrics::Outcome) -> &'static str {
    match outcome {
        metrics::Outcome::Ok => "ok",
        metrics::Outcome::ClientError => "client_error",
        metrics::Outcome::UpstreamError => "upstream_error",
        metrics::Outcome::Unavailable => "unavailable",
    }
}

/// The streaming response.
async fn stream(
    state: &SharedState,
    prepared: &Prepared,
    options: ResponseOptions,
    mode: crate::config::ExhaustedErrorMode,
    started: std::time::Instant,
) -> Result<Response, ApiError> {
    let begun = begin(state, prepared, mode).await?;
    note_attempts(&begun, &prepared.resolved.wire_model);
    let session_key = options.session_key.clone();
    let attempts = begun.attempts;
    let account_id = begun.account_id.clone();

    // The translator and the tail both need the engine's cache, which outlives
    // the stream, so the Arc is moved in rather than borrowed.
    let engine = state.clone();
    let model = prepared.requested_model.clone();
    let wire_model = prepared.resolved.wire_model.clone();

    let body = async_stream::stream! {
        let mut translator = StreamTranslator::new(options);

        for payload in &begun.events {
            for chunk in translator.on_payload(payload, Some(&engine.engine.signatures)) {
                yield sse_chunk(&chunk);
            }
        }

        if let Some(mut tail) = begun.tail {
            let mut decoder = SseDecoder::new();
            while let Some(chunk) = tail.next().await {
                let Ok(bytes) = chunk else {
                    break;
                };
                match decoder.push(&bytes) {
                    Ok(decoded) => {
                        for event in &decoded.events {
                            for chunk in
                                translator.on_payload(&event.payload, Some(&engine.engine.signatures))
                            {
                                yield sse_chunk(&chunk);
                            }
                        }
                    }
                    // A decoder that cannot make progress will not recover.
                    Err(_) => break,
                }
            }
            if let Ok(decoded) = decoder.finish() {
                for event in &decoded.events {
                    for chunk in translator.on_payload(&event.payload, Some(&engine.engine.signatures)) {
                        yield sse_chunk(&chunk);
                    }
                }
            }
        }

        for chunk in translator.finish() {
            yield sse_chunk(&chunk);
        }
        yield Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: [DONE]\n\n"));

        // Mark the turn complete so the next request in this conversation counts
        // a further step, which is what the upstream's metadata tracks.
        engine.engine.sessions.complete_execution(&session_key);

        // Recorded here rather than before the stream starts: the latency a
        // client experiences runs to the last byte, not the first, and usage is
        // only complete on the final event.
        record(
            &engine,
            &model,
            &wire_model,
            &account_id,
            true,
            200,
            attempts,
            translator.usage(),
            started.elapsed(),
        );
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        // Tells nginx and friends not to buffer, which would defeat streaming.
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(body))
        .expect("the response head is well-formed"))
}

/// The buffered response.
async fn buffered(
    state: &SharedState,
    prepared: &Prepared,
    options: ResponseOptions,
    mode: crate::config::ExhaustedErrorMode,
    started: std::time::Instant,
) -> Result<Response, ApiError> {
    let begun = begin(state, prepared, mode).await?;
    note_attempts(&begun, &prepared.resolved.wire_model);

    let mut accumulator = ResponseAccumulator::new();
    for payload in &begun.events {
        absorb(&mut accumulator, payload);
    }

    if let Some(mut tail) = begun.tail {
        let mut decoder = SseDecoder::new();
        while let Some(chunk) = tail.next().await {
            let Ok(bytes) = chunk else {
                break;
            };
            match decoder.push(&bytes) {
                Ok(decoded) => {
                    for event in &decoded.events {
                        absorb(&mut accumulator, &event.payload);
                    }
                }
                Err(_) => break,
            }
        }
        if let Ok(decoded) = decoder.finish() {
            for event in &decoded.events {
                absorb(&mut accumulator, &event.payload);
            }
        }
    }

    let session_key = options.session_key.clone();
    let usage = accumulator.usage().cloned();
    let completion: ChatCompletion =
        accumulator.into_completion(&options, Some(&state.engine.signatures));
    state.engine.sessions.complete_execution(&session_key);

    record(
        state,
        &prepared.requested_model,
        &prepared.resolved.wire_model,
        &begun.account_id,
        false,
        200,
        begun.attempts,
        usage.as_ref(),
        started.elapsed(),
    );

    Ok((StatusCode::OK, Json(completion)).into_response())
}

/// Log when a request needed more than one attempt.
///
/// A retry is invisible from the client's side, which means the only signal that
/// an account is flaky is this line appearing in the log.
fn note_attempts(begun: &Begun, wire_model: &str) {
    if begun.attempts > 1 {
        tracing::warn!(
            attempts = begun.attempts,
            wire_model,
            "request succeeded after retrying"
        );
    }
}

/// Deserialize one payload into the accumulator, ignoring anything unparseable.
fn absorb(accumulator: &mut ResponseAccumulator, payload: &Value) {
    if let Ok(response) = GenerateContentResponse::deserialize(payload) {
        accumulator.absorb(&response);
    }
}

/// What the prelude established.
struct Begun {
    /// Events read before committing, to be replayed into whichever consumer.
    events: Vec<Value>,
    /// The remaining body, or `None` when the stream ended during the prelude.
    tail: Option<BodyStream>,
    /// Upstream attempts this took, including the successful one.
    attempts: u32,
    /// Which account served it, for the audit log.
    account_id: String,
}

/// Dispatch and read until a real answer is known to be coming.
///
/// Retries an empty response up to the configured limit. Returning an error here
/// is safe: nothing has been written to the client yet.
async fn begin(
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
        // completion truthfully describes an empty response, and a client can
        // act on it. An error here would be indistinguishable from a gateway
        // fault.
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
async fn read_prelude(mut body: BodyStream) -> (Vec<Value>, Option<BodyStream>, bool) {
    let mut decoder = SseDecoder::new();
    let mut accumulator = ResponseAccumulator::new();
    let mut events: Vec<Value> = Vec::new();
    let deadline = tokio::time::Instant::now() + PRELUDE_TIMEOUT;

    loop {
        if tokio::time::Instant::now() >= deadline {
            // Out of time. Commit with what we have; a model this slow to start
            // is not going to be helped by waiting longer.
            return (events, Some(body), accumulator.has_client_content());
        }

        match tokio::time::timeout_at(deadline, body.next()).await {
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

            // Timed out waiting for the next chunk. The body is still live, so
            // it becomes the tail.
            Err(_) => return (events, Some(body), accumulator.has_client_content()),
        }
    }
}

/// Encode one chunk as an SSE frame.
fn sse_chunk(chunk: &ChatCompletionChunk) -> Result<Bytes, std::io::Error> {
    match serde_json::to_string(chunk) {
        Ok(json) => Ok(Bytes::from(format!("data: {json}\n\n"))),
        // A chunk that cannot be serialised is a bug, not a client problem.
        // Skipping it keeps the stream alive rather than truncating it.
        Err(error) => {
            tracing::error!(%error, "could not serialise a chunk; skipping");
            Ok(Bytes::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn body_of(events: &[Value]) -> BodyStream {
        let text: String = events
            .iter()
            .map(|event| format!("data: {}\n\n", serde_json::to_string(event).unwrap()))
            .collect();
        Box::pin(futures::stream::iter(vec![Ok(Bytes::from(text))]))
    }

    #[tokio::test]
    async fn the_prelude_stops_at_the_first_content() {
        // Text in the first event: the reader must not wait for more.
        let events = vec![json!({
            "candidates": [{ "content": { "parts": [{ "text": "hello" }] } }]
        })];

        let (read, tail, has_content) = read_prelude(body_of(&events)).await;
        assert!(has_content);
        assert_eq!(read.len(), 1);
        assert!(tail.is_some(), "the rest of the stream is still available");
    }

    #[tokio::test]
    async fn the_prelude_skips_carrier_events_and_finds_later_content() {
        // The Claude shape: an empty text carrier first, content second.
        let events = vec![
            json!({ "candidates": [{ "content": { "parts": [{ "text": "" }] } }] }),
            json!({ "candidates": [{ "content": { "parts": [{ "thought": true, "text": "ok" }] } }] }),
        ];

        let (read, _, has_content) = read_prelude(body_of(&events)).await;
        assert!(has_content, "content in the second event must be found");
        assert_eq!(read.len(), 2, "the carrier is kept for replay");
    }

    #[tokio::test]
    async fn a_signature_only_stream_reports_no_content() {
        let events = vec![json!({
            "candidates": [{
                "content": { "parts": [{ "text": "", "thoughtSignature": "S".repeat(80) }] },
                "finishReason": "STOP"
            }]
        })];

        let (read, tail, has_content) = read_prelude(body_of(&events)).await;
        assert!(!has_content, "signatures are not an answer");
        assert_eq!(read.len(), 1, "the carrier is kept so it can be replayed");
        assert!(
            tail.is_none(),
            "the stream ended while reading, so nothing remains"
        );
    }

    #[tokio::test]
    async fn an_empty_stream_reports_no_content_and_no_tail() {
        let stream: BodyStream = Box::pin(futures::stream::iter(vec![]));
        let (read, tail, has_content) = read_prelude(stream).await;
        assert!(read.is_empty());
        assert!(tail.is_none(), "an ended stream has nothing left to read");
        assert!(!has_content);
    }

    #[tokio::test]
    async fn the_prelude_stops_at_its_event_budget() {
        // A stream of pure carriers with no content must not be drained. The
        // budget bounds how long the prelude waits, so this uses one event per
        // chunk: a single chunk carrying many events is absorbed whole, which is
        // bounded by the chunk rather than by the budget.
        let carriers: Vec<Value> = (0..PRELUDE_MAX_EVENTS * 3)
            .map(|_| json!({ "candidates": [{ "content": { "parts": [{ "text": "" }] } }] }))
            .collect();

        let chunks: Vec<_> = carriers
            .iter()
            .map(|event| {
                Ok(Bytes::from(format!(
                    "data: {}

",
                    serde_json::to_string(event).unwrap()
                )))
            })
            .collect();
        let stream: BodyStream = Box::pin(futures::stream::iter(chunks));

        let (read, tail, has_content) = read_prelude(stream).await;
        assert!(!has_content);
        assert!(
            read.len() <= PRELUDE_MAX_EVENTS,
            "the budget must bound how far the prelude reads, got {}",
            read.len()
        );
        assert!(tail.is_some(), "the unread remainder stays available");
    }

    #[tokio::test]
    async fn a_second_chunk_is_kept_for_the_tail() {
        // The prelude must not swallow events that arrive in a later chunk.
        let first = b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"a\"}]}}]}\n\n";
        let second = b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"b\"}]}}]}\n\n";
        let stream: BodyStream = Box::pin(futures::stream::iter(vec![
            Ok(Bytes::from_static(first)),
            Ok(Bytes::from_static(second)),
        ]));

        let (read, tail, has_content) = read_prelude(stream).await;
        assert!(has_content);
        assert_eq!(read.len(), 1);

        // Draining the tail must yield the second event.
        let mut tail = tail.unwrap();
        let mut decoder = SseDecoder::new();
        let mut tail_events = 0;
        while let Some(chunk) = tail.next().await {
            let bytes = chunk.unwrap();
            tail_events += decoder.push(&bytes).unwrap().events.len();
        }
        assert_eq!(tail_events, 1, "the second event belongs to the tail");
    }

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

    #[test]
    fn a_chunk_frames_as_sse() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-x".into(),
            object: "chat.completion.chunk",
            created: 1,
            model: "m".into(),
            choices: Vec::new(),
            usage: None,
        };
        let bytes = sse_chunk(&chunk).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.starts_with("data: {"));
        assert!(text.ends_with("\n\n"), "SSE frames end with a blank line");
        assert!(text.contains("chatcmpl-x"));
    }
}
