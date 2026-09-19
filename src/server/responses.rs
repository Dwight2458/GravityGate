//! `POST /v1/responses`.
//!
//! The Responses API. Requests are converted into the Chat Completions shape and
//! dispatched through the same pipeline, so signature replay, account rotation,
//! and the empty-response retry all behave identically on both routes. Only the
//! rendering differs.
//!
//! Streaming here uses named SSE events with a monotonic `sequence_number`, and
//! the response object is repeated in `response.created`, `response.in_progress`,
//! and `response.completed`. That is the protocol, not redundancy: a client that
//! only sees the terminal event has the whole picture, and one that streams does
//! not have to accumulate it.

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures::StreamExt;
use serde::Deserialize as _;

use crate::engine::dispatch::Prepared;
use crate::transform::ir::GenerateContentResponse;
use crate::transform::responses::{
    ResponsesRequest, ResponsesResponse, ResponsesStream, format_event, new_response_id,
    to_chat_request, to_response,
};
use crate::transform::response::ResponseOptions;
use crate::upstream::sse::SseDecoder;

use super::SharedState;
use super::error::{self, ApiError};
use super::execute;

pub async fn create(
    State(state): State<SharedState>,
    Json(request): Json<ResponsesRequest>,
) -> Result<Response, ApiError> {
    let mode = execute::exhausted_mode(&state);
    let started = std::time::Instant::now();

    let chat = to_chat_request(&request).map_err(ApiError::invalid_request)?;

    let prepared = match state.engine.prepare(&chat) {
        Ok(prepared) => prepared,
        Err(error) => {
            let api = error::from_dispatch(error, mode);
            execute::record(
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

    let outcome = if request.stream {
        stream(&state, &prepared, &request, started).await
    } else {
        buffered(&state, &prepared, &request, mode, started).await
    };

    if let Err(api) = &outcome {
        execute::record(
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

/// The buffered response.
async fn buffered(
    state: &SharedState,
    prepared: &Prepared,
    request: &ResponsesRequest,
    mode: crate::config::ExhaustedErrorMode,
    started: std::time::Instant,
) -> Result<Response, ApiError> {
    let begun = execute::begin(state, prepared, mode).await?;

    let mut accumulator = crate::transform::response::ResponseAccumulator::new();
    for payload in &begun.events {
        execute::absorb(&mut accumulator, payload);
    }
    if let Some(tail) = begun.tail {
        execute::drain_into(&mut accumulator, tail).await;
    }

    let options = options_for(prepared, request);
    let usage = accumulator.usage().cloned().unwrap_or_default();
    let finish_reason = accumulator.finish_reason().map(str::to_string);
    let parts = accumulator.parts().to_vec();

    let response: ResponsesResponse = to_response(
        &parts,
        &usage,
        finish_reason.as_deref(),
        &options,
        request,
    );

    execute::record(
        state,
        &prepared.requested_model,
        &prepared.resolved.wire_model,
        &begun.account_id,
        false,
        200,
        begun.attempts,
        Some(&usage),
        started.elapsed(),
    );

    Ok((StatusCode::OK, Json(response)).into_response())
}

/// The streaming response.
async fn stream(
    state: &SharedState,
    prepared: &Prepared,
    request: &ResponsesRequest,
    started: std::time::Instant,
) -> Result<Response, ApiError> {
    let mode = execute::exhausted_mode(state);
    let begun = execute::begin(state, prepared, mode).await?;

    // Everything the stream touches is captured up front: it outlives the
    // handler's borrows.
    let attempts = begun.attempts;
    let account_id = begun.account_id.clone();
    let model = prepared.requested_model.clone();
    let wire_model = prepared.resolved.wire_model.clone();
    let session_key = prepared.session_key.clone();
    let echoed_model = request.model.clone();
    let engine = state.clone();

    let response_id = new_response_id();
    let created_at = now_secs();

    let body = async_stream::stream! {
        let mut stream = ResponsesStream::new(response_id, echoed_model, created_at);

        for payload in &begun.events {
            if let Ok(parsed) = GenerateContentResponse::deserialize(payload) {
                for event in stream.on_response(&parsed) {
                    yield Ok::<Bytes, std::io::Error>(Bytes::from(format_event(&event)));
                }
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
                            if let Ok(parsed) = GenerateContentResponse::deserialize(&event.payload)
                            {
                                for rendered in stream.on_response(&parsed) {
                                    yield Ok(Bytes::from(format_event(&rendered)));
                                }
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            if let Ok(decoded) = decoder.finish() {
                for event in &decoded.events {
                    if let Ok(parsed) = GenerateContentResponse::deserialize(&event.payload) {
                        for rendered in stream.on_response(&parsed) {
                            yield Ok(Bytes::from(format_event(&rendered)));
                        }
                    }
                }
            }
        }

        for event in stream.finish() {
            yield Ok(Bytes::from(format_event(&event)));
        }

        engine.engine.sessions.complete_execution(&session_key);

        execute::record(
            &engine,
            &model,
            &wire_model,
            &account_id,
            true,
            200,
            attempts,
            stream.usage(),
            started.elapsed(),
        );
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(body))
        .expect("the response head is well-formed"))
}

fn options_for(prepared: &Prepared, request: &ResponsesRequest) -> ResponseOptions {
    ResponseOptions {
        // The response id shares the completion id so the two routes produce
        // consistent identifiers for the same conversation.
        completion_id: crate::transform::response::new_completion_id(),
        model: request.model.clone(),
        created: now_secs(),
        family: prepared.resolved.family,
        reasoning_field: crate::config::ReasoningField::ReasoningContent,
        // The Responses protocol carries usage on the terminal event
        // unconditionally, so there is nothing to opt into.
        include_usage: true,
        session_key: prepared.session_key.clone(),
    }
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use crate::transform::responses::ResponsesRequest;
    use serde_json::json;

    fn request(value: Value) -> ResponsesRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn the_response_id_is_generated_per_request() {
        assert_ne!(new_response_id(), new_response_id());
    }

    #[test]
    fn the_request_is_converted_before_dispatch() {
        // A malformed request must be rejected without reaching the upstream.
        let error = to_chat_request(&request(json!({ "model": "m" }))).unwrap_err();
        assert!(error.contains("input"), "got {error}");
    }

}
