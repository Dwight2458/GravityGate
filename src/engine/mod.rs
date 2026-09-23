//! Request execution: credentials, project context, envelope, upstream call.
//!
//! This is the seam between the account pool and the wire. Everything above it
//! deals in accounts and models; everything below it deals in envelopes and
//! HTTP. Keeping it in one place is what makes the routing layer testable
//! without a network.

pub mod credentials;
pub mod dispatch;
pub mod retry;

use std::time::Duration;

use serde_json::Value;

use crate::accounts::account::Account;
use crate::accounts::project::{ProjectResolver, describe_tier};
use crate::accounts::router::{AccountRouter, ScoringConfig};
use crate::accounts::store::AccountStore;
use crate::config::Config;
use crate::engine::credentials::CredentialCache;
use crate::oauth::token::OAuthClient;
use crate::registry::live::LiveCatalogue;

use crate::transform::openai::{ChatCompletion, ChatCompletionRequest};
use crate::transform::response::{ResponseOptions, StreamTranslator, to_completion};
use crate::transform::signature_cache::SignatureCache;
use crate::upstream::metadata::SessionStore;
use crate::upstream::sse::SseDecoder;
use crate::upstream::transport::{TransportError, UpstreamClient};

/// Longest probe body we will echo back before truncating.
///
/// Display only: the response is always interpreted from the full decoded
/// stream, so capping this cannot cost an answer. A caller that wants the whole
/// body — `probe --raw` — passes a larger limit to `probe_request`.
pub const PROBE_BODY_LIMIT: usize = 4096;

/// Shared state for the whole gateway.
pub struct Engine {
    pub config: Config,
    pub accounts: AccountStore,
    pub upstream: UpstreamClient,
    pub oauth: OAuthClient,
    pub credentials: CredentialCache,
    pub projects: ProjectResolver,
    pub sessions: SessionStore,
    /// Cached thinking signatures. Shared across requests because a signature
    /// minted on one turn is needed on the next.
    pub signatures: SignatureCache,
    /// Account selection and runtime health. Not persisted: token buckets and
    /// health scores describe this process's recent behaviour, not the account's
    /// durable state.
    pub router: AccountRouter,
    /// The upstream's model list, shared across accounts because membership is
    /// a deployment property rather than an account one.
    pub live_catalogue: LiveCatalogue,
}

impl Engine {
    pub fn new(config: Config, accounts: AccountStore) -> Result<Self, TransportError> {
        let scoring = ScoringConfig {
            token_max: config.accounts.token_bucket_max,
            token_refill_per_min: config.accounts.token_bucket_refill_per_min,
            ..ScoringConfig::default()
        };
        let strategy = config.accounts.strategy;
        Ok(Self {
            config,
            accounts,
            upstream: UpstreamClient::new()?,
            oauth: OAuthClient::new().map_err(|e| match e {
                crate::oauth::token::OAuthError::Transport(t) => t,
                other => TransportError::HttpStatus {
                    status: reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                    body: other.to_string(),
                },
            })?,
            credentials: CredentialCache::new(),
            projects: ProjectResolver::new(),
            sessions: SessionStore::new(),
            signatures: SignatureCache::new(),
            router: AccountRouter::new(strategy, scoring),
            live_catalogue: LiveCatalogue::new(),
        })
    }

    /// Endpoints to try, from config.
    fn endpoints(&self) -> Vec<String> {
        self.config.upstream.endpoints.clone()
    }

    /// Ask the upstream which models an account can reach.
    ///
    /// Uses the first account that can produce a token and a project. The answer
    /// is a property of the deployment rather than of that particular account,
    /// which is why one call is enough.
    pub async fn fetch_available_models(
        &self,
    ) -> Result<Vec<crate::registry::LiveModel>, ProbeError> {
        let snapshot = self.accounts.snapshot();
        let account = snapshot
            .accounts
            .iter()
            .find(|account| account.is_available(crate::accounts::account::now_ms()))
            .ok_or_else(|| {
                ProbeError::Dispatch(crate::engine::dispatch::DispatchError::NoAccounts)
            })?
            .clone();

        let prepared = self.prepare_account(&account).await?;
        let body = serde_json::json!({ "project": prepared.project_id });

        let mut last_error = None;
        for endpoint in self.endpoints() {
            let url = format!(
                "{}{}",
                endpoint,
                crate::upstream::constants::API_FETCH_AVAILABLE_MODELS
            );
            match self
                .upstream
                .post_json_buffered(&url, &prepared.access_token, body.to_string().as_bytes())
                .await
            {
                Ok(response) if response.is_success() => {
                    match crate::registry::live::parse(&response.body_text()) {
                        Ok(models) => return Ok(models),
                        Err(error) => last_error = Some(error.to_string()),
                    }
                }
                Ok(response) => last_error = Some(format!("{} at {endpoint}", response.status)),
                Err(error) => last_error = Some(error.to_string()),
            }
        }

        Err(ProbeError::Unavailable(
            last_error.unwrap_or_else(|| "no endpoints configured".into()),
        ))
    }

    /// The upstream's model list, refreshing it when the cache is cold.
    ///
    /// Never fails. A refresh that does not work falls back to whatever was
    /// cached before; an entirely cold cache yields an empty list and leaves the
    /// caller to fall back to the static catalogue. A model list that empties on
    /// a network hiccup would be worse than a stale one, because clients fetch
    /// it once at startup and keep it.
    pub async fn live_models(&self) -> Vec<crate::registry::LiveModel> {
        if let Some(models) = self.live_catalogue.fresh() {
            return models;
        }

        match self.fetch_available_models().await {
            Ok(models) => {
                tracing::debug!(count = models.len(), "refreshed the live model list");
                self.live_catalogue.store(models.clone());
                models
            }
            Err(error) => {
                if let Some(stale) = self.live_catalogue.stale() {
                    tracing::warn!(%error, "model list refresh failed; serving the cached list");
                    return stale;
                }
                tracing::debug!(%error, "model list unavailable; static catalogue only");
                Vec::new()
            }
        }
    }

    /// Resolve an access token and a project for an account.
    ///
    /// Returns the token, the project id, and whether the project came from the
    /// shared fallback rather than real provisioning.
    pub async fn prepare_account(&self, account: &Account) -> Result<PreparedAccount, ProbeError> {
        let access_token = self
            .credentials
            .access_token(&self.oauth, account)
            .await
            .map_err(ProbeError::OAuth)?;

        let context = self
            .projects
            .resolve(
                &self.upstream,
                &access_token,
                &account.credential_id(),
                &self.endpoints(),
            )
            .await
            .map_err(ProbeError::Project)?;

        let tier = describe_tier(&context);
        Ok(PreparedAccount {
            access_token,
            project_id: context.project_id,
            used_fallback_project: context.used_fallback,
            tier,
            tier_id: context.tier_id,
            paid_tier_id: context.paid_tier_id,
        })
    }

    /// Send one request through the full pipeline and report what happened.
    ///
    /// This runs the real production path — model resolution, request
    /// translation, signature replay, upstream call, SSE decoding, and response
    /// translation — rather than a simplified one, so a probe that looks healthy
    /// is evidence that the gateway path is healthy. It additionally surfaces
    /// the raw upstream body, because when something is wrong that is the only
    /// thing worth reading.
    pub async fn probe(
        &self,
        account: &Account,
        model: &str,
        prompt: &str,
    ) -> Result<ProbeReport, ProbeError> {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": model,
            "messages": [{ "role": "user", "content": prompt }],
            // Not a tight limit: thinking is charged against this budget, and a
            // 64-token cap was consumed entirely by thinking, returning an empty
            // answer with `finishReason: MAX_TOKENS`.
            "max_tokens": 1024,
        }))
        .map_err(ProbeError::Serialise)?;
        self.probe_request(Some(account), request, PROBE_BODY_LIMIT)
            .await
    }

    /// Send a caller-supplied OpenAI request through the full pipeline.
    ///
    /// `body_limit` caps only the raw copy retained for display. The response is
    /// interpreted from the full decoded stream either way, so a caller that
    /// wants to print everything (`--raw`) can raise it without changing what
    /// the probe concludes.
    pub async fn probe_request(
        &self,
        account: Option<&Account>,
        request: ChatCompletionRequest,
        body_limit: usize,
    ) -> Result<ProbeReport, ProbeError> {
        let prepared = self.prepare(&request)?;
        let signatures_before = self.signatures.stats();
        let started = std::time::Instant::now();

        // Pinned when an account was named, otherwise the router chooses. The
        // unpinned form is the useful default: it exercises the same selection,
        // rotation, and backoff path that serving real traffic will.
        let call = match account {
            Some(account) => self.dispatch_to(&prepared, &account.credential_id()).await?,
            None => self.dispatch(&prepared).await?,
        };
        let elapsed = started.elapsed();
        let status = call.response.status;

        let mut translator = StreamTranslator::new(ResponseOptions::new(
            request.model.clone(),
            call.family,
            call.session_key.clone(),
        ));
        let run = drain_stream(
            call.response.body,
            &mut translator,
            &self.signatures,
            body_limit,
        )
        .await;

        let parsed = interpret_body(&run.raw, &run.events);

        // Assemble a non-streaming view from the same events, so a healthy probe
        // shows the answer a client would have received.
        //
        // `None` for the cache on purpose: this assembly is for display only,
        // and letting it write would cache every signature a second time under
        // ids no client ever saw. The streaming translator already captured
        // them properly.
        let completion = parsed.as_ref().and_then(|payload| {
            let response =
                <crate::transform::ir::GenerateContentResponse as serde::Deserialize>::deserialize(
                    payload,
                )
                .ok()?;
            let parts = response
                .candidates
                .first()
                .and_then(|candidate| candidate.content.as_ref())
                .map(|content| content.parts.clone())
                .unwrap_or_default();
            Some(to_completion(
                &parts,
                response.usage_metadata.as_ref(),
                response
                    .candidates
                    .first()
                    .and_then(|candidate| candidate.finish_reason.as_deref()),
                &ResponseOptions::new(
                    request.model.clone(),
                    call.family,
                    call.session_key.clone(),
                ),
                None,
            ))
        });

        let signatures_after = self.signatures.stats();

        let snapshot = self.accounts.snapshot();
        let served = snapshot.accounts.get(call.account_index);

        Ok(ProbeReport {
            model: request.model.clone(),
            wire_model: call.wire_model.clone(),
            thinking_tier: prepared.resolved.tier.as_str().to_string(),
            account: served.map(Account::label).unwrap_or_else(|| "unknown".into()),
            account_id: call.credential_id.chars().take(8).collect(),
            project_id: call.project_id.clone(),
            used_fallback_project: call.used_fallback_project,
            tier: call.tier.clone(),
            trace_id: run.trace_id.clone().unwrap_or_default(),
            session_id: call.session_key.clone(),
            attempts: vec![ProbeAttempt {
                endpoint: call.endpoint.clone(),
                status: Some(status),
                elapsed,
                body: run.raw.clone(),
            }],
            status,
            body: run.raw,
            truncated: run.truncated,
            parsed,
            completion,
            chunk_count: run.chunks,
            malformed_events: run.malformed,
            upstream_attempts: call.attempts,
            signatures_captured: signatures_after
                .tool_signatures
                .saturating_sub(signatures_before.tool_signatures)
                + signatures_after
                    .session_signatures
                    .saturating_sub(signatures_before.session_signatures),
        })
    }
}

/// An account with its token and project resolved.
#[derive(Debug, Clone)]
pub struct PreparedAccount {
    pub access_token: String,
    pub project_id: String,
    pub used_fallback_project: bool,
    /// Human-readable tier, for display.
    pub tier: String,
    /// Raw tier id as the upstream reported it, for persistence.
    pub tier_id: Option<String>,
    /// Raw paid-tier id, when the account has one.
    pub paid_tier_id: Option<String>,
}

/// One endpoint attempt during a probe.
#[derive(Debug, Clone)]
pub struct ProbeAttempt {
    pub endpoint: String,
    pub status: Option<reqwest::StatusCode>,
    pub elapsed: Duration,
    pub body: String,
}

/// Everything a probe observed.
#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub model: String,
    /// Account that served the request.
    pub account: String,
    /// Short credential id of the account that served it. Needed because two
    /// accounts can share a display label, which makes the label ambiguous
    /// exactly when rotation matters.
    pub account_id: String,
    /// Wire model the request actually named, after tier resolution.
    pub wire_model: String,
    /// Thinking tier applied.
    pub thinking_tier: String,
    pub project_id: String,
    pub used_fallback_project: bool,
    /// Subscription tier of the account, e.g. `free-tier`.
    pub tier: String,
    /// Upstream trace id, useful in support requests.
    pub trace_id: String,
    pub session_id: String,
    pub attempts: Vec<ProbeAttempt>,
    pub status: reqwest::StatusCode,
    pub body: String,
    pub truncated: bool,
    /// Parsed payload, from the raw body or unwrapped from SSE `data:` lines.
    pub parsed: Option<Value>,
    /// The assembled OpenAI response, produced by the same translator the
    /// gateway uses. Present only on a successful call.
    pub completion: Option<ChatCompletion>,
    /// Chunks the streaming translator emitted, proving that path ran.
    pub chunk_count: usize,
    /// Upstream attempts this took, including the successful one. Above one
    /// means a retry or an account rotation happened.
    pub upstream_attempts: u32,
    /// SSE lines that failed to parse, which a healthy stream never has.
    pub malformed_events: usize,
    /// Signatures captured during this call.
    pub signatures_captured: usize,
}

impl ProbeReport {
    pub fn succeeded(&self) -> bool {
        self.status.is_success()
    }

    /// Visible answer text, excluding reasoning.
    ///
    /// Thinking parts carry `text` just like answer parts; the `thought` flag is
    /// the only thing distinguishing them, so a naive join would print the
    /// model's reasoning as though it were the reply.
    pub fn text(&self) -> Option<String> {
        let text: String = self
            .parts()?
            .iter()
            .filter(|part| part.get("thought").and_then(Value::as_bool) != Some(true))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect();
        (!text.is_empty()).then_some(text)
    }

    /// Reasoning text, if the response carried any.
    pub fn reasoning(&self) -> Option<String> {
        let text: String = self
            .parts()?
            .iter()
            .filter(|part| part.get("thought").and_then(Value::as_bool) == Some(true))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect();
        (!text.is_empty()).then_some(text)
    }

    /// Signatures the response carried, whatever part they were attached to.
    pub fn signatures(&self) -> Vec<String> {
        self.parts()
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|part| part.get("thoughtSignature").and_then(Value::as_str))
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn parts(&self) -> Option<&Vec<Value>> {
        self.parsed
            .as_ref()?
            .get("candidates")?
            .as_array()?
            .first()?
            .get("content")?
            .get("parts")?
            .as_array()
    }

    /// Finish reason, when the response carried one.
    pub fn finish_reason(&self) -> Option<&str> {
        self.parsed
            .as_ref()?
            .get("candidates")?
            .as_array()?
            .first()?
            .get("finishReason")?
            .as_str()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("authentication failed: {0}")]
    OAuth(#[from] crate::oauth::token::OAuthError),

    #[error("could not resolve the requested model: {0}")]
    Resolve(#[from] crate::registry::ResolveError),

    #[error("could not translate the request: {0}")]
    Translate(#[from] crate::transform::request::TranslateError),

    #[error("{0}")]
    Dispatch(#[from] crate::engine::dispatch::DispatchError),

    /// An auxiliary call failed in a way that is not worth a typed error.
    #[error("{0}")]
    Unavailable(String),

    #[error("project discovery failed: {0}")]
    Project(#[from] crate::accounts::project::ProjectError),

    #[error("could not serialise the request: {0}")]
    Serialise(serde_json::Error),

    #[error("every endpoint failed: {}", format_attempts(.0))]
    AllEndpointsFailed(Vec<ProbeAttempt>),
}

fn format_attempts(attempts: &[ProbeAttempt]) -> String {
    attempts
        .iter()
        .map(|attempt| format!("{} -> {}", attempt.endpoint, attempt.body))
        .collect::<Vec<_>>()
        .join("; ")
}

/// What running a stream through the pipeline produced.
struct StreamRun {
    /// Raw upstream bytes, for diagnosis. Capped: once the limit is reached the
    /// tail is dropped, so this is for display and must never be parsed for an
    /// answer.
    raw: String,
    truncated: bool,
    /// Every decoded event, in order, so the response is interpreted from the
    /// whole stream rather than from the capped copy. Reading the answer out of
    /// `raw` was a bug: a long response lost its tail to the cap, and with the
    /// thinking prose filling the cap first, the answer that followed looked
    /// like no answer at all.
    events: Vec<Value>,
    /// Trace id the upstream stamped on its events, for support requests.
    trace_id: Option<String>,
    /// Chunks the translator emitted.
    chunks: usize,
    /// SSE lines that failed to parse.
    malformed: usize,
}

/// Decode an upstream body and feed it through the response translator.
///
/// Every byte goes to the SSE decoder and every decoded event to the
/// translator, while a copy is retained for the report. Retaining the copy must
/// not affect what the translator sees, so truncation applies only to the copy.
async fn drain_stream(
    mut stream: crate::upstream::transport::BodyStream,
    translator: &mut StreamTranslator,
    cache: &SignatureCache,
    limit: usize,
) -> StreamRun {
    use futures::StreamExt;

    let mut decoder = SseDecoder::new();
    let mut raw: Vec<u8> = Vec::new();
    let mut truncated = false;
    let mut malformed = 0usize;
    let mut chunks = 0usize;
    let mut events: Vec<Value> = Vec::new();
    let mut trace_id: Option<String> = None;

    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else {
            break;
        };

        if raw.len() < limit {
            let take = (limit - raw.len()).min(chunk.len());
            raw.extend_from_slice(&chunk[..take]);
            truncated |= take < chunk.len();
        } else {
            truncated = true;
        }

        match decoder.push(&chunk) {
            Ok(decoded) => {
                malformed += decoded.malformed;
                for event in decoded.events {
                    chunks += translator.on_payload(&event.payload, Some(cache)).len();
                    trace_id = trace_id.or(event.trace_id);
                    events.push(event.payload);
                }
            }
            // A decoder that cannot make progress will not recover mid-stream.
            Err(_) => break,
        }
    }

    if let Ok(decoded) = decoder.finish() {
        malformed += decoded.malformed;
        for event in decoded.events {
            chunks += translator.on_payload(&event.payload, Some(cache)).len();
            trace_id = trace_id.or(event.trace_id);
            events.push(event.payload);
        }
    }

    chunks += translator.finish().len();

    StreamRun {
        raw: String::from_utf8_lossy(&raw).into_owned(),
        truncated,
        events,
        trace_id,
        chunks,
        malformed,
    }
}

/// Interpret an upstream body as one response.
///
/// A non-streaming body is a single JSON object and parses directly; a streaming
/// one is folded from its decoded events. Folding the *events* rather than the
/// raw text is what makes a long answer readable — `raw` is capped for display,
/// so anything past the cap would otherwise be invisible here.
fn interpret_body(raw: &str, events: &[Value]) -> Option<Value> {
    serde_json::from_str::<Value>(raw)
        .ok()
        .or_else(|| merge_events(events))
}

/// Fold decoded payloads into one response.
///
/// The payloads arrive already unwrapped from their `{response, traceId}`
/// envelope, in stream order.
///
/// The upstream splits a single logical response across events: text arrives
/// first, then a trailing event carries the signature with an *empty* text part
/// and the finish reason. So neither end of the stream is a complete answer —
/// reading only the first loses the finish reason and the signature, reading only
/// the last loses the text. Parts therefore accumulate, while usage, finish
/// reason, and model version are taken from the newest event.
fn merge_events(events: &[Value]) -> Option<Value> {
    let mut parts: Vec<Value> = Vec::new();
    let mut newest: Option<Value> = None;

    for event in events {
        if let Some(event_parts) = event
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
        {
            parts.extend(event_parts.iter().cloned());
        }
        newest = Some(event.clone());
    }

    let mut merged = newest?;

    // Splice the accumulated parts into the newest event, creating the
    // intermediate containers if the newest event happened to omit them.
    // Everything else — finish reason, usage, model version — is newest-wins.
    if !parts.is_empty() {
        let object = merged.as_object_mut()?;
        let candidates = object
            .entry("candidates")
            .or_insert_with(|| Value::Array(Vec::new()));
        let candidates = candidates.as_array_mut()?;
        if candidates.is_empty() {
            candidates.push(Value::Object(serde_json::Map::new()));
        }

        let first = candidates.first_mut()?.as_object_mut()?;
        let content = first
            .entry("content")
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
        content
            .as_object_mut()?
            .insert("parts".into(), Value::Array(parts));
    }

    Some(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode a body into payloads, the way the drain does.
    fn decode(body: &str) -> Vec<Value> {
        let mut decoder = SseDecoder::new();
        let mut events = decoder
            .push(body.as_bytes())
            .expect("the body should decode")
            .events;
        events.extend(decoder.finish().expect("nothing should be pending").events);
        events.into_iter().map(|event| event.payload).collect()
    }

    #[test]
    fn sse_events_accumulate_their_parts() {
        let body = concat!(
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"first\"}]}}]},\"traceId\":\"a\"}

",
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"second\"}]}}]},\"traceId\":\"b\"}

",
        );
        let parsed = merge_events(&decode(body)).expect("should find a payload");
        let parts = parsed["candidates"][0]["content"]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], "first");
        assert_eq!(parts[1]["text"], "second");
    }

    #[test]
    fn text_and_signature_are_merged_across_events() {
        // The shape a live gemini-3.8-flash response actually had: the answer
        // text in one event, then a trailing event carrying the signature on an
        // empty text part together with the finish reason. Reading either end of
        // the stream alone produces an incomplete answer.
        let body = concat!(
            "data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]}}],\"usageMetadata\":{\"promptTokenCount\":6,\"candidatesTokenCount\":1,\"thoughtsTokenCount\":93},\"modelVersion\":\"gemini-3.8-flash\"}}

",
            "data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"thoughtSignature\":\"EvYDCvMDARFN\",\"text\":\"\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":6,\"candidatesTokenCount\":1,\"thoughtsTokenCount\":93},\"modelVersion\":\"gemini-3.8-flash\"}}

",
        );

        let report = report_with(merge_events(&decode(body)));
        assert_eq!(report.text().as_deref(), Some("ok"));
        assert_eq!(report.finish_reason(), Some("STOP"));
        assert_eq!(report.signatures(), vec!["EvYDCvMDARFN"]);
    }

    #[test]
    fn usage_and_finish_reason_come_from_the_newest_event() {
        let body = concat!(
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"x\"}]}}],\"usageMetadata\":{\"candidatesTokenCount\":1}}}

",
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"candidatesTokenCount\":42}}}

",
        );
        let parsed = merge_events(&decode(body)).unwrap();
        assert_eq!(parsed["usageMetadata"]["candidatesTokenCount"], 42);
        assert_eq!(parsed["candidates"][0]["finishReason"], "STOP");
        assert_eq!(parsed["candidates"][0]["content"]["parts"][0]["text"], "x");
    }

    #[test]
    fn a_signature_only_event_still_yields_a_candidate() {
        // The newest event may describe no candidates of its own; the merged
        // result must still carry the accumulated parts.
        let body = concat!(
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"x\"}]}}]}}

",
            "data: {\"response\":{\"usageMetadata\":{\"candidatesTokenCount\":7}}}

",
        );
        let parsed = merge_events(&decode(body)).unwrap();
        assert_eq!(parsed["candidates"][0]["content"]["parts"][0]["text"], "x");
        assert_eq!(parsed["usageMetadata"]["candidatesTokenCount"], 7);
    }

    #[test]
    fn merging_skips_malformed_and_unrelated_lines() {
        let body = concat!(
            ": keep-alive comment
",
            "data: not json
",
            "event: ignored
",
            "data: {\"traceId\":\"only-trace\"}
",
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"a\"}]}}]}}
",
        );
        let parsed = merge_events(&decode(body)).unwrap();
        assert_eq!(parsed["candidates"][0]["content"]["parts"][0]["text"], "a");
    }

    #[test]
    fn interpreting_a_plain_json_body_parses_it_directly() {
        // A non-streaming body is a single JSON object with no `data:` lines for
        // the decoder to find; the direct parse is what handles it, and the
        // event fold must not claim it on its own.
        let body = r#"{"candidates":[]}"#;
        assert!(merge_events(&decode(body)).is_none());
        let parsed = interpret_body(body, &decode(body)).expect("a plain body parses");
        assert_eq!(parsed["candidates"], serde_json::json!([]));
    }

    #[test]
    fn interpreting_an_empty_body_yields_nothing() {
        assert!(interpret_body("", &decode("")).is_none());
    }

    #[tokio::test]
    async fn draining_a_stream_counts_the_chunks_it_emitted() {
        let body = concat!(
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hello\"}]}}]}}

",
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}}

",
        );
        let stream = body_stream(vec![Ok(bytes::Bytes::from(body))]);
        let mut translator = StreamTranslator::new(ResponseOptions::new(
            "m",
            crate::registry::ModelFamily::Gemini,
            "s",
        ));
        let cache = SignatureCache::new();

        let run = drain_stream(stream, &mut translator, &cache, 4096).await;

        assert_eq!(run.malformed, 0);
        // Two content chunks plus the terminal one.
        assert_eq!(run.chunks, 2, "one text chunk and one finish chunk");
        assert!(run.raw.contains("hello"), "the raw copy is retained");
        assert!(!run.truncated);
    }

    #[tokio::test]
    async fn truncating_the_raw_copy_does_not_truncate_the_translation() {
        // The copy exists for the report; truncating it must not cost the
        // translator any content.
        let body = concat!(
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"aaaaaaaaaa\"}]}}]}}

",
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"bbbbbbbbbb\"}]}}]}}

",
        );
        let stream = body_stream(vec![Ok(bytes::Bytes::from(body))]);
        let mut translator = StreamTranslator::new(ResponseOptions::new(
            "m",
            crate::registry::ModelFamily::Gemini,
            "s",
        ));
        let cache = SignatureCache::new();

        let run = drain_stream(stream, &mut translator, &cache, 40).await;

        assert!(run.truncated, "the copy is over the limit");
        assert!(run.raw.len() <= 40);
        assert_eq!(run.chunks, 3, "both text chunks still reached the client");
    }

    #[tokio::test]
    async fn an_answer_past_the_raw_cap_is_still_read() {
        // The bug this guards: the report folded the *capped* copy, so once
        // reasoning had filled the cap the answer that followed was invisible,
        // and a long response looked like an empty one.
        let body = concat!(
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"deliberating at some length\",\"thought\":true}]}}]}}\n",
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"the answer\"}]}}]}}\n",
        );
        let stream = body_stream(vec![Ok(bytes::Bytes::from(body))]);
        let mut translator = StreamTranslator::new(ResponseOptions::new(
            "m",
            crate::registry::ModelFamily::Gemini,
            "s",
        ));
        let cache = SignatureCache::new();

        // A cap that the first event alone already exceeds.
        let run = drain_stream(stream, &mut translator, &cache, 32).await;

        assert!(run.truncated);
        assert!(
            !run.raw.contains("the answer"),
            "the answer is past the cap, so the copy cannot be where it came from"
        );
        assert_eq!(run.events.len(), 2, "every event is retained");

        let report = report_with(interpret_body(&run.raw, &run.events));
        assert_eq!(report.text().as_deref(), Some("the answer"));
        assert_eq!(
            report.reasoning().as_deref(),
            Some("deliberating at some length")
        );
    }

    #[tokio::test]
    async fn the_trace_id_comes_from_the_decoded_envelope() {
        let body = "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]}}]},\"traceId\":\"trace-42\"}\n";
        let stream = body_stream(vec![Ok(bytes::Bytes::from(body))]);
        let mut translator = StreamTranslator::new(ResponseOptions::new(
            "m",
            crate::registry::ModelFamily::Gemini,
            "s",
        ));
        let cache = SignatureCache::new();

        let run = drain_stream(stream, &mut translator, &cache, 4096).await;
        assert_eq!(run.trace_id.as_deref(), Some("trace-42"));
    }

    #[tokio::test]
    async fn malformed_lines_are_counted_through_the_drain() {
        let body = "data: {broken
data: {\"response\":{\"candidates\":[]}}
";
        let stream = body_stream(vec![Ok(bytes::Bytes::from(body))]);
        let mut translator = StreamTranslator::new(ResponseOptions::new(
            "m",
            crate::registry::ModelFamily::Gemini,
            "s",
        ));
        let cache = SignatureCache::new();

        let run = drain_stream(stream, &mut translator, &cache, 4096).await;
        assert_eq!(run.malformed, 1);
    }

    #[tokio::test]
    async fn a_body_split_across_chunks_is_stitched_by_the_drain() {
        // The transport delivers arbitrary chunks; the decoder must not care.
        let stream = body_stream(vec![
            Ok(bytes::Bytes::from_static(
                b"data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"te",
            )),
            Ok(bytes::Bytes::from_static(b"xt\":\"split\"}]}}]}}

")),
        ]);
        let mut translator = StreamTranslator::new(ResponseOptions::new(
            "m",
            crate::registry::ModelFamily::Gemini,
            "s",
        ));
        let cache = SignatureCache::new();

        let run = drain_stream(stream, &mut translator, &cache, 4096).await;
        assert_eq!(run.malformed, 0);
        assert_eq!(run.chunks, 2);
        assert!(run.raw.contains("split"));
    }

    /// Build a body stream from a list of chunks.
    fn body_stream(
        chunks: Vec<Result<bytes::Bytes, reqwest::Error>>,
    ) -> crate::upstream::transport::BodyStream {
        Box::pin(futures::stream::iter(chunks))
    }

    fn report_with(parsed: Option<Value>) -> ProbeReport {
        ProbeReport {
            model: "m".into(),
            account: "test@example.com".into(),
            account_id: "abcdef01".into(),
            wire_model: "m-medium".into(),
            thinking_tier: "medium".into(),
            project_id: "p".into(),
            used_fallback_project: false,
            tier: "free-tier".into(),
            trace_id: "trace-x".into(),
            session_id: "1".into(),
            attempts: Vec::new(),
            status: reqwest::StatusCode::OK,
            body: String::new(),
            truncated: false,
            parsed,
            completion: None,
            chunk_count: 0,
            upstream_attempts: 1,
            malformed_events: 0,
            signatures_captured: 0,
        }
    }

    #[test]
    fn report_extracts_text_from_candidates() {
        let report = report_with(Some(serde_json::json!({
            "candidates": [{ "content": { "parts": [{ "text": "hello " }, { "text": "world" }] } }]
        })));
        assert_eq!(report.text().as_deref(), Some("hello world"));
    }

    #[test]
    fn report_separates_reasoning_from_the_answer() {
        let report = report_with(Some(serde_json::json!({
            "candidates": [{ "content": { "parts": [
                { "text": "reasoning", "thought": true },
                { "text": "answer" }
            ] } }]
        })));
        assert_eq!(report.text().as_deref(), Some("answer"));
        assert_eq!(report.reasoning().as_deref(), Some("reasoning"));
    }

    #[test]
    fn report_reasoning_is_none_when_absent() {
        let report = report_with(Some(serde_json::json!({
            "candidates": [{ "content": { "parts": [{ "text": "answer" }] } }]
        })));
        assert!(report.reasoning().is_none());
    }

    #[test]
    fn report_collects_signatures_from_any_part() {
        let report = report_with(Some(serde_json::json!({
            "candidates": [{ "content": { "parts": [
                { "text": "thinking", "thought": true, "thoughtSignature": "sig-a" },
                { "functionCall": { "name": "f" }, "thoughtSignature": "sig-b" }
            ] } }]
        })));
        assert_eq!(report.signatures(), vec!["sig-a", "sig-b"]);
    }

    #[test]
    fn report_has_no_signatures_when_none_present() {
        let report = report_with(Some(serde_json::json!({
            "candidates": [{ "content": { "parts": [{ "text": "answer" }] } }]
        })));
        assert!(report.signatures().is_empty());
    }

    #[test]
    fn report_returns_none_for_empty_candidates() {
        let report = report_with(Some(serde_json::json!({ "candidates": [] })));
        assert!(report.text().is_none());
        assert!(report.finish_reason().is_none());
    }

    #[test]
    fn report_reads_finish_reason() {
        let report = report_with(Some(serde_json::json!({
            "candidates": [{ "finishReason": "STOP", "content": { "parts": [] } }]
        })));
        assert_eq!(report.finish_reason(), Some("STOP"));
    }
}
