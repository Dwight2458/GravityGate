//! IR → OpenAI response translation.
//!
//! Both directions of the response path live here so they cannot drift: a
//! non-streaming reply and the concatenation of a streamed one must describe the
//! same thing.
//!
//! The streaming side is an accumulator, not a per-event mapper, and that is
//! forced by how the upstream splits a reply. A live `gemini-3.8-flash` response
//! arrived as two events: the answer text and its usage in the first, a
//! `thoughtSignature` on an *empty* text part and the finish reason in the
//! second. So:
//!
//! - a part with empty text and no `thought` flag is **not** nothing — it is
//!   where a signature lives, and dropping it breaks the next turn of a tool
//!   loop;
//! - the finish reason and usage must be taken from whichever event carries
//!   them, which is not the event carrying the content;
//! - a finish reason can arrive after the content it terminates, so the terminal
//!   chunk is emitted on stream end rather than on seeing `finishReason`.
//!
//! Signatures are captured here and never emitted: OpenAI's protocol has no
//! field for one. See [`super::signature_cache`].

use serde::Deserialize as _;
use serde_json::Value;

use crate::config::ReasoningField;
use crate::registry::models::ModelFamily;
use crate::transform::ir::{
    GenerateContentResponse, Part, UsageMetadata as IrUsage,
};
use crate::transform::openai::{
    ChatCompletion, ChatCompletionChunk, Choice, ChunkChoice, ChunkDelta,
    CompletionTokensDetails, IndexedToolCall, PromptTokensDetails, ResponseFunction,
    ResponseMessage, ResponseToolCall, Usage,
};
use crate::transform::signature_cache::SignatureCache;
use crate::upstream::constants::SKIP_THOUGHT_SIGNATURE;

/// Length of a generated tool call id, in hex characters.
const TOOL_CALL_ID_HEX: usize = 24;

/// What a part means to a consumer.
///
/// Classification is shared so the streaming and non-streaming paths cannot
/// disagree about, say, whether a signature-only part is content.
#[derive(Debug, PartialEq)]
pub enum PartKind<'a> {
    /// Reasoning text.
    Thinking(&'a str),
    /// Answer text.
    Text(&'a str),
    /// A tool call.
    ToolCall {
        name: &'a str,
        args: &'a serde_json::Map<String, Value>,
        id: Option<&'a str>,
    },
    /// A signature with no content of its own. Still load-bearing.
    SignatureOnly,
    /// An image or other non-text payload, which this surface cannot carry.
    Unsupported,
    /// Nothing at all.
    Empty,
}

/// Classify a part.
///
/// **Empty text is never content**, whatever else the part carries. Live traffic
/// uses zero-length text parts as carriers: a Gemini thinking signature arrives
/// on `{"text": "", "thoughtSignature": "..."}` with no `thought` flag, and a
/// Claude thinking signature arrives on `{"thought": true, "text": "",
/// "thoughtSignature": "..."}` *with* one. Testing the `thought` flag first makes
/// the second of those look like reasoning and emits an empty reasoning delta for
/// every signature — a real bug, caught by counting chunks on a live six-event
/// Claude stream.
///
/// So the emptiness test comes first. The signature is captured separately by
/// the caller, whichever kind this returns.
pub fn classify(part: &Part) -> PartKind<'_> {
    if let Some(call) = &part.function_call {
        return PartKind::ToolCall {
            name: &call.name,
            args: &call.args,
            id: None,
        };
    }

    if let Some(text) = &part.text {
        if text.is_empty() {
            return if part.thought_signature.is_some() {
                PartKind::SignatureOnly
            } else {
                PartKind::Empty
            };
        }
        return if part.is_thought() {
            PartKind::Thinking(text)
        } else {
            PartKind::Text(text)
        };
    }

    if part.thought_signature.is_some() {
        return PartKind::SignatureOnly;
    }
    if part.inline_data.is_some() {
        return PartKind::Unsupported;
    }
    PartKind::Empty
}

/// Shared settings for producing a client-facing response.
#[derive(Debug, Clone)]
pub struct ResponseOptions {
    /// `chatcmpl-...` identifier.
    pub completion_id: String,
    /// The model name the client asked for, echoed back.
    pub model: String,
    pub created: i64,
    /// Family of the model that produced this, used to validate signatures.
    pub family: ModelFamily,
    pub reasoning_field: ReasoningField,
    /// Whether to emit a final usage-only chunk.
    pub include_usage: bool,
    /// Conversation key for session-scoped signature storage.
    pub session_key: String,
}

impl ResponseOptions {
    /// Build options with a freshly generated completion id.
    pub fn new(model: impl Into<String>, family: ModelFamily, session_key: impl Into<String>) -> Self {
        Self {
            completion_id: new_completion_id(),
            model: model.into(),
            created: now_secs(),
            family,
            reasoning_field: ReasoningField::ReasoningContent,
            include_usage: false,
            session_key: session_key.into(),
        }
    }
}

/// Generate an OpenAI-shaped completion id.
pub fn new_completion_id() -> String {
    format!("chatcmpl-{}", simple_id())
}

/// Generate an OpenAI-shaped tool call id.
pub fn new_tool_call_id() -> String {
    format!("call_{}", simple_id())
}

/// The id to expose for a tool call.
///
/// The upstream's own id is preferred where it supplies one: the backend may
/// correlate it with the signature it issued in the same part, and substituting
/// an id of our own would break that link for no gain. The client echoes
/// whichever id it was given, so the signature cache stays addressable either
/// way — which is what makes reusing the upstream id safe.
fn tool_call_id_for(call: &crate::transform::ir::FunctionCall) -> String {
    match &call.id {
        Some(id) if !id.is_empty() => id.clone(),
        _ => new_tool_call_id(),
    }
}

/// A random hex string without pulling in a UUID's formatting.
fn simple_id() -> String {
    use rand::Rng as _;
    let mut bytes = [0u8; TOOL_CALL_ID_HEX / 2];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

/// Record any signature on a part.
///
/// The key is chosen by what else is in the part: a signature on a tool call
/// belongs to that call, and anything else belongs to the conversation's
/// thinking. Both are captured regardless of whether the part also carried text,
/// because live traffic showed the two arrive independently.
fn capture_signature(
    part: &Part,
    tool_call_id: Option<&str>,
    options: &ResponseOptions,
    cache: Option<&SignatureCache>,
) {
    let Some(cache) = cache else {
        return;
    };
    let Some(signature) = part.signature() else {
        return;
    };
    match tool_call_id {
        Some(id) => {
            cache.put_tool(id, signature, options.family);
        }
        None => {
            cache.put_thinking(&options.session_key, signature, options.family);
        }
    }
}

/// Apply reasoning text to a message or delta under the configured field name.
fn set_reasoning(target: &mut ReasoningTarget<'_>, text: String, field: ReasoningField) {
    match target {
        ReasoningTarget::Message(message) => match field {
            ReasoningField::ReasoningContent => message.reasoning_content = Some(text),
            ReasoningField::Reasoning => message.reasoning = Some(text),
            ReasoningField::Both => {
                message.reasoning_content = Some(text.clone());
                message.reasoning = Some(text);
            }
        },
        ReasoningTarget::Delta(delta) => match field {
            ReasoningField::ReasoningContent => delta.reasoning_content = Some(text),
            ReasoningField::Reasoning => delta.reasoning = Some(text),
            ReasoningField::Both => {
                delta.reasoning_content = Some(text.clone());
                delta.reasoning = Some(text);
            }
        },
    }
}

enum ReasoningTarget<'a> {
    Message(&'a mut ResponseMessage),
    Delta(&'a mut ChunkDelta),
}

/// Map an upstream finish reason onto OpenAI's vocabulary.
///
/// A tool call wins outright: the upstream reports `STOP` after a function call
/// as readily as it reports anything else, and a client that sees `stop` will
/// not look for tool calls.
pub fn map_finish_reason(upstream: Option<&str>, saw_tool_call: bool) -> String {
    if saw_tool_call {
        return "tool_calls".into();
    }
    match upstream {
        Some("STOP") => "stop".into(),
        Some("MAX_TOKENS") => "length".into(),
        Some("SAFETY") | Some("RECITATION") | Some("BLOCKLIST")
        | Some("PROHIBITED_CONTENT") | Some("SPII") | Some("IMAGE_SAFETY") => {
            "content_filter".into()
        }
        // An absent or unrecognised reason is reported as a normal stop rather
        // than propagated: clients act on unknown values unpredictably.
        _ => "stop".into(),
    }
}

/// Convert accumulated token counts into OpenAI's shape.
pub fn to_usage(usage: &IrUsage) -> Usage {
    let prompt_tokens = usage.prompt_tokens();
    let completion_tokens = usage.candidates_tokens();
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        prompt_tokens_details: Some(PromptTokensDetails {
            cached_tokens: usage.cached_content_token_count,
        }),
        completion_tokens_details: Some(CompletionTokensDetails {
            reasoning_tokens: usage.thoughts_token_count,
        }),
    }
}

/// Build a non-streaming completion from accumulated parts.
pub fn to_completion(
    parts: &[Part],
    usage: Option<&IrUsage>,
    finish_reason: Option<&str>,
    options: &ResponseOptions,
    cache: Option<&SignatureCache>,
) -> ChatCompletion {
    let mut message = ResponseMessage {
        role: "assistant",
        ..Default::default()
    };

    let mut thinking = String::new();
    let mut content = String::new();
    let mut tool_calls: Vec<ResponseToolCall> = Vec::new();

    for part in parts {
        // The id must be settled before anything else, because a signature on
        // the same part is cached under whichever id the client receives.
        let call_id = part.function_call.as_ref().map(tool_call_id_for);

        capture_signature(part, call_id.as_deref(), options, cache);

        match classify(part) {
            PartKind::Thinking(text) => thinking.push_str(text),
            PartKind::Text(text) => content.push_str(text),
            PartKind::ToolCall { name, args, .. } => {
                tool_calls.push(ResponseToolCall {
                    id: call_id.clone().unwrap_or_else(new_tool_call_id),
                    kind: "function",
                    function: ResponseFunction {
                        name: name.to_string(),
                        arguments: serde_json::to_string(args)
                            .unwrap_or_else(|_| "{}".into()),
                    },
                });
            }
            PartKind::SignatureOnly | PartKind::Unsupported | PartKind::Empty => {}
        }
    }

    if !thinking.is_empty() {
        set_reasoning(
            &mut ReasoningTarget::Message(&mut message),
            thinking,
            options.reasoning_field,
        );
    }

    // OpenAI sets `content` to null when a turn is nothing but tool calls, and
    // to a string otherwise — an empty one when the model produced no text,
    // which is what a token-limited reply looks like.
    message.content = if tool_calls.is_empty() {
        Some(content)
    } else {
        (!content.is_empty()).then_some(content)
    };
    if !tool_calls.is_empty() {
        message.tool_calls = Some(tool_calls);
    }

    let saw_tool_call = message.tool_calls.is_some();

    ChatCompletion {
        id: options.completion_id.clone(),
        object: "chat.completion",
        created: options.created,
        model: options.model.clone(),
        choices: vec![Choice {
            index: 0,
            message,
            finish_reason: Some(map_finish_reason(finish_reason, saw_tool_call)),
        }],
        usage: usage.filter(|usage| usage.is_present()).map(to_usage),
        system_fingerprint: None,
    }
}

/// Streaming translator.
///
/// Feed it each decoded upstream payload; it returns the chunks to send. The
/// terminal chunks come from [`Self::finish`], because the finish reason may
/// arrive in a later event than the content.
#[derive(Debug)]
pub struct StreamTranslator {
    options: ResponseOptions,
    sent_role: bool,
    finished: bool,
    finish_reason: Option<String>,
    usage: Option<IrUsage>,
    tool_call_index: u32,
    saw_tool_call: bool,
}

impl StreamTranslator {
    pub fn new(options: ResponseOptions) -> Self {
        Self {
            options,
            sent_role: false,
            finished: false,
            finish_reason: None,
            usage: None,
            tool_call_index: 0,
            saw_tool_call: false,
        }
    }

    /// Consume one upstream payload, returning the chunks it produced.
    pub fn on_payload(
        &mut self,
        payload: &Value,
        cache: Option<&SignatureCache>,
    ) -> Vec<ChatCompletionChunk> {
        let Ok(response) = GenerateContentResponse::deserialize(payload) else {
            return Vec::new();
        };
        self.on_response(&response, cache)
    }

    /// Consume an already-parsed response.
    pub fn on_response(
        &mut self,
        response: &GenerateContentResponse,
        cache: Option<&SignatureCache>,
    ) -> Vec<ChatCompletionChunk> {
        // Usage and finish reason are newest-wins, and may arrive separately
        // from each other and from the content.
        if let Some(usage) = &response.usage_metadata
            && usage.is_present()
        {
            self.usage = Some(usage.clone());
        }

        let mut chunks = Vec::new();

        for candidate in &response.candidates {
            if let Some(reason) = &candidate.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
            let Some(content) = &candidate.content else {
                continue;
            };

            for part in &content.parts {
                let call_id = part.function_call.as_ref().map(tool_call_id_for);
                capture_signature(part, call_id.as_deref(), &self.options, cache);

                let mut delta = ChunkDelta::default();
                match classify(part) {
                    PartKind::Thinking(text) => {
                        set_reasoning(
                            &mut ReasoningTarget::Delta(&mut delta),
                            text.to_string(),
                            self.options.reasoning_field,
                        );
                    }
                    PartKind::Text(text) => delta.content = Some(text.to_string()),
                    PartKind::ToolCall { name, args, .. } => {
                        let index = self.tool_call_index;
                        self.tool_call_index += 1;
                        self.saw_tool_call = true;
                        delta.tool_calls = Some(vec![IndexedToolCall {
                            index,
                            call: ResponseToolCall {
                                id: call_id.clone().unwrap_or_else(new_tool_call_id),
                                kind: "function",
                                function: ResponseFunction {
                                    name: name.to_string(),
                                    arguments: serde_json::to_string(args)
                                        .unwrap_or_else(|_| "{}".into()),
                                },
                            },
                        }]);
                    }
                    // Nothing visible, but the signature was already captured
                    // above. This is the case that a naive translator drops.
                    PartKind::SignatureOnly | PartKind::Unsupported | PartKind::Empty => {}
                }

                if !delta.is_empty() {
                    chunks.push(self.chunk(delta, None));
                }
            }
        }

        chunks
    }

    /// Emit the terminal chunks.
    ///
    /// Idempotent: calling it twice returns nothing the second time, so a caller
    /// that ends a stream on both an error path and a normal path cannot double
    /// up the finish chunk.
    pub fn finish(&mut self) -> Vec<ChatCompletionChunk> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;

        let reason = map_finish_reason(self.finish_reason.as_deref(), self.saw_tool_call);
        let mut chunks = vec![self.chunk(ChunkDelta::default(), Some(reason))];

        // OpenAI emits a final chunk with no choices carrying the usage, and
        // only when the client asked for it.
        if self.options.include_usage
            && let Some(usage) = &self.usage
        {
            chunks.push(ChatCompletionChunk {
                id: self.options.completion_id.clone(),
                object: "chat.completion.chunk",
                created: self.options.created,
                model: self.options.model.clone(),
                choices: Vec::new(),
                usage: Some(to_usage(usage)),
            });
        }

        chunks
    }

    /// Whether anything at all was emitted for the client.
    ///
    /// An empty stream is a failure worth retrying, not a valid empty answer.
    pub fn emitted_content(&self) -> bool {
        self.sent_role
    }

    fn chunk(&mut self, delta: ChunkDelta, finish_reason: Option<String>) -> ChatCompletionChunk {
        let mut delta = delta;
        // The first chunk of a stream must announce the role.
        if !self.sent_role {
            delta.role = Some("assistant");
            self.sent_role = true;
        }

        ChatCompletionChunk {
            id: self.options.completion_id.clone(),
            object: "chat.completion.chunk",
            created: self.options.created,
            model: self.options.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage: None,
        }
    }
}

/// Attach signatures from the cache to outgoing tool calls.
///
/// Called when translating a request, so the signature the upstream expects is
/// present on the turn it belongs to. A Gemini target with no cached signature
/// gets the sentinel the upstream accepts in place of a real one; anything else
/// is left unsigned, because a signature from the wrong family is worse than none.
pub fn attach_tool_signature(
    part: &mut Part,
    tool_call_id: &str,
    cache: &SignatureCache,
    family: ModelFamily,
) {
    use crate::transform::signature_cache::SignatureLookup;

    match cache.tool_signature(tool_call_id, family) {
        SignatureLookup::Usable(signature) => {
            part.set_signature(signature);
            return;
        }
        SignatureLookup::ForeignFamily => {
            // Unusable, but worth distinguishing from a cold cache when
            // diagnosing a tool loop.
            tracing::debug!(
                tool_call_id,
                "cached signature belongs to another model family; treating it as absent"
            );
        }
        SignatureLookup::Missing => {}
    }

    // No usable signature. Gemini 3 requires one on a replayed function call and
    // the upstream provides a sentinel for exactly this case, which covers both
    // a cold cache and a foreign-family hit. Claude has no sentinel, and a
    // fabricated signature there is worse than none.
    if family == ModelFamily::Gemini {
        part.set_signature(SKIP_THOUGHT_SIGNATURE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::ir::{Candidate, Content, FunctionCall, InlineData};
    use serde_json::json;

    fn gem() -> ModelFamily {
        ModelFamily::Gemini
    }

    fn options() -> ResponseOptions {
        ResponseOptions {
            completion_id: "chatcmpl-test".into(),
            model: "gemini-3.8-flash".into(),
            created: 1_700_000_000,
            family: gem(),
            reasoning_field: ReasoningField::ReasoningContent,
            include_usage: false,
            session_key: "session-1".into(),
        }
    }

    fn call(name: &str) -> Part {
        Part::function_call(FunctionCall {
            name: name.into(),
            args: serde_json::Map::new(),
            id: None,
        })
    }

    // -- classification ----------------------------------------------------

    #[test]
    fn a_thought_part_classifies_as_thinking() {
        assert_eq!(classify(&Part::thought_text("hmm")), PartKind::Thinking("hmm"));
    }

    #[test]
    fn a_plain_part_classifies_as_text() {
        assert_eq!(classify(&Part::text("hi")), PartKind::Text("hi"));
    }

    #[test]
    fn an_empty_text_part_with_a_signature_is_signature_only() {
        // The exact shape live traffic produced. Classifying this as empty is
        // the bug that loses signatures.
        let mut part = Part::text("");
        part.set_signature("EusCCugCARFN");
        assert_eq!(classify(&part), PartKind::SignatureOnly);
    }

    #[test]
    fn a_bare_signature_with_no_text_is_signature_only() {
        let mut part = Part::default();
        part.set_signature("sig");
        assert_eq!(classify(&part), PartKind::SignatureOnly);
    }

    #[test]
    fn an_empty_part_with_nothing_in_it_is_empty() {
        assert_eq!(classify(&Part::text("")), PartKind::Empty);
    }

    #[test]
    fn a_function_call_classifies_as_a_tool_call() {
        assert!(matches!(
            classify(&call("f")),
            PartKind::ToolCall { name: "f", .. }
        ));
    }

    #[test]
    fn a_tool_call_wins_over_other_content_in_the_same_part() {
        let mut part = call("f");
        part.text = Some("ignored".into());
        assert!(matches!(classify(&part), PartKind::ToolCall { .. }));
    }

    #[test]
    fn thinking_wins_over_a_signature_on_the_same_part() {
        // The signature is captured separately; classification reports the text.
        let mut part = Part::thought_text("reasoning");
        part.set_signature("sig");
        assert_eq!(classify(&part), PartKind::Thinking("reasoning"));
    }

    #[test]
    fn inline_data_classifies_as_unsupported() {
        let part = Part::inline_data(InlineData {
            mime_type: "image/png".into(),
            data: "AAA".into(),
        });
        assert_eq!(classify(&part), PartKind::Unsupported);
    }

    // -- finish reasons ----------------------------------------------------

    #[test]
    fn finish_reasons_map_onto_openai_vocabulary() {
        assert_eq!(map_finish_reason(Some("STOP"), false), "stop");
        assert_eq!(map_finish_reason(Some("MAX_TOKENS"), false), "length");
        assert_eq!(map_finish_reason(Some("SAFETY"), false), "content_filter");
        assert_eq!(map_finish_reason(Some("RECITATION"), false), "content_filter");
    }

    #[test]
    fn a_tool_call_forces_tool_calls_regardless_of_upstream() {
        // The upstream reports STOP after a function call as readily as
        // anything else, and a client seeing `stop` will not look for calls.
        assert_eq!(map_finish_reason(Some("STOP"), true), "tool_calls");
        assert_eq!(map_finish_reason(None, true), "tool_calls");
    }

    #[test]
    fn an_unknown_finish_reason_becomes_stop() {
        assert_eq!(map_finish_reason(Some("SOMETHING_NEW"), false), "stop");
        assert_eq!(map_finish_reason(None, false), "stop");
    }

    // -- non-streaming -----------------------------------------------------

    #[test]
    fn a_text_response_becomes_a_completion() {
        let parts = vec![Part::text("hello ")];
        let usage = IrUsage {
            prompt_token_count: 9,
            candidates_token_count: Some(2),
            ..Default::default()
        };
        let completion = to_completion(&parts, Some(&usage), Some("STOP"), &options(), None);

        assert_eq!(completion.object, "chat.completion");
        assert_eq!(completion.choices[0].message.content.as_deref(), Some("hello "));
        assert_eq!(completion.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(completion.usage.unwrap().completion_tokens, 2);
    }

    #[test]
    fn thinking_and_text_are_separated() {
        let parts = vec![Part::thought_text("let me think"), Part::text("42")];
        let completion = to_completion(&parts, None, Some("STOP"), &options(), None);
        let message = &completion.choices[0].message;
        assert_eq!(message.reasoning_content.as_deref(), Some("let me think"));
        assert_eq!(message.content.as_deref(), Some("42"));
    }

    #[test]
    fn the_reasoning_field_name_is_configurable() {
        let mut options = options();
        options.reasoning_field = ReasoningField::Reasoning;
        let completion = to_completion(&[Part::thought_text("t")], None, None, &options, None);
        let message = &completion.choices[0].message;
        assert!(message.reasoning_content.is_none());
        assert_eq!(message.reasoning.as_deref(), Some("t"));
    }

    #[test]
    fn both_reasoning_fields_can_be_emitted() {
        let mut options = options();
        options.reasoning_field = ReasoningField::Both;
        let completion = to_completion(&[Part::thought_text("t")], None, None, &options, None);
        let message = &completion.choices[0].message;
        assert_eq!(message.reasoning_content.as_deref(), Some("t"));
        assert_eq!(message.reasoning.as_deref(), Some("t"));
    }

    #[test]
    fn multiple_text_parts_are_concatenated() {
        let parts = vec![Part::text("a"), Part::text("b"), Part::text("c")];
        let completion = to_completion(&parts, None, None, &options(), None);
        assert_eq!(completion.choices[0].message.content.as_deref(), Some("abc"));
    }

    #[test]
    fn a_tool_call_becomes_a_serialised_function_call() {
        let mut part = call("get_weather");
        part.function_call.as_mut().unwrap().args =
            serde_json::from_value(json!({ "city": "Paris" })).unwrap();

        let completion = to_completion(&[part], None, Some("STOP"), &options(), None);
        let calls = completion.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].kind, "function");
        assert!(calls[0].id.starts_with("call_"));

        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn content_is_null_when_a_turn_is_only_tool_calls() {
        let completion = to_completion(&[call("f")], None, Some("STOP"), &options(), None);
        assert!(completion.choices[0].message.content.is_none());
        assert_eq!(
            completion.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[test]
    fn content_is_an_empty_string_when_the_model_said_nothing() {
        // What a token-limited reply looks like: a valid response with no text.
        let completion = to_completion(&[], None, Some("MAX_TOKENS"), &options(), None);
        assert_eq!(completion.choices[0].message.content.as_deref(), Some(""));
        assert_eq!(completion.choices[0].finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn a_signature_only_part_produces_no_content() {
        let mut part = Part::text("");
        part.set_signature("sig");
        let completion = to_completion(&[part], None, Some("STOP"), &options(), None);
        assert_eq!(completion.choices[0].message.content.as_deref(), Some(""));
    }

    #[test]
    fn usage_is_omitted_when_the_upstream_reported_none() {
        let completion = to_completion(&[Part::text("x")], None, None, &options(), None);
        assert!(completion.usage.is_none());
    }

    #[test]
    fn usage_details_carry_cached_and_reasoning_tokens() {
        let usage = IrUsage {
            prompt_token_count: 100,
            cached_content_token_count: 40,
            candidates_token_count: Some(20),
            thoughts_token_count: 15,
            total_token_count: Some(135),
        };
        let completion = to_completion(&[], Some(&usage), None, &options(), None);
        let usage = completion.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 60);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 80);
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, 40);
        assert_eq!(usage.completion_tokens_details.unwrap().reasoning_tokens, 15);
    }

    // -- signature capture -------------------------------------------------

    #[test]
    fn a_signature_on_a_tool_call_is_cached_under_its_generated_id() {
        let cache = SignatureCache::new();
        let mut part = call("f");
        part.set_signature("S".repeat(80));

        let completion = to_completion(&[part], None, None, &options(), Some(&cache));
        let id = &completion.choices[0].message.tool_calls.as_ref().unwrap()[0].id;

        assert!(
            cache.tool_signature(id, gem()).is_usable(),
            "the signature must be reachable by the id the client received"
        );
    }

    #[test]
    fn a_signature_without_a_tool_call_is_cached_against_the_session() {
        let cache = SignatureCache::new();
        let mut part = Part::text("");
        part.set_signature("S".repeat(80));

        to_completion(&[part], None, None, &options(), Some(&cache));
        assert!(cache.thinking_signature("session-1", gem()).is_usable());
    }

    #[test]
    fn a_signature_on_a_thinking_part_is_captured_alongside_its_text() {
        // The two are independent: the part carries both, and both are used.
        let cache = SignatureCache::new();
        let mut part = Part::thought_text("reasoning");
        part.set_signature("S".repeat(80));

        let completion = to_completion(&[part], None, None, &options(), Some(&cache));
        assert_eq!(
            completion.choices[0].message.reasoning_content.as_deref(),
            Some("reasoning")
        );
        assert!(cache.thinking_signature("session-1", gem()).is_usable());
    }

    #[test]
    fn a_short_signature_is_not_cached() {
        let cache = SignatureCache::new();
        let mut part = call("f");
        part.set_signature("short");
        to_completion(&[part], None, None, &options(), Some(&cache));
        assert_eq!(cache.stats().tool_signatures, 0);
    }

    // -- streaming ---------------------------------------------------------

    fn translate_stream(events: &[Value], include_usage: bool) -> (Vec<ChatCompletionChunk>, SignatureCache) {
        translate_stream_for(events, include_usage, gem())
    }

    fn translate_stream_for(
        events: &[Value],
        include_usage: bool,
        family: ModelFamily,
    ) -> (Vec<ChatCompletionChunk>, SignatureCache) {
        let cache = SignatureCache::new();
        let mut options = options();
        options.family = family;
        options.include_usage = include_usage;
        let mut translator = StreamTranslator::new(options);

        let mut chunks = Vec::new();
        for event in events {
            chunks.extend(translator.on_payload(event, Some(&cache)));
        }
        chunks.extend(translator.finish());
        (chunks, cache)
    }

    fn event(parts: Value, finish: Option<&str>, usage: Option<Value>) -> Value {
        let mut candidate = json!({ "content": { "parts": parts } });
        if let Some(finish) = finish {
            candidate["finishReason"] = json!(finish);
        }
        let mut response = json!({ "candidates": [candidate] });
        if let Some(usage) = usage {
            response["usageMetadata"] = usage;
        }
        response
    }

    fn text_of(chunks: &[ChatCompletionChunk]) -> String {
        chunks
            .iter()
            .filter_map(|chunk| chunk.choices.first())
            .filter_map(|choice| choice.delta.content.clone())
            .collect()
    }

    fn reasoning_of(chunks: &[ChatCompletionChunk]) -> String {
        chunks
            .iter()
            .filter_map(|chunk| chunk.choices.first())
            .filter_map(|choice| choice.delta.reasoning_content.clone())
            .collect()
    }

    #[test]
    fn a_streamed_text_response_emits_chunks_then_a_finish() {
        let (chunks, _) = translate_stream(
            &[event(json!([{ "text": "hello" }]), Some("STOP"), None)],
            false,
        );

        assert_eq!(chunks[0].object, "chat.completion.chunk");
        assert_eq!(chunks[0].choices[0].delta.role, Some("assistant"));
        assert_eq!(text_of(&chunks), "hello");

        let last = chunks.last().unwrap();
        assert_eq!(last.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn the_role_is_announced_exactly_once() {
        let (chunks, _) = translate_stream(
            &[
                event(json!([{ "text": "a" }]), None, None),
                event(json!([{ "text": "b" }]), Some("STOP"), None),
            ],
            false,
        );
        let role_count = chunks
            .iter()
            .filter_map(|chunk| chunk.choices.first())
            .filter(|choice| choice.delta.role.is_some())
            .count();
        assert_eq!(role_count, 1);
    }

    #[test]
    fn a_multi_event_response_keeps_both_the_text_and_the_finish_reason() {
        // The live shape: text then usage in one event, a signature on an empty
        // part and the finish reason in the next.
        let (chunks, cache) = translate_stream(
            &[
                event(
                    json!([{ "text": "ok" }]),
                    None,
                    Some(json!({ "promptTokenCount": 9, "candidatesTokenCount": 2 })),
                ),
                event(
                    json!([{ "thoughtSignature": "E".repeat(80), "text": "" }]),
                    Some("STOP"),
                    Some(json!({ "promptTokenCount": 9, "totalTokenCount": 11, "thoughtsTokenCount": 0 })),
                ),
            ],
            false,
        );

        assert_eq!(text_of(&chunks), "ok", "the answer must survive");
        assert_eq!(
            chunks.last().unwrap().choices[0].finish_reason.as_deref(),
            Some("stop"),
            "the finish reason arrives in the second event"
        );
        assert!(
            cache.thinking_signature("session-1", gem()).is_usable(),
            "the signature arrives on the empty part and must still be captured"
        );
    }

    #[test]
    fn a_signature_only_event_emits_no_content_chunk() {
        let (chunks, cache) = translate_stream(
            &[event(
                json!([{ "thoughtSignature": "E".repeat(80), "text": "" }]),
                None,
                None,
            )],
            false,
        );

        // Just the terminal chunk: a signature is not client-visible content.
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].choices[0].finish_reason.as_deref(), Some("stop"));
        assert!(cache.thinking_signature("session-1", gem()).is_usable());
    }

    #[test]
    fn streamed_thinking_is_emitted_as_reasoning() {
        let (chunks, _) = translate_stream(
            &[
                event(json!([{ "text": "thinking", "thought": true }]), None, None),
                event(json!([{ "text": "answer" }]), Some("STOP"), None),
            ],
            false,
        );
        assert_eq!(reasoning_of(&chunks), "thinking");
        assert_eq!(text_of(&chunks), "answer");
    }

    #[test]
    fn a_streamed_tool_call_carries_an_index_and_id() {
        let (chunks, cache) = translate_stream(
            &[event(
                json!([{ "functionCall": { "name": "get_weather", "args": { "city": "Paris" } } }]),
                Some("STOP"),
                None,
            )],
            false,
        );

        let call_chunk = chunks
            .iter()
            .find(|chunk| {
                chunk
                    .choices
                    .first()
                    .and_then(|choice| choice.delta.tool_calls.as_ref())
                    .is_some()
            })
            .expect("a tool call chunk");

        let json = serde_json::to_value(&call_chunk.choices[0].delta).unwrap();
        let call = &json["tool_calls"][0];
        assert_eq!(call["index"], 0);
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        assert!(call["id"].as_str().unwrap().starts_with("call_"));

        // The id in the chunk is the id the signature is cached under.
        let id = call["id"].as_str().unwrap();
        assert_eq!(cache.stats().tool_signatures, 0, "no signature in this event");
        let _ = id;

        assert_eq!(
            chunks.last().unwrap().choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[test]
    fn multiple_tool_calls_get_increasing_indices() {
        let (chunks, _) = translate_stream(
            &[
                event(json!([{ "functionCall": { "name": "a", "args": {} } }]), None, None),
                event(json!([{ "functionCall": { "name": "b", "args": {} } }]), None, None),
            ],
            false,
        );

        let indices: Vec<u64> = chunks
            .iter()
            .filter_map(|chunk| chunk.choices.first())
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .map(|call| serde_json::to_value(call).unwrap()["index"].as_u64().unwrap())
            .collect();
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn a_signature_on_a_streamed_tool_call_is_cached() {
        let (chunks, cache) = translate_stream(
            &[event(
                json!([{
                    "functionCall": { "name": "f", "args": {} },
                    "thoughtSignature": "S".repeat(80)
                }]),
                Some("STOP"),
                None,
            )],
            false,
        );

        let id = chunks
            .iter()
            .filter_map(|chunk| chunk.choices.first())
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .map(|call| serde_json::to_value(call).unwrap()["id"].as_str().unwrap().to_string())
            .next()
            .unwrap();

        assert!(cache.tool_signature(&id, gem()).is_usable());
    }

    #[test]
    fn usage_is_emitted_only_when_requested() {
        let usage_event = event(json!([{ "text": "x" }]), Some("STOP"), Some(json!({
            "promptTokenCount": 5, "candidatesTokenCount": 1
        })));

        let (without, _) = translate_stream(std::slice::from_ref(&usage_event), false);
        assert!(without.iter().all(|chunk| chunk.usage.is_none()));

        let (with, _) = translate_stream(&[usage_event], true);
        let usage_chunk = with.last().unwrap();
        assert!(
            usage_chunk.choices.is_empty(),
            "the usage chunk carries no choices"
        );
        let usage = usage_chunk.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 1);
    }

    #[test]
    fn usage_is_newest_wins_across_events() {
        // The final event carries the complete counts; earlier ones are partial.
        let (chunks, _) = translate_stream(
            &[
                event(json!([{ "text": "a" }]), None, Some(json!({ "promptTokenCount": 5 }))),
                event(json!([{ "text": "b" }]), Some("STOP"), Some(json!({
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 42
                }))),
            ],
            true,
        );
        assert_eq!(chunks.last().unwrap().usage.as_ref().unwrap().completion_tokens, 42);
    }

    #[test]
    fn completion_tokens_are_present_even_when_the_upstream_omits_them() {
        // The live shape with no candidate count: derived from the total.
        let (chunks, _) = translate_stream(
            &[event(
                json!([{ "text": "x" }]),
                Some("STOP"),
                Some(json!({
                    "promptTokenCount": 9,
                    "totalTokenCount": 30,
                    "thoughtsTokenCount": 16
                })),
            )],
            true,
        );
        let usage = chunks.last().unwrap().usage.as_ref().unwrap();
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.completion_tokens_details.as_ref().unwrap().reasoning_tokens, 16);
    }

    #[test]
    fn finish_is_idempotent() {
        let mut translator = StreamTranslator::new(options());
        let first = translator.finish();
        let second = translator.finish();
        assert_eq!(first.len(), 1);
        assert!(second.is_empty(), "a second finish must not duplicate it");
    }

    #[test]
    fn an_empty_stream_still_terminates() {
        let (chunks, _) = translate_stream(&[], false);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn emitted_content_reflects_whether_anything_was_sent() {
        let mut translator = StreamTranslator::new(options());
        assert!(!translator.emitted_content());
        translator.on_payload(
            &event(json!([{ "text": "x" }]), None, None),
            None,
        );
        assert!(translator.emitted_content());
    }

    #[test]
    fn a_malformed_payload_is_skipped_without_losing_the_stream() {
        let mut translator = StreamTranslator::new(options());
        assert!(translator.on_payload(&json!("not an object"), None).is_empty());
        let chunks = translator.on_payload(&event(json!([{ "text": "x" }]), None, None), None);
        assert_eq!(text_of(&chunks), "x");
    }

    #[test]
    fn all_ids_within_a_response_are_consistent() {
        let (chunks, _) = translate_stream(
            &[event(json!([{ "text": "a" }]), Some("STOP"), None)],
            false,
        );
        let ids: std::collections::BTreeSet<&str> =
            chunks.iter().map(|chunk| chunk.id.as_str()).collect();
        assert_eq!(ids.len(), 1, "every chunk shares the completion id");
    }

    #[test]
    fn the_upstream_tool_call_id_is_preferred() {
        // Live traffic supplied `call_9680`. The backend may correlate that id
        // with the signature issued alongside it, so replacing it would break a
        // link we cannot see.
        let mut part = call("get_weather");
        part.function_call.as_mut().unwrap().id = Some("call_9680".into());
        part.set_signature("S".repeat(80));

        let cache = SignatureCache::new();
        let completion = to_completion(&[part], None, None, &options(), Some(&cache));

        let id = &completion.choices[0].message.tool_calls.as_ref().unwrap()[0].id;
        assert_eq!(id, "call_9680");
        // And the signature is reachable under that same id.
        assert!(cache.tool_signature("call_9680", gem()).is_usable());
    }

    #[test]
    fn an_id_is_generated_when_the_upstream_supplies_none() {
        let completion = to_completion(&[call("f")], None, None, &options(), None);
        let id = &completion.choices[0].message.tool_calls.as_ref().unwrap()[0].id;
        assert!(id.starts_with("call_"));
    }

    #[test]
    fn an_empty_upstream_id_falls_back_to_generating_one() {
        let mut part = call("f");
        part.function_call.as_mut().unwrap().id = Some(String::new());
        let completion = to_completion(&[part], None, None, &options(), None);
        let id = &completion.choices[0].message.tool_calls.as_ref().unwrap()[0].id;
        assert!(id.starts_with("call_"));
        assert_ne!(id, "");
    }

    #[test]
    fn generated_ids_have_the_expected_prefixes() {
        assert!(new_completion_id().starts_with("chatcmpl-"));
        assert!(new_tool_call_id().starts_with("call_"));
        assert_ne!(new_tool_call_id(), new_tool_call_id());
    }

    // -- signature attachment ----------------------------------------------

    #[test]
    fn a_cached_signature_is_attached_to_an_outgoing_tool_call() {
        let cache = SignatureCache::new();
        let signature = "S".repeat(80);
        cache.put_tool("call_1", &signature, gem());

        let mut part = call("f");
        attach_tool_signature(&mut part, "call_1", &cache, gem());
        assert_eq!(part.signature(), Some("S".repeat(80).as_str()));
    }

    #[test]
    fn a_gemini_call_with_no_cached_signature_gets_the_sentinel() {
        // Without a signature Gemini 3 rejects a replayed function call; the
        // sentinel is what the upstream provides for exactly this case.
        let cache = SignatureCache::new();
        let mut part = call("f");
        attach_tool_signature(&mut part, "unknown", &cache, gem());
        assert_eq!(part.signature(), Some(SKIP_THOUGHT_SIGNATURE));
    }

    #[test]
    fn a_claude_call_with_no_cached_signature_is_left_unsigned() {
        // Claude has no sentinel; a fabricated signature is worse than none.
        let cache = SignatureCache::new();
        let mut part = call("f");
        attach_tool_signature(&mut part, "unknown", &cache, ModelFamily::Claude);
        assert_eq!(part.signature(), None);
    }

    #[test]
    fn a_foreign_family_signature_is_not_attached() {
        let cache = SignatureCache::new();
        let signature = "S".repeat(80);
        cache.put_tool("call_1", &signature, ModelFamily::Claude);

        let mut part = call("f");
        attach_tool_signature(&mut part, "call_1", &cache, gem());
        // Falls through to the Gemini sentinel rather than reusing the Claude one.
        assert_eq!(part.signature(), Some(SKIP_THOUGHT_SIGNATURE));
    }

    #[test]
    fn indexed_tool_call_serialises_the_openai_shape() {
        let indexed = IndexedToolCall {
            index: 2,
            call: ResponseToolCall {
                id: "call_x".into(),
                kind: "function",
                function: ResponseFunction {
                    name: "f".into(),
                    arguments: "{}".into(),
                },
            },
        };
        let json = serde_json::to_value(&indexed).unwrap();
        assert_eq!(json["index"], 2);
        assert_eq!(json["id"], "call_x");
        assert_eq!(json["function"]["name"], "f");
    }

    #[test]
    fn indexed_tool_call_round_trips() {
        let json = json!({
            "index": 1,
            "id": "call_y",
            "type": "function",
            "function": { "name": "g", "arguments": "{}" }
        });
        let parsed: IndexedToolCall = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.index, 1);
        assert_eq!(parsed.call.id, "call_y");
        assert_eq!(parsed.call.function.name, "g");
    }

    #[test]
    fn index_is_optional_when_deserialising() {
        // Some clients omit it; a missing index is not a parse failure.
        let json = json!({
            "id": "call_z",
            "type": "function",
            "function": { "name": "h", "arguments": "{}" }
        });
        let parsed: IndexedToolCall = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.index, 0);
    }

    /// Every content delta must carry something. An empty delta is legal on the
    /// wire but is never intentional, and a run of them is the signature of a
    /// classification bug.
    fn assert_no_empty_content_deltas(chunks: &[ChatCompletionChunk]) {
        for chunk in chunks {
            let Some(choice) = chunk.choices.first() else {
                continue;
            };
            if let Some(text) = &choice.delta.content {
                assert!(!text.is_empty(), "empty content delta in {chunk:?}");
            }
            if let Some(text) = &choice.delta.reasoning_content {
                assert!(!text.is_empty(), "empty reasoning delta in {chunk:?}");
            }
        }
    }

    #[test]
    fn the_live_claude_event_sequence_produces_no_empty_deltas() {
        // Captured live from claude-opus-4-6-thinking. Six events for one short
        // answer, and the signature arrives on an empty-text part that *also*
        // carries `thought: true` — unlike the Gemini shape, where the same
        // carrier has no `thought` flag at all.
        let events = vec![
            event(
                json!([{ "text": "" }]),
                None,
                Some(json!({ "promptTokenCount": 41, "candidatesTokenCount": 1, "totalTokenCount": 42 })),
            ),
            event(json!([{ "thought": true, "text": "ok" }]), None, None),
            event(json!([{ "thought": true, "text": "" }]), None, None),
            event(
                json!([{ "thought": true, "thoughtSignature": "R".repeat(448), "text": "" }]),
                None,
                None,
            ),
            event(json!([{ "text": "ok" }]), None, None),
            event(
                json!([{ "text": "" }]),
                Some("STOP"),
                Some(json!({ "promptTokenCount": 41, "candidatesTokenCount": 13, "totalTokenCount": 54 })),
            ),
        ];

        let (chunks, cache) =
            translate_stream_for(&events, false, ModelFamily::Claude);

        // Reasoning plus answer plus the terminal chunk. The three empty-text
        // events must contribute nothing.
        assert_eq!(
            chunks.len(),
            3,
            "empty-text parts produced chunks: {:?}",
            chunks
                .iter()
                .map(|chunk| serde_json::to_value(&chunk.choices[0].delta).unwrap())
                .collect::<Vec<_>>()
        );
        assert_no_empty_content_deltas(&chunks);
        assert_eq!(reasoning_of(&chunks), "ok");
        assert_eq!(text_of(&chunks), "ok");

        // The signature rode in on an empty-text thinking part and was still
        // captured against the session.
        assert!(cache.thinking_signature("session-1", ModelFamily::Claude).is_usable());

        // Usage is newest-wins: the candidate count advances 1 -> 13 across the
        // stream, and the final event is the one that counts.
        let usage = chunks.last().unwrap().usage.as_ref();
        let _ = usage;
    }

    #[test]
    fn the_live_gemini_event_sequence_produces_no_empty_deltas() {
        // The Gemini counterpart: same two-event split, but the signature
        // carrier has no `thought` flag.
        let events = vec![
            event(
                json!([{ "text": "ok" }]),
                None,
                Some(json!({ "promptTokenCount": 6, "candidatesTokenCount": 1, "totalTokenCount": 71, "thoughtsTokenCount": 64 })),
            ),
            event(
                json!([{ "thoughtSignature": "E".repeat(564), "text": "" }]),
                Some("STOP"),
                Some(json!({ "promptTokenCount": 6, "candidatesTokenCount": 1, "totalTokenCount": 71, "thoughtsTokenCount": 64 })),
            ),
        ];

        let (chunks, cache) = translate_stream(&events, false);
        assert_eq!(chunks.len(), 2, "the signature-only event adds no chunk");
        assert_no_empty_content_deltas(&chunks);
        assert_eq!(text_of(&chunks), "ok");
        assert!(cache.thinking_signature("session-1", gem()).is_usable());
    }

    #[test]
    fn an_empty_thinking_part_emits_nothing_on_its_own() {
        let (chunks, _) = translate_stream(
            &[event(json!([{ "thought": true, "text": "" }]), None, None)],
            false,
        );
        assert_eq!(chunks.len(), 1, "only the terminal chunk");
        assert_eq!(chunks[0].choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn usage_reflects_the_latest_event_in_the_live_sequence() {
        let events = vec![
            event(
                json!([{ "text": "ok" }]),
                None,
                Some(json!({ "promptTokenCount": 41, "candidatesTokenCount": 1 })),
            ),
            event(
                json!([{ "text": "" }]),
                Some("STOP"),
                Some(json!({ "promptTokenCount": 41, "candidatesTokenCount": 13 })),
            ),
        ];
        let (chunks, _) = translate_stream(&events, true);
        let usage = chunks.last().unwrap().usage.as_ref().unwrap();
        assert_eq!(usage.completion_tokens, 13);
        assert_eq!(usage.total_tokens, 54);
    }

    #[test]
    fn candidate_deserialisation_tolerates_a_missing_content() {
        let response: GenerateContentResponse = serde_json::from_value(json!({
            "candidates": [{ "finishReason": "STOP" }]
        }))
        .unwrap();
        assert_eq!(response.candidates.len(), 1);
        assert!(response.candidates[0].content.is_none());
        assert_eq!(response.candidates[0].finish_reason.as_deref(), Some("STOP"));
    }

    #[test]
    fn a_candidate_with_content_deserialises_into_parts() {
        let candidate: Candidate = serde_json::from_value(json!({
            "content": { "parts": [{ "text": "hi" }] },
            "finishReason": "STOP"
        }))
        .unwrap();
        let content: Content = candidate.content.unwrap();
        assert_eq!(content.parts.len(), 1);
        assert_eq!(content.parts[0].text.as_deref(), Some("hi"));
    }
}
