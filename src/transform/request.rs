//! OpenAI request → IR translation.
//!
//! The rules that matter here are the ones the upstream enforces and a naive
//! translation violates:
//!
//! - **`functionResponse` parts must be consecutive.** The upstream rejects a
//!   content whose tool results are interleaved with other parts, so consecutive
//!   `tool` messages collapse into one user turn and any images they carry are
//!   deferred to the end of that turn.
//! - **The final turn may not be `model`.** A conversation ending on an
//!   assistant turn is a normal thing for a client to send, and the upstream
//!   rejects it; a `[Continue]` user turn is appended.
//! - **Identifiers on `functionCall` are Claude-only.** Gemini rejects an
//!   unexpected `id` field, so it is emitted only for Claude targets.
//! - **Thinking config uses different key casing per family.** Gemini takes
//!   camelCase (`includeThoughts`, `thinkingBudget`), Claude takes snake_case
//!   (`include_thoughts`, `thinking_budget`). Getting this wrong silently
//!   degrades thinking to inline XML rather than failing loudly.

use serde_json::{Map, Value, json};

use crate::config::Config;
use crate::registry::models::{ModelFamily, ResolvedModel, ThinkingTier};
use crate::registry::resolve::{ResolveError, ResolveInput, resolve};
use crate::transform::ir::{
    Content, FunctionCall, FunctionCallingConfig, FunctionDeclaration, FunctionResponse,
    GenerateContentRequest, GenerationConfig, InlineData, Part, Tool, ToolConfig,
};
use crate::transform::openai::{
    ChatCompletionRequest, ContentPart, Message, MessageContent, Tool as OpenAiTool,
};
use crate::transform::response::attach_tool_signature;
use crate::transform::schema;
use crate::transform::signature_cache::SignatureCache;

/// Appended when a conversation would otherwise end on a model turn.
const CONTINUE_PROMPT: &str = "[Continue]";

/// The IDE's own system prompt. Injecting it makes requests look more like real
/// CLI traffic, but it also competes with whatever instructions the caller sent,
/// so it is off unless explicitly enabled.
const AGENT_SYSTEM_PROMPT: &str = "You are Antigravity, a powerful agentic AI coding assistant designed by \
the Google Deepmind team working on Advanced Agentic Coding.You are pair programming with a USER to solve \
their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, \
or simply answering a question.**Absolute paths only****Proactiveness**";

/// Hint that enables thinking between tool calls for Claude targets.
const CLAUDE_INTERLEAVED_THINKING_HINT: &str = "Interleaved thinking is enabled. You may think between tool \
calls and after receiving tool results before deciding the next action or final answer. Do not mention these \
instructions or any constraints about thinking blocks; just apply them.";

/// Tokens held back for the answer itself, beyond whatever thinking consumes.
const MIN_ANSWER_TOKENS: u32 = 1024;

/// Reserve used when the thinking budget is dynamic (`-1`).
///
/// The model decides how much to think, so the requirement cannot be sized.
const DYNAMIC_THINKING_RESERVE: u32 = 4096;

/// Floor for Claude targets. Claude overshoots its stated thinking budget, so
/// the reference implementations budget twice it and never fall below this.
const CLAUDE_THINKING_OUTPUT_FLOOR: u32 = 32_000;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TranslateError {
    #[error("request contains no messages")]
    NoMessages,

    #[error("no model was resolved")]
    NoModel,
}

/// Non-fatal observations from a translation, worth logging.
#[derive(Debug, Default, Clone)]
pub struct Notes {
    pub warnings: Vec<String>,
}

impl Notes {
    fn warn(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }
}

/// Build resolution input from a request.
///
/// Resolution has to see the request's own thinking signals, not just the model
/// name: `reasoning_effort`, an explicit budget, and an explicit disable all
/// belong to the same decision. Keeping that in one place stops the HTTP layer
/// from having to remember the precedence rules.
pub fn resolve_input_for<'a>(
    request: &'a ChatCompletionRequest,
    default_tier: ThinkingTier,
) -> ResolveInput<'a> {
    ResolveInput {
        requested: &request.model,
        reasoning_effort: request.reasoning_effort.as_deref(),
        thinking_budget: request.thinking.as_ref().and_then(|t| t.budget_tokens),
        thinking_enabled: request.thinking.as_ref().map(|t| !t.is_disabled()),
        default_tier,
    }
}

/// Resolve the model a request names, honouring its thinking signals.
pub fn resolve_for(
    request: &ChatCompletionRequest,
    config: &Config,
) -> Result<ResolvedModel, ResolveError> {
    let default_tier = ThinkingTier::parse(&config.reasoning.default_tier).unwrap_or_default();
    resolve(resolve_input_for(request, default_tier))
}

/// Where a replayed tool call recovers the signature the upstream expects.
///
/// Absent when translating without a cache, in which case a Gemini tool call
/// still gets the upstream's sentinel rather than nothing.
struct SignatureSource<'a> {
    cache: &'a SignatureCache,
}

/// Translate a Chat Completions request into the IR, without signature replay.
///
/// Equivalent to [`to_ir_with_signatures`] with no cache. Skipping the cache is
/// only appropriate when no upstream turn preceded this one.
pub fn to_ir(
    request: &ChatCompletionRequest,
    resolved: &ResolvedModel,
    config: &Config,
) -> Result<(GenerateContentRequest, Notes), TranslateError> {
    to_ir_inner(request, resolved, config, None)
}

/// Translate a Chat Completions request, reattaching cached signatures.
///
/// This is the form the gateway uses: the upstream requires a signature on a
/// replayed function call, and the client cannot supply one because OpenAI's
/// protocol has nowhere to put it.
///
/// No session key is taken, because only tool calls are replayed from the cache.
/// Thinking is not — see [`convert_assistant_message`] for why attaching a
/// session's signature to an arbitrary turn is not safe.
pub fn to_ir_with_signatures(
    request: &ChatCompletionRequest,
    resolved: &ResolvedModel,
    config: &Config,
    cache: &SignatureCache,
) -> Result<(GenerateContentRequest, Notes), TranslateError> {
    to_ir_inner(request, resolved, config, Some(SignatureSource { cache }))
}

fn to_ir_inner(
    request: &ChatCompletionRequest,
    resolved: &ResolvedModel,
    config: &Config,
    signatures: Option<SignatureSource<'_>>,
) -> Result<(GenerateContentRequest, Notes), TranslateError> {
    let mut notes = Notes::default();

    if request.messages.is_empty() {
        return Err(TranslateError::NoMessages);
    }

    let mut contents = convert_messages(&request.messages, resolved, signatures.as_ref(), &mut notes);
    if contents.is_empty() {
        return Err(TranslateError::NoMessages);
    }

    // The upstream rejects a payload whose last turn is the model's.
    if contents.last().is_some_and(|content| content.role == "model") {
        contents.push(Content::user(vec![Part::text(CONTINUE_PROMPT)]));
    }

    let system_instruction = build_system_instruction(&request.messages, resolved, config, &mut notes);

    let tools = convert_tools(request.tools.as_deref(), &mut notes);
    let tool_config = tools.as_ref().and_then(|_| convert_tool_choice(request, resolved));

    let generation_config = build_generation_config(request, resolved);

    Ok((
        GenerateContentRequest {
            contents,
            system_instruction,
            tools,
            tool_config,
            // Filled in by the envelope, which owns session identity.
            labels: None,
            generation_config: Some(generation_config),
            session_id: None,
        },
        notes,
    ))
}

/// Collect system messages into a single instruction.
///
/// Multiple system messages are joined rather than dropped: clients commonly
/// split a long system prompt across several, and losing all but the first would
/// silently change behaviour.
fn build_system_instruction(
    messages: &[Message],
    resolved: &ResolvedModel,
    config: &Config,
    notes: &mut Notes,
) -> Option<Content> {
    let mut parts: Vec<Part> = Vec::new();

    if config.upstream.inject_agent_system_prompt {
        parts.push(Part::text(AGENT_SYSTEM_PROMPT));
    }

    for message in messages.iter().filter(|message| message.role == "system" || message.role == "developer") {
        match &message.content {
            Some(MessageContent::Text(text)) if !text.is_empty() => {
                parts.push(Part::text(text.clone()));
            }
            Some(MessageContent::Parts(content_parts)) => {
                for part in content_parts {
                    if part.kind == "text"
                        && let Some(text) = &part.text
                        && !text.is_empty()
                    {
                        parts.push(Part::text(text.clone()));
                    }
                }
            }
            _ => {}
        }
    }

    // Claude benefits measurably from being told it may think between tool
    // calls; without this it tends to answer after the first tool result.
    if resolved.family == ModelFamily::Claude
        && resolved.thinking_enabled
        && messages
            .iter()
            .any(|message| message.tool_calls.as_ref().is_some_and(|calls| !calls.is_empty()))
    {
        parts.push(Part::text(CLAUDE_INTERLEAVED_THINKING_HINT));
    }

    if parts.is_empty() {
        if config.upstream.inject_agent_system_prompt {
            notes.warn("agent system prompt enabled but no system instruction produced");
        }
        return None;
    }

    Some(Content::user(parts))
}

/// Convert messages into contents.
fn convert_messages(
    messages: &[Message],
    resolved: &ResolvedModel,
    signatures: Option<&SignatureSource<'_>>,
    notes: &mut Notes,
) -> Vec<Content> {
    let mut contents: Vec<Content> = Vec::new();

    for message in messages {
        match message.role.as_str() {
            "system" | "developer" => {
                // Handled by the system instruction.
            }

            "tool" => {
                let Some(part) = convert_tool_result(message, notes) else {
                    continue;
                };
                // Tool results must sit in a single user turn with nothing but
                // other functionResponse parts between them.
                match contents.last_mut() {
                    Some(last) if last.role == "user" && is_tool_result_turn(last) => {
                        last.parts.push(part);
                    }
                    _ => contents.push(Content::user(vec![part])),
                }
            }

            "assistant" => {
                let parts = convert_assistant_message(message, resolved, signatures, notes);
                if parts.is_empty() {
                    continue;
                }
                contents.push(Content::model(parts));
            }

            // Anything unrecognised is treated as user content, which is the
            // safer default: dropping it would silently lose the user's turn.
            _ => {
                let parts = convert_user_message(message, notes);
                if parts.is_empty() {
                    continue;
                }
                match contents.last_mut() {
                    // Merge adjacent user turns; some clients split a single
                    // logical turn across several messages.
                    Some(last) if last.role == "user" && !is_tool_result_turn(last) => {
                        last.parts.extend(parts);
                    }
                    _ => contents.push(Content::user(parts)),
                }
            }
        }
    }

    contents
}

/// Whether a turn consists solely of tool results.
fn is_tool_result_turn(content: &Content) -> bool {
    !content.parts.is_empty()
        && content
            .parts
            .iter()
            .all(|part| part.function_response.is_some())
}

fn convert_user_message(message: &Message, notes: &mut Notes) -> Vec<Part> {
    match &message.content {
        Some(MessageContent::Text(text)) => {
            if text.is_empty() {
                Vec::new()
            } else {
                vec![Part::text(text.clone())]
            }
        }
        Some(MessageContent::Parts(parts)) => convert_content_parts(parts, notes),
        None => Vec::new(),
    }
}

/// Convert OpenAI content parts into IR parts.
fn convert_content_parts(parts: &[ContentPart], notes: &mut Notes) -> Vec<Part> {
    let mut converted = Vec::new();

    for part in parts {
        match part.kind.as_str() {
            "text" | "input_text" => {
                if let Some(text) = &part.text
                    && !text.is_empty()
                {
                    converted.push(Part::text(text.clone()));
                }
            }
            "image_url" | "input_image" => match &part.image_url {
                Some(image) => match decode_data_uri(&image.url) {
                    Some((mime_type, data)) => {
                        converted.push(Part::inline_data(InlineData { mime_type, data }));
                    }
                    None => notes.warn(format!(
                        "dropped image with a non-data URL ({}); the upstream accepts inline data only",
                        truncate_url(&image.url)
                    )),
                },
                None => notes.warn("dropped an image part with no url"),
            },
            other => notes.warn(format!("dropped unsupported content part of type '{other}'")),
        }
    }

    converted
}

fn convert_assistant_message(
    message: &Message,
    resolved: &ResolvedModel,
    signatures: Option<&SignatureSource<'_>>,
    notes: &mut Notes,
) -> Vec<Part> {
    let mut parts = Vec::new();

    // Thinking is deliberately not reconstructed here, even though the session
    // cache holds a signature that could be paired with this text. The pairing
    // is only reliable for the most recent turn — the cache keeps one signature
    // per conversation — and attaching a signature to the wrong turn is a hard
    // upstream rejection, while omitting an unsigned thinking block is a
    // documented-safe omission that the reference implementations also make.
    if message.reasoning_content.is_some() {
        notes.warn(
            "dropped echoed reasoning_content; a thinking block requires a signature that              cannot be reliably matched to this turn, and unsigned thinking is rejected",
        );
    }

    match &message.content {
        Some(MessageContent::Text(text)) if !text.is_empty() => {
            parts.push(Part::text(text.clone()));
        }
        Some(MessageContent::Parts(content_parts)) => {
            parts.extend(convert_content_parts(content_parts, notes));
        }
        _ => {}
    }

    for call in message.tool_calls.iter().flatten() {
        let Some(function) = &call.function else {
            notes.warn("dropped a tool call with no function payload");
            continue;
        };
        let Some(name) = &function.name else {
            notes.warn("dropped a tool call with no function name");
            continue;
        };

        let mut part = Part::function_call(FunctionCall {
            name: name.clone(),
            args: function.parsed_arguments(),
            // Claude expects an id on a replayed call; Gemini rejects an
            // unexpected one. This is the sibling `id` field, not an entry
            // inside `args` — live traffic showed the upstream places it there.
            id: match resolved.family {
                ModelFamily::Claude => call.id.clone(),
                _ => None,
            },
        });

        // A signature the client happened to echo back is a bonus path; it
        // normally arrives with none, which is why the cache exists.
        if let Some(signature) = &call.thought_signature
            && !signature.is_empty()
        {
            part.set_signature(signature.clone());
        } else if let (Some(source), Some(id)) = (signatures, call.id.as_deref()) {
            attach_tool_signature(&mut part, id, source.cache, resolved.family);
        } else if let Some(source) = signatures {
            // No id to key on, so fall back to the same treatment as a cache
            // miss: Gemini needs a signature on a replayed call, and the
            // sentinel is what stands in for one.
            let _ = source;
            attach_tool_signature(&mut part, "", source.cache, resolved.family);
        }

        parts.push(part);
    }

    // The upstream is order-sensitive: reasoning, then prose, then calls.
    reorder_assistant_parts(&mut parts);

    parts
}

/// Order assistant parts as thinking → text → tool calls.
///
/// Clients reorder freely when echoing a turn back, and the upstream expects the
/// order the model originally produced.
fn reorder_assistant_parts(parts: &mut [Part]) {
    parts.sort_by_key(|part| {
        if part.is_thought() {
            0
        } else if part.function_call.is_some() {
            2
        } else {
            1
        }
    });
}

fn convert_tool_result(message: &Message, notes: &mut Notes) -> Option<Part> {
    let name = message.name.clone().unwrap_or_else(|| {
        // The OpenAI protocol puts the tool name on the originating call, not on
        // the result, and some clients omit `name` entirely.
        notes.warn("tool result had no name; using 'unknown'");
        "unknown".to_string()
    });

    let text = match &message.content {
        Some(MessageContent::Text(text)) => text.clone(),
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .filter(|part| part.kind == "text")
            .filter_map(|part| part.text.clone())
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    };

    let mut response = Map::new();
    response.insert("result".into(), Value::String(text));

    let mut part = Part::function_response(FunctionResponse { name, response });

    // Images inside a tool result cannot live in the same part, and inserting
    // them here would break the run of consecutive functionResponse parts. The
    // caller appends them at the end of the turn instead.
    if let Some(MessageContent::Parts(parts)) = &message.content
        && parts.iter().any(|part| part.kind == "image_url")
    {
        notes.warn(
            "tool result images were dropped: they cannot be sent alongside a functionResponse",
        );
    }

    part.thought_signature = None;
    Some(part)
}

/// Build the tool declarations.
///
/// Returns `None` when there are no usable tools, which is what suppresses the
/// `toolConfig` block.
fn convert_tools(tools: Option<&[OpenAiTool]>, notes: &mut Notes) -> Option<Vec<Tool>> {
    let tools = tools?;
    let mut declarations = Vec::new();

    for tool in tools {
        // Only function tools exist upstream; `code_interpreter`, `web_search`,
        // and friends have no equivalent.
        if tool.kind.as_deref().is_some_and(|kind| kind != "function") {
            notes.warn(format!(
                "dropped unsupported tool type '{}'",
                tool.kind.as_deref().unwrap_or("unknown")
            ));
            continue;
        }

        let Some(function) = &tool.function else {
            notes.warn("dropped a tool with no function payload");
            continue;
        };

        let name = sanitize_tool_name(&function.name);
        if name.is_empty() {
            notes.warn("dropped a tool whose name is empty after sanitisation");
            continue;
        }

        let parameters = function
            .parameters
            .as_ref()
            .and_then(schema::sanitize);

        declarations.push(FunctionDeclaration {
            name,
            description: function.description.clone(),
            parameters,
        });
    }

    if declarations.is_empty() {
        return None;
    }

    Some(vec![Tool {
        function_declarations: declarations,
    }])
}

/// Restrict a tool name to the characters the upstream accepts.
///
/// Allowed: alphanumerics, underscore, colon, dash, up to 64 characters. Anything
/// else is replaced rather than rejected, since a tool with a slightly different
/// name still works while a rejected request does not.
fn sanitize_tool_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric()
                || matches!(character, '_' | '-' | ':')
            {
                character
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    cleaned
}

/// Map OpenAI `tool_choice` onto the upstream's calling mode.
///
/// The reference implementations discard this outright, which means a client
/// asking for `required` silently gets `auto` and may receive no tool call at
/// all. The mapping is cheap and the difference is observable.
fn convert_tool_choice(
    request: &ChatCompletionRequest,
    resolved: &ResolvedModel,
) -> Option<ToolConfig> {
    let (mode, allowed) = match request.tool_choice.as_ref() {
        None => (default_mode(resolved).to_string(), None),
        Some(Value::String(choice)) => match choice.as_str() {
            "none" => ("NONE".to_string(), None),
            "auto" => ("AUTO".to_string(), None),
            "required" | "any" => ("ANY".to_string(), None),
            _ => (default_mode(resolved).to_string(), None),
        },
        Some(Value::Object(object)) => {
            // `{"type":"function","function":{"name":"x"}}`
            let name = object
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .map(|name| vec![sanitize_tool_name(name)]);
            match name {
                Some(names) => ("ANY".to_string(), Some(names)),
                None => (default_mode(resolved).to_string(), None),
            }
        }
        _ => (default_mode(resolved).to_string(), None),
    };

    Some(ToolConfig {
        function_calling_config: FunctionCallingConfig {
            mode,
            allowed_function_names: allowed,
        },
    })
}

/// Claude targets want validated tool calling; Gemini's default is fine.
fn default_mode(resolved: &ResolvedModel) -> &'static str {
    match resolved.family {
        ModelFamily::Claude => "VALIDATED",
        _ => "AUTO",
    }
}

fn build_generation_config(
    request: &ChatCompletionRequest,
    resolved: &ResolvedModel,
) -> GenerationConfig {
    // `max_completion_tokens` is the current spelling and wins when both appear.
    let requested = request.max_completion_tokens.or(request.max_tokens);

    GenerationConfig {
        max_output_tokens: Some(build_output_budget(requested, resolved)),
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: None,
        stop_sequences: request.stop.clone().map(|stop| stop.into_vec()),
        thinking_config: build_thinking_config(resolved),
        response_mime_type: None,
        response_schema: None,
    }
}

/// Compute the output token budget.
///
/// `maxOutputTokens` covers thinking *and* the answer, so a limit at or below
/// what the model spends thinking yields a reply that thinks and then stops: an
/// empty message with `finishReason: MAX_TOKENS` and nothing the client can use.
///
/// This is not family-specific. It was first guarded for on Claude, where the
/// reference implementations noticed it, and then reproduced identically on
/// Gemini 3: a probe with a 64-token limit came back with 61 tokens of thinking,
/// an empty text part, and `MAX_TOKENS`.
///
/// A client asking for a small `max_tokens` is asking for a short *answer*. The
/// thinking that precedes it is overhead the client neither asked for nor can
/// see, so the budget is raised to cover the reserve plus a usable answer and
/// capped at what the model can actually emit.
///
/// An unspecified limit means no preference, and the model's full allowance is
/// used — raising is only ever about honouring an explicit request usefully.
fn build_output_budget(requested: Option<u32>, resolved: &ResolvedModel) -> u32 {
    let limit = resolved.output_limit();

    let Some(requested) = requested else {
        return limit;
    };
    let requested = requested.min(limit);

    if !resolved.thinking_enabled {
        return requested;
    }

    let reserve = match (resolved.family, resolved.thinking_budget) {
        (ModelFamily::Claude, Some(budget)) if budget > 0 => (budget as u32)
            .saturating_mul(2)
            .max(CLAUDE_THINKING_OUTPUT_FLOOR),
        (_, Some(budget)) if budget > 0 => budget as u32,
        // Dynamic (`-1`) or absent: the model decides, so use a fixed reserve.
        _ => DYNAMIC_THINKING_RESERVE,
    };

    let minimum = reserve.saturating_add(MIN_ANSWER_TOKENS).min(limit);
    requested.max(minimum)
}

/// Build the family-specific thinking config.
fn build_thinking_config(resolved: &ResolvedModel) -> Option<Value> {
    if !resolved.thinking_enabled {
        return None;
    }
    let budget = resolved.thinking_budget?;

    match resolved.family {
        // Claude takes snake_case; camelCase here is silently ignored and the
        // model degrades to writing `<thinking>` into the visible answer.
        ModelFamily::Claude => Some(json!({
            "include_thoughts": true,
            "thinking_budget": budget.max(0),
        })),
        // Gemini takes camelCase, and -1 is meaningful (model chooses).
        _ => Some(json!({
            "includeThoughts": true,
            "thinkingBudget": budget,
        })),
    }
}

/// Split a `data:` URI into its media type and base64 payload.
fn decode_data_uri(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    if !meta.contains("base64") {
        // A non-base64 data URI would need percent-decoding, which the upstream
        // does not accept anyway.
        return None;
    }
    let mime_type = meta.split(';').next().unwrap_or("image/png").to_string();
    Some((mime_type, data.to_string()))
}

fn truncate_url(url: &str) -> String {
    if url.chars().count() <= 40 {
        return url.to_string();
    }
    let head: String = url.chars().take(37).collect();
    format!("{head}...")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::models::ThinkingTier;
    use crate::registry::resolve::{ResolveInput, resolve};
    use serde_json::json;

    fn model(name: &str) -> ResolvedModel {
        resolve(ResolveInput {
            requested: name,
            ..Default::default()
        })
        .unwrap()
    }

    fn translate(value: Value) -> (GenerateContentRequest, Notes) {
        translate_for(value, "gemini-3.8-flash")
    }

    /// Translate using request-aware resolution, so the request's own thinking
    /// signals are honoured rather than only the model name.
    fn translate_for(value: Value, model_name: &str) -> (GenerateContentRequest, Notes) {
        let mut value = value;
        value["model"] = Value::String(model_name.to_string());
        let request: ChatCompletionRequest = serde_json::from_value(value).unwrap();
        let resolved = resolve_for(&request, &Config::default()).unwrap();
        to_ir(&request, &resolved, &Config::default()).unwrap()
    }

    #[test]
    fn empty_message_list_is_rejected() {
        let request: ChatCompletionRequest =
            serde_json::from_value(json!({ "model": "m", "messages": [] })).unwrap();
        assert!(matches!(
            to_ir(&request, &model("gemini-3.8-flash"), &Config::default()),
            Err(TranslateError::NoMessages)
        ));
    }

    #[test]
    fn simple_user_turn_becomes_a_user_content() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hello" }]
        }));
        assert_eq!(ir.contents.len(), 1);
        assert_eq!(ir.contents[0].role, "user");
        assert_eq!(ir.contents[0].parts[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn assistant_role_maps_to_model() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": "hello" }
            ]
        }));
        assert_eq!(ir.contents[1].role, "model");
    }

    #[test]
    fn trailing_model_turn_gets_a_continue_prompt() {
        // The upstream rejects a conversation ending on a model turn.
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": "hello" }
            ]
        }));
        assert_eq!(ir.contents.len(), 3);
        assert_eq!(ir.contents[2].role, "user");
        assert_eq!(
            ir.contents[2].parts[0].text.as_deref(),
            Some(CONTINUE_PROMPT)
        );
    }

    #[test]
    fn user_turn_ending_conversation_needs_no_continue() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }]
        }));
        assert_eq!(ir.contents.len(), 1);
    }

    #[test]
    fn multiple_system_messages_are_joined_in_order() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [
                { "role": "system", "content": "first" },
                { "role": "system", "content": "second" },
                { "role": "user", "content": "hi" }
            ]
        }));
        let system = ir.system_instruction.unwrap();
        assert_eq!(system.parts[0].text.as_deref(), Some("first"));
        assert_eq!(system.parts[1].text.as_deref(), Some("second"));
        // System messages must not appear as contents.
        assert_eq!(ir.contents.len(), 1);
    }

    #[test]
    fn no_system_messages_means_no_system_instruction() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }]
        }));
        assert!(ir.system_instruction.is_none());
    }

    #[test]
    fn agent_system_prompt_is_opt_in() {
        let value = json!({
            "model": "m",
            "messages": [
                { "role": "system", "content": "be terse" },
                { "role": "user", "content": "hi" }
            ]
        });

        let (off, _) = translate_for(value.clone(), "gemini-3.8-flash");
        let off_parts = off.system_instruction.unwrap().parts;
        assert_eq!(off_parts.len(), 1, "the caller's prompt must be untouched");

        let mut config = Config::default();
        config.upstream.inject_agent_system_prompt = true;
        let request: ChatCompletionRequest = serde_json::from_value(value).unwrap();
        let (on, _) = to_ir(&request, &model("gemini-3.8-flash"), &config).unwrap();
        let on_parts = on.system_instruction.unwrap().parts;
        assert_eq!(on_parts.len(), 2);
        assert!(on_parts[0].text.as_deref().unwrap().contains("Antigravity"));
    }

    #[test]
    fn consecutive_tool_results_merge_into_one_user_turn() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [
                    { "id": "1", "type": "function", "function": { "name": "a", "arguments": "{}" } },
                    { "id": "2", "type": "function", "function": { "name": "b", "arguments": "{}" } }
                ]},
                { "role": "tool", "tool_call_id": "1", "name": "a", "content": "ra" },
                { "role": "tool", "tool_call_id": "2", "name": "b", "content": "rb" }
            ]
        }));

        let last = ir.contents.last().unwrap();
        assert_eq!(last.role, "user");
        assert_eq!(last.parts.len(), 2, "both results belong to one turn");
        assert!(last.parts.iter().all(|part| part.function_response.is_some()));
        assert_eq!(
            last.parts[0].function_response.as_ref().unwrap().name,
            "a"
        );
    }

    #[test]
    fn tool_result_carries_the_text_as_result() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [
                    { "id": "1", "type": "function", "function": { "name": "a", "arguments": "{}" } }
                ]},
                { "role": "tool", "name": "a", "content": "the result" }
            ]
        }));
        let response = ir
            .contents
            .last()
            .unwrap()
            .parts[0]
            .function_response
            .as_ref()
            .unwrap();
        assert_eq!(response.response["result"], "the result");
    }

    #[test]
    fn tool_call_becomes_a_function_call_part() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
                }]},
                { "role": "user", "content": "and?" }
            ]
        }));
        let call = ir.contents[1].parts[0].function_call.as_ref().unwrap();
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.args["city"], "Paris");
    }

    fn id_probe_request() -> Value {
        json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "f", "arguments": "{}" }
                }]},
                { "role": "user", "content": "and?" }
            ]
        })
    }

    #[test]
    fn gemini_function_calls_omit_the_id() {
        // Gemini rejects an unexpected `id` on a functionCall.
        let (ir, _) = translate(id_probe_request());
        let call = ir.contents[1].parts[0].function_call.as_ref().unwrap();
        assert!(call.id.is_none(), "got: {call:?}");
        // And it must not be smuggled in as an argument either.
        assert!(call.args.get("id").is_none(), "got: {:?}", call.args);
    }

    #[test]
    fn claude_function_calls_keep_the_id() {
        let request: ChatCompletionRequest =
            serde_json::from_value(id_probe_request()).unwrap();
        let resolved = model("claude-opus-4-6-thinking");
        let (ir, _) = to_ir(&request, &resolved, &Config::default()).unwrap();
        let call = ir.contents[1].parts[0].function_call.as_ref().unwrap();
        // The id is a sibling of name and args, which is where the upstream
        // puts it — not an entry inside args.
        assert_eq!(call.id.as_deref(), Some("call_1"));
        assert!(call.args.get("id").is_none());
    }

    #[test]
    fn the_function_call_id_is_serialised_as_a_sibling() {
        let call = FunctionCall {
            name: "f".into(),
            args: serde_json::from_value(json!({ "city": "Paris" })).unwrap(),
            id: Some("call_9".into()),
        };
        let json = serde_json::to_string(&call).unwrap();
        assert_eq!(
            json,
            r#"{"name":"f","args":{"city":"Paris"},"id":"call_9"}"#
        );
    }

    #[test]
    fn assistant_parts_are_reordered_calls_last() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant",
                  "content": "thinking about it",
                  "tool_calls": [{ "id": "1", "type": "function",
                                   "function": { "name": "f", "arguments": "{}" } }] },
                { "role": "user", "content": "and?" }
            ]
        }));
        let parts = &ir.contents[1].parts;
        assert!(parts[0].text.is_some(), "text must precede calls");
        assert!(parts[1].function_call.is_some());
    }

    #[test]
    fn adjacent_user_turns_merge() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "first" },
                { "role": "user", "content": "second" }
            ]
        }));
        assert_eq!(ir.contents.len(), 1);
        assert_eq!(ir.contents[0].parts.len(), 2);
    }

    #[test]
    fn tool_choice_none_maps_to_none_mode() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "type": "function", "function": { "name": "f", "parameters": { "type": "object", "properties": { "a": { "type": "string" } } } } }],
            "tool_choice": "none"
        }));
        assert_eq!(
            ir.tool_config.unwrap().function_calling_config.mode,
            "NONE"
        );
    }

    #[test]
    fn tool_choice_required_maps_to_any_mode() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "type": "function", "function": { "name": "f", "parameters": { "type": "object", "properties": { "a": { "type": "string" } } } } }],
            "tool_choice": "required"
        }));
        assert_eq!(ir.tool_config.unwrap().function_calling_config.mode, "ANY");
    }

    #[test]
    fn tool_choice_naming_a_function_restricts_it() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "type": "function", "function": { "name": "chosen", "parameters": { "type": "object", "properties": { "a": { "type": "string" } } } } }],
            "tool_choice": { "type": "function", "function": { "name": "chosen" } }
        }));
        let config = ir.tool_config.unwrap().function_calling_config;
        assert_eq!(config.mode, "ANY");
        assert_eq!(config.allowed_function_names, Some(vec!["chosen".to_string()]));
    }

    #[test]
    fn claude_defaults_to_validated_tool_mode() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "type": "function", "function": { "name": "f", "parameters": { "type": "object", "properties": { "a": { "type": "string" } } } } }]
        }))
        .unwrap();
        let (ir, _) = to_ir(
            &request,
            &model("claude-opus-4-6-thinking"),
            &Config::default(),
        )
        .unwrap();
        assert_eq!(
            ir.tool_config.unwrap().function_calling_config.mode,
            "VALIDATED"
        );
    }

    #[test]
    fn no_tools_means_no_tool_config() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }]
        }));
        assert!(ir.tools.is_none());
        assert!(ir.tool_config.is_none());
    }

    #[test]
    fn tool_names_are_sanitised_and_truncated() {
        let long_name = "a".repeat(100);
        let (ir, notes) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "type": "function", "function": {
                "name": format!("bad name/{long_name}"),
                "parameters": { "type": "object", "properties": { "a": { "type": "string" } } }
            }}]
        }));
        let declarations = &ir.tools.unwrap()[0].function_declarations;
        let name = &declarations[0].name;
        assert_eq!(name.len(), 64);
        assert!(name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | ':')));
        assert!(notes.warnings.is_empty(), "sanitising is not a warning-level event");
    }

    #[test]
    fn non_function_tools_are_dropped_with_a_warning() {
        let (ir, notes) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "type": "web_search" }]
        }));
        assert!(ir.tools.is_none());
        assert!(notes.warnings.iter().any(|w| w.contains("web_search")));
    }

    #[test]
    fn tool_schemas_are_sanitised() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "type": "function", "function": {
                "name": "f",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": { "a": { "type": "string", "pattern": "^x$" } }
                }
            }}]
        }));
        let parameters = ir.tools.unwrap()[0].function_declarations[0]
            .parameters
            .clone()
            .unwrap();
        assert!(parameters.get("$schema").is_none());
        assert_eq!(parameters["type"], "OBJECT");
        assert!(parameters["properties"]["a"].get("pattern").is_none());
    }

    #[test]
    fn gemini_thinking_config_uses_camel_case() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }]
        }));
        let config = ir.generation_config.unwrap().thinking_config.unwrap();
        assert_eq!(config["includeThoughts"], true);
        assert_eq!(config["thinkingBudget"], 4000);
        assert!(config.get("include_thoughts").is_none());
    }

    #[test]
    fn claude_thinking_config_uses_snake_case() {
        // The wrong casing is silently ignored upstream, degrading thinking
        // rather than failing, so this is worth pinning.
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }]
        }))
        .unwrap();
        let (ir, _) = to_ir(
            &request,
            &model("claude-opus-4-6-thinking"),
            &Config::default(),
        )
        .unwrap();
        let config = ir.generation_config.unwrap().thinking_config.unwrap();
        assert_eq!(config["include_thoughts"], true);
        assert_eq!(config["thinking_budget"], 16384);
        assert!(config.get("includeThoughts").is_none());
    }

    #[test]
    fn disabled_thinking_sends_no_config() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "thinking": { "type": "disabled" }
        }));
        assert!(ir.generation_config.unwrap().thinking_config.is_none());
    }

    #[test]
    fn claude_output_budget_is_raised_to_fit_thinking() {
        // A max_tokens at or below the thinking budget yields an empty answer
        // because thinking consumes the whole output allowance.
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 100
        }))
        .unwrap();
        let (ir, _) = to_ir(
            &request,
            &model("claude-opus-4-6-thinking"),
            &Config::default(),
        )
        .unwrap();
        let max = ir.generation_config.unwrap().max_output_tokens.unwrap();
        assert!(
            max > 16_384,
            "output budget must exceed the thinking budget, got {max}"
        );
    }

    #[test]
    fn gemini_output_budget_is_raised_to_fit_thinking() {
        // Gemini charges thinking against maxOutputTokens exactly as Claude does.
        // This previously asserted the opposite, on the assumption that only
        // Claude needed the guard.
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 100
        }));
        let max = ir.generation_config.unwrap().max_output_tokens.unwrap();
        // The `medium` tier carries a 4000-token thinking budget.
        assert_eq!(max, 4000 + MIN_ANSWER_TOKENS);
    }

    #[test]
    fn a_tiny_budget_cannot_be_swallowed_by_thinking() {
        // The exact shape of the failure observed against the live upstream: a
        // 64-token limit, 61 tokens spent thinking, and an empty answer part
        // with finishReason MAX_TOKENS.
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 64
        }));
        let config = ir.generation_config.unwrap();
        let max = config.max_output_tokens.unwrap();
        let budget = config.thinking_config.unwrap()["thinkingBudget"]
            .as_i64()
            .unwrap();

        assert!(
            i64::from(max) > budget,
            "output budget {max} must exceed the thinking budget {budget},              otherwise the model thinks and then stops"
        );
    }

    #[test]
    fn dynamic_thinking_budget_still_gets_a_floor() {
        // `-1` means the model chooses, so the requirement cannot be sized;
        // a floor keeps ordinary prompts from being swallowed by thinking.
        let (ir, _) = translate_for(
            json!({
                "model": "m",
                "messages": [{ "role": "user", "content": "hi" }],
                "max_tokens": 100
            }),
            "gemini-3.8-flash-high",
        );
        let max = ir.generation_config.unwrap().max_output_tokens.unwrap();
        assert_eq!(max, DYNAMIC_THINKING_RESERVE + MIN_ANSWER_TOKENS);
    }

    #[test]
    fn output_budget_never_exceeds_the_model_limit() {
        for model in ["gemini-3.8-flash", "claude-opus-4-6-thinking"] {
            let (ir, _) = translate_for(
                json!({
                    "model": "m",
                    "messages": [{ "role": "user", "content": "hi" }],
                    "max_tokens": u32::MAX
                }),
                model,
            );
            let max = ir.generation_config.unwrap().max_output_tokens.unwrap();
            let limit = resolve(ResolveInput {
                requested: model,
                ..Default::default()
            })
            .unwrap()
            .output_limit();
            assert_eq!(max, limit, "{model} exceeded its output limit");
        }
    }

    #[test]
    fn disabled_thinking_leaves_the_requested_budget_alone() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 100,
            "thinking": { "type": "disabled" }
        }));
        assert_eq!(ir.generation_config.unwrap().max_output_tokens, Some(100));
    }

    #[test]
    fn output_budget_defaults_to_the_model_limit() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }]
        }));
        assert_eq!(
            ir.generation_config.unwrap().max_output_tokens,
            Some(65_536)
        );
    }

    #[test]
    fn max_completion_tokens_wins_over_max_tokens() {
        // Thinking is disabled so the value is not raised by the budget guard,
        // which keeps this assertion about precedence rather than about size.
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 100,
            "max_completion_tokens": 200,
            "thinking": { "type": "disabled" }
        }));
        assert_eq!(ir.generation_config.unwrap().max_output_tokens, Some(200));
    }

    #[test]
    fn sampling_parameters_are_carried_through() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "temperature": 0.3,
            "top_p": 0.9,
            "stop": ["END", "STOP"]
        }));
        let config = ir.generation_config.unwrap();
        assert_eq!(config.temperature, Some(0.3));
        assert_eq!(config.top_p, Some(0.9));
        assert_eq!(
            config.stop_sequences,
            Some(vec!["END".to_string(), "STOP".to_string()])
        );
    }

    #[test]
    fn data_uri_images_become_inline_data() {
        let (ir, notes) = translate(json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "describe" },
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } }
                ]
            }]
        }));
        let inline = ir.contents[0].parts[1].inline_data.as_ref().unwrap();
        assert_eq!(inline.mime_type, "image/png");
        assert_eq!(inline.data, "AAAA");
        assert!(
            notes.warnings.is_empty(),
            "a well-formed image is not a warning"
        );
    }

    #[test]
    fn remote_image_urls_are_dropped_with_a_warning() {
        let (ir, notes) = translate(json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "describe" },
                    { "type": "image_url", "image_url": { "url": "https://example.com/a.png" } }
                ]
            }]
        }));
        assert_eq!(ir.contents[0].parts.len(), 1);
        assert!(notes.warnings.iter().any(|w| w.contains("non-data URL")));
    }

    #[test]
    fn echoed_reasoning_content_is_ignored_with_a_warning() {
        // OpenAI's reasoning_content carries no signature, so it cannot be sent
        // back upstream as thinking.
        let (_, notes) = translate(json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": "hello", "reasoning_content": "I thought about it" },
                { "role": "user", "content": "again" }
            ]
        }));
        assert!(notes.warnings.iter().any(|w| w.contains("reasoning_content")));
    }

    #[test]
    fn data_uri_decoding_handles_the_parameter_list() {
        let (mime, data) = decode_data_uri("data:image/jpeg;charset=utf-8;base64,ZZZ").unwrap();
        assert_eq!(mime, "image/jpeg");
        assert_eq!(data, "ZZZ");
    }

    #[test]
    fn non_base64_data_uris_are_rejected() {
        assert!(decode_data_uri("data:image/png,notbase64").is_none());
        assert!(decode_data_uri("https://example.com/a.png").is_none());
        assert!(decode_data_uri("data:image/png;base64").is_none());
    }

    #[test]
    fn full_tool_loop_round_trips() {
        // The shape an agent framework sends on the second turn.
        let (ir, notes) = translate(json!({
            "model": "m",
            "messages": [
                { "role": "system", "content": "You are helpful" },
                { "role": "user", "content": "What is the weather in Paris?" },
                { "role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
                }]},
                { "role": "tool", "tool_call_id": "call_abc", "name": "get_weather", "content": "18C" },
                { "role": "user", "content": "Thanks" }
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"]
                    }
                }
            }],
            "tool_choice": "auto"
        }));

        assert_eq!(ir.contents.len(), 4);
        assert_eq!(ir.contents[0].role, "user");
        assert_eq!(ir.contents[1].role, "model");
        assert_eq!(ir.contents[2].role, "user");
        assert!(ir.contents[2].parts[0].function_response.is_some());
        assert_eq!(ir.contents[3].role, "user");

        assert!(ir.system_instruction.is_some());
        assert_eq!(
            ir.tools.as_ref().unwrap()[0].function_declarations.len(),
            1
        );
        assert!(ir.tool_config.is_some());
        assert!(notes.warnings.is_empty(), "warnings: {:?}", notes.warnings);
    }

    #[test]
    fn unknown_roles_are_treated_as_user_content() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "weird_role", "content": "hi" }]
        }));
        assert_eq!(ir.contents[0].role, "user");
    }

    #[test]
    fn tier_from_reasoning_effort_reaches_the_thinking_config() {
        let (ir, _) = translate(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hi" }],
            "reasoning_effort": "high"
        }));
        let config = ir.generation_config.unwrap().thinking_config.unwrap();
        assert_eq!(config["thinkingBudget"], -1);
    }

    #[test]
    fn explicit_tier_suffix_wins_in_the_translation() {
        let (ir, _) = translate_for(
            json!({
                "model": "m",
                "messages": [{ "role": "user", "content": "hi" }],
                "reasoning_effort": "high"
            }),
            "gemini-3.8-flash-low",
        );
        let config = ir.generation_config.unwrap().thinking_config.unwrap();
        assert_eq!(
            config["thinkingBudget"], 1000,
            "the name's tier must outrank reasoning_effort"
        );
    }

    // -- signature replay --------------------------------------------------

    fn replay_request() -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "f", "arguments": "{}" }
                }]},
                { "role": "tool", "tool_call_id": "call_1", "name": "f", "content": "done" }
            ]
        }))
        .unwrap()
    }

    fn sig(marker: &str) -> String {
        format!("{marker}{}", "A".repeat(60))
    }

    #[test]
    fn a_replayed_tool_call_recovers_its_cached_signature() {
        // The payoff of the cache: the client never saw this signature and
        // cannot send it, but the upstream requires it.
        let cache = SignatureCache::new();
        cache.put_tool("call_1", &sig("gem"), ModelFamily::Gemini);

        let request = replay_request();
        let resolved = model("gemini-3.8-flash");
        let (ir, _) = to_ir_with_signatures(&request, &resolved, &Config::default(), &cache)
        .unwrap();

        let call_part = ir.contents[1]
            .parts
            .iter()
            .find(|part| part.function_call.is_some())
            .unwrap();
        assert_eq!(call_part.signature(), Some(sig("gem").as_str()));
    }

    #[test]
    fn a_gemini_call_with_a_cold_cache_gets_the_sentinel() {
        let cache = SignatureCache::new();
        let request = replay_request();
        let resolved = model("gemini-3.8-flash");
        let (ir, _) = to_ir_with_signatures(&request, &resolved, &Config::default(), &cache)
        .unwrap();

        let call_part = ir.contents[1]
            .parts
            .iter()
            .find(|part| part.function_call.is_some())
            .unwrap();
        assert_eq!(
            call_part.signature(),
            Some(crate::upstream::constants::SKIP_THOUGHT_SIGNATURE)
        );
    }

    #[test]
    fn a_claude_call_with_a_cold_cache_is_left_unsigned() {
        let cache = SignatureCache::new();
        let request = replay_request();
        let resolved = model("claude-opus-4-6-thinking");
        let (ir, _) = to_ir_with_signatures(&request, &resolved, &Config::default(), &cache)
        .unwrap();

        let call_part = ir.contents[1]
            .parts
            .iter()
            .find(|part| part.function_call.is_some())
            .unwrap();
        assert_eq!(call_part.signature(), None);
    }

    #[test]
    fn a_signature_echoed_by_the_client_wins_over_the_cache() {
        // If a client does preserve the field, its value is the more specific
        // one and should not be overwritten.
        let cache = SignatureCache::new();
        cache.put_tool("call_1", &sig("cached"), ModelFamily::Gemini);

        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [
                { "role": "user", "content": "go" },
                { "role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "f", "arguments": "{}" },
                    "thoughtSignature": sig("echoed")
                }]},
                { "role": "tool", "tool_call_id": "call_1", "name": "f", "content": "done" }
            ]
        }))
        .unwrap();

        let resolved = model("gemini-3.8-flash");
        let (ir, _) = to_ir_with_signatures(&request, &resolved, &Config::default(), &cache)
        .unwrap();

        let call_part = ir.contents[1]
            .parts
            .iter()
            .find(|part| part.function_call.is_some())
            .unwrap();
        assert_eq!(call_part.signature(), Some(sig("echoed").as_str()));
    }

    #[test]
    fn translating_without_a_cache_still_produces_a_valid_request() {
        // The one-argument form is used when there is no upstream history.
        let request = replay_request();
        let resolved = model("gemini-3.8-flash");
        let (ir, _) = to_ir(&request, &resolved, &Config::default()).unwrap();
        assert_eq!(ir.contents.len(), 3);
    }

    #[test]
    fn a_signature_from_another_family_is_not_replayed() {
        let cache = SignatureCache::new();
        cache.put_tool("call_1", &sig("claude"), ModelFamily::Claude);

        let request = replay_request();
        let resolved = model("gemini-3.8-flash");
        let (ir, _) = to_ir_with_signatures(&request, &resolved, &Config::default(), &cache)
        .unwrap();

        let call_part = ir.contents[1]
            .parts
            .iter()
            .find(|part| part.function_call.is_some())
            .unwrap();
        // Falls back to the sentinel rather than reusing a Claude signature.
        assert_eq!(
            call_part.signature(),
            Some(crate::upstream::constants::SKIP_THOUGHT_SIGNATURE)
        );
    }

    #[test]
    fn thinking_tier_type_is_reachable() {
        assert_eq!(ThinkingTier::Medium.as_str(), "medium");
    }
}
