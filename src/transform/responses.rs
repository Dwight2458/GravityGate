//! OpenAI Responses API translation.
//!
//! The Responses API is a different protocol from Chat Completions, not a
//! renamed one. Three differences shape everything below:
//!
//! - **Input items are typed, not just role-tagged.** A conversation is a list of
//!   `message`, `function_call`, and `function_call_output` items, and a tool
//!   call is a *sibling* of the message that produced it rather than a field on
//!   it.
//! - **Output is a list of items.** Reasoning, messages, and function calls each
//!   become their own item with their own id, and the client is expected to
//!   track them by id.
//! - **Streaming uses named SSE events** with a `sequence_number`, unlike Chat
//!   Completions which sends only `data:` lines. A client that ignores the event
//!   name and only parses `data:` will still work, because the type is also
//!   inside the payload.
//!
//! The request side is converted into a [`ChatCompletionRequest`] and handed to
//! the same pipeline the other route uses. That is deliberate: translating twice
//! would mean two places to keep correct, and the shared path already handles
//! signature replay, account rotation, and the empty-response retry.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::transform::openai::ChatCompletionRequest;
use crate::transform::ir::{GenerateContentResponse, Part, UsageMetadata};
use crate::transform::response::{PartKind, ResponseOptions, classify};

/// A request to `POST /v1/responses`.
///
/// Fields the gateway does not act on are accepted and ignored rather than
/// rejected: clients send supersets, and refusing an unknown key would make the
/// route unusable with real SDKs.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    /// A string, or a list of typed input items.
    #[serde(default)]
    pub input: Option<Value>,
    /// System-level instructions, the analogue of a system message.
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    /// Responses-style tools, which differ from Chat Completions: `name`,
    /// `description`, and `parameters` sit at the top level, not under a
    /// `function` key.
    #[serde(default)]
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// `{"effort": "low" | "medium" | "high"}`.
    #[serde(default)]
    pub reasoning: Option<Value>,
    /// End-user identifier, used here as a session hint.
    #[serde(default)]
    pub user: Option<String>,
    /// Accepted and ignored. Storing responses server-side is not something a
    /// stateless gateway does, and pretending otherwise would be worse.
    #[serde(default)]
    pub store: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesTool {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
    /// Some clients nest the definition the Chat Completions way instead.
    #[serde(default)]
    pub function: Option<Value>,
}

/// Convert a Responses request into the Chat Completions shape.
///
/// Everything downstream — translation to the IR, dispatch, signature replay —
/// then works unchanged.
pub fn to_chat_request(request: &ResponsesRequest) -> Result<ChatCompletionRequest, String> {
    let mut messages: Vec<Value> = Vec::new();

    // `instructions` is a system message by another name, and comes first.
    if let Some(instructions) = request.instructions.as_deref()
        && !instructions.is_empty()
    {
        messages.push(json!({ "role": "system", "content": instructions }));
    }

    match request.input.as_ref() {
        // A bare string is the whole user turn.
        Some(Value::String(text)) => {
            messages.push(json!({ "role": "user", "content": text }));
        }
        Some(Value::Array(items)) => {
            for item in items {
                messages.push(convert_input_item(item)?);
            }
        }
        // No input at all is a client error, not an empty conversation.
        Some(other) => {
            return Err(format!(
                "input must be a string or an array of items, got {}",
                type_name(other)
            ));
        }
        None => return Err("input is required".into()),
    }

    if messages.is_empty() {
        return Err("input produced no messages".into());
    }

    let tools = request.tools.as_ref().map(|tools| {
        tools
            .iter()
            .filter_map(convert_tool)
            .collect::<Vec<Value>>()
    });

    let reasoning_effort = request
        .reasoning
        .as_ref()
        .and_then(|reasoning| reasoning.get("effort"))
        .and_then(Value::as_str)
        .map(str::to_string);

    let mut converted = json!({
        "model": request.model,
        "messages": messages,
        "stream": request.stream,
    });

    if let Some(max) = request.max_output_tokens {
        converted["max_tokens"] = json!(max);
    }
    if let Some(temperature) = request.temperature {
        converted["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.top_p {
        converted["top_p"] = json!(top_p);
    }
    if let Some(tools) = tools.filter(|tools| !tools.is_empty()) {
        converted["tools"] = json!(tools);
    }
    if let Some(choice) = &request.tool_choice {
        converted["tool_choice"] = choice.clone();
    }
    if let Some(effort) = reasoning_effort {
        converted["reasoning_effort"] = json!(effort);
    }
    if let Some(user) = &request.user {
        converted["user"] = json!(user);
    }

    serde_json::from_value(converted)
        .map_err(|error| format!("could not convert the request: {error}"))
}

/// Convert one input item into a chat message.
fn convert_input_item(item: &Value) -> Result<Value, String> {
    let Some(object) = item.as_object() else {
        return Err(format!("input items must be objects, got {}", type_name(item)));
    };

    // An item with no `type` is a plain message, which is what most clients
    // send. The typed forms are checked first because `message` is also a type.
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message");

    match kind {
        "message" => {
            let role = object
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user");
            let content = object.get("content").cloned().unwrap_or(Value::Null);
            Ok(json!({ "role": role, "content": convert_content(&content) }))
        }

        // A tool call the model made, being replayed on a later turn.
        "function_call" => {
            let call_id = object
                .get("call_id")
                .or_else(|| object.get("id"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let name = object.get("name").and_then(Value::as_str).unwrap_or_default();
            let arguments = object.get("arguments").cloned().unwrap_or(json!("{}"));
            Ok(json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": { "name": name, "arguments": arguments }
                }]
            }))
        }

        // The result of that call, sent back as a tool message.
        "function_call_output" => {
            let call_id = object
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let output = match object.get("output") {
                Some(Value::String(text)) => text.clone(),
                Some(other) => other.to_string(),
                None => String::new(),
            };
            Ok(json!({
                "role": "tool",
                "tool_call_id": call_id,
                // The Responses protocol does not repeat the tool name on the
                // result; the Chat Completions shape wants one, and the name is
                // recovered from the call it answers further up the stack.
                "name": object.get("name").and_then(Value::as_str).unwrap_or(""),
                "content": output
            }))
        }

        // Reasoning items are the model's own output being echoed back. They
        // carry no signature in this protocol, so they cannot be replayed as
        // thinking; the signature cache handles that on the Chat Completions
        // path where signatures do round-trip.
        "reasoning" => Ok(json!({ "role": "assistant", "content": null })),

        other => Err(format!("unsupported input item type '{other}'")),
    }
}

/// Flatten Responses content parts into a chat content value.
fn convert_content(content: &Value) -> Value {
    match content {
        Value::String(_) | Value::Null => content.clone(),
        Value::Array(parts) => {
            let converted: Vec<Value> = parts
                .iter()
                .filter_map(|part| {
                    let kind = part.get("type").and_then(Value::as_str)?;
                    match kind {
                        "input_text" | "output_text" | "text" | "summary_text" => part
                            .get("text")
                            .map(|text| json!({ "type": "text", "text": text })),
                        "input_image" => part
                            .get("image_url")
                            .map(|url| json!({ "type": "image_url", "image_url": { "url": url } })),
                        _ => None,
                    }
                })
                .collect();
            json!(converted)
        }
        other => json!(other.to_string()),
    }
}

/// Convert one Responses tool into the Chat Completions shape.
///
/// The two protocols disagree on where the definition lives: Responses puts
/// `name` at the top level, Chat Completions nests it under `function`. Clients
/// in the wild do both, so both are accepted.
fn convert_tool(tool: &ResponsesTool) -> Option<Value> {
    if tool.kind.as_deref().is_some_and(|kind| kind != "function") {
        return None;
    }
    if let Some(function) = &tool.function {
        return Some(json!({ "type": "function", "function": function }));
    }
    let name = tool.name.as_ref()?;
    Some(json!({
        "type": "function",
        "function": {
            "name": name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    }))
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// A response object, as returned by the non-streaming route and by
/// `response.completed`.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub status: &'static str,
    pub model: String,
    pub output: Vec<Value>,
    pub parallel_tool_calls: bool,
    pub tool_choice: Value,
    pub tools: Vec<Value>,
    pub usage: ResponsesUsage,
    /// Present only when `status` is `incomplete`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<Value>,
    /// Always null: this gateway does not store responses, so there is nothing
    /// to reference.
    pub error: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ResponsesUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub total_tokens: i64,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens_details: OutputTokensDetails,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct InputTokensDetails {
    pub cached_tokens: i64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: i64,
}

impl ResponsesUsage {
    pub fn from_ir(usage: &UsageMetadata) -> Self {
        let input = usage.prompt_tokens();
        let output = usage.completion_tokens();
        Self {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            input_tokens_details: InputTokensDetails {
                cached_tokens: usage.cached_content_token_count,
            },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: usage.thoughts_token_count,
            },
        }
    }
}

/// Build a fresh response id.
pub fn new_response_id() -> String {
    use rand::Rng as _;
    let mut bytes = [0u8; 8];
    rand::rng().fill_bytes(&mut bytes);
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!("resp_{hex}")
}

fn item_id(prefix: &str) -> String {
    use rand::Rng as _;
    let mut bytes = [0u8; 8];
    rand::rng().fill_bytes(&mut bytes);
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!("{prefix}_{hex}")
}

/// Build the output items for a finished response.
///
/// Reasoning comes first, then the message, then the tool calls — the order the
/// model produced them and the order the protocol expects.
pub fn build_output(parts: &[Part]) -> Vec<Value> {
    let mut thinking = String::new();
    let mut text = String::new();
    let mut calls: Vec<Value> = Vec::new();

    for part in parts {
        match classify(part) {
            PartKind::Thinking(content) => thinking.push_str(content),
            PartKind::Text(content) => text.push_str(content),
            PartKind::ToolCall { name, args, .. } => {
                let call_id = part
                    .function_call
                    .as_ref()
                    .and_then(|call| call.id.clone())
                    .unwrap_or_else(|| item_id("call"));
                calls.push(json!({
                    "type": "function_call",
                    "id": item_id("fc"),
                    "call_id": call_id,
                    "name": name,
                    "arguments": serde_json::to_string(args).unwrap_or_else(|_| "{}".into()),
                    "status": "completed"
                }));
            }
            PartKind::SignatureOnly | PartKind::Unsupported | PartKind::Empty => {}
        }
    }

    let mut output = Vec::new();

    if !thinking.trim().is_empty() {
        output.push(json!({
            "type": "reasoning",
            "id": item_id("rs"),
            "summary": [{ "type": "summary_text", "text": thinking }],
            "status": "completed"
        }));
    }

    if !text.is_empty() || calls.is_empty() {
        output.push(json!({
            "type": "message",
            "id": item_id("msg"),
            "role": "assistant",
            "status": "completed",
            "content": if text.is_empty() {
                Vec::new()
            } else {
                vec![json!({ "type": "output_text", "text": text, "annotations": [] })]
            }
        }));
    }

    output.extend(calls);
    output
}

/// Turn an assembled upstream response into a client-facing response object.
pub fn to_response(
    parts: &[Part],
    usage: &UsageMetadata,
    finish_reason: Option<&str>,
    options: &ResponseOptions,
    request: &ResponsesRequest,
) -> ResponsesResponse {
    let output = build_output(parts);

    // The protocol distinguishes a finished response from one the model ran out
    // of room for, and clients act on the difference — a truncated answer that
    // claims to be complete is worse than one that admits it.
    let (status, incomplete_details) = match finish_reason {
        Some("MAX_TOKENS") => (
            "incomplete",
            Some(json!({ "reason": "max_output_tokens" })),
        ),
        _ => ("completed", None),
    };

    ResponsesResponse {
        id: options.completion_id.replace("chatcmpl-", "resp_"),
        object: "response",
        created_at: options.created,
        status,
        model: request.model.clone(),
        output,
        parallel_tool_calls: true,
        tool_choice: request.tool_choice.clone().unwrap_or(json!("auto")),
        tools: Vec::new(),
        usage: ResponsesUsage::from_ir(usage),
        incomplete_details,
        error: None,
    }
}

/// Streaming event state machine.
///
/// Emits the protocol's named events in order, with a monotonic
/// `sequence_number`. The bounds that matter:
///
/// - A message item is opened at most once, on the first text, and closed once
///   at the end. Emitting `output_item.added` per delta would produce a client
///   that renders one message per token.
/// - Item ids are stable across the deltas that belong to them, because that is
///   how a client associates a delta with the item it extends.
#[derive(Debug)]
pub struct ResponsesStream {
    response_id: String,
    model: String,
    created_at: i64,
    sequence: u64,
    output_index: u32,
    /// Item id of the open message, if any.
    message_item: Option<String>,
    /// Text accumulated for the message, for `output_text.done`.
    message_text: String,
    /// Reasoning item emitted so far.
    reasoning_item: Option<String>,
    /// Completed output items, for the terminal event.
    output: Vec<Value>,
    /// Tool calls, keyed by the index they were emitted at.
    calls: Vec<Value>,
    usage: Option<UsageMetadata>,
    started: bool,
}

impl ResponsesStream {
    pub fn new(response_id: String, model: String, created_at: i64) -> Self {
        Self {
            response_id,
            model,
            created_at,
            sequence: 0,
            output_index: 0,
            message_item: None,
            message_text: String::new(),
            reasoning_item: None,
            output: Vec::new(),
            calls: Vec::new(),
            usage: None,
            started: false,
        }
    }

    fn next_sequence(&mut self) -> u64 {
        let current = self.sequence;
        self.sequence += 1;
        current
    }

    /// A `response` object for the opening events.
    ///
    /// Carries no usage and no output, because neither exists yet. Reporting a
    /// partial usage figure at creation would be worse than reporting none: a
    /// client that reads it there gets a number that is simply wrong.
    fn opening_object(&self) -> Value {
        json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "in_progress",
            "model": self.model,
            "output": [],
            "usage": Value::Null,
        })
    }

    /// The terminal `response` object.
    fn completed_object(&self) -> Value {
        json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "completed",
            "model": self.model,
            "output": self.output,
            "usage": self.usage.as_ref().map(ResponsesUsage::from_ir),
        })
    }

    /// Events emitted once, before any content.
    pub fn start(&mut self) -> Vec<Value> {
        if self.started {
            return Vec::new();
        }
        self.started = true;
        vec![
            self.event("response.created", json!({ "response": self.opening_object() })),
            self.event("response.in_progress", json!({ "response": self.opening_object() })),
        ]
    }

    fn event(&mut self, kind: &str, body: Value) -> Value {
        let mut event = json!({
            "type": kind,
            "sequence_number": self.next_sequence(),
        });
        if let (Some(event), Some(body)) = (event.as_object_mut(), body.as_object()) {
            for (key, value) in body {
                event.insert(key.clone(), value.clone());
            }
        }
        event
    }

    /// Absorb one upstream event, returning the events to send.
    pub fn on_response(&mut self, response: &GenerateContentResponse) -> Vec<Value> {
        if let Some(usage) = &response.usage_metadata
            && usage.is_present()
        {
            self.usage = Some(usage.clone());
        }

        let mut events = self.start();

        for candidate in &response.candidates {
            let Some(content) = &candidate.content else {
                continue;
            };
            for part in &content.parts {
                match classify(part) {
                    PartKind::Thinking(text) => {
                        self.open_reasoning(&mut events);
                        events.push(self.event(
                            "response.reasoning_summary_text.delta",
                            json!({
                                "item_id": self.reasoning_item,
                                "output_index": self.output_index,
                                "summary_index": 0,
                                "delta": text,
                            }),
                        ));
                    }
                    PartKind::Text(text) => {
                        self.open_message(&mut events);
                        self.message_text.push_str(text);
                        events.push(self.event(
                            "response.output_text.delta",
                            json!({
                                "item_id": self.message_item,
                                "output_index": self.output_index,
                                "content_index": 0,
                                "delta": text,
                            }),
                        ));
                    }
                    PartKind::ToolCall { name, args, .. } => {
                        self.close_message(&mut events);
                        events.extend(self.open_call(name, part, args));
                    }
                    PartKind::SignatureOnly | PartKind::Unsupported | PartKind::Empty => {}
                }
            }
        }

        events
    }

    /// Open the reasoning item, if it is not already open.
    ///
    /// Announce it like any other item: a client tracks items by the
    /// `output_item.added` event, and one that appears only in the final output
    /// has no id to associate its deltas with.
    fn open_reasoning(&mut self, events: &mut Vec<Value>) {
        if self.reasoning_item.is_some() {
            return;
        }
        self.output_index = self.output.len() as u32;
        let id = item_id("rs");
        self.reasoning_item = Some(id.clone());

        let item = json!({
            "type": "reasoning",
            "id": id,
            "summary": [],
            "status": "completed"
        });
        events.push(self.event(
            "response.output_item.added",
            json!({ "output_index": self.output_index, "item": item.clone() }),
        ));
        self.output.push(item);
    }

    /// Open the message item, if it is not already open.
    fn open_message(&mut self, events: &mut Vec<Value>) {
        if self.message_item.is_some() {
            return;
        }
        // The index of an item is how many have been emitted before it. Deriving
        // it beats incrementing a counter and hoping the paths agree.
        self.output_index = self.output.len() as u32;
        let id = item_id("msg");
        self.message_item = Some(id.clone());

        events.push(self.event(
            "response.output_item.added",
            json!({
                "output_index": self.output_index,
                "item": {
                    "type": "message",
                    "id": id,
                    "role": "assistant",
                    "status": "in_progress",
                    "content": []
                }
            }),
        ));
        events.push(self.event(
            "response.content_part.added",
            json!({
                "item_id": id,
                "output_index": self.output_index,
                "content_index": 0,
                "part": { "type": "output_text", "text": "", "annotations": [] }
            }),
        ));
    }

    /// Close the open message, if any.
    fn close_message(&mut self, events: &mut Vec<Value>) {
        let Some(id) = self.message_item.take() else {
            return;
        };
        let text = std::mem::take(&mut self.message_text);

        events.push(self.event(
            "response.output_text.done",
            json!({
                "item_id": id,
                "output_index": self.output_index,
                "content_index": 0,
                "text": text,
            }),
        ));

        let item = json!({
            "type": "message",
            "id": id,
            "role": "assistant",
            "status": "completed",
            "content": if text.is_empty() {
                Vec::new()
            } else {
                vec![json!({ "type": "output_text", "text": text, "annotations": [] })]
            }
        });
        events.push(self.event(
            "response.output_item.done",
            json!({ "output_index": self.output_index, "item": item.clone() }),
        ));
        self.output.push(item);
    }

    /// Emit a tool call in full.
    ///
    /// The upstream delivers arguments whole, so there is nothing to stream
    /// incrementally; the delta and done events carry the same text.
    fn open_call(&mut self, name: &str, part: &Part, args: &serde_json::Map<String, Value>) -> Vec<Value> {
        let call_id = part
            .function_call
            .as_ref()
            .and_then(|call| call.id.clone())
            .unwrap_or_else(|| item_id("call"));
        let arguments = serde_json::to_string(args).unwrap_or_else(|_| "{}".into());
        self.output_index = self.output.len() as u32;
        let id = item_id("fc");

        let item = json!({
            "type": "function_call",
            "id": id,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed"
        });

        let events = vec![
            self.event(
                "response.output_item.added",
                json!({
                    "output_index": self.output_index,
                    "item": {
                        "type": "function_call",
                        "id": id,
                        "call_id": call_id,
                        "name": name,
                        "arguments": "",
                        "status": "in_progress"
                    }
                }),
            ),
            self.event(
                "response.function_call_arguments.delta",
                json!({
                    "item_id": id,
                    "output_index": self.output_index,
                    "delta": arguments,
                }),
            ),
            self.event(
                "response.function_call_arguments.done",
                json!({
                    "item_id": id,
                    "output_index": self.output_index,
                    "name": name,
                    "call_id": call_id,
                    "arguments": arguments,
                }),
            ),
            self.event(
                "response.output_item.done",
                json!({ "output_index": self.output_index, "item": item.clone() }),
            ),
        ];

        self.calls.push(item.clone());
        self.output.push(item);
        events
    }

    /// The terminal events: close the open item, then complete.
    pub fn finish(&mut self) -> Vec<Value> {
        let mut events = self.start();
        self.close_message(&mut events);
        events.push(self.event(
            "response.completed",
            json!({ "response": self.completed_object() }),
        ));
        events
    }

    pub fn usage(&self) -> Option<&UsageMetadata> {
        self.usage.as_ref()
    }

    pub fn has_output(&self) -> bool {
        !self.output.is_empty()
    }

    pub fn response_id(&self) -> &str {
        &self.response_id
    }

    pub fn output(&self) -> &[Value] {
        &self.output
    }
}

/// Format one event as an SSE frame.
///
/// Named events, unlike the Chat Completions route: the protocol expects the
/// event name to be present, even though it also appears in the payload.
pub fn format_event(event: &Value) -> String {
    let kind = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message");
    format!("event: {kind}\ndata: {event}\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ModelFamily;
    use crate::transform::ir::{FunctionCall, Part};

    fn parse(value: Value) -> ResponsesRequest {
        serde_json::from_value(value).unwrap()
    }

    fn chat(value: Value) -> ChatCompletionRequest {
        to_chat_request(&parse(value)).unwrap()
    }

    // -- request conversion -------------------------------------------------

    #[test]
    fn a_string_input_becomes_one_user_message() {
        let converted = chat(json!({ "model": "m", "input": "hello" }));
        assert_eq!(converted.messages.len(), 1);
        assert_eq!(converted.messages[0].role, "user");
    }

    #[test]
    fn instructions_become_a_system_message_first() {
        let converted = chat(json!({
            "model": "m", "instructions": "be terse", "input": "hello"
        }));
        assert_eq!(converted.messages.len(), 2);
        assert_eq!(converted.messages[0].role, "system");
        assert_eq!(converted.messages[1].role, "user");
    }

    #[test]
    fn an_empty_instructions_field_is_skipped() {
        let converted = chat(json!({ "model": "m", "instructions": "", "input": "hi" }));
        assert_eq!(converted.messages.len(), 1);
    }

    #[test]
    fn typed_input_items_become_messages() {
        let converted = chat(json!({
            "model": "m",
            "input": [
                { "type": "message", "role": "user", "content": "first" },
                { "type": "message", "role": "assistant", "content": "second" }
            ]
        }));
        assert_eq!(converted.messages.len(), 2);
        assert_eq!(converted.messages[0].role, "user");
        assert_eq!(converted.messages[1].role, "assistant");
    }

    #[test]
    fn an_item_without_a_type_is_treated_as_a_message() {
        // Most clients omit it.
        let converted = chat(json!({
            "model": "m",
            "input": [{ "role": "user", "content": "hello" }]
        }));
        assert_eq!(converted.messages[0].role, "user");
    }

    #[test]
    fn a_function_call_item_becomes_a_tool_call() {
        let converted = chat(json!({
            "model": "m",
            "input": [
                { "type": "message", "role": "user", "content": "weather?" },
                { "type": "function_call", "call_id": "call_1", "name": "get_weather",
                  "arguments": "{\"city\":\"Paris\"}" },
                { "type": "function_call_output", "call_id": "call_1", "output": "18C" }
            ]
        }));

        let assistant = &converted.messages[1];
        assert_eq!(assistant.role, "assistant");
        let calls = assistant.tool_calls.as_ref().expect("a tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(
            calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );

        let result = &converted.messages[2];
        assert_eq!(result.role, "tool");
        assert_eq!(result.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn an_item_id_is_used_when_there_is_no_call_id() {
        let converted = chat(json!({
            "model": "m",
            "input": [{ "type": "function_call", "id": "fc_1", "name": "f", "arguments": "{}" }]
        }));
        let calls = converted.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].id.as_deref(), Some("fc_1"));
    }

    #[test]
    fn content_parts_are_flattened() {
        let converted = chat(json!({
            "model": "m",
            "input": [{ "role": "user", "content": [
                { "type": "input_text", "text": "look" }
            ] }]
        }));
        let content = converted.messages[0].content.as_ref().unwrap();
        assert!(matches!(
            content,
            crate::transform::openai::MessageContent::Parts(_)
        ));
    }

    #[test]
    fn a_reasoning_item_becomes_an_empty_assistant_turn() {
        // It carries no signature in this protocol, so it cannot be replayed as
        // thinking; the surrounding turns still need it for ordering.
        let converted = chat(json!({
            "model": "m",
            "input": [
                { "type": "reasoning", "id": "rs_1", "summary": [] },
                { "role": "user", "content": "hi" }
            ]
        }));
        assert_eq!(converted.messages.len(), 2);
        assert_eq!(converted.messages[0].role, "assistant");
    }

    #[test]
    fn responses_tools_are_reshaped_for_the_chat_layer() {
        // Responses puts `name` at the top level; the chat shape nests it.
        let converted = chat(json!({
            "model": "m", "input": "hi",
            "tools": [{ "type": "function", "name": "get_weather",
                        "description": "d", "parameters": { "type": "object" } }]
        }));
        let tools = converted.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools[0].function.as_ref().unwrap().name,
            "get_weather"
        );
    }

    #[test]
    fn an_already_nested_tool_is_accepted() {
        let converted = chat(json!({
            "model": "m", "input": "hi",
            "tools": [{ "type": "function", "function": { "name": "f", "parameters": {} } }]
        }));
        assert_eq!(converted.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn reasoning_effort_is_carried_through() {
        let converted = chat(json!({
            "model": "m", "input": "hi", "reasoning": { "effort": "high" }
        }));
        assert_eq!(converted.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn output_limits_and_sampling_are_carried_through() {
        let converted = chat(json!({
            "model": "m", "input": "hi",
            "max_output_tokens": 512, "temperature": 0.3, "top_p": 0.9
        }));
        assert_eq!(converted.max_tokens, Some(512));
        assert_eq!(converted.temperature, Some(0.3));
        assert_eq!(converted.top_p, Some(0.9));
    }

    #[test]
    fn a_missing_input_is_rejected() {
        let error = to_chat_request(&parse(json!({ "model": "m" }))).unwrap_err();
        assert!(error.contains("required"), "got {error}");
    }

    #[test]
    fn a_non_string_non_array_input_is_rejected_with_the_type_named() {
        let error = to_chat_request(&parse(json!({ "model": "m", "input": 42 }))).unwrap_err();
        assert!(error.contains("number"), "got {error}");
    }

    #[test]
    fn an_unsupported_item_type_is_rejected() {
        let error = to_chat_request(&parse(json!({
            "model": "m", "input": [{ "type": "web_search_call" }]
        })))
        .unwrap_err();
        assert!(error.contains("web_search_call"), "got {error}");
    }

    #[test]
    fn unknown_top_level_fields_are_ignored() {
        // The field set includes `store`, `metadata`, and others this gateway
        // does not act on; rejecting them would break real SDKs.
        let converted = chat(json!({
            "model": "m", "input": "hi",
            "store": false, "metadata": { "a": 1 }, "truncation": "auto",
            "parallel_tool_calls": true, "some_future_field": [1, 2]
        }));
        assert_eq!(converted.messages.len(), 1);
    }

    // -- output -------------------------------------------------------------

    fn options() -> ResponseOptions {
        ResponseOptions {
            completion_id: "chatcmpl-test".into(),
            model: "m".into(),
            created: 1,
            family: ModelFamily::Gemini,
            reasoning_field: crate::config::ReasoningField::ReasoningContent,
            include_usage: false,
            session_key: "s".into(),
        }
    }

    fn call_part(name: &str) -> Part {
        Part::function_call(FunctionCall {
            name: name.into(),
            args: serde_json::from_value(json!({ "city": "Paris" })).unwrap(),
            id: Some("call_1".into()),
        })
    }

    #[test]
    fn a_text_response_becomes_a_message_item() {
        let output = build_output(&[Part::text("hello")]);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["content"][0]["type"], "output_text");
        assert_eq!(output[0]["content"][0]["text"], "hello");
    }

    #[test]
    fn thinking_becomes_a_reasoning_item_before_the_message() {
        let output = build_output(&[Part::thought_text("hmm"), Part::text("answer")]);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["summary"][0]["text"], "hmm");
        assert_eq!(output[1]["type"], "message");
    }

    #[test]
    fn a_tool_call_becomes_a_function_call_item() {
        let output = build_output(&[call_part("get_weather")]);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "function_call");
        assert_eq!(output[0]["name"], "get_weather");
        assert_eq!(output[0]["call_id"], "call_1");
        let args: Value = serde_json::from_str(output[0]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn a_tool_only_turn_has_no_message_item() {
        // A message item with empty content would make a client render an empty
        // bubble alongside the call.
        let output = build_output(&[call_part("f")]);
        assert!(output.iter().all(|item| item["type"] != "message"));
    }

    #[test]
    fn an_empty_response_still_produces_a_message_item() {
        let output = build_output(&[]);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["content"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn item_ids_are_unique_within_a_response() {
        let output = build_output(&[
            Part::thought_text("t"),
            Part::text("a"),
            call_part("f"),
        ]);
        let ids: std::collections::BTreeSet<&str> =
            output.iter().filter_map(|item| item["id"].as_str()).collect();
        assert_eq!(ids.len(), output.len());
    }

    #[test]
    fn usage_maps_onto_the_responses_shape() {
        let usage = UsageMetadata {
            prompt_token_count: 100,
            cached_content_token_count: 40,
            candidates_token_count: Some(20),
            thoughts_token_count: 15,
            total_token_count: Some(135),
        };
        let converted = ResponsesUsage::from_ir(&usage);
        assert_eq!(converted.input_tokens, 60);
        assert_eq!(converted.output_tokens, 35);
        assert_eq!(converted.total_tokens, 95);
        assert_eq!(converted.input_tokens_details.cached_tokens, 40);
        assert_eq!(converted.output_tokens_details.reasoning_tokens, 15);
    }

    #[test]
    fn output_indices_are_sequential_and_match_the_output() {
        // A client routes deltas by index; a gap or a repeat associates a delta
        // with the wrong item.
        let mut stream = stream();
        let events = absorb(
            &mut stream,
            json!({ "candidates": [{ "content": { "parts": [
                { "text": "thinking", "thought": true },
                { "text": "answer" },
                { "functionCall": { "name": "f", "args": {}, "id": "c1" } }
            ] } }] }),
        );

        let added: Vec<u64> = events
            .iter()
            .filter(|event| event["type"] == "response.output_item.added")
            .filter_map(|event| event["output_index"].as_u64())
            .collect();
        assert_eq!(added, vec![0, 1, 2], "got {added:?}");
    }

    #[test]
    fn a_truncated_response_reports_itself_as_incomplete() {
        let response = to_response(
            &[Part::text("cut off")],
            &UsageMetadata::default(),
            Some("MAX_TOKENS"),
            &options(),
            &parse(json!({ "model": "m", "input": "hi" })),
        );
        assert_eq!(response.status, "incomplete");
        assert_eq!(
            response.incomplete_details.unwrap()["reason"],
            "max_output_tokens"
        );
    }

    #[test]
    fn a_stopped_response_has_no_incomplete_details() {
        let response = to_response(
            &[Part::text("done")],
            &UsageMetadata::default(),
            Some("STOP"),
            &options(),
            &parse(json!({ "model": "m", "input": "hi" })),
        );
        assert_eq!(response.status, "completed");
        assert!(response.incomplete_details.is_none());
    }

    #[test]
    fn the_response_id_uses_the_protocols_prefix() {
        let response = to_response(
            &[Part::text("x")],
            &UsageMetadata::default(),
            None,
            &options(),
            &parse(json!({ "model": "m", "input": "hi" })),
        );
        assert!(response.id.starts_with("resp_"), "got {}", response.id);
        assert_eq!(response.object, "response");
        assert_eq!(response.status, "completed");
    }

    // -- streaming ----------------------------------------------------------

    fn stream() -> ResponsesStream {
        ResponsesStream::new("resp_test".into(), "m".into(), 1)
    }

    fn absorb(stream: &mut ResponsesStream, body: Value) -> Vec<Value> {
        let response: GenerateContentResponse = serde_json::from_value(body).unwrap();
        stream.on_response(&response)
    }

    fn kinds(events: &[Value]) -> Vec<String> {
        events
            .iter()
            .filter_map(|event| event["type"].as_str().map(str::to_string))
            .collect()
    }

    #[test]
    fn the_first_events_open_the_response() {
        let mut stream = stream();
        let events = absorb(
            &mut stream,
            json!({ "candidates": [{ "content": { "parts": [{ "text": "hi" }] } }] }),
        );
        let kinds = kinds(&events);
        assert_eq!(kinds[0], "response.created");
        assert_eq!(kinds[1], "response.in_progress");
        assert_eq!(kinds[2], "response.output_item.added");
        assert_eq!(kinds[3], "response.content_part.added");
        assert_eq!(kinds[4], "response.output_text.delta");
    }

    #[test]
    fn the_response_is_opened_only_once() {
        let mut stream = stream();
        absorb(&mut stream, json!({ "candidates": [{ "content": { "parts": [{ "text": "a" }] } }] }));
        let second = absorb(
            &mut stream,
            json!({ "candidates": [{ "content": { "parts": [{ "text": "b" }] } }] }),
        );
        assert!(
            !kinds(&second).contains(&"response.created".to_string()),
            "created must not repeat"
        );
    }

    #[test]
    fn the_message_item_is_opened_once_across_many_deltas() {
        // One message item per token would make a client render a message per
        // token.
        let mut stream = stream();
        let mut events = Vec::new();
        for text in ["a", "b", "c"] {
            events.extend(absorb(
                &mut stream,
                json!({ "candidates": [{ "content": { "parts": [{ "text": text }] } }] }),
            ));
        }
        let opens = kinds(&events)
            .into_iter()
            .filter(|kind| kind == "response.output_item.added")
            .count();
        assert_eq!(opens, 1, "got {opens} item opens");
    }

    #[test]
    fn sequence_numbers_are_monotonic_and_unique() {
        let mut stream = stream();
        let mut events = absorb(
            &mut stream,
            json!({ "candidates": [{ "content": { "parts": [
                { "text": "t", "thought": true }, { "text": "a" }
            ] } }] }),
        );
        events.extend(stream.finish());

        let numbers: Vec<u64> = events
            .iter()
            .filter_map(|event| event["sequence_number"].as_u64())
            .collect();
        assert!(numbers.len() >= events.len() - 1);
        for window in numbers.windows(2) {
            assert!(window[1] > window[0], "sequence went backwards: {numbers:?}");
        }
    }

    #[test]
    fn thinking_streams_as_reasoning_summary_deltas() {
        let mut stream = stream();
        let events = absorb(
            &mut stream,
            json!({ "candidates": [{ "content": { "parts": [
                { "text": "thinking", "thought": true }
            ] } }] }),
        );
        let kinds = kinds(&events);
        assert!(kinds.contains(&"response.reasoning_summary_text.delta".to_string()));
    }

    #[test]
    fn a_tool_call_streams_a_full_argument_sequence() {
        let mut stream = stream();
        let events = absorb(
            &mut stream,
            json!({ "candidates": [{ "content": { "parts": [
                { "functionCall": { "name": "f", "args": { "a": 1 }, "id": "call_9" } }
            ] } }] }),
        );
        let kinds = kinds(&events);
        assert!(kinds.contains(&"response.output_item.added".to_string()));
        assert!(kinds.contains(&"response.function_call_arguments.delta".to_string()));
        assert!(kinds.contains(&"response.function_call_arguments.done".to_string()));
        assert!(kinds.contains(&"response.output_item.done".to_string()));

        let done = events
            .iter()
            .find(|event| event["type"] == "response.function_call_arguments.done")
            .unwrap();
        assert_eq!(done["call_id"], "call_9");
        assert_eq!(done["name"], "f");
    }

    #[test]
    fn finishing_closes_the_message_and_completes() {
        let mut stream = stream();
        absorb(&mut stream, json!({ "candidates": [{ "content": { "parts": [{ "text": "hi" }] } }] }));
        let events = stream.finish();
        let kinds = kinds(&events);
        assert!(kinds.contains(&"response.output_text.done".to_string()));
        assert!(kinds.contains(&"response.output_item.done".to_string()));
        assert_eq!(kinds.last().unwrap(), "response.completed");
    }

    #[test]
    fn the_done_event_carries_the_accumulated_text() {
        // The reference implementation sends an empty string here, which makes
        // the event useless to a client that relies on it.
        let mut stream = stream();
        for text in ["he", "llo"] {
            absorb(&mut stream, json!({ "candidates": [{ "content": { "parts": [{ "text": text }] } }] }));
        }
        let events = stream.finish();
        let done = events
            .iter()
            .find(|event| event["type"] == "response.output_text.done")
            .unwrap();
        assert_eq!(done["text"], "hello");
    }

    #[test]
    fn the_completed_event_carries_the_output() {
        let mut stream = stream();
        absorb(&mut stream, json!({ "candidates": [{ "content": { "parts": [{ "text": "hi" }] } }] }));
        let events = stream.finish();
        let completed = events
            .iter()
            .find(|event| event["type"] == "response.completed")
            .unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "the completed event must describe the output");
        assert_eq!(output[0]["type"], "message");
    }

    #[test]
    fn the_opening_events_carry_no_usage() {
        // Usage is not known when the response opens, and a client that reads it
        // there would get a wrong number rather than none.
        let mut stream = stream();
        let events = absorb(
            &mut stream,
            json!({
                "candidates": [{ "content": { "parts": [{ "text": "hi" }] } }],
                "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 4 }
            }),
        );
        let created = events
            .iter()
            .find(|event| event["type"] == "response.created")
            .unwrap();
        assert!(created["response"]["usage"].is_null());
        assert_eq!(
            created["response"]["output"].as_array().unwrap().len(),
            0
        );
    }

    #[test]
    fn the_completed_event_carries_usage() {
        let mut stream = stream();
        absorb(
            &mut stream,
            json!({
                "candidates": [{ "content": { "parts": [{ "text": "hi" }] } }],
                "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 4 }
            }),
        );
        let events = stream.finish();
        let completed = events
            .iter()
            .find(|event| event["type"] == "response.completed")
            .unwrap();
        assert_eq!(completed["response"]["usage"]["input_tokens"], 10);
        assert_eq!(completed["response"]["usage"]["output_tokens"], 4);
    }

    #[test]
    fn an_empty_stream_still_completes_cleanly() {
        let mut stream = stream();
        let events = stream.finish();
        let kinds = kinds(&events);
        assert_eq!(kinds[0], "response.created");
        assert_eq!(kinds.last().unwrap(), "response.completed");
        assert!(!stream.has_output());
    }

    #[test]
    fn events_frame_with_their_name() {
        let event = json!({ "type": "response.completed", "sequence_number": 3 });
        let framed = format_event(&event);
        assert!(framed.starts_with("event: response.completed\n"));
        assert!(framed.contains("data: {"));
        assert!(framed.ends_with("\n\n"));
    }

    #[test]
    fn a_payload_without_a_type_still_frames() {
        let framed = format_event(&json!({ "sequence_number": 0 }));
        assert!(framed.starts_with("event: message\n"));
    }

    #[test]
    fn ids_have_the_protocols_prefixes() {
        assert!(new_response_id().starts_with("resp_"));
        assert!(item_id("msg").starts_with("msg_"));
        assert_ne!(new_response_id(), new_response_id());
    }
}
