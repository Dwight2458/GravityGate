//! OpenAI Chat Completions wire types.
//!
//! Deserialisation here is deliberately permissive. Clients in this ecosystem
//! send supersets of the spec — extra fields, `null` where a field is optional,
//! content parts this gateway does not model — and rejecting a request because
//! of an unrecognised key would make the gateway useless in practice. Unknown
//! fields are ignored; unknown content part types are dropped with a warning.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    /// Legacy spelling, still what most clients send.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Current spelling. Takes precedence when both are present.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,

    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stop: Option<StopSequences>,

    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    #[serde(default)]
    pub response_format: Option<Value>,
    /// OpenAI's coarse reasoning control.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// End-user identifier, used here as a session hint for prompt-cache affinity.
    #[serde(default)]
    pub user: Option<String>,
    /// Non-standard extension used by several gateways; accepted so clients that
    /// send it get the behaviour they expect.
    #[serde(default)]
    pub thinking: Option<ThinkingParam>,

    /// Anthropic-flavoured alias occasionally sent by polyglot clients.
    #[serde(default)]
    pub anthropic_beta: Option<Value>,
}

/// Non-standard `thinking` extension.
#[derive(Debug, Clone, Deserialize)]
pub struct ThinkingParam {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub budget_tokens: Option<i64>,
}

impl ThinkingParam {
    /// Whether thinking was explicitly turned off.
    pub fn is_disabled(&self) -> bool {
        matches!(self.kind.as_deref(), Some("disabled" | "off" | "none"))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

/// `stop` accepts a string or an array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(one) => vec![one],
            Self::Many(many) => many,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: Option<MessageContent>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    /// Some clients echo reasoning back on the next turn. It carries no
    /// signature, so it is informational only.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

/// Message content: a bare string or a list of parts.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Parts(parts) => parts.is_empty(),
        }
    }
}

/// A content part.
///
/// Modelled flat with a `type` discriminator rather than as a tagged enum, so an
/// unrecognised part type is a value we can skip instead of a deserialisation
/// failure that rejects the whole request.
#[derive(Debug, Clone, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<ImageUrl>,
}

impl ContentPart {
    pub fn text_part(text: impl Into<String>) -> Self {
        Self {
            kind: "text".into(),
            text: Some(text.into()),
            image_url: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionCall>,
    /// Present in some clients' echoes of a Gemini tool call. Treated as a
    /// signature of last resort — the cache is the real mechanism.
    #[serde(default, rename = "thoughtSignature")]
    pub thought_signature: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionCall {
    #[serde(default)]
    pub name: Option<String>,
    /// Arguments as a JSON string, per the OpenAI spec. A few clients send an
    /// object instead, which is tolerated.
    #[serde(default)]
    pub arguments: Option<Value>,
}

impl FunctionCall {
    /// Parse arguments into an object, tolerating both spellings.
    pub fn parsed_arguments(&self) -> serde_json::Map<String, Value> {
        match &self.arguments {
            Some(Value::String(text)) => {
                serde_json::from_str(text).unwrap_or_else(|_| serde_json::Map::new())
            }
            Some(Value::Object(map)) => map.clone(),
            _ => serde_json::Map::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Tool {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: Option<ToolFunction>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
    /// Some clients send a strict flag; the upstream has no equivalent.
    #[serde(default)]
    pub strict: Option<bool>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Present when the upstream reported one; useful for support requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ResponseMessage {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ResponseFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFunction {
    pub name: String,
    /// JSON-encoded arguments, per the OpenAI spec.
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct PromptTokensDetails {
    pub cached_tokens: i64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// A tool call with the index OpenAI's streaming format requires.
///
/// `ResponseToolCall` is the wire shape for a single call; the streaming delta
/// adds an `index` field alongside `id`, `type`, and `function`.
#[derive(Debug, Clone)]
pub struct IndexedToolCall {
    pub index: u32,
    pub call: ResponseToolCall,
}

impl serde::Serialize for IndexedToolCall {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("index", &self.index)?;
        map.serialize_entry("id", &self.call.id)?;
        map.serialize_entry("type", &self.call.kind)?;
        map.serialize_entry("function", &self.call.function)?;
        map.end()
    }
}

impl<'de> serde::Deserialize<'de> for IndexedToolCall {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Raw {
            #[serde(default)]
            index: u32,
            #[serde(default)]
            id: String,
            // `type` needs no field: only `function` calls exist on this
            // surface, and serde skips unknown fields.
            function: ResponseFunction,
        }
        let raw = Raw::deserialize(deserializer)?;
        Ok(Self {
            index: raw.index,
            call: ResponseToolCall {
                id: raw.id,
                kind: "function",
                function: raw.function,
            },
        })
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<IndexedToolCall>>,
}

impl ChunkDelta {
    pub fn is_empty(&self) -> bool {
        self.role.is_none()
            && self.content.is_none()
            && self.reasoning_content.is_none()
            && self.reasoning.is_none()
            && self.tool_calls.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(value: Value) -> ChatCompletionRequest {
        serde_json::from_value(value).expect("request should deserialise")
    }

    #[test]
    fn minimal_request_deserialises() {
        let request = parse(json!({
            "model": "gemini-3.8-flash",
            "messages": [{ "role": "user", "content": "hi" }]
        }));
        assert_eq!(request.model, "gemini-3.8-flash");
        assert_eq!(request.messages.len(), 1);
        assert!(!request.stream);
    }

    #[test]
    fn unknown_top_level_fields_are_ignored() {
        // Clients send supersets; rejecting them would make the gateway unusable.
        let request = parse(json!({
            "model": "m",
            "messages": [],
            "some_future_field": {"nested": true},
            "another": 42
        }));
        assert_eq!(request.model, "m");
    }

    #[test]
    fn content_accepts_a_bare_string_or_parts() {
        let text = parse(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": "hello" }]
        }));
        assert!(matches!(
            text.messages[0].content,
            Some(MessageContent::Text(_))
        ));

        let parts = parse(json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "look" },
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAA" } }
                ]
            }]
        }));
        assert!(matches!(
            parts.messages[0].content,
            Some(MessageContent::Parts(_))
        ));
    }

    #[test]
    fn unrecognised_content_part_type_does_not_break_parsing() {
        let request = parse(json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "ok" },
                    { "type": "input_audio", "input_audio": { "data": "..." } }
                ]
            }]
        }));
        assert!(matches!(
            request.messages[0].content,
            Some(MessageContent::Parts(_))
        ));
    }

    #[test]
    fn stop_accepts_either_shape() {
        let single = parse(json!({
            "model": "m", "messages": [], "stop": "\n"
        }));
        assert_eq!(single.stop.unwrap().into_vec(), vec!["\n"]);

        let many = parse(json!({
            "model": "m", "messages": [], "stop": ["a", "b"]
        }));
        assert_eq!(many.stop.unwrap().into_vec(), vec!["a", "b"]);
    }

    #[test]
    fn tool_call_arguments_parse_from_a_json_string() {
        let request = parse(json!({
            "model": "m",
            "messages": [{
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
                }]
            }]
        }));
        let call = &request.messages[0].tool_calls.as_ref().unwrap()[0];
        let arguments = call.function.as_ref().unwrap().parsed_arguments();
        assert_eq!(arguments["city"], "Paris");
    }

    #[test]
    fn tool_call_arguments_also_parse_from_an_object() {
        // Some clients send an object where the spec says string.
        let request = parse(json!({
            "model": "m",
            "messages": [{
                "role": "assistant",
                "tool_calls": [{
                    "function": { "name": "f", "arguments": { "city": "Paris" } }
                }]
            }]
        }));
        let call = &request.messages[0].tool_calls.as_ref().unwrap()[0];
        assert_eq!(
            call.function.as_ref().unwrap().parsed_arguments()["city"],
            "Paris"
        );
    }

    #[test]
    fn malformed_tool_arguments_yield_an_empty_object() {
        let request = parse(json!({
            "model": "m",
            "messages": [{
                "role": "assistant",
                "tool_calls": [{ "function": { "name": "f", "arguments": "{not json" } }]
            }]
        }));
        let call = &request.messages[0].tool_calls.as_ref().unwrap()[0];
        assert!(call.function.as_ref().unwrap().parsed_arguments().is_empty());
    }

    #[test]
    fn null_content_is_accepted() {
        // An assistant turn that only makes tool calls has null content.
        let request = parse(json!({
            "model": "m",
            "messages": [{ "role": "assistant", "content": null, "tool_calls": [] }]
        }));
        assert!(request.messages[0].content.is_none());
    }

    #[test]
    fn thinking_extension_is_recognised() {
        let enabled = parse(json!({
            "model": "m", "messages": [],
            "thinking": { "type": "enabled", "budget_tokens": 8192 }
        }));
        let thinking = enabled.thinking.unwrap();
        assert!(!thinking.is_disabled());
        assert_eq!(thinking.budget_tokens, Some(8192));

        let disabled = parse(json!({
            "model": "m", "messages": [],
            "thinking": { "type": "disabled" }
        }));
        assert!(disabled.thinking.unwrap().is_disabled());
    }

    #[test]
    fn response_omits_absent_fields() {
        let message = ResponseMessage {
            role: "assistant",
            content: Some("hi".into()),
            ..Default::default()
        };
        let json = serde_json::to_value(&message).unwrap();
        assert!(json.get("reasoning_content").is_none());
        assert!(json.get("tool_calls").is_none());
        assert_eq!(json["role"], "assistant");
    }

    #[test]
    fn chunk_delta_emptiness_is_detected() {
        assert!(ChunkDelta::default().is_empty());
        assert!(!ChunkDelta {
            content: Some("x".into()),
            ..Default::default()
        }
        .is_empty());
    }

    #[test]
    fn usage_serialises_token_details_only_when_present() {
        let bare = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        };
        let json = serde_json::to_value(&bare).unwrap();
        assert!(json.get("prompt_tokens_details").is_none());

        let detailed = Usage {
            prompt_tokens_details: Some(PromptTokensDetails { cached_tokens: 4 }),
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: 2,
            }),
            ..bare
        };
        let json = serde_json::to_value(&detailed).unwrap();
        assert_eq!(json["prompt_tokens_details"]["cached_tokens"], 4);
        assert_eq!(json["completion_tokens_details"]["reasoning_tokens"], 2);
    }
}
