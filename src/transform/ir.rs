//! Canonical intermediate representation.
//!
//! GravityGate uses the Google Generative AI shape (`contents` / `parts`) as its
//! normalised IR rather than Anthropic's Messages shape. The reason is that the
//! upstream is the only constraint surface that matters: an OpenAI request
//! crosses exactly one translation boundary on the way out, and an upstream
//! response crosses exactly one on the way back. Thinking has a native
//! representation here (`Part::thought` / `Part::thought_signature`), so
//! signatures ride along with the parts that carry them instead of needing a
//! side channel.
//!
//! Field declaration order is load-bearing. The upstream is calibrated against a
//! captured CLI request, and the reference implementation explicitly re-orders
//! the envelope and inner request to match. `serde` serialises struct fields in
//! declaration order, so the order below is the order on the wire — do not
//! alphabetise or reorder without re-checking against the capture.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Role-bearing turn. The upstream knows exactly two roles: `user` and `model`.
///
/// `role` is optional because a *response* candidate omits it — the upstream
/// sends `{"content": {"parts": [...]}}` with no role at all. Requiring it here
/// would make every streamed response fail to parse. Requests always set it, so
/// the field is still emitted on the way out.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Content {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub role: String,
    pub parts: Vec<Part>,
}

impl Content {
    pub fn user(parts: Vec<Part>) -> Self {
        Self {
            role: "user".into(),
            parts,
        }
    }

    pub fn model(parts: Vec<Part>) -> Self {
        Self {
            role: "model".into(),
            parts,
        }
    }
}

/// A single piece of content.
///
/// Modelled as a flat struct with optional fields rather than an untagged enum.
/// An untagged enum has to guess a variant by trying each in turn, which turns
/// any schema drift into a silently-wrong variant; a flat struct either matches
/// or produces a visible `None`.
///
/// Exactly one of `text` / `function_call` / `function_response` / `inline_data`
/// is expected to be set. The combinations that actually occur:
///
/// - plain text: `text`
/// - reasoning: `text` + `thought: true`, optionally `thought_signature`
/// - tool call: `function_call`, optionally `thought_signature` (Gemini 3 places
///   the signature here rather than on the thinking part)
/// - tool result: `function_response`
/// - image: `inline_data`
///
/// `thought_signature` is declared **last** on purpose. The reference
/// implementations build a call part as `{ functionCall }` and only then attach
/// `thoughtSignature`, so `functionCall` must precede it; equally, a reasoning
/// part reads `{ text, thought, thoughtSignature }`. Parking the signature at the
/// end satisfies both shapes from a single declaration order.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Present and `true` only for reasoning content. Left as `None` rather than
    /// `Some(false)` for ordinary text so the key is omitted entirely.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
    #[serde(rename = "functionCall", skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(rename = "functionResponse", skip_serializing_if = "Option::is_none")]
    pub function_response: Option<FunctionResponse>,
    #[serde(rename = "inlineData", skip_serializing_if = "Option::is_none")]
    pub inline_data: Option<InlineData>,
    #[serde(
        rename = "thoughtSignature",
        skip_serializing_if = "Option::is_none"
    )]
    pub thought_signature: Option<String>,
}

impl Part {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            ..Default::default()
        }
    }

    pub fn thought_text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            thought: Some(true),
            ..Default::default()
        }
    }

    pub fn function_call(call: FunctionCall) -> Self {
        Self {
            function_call: Some(call),
            ..Default::default()
        }
    }

    pub fn function_response(response: FunctionResponse) -> Self {
        Self {
            function_response: Some(response),
            ..Default::default()
        }
    }

    pub fn inline_data(data: InlineData) -> Self {
        Self {
            inline_data: Some(data),
            ..Default::default()
        }
    }

    /// Reasoning content, the `thought` flag being what distinguishes it.
    pub fn is_thought(&self) -> bool {
        self.thought == Some(true)
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_none()
            && self.function_call.is_none()
            && self.function_response.is_none()
            && self.inline_data.is_none()
    }

    /// Signature carried by this part, whether on a thinking part or a Gemini 3
    /// function call.
    pub fn signature(&self) -> Option<&str> {
        self.thought_signature.as_deref()
    }

    pub fn set_signature(&mut self, signature: impl Into<String>) {
        self.thought_signature = Some(signature.into());
    }

    pub fn clear_signature(&mut self) {
        self.thought_signature = None;
    }
}

/// A tool call.
///
/// `id` is a sibling of `name` and `args`, not an entry inside `args`. Live
/// traffic disproved the assumption that it is usually absent: the upstream sent
/// `{"name": "get_weather", "args": {...}, "id": "call_9680"}`. The id is
/// preserved because the backend may correlate it with the signature it issued
/// alongside, and replacing it with one of our own would break that link for no
/// benefit. Claude targets expect an id on the way back; Gemini targets do not.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub args: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FunctionResponse {
    pub name: String,
    pub response: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct InlineData {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub data: String,
}

/// Tool declarations. The upstream takes a single-element array holding one
/// `functionDeclarations` list, not one entry per tool.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Tool {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ToolConfig {
    #[serde(rename = "functionCallingConfig")]
    pub function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FunctionCallingConfig {
    /// `AUTO` | `ANY` | `NONE` | `VALIDATED`
    pub mode: String,
    #[serde(
        rename = "allowedFunctionNames",
        skip_serializing_if = "Option::is_none"
    )]
    pub allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct GenerationConfig {
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(rename = "topK", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<Value>,
    #[serde(rename = "responseMimeType", skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    #[serde(rename = "responseSchema", skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Value>,
}

/// The inner `request` object of the envelope.
///
/// Declaration order here reproduces the captured `requestKeys`:
/// `contents, systemInstruction, tools, toolConfig, labels, generationConfig, sessionId`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct GenerateContentRequest {
    pub contents: Vec<Content>,
    #[serde(
        rename = "systemInstruction",
        skip_serializing_if = "Option::is_none"
    )]
    pub system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<Map<String, Value>>,
    #[serde(
        rename = "generationConfig",
        skip_serializing_if = "Option::is_none"
    )]
    pub generation_config: Option<GenerationConfig>,
    #[serde(rename = "sessionId", skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// A candidate returned by the upstream.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Candidate {
    #[serde(default)]
    pub content: Option<Content>,
    #[serde(rename = "finishReason", default)]
    pub finish_reason: Option<String>,
}

/// Token accounting.
///
/// Two properties were learned from live responses and both matter:
///
/// - The upstream counts cached tokens *inside* `promptTokenCount`, so a
///   client-facing prompt count is `prompt - cached`. See [`Self::prompt_tokens`].
/// - `candidatesTokenCount` is not always present. One live usage block carried
///   only `promptTokenCount`, `totalTokenCount`, and `thoughtsTokenCount`. See
///   [`Self::candidates_tokens`].
#[derive(Debug, Clone, Deserialize, Default)]
pub struct UsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    pub prompt_token_count: i64,
    #[serde(rename = "candidatesTokenCount", default)]
    pub candidates_token_count: Option<i64>,
    #[serde(rename = "cachedContentTokenCount", default)]
    pub cached_content_token_count: i64,
    #[serde(rename = "thoughtsTokenCount", default)]
    pub thoughts_token_count: i64,
    #[serde(rename = "totalTokenCount", default)]
    pub total_token_count: Option<i64>,
}

impl UsageMetadata {
    /// Prompt tokens excluding cached ones — what an OpenAI client expects in
    /// `usage.prompt_tokens`.
    pub fn prompt_tokens(&self) -> i64 {
        (self.prompt_token_count - self.cached_content_token_count).max(0)
    }

    /// Completion tokens, derived when the upstream omits the count.
    ///
    /// `totalTokenCount` covers prompt, candidates, *and* thinking, so it cannot
    /// be used directly as the completion count. Subtracting the parts we do
    /// know isolates the candidate count, which is what an OpenAI client expects:
    /// thinking is reported separately as `reasoning_tokens`.
    pub fn candidates_tokens(&self) -> i64 {
        if let Some(explicit) = self.candidates_token_count {
            return explicit.max(0);
        }
        let Some(total) = self.total_token_count else {
            return 0;
        };
        (total - self.prompt_token_count - self.thoughts_token_count).max(0)
    }

    /// Whether any token accounting at all is present.
    pub fn is_present(&self) -> bool {
        self.prompt_token_count > 0
            || self.candidates_token_count.is_some()
            || self.total_token_count.is_some()
            || self.thoughts_token_count > 0
    }
}

/// The payload the upstream wraps in `{"response": ...}` on both the streaming
/// and buffering paths.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct GenerateContentResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(rename = "usageMetadata", default)]
    pub usage_metadata: Option<UsageMetadata>,
    #[serde(rename = "modelVersion", default)]
    pub model_version: Option<String>,
    #[serde(rename = "responseId", default)]
    pub response_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_part_omits_thought_key() {
        let part = Part::text("hello");
        let json = serde_json::to_string(&part).unwrap();
        assert_eq!(json, r#"{"text":"hello"}"#);
    }

    #[test]
    fn thought_part_serialises_text_then_thought_then_signature() {
        let mut part = Part::thought_text("reasoning");
        part.set_signature("sig");
        let json = serde_json::to_string(&part).unwrap();
        // Order is asserted literally: the upstream is calibrated against a capture.
        assert_eq!(json, r#"{"text":"reasoning","thought":true,"thoughtSignature":"sig"}"#);
    }

    #[test]
    fn function_call_part_puts_signature_last() {
        let mut part = Part::function_call(FunctionCall {
            name: "get_weather".into(),
            args: Map::new(),
            id: None,
        });
        part.set_signature("sig");
        let json = serde_json::to_string(&part).unwrap();
        // Matches how the reference builds it: `{ functionCall }`, then attach
        // `thoughtSignature`.
        assert_eq!(
            json,
            r#"{"functionCall":{"name":"get_weather"},"thoughtSignature":"sig"}"#
        );
    }

    #[test]
    fn request_field_order_matches_capture() {
        let req = GenerateContentRequest {
            contents: vec![Content::user(vec![Part::text("hi")])],
            system_instruction: None,
            tools: None,
            tool_config: None,
            labels: None,
            generation_config: Some(GenerationConfig {
                max_output_tokens: Some(1024),
                ..Default::default()
            }),
            session_id: Some("-123".into()),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(
            json,
            r#"{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"maxOutputTokens":1024},"sessionId":"-123"}"#
        );
    }

    #[test]
    fn usage_excludes_cached_tokens_from_prompt_count() {
        let usage = UsageMetadata {
            prompt_token_count: 1000,
            cached_content_token_count: 400,
            ..Default::default()
        };
        assert_eq!(usage.prompt_tokens(), 600);
    }

    #[test]
    fn usage_never_reports_negative_prompt_tokens() {
        let usage = UsageMetadata {
            prompt_token_count: 10,
            cached_content_token_count: 40,
            ..Default::default()
        };
        assert_eq!(usage.prompt_tokens(), 0);
    }

    #[test]
    fn explicit_candidate_count_is_used_directly() {
        let usage = UsageMetadata {
            prompt_token_count: 100,
            candidates_token_count: Some(20),
            total_token_count: Some(130),
            thoughts_token_count: 10,
            ..Default::default()
        };
        assert_eq!(usage.candidates_tokens(), 20);
    }

    #[test]
    fn candidate_count_is_derived_when_absent() {
        // The live shape: no candidates count, but a total that includes
        // thinking. Completion tokens exclude thinking, which is reported
        // separately.
        let usage = UsageMetadata {
            prompt_token_count: 9,
            candidates_token_count: None,
            total_token_count: Some(30),
            thoughts_token_count: 16,
            ..Default::default()
        };
        assert_eq!(usage.candidates_tokens(), 5);
    }

    #[test]
    fn derived_candidate_count_never_goes_negative() {
        // Possible if the upstream's total lags its parts mid-stream.
        let usage = UsageMetadata {
            prompt_token_count: 100,
            candidates_token_count: None,
            total_token_count: Some(10),
            thoughts_token_count: 50,
            ..Default::default()
        };
        assert_eq!(usage.candidates_tokens(), 0);
    }

    #[test]
    fn missing_totals_yield_zero_rather_than_a_panic() {
        let usage = UsageMetadata::default();
        assert_eq!(usage.candidates_tokens(), 0);
        assert!(!usage.is_present());
    }

    #[test]
    fn presence_is_detected_from_any_field() {
        assert!(UsageMetadata {
            prompt_token_count: 1,
            ..Default::default()
        }
        .is_present());
        assert!(UsageMetadata {
            thoughts_token_count: 3,
            ..Default::default()
        }
        .is_present());
        assert!(UsageMetadata {
            total_token_count: Some(5),
            ..Default::default()
        }
        .is_present());
        assert!(UsageMetadata {
            candidates_token_count: Some(0),
            ..Default::default()
        }
        .is_present());
    }
}
