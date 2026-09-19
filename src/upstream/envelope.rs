//! Envelope assembly.
//!
//! Every generation call is wrapped in a fixed outer object. The captured CLI
//! request pins both the key set and their order:
//!
//! ```text
//! envelopeKeys: project, requestId, request, model, userAgent, requestType
//! requestKeys:  contents, systemInstruction, tools, toolConfig, labels, generationConfig, sessionId
//! ```
//!
//! `serde` emits struct fields in declaration order, so the two structs below
//! reproduce that ordering by construction rather than by post-hoc reordering.
//! Reordering the declarations is a wire-format change.

use serde::{Deserialize, Serialize};

use crate::transform::ir::GenerateContentRequest;
use crate::upstream::metadata::{RequestMetadata, SessionStore};

/// Value `userAgent` always carries — the string `antigravity`, not a UA header.
const ENVELOPE_USER_AGENT: &str = "antigravity";

/// Value `requestType` always carries for agent traffic.
const ENVELOPE_REQUEST_TYPE: &str = "agent";

/// The outer request object sent to `/v1internal:generateContent` and
/// `/v1internal:streamGenerateContent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub project: String,
    #[serde(rename = "requestId")]
    pub request_id: String,
    pub request: GenerateContentRequest,
    pub model: String,
    #[serde(rename = "userAgent")]
    pub user_agent: String,
    #[serde(rename = "requestType")]
    pub request_type: String,
}

impl Envelope {
    /// Wrap a prepared request, filling in session metadata.
    ///
    /// `session_key` identifies the conversation; `step_count` is derived from
    /// the payload so that `labels.last_step_index` tracks real progress.
    pub fn build(
        request: GenerateContentRequest,
        project: impl Into<String>,
        model: impl Into<String>,
        session_key: &str,
        sessions: &SessionStore,
    ) -> Self {
        let model = model.into();
        let step_count = count_steps(&request);
        let metadata = sessions.begin_request(session_key, &model, step_count);
        Self::build_with_metadata(request, project, model, metadata)
    }

    /// Wrap a request using metadata that has already been produced.
    pub fn build_with_metadata(
        mut request: GenerateContentRequest,
        project: impl Into<String>,
        model: impl Into<String>,
        metadata: RequestMetadata,
    ) -> Self {
        request.labels = Some(metadata.labels);
        request.session_id = Some(metadata.session_id);
        Self {
            project: project.into(),
            request_id: metadata.request_id,
            request,
            model: model.into(),
            user_agent: ENVELOPE_USER_AGENT.into(),
            request_type: ENVELOPE_REQUEST_TYPE.into(),
        }
    }
}

/// Count the parts across all contents, which is how the CLI defines a step.
///
/// A payload with no contents still counts as one step so that
/// `last_step_index` is never zero.
pub fn count_steps(request: &GenerateContentRequest) -> usize {
    let parts: usize = request.contents.iter().map(|c| c.parts.len()).sum();
    parts.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::ir::{Content, Part};
    use serde_json::json;

    fn sample_request() -> GenerateContentRequest {
        GenerateContentRequest {
            contents: vec![Content::user(vec![Part::text("hello")])],
            system_instruction: Some(Content::user(vec![Part::text("be useful")])),
            ..Default::default()
        }
    }

    fn build() -> Envelope {
        Envelope::build(
            sample_request(),
            "my-project",
            "gemini-3.8-flash-medium",
            "session-1",
            &SessionStore::new(),
        )
    }

    #[test]
    fn envelope_key_order_matches_capture() {
        let json = serde_json::to_string(&build()).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let keys: Vec<&str> = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            vec![
                "project",
                "requestId",
                "request",
                "model",
                "userAgent",
                "requestType"
            ]
        );
    }

    #[test]
    fn request_key_order_matches_capture() {
        let value: serde_json::Value = serde_json::to_value(build()).unwrap();
        let keys: Vec<&str> = value["request"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            vec![
                "contents",
                "systemInstruction",
                "labels",
                "sessionId"
            ],
            "absent optional keys collapse, but the relative order of present keys must hold"
        );
    }

    #[test]
    fn request_key_order_with_tools_and_config() {
        let mut request = sample_request();
        request.tools = Some(vec![Default::default()]);
        request.tool_config = Some(Default::default());
        request.generation_config = Some(Default::default());

        let envelope = Envelope::build(
            request,
            "p",
            "gemini-3.8-flash-medium",
            "s",
            &SessionStore::new(),
        );
        let value: serde_json::Value = serde_json::to_value(envelope).unwrap();
        let keys: Vec<&str> = value["request"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            vec![
                "contents",
                "systemInstruction",
                "tools",
                "toolConfig",
                "labels",
                "generationConfig",
                "sessionId"
            ]
        );
    }

    #[test]
    fn envelope_carries_literal_agent_markers() {
        let value: serde_json::Value = serde_json::to_value(build()).unwrap();
        assert_eq!(value["userAgent"], "antigravity");
        assert_eq!(value["requestType"], "agent");
        assert_eq!(value["project"], "my-project");
        assert_eq!(value["model"], "gemini-3.8-flash-medium");
    }

    #[test]
    fn request_id_has_agent_prefix_and_five_segments() {
        let value: serde_json::Value = serde_json::to_value(build()).unwrap();
        let request_id = value["requestId"].as_str().unwrap();
        let segments: Vec<&str> = request_id.split('/').collect();
        assert_eq!(segments[0], "agent");
        assert_eq!(segments.len(), 5, "agent/<conversation>/<ts>/<trajectory>/<step>");
        assert!(segments[2].parse::<i64>().is_ok(), "timestamp must be numeric");
        assert!(segments[4].parse::<u64>().is_ok(), "step must be numeric");
    }

    #[test]
    fn session_id_is_written_into_the_request_object() {
        let envelope = build();
        assert!(envelope.request.session_id.is_some());
        assert!(envelope.request.labels.is_some());
    }

    #[test]
    fn step_count_never_zero() {
        let empty = GenerateContentRequest::default();
        assert_eq!(count_steps(&empty), 1);
    }

    #[test]
    fn step_count_sums_parts_across_contents() {
        let request = GenerateContentRequest {
            contents: vec![
                Content::user(vec![Part::text("a"), Part::text("b")]),
                Content::model(vec![Part::text("c")]),
            ],
            ..Default::default()
        };
        assert_eq!(count_steps(&request), 3);
    }

    #[test]
    fn metadata_labels_reach_the_wire() {
        let value: serde_json::Value = serde_json::to_value(build()).unwrap();
        let labels = value["request"]["labels"].as_object().unwrap();
        assert_eq!(labels["used_claude"], "false");
        assert_eq!(labels["model_enum"], "MODEL_PLACEHOLDER_M319");
        assert!(labels["trajectory_id"].is_string());
        // Sanity: the label set matches the reference's shape exactly.
        let expected: std::collections::BTreeSet<&str> = [
            "last_step_index",
            "model_enum",
            "trajectory_id",
            "used_claude",
            "used_claude_conservative",
            "used_non_gemini_model",
        ]
        .into_iter()
        .collect();
        let actual: std::collections::BTreeSet<&str> =
            labels.keys().map(String::as_str).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn round_trips_through_json() {
        let value = json!(build());
        let parsed: Envelope = serde_json::from_value(value).unwrap();
        assert_eq!(parsed.project, "my-project");
        assert_eq!(parsed.request.contents.len(), 1);
    }
}
