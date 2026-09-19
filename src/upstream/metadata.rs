//! Agent-shaped request metadata.
//!
//! Real `agy` traffic does not send a bare request. Every call carries a
//! `requestId` encoding a conversation, a trajectory, and a step index, plus a
//! `labels` block describing what the agent has been doing. The reference
//! implementation reproduces this from a per-session context, and the captured
//! request fixture records the field ordering.
//!
//! A gateway has no real workspace or conversation, so we synthesise an
//! equivalent: a session context keyed by a stable conversation identifier,
//! holding the conversation id, trajectory id, and the numeric `sessionId`. The
//! numeric id being *stable across turns* is the part that matters — it is what
//! lets the upstream keep prompt-cache and quota accounting coherent, and it is
//! exactly what the naive "random project per request" approach in
//! `opencode-antigravity-auth` gets wrong.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};
use uuid::Uuid;

/// Sessions idle for longer than this are dropped.
const SESSION_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Upper bound on tracked sessions, to keep a long-running gateway bounded.
const MAX_SESSIONS: usize = 512;

/// Wire model name to the opaque enum the CLI reports in `labels.model_enum`.
///
/// Only models observed in captured CLI traffic appear here. A model absent from
/// this table simply omits `model_enum`, which is what the CLI does for models it
/// does not have an enum for.
const MODEL_ENUM: &[(&str, &str)] = &[
    ("gemini-3.5-flash-extra-low", "MODEL_PLACEHOLDER_M187"),
    ("gemini-3.5-flash-low", "MODEL_PLACEHOLDER_M20"),
    ("gemini-3-flash-agent", "MODEL_PLACEHOLDER_M84"),
    ("gemini-3.6-flash-low", "MODEL_PLACEHOLDER_M73"),
    ("gemini-3.6-flash-medium", "MODEL_PLACEHOLDER_M72"),
    ("gemini-3.6-flash-high", "MODEL_PLACEHOLDER_M71"),
    ("gemini-3.7-flash-low", "MODEL_PLACEHOLDER_M300"),
    ("gemini-3.7-flash-medium", "MODEL_PLACEHOLDER_M299"),
    ("gemini-3.7-flash-high", "MODEL_PLACEHOLDER_M298"),
    ("gemini-3.8-flash-low", "MODEL_PLACEHOLDER_M320"),
    ("gemini-3.8-flash-medium", "MODEL_PLACEHOLDER_M319"),
    ("gemini-3.8-flash-high", "MODEL_PLACEHOLDER_M318"),
    ("gemini-3.1-pro-low", "MODEL_PLACEHOLDER_M36"),
    ("gemini-pro-agent", "MODEL_PLACEHOLDER_M16"),
    ("claude-sonnet-4-6", "MODEL_PLACEHOLDER_M35"),
    ("claude-opus-4-6-thinking", "MODEL_PLACEHOLDER_M26"),
    ("gemini-3.1-flash-image", "MODEL_PLACEHOLDER_M21"),
    ("gpt-oss-120b-medium", "MODEL_OPENAI_GPT_OSS_120B_MEDIUM"),
];

/// Look up the opaque enum for a wire model name, if one is known.
pub fn model_enum(model: &str) -> Option<&'static str> {
    let lowered = model.to_ascii_lowercase();
    MODEL_ENUM
        .iter()
        .find(|(name, _)| *name == lowered)
        .map(|(_, value)| *value)
}

/// FNV-1a over 64 bits, reinterpreted as a signed integer.
///
/// The CLI's numeric `sessionId` is a signed 64-bit value rendered as a decimal
/// string. Matching the exact derivation means our ids are indistinguishable in
/// shape from the CLI's.
pub fn fnv1a64_signed(input: &str) -> i64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET_BASIS;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash as i64
}

/// Per-conversation state that must survive across turns.
#[derive(Debug, Clone)]
pub struct SessionContext {
    pub conversation_id: String,
    pub trajectory_id: String,
    /// Stable numeric id, derived from the conversation key.
    pub numeric_session_id: String,
    /// Whether any Claude model has been used in this conversation.
    pub used_claude: bool,
    /// Whether any non-Gemini model (Claude or GPT-OSS) has been used.
    pub used_non_gemini_model: bool,
    /// Set after a completed execution; surfaces as `labels.last_execution_id`.
    pub last_execution_id: Option<String>,
    last_accessed: SystemTime,
    last_request_ms: i64,
}

impl SessionContext {
    fn new(key: &str, now: SystemTime) -> Self {
        Self {
            conversation_id: Uuid::new_v4().to_string(),
            trajectory_id: Uuid::new_v4().to_string(),
            numeric_session_id: fnv1a64_signed(key).to_string(),
            used_claude: false,
            used_non_gemini_model: false,
            last_execution_id: None,
            last_accessed: now,
            last_request_ms: 0,
        }
    }
}

/// Everything needed to fill in `requestId`, `sessionId`, and `labels`.
#[derive(Debug, Clone)]
pub struct RequestMetadata {
    pub request_id: String,
    pub session_id: String,
    pub labels: Map<String, Value>,
}

/// Bounded, TTL'd session store.
#[derive(Debug)]
pub struct SessionStore {
    sessions: Mutex<HashMap<String, SessionContext>>,
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionStore {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Build metadata for a request, creating or touching the session as needed.
    ///
    /// `step_count` is the number of parts in the outgoing payload, matching the
    /// CLI's own definition of a step.
    pub fn begin_request(&self, key: &str, model: &str, step_count: usize) -> RequestMetadata {
        let now = SystemTime::now();
        let mut sessions = self.sessions.lock().expect("session store poisoned");
        sessions.retain(|_, ctx| {
            now.duration_since(ctx.last_accessed)
                .map(|idle| idle < SESSION_TTL)
                .unwrap_or(true)
        });
        if sessions.len() >= MAX_SESSIONS && !sessions.contains_key(key) {
            // Evict the least recently touched session.
            if let Some(oldest) = sessions
                .iter()
                .min_by_key(|(_, ctx)| ctx.last_accessed)
                .map(|(k, _)| k.clone())
            {
                sessions.remove(&oldest);
            }
        }

        let ctx = sessions
            .entry(key.to_string())
            .or_insert_with(|| SessionContext::new(key, now));

        // Timestamps must strictly increase within a session, otherwise the
        // requestId of a rapid retry would collide with its predecessor.
        let wall = now
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let timestamp = wall.max(ctx.last_request_ms + 1);
        ctx.last_request_ms = timestamp;
        ctx.last_accessed = now;

        let lowered = model.to_ascii_lowercase();
        let is_claude = lowered.starts_with("claude-");
        let is_non_gemini = is_claude || lowered.starts_with("gpt-");
        ctx.used_claude |= is_claude;
        ctx.used_non_gemini_model |= is_non_gemini;

        // Mirrors the reference: step count plus one when a prior execution is
        // known, since the CLI counts the tool round trip as its own step.
        let last_step_index = step_count + usize::from(ctx.last_execution_id.is_some());

        let mut labels = Map::new();
        if let Some(execution_id) = &ctx.last_execution_id {
            labels.insert("last_execution_id".into(), Value::String(execution_id.clone()));
        }
        labels.insert(
            "last_step_index".into(),
            Value::String(last_step_index.to_string()),
        );
        if let Some(enum_value) = model_enum(model) {
            labels.insert("model_enum".into(), Value::String(enum_value.into()));
        }
        labels.insert(
            "trajectory_id".into(),
            Value::String(ctx.trajectory_id.clone()),
        );
        labels.insert(
            "used_claude".into(),
            Value::String(ctx.used_claude.to_string()),
        );
        labels.insert(
            "used_claude_conservative".into(),
            Value::String(ctx.used_claude.to_string()),
        );
        labels.insert(
            "used_non_gemini_model".into(),
            Value::String(ctx.used_non_gemini_model.to_string()),
        );

        RequestMetadata {
            request_id: format!(
                "agent/{}/{}/{}/{}",
                ctx.conversation_id,
                timestamp,
                ctx.trajectory_id,
                last_step_index + 1
            ),
            session_id: ctx.numeric_session_id.clone(),
            labels,
        }
    }

    /// Mark that a tool-execution round completed, so the next turn reports a
    /// `last_execution_id` and advances the step index.
    pub fn complete_execution(&self, key: &str) {
        let mut sessions = self.sessions.lock().expect("session store poisoned");
        if let Some(ctx) = sessions.get_mut(key) {
            ctx.last_execution_id = Some(Uuid::new_v4().to_string());
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.sessions.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a64_matches_reference_vector() {
        // Known FNV-1a 64-bit vectors, folded to signed.
        assert_eq!(fnv1a64_signed(""), 0xcbf2_9ce4_8422_2325u64 as i64);
        assert_eq!(fnv1a64_signed("a"), 0xaf63_dc4c_8601_ec8cu64 as i64);
        assert_eq!(fnv1a64_signed("foobar"), 0x85944171f73967e8u64 as i64);
    }

    #[test]
    fn session_id_is_stable_across_turns() {
        let store = SessionStore::new();
        let first = store.begin_request("conv-1", "gemini-3.8-flash", 2);
        let second = store.begin_request("conv-1", "gemini-3.8-flash", 4);
        assert_eq!(first.session_id, second.session_id);
        assert_eq!(first.labels["trajectory_id"], second.labels["trajectory_id"]);
    }

    #[test]
    fn distinct_conversations_get_distinct_sessions() {
        let store = SessionStore::new();
        let a = store.begin_request("conv-a", "gemini-3.8-flash", 1);
        let b = store.begin_request("conv-b", "gemini-3.8-flash", 1);
        assert_ne!(a.session_id, b.session_id);
        assert_ne!(a.labels["trajectory_id"], b.labels["trajectory_id"]);
    }

    #[test]
    fn request_ids_strictly_increase_within_a_session() {
        let store = SessionStore::new();
        let first = store.begin_request("conv-1", "gemini-3.8-flash", 1);
        let second = store.begin_request("conv-1", "gemini-3.8-flash", 1);
        // Both requests land in the same millisecond under test, so the
        // monotonic guard is what keeps them distinct.
        assert_ne!(first.request_id, second.request_id);

        let ts = |id: &str| -> i64 {
            id.split('/').nth(2).unwrap().parse().unwrap()
        };
        assert!(ts(&second.request_id) > ts(&first.request_id));
    }

    /// Read a label without panicking, so a missing key reports as a failed
    /// assertion rather than a test crash.
    fn label<'a>(meta: &'a RequestMetadata, key: &str) -> Option<&'a str> {
        meta.labels.get(key).and_then(Value::as_str)
    }

    #[test]
    fn labels_track_model_family_usage() {
        let store = SessionStore::new();

        // The enum table is keyed by *resolved wire* model names, so the base
        // name `gemini-3.8-flash` correctly has no enum — only its tiered
        // variants do. Tier resolution happens before this layer.
        let gemini = store.begin_request("c", "gemini-3.8-flash-medium", 1);
        assert_eq!(label(&gemini, "used_claude"), Some("false"));
        assert_eq!(label(&gemini, "used_non_gemini_model"), Some("false"));
        assert_eq!(label(&gemini, "model_enum"), Some("MODEL_PLACEHOLDER_M319"));

        // Usage flags are sticky for the life of the conversation.
        let claude = store.begin_request("c", "claude-opus-4-6-thinking", 1);
        assert_eq!(label(&claude, "used_claude"), Some("true"));
        assert_eq!(label(&claude, "used_non_gemini_model"), Some("true"));
        assert_eq!(label(&claude, "model_enum"), Some("MODEL_PLACEHOLDER_M26"));

        let gemini_again = store.begin_request("c", "gemini-3.8-flash-high", 1);
        assert_eq!(label(&gemini_again, "used_claude"), Some("true"));
        assert_eq!(label(&gemini_again, "model_enum"), Some("MODEL_PLACEHOLDER_M318"));
    }

    #[test]
    fn base_model_names_have_no_enum() {
        // Guards the assumption above: enums exist for wire names, not for the
        // unresolved public names clients send.
        assert!(model_enum("gemini-3.8-flash").is_none());
        assert!(model_enum("gemini-3.8-flash-medium").is_some());
        assert!(model_enum("claude-sonnet-4-6").is_some());
    }

    #[test]
    fn completed_execution_sets_label_and_advances_step() {
        let store = SessionStore::new();
        let before = store.begin_request("c", "gemini-3.8-flash", 2);
        assert!(before.labels.get("last_execution_id").is_none());

        store.complete_execution("c");
        let after = store.begin_request("c", "gemini-3.8-flash", 2);
        assert!(after.labels.get("last_execution_id").is_some());

        let step = |m: &RequestMetadata| -> i64 {
            m.labels["last_step_index"].as_str().unwrap().parse().unwrap()
        };
        assert_eq!(step(&after), step(&before) + 1);
    }

    #[test]
    fn session_count_is_bounded() {
        let store = SessionStore::new();
        for i in 0..MAX_SESSIONS + 50 {
            store.begin_request(&format!("conv-{i}"), "gemini-3.8-flash", 1);
        }
        assert!(store.len() <= MAX_SESSIONS);
    }

    #[test]
    fn unknown_model_omits_enum_label() {
        let store = SessionStore::new();
        let meta = store.begin_request("c", "some-unknown-model", 1);
        assert!(meta.labels.get("model_enum").is_none());
    }
}
