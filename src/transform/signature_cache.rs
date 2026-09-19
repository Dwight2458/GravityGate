//! Thinking signature cache.
//!
//! This is the subsystem with no prior art to copy, and the reason is worth
//! stating plainly.
//!
//! The upstream requires a conversation's prior thinking to be sent back with
//! its signature attached. The reference implementations speak the Anthropic
//! protocol, which has somewhere to put one: `signature_delta` events on the way
//! out, `thinking.signature` on the way back. OpenAI's protocol has no such
//! field. `tool_calls` carries an id, a name, and a JSON string — nothing else —
//! and clients drop unknown fields when they echo a turn back. There is no
//! arrangement of the OpenAI surface that carries a signature from one turn to
//! the next.
//!
//! So the gateway carries it. A signature is captured when it is produced and
//! reattached when the turn it belongs to comes back. The client never sees it
//! and never needs to preserve it.
//!
//! Three properties matter:
//!
//! - **Signatures attach to more than thinking.** A live response produced
//!   `{"thoughtSignature": "EusCCug...", "text": ""}` — a signature on a part
//!   with no thinking text and no `thought` flag. And for Gemini 3, the signature
//!   that matters most sits on a `functionCall` part. Anything keyed off "this
//!   part is thinking" would miss both.
//! - **Signatures are family-scoped.** A signature minted by a Gemini model is
//!   rejected on a Claude request and vice versa. Reuse across families has to be
//!   refused here, because the upstream's error is opaque.
//! - **A shorter signature is not a signature.** Real ones run to hundreds of
//!   bytes. Anything under [`MIN_SIGNATURE_LENGTH`] is treated as absent.
//!
//! Entries expire, because a signature is only meaningful for a conversation
//! that is still alive, and holding them forever would grow without bound.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::registry::models::ModelFamily;
use crate::upstream::constants::MIN_SIGNATURE_LENGTH;

/// How long a signature stays usable. Matches the reference implementations:
/// long enough for a slow multi-turn exchange, short enough to bound the cache.
const TTL: Duration = Duration::from_secs(2 * 60 * 60);

/// Upper bound on entries per map. A long-running gateway sees many
/// conversations; this keeps memory flat without evicting so aggressively that
/// ordinary tool loops lose their signatures.
const MAX_ENTRIES: usize = 4096;

/// What a lookup produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignatureLookup {
    /// A signature usable for the requesting model's family.
    Usable(String),
    /// A signature is cached, but was minted by a different family. Sending it
    /// would be rejected upstream, so the caller must treat this as absent —
    /// but it is worth distinguishing from a cold cache when diagnosing.
    ForeignFamily,
    /// Nothing cached, or what was cached has expired.
    Missing,
}

impl SignatureLookup {
    pub fn into_option(self) -> Option<String> {
        match self {
            Self::Usable(signature) => Some(signature),
            _ => None,
        }
    }

    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Usable(_))
    }
}

#[derive(Debug, Clone)]
struct Entry {
    signature: String,
    family: ModelFamily,
    inserted: Instant,
}

#[derive(Debug)]
struct Inner {
    /// Keyed by the tool call id *we* generated for the client. The upstream
    /// often omits an id on `functionCall`, so the id the client echoes back is
    /// one we minted — which makes it a sound cache key.
    by_tool_call: HashMap<String, Entry>,
    /// Keyed by session, holding the most recent thinking signature. Used to
    /// pair a client's echoed `reasoning_content` with the signature that
    /// belongs to it.
    by_session: HashMap<String, Entry>,
}

impl Inner {
    fn new() -> Self {
        Self {
            by_tool_call: HashMap::new(),
            by_session: HashMap::new(),
        }
    }

    /// Drop expired entries, then evict the oldest if still over the bound.
    fn prune(&mut self, now: Instant) {
        for map in [&mut self.by_tool_call, &mut self.by_session] {
            map.retain(|_, entry| now.duration_since(entry.inserted) < TTL);
        }
        for map in [&mut self.by_tool_call, &mut self.by_session] {
            while map.len() > MAX_ENTRIES {
                let Some(oldest) = map
                    .iter()
                    .min_by_key(|(_, entry)| entry.inserted)
                    .map(|(key, _)| key.clone())
                else {
                    break;
                };
                map.remove(&oldest);
            }
        }
    }
}

/// Cache statistics, for `/health` and diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub tool_signatures: usize,
    pub session_signatures: usize,
}

#[derive(Debug)]
pub struct SignatureCache {
    inner: Mutex<Inner>,
}

impl Default for SignatureCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SignatureCache {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner::new()),
        }
    }

    /// Record the signature attached to a tool call.
    ///
    /// `tool_call_id` must be the id that was handed to the client, since that
    /// is what comes back on the next turn.
    pub fn put_tool(&self, tool_call_id: &str, signature: &str, family: ModelFamily) -> bool {
        self.put(|inner| &mut inner.by_tool_call, tool_call_id, signature, family)
    }

    /// Record the most recent thinking signature for a conversation.
    pub fn put_thinking(&self, session_key: &str, signature: &str, family: ModelFamily) -> bool {
        self.put(|inner| &mut inner.by_session, session_key, signature, family)
    }

    fn put(
        &self,
        select: impl FnOnce(&mut Inner) -> &mut HashMap<String, Entry>,
        key: &str,
        signature: &str,
        family: ModelFamily,
    ) -> bool {
        if !is_plausible(signature) {
            return false;
        }
        let now = Instant::now();
        let mut inner = self.inner.lock().expect("signature cache poisoned");
        let map = select(&mut inner);
        map.insert(
            key.to_string(),
            Entry {
                signature: signature.to_string(),
                family,
                inserted: now,
            },
        );
        // Prune after inserting, so the bound accounts for the entry just added.
        // Pruning first would let the map settle one entry above its limit. The
        // new entry is the most recent, so eviction cannot remove it.
        inner.prune(now);
        true
    }

    /// Look up the signature for a tool call, validated against the target family.
    pub fn tool_signature(&self, tool_call_id: &str, target: ModelFamily) -> SignatureLookup {
        self.lookup(|inner| &inner.by_tool_call, tool_call_id, target)
    }

    /// Look up the thinking signature for a session.
    pub fn thinking_signature(&self, session_key: &str, target: ModelFamily) -> SignatureLookup {
        self.lookup(|inner| &inner.by_session, session_key, target)
    }

    fn lookup(
        &self,
        select: impl FnOnce(&Inner) -> &HashMap<String, Entry>,
        key: &str,
        target: ModelFamily,
    ) -> SignatureLookup {
        let now = Instant::now();
        let mut inner = self.inner.lock().expect("signature cache poisoned");
        inner.prune(now);

        let Some(entry) = select(&inner).get(key) else {
            return SignatureLookup::Missing;
        };
        if entry.family != target {
            return SignatureLookup::ForeignFamily;
        }
        SignatureLookup::Usable(entry.signature.clone())
    }

    /// Forget everything. Used when credentials change and cached signatures
    /// can no longer be trusted to belong to the current accounts.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("signature cache poisoned");
        inner.by_tool_call.clear();
        inner.by_session.clear();
    }

    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock().expect("signature cache poisoned");
        CacheStats {
            tool_signatures: inner.by_tool_call.len(),
            session_signatures: inner.by_session.len(),
        }
    }
}

/// Whether a string is plausibly a signature rather than a placeholder.
pub fn is_plausible(signature: &str) -> bool {
    signature.len() >= MIN_SIGNATURE_LENGTH
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A signature-shaped string of the length the upstream actually produces.
    fn sig(marker: &str) -> String {
        format!("{marker}{}", "A".repeat(60))
    }

    #[test]
    fn a_stored_tool_signature_comes_back_for_the_same_family() {
        let cache = SignatureCache::new();
        assert!(cache.put_tool("call_1", &sig("gem"), ModelFamily::Gemini));
        assert_eq!(
            cache.tool_signature("call_1", ModelFamily::Gemini),
            SignatureLookup::Usable(sig("gem"))
        );
    }

    #[test]
    fn an_unknown_tool_call_reports_missing() {
        let cache = SignatureCache::new();
        assert_eq!(
            cache.tool_signature("never_seen", ModelFamily::Gemini),
            SignatureLookup::Missing
        );
    }

    #[test]
    fn a_foreign_family_signature_is_refused() {
        // Sending a Gemini signature on a Claude request is rejected upstream
        // with an opaque error, so it must be refused here.
        let cache = SignatureCache::new();
        cache.put_tool("call_1", &sig("gem"), ModelFamily::Gemini);
        assert_eq!(
            cache.tool_signature("call_1", ModelFamily::Claude),
            SignatureLookup::ForeignFamily
        );
        assert!(cache
            .tool_signature("call_1", ModelFamily::Claude)
            .into_option()
            .is_none());
    }

    #[test]
    fn session_thinking_signatures_are_kept_separately_from_tool_ones() {
        let cache = SignatureCache::new();
        cache.put_tool("call_1", &sig("tool"), ModelFamily::Gemini);
        cache.put_thinking("session_1", &sig("think"), ModelFamily::Gemini);

        assert_eq!(
            cache.tool_signature("call_1", ModelFamily::Gemini),
            SignatureLookup::Usable(sig("tool"))
        );
        assert_eq!(
            cache.thinking_signature("session_1", ModelFamily::Gemini),
            SignatureLookup::Usable(sig("think"))
        );
        // The maps must not bleed into each other.
        assert_eq!(
            cache.thinking_signature("call_1", ModelFamily::Gemini),
            SignatureLookup::Missing
        );
    }

    #[test]
    fn the_latest_thinking_signature_replaces_the_previous_one() {
        let cache = SignatureCache::new();
        cache.put_thinking("s", &sig("first"), ModelFamily::Gemini);
        cache.put_thinking("s", &sig("second"), ModelFamily::Gemini);
        assert_eq!(
            cache.thinking_signature("s", ModelFamily::Gemini),
            SignatureLookup::Usable(sig("second"))
        );
    }

    #[test]
    fn a_short_string_is_not_accepted_as_a_signature() {
        // Guards against caching a placeholder or a truncated read.
        let cache = SignatureCache::new();
        assert!(!cache.put_tool("call_1", "too-short", ModelFamily::Gemini));
        assert_eq!(
            cache.tool_signature("call_1", ModelFamily::Gemini),
            SignatureLookup::Missing
        );
    }

    #[test]
    fn the_length_gate_is_inclusive_at_the_threshold() {
        let exact = "A".repeat(MIN_SIGNATURE_LENGTH);
        assert!(is_plausible(&exact));
        assert!(!is_plausible(&"A".repeat(MIN_SIGNATURE_LENGTH - 1)));
    }

    #[test]
    fn clearing_empties_both_maps() {
        let cache = SignatureCache::new();
        cache.put_tool("call_1", &sig("a"), ModelFamily::Gemini);
        cache.put_thinking("s", &sig("b"), ModelFamily::Gemini);
        cache.clear();

        let stats = cache.stats();
        assert_eq!(stats.tool_signatures, 0);
        assert_eq!(stats.session_signatures, 0);
    }

    #[test]
    fn stats_report_what_is_held() {
        let cache = SignatureCache::new();
        cache.put_tool("call_1", &sig("a"), ModelFamily::Gemini);
        cache.put_tool("call_2", &sig("b"), ModelFamily::Gemini);
        cache.put_thinking("s", &sig("c"), ModelFamily::Gemini);

        let stats = cache.stats();
        assert_eq!(stats.tool_signatures, 2);
        assert_eq!(stats.session_signatures, 1);
    }

    #[test]
    fn expired_entries_are_dropped_on_the_next_touch() {
        let cache = SignatureCache::new();
        cache.put_tool("call_1", &sig("a"), ModelFamily::Gemini);

        // Backdate the entry past the TTL.
        {
            let mut inner = cache.inner.lock().unwrap();
            let entry = inner.by_tool_call.get_mut("call_1").unwrap();
            entry.inserted = Instant::now() - TTL - Duration::from_secs(1);
        }

        assert_eq!(
            cache.tool_signature("call_1", ModelFamily::Gemini),
            SignatureLookup::Missing
        );
        assert_eq!(cache.stats().tool_signatures, 0, "expiry must also evict");
    }

    #[test]
    fn an_entry_inside_the_ttl_survives() {
        let cache = SignatureCache::new();
        cache.put_tool("call_1", &sig("a"), ModelFamily::Gemini);
        {
            let mut inner = cache.inner.lock().unwrap();
            let entry = inner.by_tool_call.get_mut("call_1").unwrap();
            entry.inserted = Instant::now() - TTL + Duration::from_secs(60);
        }
        assert!(cache
            .tool_signature("call_1", ModelFamily::Gemini)
            .is_usable());
    }

    #[test]
    fn the_cache_is_bounded() {
        let cache = SignatureCache::new();
        for index in 0..MAX_ENTRIES + 200 {
            cache.put_tool(&format!("call_{index}"), &sig("x"), ModelFamily::Gemini);
        }
        assert!(cache.stats().tool_signatures <= MAX_ENTRIES);
    }

    #[test]
    fn eviction_removes_the_oldest_entries_first() {
        let cache = SignatureCache::new();
        // Fill to the bound, then add one more with a distinctly newer entry.
        for index in 0..MAX_ENTRIES {
            cache.put_tool(&format!("old_{index}"), &sig("x"), ModelFamily::Gemini);
        }
        cache.put_tool("newest", &sig("y"), ModelFamily::Gemini);

        assert!(
            cache.tool_signature("newest", ModelFamily::Gemini).is_usable(),
            "the most recent entry must survive eviction"
        );
        assert!(cache.stats().tool_signatures <= MAX_ENTRIES);
    }

    #[test]
    fn lookups_are_safe_under_concurrency() {
        let cache = std::sync::Arc::new(SignatureCache::new());
        let threads: Vec<_> = (0..8)
            .map(|worker| {
                let cache = cache.clone();
                std::thread::spawn(move || {
                    for index in 0..64 {
                        let key = format!("call_{worker}_{index}");
                        cache.put_tool(&key, &sig("z"), ModelFamily::Gemini);
                        let _ = cache.tool_signature(&key, ModelFamily::Gemini);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert!(cache.stats().tool_signatures <= MAX_ENTRIES);
    }

    #[test]
    fn lookup_helpers_behave() {
        assert!(SignatureLookup::Usable("s".into()).is_usable());
        assert!(!SignatureLookup::Missing.is_usable());
        assert!(!SignatureLookup::ForeignFamily.is_usable());
        assert_eq!(SignatureLookup::Missing.into_option(), None);
    }
}
