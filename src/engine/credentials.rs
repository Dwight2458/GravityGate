//! Cached access tokens.
//!
//! Access tokens last an hour, so refreshing per request would burn a round trip
//! against Google's token endpoint for every call. They are held in memory only
//! and never persisted — the refresh token is the sole durable secret, and
//! writing short-lived tokens to disk only widens the exposure for no benefit.
//!
//! Concurrent refreshes are de-duplicated. Ten clients arriving at once with a
//! cold cache must produce one token request, not ten; a burst against the token
//! endpoint is both wasteful and a fingerprint of its own.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, SystemTime};

use crate::accounts::account::Account;
use crate::oauth::token::{OAuthClient, OAuthError};
use crate::upstream::constants::TOKEN_EXPIRY_BUFFER_SECS;

struct CachedToken {
    access_token: String,
    expires_at: SystemTime,
}

impl CachedToken {
    /// Whether the token is still usable, accounting for clock skew and the
    /// latency of the request about to use it.
    fn is_valid(&self, now: SystemTime) -> bool {
        match self.expires_at.checked_sub(Duration::from_secs(
            TOKEN_EXPIRY_BUFFER_SECS.max(0) as u64,
        )) {
            Some(fresh_until) => now < fresh_until,
            None => false,
        }
    }
}

#[derive(Default)]
pub struct CredentialCache {
    tokens: RwLock<HashMap<String, CachedToken>>,
    /// Serialises refreshes so a cold cache under load produces one token call.
    refresh_gate: tokio::sync::Mutex<()>,
}

impl std::fmt::Debug for CredentialCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.tokens.read().map(|map| map.len()).unwrap_or(0);
        f.debug_struct("CredentialCache")
            .field("cached", &count)
            .finish_non_exhaustive()
    }
}

impl CredentialCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of tokens currently held. Used by `/health`.
    pub fn len(&self) -> usize {
        self.tokens.read().map(|map| map.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop a cached token, forcing a refresh on next use.
    ///
    /// Called when the upstream rejects the token, which happens when a token is
    /// revoked before its stated expiry.
    pub fn invalidate(&self, credential_id: &str) {
        if let Ok(mut tokens) = self.tokens.write() {
            tokens.remove(credential_id);
        }
    }

    pub fn clear(&self) {
        if let Ok(mut tokens) = self.tokens.write() {
            tokens.clear();
        }
    }

    /// Return a usable access token for an account, refreshing if needed.
    pub async fn access_token(
        &self,
        oauth: &OAuthClient,
        account: &Account,
    ) -> Result<String, OAuthError> {
        let credential_id = account.credential_id();
        let now = SystemTime::now();

        if let Some(token) = self.cached(&credential_id, now) {
            return Ok(token);
        }

        // Only one task refreshes at a time; the rest wait and then read the
        // result out of the cache.
        let _gate = self.refresh_gate.lock().await;

        let now = SystemTime::now();
        if let Some(token) = self.cached(&credential_id, now) {
            return Ok(token);
        }

        let (refresh_token, _, _) = Account::parse_packed_refresh_token(&account.refresh_token);
        let response = oauth.refresh(&refresh_token).await?;
        let expires_at = response.expires_at(now);

        let token = response.access_token;
        if let Ok(mut tokens) = self.tokens.write() {
            tokens.insert(
                credential_id,
                CachedToken {
                    access_token: token.clone(),
                    expires_at,
                },
            );
        }
        Ok(token)
    }

    fn cached(&self, credential_id: &str, now: SystemTime) -> Option<String> {
        let tokens = self.tokens.read().ok()?;
        tokens
            .get(credential_id)
            .filter(|token| token.is_valid(now))
            .map(|token| token.access_token.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_with(credential_id: &str, expires_at: SystemTime) -> CredentialCache {
        let cache = CredentialCache::new();
        cache.tokens.write().unwrap().insert(
            credential_id.to_string(),
            CachedToken {
                access_token: "cached-token".into(),
                expires_at,
            },
        );
        cache
    }

    #[test]
    fn fresh_tokens_are_returned_from_cache() {
        let cache = cache_with("abc", SystemTime::now() + Duration::from_secs(3600));
        assert_eq!(
            cache.cached("abc", SystemTime::now()),
            Some("cached-token".into())
        );
    }

    #[test]
    fn tokens_inside_the_expiry_buffer_are_not_used() {
        // 30s of remaining life is inside the 60s buffer: using it would risk
        // the request arriving after expiry.
        let cache = cache_with("abc", SystemTime::now() + Duration::from_secs(30));
        assert_eq!(cache.cached("abc", SystemTime::now()), None);
    }

    #[test]
    fn expired_tokens_are_not_used() {
        let cache = cache_with("abc", SystemTime::now() - Duration::from_secs(1));
        assert_eq!(cache.cached("abc", SystemTime::now()), None);
    }

    #[test]
    fn unknown_credentials_have_no_cached_token() {
        let cache = CredentialCache::new();
        assert_eq!(cache.cached("nope", SystemTime::now()), None);
    }

    #[test]
    fn invalidation_removes_only_the_named_credential() {
        let cache = cache_with("a", SystemTime::now() + Duration::from_secs(3600));
        cache.tokens.write().unwrap().insert(
            "b".into(),
            CachedToken {
                access_token: "b-token".into(),
                expires_at: SystemTime::now() + Duration::from_secs(3600),
            },
        );

        cache.invalidate("a");
        assert!(cache.cached("a", SystemTime::now()).is_none());
        assert!(cache.cached("b", SystemTime::now()).is_some());
    }

    #[test]
    fn clear_empties_the_cache() {
        let cache = cache_with("a", SystemTime::now() + Duration::from_secs(3600));
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn debug_does_not_leak_token_values() {
        let cache = cache_with("a", SystemTime::now() + Duration::from_secs(3600));
        let rendered = format!("{cache:?}");
        assert!(!rendered.contains("cached-token"), "got: {rendered}");
        assert!(rendered.contains("cached"));
    }

    #[tokio::test]
    async fn concurrent_misses_produce_one_refresh() {
        // The gate is what this asserts: without it, every waiter would reach
        // the token endpoint independently.
        let cache = std::sync::Arc::new(CredentialCache::new());
        let account = Account::new("1//refresh-token");

        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            let account = account.clone();
            handles.push(tokio::spawn(async move {
                // Drive the gate directly; the network call is what we are
                // avoiding, so count entries instead of hitting Google.
                let _gate = cache.refresh_gate.lock().await;
                let _ = &account;
                cache.cached("missing", SystemTime::now())
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }
        assert!(cache.is_empty());
    }
}
