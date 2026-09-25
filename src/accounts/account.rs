//! Account model and on-disk schema.
//!
//! The schema follows `antigravity-auth`'s V3/V4 shape, which is the most
//! complete of the reference implementations and, more importantly, the one
//! that records *why* an account is unusable. A single `is_invalid` boolean —
//! all `antigravity-gateway` offers — tells an operator nothing about whether
//! they are rate-limited for five minutes or permanently banned.
//!
//! The refresh token is the only durable secret. Its `Debug` representation is
//! redacted so it cannot leak through a log line or a panic message.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Current on-disk schema version. Bump when the shape changes incompatibly.
pub const SCHEMA_VERSION: u32 = 1;

/// Why an account is temporarily out of rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CooldownReason {
    /// Repeated authentication failures.
    AuthFailure,
    /// Repeated transport failures.
    NetworkError,
    /// The upstream rejected the project we are using.
    ProjectError,
    /// The upstream is demanding account-holder verification.
    ValidationRequired,
}

/// A single account in the pool.
#[derive(Clone, Serialize, Deserialize)]
pub struct Account {
    /// Display label, normally the Google account email.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,

    /// The long-lived secret. Treated as opaque: the upstream also accepts the
    /// packed `refresh|project|managedProject` form, which is preserved verbatim.
    pub refresh_token: String,

    /// Project used for generation, once resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Project the backend provisioned, when it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub managed_project_id: Option<String>,

    #[serde(default)]
    pub added_at: i64,
    #[serde(default)]
    pub last_used: i64,
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Per-quota-key rate-limit expiry, in epoch milliseconds. Keys are quota
    /// pools (`claude`, `gemini`) rather than models, because that is the
    /// granularity the upstream actually enforces.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub rate_limit_reset_times: BTreeMap<String, i64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooling_down_until: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooldown_reason: Option<CooldownReason>,

    /// Set when the upstream demands account-holder verification.
    ///
    /// The hold blocks dispatch only until the recheck window opens
    /// ([`Self::verification_recheck_at`]): completing the challenge in the
    /// browser is invisible to the gateway, so the only way to learn it is to
    /// let a real request through and see what the upstream says.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub verification_required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_required_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_required_reason: Option<String>,
    /// URL the account holder must visit. Surfaced in `/health` so an operator
    /// can act on it without digging through logs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_url: Option<String>,
    /// When dispatch may test the hold again, in epoch milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_recheck_at: Option<i64>,
    /// How many times the hold has been (re)asserted, which sets the recheck
    /// interval. Reset when the upstream serves the account again.
    #[serde(default)]
    pub verification_attempts: u32,

    /// Set when the upstream reports the account as ineligible.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub account_ineligible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_ineligible_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_ineligible_reason: Option<String>,

    /// Last known subscription tier, captured from `loadCodeAssist`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captured_tier_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captured_paid_tier_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captured_tier_at: Option<i64>,
}

fn default_true() -> bool {
    true
}

/// How long a verification hold waits before letting dispatch re-check it,
/// indexed by how many times the hold has been asserted minus one.
///
/// Escalating, because a hold that keeps re-asserting itself is probably real
/// and each re-check costs a guaranteed 403. The first window is short on
/// purpose: the common case is an operator who has just completed the
/// challenge in the browser, and minutes of delay after that is dead time.
const VERIFICATION_RECHECK_LADDER_MS: &[i64] = &[
    60 * 1000,
    5 * 60 * 1000,
    15 * 60 * 1000,
    60 * 60 * 1000,
];

fn verification_recheck_interval_ms(attempts: u32) -> i64 {
    let index = (attempts.saturating_sub(1) as usize).min(VERIFICATION_RECHECK_LADDER_MS.len() - 1);
    VERIFICATION_RECHECK_LADDER_MS[index]
}

impl std::fmt::Debug for Account {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Account")
            .field("email", &self.email)
            .field("refresh_token", &"<redacted>")
            .field("project_id", &self.project_id)
            .field("managed_project_id", &self.managed_project_id)
            .field("enabled", &self.enabled)
            .field("rate_limit_reset_times", &self.rate_limit_reset_times)
            .field("cooling_down_until", &self.cooling_down_until)
            .field("verification_required", &self.verification_required)
            .field("verification_recheck_at", &self.verification_recheck_at)
            .field("verification_attempts", &self.verification_attempts)
            .field("account_ineligible", &self.account_ineligible)
            .field("captured_tier_id", &self.captured_tier_id)
            .finish()
    }
}

impl Account {
    pub fn new(refresh_token: impl Into<String>) -> Self {
        Self {
            email: None,
            refresh_token: refresh_token.into(),
            project_id: None,
            managed_project_id: None,
            added_at: now_ms(),
            last_used: 0,
            enabled: true,
            rate_limit_reset_times: BTreeMap::new(),
            cooling_down_until: None,
            cooldown_reason: None,
            verification_required: false,
            verification_required_at: None,
            verification_required_reason: None,
            verification_url: None,
            verification_recheck_at: None,
            verification_attempts: 0,
            account_ineligible: false,
            account_ineligible_at: None,
            account_ineligible_reason: None,
            captured_tier_id: None,
            captured_paid_tier_id: None,
            captured_tier_at: None,
        }
    }

    /// Stable, non-reversible identifier derived from the refresh token.
    ///
    /// Used as a cache key and as an audit-log key so that neither has to hold
    /// the secret itself.
    pub fn credential_id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.refresh_token.as_bytes());
        let digest = hasher.finalize();
        digest[..12].iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Label for logs and `/health`, preferring email and falling back to the
    /// credential id so the account is still identifiable.
    pub fn label(&self) -> String {
        self.email
            .clone()
            .unwrap_or_else(|| format!("<{}>", self.credential_id()))
    }

    /// Whether this account may be dispatched to right now.
    ///
    /// Excludes disabled accounts, accounts inside an unexpired verification or
    /// ineligibility hold, and accounts still inside a cooldown. A verification
    /// hold is not indefinite — see [`Self::verification_blocks_dispatch`].
    pub fn is_available(&self, now: i64) -> bool {
        self.enabled
            && !self.verification_blocks_dispatch(now)
            && !self.account_ineligible
            && self.cooling_down_until.is_none_or(|until| until <= now)
            && !self.is_rate_limited_for_any(now)
    }

    /// Whether the verification hold still blocks dispatch at `now`.
    ///
    /// Completing the challenge upstream is invisible to the gateway — nothing
    /// pushes the news — so the hold expires on its own and lets a real request
    /// learn the truth. That request either succeeds, which clears the hold, or
    /// draws the 403 that re-arms it. An absent window (an account marked by an
    /// older build) counts as due immediately, so pre-existing holds self-heal.
    fn verification_blocks_dispatch(&self, now: i64) -> bool {
        self.verification_required
            && self.verification_recheck_at.is_some_and(|at| at > now)
    }

    /// Whether any quota pool is currently rate-limited.
    pub fn is_rate_limited_for_any(&self, now: i64) -> bool {
        self.rate_limit_reset_times
            .values()
            .any(|reset| *reset > now)
    }

    /// Whether a specific quota pool is rate-limited.
    pub fn is_rate_limited(&self, pool: &str, now: i64) -> bool {
        self.rate_limit_reset_times
            .get(pool)
            .is_some_and(|reset| *reset > now)
    }

    /// Earliest moment any pool frees up, if any is limited.
    pub fn next_reset_at(&self, now: i64) -> Option<i64> {
        self.rate_limit_reset_times
            .values()
            .filter(|reset| **reset > now)
            .min()
            .copied()
    }

    /// Clear expired rate-limit entries. Returns true when something changed, so
    /// callers only persist when it matters.
    pub fn clear_expired_limits(&mut self, now: i64) -> bool {
        let before = self.rate_limit_reset_times.len();
        self.rate_limit_reset_times.retain(|_, reset| *reset > now);
        let cooldown_cleared = match self.cooling_down_until {
            Some(until) if until <= now => {
                self.cooling_down_until = None;
                self.cooldown_reason = None;
                true
            }
            _ => false,
        };
        before != self.rate_limit_reset_times.len() || cooldown_cleared
    }

    /// Mark a quota pool as rate-limited until `reset_at`.
    ///
    /// Keeps the furthest-out expiry so that a short limit reported after a long
    /// one cannot shorten it — the upstream's resets are the authority, and
    /// shortening would produce a request that is guaranteed to fail.
    pub fn mark_rate_limited(&mut self, pool: &str, reset_at: i64) {
        let entry = self
            .rate_limit_reset_times
            .entry(pool.to_string())
            .or_insert(reset_at);
        if reset_at > *entry {
            *entry = reset_at;
        }
    }

    pub fn mark_cooling_down(&mut self, until: i64, reason: CooldownReason) {
        self.cooling_down_until = Some(until);
        self.cooldown_reason = Some(reason);
    }

    pub fn mark_verification_required(&mut self, url: Option<String>, reason: impl Into<String>) {
        let now = now_ms();
        self.verification_required = true;
        self.verification_required_at = Some(now);
        self.verification_attempts = self.verification_attempts.saturating_add(1);
        self.verification_recheck_at =
            Some(now + verification_recheck_interval_ms(self.verification_attempts));
        self.verification_url = url;
        self.verification_required_reason = Some(reason.into());
    }

    /// Clear the verification hold: the upstream just served this account, so
    /// the demand, whatever it once was, is satisfied.
    pub fn clear_verification(&mut self) {
        self.verification_required = false;
        self.verification_required_at = None;
        self.verification_required_reason = None;
        self.verification_url = None;
        self.verification_recheck_at = None;
        self.verification_attempts = 0;
    }

    pub fn mark_ineligible(&mut self, reason: impl Into<String>) {
        self.account_ineligible = true;
        self.account_ineligible_at = Some(now_ms());
        self.account_ineligible_reason = Some(reason.into());
    }

    /// Clear operator-resolvable holds so an account can be retried.
    pub fn clear_holds(&mut self) {
        self.clear_verification();
        self.account_ineligible = false;
        self.account_ineligible_at = None;
        self.account_ineligible_reason = None;
        self.cooling_down_until = None;
        self.cooldown_reason = None;
        self.rate_limit_reset_times.clear();
    }

    /// Split a packed refresh token into its parts.
    ///
    /// The upstream tooling stores `refresh|project|managedProject` as one
    /// opaque string. We preserve the packed form verbatim for round-tripping
    /// and expose the parts for display.
    pub fn parse_packed_refresh_token(token: &str) -> (String, Option<String>, Option<String>) {
        let mut parts = token.split('|');
        let refresh = parts.next().unwrap_or_default().to_string();
        let project = parts.next().filter(|s| !s.is_empty()).map(str::to_string);
        let managed = parts.next().filter(|s| !s.is_empty()).map(str::to_string);
        (refresh, project, managed)
    }

    /// Adopt the project ids embedded in a packed refresh token.
    pub fn absorb_packed_project(&mut self) {
        let (_, project, managed) = Self::parse_packed_refresh_token(&self.refresh_token);
        if self.project_id.is_none() {
            self.project_id = project;
        }
        if self.managed_project_id.is_none() {
            self.managed_project_id = managed;
        }
    }
}

/// The whole on-disk pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountStorage {
    pub version: u32,
    #[serde(default)]
    pub accounts: Vec<Account>,
    #[serde(default)]
    pub active_index: usize,
    /// Separate rotation cursor per model family, so Claude traffic and Gemini
    /// traffic do not drag each other's sticky account around.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub active_index_by_family: BTreeMap<String, usize>,
}

impl Default for AccountStorage {
    fn default() -> Self {
        Self {
            version: SCHEMA_VERSION,
            accounts: Vec::new(),
            active_index: 0,
            active_index_by_family: BTreeMap::new(),
        }
    }
}

impl AccountStorage {
    pub fn is_empty(&self) -> bool {
        self.accounts.is_empty()
    }

    pub fn len(&self) -> usize {
        self.accounts.len()
    }

    /// Clamp the rotation cursor into range. Called after every load and every
    /// removal, since a stale index would otherwise panic on the next select.
    pub fn clamp_indices(&mut self) {
        if self.accounts.is_empty() {
            self.active_index = 0;
            self.active_index_by_family.clear();
            return;
        }
        self.active_index = self.active_index.min(self.accounts.len() - 1);
        let len = self.accounts.len();
        self.active_index_by_family
            .retain(|_, index| *index < len);
    }
}

/// Current wall clock in epoch milliseconds.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_output_redacts_the_refresh_token() {
        let account = Account::new("1//super-secret-refresh-token");
        let rendered = format!("{account:?}");
        assert!(!rendered.contains("super-secret"), "got: {rendered}");
        assert!(rendered.contains("<redacted>"));
    }

    #[test]
    fn credential_id_is_stable_and_not_the_token() {
        let account = Account::new("token-a");
        let id = account.credential_id();
        assert_eq!(id, Account::new("token-a").credential_id());
        assert_ne!(id, Account::new("token-b").credential_id());
        assert_eq!(id.len(), 24, "12 bytes rendered as hex");
        assert!(!id.contains("token-a"));
    }

    #[test]
    fn label_prefers_email_and_falls_back_to_credential_id() {
        let mut account = Account::new("t");
        assert!(account.label().starts_with('<'));
        account.email = Some("someone@example.com".into());
        assert_eq!(account.label(), "someone@example.com");
    }

    #[test]
    fn fresh_accounts_are_available() {
        let account = Account::new("t");
        assert!(account.is_available(now_ms()));
    }

    #[test]
    fn disabled_accounts_are_not_available() {
        let mut account = Account::new("t");
        account.enabled = false;
        assert!(!account.is_available(now_ms()));
    }

    #[test]
    fn verification_hold_blocks_dispatch() {
        let mut account = Account::new("t");
        account.mark_verification_required(Some("https://accounts.google.com/x".into()), "asked");
        assert!(!account.is_available(now_ms()));
        assert_eq!(
            account.verification_url.as_deref(),
            Some("https://accounts.google.com/x")
        );
    }

    #[test]
    fn a_verification_hold_opens_for_recheck_when_its_window_expires() {
        // Completing the challenge is invisible to the gateway, so the hold has
        // to expire on its own: the next request is what learns the truth.
        let mut account = Account::new("t");
        account.mark_verification_required(Some("https://accounts.google.com/x".into()), "asked");
        let recheck_at = account.verification_recheck_at.expect("a recheck window");

        assert!(!account.is_available(recheck_at - 1));
        assert!(account.is_available(recheck_at), "due is available");
        assert!(
            account.verification_required,
            "availability is not proof; the flag clears on success"
        );
    }

    #[test]
    fn a_verification_hold_without_a_window_is_due_immediately() {
        // Accounts marked by an older build carry no recheck time; without this
        // they would be held forever.
        let mut account = Account::new("t");
        account.verification_required = true;
        assert!(account.is_available(now_ms()));
    }

    #[test]
    fn verification_rechecks_escalate_and_plateau() {
        let interval = |attempts: u32| verification_recheck_interval_ms(attempts);
        assert_eq!(interval(1), 60 * 1000);
        assert_eq!(interval(2), 5 * 60 * 1000);
        assert_eq!(interval(3), 15 * 60 * 1000);
        assert_eq!(interval(4), 60 * 60 * 1000);
        assert_eq!(interval(99), 60 * 60 * 1000, "past the end, it stays put");
    }

    #[test]
    fn a_reasserted_hold_arms_a_longer_window() {
        let mut account = Account::new("t");
        account.mark_verification_required(None, "asked");
        let first = account.verification_recheck_at.expect("a window");

        account.mark_verification_required(None, "asked again");
        let second = account.verification_recheck_at.expect("a window");
        assert!(second > first, "each assertion waits longer");
        assert_eq!(account.verification_attempts, 2);
    }

    #[test]
    fn clearing_verification_resets_the_ladder() {
        let mut account = Account::new("t");
        account.mark_verification_required(None, "asked");
        account.mark_verification_required(None, "asked again");
        assert_eq!(account.verification_attempts, 2);

        account.clear_verification();
        assert!(!account.verification_required);
        assert_eq!(account.verification_attempts, 0);
        assert!(account.verification_recheck_at.is_none());
        assert!(account.verification_url.is_none());
    }

    #[test]
    fn ineligibility_blocks_dispatch() {
        let mut account = Account::new("t");
        account.mark_ineligible("ACCOUNT_INELIGIBLE");
        assert!(!account.is_available(now_ms()));
    }

    #[test]
    fn expired_cooldown_does_not_block() {
        let now = now_ms();
        let mut account = Account::new("t");
        account.mark_cooling_down(now - 1, CooldownReason::NetworkError);
        assert!(account.is_available(now));
    }

    #[test]
    fn active_cooldown_blocks() {
        let now = now_ms();
        let mut account = Account::new("t");
        account.mark_cooling_down(now + 60_000, CooldownReason::AuthFailure);
        assert!(!account.is_available(now));
        assert!(account.is_available(now + 60_001));
    }

    #[test]
    fn rate_limit_is_scoped_to_its_pool() {
        let now = now_ms();
        let mut account = Account::new("t");
        account.mark_rate_limited("claude", now + 60_000);

        assert!(account.is_rate_limited("claude", now));
        assert!(!account.is_rate_limited("gemini", now));
        // Any active limit takes the account out of rotation entirely.
        assert!(account.is_rate_limited_for_any(now));
        assert!(!account.is_available(now));
    }

    #[test]
    fn marking_a_shorter_limit_does_not_shorten_an_existing_one() {
        let now = now_ms();
        let mut account = Account::new("t");
        account.mark_rate_limited("claude", now + 120_000);
        account.mark_rate_limited("claude", now + 5_000);
        assert_eq!(account.rate_limit_reset_times["claude"], now + 120_000);
    }

    #[test]
    fn marking_a_longer_limit_extends_it() {
        let now = now_ms();
        let mut account = Account::new("t");
        account.mark_rate_limited("claude", now + 5_000);
        account.mark_rate_limited("claude", now + 120_000);
        assert_eq!(account.rate_limit_reset_times["claude"], now + 120_000);
    }

    #[test]
    fn clearing_expired_limits_reports_change_once() {
        let now = now_ms();
        let mut account = Account::new("t");
        account.mark_rate_limited("claude", now - 1);
        account.mark_rate_limited("gemini", now + 60_000);

        assert!(account.clear_expired_limits(now), "should report a change");
        assert!(!account.clear_expired_limits(now), "second call is a no-op");
        assert_eq!(account.rate_limit_reset_times.len(), 1);
        assert!(account.rate_limit_reset_times.contains_key("gemini"));
    }

    #[test]
    fn clearing_expired_limits_also_clears_finished_cooldown() {
        let now = now_ms();
        let mut account = Account::new("t");
        account.mark_cooling_down(now - 1, CooldownReason::AuthFailure);
        assert!(account.clear_expired_limits(now));
        assert!(account.cooling_down_until.is_none());
        assert!(account.cooldown_reason.is_none());
    }

    #[test]
    fn next_reset_picks_the_earliest_future_limit() {
        let now = now_ms();
        let mut account = Account::new("t");
        account.mark_rate_limited("claude", now + 90_000);
        account.mark_rate_limited("gemini", now + 30_000);
        account.mark_rate_limited("stale", now - 10_000);
        assert_eq!(account.next_reset_at(now), Some(now + 30_000));
    }

    #[test]
    fn next_reset_is_none_when_unlimited() {
        let account = Account::new("t");
        assert_eq!(account.next_reset_at(now_ms()), None);
    }

    #[test]
    fn clear_holds_releases_everything_except_enabled() {
        let now = now_ms();
        let mut account = Account::new("t");
        account.mark_verification_required(Some("u".into()), "reason");
        account.mark_ineligible("nope");
        account.mark_cooling_down(now + 60_000, CooldownReason::NetworkError);
        account.mark_rate_limited("claude", now + 60_000);

        account.clear_holds();

        assert!(account.enabled, "clear_holds must not re-enable a disabled account");
        assert!(!account.verification_required);
        assert!(!account.account_ineligible);
        assert!(account.cooling_down_until.is_none());
        assert!(account.rate_limit_reset_times.is_empty());
        assert!(account.is_available(now));
    }

    #[test]
    fn packed_refresh_token_is_split_into_parts() {
        let (refresh, project, managed) =
            Account::parse_packed_refresh_token("1//refresh|proj-a|managed-b");
        assert_eq!(refresh, "1//refresh");
        assert_eq!(project.as_deref(), Some("proj-a"));
        assert_eq!(managed.as_deref(), Some("managed-b"));
    }

    #[test]
    fn unpacked_refresh_token_yields_no_project() {
        let (refresh, project, managed) = Account::parse_packed_refresh_token("1//plain");
        assert_eq!(refresh, "1//plain");
        assert_eq!(project, None);
        assert_eq!(managed, None);
    }

    #[test]
    fn empty_packed_segments_are_treated_as_absent() {
        let (_, project, managed) = Account::parse_packed_refresh_token("1//r||");
        assert_eq!(project, None);
        assert_eq!(managed, None);
    }

    #[test]
    fn absorb_packed_project_fills_only_missing_fields() {
        let mut account = Account::new("1//r|from-token|managed-from-token");
        account.project_id = Some("already-set".into());
        account.absorb_packed_project();
        assert_eq!(account.project_id.as_deref(), Some("already-set"));
        assert_eq!(
            account.managed_project_id.as_deref(),
            Some("managed-from-token")
        );
    }

    #[test]
    fn storage_round_trips_through_json() {
        let mut storage = AccountStorage::default();
        storage.accounts.push(Account::new("1//r"));
        storage.active_index = 0;

        let json = serde_json::to_string(&storage).unwrap();
        let parsed: AccountStorage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, SCHEMA_VERSION);
        assert_eq!(parsed.accounts.len(), 1);
        assert_eq!(parsed.accounts[0].refresh_token, "1//r");
    }

    #[test]
    fn empty_optional_fields_are_omitted_from_json() {
        let json = serde_json::to_value(Account::new("1//r")).unwrap();
        let object = json.as_object().unwrap();
        // Keeps the file small and readable, and avoids null-vs-absent ambiguity.
        assert!(!object.contains_key("email"));
        assert!(!object.contains_key("rate_limit_reset_times"));
        assert!(!object.contains_key("verification_required"));
        assert!(object.contains_key("refresh_token"));
    }

    #[test]
    fn clamp_handles_empty_and_overrun_indices() {
        let mut storage = AccountStorage {
            active_index: 7,
            ..Default::default()
        };
        storage.clamp_indices();
        assert_eq!(storage.active_index, 0);

        storage.accounts.push(Account::new("a"));
        storage.accounts.push(Account::new("b"));
        storage.active_index = 9;
        storage.active_index_by_family.insert("claude".into(), 9);
        storage.clamp_indices();
        assert_eq!(storage.active_index, 1);
        assert!(storage.active_index_by_family.is_empty());
    }

    #[test]
    fn missing_version_is_a_parse_error_not_a_silent_default() {
        // A file without a version is not something we wrote.
        let json = r#"{"accounts":[]}"#;
        let parsed: Result<AccountStorage, _> = serde_json::from_str(json);
        assert!(parsed.is_err());
    }
}
