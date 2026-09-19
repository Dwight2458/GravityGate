//! Configuration loading.
//!
//! Precedence, lowest to highest: built-in defaults, the TOML file, `GG_*`
//! environment variables, command-line flags. Every field has a default, so a
//! missing config file is not an error.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::upstream::constants;

/// Environment variable naming the config directory.
pub const ENV_CONFIG_DIR: &str = "GRAVITYGATE_CONFIG_DIR";
/// Environment variable naming the config file explicitly.
pub const ENV_CONFIG_FILE: &str = "GRAVITYGATE_CONFIG_FILE";

/// Default listen port. Matches every reference implementation, which keeps
/// client documentation copy-pasteable.
pub const DEFAULT_PORT: u16 = 8080;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    pub upstream: UpstreamConfig,
    pub accounts: AccountsConfig,
    pub routing: RoutingConfig,
    pub reasoning: ReasoningConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Accepted client credentials. Empty means no authentication, which is the
    /// sensible default for a loopback-only gateway.
    pub api_keys: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: DEFAULT_PORT,
            api_keys: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UpstreamConfig {
    /// Overridable so tests can point at a local server.
    pub endpoints: Vec<String>,
    /// Random delay added before dispatch, in milliseconds. Zero by default:
    /// a gateway serves many clients and predictable timing is easier to reason
    /// about than artificial variance.
    pub request_jitter_max_ms: u64,
    /// Prepend the Antigravity IDE's own system prompt to every request.
    ///
    /// Makes traffic look more like the real CLI, but that prompt competes with
    /// whatever the caller sent, so it is off by default. Turn it on only if
    /// upstream validation turns out to care.
    pub inject_agent_system_prompt: bool,
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        Self {
            endpoints: constants::ENDPOINT_FALLBACKS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            request_jitter_max_ms: 0,
            inject_agent_system_prompt: false,
        }
    }
}

/// Account selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Strategy {
    /// Health score, token bucket, and freshness combined. The default.
    #[default]
    Hybrid,
    /// Stay on one account to protect the prompt cache; move only on failure.
    Sticky,
    /// Rotate strictly in order.
    RoundRobin,
    /// Always pick the least recently used account.
    LeastRecentlyUsed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AccountsConfig {
    pub strategy: Strategy,
    /// Cooldown applied after repeated transport failures.
    pub cooldown_secs: u64,
    /// Consecutive failures of any kind before an account is cooled down.
    pub max_consecutive_failures: u32,
    /// Rotate away once remaining quota drops below this fraction.
    pub soft_quota_threshold: f64,
    /// How often quota is refreshed, in seconds.
    pub quota_refresh_secs: u64,
    /// Token bucket ceiling per account.
    pub token_bucket_max: f64,
    /// Token bucket refill rate per minute.
    pub token_bucket_refill_per_min: f64,
}

impl Default for AccountsConfig {
    fn default() -> Self {
        Self {
            strategy: Strategy::default(),
            cooldown_secs: 60,
            max_consecutive_failures: 3,
            soft_quota_threshold: 0.20,
            quota_refresh_secs: 30 * 60,
            token_bucket_max: 50.0,
            token_bucket_refill_per_min: 6.0,
        }
    }
}

impl Strategy {
    /// Parse a strategy name.
    ///
    /// Accepts the canonical kebab-case spelling used in TOML and a short alias
    /// for each, so the environment variable and the config file cannot diverge
    /// into two vocabularies for the same setting.
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "hybrid" => Some(Self::Hybrid),
            "sticky" => Some(Self::Sticky),
            "round-robin" | "roundrobin" | "rr" => Some(Self::RoundRobin),
            "least-recently-used" | "lru" => Some(Self::LeastRecentlyUsed),
            _ => None,
        }
    }
}

impl std::str::FromStr for Strategy {
    type Err = ();

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse(text).ok_or(())
    }
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Hybrid => "hybrid",
            Self::Sticky => "sticky",
            Self::RoundRobin => "round-robin",
            Self::LeastRecentlyUsed => "least-recently-used",
        };
        f.write_str(name)
    }
}

/// How an exhausted account pool is reported to clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ExhaustedErrorMode {
    /// `429 Too Many Requests` with `Retry-After`. Semantically correct for the
    /// OpenAI ecosystem, where clients back off and honour the header.
    #[default]
    TooManyRequests,
    /// `400 Bad Request` with `invalid_request_error`. Matches the reference
    /// implementations, which chose it to stop agentic clients retrying a
    /// condition that will not clear.
    BadRequest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RoutingConfig {
    /// Longest a request will wait for a rate-limited pool before erroring.
    pub max_wait_before_error_secs: u64,
    /// Retries across accounts for a single client request.
    pub max_account_attempts: u32,
    /// Extra retries against the same account when the model is capacity-bound.
    pub max_capacity_retries: u32,
    /// Retries when the upstream returns a well-formed but empty stream.
    pub max_empty_response_retries: u32,
    pub exhausted_error_mode: ExhaustedErrorMode,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            max_wait_before_error_secs: 120,
            max_account_attempts: 10,
            max_capacity_retries: 5,
            max_empty_response_retries: 2,
            exhausted_error_mode: ExhaustedErrorMode::default(),
        }
    }
}

/// Which field carries reasoning content on the OpenAI surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ReasoningField {
    /// `reasoning_content`, understood by Cline, Roo, and most OpenAI-compatible
    /// forks.
    #[default]
    ReasoningContent,
    /// `reasoning`, the OpenRouter spelling.
    Reasoning,
    /// Emit both. Broadest compatibility, at the cost of duplicate payload.
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReasoningConfig {
    pub output_field: ReasoningField,
    /// Default thinking tier when the client expresses no preference.
    pub default_tier: String,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            output_field: ReasoningField::default(),
            default_tier: "medium".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingConfig {
    pub level: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
        }
    }
}

/// Resolve the configuration directory.
///
/// Order: `$GRAVITYGATE_CONFIG_DIR`, then `$XDG_CONFIG_HOME/gravitygate`, then
/// `%APPDATA%\gravitygate` on Windows or `~/.config/gravitygate` elsewhere.
pub fn config_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(ENV_CONFIG_DIR) {
        return PathBuf::from(dir);
    }
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME")
        && !xdg.is_empty()
    {
        return PathBuf::from(xdg).join("gravitygate");
    }
    if cfg!(windows)
        && let Some(appdata) = std::env::var_os("APPDATA")
    {
        return PathBuf::from(appdata).join("gravitygate");
    }
    if let Some(home) = home_dir() {
        return home.join(".config").join("gravitygate");
    }
    PathBuf::from(".gravitygate")
}

/// Path to the account pool file.
pub fn accounts_path() -> PathBuf {
    config_dir().join("accounts.json")
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

impl Config {
    /// Load configuration, treating a missing file as all-defaults.
    pub fn load(path: Option<&Path>) -> Result<Self, ConfigError> {
        let path = match path {
            Some(p) => p.to_path_buf(),
            None => match std::env::var_os(ENV_CONFIG_FILE) {
                Some(p) => PathBuf::from(p),
                None => config_dir().join("config.toml"),
            },
        };

        let mut config = if path.exists() {
            let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Read {
                path: path.clone(),
                source,
            })?;
            toml::from_str(&text).map_err(|source| ConfigError::Parse {
                path: path.clone(),
                source,
            })?
        } else {
            Self::default()
        };

        config.apply_env();
        config.validate()?;
        Ok(config)
    }

    /// Apply `GG_*` overrides for the small set of values that are commonly
    /// changed in containers.
    fn apply_env(&mut self) {
        if let Ok(host) = std::env::var("GG_HOST") {
            self.server.host = host;
        }
        if let Ok(port) = std::env::var("GG_PORT")
            && let Ok(port) = port.parse()
        {
            self.server.port = port;
        }
        if let Ok(level) = std::env::var("GG_LOG_LEVEL") {
            self.logging.level = level;
        }
        if let Ok(strategy) = std::env::var("GG_ACCOUNT_STRATEGY") {
            match Strategy::parse(&strategy) {
                Some(parsed) => self.accounts.strategy = parsed,
                // Silently ignoring a typo here would leave the operator
                // believing a strategy is in force that is not.
                None => tracing::warn!(
                    value = %strategy,
                    "GG_ACCOUNT_STRATEGY is not a known strategy; ignoring"
                ),
            }
        }
        if let Ok(keys) = std::env::var("GG_API_KEYS") {
            self.server.api_keys = keys
                .split(',')
                .map(str::trim)
                .filter(|k| !k.is_empty())
                .map(str::to_string)
                .collect();
        }
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.server.port == 0 {
            return Err(ConfigError::Invalid(
                "server.port must not be zero".into(),
            ));
        }
        if self.upstream.endpoints.is_empty() {
            return Err(ConfigError::Invalid(
                "upstream.endpoints must contain at least one endpoint".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.accounts.soft_quota_threshold) {
            return Err(ConfigError::Invalid(
                "accounts.soft_quota_threshold must be within 0.0..=1.0".into(),
            ));
        }
        let known_tiers = ["minimal", "low", "medium", "high"];
        if !known_tiers.contains(&self.reasoning.default_tier.as_str()) {
            return Err(ConfigError::Invalid(format!(
                "reasoning.default_tier must be one of {known_tiers:?}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("could not read config at {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not parse config at {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("invalid config: {0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_usable_without_a_file() {
        let config = Config::default();
        assert_eq!(config.server.port, DEFAULT_PORT);
        assert_eq!(config.server.host, "127.0.0.1");
        assert!(config.server.api_keys.is_empty());
        assert_eq!(config.accounts.strategy, Strategy::Hybrid);
        assert_eq!(config.reasoning.output_field, ReasoningField::ReasoningContent);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn upstream_defaults_to_daily_then_prod() {
        let config = Config::default();
        assert_eq!(
            config.upstream.endpoints,
            vec![constants::ENDPOINT_DAILY, constants::ENDPOINT_PROD]
        );
    }

    #[test]
    fn partial_toml_fills_remaining_fields_with_defaults() {
        let toml = r#"
            [server]
            port = 9999
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.server.port, 9999);
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.accounts.strategy, Strategy::Hybrid);
    }

    #[test]
    fn strategy_parsing_accepts_both_spellings() {
        // The env override and the config file must not use different
        // vocabularies for the same setting.
        assert_eq!(Strategy::parse("least-recently-used"), Some(Strategy::LeastRecentlyUsed));
        assert_eq!(Strategy::parse("lru"), Some(Strategy::LeastRecentlyUsed));
        assert_eq!(Strategy::parse("round-robin"), Some(Strategy::RoundRobin));
        assert_eq!(Strategy::parse("RR"), Some(Strategy::RoundRobin));
        assert_eq!(Strategy::parse("  Hybrid "), Some(Strategy::Hybrid));
        assert_eq!(Strategy::parse("nonsense"), None);
    }

    #[test]
    fn every_strategy_round_trips_through_its_own_name() {
        for strategy in [
            Strategy::Hybrid,
            Strategy::Sticky,
            Strategy::RoundRobin,
            Strategy::LeastRecentlyUsed,
        ] {
            assert_eq!(
                Strategy::parse(&strategy.to_string()),
                Some(strategy),
                "{strategy} did not round-trip"
            );
        }
    }

    #[test]
    fn strategy_names_parse_kebab_case() {
        for (text, expected) in [
            ("hybrid", Strategy::Hybrid),
            ("sticky", Strategy::Sticky),
            ("round-robin", Strategy::RoundRobin),
            ("least-recently-used", Strategy::LeastRecentlyUsed),
        ] {
            let toml = format!("[accounts]\nstrategy = \"{text}\"");
            let config: Config = toml::from_str(&toml).unwrap();
            assert_eq!(config.accounts.strategy, expected, "parsing {text}");
        }
    }

    #[test]
    fn exhausted_error_mode_defaults_to_429_semantics() {
        assert_eq!(
            RoutingConfig::default().exhausted_error_mode,
            ExhaustedErrorMode::TooManyRequests
        );
    }

    #[test]
    fn unknown_keys_are_rejected() {
        // A typo in a config file should be loud, not silently ignored.
        let toml = "[server]\nprot = 8080";
        assert!(toml::from_str::<Config>(toml).is_err());
    }

    #[test]
    fn zero_port_is_rejected() {
        let mut config = Config::default();
        config.server.port = 0;
        assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn empty_endpoints_are_rejected() {
        let mut config = Config::default();
        config.upstream.endpoints.clear();
        assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn quota_threshold_outside_unit_range_is_rejected() {
        let mut config = Config::default();
        config.accounts.soft_quota_threshold = 1.5;
        assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn unknown_default_tier_is_rejected() {
        let mut config = Config::default();
        config.reasoning.default_tier = "enormous".into();
        assert!(matches!(config.validate(), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn config_dir_env_override_wins() {
        // Serialised because env vars are process-global and other tests read
        // the same variables.
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var(ENV_CONFIG_DIR, "/tmp/gg-test") };
        assert_eq!(config_dir(), PathBuf::from("/tmp/gg-test"));
        unsafe { std::env::remove_var(ENV_CONFIG_DIR) };
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
}
