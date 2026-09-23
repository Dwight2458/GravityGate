//! GravityGate entry point.

mod cli;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;

use cli::{AccountCommand, Cli, Command, ConfigCommand};
use gravitygate::accounts::account::{Account, AccountStorage, CooldownReason};
use gravitygate::accounts::store::AccountStore;
use gravitygate::config::{Config, accounts_path, config_dir};
use gravitygate::engine::Engine;
use gravitygate::oauth::login::{LoginOptions, login};
use gravitygate::oauth::token::OAuthClient;

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing(cli.verbose);
    // Logged at startup so a stale binary announces itself rather than
    // producing a mystery. This is how the 404 above should have been caught.
    tracing::debug!(version = cli::VERSION, "starting");

    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // `{:#}` renders the whole context chain on one line, which is what
            // an operator wants from a CLI rather than a backtrace.
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Serve { host, port } => cmd_serve(cli.config, host, port).await,
        Command::Account { command } => cmd_account(cli.config, command).await,
        Command::Probe {
            model,
            account,
            prompt,
            raw,
            tool,
            repeat,
        } => cmd_probe(cli.config, account, model, prompt, raw, tool, repeat).await,
        Command::Config { command } => cmd_config(cli.config, command),
    }
}

fn init_tracing(verbosity: u8) {
    use tracing_subscriber::EnvFilter;

    // `RUST_LOG` wins when set, so an operator can get fine-grained control
    // without a rebuild; otherwise verbosity flags set a sane default.
    let default = match verbosity {
        0 => "gravitygate=info,warn",
        1 => "gravitygate=debug,info",
        _ => "gravitygate=trace,debug",
    };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

fn load_config(path: Option<PathBuf>) -> Result<Config> {
    Config::load(path.as_deref()).context("loading configuration")
}

fn open_store() -> Result<AccountStore> {
    let path = accounts_path();
    AccountStore::load(&path)
        .with_context(|| format!("opening account store at {}", path.display()))
}

// ---------------------------------------------------------------------------
// serve
// ---------------------------------------------------------------------------

async fn cmd_serve(
    config_path: Option<PathBuf>,
    host_override: Option<String>,
    port_override: Option<u16>,
) -> Result<()> {
    let mut config = load_config(config_path)?;
    if let Some(host) = host_override {
        config.server.host = host;
    }
    if let Some(port) = port_override {
        config.server.port = port;
    }

    let store = open_store()?;
    if store.snapshot().is_empty() {
        tracing::warn!(
            path = %store.path().display(),
            "no accounts configured; add one with `gravitygate account add-token`"
        );
    }

    gravitygate::server::serve(config, store)
        .await
        .map_err(|error| anyhow::anyhow!("{error:#}"))
}

// ---------------------------------------------------------------------------
// account
// ---------------------------------------------------------------------------

async fn cmd_account(config_path: Option<PathBuf>, command: AccountCommand) -> Result<()> {
    match command {
        AccountCommand::Login {
            no_browser,
            no_open,
            timeout,
            email,
            project,
            no_verify,
        } => {
            let config = load_config(config_path)?;
            let store = open_store()?;
            account_login(
                config,
                store,
                LoginArgs {
                    no_browser,
                    no_open,
                    timeout: Duration::from_secs(timeout),
                    email,
                    project,
                    verify: !no_verify,
                },
            )
            .await
        }

        AccountCommand::AddToken {
            token,
            email,
            project,
        } => {
            let store = open_store()?;
            add_token(&store, &token, email, project)
        }

        AccountCommand::List { json } => {
            let store = open_store()?;
            let storage = store.snapshot();
            if json {
                println!("{}", serde_json::to_string_pretty(&storage)?);
            } else {
                print_accounts(&storage);
            }
            Ok(())
        }

        AccountCommand::Remove { selector } => {
            let store = open_store()?;
            let label = mutate_by_selector(&store, &selector, |storage, index| {
                storage.accounts.remove(index);
            })?;
            println!("Removed {label}.");
            Ok(())
        }

        AccountCommand::Enable { selector } => {
            let store = open_store()?;
            let label = mutate_by_selector(&store, &selector, |storage, index| {
                storage.accounts[index].enabled = true;
            })?;
            println!("Enabled {label}.");
            Ok(())
        }

        AccountCommand::Disable { selector } => {
            let store = open_store()?;
            let label = mutate_by_selector(&store, &selector, |storage, index| {
                storage.accounts[index].enabled = false;
            })?;
            println!("Disabled {label}.");
            Ok(())
        }

        AccountCommand::ClearHolds { selector } => {
            let store = open_store()?;
            let label = mutate_by_selector(&store, &selector, |storage, index| {
                storage.accounts[index].clear_holds();
            })?;
            println!("Cleared rate limits, cooldowns, and holds for {label}.");
            Ok(())
        }

        AccountCommand::Verify { selector } => {
            let config = load_config(config_path)?;
            let store = open_store()?;
            verify_accounts(config, store, selector).await
        }
    }
}

fn add_token(
    store: &AccountStore,
    token: &str,
    email: Option<String>,
    project: Option<String>,
) -> Result<()> {
    let token = token.trim();
    if token.is_empty() {
        bail!("refresh token is empty");
    }

    let mut account = Account::new(token);
    account.email = email;
    account.project_id = project;
    // A packed token carries its own project ids; adopt them unless overridden.
    account.absorb_packed_project();

    report_upsert(&upsert_account(store, account)?);
    Ok(())
}

/// What [`upsert_account`] did.
struct UpsertOutcome {
    label: String,
    short_id: String,
    updated: bool,
}

/// Insert an account, or update it in place if its credentials are already known.
///
/// Updating in place matters: adding the same refresh token twice must not create
/// two entries that split traffic between them and accumulate rate limits
/// independently.
fn upsert_account(store: &AccountStore, account: Account) -> Result<UpsertOutcome> {
    let credential_id = account.credential_id();
    let label = account.label();
    let mut updated = false;

    store.mutate(|storage| {
        if let Some(existing) = storage
            .accounts
            .iter_mut()
            .find(|existing| existing.credential_id() == credential_id)
        {
            if account.email.is_some() {
                existing.email = account.email.clone();
            }
            if account.project_id.is_some() {
                existing.project_id = account.project_id.clone();
            }
            if account.managed_project_id.is_some() {
                existing.managed_project_id = account.managed_project_id.clone();
            }
            // Re-adding is an explicit statement of intent, so re-enable.
            existing.enabled = true;
            updated = true;
            return true;
        }
        storage.accounts.push(account);
        true
    })?;

    Ok(UpsertOutcome {
        label,
        short_id: credential_id[..8].to_string(),
        updated,
    })
}

fn report_upsert(outcome: &UpsertOutcome) {
    let verb = if outcome.updated { "Updated" } else { "Added" };
    println!("{} {} ({}).", verb, outcome.label, outcome.short_id);
    println!("Run `gravitygate probe` to confirm it works before serving traffic.");
}

/// Apply a change to the account a selector names, returning its label.
fn mutate_by_selector<F>(store: &AccountStore, selector: &str, change: F) -> Result<String>
where
    F: FnOnce(&mut AccountStorage, usize),
{
    let storage = store.snapshot();
    let index = resolve_selector(&storage, selector)?;
    let label = storage.accounts[index].label();

    store.mutate(|current| {
        // The snapshot index may be stale if the file changed underneath us.
        if index >= current.accounts.len() {
            return false;
        }
        change(current, index);
        true
    })?;

    Ok(label)
}

/// Resolve a selector to an index.
///
/// Accepts an index, an exact email, or a credential id prefix, so an operator
/// can copy whatever the table printed.
fn resolve_selector(storage: &AccountStorage, selector: &str) -> Result<usize> {
    if storage.accounts.is_empty() {
        bail!("no accounts configured");
    }

    if selector.is_empty() {
        bail!("selector is empty");
    }

    if let Ok(index) = selector.parse::<usize>() {
        if index < storage.accounts.len() {
            return Ok(index);
        }
        bail!("index {index} is out of range (0..{})", storage.accounts.len());
    }

    if let Some(index) = storage
        .accounts
        .iter()
        .position(|account| account.email.as_deref() == Some(selector))
    {
        return Ok(index);
    }

    let matches: Vec<usize> = storage
        .accounts
        .iter()
        .enumerate()
        .filter(|(_, account)| account.credential_id().starts_with(selector))
        .map(|(index, _)| index)
        .collect();

    match matches.as_slice() {
        [index] => Ok(*index),
        [] => bail!("no account matches '{selector}'"),
        _ => bail!(
            "'{selector}' matches {} accounts; use more characters",
            matches.len()
        ),
    }
}

fn print_accounts(storage: &AccountStorage) {
    if storage.is_empty() {
        println!("No accounts configured.");
        println!(
            "Add one with: gravitygate account add-token <refresh-token> --email you@example.com"
        );
        return;
    }

    let now = gravitygate::accounts::account::now_ms();
    println!("{:<4} {:<28} {:<10} {:<11} DETAIL", "#", "ACCOUNT", "ID", "STATUS");

    for (index, account) in storage.accounts.iter().enumerate() {
        let (status, detail) = describe_status(account, now);
        println!(
            "{:<4} {:<28} {:<10} {:<11} {}",
            index,
            truncate(&account.label(), 27),
            &account.credential_id()[..8],
            status,
            detail
        );
    }
}

/// Summarise an account's state for the table.
///
/// Ordered by severity: a ban outranks a verification hold, which outranks a
/// rate limit. An operator scanning the table should see the worst problem.
fn describe_status(account: &Account, now: i64) -> (&'static str, String) {
    if !account.enabled {
        return ("disabled", String::new());
    }
    if account.account_ineligible {
        return (
            "ineligible",
            account
                .account_ineligible_reason
                .clone()
                .unwrap_or_else(|| "account reported as ineligible".into()),
        );
    }
    if account.verification_required {
        return (
            "verify",
            account
                .verification_url
                .clone()
                .unwrap_or_else(|| "verification required".into()),
        );
    }
    if let Some(until) = account.cooling_down_until.filter(|until| *until > now) {
        let reason = account
            .cooldown_reason
            .map(describe_cooldown)
            .unwrap_or("cooldown");
        return (
            "cooling",
            format!("{reason}, {} left", humanise_ms(until - now)),
        );
    }
    if let Some(reset) = account.next_reset_at(now) {
        let pools: Vec<&str> = account
            .rate_limit_reset_times
            .iter()
            .filter(|(_, value)| **value > now)
            .map(|(key, _)| key.as_str())
            .collect();
        return (
            "limited",
            format!("{}, {} left", pools.join("+"), humanise_ms(reset - now)),
        );
    }
    ("ready", String::new())
}

fn describe_cooldown(reason: CooldownReason) -> &'static str {
    match reason {
        CooldownReason::AuthFailure => "auth failure",
        CooldownReason::NetworkError => "network error",
        CooldownReason::ProjectError => "project error",
        CooldownReason::ValidationRequired => "verification",
    }
}

/// Render a millisecond duration compactly, e.g. `2h10m`, `45s`.
fn humanise_ms(ms: i64) -> String {
    let seconds = (ms / 1000).max(0);
    let (hours, minutes, secs) = (seconds / 3600, (seconds % 3600) / 60, seconds % 60);
    if hours > 0 {
        format!("{hours}h{minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m{secs:02}s")
    } else {
        format!("{secs}s")
    }
}

fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let mut out: String = text.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Arguments for [`account_login`], bundled to keep the signature readable.
struct LoginArgs {
    no_browser: bool,
    no_open: bool,
    timeout: Duration,
    email: Option<String>,
    project: Option<String>,
    /// Run a connectivity check after saving. Best effort: a saved account is
    /// not rolled back because the check failed.
    verify: bool,
}

async fn account_login(config: Config, store: AccountStore, args: LoginArgs) -> Result<()> {
    let oauth = OAuthClient::new().context("building the OAuth client")?;

    let options = LoginOptions {
        no_browser: args.no_browser,
        open_browser: !args.no_open,
        timeout: args.timeout,
        ..Default::default()
    };

    let no_browser = args.no_browser;
    let outcome = login(&oauth, options, |prompt| {
        println!("Open this URL and sign in with the Google account you want to add:");
        println!();
        println!("  {}", prompt.url);
        println!();
        if no_browser {
            println!("After approving, the browser will fail to load the redirect page.");
            println!("That is expected. Paste the full URL from the address bar below.");
        } else {
            println!("Waiting for the browser redirect...");
        }
        println!();
    })
    .await?;

    let mut account = Account::new(outcome.refresh_token);
    // An explicit --email wins; otherwise use whatever Google told us.
    account.email = args.email.or(outcome.email);
    account.project_id = args.project;

    let verification_target = account.clone();
    let saved = upsert_account(&store, account)?;
    report_upsert(&saved);

    if !args.verify {
        return Ok(());
    }

    println!();
    println!("Checking the account against the upstream...");

    let engine = Engine::new(config, store)?;
    match engine.prepare_account(&verification_target).await {
        Ok(prepared) => {
            println!("  tier    : {}", prepared.tier);
            if prepared.used_fallback_project {
                println!(
                    "  project : {} (fallback - no project was provisioned)",
                    prepared.project_id
                );
            } else {
                println!("  project : {}", prepared.project_id);
            }
            println!("  result  : ok");
            println!();
            println!("Ready. Try `gravitygate probe` for an end-to-end request.");
        }
        Err(error) => {
            // The credentials are already saved. A failed check is useful
            // information, not a reason to discard a working refresh token.
            println!("  result  : FAILED");
            println!("  error   : {error}");
            println!();
            println!("The account was saved anyway. Address the error above, then re-check with");
            println!("`gravitygate account verify {}`.", saved.short_id);
        }
    }

    Ok(())
}

async fn verify_accounts(
    config: Config,
    store: AccountStore,
    selector: Option<String>,
) -> Result<()> {
    let storage = store.snapshot();
    if storage.is_empty() {
        bail!("no accounts configured");
    }

    let indices: Vec<usize> = match &selector {
        Some(selector) => vec![resolve_selector(&storage, selector)?],
        None => (0..storage.accounts.len()).collect(),
    };

    let path = store.path().to_path_buf();
    let engine = Engine::new(config, AccountStore::load(&path)?)?;
    let mut failures = 0;

    for index in indices {
        let account = &storage.accounts[index];
        println!("{} ({})", account.label(), &account.credential_id()[..8]);

        match engine.prepare_account(account).await {
            Ok(prepared) => {
                println!("  tier      : {}", prepared.tier);
                if prepared.used_fallback_project {
                    println!(
                        "  project   : {} (fallback — no project was provisioned)",
                        prepared.project_id
                    );
                } else {
                    println!("  project   : {}", prepared.project_id);
                }
                println!("  result    : ok");
            }
            Err(error) => {
                failures += 1;
                println!("  result    : FAILED");
                println!("  error     : {error}");
            }
        }
    }

    if failures > 0 {
        bail!("{failures} account(s) failed verification");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// probe
// ---------------------------------------------------------------------------

async fn cmd_probe(
    config_path: Option<PathBuf>,
    selector: Option<String>,
    model: String,
    prompt: String,
    raw: bool,
    with_tool: bool,
    repeat: u32,
) -> Result<()> {
    let config = load_config(config_path)?;
    let store = open_store()?;
    let storage = store.snapshot();

    if storage.is_empty() {
        bail!(
            "no accounts configured; add one with `gravitygate account add-token <refresh-token>`"
        );
    }

    let pinned = match &selector {
        Some(selector) => Some(storage.accounts[resolve_selector(&storage, selector)?].clone()),
        None => None,
    };

    let engine = Engine::new(config, store)?;

    println!("model     : {model}");
    match &pinned {
        Some(account) => println!(
            "requested : {} ({})",
            account.label(),
            &account.credential_id()[..8]
        ),
        // With no pin the router selects, so a rotation is observable.
        None => println!("requested : any account (router selects)"),
    }

    let repeat = repeat.max(1);

    // Repeating shares one engine, so the router's per-process state accumulates
    // across calls and rotation becomes observable.
    if repeat > 1 {
        return probe_repeated(&engine, pinned.as_ref(), &model, &prompt, with_tool, repeat).await;
    }

    let outcome = if with_tool {
        engine
            .probe_request(pinned.as_ref(), probe_request_with_tool(&model, &prompt))
            .await
    } else {
        engine.probe_request(pinned.as_ref(), probe_request(&model, &prompt)).await
    };

    let report = match outcome {
        Ok(report) => report,
        Err(error) => {
            println!("result    : FAILED");
            bail!("{error}");
        }
    };

    println!(
        "resolved  : {} (tier {}, {} chunks)",
        report.wire_model, report.thinking_tier, report.chunk_count
    );
    println!("served by : {} ({})", report.account, report.account_id);
    if report.upstream_attempts > 1 {
        println!(
            "retries   : {} upstream attempt(s) before success",
            report.upstream_attempts
        );
    }
    println!("tier      : {}", report.tier);
    if report.used_fallback_project {
        println!(
            "project   : {} (fallback — no project was provisioned for this account)",
            report.project_id
        );
    } else {
        println!("project   : {}", report.project_id);
    }

    for attempt in &report.attempts {
        let status = attempt
            .status
            .map(|s| s.to_string())
            .unwrap_or_else(|| "transport error".into());
        println!(
            "endpoint  : {} -> {} ({}ms)",
            attempt.endpoint,
            status,
            attempt.elapsed.as_millis()
        );
    }

    println!("traceId   : {}", report.trace_id);
    println!("sessionId : {}", report.session_id);

    if report.succeeded() {
        println!("result    : {}", report.status);

        match report.text() {
            Some(text) => println!("answer    : {}", text.trim()),
            None => println!("answer    : (no text content in response)"),
        }
        if let Some(reasoning) = report.reasoning() {
            println!("reasoning : {}", reasoning.trim());
        }
        if let Some(reason) = report.finish_reason() {
            // The upstream vocabulary is not the client-facing one: a tool call
            // reports STOP upstream and `tool_calls` downstream.
            println!("finish    : {reason} (upstream)");
        }

        if report.malformed_events > 0 {
            println!(
                "warning   : {} unparseable SSE line(s)",
                report.malformed_events
            );
        }

        let signatures = report.signatures();
        if signatures.is_empty() {
            println!("signatures: none");
        } else {
            println!("signatures: {} present", signatures.len());
            for signature in &signatures {
                let head = &signature[..12.min(signature.len())];
                println!("            {head}… ({} chars)", signature.len());
            }
        }
        // What the client would actually have received, assembled by the same
        // translator the gateway uses.
        if let Some(completion) = &report.completion {
            let message = &completion.choices[0].message;
            println!(
                "client    : finish_reason={}",
                completion.choices[0].finish_reason.as_deref().unwrap_or("none")
            );
            println!("            content={:?}", message.content);
            if let Some(reasoning) = &message.reasoning_content {
                println!("            reasoning={:?}", truncate(reasoning.trim(), 60));
            }
            if let Some(calls) = &message.tool_calls {
                for call in calls {
                    println!(
                        "            tool_call {} {}({})",
                        call.id, call.function.name, call.function.arguments
                    );
                }
            }
            if let Some(usage) = &completion.usage {
                println!(
                    "            usage prompt={} completion={} total={}",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
                );
            }
        }
        println!(
            "cached    : {} signature(s) captured this call",
            report.signatures_captured
        );
    } else {
        println!("result    : FAILED ({})", report.status);
    }

    println!();
    println!("--- raw response ---");
    println!("{}", report.body);
    if report.truncated && !raw {
        println!("(truncated; re-run with --raw for the full body)");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

fn cmd_config(config_path: Option<PathBuf>, command: ConfigCommand) -> Result<()> {
    match command {
        ConfigCommand::Path => {
            println!("config dir    : {}", config_dir().display());
            println!("config file   : {}", config_file_path(config_path).display());
            println!("accounts file : {}", accounts_path().display());
            Ok(())
        }

        ConfigCommand::Show => {
            let config = load_config(config_path)?;
            println!("{}", toml::to_string_pretty(&config)?);
            Ok(())
        }

        ConfigCommand::Init => {
            let path = config_file_path(config_path);
            if path.exists() {
                bail!("{} already exists; not overwriting", path.display());
            }
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            std::fs::write(&path, SAMPLE_CONFIG)
                .with_context(|| format!("writing {}", path.display()))?;
            println!("Wrote {}", path.display());
            Ok(())
        }
    }
}

/// Send the same request repeatedly, reporting where each one landed.
///
/// Deliberately terse: the point is the routing decision, and a full report per
/// call would bury it.
async fn probe_repeated(
    engine: &gravitygate::engine::Engine,
    pinned: Option<&gravitygate::accounts::account::Account>,
    model: &str,
    prompt: &str,
    with_tool: bool,
    repeat: u32,
) -> Result<()> {
    println!("strategy  : {}", engine.config.accounts.strategy);
    println!("repeating : {repeat} request(s) in one process");
    println!();
    println!("{:<4} {:<12} {:<10} OUTCOME", "#", "ACCOUNT", "ATTEMPTS");

    let mut served: Vec<String> = Vec::new();
    let mut failures = 0;

    for index in 1..=repeat {
        let request = if with_tool {
            probe_request_with_tool(model, prompt)
        } else {
            probe_request(model, prompt)
        };

        // Each iteration goes through the same engine, so the router sees a
        // fresh request against accumulated state.
        let outcome = engine.probe_request(pinned, request).await;

        match outcome {
            Ok(report) => {
                let answer = report.text().unwrap_or_default();
                println!(
                    "{:<4} {:<12} {:<10} {} {}",
                    index,
                    report.account_id,
                    report.upstream_attempts,
                    report.status,
                    truncate(answer.trim(), 40)
                );
                served.push(report.account_id.clone());
            }
            Err(error) => {
                failures += 1;
                println!("{:<4} {:<12} {:<10} FAILED: {error}", index, "-", "-");
            }
        }
    }

    let distinct: std::collections::BTreeSet<&String> = served.iter().collect();
    println!();
    println!("distinct accounts used : {}", distinct.len());
    if failures > 0 {
        bail!("{failures} of {repeat} request(s) failed");
    }
    Ok(())
}

/// Build a plain probe request.
fn probe_request(model: &str, prompt: &str) -> gravitygate::transform::openai::ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": model,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": 1024,
    }))
    .expect("the probe request is well-formed")
}

/// Build a probe request carrying a dummy tool the model is required to call.
///
/// `tool_choice: "required"` is what makes this deterministic: the point is to
/// exercise the tool-call path, not to find out whether the model felt like
/// using a tool.
fn probe_request_with_tool(model: &str, prompt: &str) -> gravitygate::transform::openai::ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": model,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": 1024,
        "tool_choice": "required",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string", "description": "City name" }
                    },
                    "required": ["city"],
                    "additionalProperties": false
                }
            }
        }]
    }))
    .expect("the probe tool request is well-formed")
}

fn config_file_path(explicit: Option<PathBuf>) -> PathBuf {
    explicit.unwrap_or_else(|| config_dir().join("config.toml"))
}

const SAMPLE_CONFIG: &str = r#"# GravityGate configuration.
#
# Every value below is the default, so this file can be deleted and the gateway
# will behave identically.

[server]
host = "127.0.0.1"
port = 8080
# Accepted client credentials. Empty disables authentication, which is the
# sensible default for a loopback-only gateway.
# api_keys = ["sk-your-key"]

[upstream]
# Tried in order. The captured CLI uses daily first.
endpoints = [
  "https://daily-cloudcode-pa.googleapis.com",
  "https://cloudcode-pa.googleapis.com",
]
request_jitter_max_ms = 0

[accounts]
# hybrid | sticky | round-robin | least-recently-used
strategy = "hybrid"
cooldown_secs = 60
max_consecutive_failures = 3
soft_quota_threshold = 0.20
quota_refresh_secs = 1800
token_bucket_max = 50.0
token_bucket_refill_per_min = 6.0

[routing]
max_wait_before_error_secs = 120
max_account_attempts = 10
max_capacity_retries = 5
max_empty_response_retries = 2
# too-many-requests returns 429 with Retry-After (OpenAI semantics).
# bad-request returns 400 invalid_request_error, which stops agentic clients
# from retrying a condition that will not clear.
exhausted_error_mode = "too-many-requests"

[reasoning]
# reasoning-content | reasoning | both
output_field = "reasoning-content"
default_tier = "medium"

[logging]
level = "info"

[metrics]
# Serve Prometheus metrics at /metrics.
enabled = true

[audit]
# Record every request to a SQLite file, for /api/stats and the dashboard.
enabled = true
# Defaults to audit.db in the config directory.
# path = "/var/lib/gravitygate/audit.db"
# Records older than this are pruned at startup and daily.
retention_days = 30
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn humanise_formats_each_magnitude() {
        assert_eq!(humanise_ms(0), "0s");
        assert_eq!(humanise_ms(45_000), "45s");
        assert_eq!(humanise_ms(90_000), "1m30s");
        assert_eq!(humanise_ms(3_600_000), "1h00m");
        assert_eq!(humanise_ms(7_800_000), "2h10m");
    }

    #[test]
    fn humanise_clamps_negative_durations() {
        // A reset time slightly in the past is possible under clock skew.
        assert_eq!(humanise_ms(-5_000), "0s");
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        // Multi-byte characters must not be split.
        assert_eq!(truncate("日本語のテキスト", 4), "日本語…");
        assert_eq!(truncate("short", 10), "short");
    }

    #[test]
    fn selector_resolves_by_index_email_and_prefix() {
        let mut first = Account::new("token-one");
        first.email = Some("a@example.com".into());
        let storage = AccountStorage {
            accounts: vec![first, Account::new("token-two")],
            ..Default::default()
        };

        assert_eq!(resolve_selector(&storage, "0").unwrap(), 0);
        assert_eq!(resolve_selector(&storage, "1").unwrap(), 1);
        assert_eq!(resolve_selector(&storage, "a@example.com").unwrap(), 0);

        let prefix = &storage.accounts[1].credential_id()[..6];
        assert_eq!(resolve_selector(&storage, prefix).unwrap(), 1);
    }

    #[test]
    fn selector_rejects_out_of_range_index() {
        let mut storage = AccountStorage::default();
        storage.accounts.push(Account::new("t"));
        assert!(resolve_selector(&storage, "5").is_err());
    }

    #[test]
    fn selector_reports_unknown_and_empty_pool() {
        let empty = AccountStorage::default();
        assert!(resolve_selector(&empty, "anything").is_err());

        let mut storage = AccountStorage::default();
        storage.accounts.push(Account::new("t"));
        let error = resolve_selector(&storage, "nobody@example.com").unwrap_err();
        assert!(error.to_string().contains("no account matches"));
    }

    #[test]
    fn duplicate_credentials_are_ambiguous_by_prefix() {
        let mut storage = AccountStorage::default();
        storage.accounts.push(Account::new("t"));
        storage.accounts.push(Account::new("t"));
        // Identical tokens produce identical credential ids, so no prefix
        // shorter than the whole id can disambiguate them.
        assert!(resolve_selector(&storage, "abc").is_err());
    }

    #[test]
    fn status_prefers_the_most_severe_condition() {
        let now = gravitygate::accounts::account::now_ms();

        let mut account = Account::new("t");
        assert_eq!(describe_status(&account, now).0, "ready");

        account.mark_rate_limited("claude", now + 60_000);
        assert_eq!(describe_status(&account, now).0, "limited");

        account.mark_cooling_down(now + 60_000, CooldownReason::NetworkError);
        assert_eq!(describe_status(&account, now).0, "cooling");

        account.mark_verification_required(Some("u".into()), "r");
        assert_eq!(describe_status(&account, now).0, "verify");

        account.mark_ineligible("banned");
        assert_eq!(describe_status(&account, now).0, "ineligible");

        account.enabled = false;
        assert_eq!(describe_status(&account, now).0, "disabled");
    }

    #[test]
    fn status_detail_names_the_limited_pools() {
        let now = gravitygate::accounts::account::now_ms();
        let mut account = Account::new("t");
        account.mark_rate_limited("claude", now + 120_000);
        let (_, detail) = describe_status(&account, now);
        assert!(detail.contains("claude"), "got: {detail}");
        assert!(detail.contains("2m00s"), "got: {detail}");
    }

    #[test]
    fn sample_config_parses_and_matches_defaults() {
        // The sample must stay loadable, or `config init` produces a broken file.
        let parsed: Config = toml::from_str(SAMPLE_CONFIG).expect("sample config must parse");
        let defaults = Config::default();
        assert_eq!(parsed.server.port, defaults.server.port);
        assert_eq!(parsed.accounts.strategy, defaults.accounts.strategy);
        assert_eq!(
            parsed.routing.exhausted_error_mode,
            defaults.routing.exhausted_error_mode
        );
        assert_eq!(parsed.reasoning.output_field, defaults.reasoning.output_field);
        assert_eq!(parsed.upstream.endpoints, defaults.upstream.endpoints);
    }
}
