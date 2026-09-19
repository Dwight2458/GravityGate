//! Command-line surface.
//!
//! The CLI is the primary way an operator configures accounts, so it is built
//! around one principle: every account operation reports the account's *state*,
//! not just success or failure. An operator needs to know whether an account is
//! rate-limited for five minutes, awaiting verification, or banned, because the
//! three call for completely different responses.

use std::path::PathBuf;

use clap::{ArgAction, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "gravitygate",
    version,
    about = "OpenAI-compatible gateway for Google Antigravity",
    long_about = "Exposes Google Antigravity (Cloud Code Assist) models through an OpenAI-compatible API.\n\n\
                  Note: using this software violates Google's Terms of Service. Accounts have been \
                  suspended for similar use. Do not configure an account you cannot afford to lose."
)]
pub struct Cli {
    /// Path to the config file. Defaults to `config.toml` in the config directory.
    #[arg(long, global = true, env = "GRAVITYGATE_CONFIG_FILE", value_name = "PATH")]
    pub config: Option<PathBuf>,

    /// Increase log verbosity. Repeat for more detail.
    #[arg(short, long, global = true, action = ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the gateway.
    Serve {
        /// Override the configured listen host.
        #[arg(long)]
        host: Option<String>,
        /// Override the configured listen port.
        #[arg(long)]
        port: Option<u16>,
    },

    /// Manage the account pool.
    Account {
        #[command(subcommand)]
        command: AccountCommand,
    },

    /// Send one request to the upstream and report exactly what happened.
    ///
    /// Use this to confirm an account works before putting the gateway in front
    /// of a client.
    Probe {
        /// Wire model to use.
        #[arg(long, default_value = "gemini-3.8-flash-medium")]
        model: String,
        /// Account selector: email, credential id prefix, or index.
        #[arg(long)]
        account: Option<String>,
        /// Prompt to send.
        #[arg(long, default_value = "Reply with exactly: ok")]
        prompt: String,
        /// Print the full response body without truncation.
        #[arg(long)]
        raw: bool,
        /// Send the request this many times in one process.
        ///
        /// Routing state — the sticky cursor, token buckets, health scores — is
        /// per-process, so a single invocation can only ever show one selection.
        /// Repeating is what makes rotation observable.
        #[arg(long, default_value_t = 1)]
        repeat: u32,
        /// Send a dummy tool and require the model to call it.
        ///
        /// Exercises the path the signature cache exists for: the upstream
        /// attaches a signature to the function call, and a follow-up turn has
        /// to send it back.
        #[arg(long)]
        tool: bool,
    },

    /// Inspect configuration.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum AccountCommand {
    /// Add an account by signing in with a browser.
    ///
    /// Runs the Google authorization-code flow with PKCE against a loopback
    /// listener, then stores the resulting refresh token.
    Login {
        /// Print the URL and read the authorization code from the terminal
        /// instead of running a callback listener.
        ///
        /// Use this over SSH or in a container, where the browser cannot reach
        /// the gateway's loopback interface.
        #[arg(long)]
        no_browser: bool,

        /// Do not try to open a browser automatically.
        ///
        /// The URL is printed either way.
        #[arg(long)]
        no_open: bool,

        /// Seconds to wait for the browser redirect.
        #[arg(long, default_value_t = 300, value_name = "SECONDS")]
        timeout: u64,

        /// Label for the account. Defaults to the Google account's email.
        #[arg(long)]
        email: Option<String>,

        /// Project id to record on the account.
        #[arg(long)]
        project: Option<String>,

        /// Skip the connectivity check that normally runs after signing in.
        #[arg(long)]
        no_verify: bool,
    },

    /// Add an account from a refresh token.
    ///
    /// Accepts either a bare refresh token or the packed
    /// `refresh|project|managedProject` form.
    AddToken {
        /// The refresh token.
        #[arg(value_name = "TOKEN")]
        token: String,
        /// Label for the account, normally the Google account email.
        #[arg(long)]
        email: Option<String>,
        /// Project id to use, when not embedded in the token.
        #[arg(long)]
        project: Option<String>,
    },

    /// List accounts and their current state.
    List {
        /// Emit JSON instead of a table.
        #[arg(long)]
        json: bool,
    },

    /// Remove an account.
    Remove {
        /// Email, credential id prefix, or index.
        selector: String,
    },

    /// Enable a previously disabled account.
    Enable {
        /// Email, credential id prefix, or index.
        selector: String,
    },

    /// Disable an account without removing it.
    Disable {
        /// Email, credential id prefix, or index.
        selector: String,
    },

    /// Clear rate limits, cooldowns, and verification holds.
    ///
    /// Use after resolving a verification prompt in a browser.
    ClearHolds {
        /// Email, credential id prefix, or index.
        selector: String,
    },

    /// Check that an account authenticates and can resolve a project.
    Verify {
        /// Email, credential id prefix, or index. Defaults to every account.
        selector: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
pub enum ConfigCommand {
    /// Print the config and account file locations.
    Path,
    /// Print the effective configuration.
    Show,
    /// Write an example config file if one does not exist.
    Init,
}
