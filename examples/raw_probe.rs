//! Send one generation request naming a model *verbatim*, with no tier
//! resolution.
//!
//! This is a diagnostic, not a feature. `/v1/chat/completions` resolves
//! `gemini-3.8-flash` to `gemini-3.8-flash-medium` before it reaches the wire,
//! which is correct but makes it impossible to ask the upstream directly whether
//! some other spelling exists or not. An older build of this gateway did not
//! resolve, and a 404 from it is only explainable by knowing what the upstream
//! does with a bare base name.
//!
//! ```text
//! cargo run --example raw_probe -- gemini-3.8-flash
//! cargo run --example raw_probe -- gemini-3.8-flash-medium
//! ```

use anyhow::{Context, Result, bail};
use gravitygate::accounts::store::AccountStore;
use gravitygate::config::{Config, accounts_path};
use gravitygate::engine::Engine;
use gravitygate::transform::ir::{Content, GenerateContentRequest, GenerationConfig, Part};
use gravitygate::upstream::constants;
use gravitygate::upstream::envelope::Envelope;
use gravitygate::upstream::transport::CallKind;

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::args()
        .nth(1)
        .context("usage: raw_probe <model>")?;

    let config = Config::load(None).context("loading configuration")?;
    let store = AccountStore::load(accounts_path())?;
    let snapshot = store.snapshot();
    let account = snapshot
        .accounts
        .iter()
        .find(|account| account.is_available(gravitygate::accounts::account::now_ms()))
        .cloned()
        .context("no available account")?;

    let engine = Engine::new(config.clone(), store)?;
    let prepared = engine.prepare_account(&account).await?;

    let request = GenerateContentRequest {
        contents: vec![Content::user(vec![Part::text("Say: ok")])],
        generation_config: Some(GenerationConfig {
            max_output_tokens: Some(512),
            ..Default::default()
        }),
        ..Default::default()
    };

    // The model string goes into the envelope untouched.
    let envelope = Envelope::build(
        request,
        &prepared.project_id,
        &model,
        "raw-probe",
        &engine.sessions,
    );
    let body = serde_json::to_vec(&envelope)?;

    let endpoint = config
        .upstream
        .endpoints
        .first()
        .context("no endpoint configured")?;
    let url = format!("{endpoint}{}?alt=sse", constants::API_STREAM_GENERATE);

    let response = engine
        .upstream
        .post_json(&url, &prepared.access_token, &body, CallKind::Buffered)
        .await?;

    let status = response.status;
    let mut text = String::new();
    {
        use futures::StreamExt;
        let mut stream = response.body;
        while let Some(chunk) = stream.next().await {
            let Ok(bytes) = chunk else { break };
            text.push_str(&String::from_utf8_lossy(&bytes));
        }
    }

    println!("model    : {model}");
    println!("endpoint : {endpoint}");
    println!("status   : {status}");
    if !status.is_success() {
        println!("body     : {}", text.trim().chars().take(300).collect::<String>());
        bail!("the upstream rejected this model name");
    }
    println!("first    : {}", text.lines().next().unwrap_or("").chars().take(160).collect::<String>());
    Ok(())
}
