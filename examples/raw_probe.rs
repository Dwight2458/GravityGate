//! Send one generation request naming a model *verbatim*, with no tier
//! resolution, and summarise exactly which kinds of part come back.
//!
//! This is a diagnostic, not a feature. The normal routes resolve model names and
//! translate the response, which is correct but hides the upstream's own shape.
//! Two questions it exists to answer:
//!
//! - **Whether the upstream recognises a given model spelling.**
//!   `/v1/chat/completions` resolves `gemini-3.8-flash` to
//!   `gemini-3.8-flash-medium` before it reaches the wire, so it can never ask
//!   whether the bare base name exists. An older build of this gateway sent the
//!   base name and got a 404 that looked like an upstream fault.
//! - **Whether a given `thinkingConfig` produces thinking *text*.** The upstream
//!   reports `thoughtsTokenCount` either way, so a token count is no evidence
//!   that the reasoning itself is being returned. Measured 2026-09-23: a
//!   positive `thinkingBudget` on a `gemini-3.8-flash-high` wire name never
//!   returned text, while `-1` on the same model did — and on `-medium` a
//!   trivial prompt returned none where a substantial one returned ~2000
//!   characters. Text is intermittent, so one sample proves little in either
//!   direction; the full table is in `docs/progress.md`.
//!
//! ```text
//! cargo run --example raw_probe -- gemini-3.8-flash
//! cargo run --example raw_probe -- gemini-3.8-flash-high '{"includeThoughts":true,"thinkingLevel":"medium"}'
//! ```

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use gravitygate::accounts::store::AccountStore;
use gravitygate::config::{Config, accounts_path};
use gravitygate::engine::Engine;
use gravitygate::transform::ir::{Content, GenerateContentRequest, GenerationConfig, Part};
use gravitygate::upstream::constants;
use gravitygate::upstream::envelope::Envelope;
use gravitygate::upstream::transport::CallKind;
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .context("usage: raw_probe <model> [thinkingConfig json] [prompt]")?;
    let thinking: Option<Value> = match args.next() {
        Some(text) => Some(serde_json::from_str(&text).context("parsing thinkingConfig")?),
        None => None,
    };
    let prompt = args.next().unwrap_or_else(|| {
        "How many times does the letter r appear in strawberry?".into()
    });

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
        contents: vec![Content::user(vec![Part::text(prompt)])],
        generation_config: Some(GenerationConfig {
            max_output_tokens: Some(2048),
            thinking_config: thinking.clone(),
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
        let mut stream = response.body;
        while let Some(chunk) = stream.next().await {
            let Ok(bytes) = chunk else { break };
            text.push_str(&String::from_utf8_lossy(&bytes));
        }
    }

    println!("model    : {model}");
    println!(
        "thinking : {}",
        thinking
            .as_ref()
            .map_or_else(|| "(none configured)".to_string(), Value::to_string)
    );
    println!("status   : {status}");

    if !status.is_success() {
        println!(
            "body     : {}",
            text.trim().chars().take(400).collect::<String>()
        );
        bail!("the upstream rejected this request");
    }

    // Summarise rather than dump: the question is which kinds of part came back,
    // and a signature is hundreds of characters of noise.
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    let (mut thinking_chars, mut answer_chars, mut signatures) = (0usize, 0usize, 0usize);
    let mut thoughts_tokens = 0i64;

    for line in text.lines() {
        let Some(payload) = line.strip_prefix("data:") else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(payload.trim()) else {
            continue;
        };
        let Some(response) = value.get("response") else {
            continue;
        };

        if let Some(count) = response
            .pointer("/usageMetadata/thoughtsTokenCount")
            .and_then(Value::as_i64)
        {
            thoughts_tokens = count.max(thoughts_tokens);
        }

        let Some(parts) = response
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
        else {
            continue;
        };

        for part in parts {
            let thought = part.get("thought").and_then(Value::as_bool) == Some(true);
            let content = part.get("text").and_then(Value::as_str).unwrap_or("");
            if part.get("thoughtSignature").is_some() {
                signatures += 1;
            }
            let kind = match (thought, content.is_empty()) {
                (true, false) => {
                    thinking_chars += content.len();
                    "thought text"
                }
                (true, true) => "thought, empty",
                (false, false) => {
                    answer_chars += content.len();
                    "answer text"
                }
                (false, true) => "empty",
            };
            *counts.entry(kind).or_insert(0) += 1;
        }
    }

    println!("parts    : {counts:?}");
    println!("thinking : {thinking_chars} chars, thoughtsTokenCount={thoughts_tokens}");
    println!("answer   : {answer_chars} chars");
    println!("signature: {signatures} present");
    Ok(())
}
