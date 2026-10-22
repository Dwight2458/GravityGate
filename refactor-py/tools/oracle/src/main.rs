//! Offline contract oracle; never constructs a transport or reads account files.
use gravitygate::{
    config::Config,
    transform::{openai::ChatCompletionRequest, request, schema, signature_cache::SignatureCache},
};
use serde_json::{Value, json};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cases: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let results: Vec<Value> = cases
        .into_iter()
        .map(|case| {
            let result = if case["kind"] == "schema" {
                json!({"schema": schema::sanitize(&case["input"])})
            } else {
                let mut config = Config::default();
                config.upstream.inject_agent_system_prompt =
                    case["inject_system"].as_bool().unwrap_or(false);
                let input: ChatCompletionRequest = if case["kind"] == "responses" {
                    let response = serde_json::from_value(case["input"].clone()).unwrap();
                    match gravitygate::transform::responses::to_chat_request(&response) {
                        Ok(chat) => chat,
                        Err(error) => return json!({"case": case, "expected": {"error": error}}),
                    }
                } else {
                    serde_json::from_value(case["input"].clone()).unwrap()
                };
                match request::resolve_for(&input, &config) {
                    Err(e) => json!({"error": e.to_string()}),
                    Ok(model) => {
                        let ir = request::to_ir_with_signatures(
                            &input,
                            &model,
                            &config,
                            &SignatureCache::new(),
                        );
                        match ir {
                            Ok((ir, _)) => {
                                json!({"wire_model": model.wire_model, "tier": model.tier.as_str(),
                            "budget": model.thinking_budget, "quota_key": model.quota_key(),
                            "thinking": model.thinking_enabled, "ir": ir})
                            }
                            Err(e) => json!({"error": e.to_string()}),
                        }
                    }
                }
            };
            json!({"case": case, "expected": result})
        })
        .collect();
    std::fs::write(&args[2], serde_json::to_string_pretty(&results).unwrap()).unwrap();
    eprintln!("Exported {} offline Rust contract cases", results.len());
}
