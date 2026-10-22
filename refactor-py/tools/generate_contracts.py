"""Regenerate synthetic contracts from the original Rust implementation."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cases = []
models = [
    "gemini-3.1-pro",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking",
    "gpt-oss-120b-medium",
    "gemini-3.1-flash-image",
    "gemini-3.8-flash-tiered",
    "future-model-high",
    "ANTIGRAVITY-GEMINI-3.8-FLASH-HIGH",
]
for model in models:
    for signal in (
        {},
        {"reasoning_effort": "low"},
        {"reasoning_effort": "high"},
        {"thinking": {"budget_tokens": 20000}},
        {"thinking": {"type": "disabled"}},
        {"max_tokens": 64},
    ):
        cases.append(
            {
                "kind": "chat",
                "input": {
                    "model": model,
                    "messages": [{"role": "user", "content": "Hello 世界"}],
                    **signal,
                },
            }
        )
messages = [
    [
        {"role": "system", "content": "one"},
        {"role": "developer", "content": "two"},
        {"role": "user", "content": "question"},
    ],
    [
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "partial"},
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
            ],
        }
    ],
    [
        {
            "role": "assistant",
            "reasoning_content": "private",
            "content": "result",
            "tool_calls": [
                {
                    "id": "call_original",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"x":1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_original", "content": "output"},
    ],
]
for model in ("gemini-3.8-flash", "claude-sonnet-4-6"):
    for conversation in messages:
        cases.append(
            {
                "kind": "chat",
                "inject_system": True,
                "input": {"model": model, "messages": conversation},
            }
        )
    for choice in ("auto", "none", "required", {"type": "function", "function": {"name": "echo"}}):
        cases.append(
            {
                "kind": "chat",
                "input": {
                    "model": model,
                    "messages": [{"role": "user", "content": "use echo"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "echo", "parameters": {"type": "object"}},
                        }
                    ],
                    "tool_choice": choice,
                },
            }
        )
schemas = [
    {},
    True,
    {"type": "object"},
    {"type": ["string", "null"]},
    {
        "type": "object",
        "properties": {"x": {"type": "string", "format": "email"}},
        "required": ["x", "missing"],
    },
    {
        "anyOf": [
            {"type": "string"},
            {"type": "object", "properties": {"x": {"const": 3}}},
            {"type": "null"},
        ]
    },
    {
        "allOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"properties": {"b": {"type": "integer"}}},
        ]
    },
    {
        "type": "array",
        "items": [{"type": "string"}, {"type": "object"}],
        "minItems": 1,
        "uniqueItems": True,
    },
    {"$ref": "#/definitions/Person", "description": "A person"},
    {"type": ["array", "string", "object"], "properties": {"x": {}}, "additionalProperties": False},
]
cases += [{"kind": "schema", "input": schema} for schema in schemas]
cases += [
    {
        "kind": "responses",
        "input": {
            "model": "gemini-3.8-flash",
            "input": "hello",
            "instructions": "be brief",
            "max_output_tokens": 64,
        },
    },
    {
        "kind": "responses",
        "input": {
            "model": "gemini-3.8-flash",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_original",
                    "name": "echo",
                    "arguments": '{"x":1}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_original",
                    "output": {"result": 1},
                },
            ],
        },
    },
]
cache = ROOT / ".cache"
cache.mkdir(exist_ok=True)
inputs = cache / "contract-inputs.json"
inputs.write_text(json.dumps(cases), encoding="utf-8")
destination = ROOT / "tests" / "fixtures" / "rust-contracts.json"
destination.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(ROOT / "tools/oracle/Cargo.toml"),
        "--target-dir",
        str(cache / "rust-target"),
        "--",
        str(inputs),
        str(destination),
    ],
    check=True,
)
