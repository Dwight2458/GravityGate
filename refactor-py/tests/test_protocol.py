import json
from pathlib import Path

import pytest

from gravitygate_py.cache import SENTINEL, SignatureCache
from gravitygate_py.config import Config
from gravitygate_py.registry import resolve
from gravitygate_py.request import responses_to_chat, to_ir
from gravitygate_py.response import Accumulator, ResponsesStream
from gravitygate_py.schema import sanitize
from gravitygate_py.sse import SSEDecoder, StreamError
from gravitygate_py.upstream import Sessions, fnv1a64_signed


def test_offline_rust_contracts():
    contracts = json.loads(
        (Path(__file__).parent / "fixtures/rust-contracts.json").read_text(encoding="utf-8")
    )
    assert len(contracts) >= 100
    mismatches = []
    for contract in contracts:
        case, expected = contract["case"], contract["expected"]
        if case["kind"] == "schema":
            actual = {"schema": sanitize(case["input"])}
        else:
            config = Config()
            config.upstream.inject_agent_system_prompt = case.get("inject_system", False)
            try:
                request = (
                    responses_to_chat(case["input"])
                    if case["kind"] == "responses"
                    else case["input"]
                )
                model = resolve(request)
                actual = {
                    "wire_model": model.wire_model,
                    "tier": model.tier,
                    "budget": model.budget,
                    "quota_key": model.quota_key,
                    "thinking": model.thinking,
                    "ir": to_ir(request, model, config, SignatureCache()),
                }
            except ValueError:
                actual = {"error": "present"}
        if "error" in expected:
            assert "error" in actual
        elif actual != expected:
            mismatches.append({"case": case, "expected": expected, "actual": actual})
    assert not mismatches, json.dumps(mismatches, ensure_ascii=False, indent=2)


@pytest.mark.parametrize("split", range(1, 14))
def test_sse_incremental_unicode_and_crlf(split):
    source = 'data: {"text":"中文🙂"}\r\n\r\ndata: {"end":true}\r\n\r\n'.encode()
    decoder, events = SSEDecoder(), []
    for offset in range(0, len(source), split):
        events += decoder.feed(source[offset : offset + split])
    events += decoder.feed(b"", final=True)
    assert events == [{"text": "中文🙂"}, {"end": True}]


def test_sse_multiline_and_failures():
    assert SSEDecoder().feed(b': comment\ndata: {\ndata: "x":1}\n\n') == [{"x": 1}]
    with pytest.raises(StreamError):
        SSEDecoder().feed(b"data: {bad}\n\n")
    with pytest.raises(StreamError):
        SSEDecoder(maximum=10).feed(b"data: " + b"x" * 20)
    with pytest.raises(StreamError):
        SSEDecoder().feed(b"data: \xff\n\n")
    with pytest.raises(StreamError):
        SSEDecoder().feed(b"data: [DONE]\n\ndata: {}\n\n")


def test_signature_only_events_and_usage():
    cache = SignatureCache()
    acc = Accumulator("test", "gemini", "conversation", cache)
    assert (
        acc.absorb(
            {
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": "", "thoughtSignature": "s" * 80}]}}
                    ]
                }
            }
        )
        == []
    )
    assert cache.get("thinking", "conversation", "gemini") == "s" * 80
    acc.absorb(
        {
            "usageMetadata": {
                "promptTokenCount": 100,
                "cachedContentTokenCount": 20,
                "thoughtsTokenCount": 30,
                "totalTokenCount": 140,
            },
            "candidates": [{"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}],
        }
    )
    assert acc.usage["prompt_tokens"] == 80
    assert acc.usage["completion_tokens"] == 40
    assert acc.usage["total_tokens"] == 120
    assert acc.final_chunks(True)[-1]["choices"] == []
    assert acc.final_chunks(True) == []


def test_tool_signatures_roundtrip_without_cross_family_replay():
    cache = SignatureCache()
    acc = Accumulator("test", "gemini", "conversation", cache)
    acc.absorb(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {"name": "echo", "args": {"x": 1}},
                                "thoughtSignature": "s" * 80,
                            }
                        ]
                    }
                }
            ]
        }
    )
    call = acc.tools[0]
    request = {
        "model": "gemini-3.8-flash",
        "messages": [
            {"role": "assistant", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "done"},
        ],
    }
    ir = to_ir(request, resolve(request), Config(), cache)
    assert ir["contents"][0]["parts"][0]["thoughtSignature"] == "s" * 80
    assert ir["contents"][1]["parts"][0]["functionResponse"]["name"] == "echo"
    request["model"] = "claude-sonnet-4-6"
    ir = to_ir(request, resolve(request), Config(), cache)
    assert "thoughtSignature" not in ir["contents"][0]["parts"][0]
    assert ir["contents"][0]["parts"][0]["functionCall"]["id"] == call["id"]


def test_signature_cache_bounds_and_sentinel():
    cache = SignatureCache(maximum=2)
    for i in range(3):
        cache.put("tool", str(i), "s" * 80, "gemini")
    assert len(cache) == 2
    assert cache.get("tool", "0", "gemini") is None
    cache.put("tool", "sentinel", SENTINEL, "gemini")
    assert cache.get("tool", "sentinel", "gemini") is None


def test_responses_lifecycle_sequence_and_snapshots():
    acc = Accumulator("test", "gemini", "session", SignatureCache())
    stream = ResponsesStream(acc)
    events = stream.start()
    assert events[0]["response"]["usage"] is None
    assert events[0]["response"]["output"] == []
    assert stream.start() == []
    events += stream.on_deltas(
        [
            {"reasoning_content": "think"},
            {"content": "hi"},
            {"tool_calls": [{"id": "call_1", "function": {"name": "echo", "arguments": "{}"}}]},
        ]
    )
    acc.finish_reason = "STOP"
    events += stream.finish()
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    assert events[-1]["type"] == "response.completed"
    assert [i["type"] for i in events[-1]["response"]["output"]] == [
        "reasoning",
        "message",
        "function_call",
    ]
    added = next(
        e
        for e in events
        if e["type"] == "response.output_item.added" and e["item"]["type"] == "message"
    )
    assert added["item"]["content"] == []
    assert stream.finish() == []


def test_metadata_order_session_stability_and_unique_retries():
    sessions = Sessions()
    ir = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}], "generationConfig": {}}
    first = sessions.envelope(ir, "project", "gemini-3.8-flash-high", "conversation")
    second = sessions.envelope(ir, "project", "gemini-3.8-flash-high", "conversation")
    assert list(first) == ["project", "requestId", "request", "model", "userAgent", "requestType"]
    assert list(first["request"]) == ["contents", "labels", "generationConfig", "sessionId"]
    assert first["requestId"] != second["requestId"]
    assert (
        first["request"]["sessionId"]
        == second["request"]["sessionId"]
        == str(fnv1a64_signed("conversation"))
    )
    assert first["request"]["labels"]["model_enum"] == "MODEL_PLACEHOLDER_M318"
    sessions.completed("conversation")
    third = sessions.envelope(ir, "project", "claude-sonnet-4-6", "conversation")
    assert third["request"]["labels"]["used_claude"] == "true"
    assert "last_execution_id" in third["request"]["labels"]
