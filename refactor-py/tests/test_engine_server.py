import json

import httpx
import pytest
from conftest import Reply, frame, payload

from gravitygate_py.errors import GatewayError, classify
from gravitygate_py.registry import resolve
from gravitygate_py.storage import now_ms


async def test_buffered_chat_and_wire_identity(system, api, request_body):
    _, _, database, fake = system
    database.add("one", "one@test")
    response = await api.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "ok"
    assert body["usage"]["completion_tokens"] == 98
    token, envelope, request = fake.requests[0]
    assert token == "one"
    assert envelope["model"] == "gemini-3.8-flash-high"
    assert request.headers["transfer-encoding"] == "chunked"
    assert "content-length" not in request.headers
    assert request.headers["accept-encoding"] == "gzip"
    names = [k.lower() for k, _ in request.headers.multi_items()]
    assert (
        names.index("user-agent")
        < names.index("authorization")
        < names.index("content-type")
        < names.index("accept-encoding")
    )
    await system[0].state.observ.flush()
    stats = (await api.get("/api/stats")).json()
    assert stats["requests"] == stats["ok"] == 1
    assert stats["reasoning_tokens"] == 97
    records = (await api.get("/api/requests")).json()["requests"]
    assert records[0]["account_id"] == database.get("0").id[:8]
    assert "hello" not in json.dumps(records)
    metrics = (await api.get("/metrics")).text
    assert 'gravitygate_requests_total{model="gemini-3.8-flash",outcome="ok"} 1.0' in metrics
    assert 'gravitygate_tokens_total{kind="reasoning",model="gemini-3.8-flash"} 97.0' in metrics
    assert "gravitygate_upstream_attempts_count" in metrics
    assert "gravitygate_request_duration_seconds_bucket" in metrics
    assert "gravitygate_account_selected_total" in metrics
    assert "gravitygate_signatures_cached" in metrics


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses"])
async def test_streaming_protocols(system, api, request_body, route):
    _, _, database, fake = system
    database.add("one")
    request = {**request_body, "stream": True, "stream_options": {"include_usage": True}}
    if route.endswith("responses"):
        request = {"model": request_body["model"], "input": "hello", "stream": True}
    fake.replies["one"].append(
        Reply(
            chunks=[frame(payload("thinking", finish=None, thought=True)), frame(payload("answer"))]
        )
    )
    response = await api.post(route, json=request)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    if route.endswith("responses"):
        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] == "response.completed"
        assert events[-1]["response"]["output"][1]["content"][0]["text"] == "answer"
        assert [e["sequence_number"] for e in events] == list(range(len(events)))
    else:
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"
        assert events[-1]["choices"] == []
        assert any(
            "reasoning_content" in c.get("delta", {}) for e in events for c in e.get("choices", [])
        )
        assert "data: [DONE]" in response.text
    assert fake.bodies[-1].closed


async def test_buffered_responses_tool_and_replay(system, api):
    _, _, database, fake = system
    database.add("one")
    fake.replies["one"].append(
        Reply(
            data={
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {"name": "echo", "args": {"x": 1}},
                                        "thoughtSignature": "s" * 100,
                                    }
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ]
                }
            }
        )
    )
    first = await api.post(
        "/v1/responses", json={"model": "gemini-3.8-flash", "input": "call echo"}
    )
    assert first.status_code == 200, first.text
    call = first.json()["output"][0]
    assert call["type"] == "function_call"
    second = await api.post(
        "/v1/responses",
        json={
            "model": "gemini-3.8-flash",
            "input": [
                call,
                {"type": "function_call_output", "call_id": call["call_id"], "output": "done"},
            ],
        },
    )
    assert second.status_code == 200, second.text
    contents = fake.requests[-1][1]["request"]["contents"]
    assert contents[0]["parts"][0]["thoughtSignature"] == "s" * 100
    assert contents[1]["parts"][0]["functionResponse"]["name"] == "echo"


async def test_verification_rotates_to_ready_account(system, api, request_body):
    _, _, database, fake = system
    one = database.add("one", "one@test")
    database.add("two", "two@test")
    fake.replies["one"].append(
        Reply(
            status=403,
            data={
                "error": {
                    "message": "VALIDATION_REQUIRED https://accounts.google.com/signin/continue?x=1"
                }
            },
        )
    )
    response = await api.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 200, response.text
    assert [r[0] for r in fake.requests] == ["one", "two"]
    held = database.get(one.id)
    assert held.state["verification_required"]
    assert held.state["verification_url"].startswith("https://accounts.google.com/")
    health = (await api.get("/health")).json()
    assert health["status"] == "degraded"
    assert health["accounts_available"] == 1


async def test_quota_is_real_and_model_scoped(system, api, request_body):
    _, _, database, fake = system
    a = database.add("one")
    fake.replies["one"].append(
        Reply(
            status=429,
            data={
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [{"reason": "QUOTA_EXHAUSTED"}],
                }
            },
        )
    )
    result = await api.post("/v1/chat/completions", json=request_body)
    assert result.status_code == 429, result.text
    holds = database.get(a.id).state["holds"]
    quota = resolve(request_body).quota_key
    assert 58000 <= holds[quota] - now_ms() <= 61000
    assert "gemini" not in holds
    request_body["model"] = "gemini-3.1-pro"
    response = await api.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 200
    assert quota in database.get(a.id).state["holds"]
    assert (await api.get("/health")).status_code == 200


async def test_held_account_plus_limited_account_reports_quota(system, api, request_body):
    _, _, database, _ = system
    one, two = database.add("one"), database.add("two")
    database.update(
        one.id, lambda a: a.state.update(verification_required=True, recheck_at=now_ms() + 60000)
    )
    database.update(
        two.id, lambda a: a.state.update(holds={resolve(request_body).quota_key: now_ms() + 30000})
    )
    response = await api.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 429, response.text
    assert response.json()["error"]["code"] == "account_pool_rate_limited"
    assert "Retry-After" in response.headers


async def test_capacity_does_not_mark_account_quota_or_rotate(system, api, request_body):
    _, _, database, fake = system
    a = database.add("one")
    database.add("two")
    fake.replies["one"].extend(
        [
            Reply(status=503, data={"error": {"message": "model overloaded"}}),
            Reply(status=503, data={"error": {"message": "model overloaded"}}),
        ]
    )
    response = await api.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 503
    assert len(fake.requests) == 2
    assert all(r[0] == "one" for r in fake.requests)
    assert not database.get(a.id).state.get("holds")


@pytest.mark.parametrize("status,code", [(404, "model_not_found"), (400, "upstream_bad_request")])
async def test_invalid_model_request_does_not_penalize_or_rotate(
    system, api, request_body, status, code
):
    _, _, database, fake = system
    a = database.add("one")
    database.add("two")
    fake.replies["one"].append(Reply(status=status, data={"error": {"message": "not found"}}))
    response = await api.post("/v1/chat/completions", json=request_body)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert len(fake.requests) == 1
    assert not database.get(a.id).state.get("failures")


async def test_auth_refresh_single_flight_and_terminal_failure_rotation(system, api, request_body):
    _, _, database, fake = system
    a = database.add("bad")
    database.add("good")
    fake.invalid_tokens.add("bad")
    response = await api.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 200, response.text
    assert database.get(a.id).enabled is False
    fake.replies["good"].append(Reply(status=401, data={"error": {"message": "expired token"}}))
    response = await api.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 200
    assert fake.refreshes.count("good") == 2


async def test_transport_fallback_and_bounded_empty_retries(system, api, request_body):
    _, config, database, fake = system
    database.add("one")
    fake.replies["one"].append(httpx.ConnectError("synthetic"))
    result = await api.post("/v1/chat/completions", json=request_body)
    assert result.status_code == 200
    assert fake.requests[-1][2].url.host == "fake-two"
    config.routing.max_empty_response_retries = 1
    fake.replies["one"].extend([Reply(data={"response": {"candidates": []}}), Reply()])
    result = await api.post("/v1/chat/completions", json=request_body)
    assert result.status_code == 200
    assert all(body.closed for body in fake.bodies)


@pytest.mark.parametrize("stream", [False, True])
async def test_interrupted_stream_is_never_success(system, api, request_body, stream):
    _, _, database, fake = system
    database.add("one")
    fake.replies["one"].append(
        Reply(chunks=[frame(payload("partial", finish=None))], break_read=True)
    )
    request_body["stream"] = stream
    result = await api.post("/v1/chat/completions", json=request_body)
    if stream:
        assert result.status_code == 200
        assert '"error"' in result.text
        assert '"finish_reason":"stop"' not in result.text
    else:
        assert result.status_code == 502, result.text
    assert fake.bodies[-1].closed
    await system[0].state.observ.flush()
    assert (await api.get("/api/stats")).json()["errors"] == 1


@pytest.mark.parametrize("raw,status", [(b"not-json", 400), (b"[]", 422), (b"{}", 422)])
async def test_bad_json_envelope_and_no_account_selection(system, api, raw, status):
    result = await api.post(
        "/v1/chat/completions", content=raw, headers={"content-type": "application/json"}
    )
    assert result.status_code == status, result.text
    assert "error" in result.json()
    assert not system[3].requests


async def test_large_zcode_payload_below_16_mib_is_accepted(system, api, request_body):
    _, _, database, _ = system
    database.add("one")
    request_body["messages"][0]["content"] = "x" * 2_098_816
    response = await api.post("/v1/chat/completions", json=request_body)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses"])
async def test_body_limit_both_declared_and_streamed(system, api, route):
    _, config, database, fake = system
    database.add("one")
    config.server.max_request_body_bytes = 100
    declared = await api.post(
        route, content=b" " * 101, headers={"content-type": "application/json"}
    )

    async def chunks():
        yield b" " * 70
        yield b" " * 70

    streamed = await api.post(route, content=chunks(), headers={"content-type": "application/json"})
    for result in (declared, streamed):
        assert result.status_code == 413, result.text
        assert result.json()["error"]["code"] == "request_body_too_large"
    assert not fake.requests
    assert not fake.refreshes
    await system[0].state.observ.flush()
    assert all(r["account_id"] == "" for r in system[0].state.observ.recent())


async def test_auth_protects_admin_and_generation(system, api, request_body):
    _, config, database, _ = system
    database.add("one")
    # Middleware reads the same mutable key list; no app rebuild required for this test.
    config.server.api_keys.append("gateway-secret")
    for route in ("/health", "/metrics", "/api/requests", "/v1/models", "/"):
        assert (await api.get(route)).status_code == 401
    for route in ("/api/accounts/0/recheck", "/refresh-token", "/v1/chat/completions"):
        assert (await api.post(route, json=request_body)).status_code == 401
    assert (
        await api.get("/health", headers={"Authorization": "bEaReR gateway-secret"})
    ).status_code == 200
    assert (await api.get("/health", headers={"x-api-key": "gateway-secret"})).status_code == 200
    assert (await api.get("/", auth=("operator", "gateway-secret"))).status_code == 200


async def test_model_catalogue_metrics_and_dashboard(system, api):
    _, _, database, _ = system
    database.add("one")
    models = (await api.get("/v1/models")).json()["data"]
    ids = {row["id"] for row in models}
    assert "gemini-3.8-flash-high" in ids and "gemini-future" in ids
    assert "chat_internal" not in ids
    limits = (await api.get("/account-limits")).json()
    assert (
        limits["accounts"][0]["models"]["gemini-3.8-flash-high"]["quotaInfo"]["remainingFraction"]
        == 0.5
    )
    assert 'condition="checking"} 0' in (await api.get("/metrics")).text
    html = await api.get("/")
    assert "Recheck now" in html.text and "gravitygate-py account login" in html.text
    assert (await api.get("/not-found")).json()["error"]["code"] == "invalid_request_error"


@pytest.mark.parametrize(
    "status,body,code",
    [
        (
            429,
            {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Resource exhausted (e.g. check quota).",
                }
            },
            "rate_limit_unknown",
        ),
        (429, {"error": {"details": [{"reason": "RATE_LIMIT_EXCEEDED"}]}}, "rate_limit_exceeded"),
        (400, {"error": {"details": [{"reason": "QUOTA_EXHAUSTED"}]}}, "quota_exhausted"),
        (403, {"error": {"message": "ACCOUNT_INELIGIBLE"}}, "account_ineligible"),
        (503, {"error": {"message": "quota"}}, "model_capacity_exhausted"),
    ],
)
def test_error_classifier(status, body, code):
    assert classify(status, {}, json.dumps(body)).code == code


def test_retry_hints_precedence():
    error = classify(
        429,
        {"retry-after-ms": "15000", "retry-after": "60"},
        '{"error":{"details":[{"retryDelay":"120s"}]}}',
    )
    assert isinstance(error, GatewayError) and error.retry_after == 15
