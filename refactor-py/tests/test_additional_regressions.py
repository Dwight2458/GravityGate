import asyncio
import json
import threading

import httpx
import pytest
from conftest import Reply, payload

from gravitygate_py.config import Accounts
from gravitygate_py.engine import apply_failure
from gravitygate_py.errors import GatewayError
from gravitygate_py.oauth import FALLBACK_PROJECT
from gravitygate_py.registry import model_list
from gravitygate_py.storage import Database, now_ms


@pytest.mark.parametrize("strategy", ["hybrid", "sticky", "round-robin", "least-recently-used"])
def test_routing_strategy_and_rotation(tmp_path, strategy):
    database = Database(tmp_path / "db")
    first, second = database.add("one"), database.add("two")
    config = Accounts(strategy=strategy)
    quota = "gemini:test"
    a, _ = database.select(quota, config, set())
    b, _ = database.select(quota, config, set())
    if strategy == "sticky":
        assert a.id == b.id
    else:
        assert a.id != b.id
        if strategy == "hybrid":
            assert database.select(quota, config, set())[0].id == b.id
    database.update(first.id, lambda a: a.state.update(holds={quota: now_ms() + 60000}))
    assert database.select(quota, config, set())[0].id == second.id


def test_model_catalogue_matches_canonical_aliases():
    rows = {row["id"]: row for row in model_list()}
    assert "gemini-3.1-pro-low" not in rows
    assert "gemini-3.1-pro-high" in rows
    assert "claude-sonnet-4-6-high" not in rows
    assert rows["claude-sonnet-4-6"]["owned_by"] == "anthropic"
    assert rows["gpt-oss-120b-medium"]["owned_by"] == "openai"
    assert rows["gemini-3.8-flash-high"]["supports_tools"]


def test_quota_and_capacity_do_not_damage_health(tmp_path):
    from gravitygate_py.config import Config

    database = Database(tmp_path / "db")
    account = database.add("one")
    apply_failure(
        account, GatewayError(503, "model_capacity_exhausted", "busy"), "gemini:test", Config()
    )
    assert not account.state.get("failures")
    assert not account.state.get("health")
    apply_failure(account, GatewayError(429, "quota_exhausted", "quota"), "gemini:test", Config())
    assert not account.state.get("failures")
    assert not account.state.get("health")
    assert account.state["holds"]["gemini:test"] > now_ms()


async def test_monitor_runs_in_service_lifespan_and_stops_after_recovery(system, monkeypatch):
    app, config, database, fake = system
    a = database.add("one")
    database.update(a.id, lambda a: a.state.update(verification_required=True, recheck_at=0))
    config.verification.enabled = True
    config.verification.scan_interval_secs = 0.01
    recovered = asyncio.Event()
    original_scan = app.state.monitor.scan

    async def scan_and_signal():
        await original_scan()
        if not database.get(a.id).state.get("verification_required"):
            recovered.set()

    monkeypatch.setattr(app.state.monitor, "scan", scan_and_signal)
    async with app.router.lifespan_context(app):
        async with asyncio.timeout(3):
            await recovered.wait()
        assert app.state.engine.monitor_active
        await asyncio.sleep(0.04)
        assert len(fake.requests) == 1
    assert not app.state.engine.monitor_active


async def test_cancellation_during_sqlite_claim_cleans_lease(system, api, monkeypatch):
    app, _, database, fake = system
    a = database.add("one")
    database.update(a.id, lambda a: a.state.update(verification_required=True, recheck_at=0))
    entered, gate = threading.Event(), threading.Event()
    original = database.update

    def delayed(selector, change):
        if change.__name__ == "claim":
            entered.set()
            assert gate.wait(3)
        return original(selector, change)

    monkeypatch.setattr(database, "update", delayed)
    task = asyncio.create_task(app.state.monitor.check(a.id))
    assert await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not database.get(a.id).running(now_ms())
    assert database.get(a.id).state["last_result"] == "verification_cancelled"
    assert not fake.requests


async def test_malformed_upstream_usage_is_a_gateway_error(system, api, request_body):
    _, _, database, fake = system
    database.add("one")
    malformed = payload()
    malformed["response"]["usageMetadata"] = {"promptTokenCount": "bad"}
    fake.replies["one"].append(Reply(data=malformed))
    result = await api.post("/v1/chat/completions", json=request_body)
    assert result.status_code == 503, result.text
    assert result.json()["error"]["code"] == "account_pool_exhausted"


async def test_oauth_network_failure_returns_gateway_error(system, api, request_body, monkeypatch):
    app, _, database, _ = system
    database.add("one")

    async def broken_refresh(token):
        raise httpx.ConnectError("synthetic OAuth network failure")

    monkeypatch.setattr(app.state.engine.credentials.oauth, "refresh", broken_refresh)
    result = await api.post("/v1/chat/completions", json=request_body)
    assert result.status_code == 503, result.text
    assert result.json()["error"]["code"] == "account_pool_exhausted"


async def test_project_discovery_fallback_is_not_persisted_as_real_project(system, monkeypatch):
    app, _, database, fake = system
    account = database.add("one")

    async def refused(*args, **kwargs):
        raise GatewayError(403, "permission_denied", "not provisioned")

    monkeypatch.setattr(app.state.engine.transport, "json", refused)
    try:
        _, project = await app.state.engine.credentials.prepare(account)
        assert project == FALLBACK_PROJECT
        assert database.get(account.id).project_id is None
        assert fake.refreshes == ["one"]
    finally:
        await app.state.engine.close()


async def test_query_validation_uses_openai_error_shape(system, api):
    result = await api.get("/api/requests?limit=invalid")
    assert result.status_code == 422
    assert "error" in result.json()
    assert "input" not in json.dumps(result.json())
