import asyncio

import pytest
from conftest import Reply, frame, payload

from gravitygate_py.engine import Engine
from gravitygate_py.errors import GatewayError
from gravitygate_py.registry import resolve
from gravitygate_py.storage import Busy, NotFound, now_ms
from gravitygate_py.verification import VerificationMonitor


def hold(database, account):
    database.update(
        account.id,
        lambda a: a.state.update(
            verification_required=True,
            verification_url="https://accounts.google.com/old",
            verification_reason="challenge",
            recheck_at=now_ms() - 1,
            revision=1,
        ),
    )


async def test_automatic_recovery_requires_generation_and_stops_polling(system, api):
    app, _, database, fake = system
    a = database.add("one")
    hold(database, a)
    database.update(
        a.id,
        lambda a: a.state.update(health=1, holds={"claude:claude-sonnet-4-6": now_ms() + 60000}),
    )
    await app.state.monitor.scan()
    current = database.get(a.id)
    assert not current.state["verification_required"]
    assert current.state["verification_url"] is None
    assert current.state["last_result"] == "recovered"
    assert current.state["health"] == 100
    assert current.state["holds"]["claude:claude-sonnet-4-6"] > now_ms()
    assert current.verification()["recovered_at"]
    await app.state.monitor.scan()
    assert len(fake.requests) == 1


@pytest.mark.parametrize(
    "reply",
    [
        Reply(data={"response": {"candidates": []}}),
        Reply(data=payload("", finish="STOP")),
        Reply(data=payload("  \n ", finish="STOP")),
        Reply(data=payload("only thought", thought=True)),
        Reply(data=payload("answer", finish="MAX_TOKENS")),
        Reply(data=payload("answer", finish=None)),
        Reply(chunks=[b"data: {bad}\n\n"]),
        Reply(chunks=[frame(payload("partial", finish=None))], break_read=True),
        Reply(data={"error": {"message": "stream error"}}),
    ],
)
async def test_no_false_recovery(system, api, reply):
    app, _, database, fake = system
    a = database.add("one")
    database.add("two")
    hold(database, a)
    fake.replies["one"].append(reply)
    view = await app.state.monitor.check(a.id)
    assert view["state"] == "waiting", view
    assert view["last_result"] != "recovered"
    assert database.get(a.id).state["verification_required"]
    assert [r[0] for r in fake.requests] == ["one"]
    assert all(body.closed for body in fake.bodies)


async def test_renewed_challenge_backoff_and_latest_url(system, api):
    app, _, database, fake = system
    a = database.add("one")
    hold(database, a)
    for count, expected in enumerate((60, 120, 300, 300), start=1):
        database.update(a.id, lambda a: a.state.update(check_started_at=0, recheck_at=0))
        fake.replies["one"].append(
            Reply(
                status=403,
                data={
                    "error": {
                        "message": f"VALIDATION_REQUIRED https://accounts.google.com/new?attempt={count}"
                    }
                },
            )
        )
        view = await app.state.monitor.check(a.id)
        assert expected * 1000 - 1000 <= view["next_check_at"] - now_ms() <= expected * 1000 + 1000
        assert database.get(a.id).state["verification_url"].endswith(f"attempt={count}")


async def test_quota_hint_extends_backoff_and_blocks_manual_bypass(system, api):
    app, _, database, fake = system
    a = database.add("one")
    hold(database, a)
    fake.replies["one"].append(
        Reply(
            status=429,
            headers={"retry-after": "900"},
            data={"error": {"details": [{"reason": "QUOTA_EXHAUSTED"}]}},
        )
    )
    view = await app.state.monitor.check(a.id)
    assert view["next_check_at"] - now_ms() > 898000
    database.update(a.id, lambda a: a.state.update(check_started_at=0))
    with pytest.raises(Busy):
        await app.state.monitor.check(a.id)
    assert len(fake.requests) == 1


async def test_duplicate_checks_excluded_across_engine_instances(system, api):
    app, config, database, fake = system
    a = database.add("one")
    hold(database, a)
    gate, entered = asyncio.Event(), asyncio.Event()
    fake.replies["one"].append(Reply(gate=gate, entered=entered))
    first = asyncio.create_task(app.state.monitor.check(a.id))
    await asyncio.wait_for(entered.wait(), 2)
    other = Engine(config, database, fake.client())
    try:
        with pytest.raises(Busy):
            await VerificationMonitor(other).check(a.id)
        health = (await api.get("/health")).json()
        assert health["accounts"][0]["status"] == "checking"
        gate.set()
        assert (await first)["state"] == "recovered"
    finally:
        gate.set()
        await other.close()
        if not first.done():
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)


@pytest.mark.parametrize("change", ["reset", "new_challenge", "disable", "remove"])
async def test_late_results_cannot_overwrite_operator_or_new_challenge(system, api, change):
    app, _, database, fake = system
    a = database.add("one")
    hold(database, a)
    gate, entered = asyncio.Event(), asyncio.Event()
    fake.replies["one"].append(Reply(gate=gate, entered=entered))
    first = asyncio.create_task(app.state.monitor.check(a.id))
    await asyncio.wait_for(entered.wait(), 2)
    if change == "remove":
        database.remove(a.id)
    elif change == "reset":
        database.update(a.id, lambda a: a.reset_holds())
    elif change == "disable":

        def disable(a):
            a.enabled = False
            a.state["revision"] += 1

        database.update(a.id, disable)
    else:
        await app.state.engine.record_failure(
            a,
            GatewayError(
                403,
                "verification_required",
                "new challenge",
                verification_url="https://accounts.google.com/new",
            ),
            resolve({"model": "gemini-3.8-flash-low"}).quota_key,
        )
    if change != "remove":
        assert database.get(a.id).running(now_ms())
        with pytest.raises(Busy):
            await app.state.monitor.check(a.id)
    gate.set()
    if change == "remove":
        with pytest.raises(NotFound):
            await first
        assert database.list() == []
    else:
        view = await first
        assert view["state"] != "recovered"
        current = database.get(a.id)
        assert not current.running(now_ms())
        assert not current.state.get("recovered_at")
        if change == "new_challenge":
            assert current.state["verification_url"] == "https://accounts.google.com/new"


async def test_timeout_cancellation_and_lease_expiry(system, api):
    app, config, database, fake = system
    a = database.add("one")
    hold(database, a)
    # Warm authentication first so this assertion measures stream cancellation,
    # rather than filesystem scheduling during the first token/project cache fill.
    await app.state.engine.credentials.prepare(a)
    config.verification.timeout_secs = 1
    fake.replies["one"].append(Reply(gate=asyncio.Event()))
    view = await app.state.monitor.check(a.id)
    assert view["last_result"] == "verification_timeout"
    assert not database.get(a.id).running(now_ms())
    assert fake.bodies[-1].closed
    config.verification.timeout_secs = 60
    database.update(
        a.id,
        lambda a: a.state.update(
            check_started_at=0, lease_id="abandoned", lease_until=now_ms() - 1
        ),
    )
    gate, entered = asyncio.Event(), asyncio.Event()
    fake.replies["one"].append(Reply(gate=gate, entered=entered))
    task = asyncio.create_task(app.state.monitor.check(a.id))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not database.get(a.id).running(now_ms())
    assert database.get(a.id).state["verification_required"]
    assert fake.bodies[-1].closed


async def test_recheck_api_status_and_recovered_health(system, api):
    _, _, database, _ = system
    a = database.add("one")
    hold(database, a)
    assert (await api.get("/health")).status_code == 503
    assert (await api.post("/api/accounts/missing/recheck")).status_code == 404
    response = await api.post(f"/api/accounts/{a.id[:8]}/recheck")
    assert response.status_code == 200, response.text
    assert response.json()["verification"]["state"] == "recovered"
    assert (await api.get("/health")).status_code == 200
    assert (await api.post(f"/api/accounts/{a.id}/recheck")).status_code == 409
