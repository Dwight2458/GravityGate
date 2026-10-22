"""Generation-backed verification recovery with cross-process leases."""

import asyncio
import logging
import math
import uuid

import httpx

from .engine import Engine, apply_failure
from .errors import GatewayError
from .registry import resolve
from .storage import Account, Busy, NotFound, now_ms

log = logging.getLogger(__name__)


class VerificationMonitor:
    def __init__(self, engine: Engine):
        self.engine = engine

    async def check(self, selector: str, *, immediate: bool = True) -> dict:
        engine, config = self.engine, self.engine.config
        identity = str(uuid.uuid4())
        request = {
            "model": config.verification.model,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            "max_tokens": 1024,
            "user": "verification:" + identity,
        }
        quota = resolve(request, config.reasoning.default_tier).quota_key

        def claim(a: Account):
            now, s = now_ms(), a.state
            if a.running(now) or s.get("check_started_at", 0) + 15000 > now:
                raise Busy(
                    "a verification check is running or was started within the last 15 seconds"
                )
            until = engine.database.blocked_until(a, quota, now, recovery=True)
            tokens = engine.database.bucket(a, config.accounts, now)
            if until is None or until > now or tokens < 1:
                raise Busy(
                    "account is disabled, ineligible, cooling down or rate limited for the check model"
                )
            if not immediate and s.get("recheck_at", 0) > now:
                raise Busy("verification recheck is not due")
            s.update(
                lease_id=identity,
                lease_until=now + math.ceil(config.verification.timeout_secs * 1000) + 30000,
                check_started_at=now,
                tokens=tokens - 1,
                tokens_at=now,
            )
            return a, s.get("revision", 0)

        claim_task = asyncio.create_task(asyncio.to_thread(engine.database.update, selector, claim))
        try:
            account, revision = await asyncio.shield(claim_task)
        except asyncio.CancelledError:
            # A cancelled to_thread await does not stop the SQLite transaction.
            # Wait for the claim, then release precisely that lease before exiting.
            try:
                account, revision = await claim_task

                def release_claim(a: Account):
                    if a.state.get("lease_id") == identity:
                        a.state.pop("lease_id", None)
                        a.state.pop("lease_until", None)
                        if a.state.get("revision", 0) == revision:
                            a.state.update(
                                last_result="verification_cancelled", recheck_at=now_ms() + 60000
                            )

                await asyncio.to_thread(engine.database.update, account.id, release_claim)
            except Exception:
                log.exception("could not release a cancelled verification claim")
            raise
        completed = False
        try:
            failure = None
            try:
                async with asyncio.timeout(config.verification.timeout_secs):
                    run = engine.execution(request, recovery_account=account)
                    run.response_limit = 1024 * 1024
                    await run.drain()
                    if (
                        not run.clean
                        or not any(text.strip() for text in run.acc.text)
                        or run.acc.finish_reason != "STOP"
                    ):
                        failure = GatewayError(
                            502,
                            "verification_unproven",
                            "visible output and a clean STOP are required",
                        )
            except TimeoutError:
                failure = GatewayError(502, "verification_timeout", "verification check timed out")
            except GatewayError as exc:
                failure = exc
            except (httpx.HTTPError, ValueError):
                failure = GatewayError(502, "verification_failed", "verification check failed")

            def finish(a: Account):
                s, now = a.state, now_ms()
                if s.get("lease_id") != identity:
                    return a.verification(now)
                s.pop("lease_id", None)
                s.pop("lease_until", None)
                if s.get("revision", 0) != revision or not a.enabled or s.get("ineligible"):
                    return a.verification(now)
                s["last_checked_at"] = now
                if failure is None:
                    s.update(
                        verification_required=False,
                        verification_url=None,
                        verification_reason=None,
                        recheck_at=None,
                        last_result="recovered",
                        recovered_at=now,
                        verification_failures=0,
                        failures=0,
                        health=100,
                        last_used=now,
                        revision=revision + 1,
                    )
                    s.get("holds", {}).pop(quota, None)
                    s.get("quota_failures", {}).pop(quota, None)
                else:
                    failures = s.get("verification_failures", 0) + 1
                    apply_failure(a, failure, quota, config)
                    delay = (60, 120, 300)[min(failures - 1, 2)]
                    s.update(
                        verification_failures=failures,
                        last_result=failure.code,
                        recheck_at=max(now + delay * 1000, s.get("holds", {}).get(quota, 0)),
                    )
                return a.verification(now)

            view = await asyncio.to_thread(engine.database.update, account.id, finish)
            completed = True
            return view
        finally:
            if not completed:

                def release(a: Account):
                    if a.state.get("lease_id") == identity:
                        a.state.pop("lease_id", None)
                        a.state.pop("lease_until", None)
                        if a.state.get("revision", 0) == revision:
                            a.state.update(
                                recheck_at=now_ms() + 60000, last_result="verification_cancelled"
                            )

                try:
                    await asyncio.shield(
                        asyncio.to_thread(engine.database.update, account.id, release)
                    )
                except NotFound:
                    pass

    async def scan(self) -> None:
        accounts = await asyncio.to_thread(self.engine.database.list)
        now = now_ms()
        for a in accounts:
            if (
                a.enabled
                and not a.state.get("ineligible")
                and a.state.get("verification_required")
                and not a.running(now)
                and (a.state.get("recheck_at") or 0) <= now
            ):
                try:
                    await self.check(a.id, immediate=False)
                except (Busy, NotFound):
                    pass
                except Exception:
                    log.exception("verification scan failed for account %s", a.id[:8])

    async def run(self) -> None:
        self.engine.monitor_active = self.engine.config.verification.enabled
        try:
            while True:
                if self.engine.config.verification.enabled:
                    try:
                        await self.scan()
                    except Exception:
                        log.exception("verification scan failed; retrying at the next scan")
                await asyncio.sleep(self.engine.config.verification.scan_interval_secs)
        finally:
            self.engine.monitor_active = False
