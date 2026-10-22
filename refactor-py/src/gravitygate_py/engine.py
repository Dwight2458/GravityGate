"""Bounded dispatch and classification, independent of HTTP presentation."""

import asyncio
import logging
import math
import random
import time
from collections.abc import AsyncIterator, Callable

import httpx

from .cache import SignatureCache
from .config import Config
from .errors import GatewayError, backoff
from .oauth import Credentials
from .registry import Resolved, resolve
from .request import ChatRequest, session_key, to_ir
from .response import Accumulator
from .sse import StreamError, decode_stream
from .storage import Account, Database, NotFound, now_ms
from .upstream import Sessions, Transport

log = logging.getLogger(__name__)


def apply_failure(a: Account, error: GatewayError, quota: str, config: Config) -> None:
    now, s = now_ms(), a.state
    if error.code in {
        "model_capacity_exhausted",
        "permission_denied",
        "model_not_found",
        "upstream_bad_request",
    }:
        return
    if error.code in {"quota_exhausted", "rate_limit_exceeded", "rate_limit_unknown"}:
        counts = s.setdefault("quota_failures", {})
        counts[quota] = counts.get(quota, 0) + 1
        s.setdefault("holds", {})[quota] = now + math.ceil(backoff(error, counts[quota]) * 1000)
        return
    s["failures"] = s.get("failures", 0) + 1
    s["health"] = max(0, Database.health(a, now) - 20)
    s["health_at"] = now
    if error.code == "verification_required":
        s.update(
            verification_required=True,
            verification_reason=error.message,
            verification_url=error.verification_url,
            revision=s.get("revision", 0) + 1,
            recheck_at=now + 60000,
            last_result=error.code,
        )
        s.pop("recovered_at", None)
    elif error.code in {"banned", "account_ineligible"}:
        if error.code == "banned":
            a.enabled = False
        s.update(
            ineligible=True, ineligible_reason=error.message, revision=s.get("revision", 0) + 1
        )
    elif error.status == 401:
        a.enabled = False
        s["cooldown_reason"] = error.code
    elif error.code in {
        "network_error",
        "auth_invalid",
        "project_error",
        "upstream_server_error",
        "upstream_stream_error",
    }:
        if s["failures"] >= config.accounts.max_consecutive_failures:
            s.update(
                cooldown_until=now + config.accounts.cooldown_secs * 1000,
                cooldown_reason=error.code,
            )


class Execution:
    def __init__(
        self,
        engine: "Engine",
        request: dict,
        selector: str | None = None,
        recovery_account: Account | None = None,
    ):
        self.engine = engine
        self.request = ChatRequest.model_validate(request).model_dump(exclude_none=True)
        self.resolved: Resolved = resolve(self.request, engine.config.reasoning.default_tier)
        self.ir = to_ir(self.request, self.resolved, engine.config, engine.signatures)
        self.session = session_key(self.request)
        self.account: Account | None = None
        self.endpoint: str | None = None
        self.endpoint_index = 0
        self.project: str | None = None
        self.attempts, self.emitted = 0, False
        self.clean = False
        self.response_limit: int | None = None
        self.capture_raw = False
        self.raw_events: list[dict] = []
        self.trace_id: str | None = None
        self.started_at = time.monotonic()
        self.acc = self.new_accumulator()
        self.events = engine._dispatch(self, selector, recovery_account)
        self.first: list[dict] | None = None

    def new_accumulator(self) -> Accumulator:
        return Accumulator(
            self.request["model"],
            self.resolved.family,
            self.session,
            self.engine.signatures,
            self.engine.config.reasoning.output_field,
        )

    async def start(self) -> "Execution":
        self.first = await anext(self.events)
        return self

    async def iterate(self) -> AsyncIterator[list[dict]]:
        try:
            if self.first is not None:
                first, self.first = self.first, None
                yield first
            async for deltas in self.events:
                yield deltas
        finally:
            await self.events.aclose()

    async def drain(self) -> None:
        async for _ in self.iterate():
            pass

    async def close(self) -> None:
        await self.events.aclose()


class Engine:
    def __init__(self, config: Config, database: Database, client: httpx.AsyncClient | None = None):
        self.config, self.database = config, database
        self.transport = Transport(config, client)
        self.credentials = Credentials(self.transport, database, config.upstream.endpoints)
        self.sessions, self.signatures = Sessions(), SignatureCache()
        self.monitor_active = False
        self.on_account_selected: Callable[[Account], None] | None = None
        self.on_adverse: Callable[[str], None] | None = None
        self.live_cache: dict[str, tuple[float, dict]] = {}
        self.live_lock = asyncio.Lock()

    async def close(self) -> None:
        await self.transport.close()

    def execution(
        self, request: dict, selector: str | None = None, recovery_account: Account | None = None
    ) -> Execution:
        return Execution(self, request, selector, recovery_account)

    async def record_failure(self, account: Account, error: GatewayError, quota: str) -> None:
        if self.on_adverse:
            self.on_adverse(error.code)
        try:
            await asyncio.to_thread(
                self.database.update,
                account.id,
                lambda a: apply_failure(a, error, quota, self.config),
            )
        except NotFound:
            pass

    async def record_success(self, account: Account, quota: str) -> None:
        def update(a: Account):
            s = a.state
            s.update(
                failures=0,
                health=min(100, Database.health(a, now_ms()) + 1),
                health_at=now_ms(),
                last_used=now_ms(),
            )
            s.get("holds", {}).pop(quota, None)
            s.get("quota_failures", {}).pop(quota, None)
            if (
                s.get("verification_required")
                and not a.running(now_ms())
                and s.get("revision", 0) == account.state.get("revision", 0)
            ):
                s.update(
                    verification_required=False,
                    verification_url=None,
                    recheck_at=None,
                    last_checked_at=now_ms(),
                    recovered_at=now_ms(),
                    last_result="recovered",
                    verification_failures=0,
                    health=100,
                    revision=s.get("revision", 0) + 1,
                )

        try:
            await asyncio.to_thread(self.database.update, account.id, update)
        except NotFound:
            pass

    async def _attempt(self, run: Execution, account: Account) -> AsyncIterator[list[dict]]:
        run.account = account
        try:
            token, project = await self.credentials.prepare(account)
        except httpx.HTTPError as exc:
            raise GatewayError(
                502, "network_error", "upstream authentication transport failed"
            ) from exc
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise GatewayError(
                502, "project_error", "upstream authentication or project response was malformed"
            ) from exc
        except NotFound as exc:
            raise GatewayError(
                503, "account_removed", "selected account was removed during authentication"
            ) from exc
        refreshed = False
        run.project = project
        for index in range(run.endpoint_index, len(self.config.upstream.endpoints)):
            endpoint = self.config.upstream.endpoints[index]
            run.endpoint_index = index
            response = None
            try:
                run.attempts += 1
                run.account, run.endpoint = account, endpoint
                envelope = self.sessions.envelope(
                    run.ir, project, run.resolved.wire_model, run.session
                )
                jitter = self.config.upstream.request_jitter_max_ms
                if jitter:
                    await asyncio.sleep(random.random() * jitter / 1000)
                url = endpoint + "/v1internal:streamGenerateContent?alt=sse"
                try:
                    response = await self.transport.post(url, token, envelope, streaming=True)
                except GatewayError as error:
                    if error.code != "auth_invalid" or refreshed:
                        raise
                    self.credentials.invalidate(account.id)
                    token, project = await self.credentials.prepare(account)
                    refreshed = True
                    run.attempts += 1
                    response = await self.transport.post(url, token, envelope, streaming=True)

                async def bounded_body(current_response=response):
                    size = 0
                    async for chunk in current_response.aiter_bytes():
                        size += len(chunk)
                        if run.response_limit is not None and size > run.response_limit:
                            raise StreamError("verification response exceeds byte limit")
                        yield chunk

                async for payload in decode_stream(
                    bounded_body(), response.headers.get("content-type", "")
                ):
                    run.trace_id = payload.get("traceId", run.trace_id)
                    if run.capture_raw:
                        run.raw_events.append(payload)
                    deltas = run.acc.absorb(payload)
                    if deltas:
                        run.emitted = True
                        yield deltas
                if not run.acc.has_output:
                    raise GatewayError(
                        502, "empty_upstream_response", "upstream returned no usable content"
                    )
                if not run.acc.finish_reason:
                    raise GatewayError(
                        502, "upstream_stream_error", "upstream stream ended before a finish reason"
                    )
                run.clean = True
                self.sessions.completed(run.session)
                return
            except httpx.HTTPError as exc:
                error = GatewayError(502, "network_error", "upstream transport failed")
                if run.emitted or index + 1 == len(self.config.upstream.endpoints):
                    raise error from exc
            except GatewayError as exc:
                if (
                    run.emitted
                    or exc.code
                    not in {"permission_denied", "upstream_server_error", "network_error"}
                    or index + 1 == len(self.config.upstream.endpoints)
                ):
                    raise
            except (
                StreamError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                RecursionError,
            ) as exc:
                raise GatewayError(
                    502,
                    "upstream_stream_error",
                    "upstream returned a malformed or interrupted stream",
                ) from exc
            finally:
                if response is not None:
                    await response.aclose()

    async def _dispatch(
        self, run: Execution, selector: str | None, recovery: Account | None
    ) -> AsyncIterator[list[dict]]:
        if recovery:
            async for deltas in self._attempt(run, recovery):
                yield deltas
            return
        config, quota = self.config, run.resolved.quota_key
        deadline = time.monotonic() + config.routing.max_wait_before_error_secs
        excluded: set[str] = set()
        last_error = None
        count = 0
        while count < config.routing.max_account_attempts:
            account, wait = await asyncio.to_thread(
                self.database.select,
                quota,
                config.accounts,
                excluded,
                selector=selector,
                monitor=self.monitor_active,
            )
            if account is None:
                if wait is not None:
                    delay = max(0.01, (wait - now_ms()) / 1000)
                    if delay <= deadline - time.monotonic():
                        await asyncio.sleep(delay)
                        excluded.clear()
                        continue
                    status = 400 if config.routing.exhausted_error_mode == "bad-request" else 429
                    raise GatewayError(
                        status,
                        "account_pool_rate_limited",
                        "no account is available for this model's quota group",
                        delay,
                    )
                if last_error:
                    raise GatewayError(
                        503,
                        "account_pool_exhausted",
                        "no remaining account can serve this request; last failure: "
                        + last_error.code,
                        last_error.retry_after,
                    )
                accounts = await asyncio.to_thread(self.database.list)
                statuses = ", ".join(f"{a.label}: {a.public()['status']}" for a in accounts)
                raise GatewayError(
                    503,
                    "account_pool_unavailable",
                    "no account can serve this request"
                    + (f" ({statuses})" if statuses else "; add an account first"),
                )
            count += 1
            run.endpoint_index = 0
            if self.on_account_selected:
                self.on_account_selected(account)
            retries, empty_retries = 0, 0
            while True:
                try:
                    async for deltas in self._attempt(run, account):
                        yield deltas
                    await self.record_success(account, quota)
                    return
                except GatewayError as error:
                    last_error = error
                    if (
                        not run.emitted
                        and error.code == "empty_upstream_response"
                        and empty_retries < config.routing.max_empty_response_retries
                    ):
                        empty_retries += 1
                        run.acc = run.new_accumulator()
                        continue
                    if error.code not in {
                        "model_not_found",
                        "upstream_bad_request",
                        "empty_upstream_response",
                    }:
                        await self.record_failure(account, error, quota)
                    if run.emitted:
                        raise
                    transient = error.code in {
                        "model_capacity_exhausted",
                        "upstream_server_error",
                        "network_error",
                    }
                    if transient and retries < config.routing.max_capacity_retries:
                        delay = backoff(error)
                        if delay <= deadline - time.monotonic():
                            retries += 1
                            await asyncio.sleep(delay)
                            run.acc = run.new_accumulator()
                            continue
                    if error.code in {
                        "model_capacity_exhausted",
                        "model_not_found",
                        "upstream_bad_request",
                    }:
                        if error.code == "model_capacity_exhausted":
                            if run.endpoint_index + 1 < len(config.upstream.endpoints):
                                run.endpoint_index += 1
                                run.acc = run.new_accumulator()
                                continue
                            error.retry_after = backoff(error)
                        raise
                    excluded.add(account.id)
                    run.acc = run.new_accumulator()
                    break
        raise GatewayError(
            503,
            "account_attempts_exhausted",
            "account attempt budget exhausted; last failure: "
            + (last_error.code if last_error else "unknown"),
        )

    async def available_models(self, *, all_accounts: bool = False) -> dict:
        async with self.live_lock:
            accounts = await asyncio.to_thread(self.database.list)
            result = {}
            for account in accounts:
                if (
                    not account.enabled
                    or account.state.get("ineligible")
                    or account.state.get("verification_required")
                ):
                    continue
                cached = self.live_cache.get(account.id)
                if cached and cached[0] > time.monotonic():
                    result[account.id] = cached[1]
                else:
                    try:
                        token, project = await self.credentials.prepare(account)
                        payload = await self.transport.json(
                            self.config.upstream.endpoints[0] + "/v1internal:fetchAvailableModels",
                            token,
                            {"project": project},
                        )
                        models = payload.get("models", {})
                        if not isinstance(models, dict):
                            raise ValueError("models must be an object")
                        self.live_cache[account.id] = (
                            time.monotonic() + min(300, self.config.accounts.quota_refresh_secs),
                            models,
                        )
                        result[account.id] = models
                    except (GatewayError, httpx.HTTPError, ValueError):
                        if cached:
                            result[account.id] = cached[1]
                if result and not all_accounts:
                    break
            return result
