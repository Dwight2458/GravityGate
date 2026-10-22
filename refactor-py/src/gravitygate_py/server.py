"""FastAPI presentation. Stream failures never become a successful finish."""

import asyncio
import base64
import hmac
import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from .config import Config, data_dir
from .engine import Engine
from .errors import GatewayError
from .observability import Observability
from .registry import model_list
from .request import responses_to_chat
from .response import ResponsesStream
from .sse import encode_event
from .storage import Busy, Database, NotFound, now_ms
from .verification import VerificationMonitor

log = logging.getLogger(__name__)
GENERATION_ROUTES = {"/v1/chat/completions", "/v1/responses"}


class Authentication:
    def __init__(self, app, keys: list[str]):
        self.app, self.keys = app, keys

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.keys:
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope["headers"]}
        token = headers.get(b"x-api-key", b"")
        authorization = headers.get(b"authorization", b"")
        scheme, _, value = authorization.partition(b" ")
        if scheme.lower() == b"bearer":
            token = value
        elif scheme.lower() == b"basic":
            try:
                token = base64.b64decode(value, validate=True).partition(b":")[2]
            except ValueError:
                token = b""
        if not any(hmac.compare_digest(token, key.encode()) for key in self.keys):
            error = GatewayError(401, "authentication_error", "missing or invalid gateway API key")
            response = JSONResponse(
                error.envelope(),
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="GravityGate (API key as password)"'}
                if scope.get("path") == "/"
                else {},
            )
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)


def create_app(
    config: Config | None = None,
    database: Database | None = None,
    client: httpx.AsyncClient | None = None,
    db_path: Path | None = None,
) -> FastAPI:
    config = config or Config.load()
    database = database or Database(db_path or data_dir() / "gravitygate.db")
    engine = Engine(config, database, client)
    monitor, observ = VerificationMonitor(engine), Observability(database, config)
    engine.on_account_selected = observ.account_selected
    engine.on_adverse = observ.adverse

    @asynccontextmanager
    async def lifespan(app):
        observ.start()
        engine.monitor_active = config.verification.enabled
        task = asyncio.create_task(monitor.run(), name="gravitygate-verification")
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await engine.close()
            await observ.close()

    app = FastAPI(
        title="GravityGate Python",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.engine, app.state.database, app.state.monitor, app.state.observ = (
        engine,
        database,
        monitor,
        observ,
    )
    app.add_middleware(Authentication, keys=config.server.api_keys)

    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, exc: GatewayError):
        if request.url.path in GENERATION_ROUTES and not getattr(request.state, "audited", False):
            observ.record(
                run=getattr(request.state, "run", None), status=exc.status, outcome=exc.code
            )
            request.state.audited = True
        return JSONResponse(exc.envelope(), status_code=exc.status, headers=exc.headers)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return await gateway_error(
            request, GatewayError(exc.status_code, "invalid_request_error", str(exc.detail))
        )

    @app.exception_handler(RequestValidationError)
    async def parameter_error(request: Request, exc: RequestValidationError):
        return await gateway_error(
            request, GatewayError(422, "invalid_request_error", "invalid request parameters")
        )

    @app.exception_handler(sqlite3.Error)
    async def storage_error(request: Request, exc: sqlite3.Error):
        log.error("database operation failed: %s", type(exc).__name__)
        return await gateway_error(
            request, GatewayError(503, "storage_unavailable", "account storage is unavailable")
        )

    async def parse(request: Request) -> dict:
        media = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media != "application/json" and not media.endswith("+json"):
            raise GatewayError(
                415, "invalid_request_error", "Content-Type must be application/json"
            )
        limit = config.server.max_request_body_bytes
        length = request.headers.get("content-length")
        if length is not None:
            try:
                declared = int(length)
                if declared < 0:
                    raise ValueError
            except ValueError as exc:
                raise GatewayError(400, "invalid_request_error", "invalid Content-Length") from exc
            if declared > limit:
                raise GatewayError(
                    413, "request_body_too_large", f"request body exceeds {limit} bytes"
                )
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                raise GatewayError(
                    413, "request_body_too_large", f"request body exceeds {limit} bytes"
                )
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise GatewayError(
                400, "invalid_request_error", "request body contains invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise GatewayError(422, "invalid_request_error", "request body must be a JSON object")
        return payload

    async def generate(request: Request, responses: bool):
        raw = await parse(request)
        try:
            converted = responses_to_chat(raw) if responses else raw
            run = engine.execution(converted)
        except (
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            RecursionError,
            ValidationError,
        ) as exc:
            message = str(exc) if type(exc) is ValueError else "invalid request fields"
            raise GatewayError(
                400 if type(exc) is ValueError else 422, "invalid_request_error", message
            ) from exc
        request.state.run = run
        if run.request.get("stream"):
            try:
                await run.start()
            except BaseException:
                await run.close()
                raise

            async def stream():
                status, outcome = 200, "ok"
                translator = ResponsesStream(run.acc, raw) if responses else None
                try:
                    if translator:
                        for event in translator.start():
                            yield encode_event(event, event["type"])
                    else:
                        yield encode_event(run.acc.chunk({"role": "assistant"}))
                    async for deltas in run.iterate():
                        if translator:
                            for event in translator.on_deltas(deltas):
                                yield encode_event(event, event["type"])
                        else:
                            for delta in deltas:
                                yield encode_event(run.acc.chunk(delta))
                    if translator:
                        for event in translator.finish():
                            yield encode_event(event, event["type"])
                    else:
                        for chunk in run.acc.final_chunks(
                            bool((run.request.get("stream_options") or {}).get("include_usage"))
                        ):
                            yield encode_event(chunk)
                        yield encode_event("[DONE]")
                except GatewayError as exc:
                    status, outcome = exc.status, exc.code
                    if translator:
                        failed = translator.response("failed")
                        failed["error"] = exc.envelope()["error"]
                        event = translator.event("response.failed", response=failed)
                        yield encode_event(event, event["type"])
                    else:
                        yield encode_event(exc.envelope())
                        yield encode_event("[DONE]")
                except (asyncio.CancelledError, GeneratorExit):
                    status, outcome = 499, "client_cancelled"
                    raise
                finally:
                    await run.close()
                    observ.record(run=run, status=status, outcome=outcome)
                    request.state.audited = True

            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        translator = ResponsesStream(run.acc, raw) if responses else None
        try:
            async for deltas in run.iterate():
                if translator:
                    translator.on_deltas(deltas)
            if translator:
                translator.finish()
                result = translator.response(
                    "incomplete" if run.acc.finish_reason == "MAX_TOKENS" else "completed"
                )
            else:
                result = run.acc.completion()
            observ.record(run=run)
            request.state.audited = True
            return JSONResponse(result)
        finally:
            await run.close()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await generate(request, False)

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await generate(request, True)

    @app.get("/v1/models")
    async def models():
        lists = await engine.available_models()
        live = next(iter(lists.values()), {})
        return {"object": "list", "data": model_list(live)}

    @app.get("/health")
    async def health():
        accounts = [a.public() for a in await asyncio.to_thread(database.list)]
        available = sum(a["status"] in {"ready", "limited"} for a in accounts)
        status = (
            "unhealthy"
            if not available
            else "healthy"
            if all(a["status"] == "ready" for a in accounts)
            else "degraded"
        )
        return JSONResponse(
            {
                "status": status,
                "serving": available > 0,
                "accounts_configured": len(accounts),
                "accounts_available": available,
                "accounts": accounts,
                "verification_monitor_enabled": engine.monitor_active,
                "signatures_cached": len(engine.signatures),
            },
            status_code=200 if available else 503,
        )

    @app.get("/account-limits")
    async def limits():
        live = await engine.available_models(all_accounts=True)
        return {
            "accounts": [
                {
                    "account": a.label,
                    "credential_id": a.id[:8],
                    "tier": a.state.get("tier"),
                    "pools": {
                        k: {"resets_in_seconds": max(0, (v - now_ms()) // 1000)}
                        for k, v in a.state.get("holds", {}).items()
                    },
                    "models": live.get(a.id, {}),
                }
                for a in await asyncio.to_thread(database.list)
            ]
        }

    @app.post("/refresh-token")
    async def refresh():
        engine.credentials.clear()
        engine.signatures.clear()
        engine.sessions.clear()
        engine.live_cache.clear()
        return {
            "status": "ok",
            "detail": "access token, project, model, session and signature caches cleared",
        }

    @app.post("/api/accounts/{selector}/recheck")
    async def recheck(selector: str):
        try:
            return {"verification": await monitor.check(selector)}
        except NotFound as exc:
            raise GatewayError(404, "account_not_found", str(exc)) from exc
        except Busy as exc:
            raise GatewayError(409, "verification_check_busy", str(exc)) from exc

    @app.get("/api/stats")
    async def stats():
        return await asyncio.to_thread(observ.stats)

    @app.get("/api/stats/accounts")
    async def account_stats():
        return await asyncio.to_thread(observ.stats, accounts=True)

    @app.get("/api/requests")
    async def requests(limit: int = Query(default=40, ge=1, le=1000)):
        return {"requests": await asyncio.to_thread(observ.recent, limit)}

    @app.get("/metrics")
    async def metrics():
        if not config.metrics.enabled:
            raise GatewayError(404, "not_found", "metrics are disabled")
        accounts = [a.public() for a in await asyncio.to_thread(database.list)]
        return Response(
            observ.metrics(accounts, len(engine.signatures)), media_type="text/plain; version=0.0.4"
        )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return HTMLResponse(
            files("gravitygate_py").joinpath("dashboard.html").read_text(encoding="utf-8")
        )

    return app
