import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from urllib.parse import parse_qs

import httpx
import pytest

from gravitygate_py.config import Config
from gravitygate_py.server import create_app
from gravitygate_py.storage import Database


def payload(text="ok", *, finish="STOP", thought=False):
    return {
        "response": {
            "candidates": [
                {"content": {"parts": [{"text": text, "thought": thought}]}, "finishReason": finish}
            ],
            "usageMetadata": {
                "promptTokenCount": 6,
                "candidatesTokenCount": 1,
                "thoughtsTokenCount": 97,
                "totalTokenCount": 104,
            },
        },
        "traceId": "synthetic-trace",
    }


def frame(value):
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


@dataclass
class Reply:
    status: int = 200
    data: object = field(default_factory=lambda: payload())
    headers: dict = field(default_factory=dict)
    chunks: list[bytes] | None = None
    gate: asyncio.Event | None = None
    entered: asyncio.Event | None = None
    break_read: bool = False


class Body(httpx.AsyncByteStream):
    def __init__(self, reply: Reply):
        self.reply, self.closed = reply, False

    async def __aiter__(self):
        reply = self.reply
        if reply.entered:
            reply.entered.set()
        if reply.gate:
            await reply.gate.wait()
        for chunk in reply.chunks if reply.chunks is not None else [frame(reply.data)]:
            yield chunk
        if reply.break_read:
            raise httpx.ReadError("synthetic read interruption")

    async def aclose(self):
        self.closed = True


class FakeUpstream:
    def __init__(self):
        self.replies = defaultdict(deque)
        self.requests, self.bodies, self.refreshes, self.discovery = [], [], [], []
        self.invalid_tokens = set()
        self.models = {
            "gemini-3.8-flash-high": {
                "displayName": "Flash",
                "quotaInfo": {"remainingFraction": 0.5},
            },
            "gemini-future": {},
            "chat_internal": {},
        }

    async def handle(self, request: httpx.Request):
        data = await request.aread()
        if request.url.path == "/token":
            token = parse_qs(data.decode())["refresh_token"][0]
            self.refreshes.append(token)
            if token in self.invalid_tokens:
                return httpx.Response(400, json={"error": "invalid_grant"})
            return httpx.Response(200, json={"access_token": "access:" + token, "expires_in": 3600})
        token = request.headers.get("authorization", "").removeprefix("Bearer access:")
        if request.url.path.endswith(":loadCodeAssist"):
            self.discovery.append(token)
            return httpx.Response(
                200,
                json={
                    "cloudaicompanionProject": "test-project",
                    "currentTier": {"id": "free-tier"},
                },
            )
        if request.url.path.endswith(":fetchAvailableModels"):
            return httpx.Response(200, json={"models": self.models})
        assert request.url.path.endswith(":streamGenerateContent"), str(request.url)
        body = json.loads(data)
        self.requests.append((token, body, request))
        reply = self.replies[token].popleft() if self.replies[token] else Reply()
        if isinstance(reply, Exception):
            raise reply
        if reply.status >= 400:
            return httpx.Response(reply.status, json=reply.data, headers=reply.headers)
        stream = Body(reply)
        self.bodies.append(stream)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream", **reply.headers}, stream=stream
        )

    def client(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))


@pytest.fixture
def system(tmp_path, monkeypatch):
    # Tests use only an explicitly injected synthetic credential endpoint.
    monkeypatch.setattr("gravitygate_py.oauth.TOKEN_URL", "http://fake/token")
    config = Config()
    config.verification.enabled = False
    config.upstream.endpoints = ["http://fake-one", "http://fake-two"]
    config.routing.max_wait_before_error_secs = 0
    config.routing.max_capacity_retries = 0
    config.routing.max_empty_response_retries = 0
    config.accounts.token_bucket_max = 50
    database = Database(tmp_path / "gateway.db")
    fake = FakeUpstream()
    app = create_app(config, database, fake.client())
    return app, config, database, fake


@pytest.fixture
async def api(system):
    app, *_ = system
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            yield client


@pytest.fixture
def request_body():
    return {"model": "gemini-3.8-flash-high", "messages": [{"role": "user", "content": "hello"}]}
