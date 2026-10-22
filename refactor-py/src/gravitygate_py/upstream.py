"""HTTP transport and the captured agent metadata envelope."""

import json
import platform
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator

import httpx

from .config import Config
from .errors import classify
from .storage import now_ms

MODEL_ENUM = {
    "gemini-3.5-flash-extra-low": "MODEL_PLACEHOLDER_M187",
    "gemini-3.5-flash-low": "MODEL_PLACEHOLDER_M20",
    "gemini-3-flash-agent": "MODEL_PLACEHOLDER_M84",
    "gemini-3.6-flash-low": "MODEL_PLACEHOLDER_M73",
    "gemini-3.6-flash-medium": "MODEL_PLACEHOLDER_M72",
    "gemini-3.6-flash-high": "MODEL_PLACEHOLDER_M71",
    "gemini-3.7-flash-low": "MODEL_PLACEHOLDER_M300",
    "gemini-3.7-flash-medium": "MODEL_PLACEHOLDER_M299",
    "gemini-3.7-flash-high": "MODEL_PLACEHOLDER_M298",
    "gemini-3.8-flash-low": "MODEL_PLACEHOLDER_M320",
    "gemini-3.8-flash-medium": "MODEL_PLACEHOLDER_M319",
    "gemini-3.8-flash-high": "MODEL_PLACEHOLDER_M318",
    "gemini-3.1-pro-low": "MODEL_PLACEHOLDER_M36",
    "gemini-pro-agent": "MODEL_PLACEHOLDER_M16",
    "claude-sonnet-4-6": "MODEL_PLACEHOLDER_M35",
    "claude-opus-4-6-thinking": "MODEL_PLACEHOLDER_M26",
    "gemini-3.1-flash-image": "MODEL_PLACEHOLDER_M21",
    "gpt-oss-120b-medium": "MODEL_OPENAI_GPT_OSS_120B_MEDIUM",
}


def user_agent() -> str:
    system = {"Darwin": "darwin", "Windows": "windows", "Linux": "linux"}.get(
        platform.system(), platform.system().lower()
    )
    arch = {"AMD64": "amd64", "x86_64": "amd64", "aarch64": "arm64"}.get(
        platform.machine(), platform.machine()
    )
    return f"antigravity/cli/1.1.24 (aidev_client; os_type={system}; arch={arch}; cl=974782877; auth_method=consumer)"


def fnv1a64_signed(value: str) -> int:
    result = 0xCBF29CE484222325
    for byte in value.encode():
        result = ((result ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return result if result < 2**63 else result - 2**64


class Sessions:
    def __init__(self):
        self.items: OrderedDict[str, dict] = OrderedDict()

    def envelope(self, request: dict, project: str, model: str, key: str) -> dict:
        now = now_ms()
        for old in [k for k, v in self.items.items() if v["touched"] + 86400 < time.monotonic()]:
            del self.items[old]
        if key not in self.items:
            self.items[key] = {
                "conversation": str(uuid.uuid4()),
                "trajectory": str(uuid.uuid4()),
                "timestamp": 0,
                "claude": False,
                "non_gemini": False,
                "execution": None,
            }
        ctx = self.items[key]
        self.items.move_to_end(key)
        while len(self.items) > 512:
            self.items.popitem(last=False)
        ctx.update(
            touched=time.monotonic(),
            timestamp=max(now, ctx["timestamp"] + 1),
            claude=ctx["claude"] or model.startswith("claude-"),
            non_gemini=ctx["non_gemini"] or model.startswith(("claude-", "gpt-")),
        )
        step = max(1, sum(len(c["parts"]) for c in request["contents"])) + bool(ctx["execution"])
        labels = {}
        if ctx["execution"]:
            labels["last_execution_id"] = ctx["execution"]
        labels["last_step_index"] = str(step)
        if model in MODEL_ENUM:
            labels["model_enum"] = MODEL_ENUM[model]
        labels.update(
            trajectory_id=ctx["trajectory"],
            used_claude=str(ctx["claude"]).lower(),
            used_claude_conservative=str(ctx["claude"]).lower(),
            used_non_gemini_model=str(ctx["non_gemini"]).lower(),
        )
        # Rebuild in captured order; Python dicts preserve insertion order.
        inner = {
            k: request[k]
            for k in ("contents", "systemInstruction", "tools", "toolConfig")
            if k in request
        }
        inner["labels"] = labels
        if "generationConfig" in request:
            inner["generationConfig"] = request["generationConfig"]
        inner["sessionId"] = str(fnv1a64_signed(key))
        return {
            "project": project,
            "requestId": f"agent/{ctx['conversation']}/{ctx['timestamp']}/{ctx['trajectory']}/{step + 1}",
            "request": inner,
            "model": model,
            "userAgent": "antigravity",
            "requestType": "agent",
        }

    def completed(self, key: str) -> None:
        if key in self.items:
            self.items[key]["execution"] = str(uuid.uuid4())

    def clear(self) -> None:
        self.items.clear()


class Transport:
    def __init__(self, config: Config, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(
            http1=True,
            http2=False,
            timeout=httpx.Timeout(
                config.upstream.read_timeout_secs, connect=config.upstream.connect_timeout_secs
            ),
            follow_redirects=False,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def post(
        self, url: str, token: str, payload: dict, *, streaming: bool = False
    ) -> httpx.Response:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        headers = [
            ("User-Agent", user_agent()),
            ("Authorization", f"Bearer {token}"),
            ("Content-Type", "application/json"),
            ("Accept-Encoding", "gzip"),
        ]

        async def chunks() -> AsyncIterator[bytes]:
            yield body

        request = httpx.Request(
            "POST", url, headers=headers, content=chunks() if streaming else body
        )
        response = await self.client.send(request, stream=True)
        if not 200 <= response.status_code < 300:
            try:
                content = await self.read_bounded(response, 1024 * 1024)
                raise classify(
                    response.status_code, response.headers, content.decode(errors="replace")
                )
            finally:
                await response.aclose()
        return response

    @staticmethod
    async def read_bounded(response: httpx.Response, maximum: int = 16 * 1024 * 1024) -> bytes:
        from .sse import StreamError

        result = bytearray()
        async for chunk in response.aiter_bytes():
            result.extend(chunk)
            if len(result) > maximum:
                raise StreamError("upstream response exceeds byte limit")
        return bytes(result)

    async def json(self, url: str, token: str, payload: dict) -> dict:
        response = await self.post(url, token, payload)
        try:
            return json.loads(await self.read_bounded(response))
        finally:
            await response.aclose()
