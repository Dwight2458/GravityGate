"""Loopback-only HTTP integration: Uvicorn -> httpx -> a real TCP fake upstream."""

import asyncio
import json
import socket

import httpx
import uvicorn
from conftest import frame, payload

from gravitygate_py.config import Config
from gravitygate_py.server import create_app
from gravitygate_py.storage import Database


async def test_real_http_streaming_and_shutdown(tmp_path, monkeypatch):
    upstream_finished = asyncio.Event()
    captures = []
    clients = set()

    async def upstream(reader, writer):
        task = asyncio.current_task()
        clients.add(task)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode().split("\r\n")
            _, target, _ = lines[0].split(" ")
            headers = dict(line.split(": ", 1) for line in lines[1:] if ": " in line)
            lower = {k.lower(): v for k, v in headers.items()}
            body = bytearray()
            if lower.get("transfer-encoding") == "chunked":
                while size := int((await reader.readline()).strip(), 16):
                    body.extend(await reader.readexactly(size))
                    assert await reader.readexactly(2) == b"\r\n"
                await reader.readline()
            elif "content-length" in lower:
                body.extend(await reader.readexactly(int(lower["content-length"])))
            if target == "/token":
                value = {"access_token": "synthetic", "expires_in": 3600}
            elif target.endswith(":loadCodeAssist"):
                value = {"cloudaicompanionProject": "loopback-project"}
            else:
                captures.append((headers, json.loads(body)))
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                )
                first = frame(payload("first", finish=None))
                writer.write(f"{len(first):x}\r\n".encode() + first + b"\r\n")
                await writer.drain()
                await asyncio.sleep(0.15)
                last = frame(payload("second"))
                writer.write(f"{len(last):x}\r\n".encode() + last + b"\r\n0\r\n\r\n")
                await writer.drain()
                upstream_finished.set()
                return
            response = json.dumps(value).encode()
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(response)}\r\nConnection: close\r\n\r\n".encode()
                + response
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            clients.discard(task)

    fake_server = await asyncio.start_server(upstream, "127.0.0.1", 0)
    fake_url = f"http://127.0.0.1:{fake_server.sockets[0].getsockname()[1]}"
    monkeypatch.setattr("gravitygate_py.oauth.TOKEN_URL", fake_url + "/token")
    config = Config()
    config.verification.enabled = False
    config.upstream.endpoints = [fake_url]
    database = Database(tmp_path / "db")
    database.add("synthetic-refresh")
    app = create_app(config, database)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", trust_env=False
        ) as client:
            assert (await client.get("/health")).status_code == 200
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "gemini-3.8-flash-high",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                saw_first = False
                async for line in response.aiter_lines():
                    if '"content":"first"' in line:
                        assert not upstream_finished.is_set(), (
                            "stream was buffered until completion"
                        )
                        saw_first = True
                assert saw_first
            assert upstream_finished.is_set()
            assert captures[0][1]["project"] == "loopback-project"
            header_names = list(captures[0][0])
            assert (
                header_names.index("User-Agent")
                < header_names.index("Authorization")
                < header_names.index("Content-Type")
            )
            assert captures[0][0]["Transfer-Encoding"] == "chunked"
            assert "requests" in (await client.get("/api/stats")).json()
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        listener.close()
        fake_server.close()
        await fake_server.wait_closed()
        for client_task in clients.copy():
            client_task.cancel()
        if clients:
            await asyncio.gather(*clients, return_exceptions=True)
    assert app.state.engine.monitor_active is False
