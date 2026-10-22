import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from gravitygate_py.cli import parser, service_url
from gravitygate_py.config import Config
from gravitygate_py.errors import GatewayError
from gravitygate_py.oauth import OAuth, authorize_url, extract_code, login, pkce


def test_pkce_and_auth_url_contract():
    verifier, challenge = pkce()
    assert 43 <= len(verifier) <= 128
    assert (
        challenge
        == base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    values = parse_qs(
        urlsplit(authorize_url("http://localhost:51121/oauth-callback", challenge, "state")).query
    )
    assert values["code_challenge_method"] == ["S256"]
    assert values["prompt"] == ["consent"] and values["access_type"] == ["offline"]
    assert (
        extract_code("http://localhost/callback?code=the-code&state=state", "state") == "the-code"
    )
    with pytest.raises(ValueError):
        extract_code("http://localhost/callback?code=the-code&state=wrong", "state")


@pytest.mark.parametrize("expires", [None, 0, -1, 3600])
async def test_token_refresh_form_and_errors(expires):
    seen = []

    async def handle(request):
        seen.append(parse_qs((await request.aread()).decode()))
        return httpx.Response(200, json={"access_token": "synthetic", "expires_in": expires})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        oauth = OAuth(client)
        assert (await oauth.refresh("refresh"))["access_token"] == "synthetic"
        assert seen[0]["refresh_token"] == ["refresh"]
        assert seen[0]["grant_type"] == ["refresh_token"]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(400, json={"error": "invalid_grant"})
        )
    ) as client:
        with pytest.raises(GatewayError) as error:
            await OAuth(client).refresh("revoked")
        assert error.value.status == 401


async def test_login_loopback_pkce_state_and_cleanup(monkeypatch):
    servers = []
    original_start = asyncio.start_server

    async def tracked_start(*args, **kwargs):
        server = await original_start(*args, **kwargs)
        servers.append(server)
        return server

    monkeypatch.setattr(asyncio, "start_server", tracked_start)
    seen = {}

    class FakeOAuth:
        async def tokens(self, **form):
            seen.update(form)
            return {"refresh_token": "synthetic-refresh", "access_token": "synthetic-access"}

        async def userinfo(self, token):
            return {"email": "synthetic@test"}

    prompted = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(
        login(FakeOAuth(), no_open=True, ports=(0,), timeout=2, prompt=prompted.set_result)
    )
    url = await asyncio.wait_for(prompted, 2)
    query = parse_qs(urlsplit(url).query)
    redirect = query["redirect_uri"][0].replace("localhost", "127.0.0.1")
    async with httpx.AsyncClient(trust_env=False) as client:
        wrong = await client.get(redirect, params={"state": "wrong", "code": "bad"})
        assert wrong.status_code == 400 and not task.done()
        right = await client.get(redirect, params={"state": query["state"][0], "code": "correct"})
        assert right.status_code == 200
    assert await task == ("synthetic-refresh", "synthetic@test")
    assert seen["code"] == "correct"
    assert (
        base64.urlsafe_b64encode(hashlib.sha256(seen["code_verifier"].encode()).digest())
        .rstrip(b"=")
        .decode()
        == query["code_challenge"][0]
    )
    assert not servers[0].is_serving()
    assert servers[0].sockets == ()


async def test_callback_timeout_releases_socket(monkeypatch):
    servers = []
    original_start = asyncio.start_server

    async def tracked_start(*args, **kwargs):
        server = await original_start(*args, **kwargs)
        servers.append(server)
        return server

    monkeypatch.setattr(asyncio, "start_server", tracked_start)
    prompted = asyncio.get_running_loop().create_future()
    with pytest.raises(TimeoutError):
        await login(None, no_open=True, ports=(0,), timeout=0.02, prompt=prompted.set_result)
    assert prompted.done()
    assert not servers[0].is_serving()
    assert servers[0].sockets == ()


async def test_credentials_single_flight(system):
    app, _, database, fake = system
    account = database.add("one")
    try:
        prepared = await asyncio.gather(
            *(app.state.engine.credentials.prepare(account) for _ in range(20))
        )
        assert len(set(prepared)) == 1
        assert fake.refreshes == ["one"]
        assert fake.discovery == ["one"]
    finally:
        await app.state.engine.close()


def cli(tmp_path, *arguments):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("GG_", "GRAVITYGATE_")):
            del env[key]
    return subprocess.run(
        [sys.executable, "-m", "gravitygate_py", "--data-dir", str(tmp_path), *arguments],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )


def test_cli_config_and_accounts(tmp_path):
    assert cli(tmp_path, "--version").returncode == 0
    path = cli(tmp_path, "config", "path")
    assert "gravitygate.db" in path.stdout
    assert not (tmp_path / "gravitygate.db").exists()
    assert cli(tmp_path, "config", "init").returncode == 0
    assert cli(tmp_path, "config", "init").returncode == 1
    added = cli(
        tmp_path, "account", "add-token", "synthetic-refresh-token", "--email", "test@example.com"
    )
    assert added.returncode == 0, added.stderr
    assert "synthetic-refresh-token" not in added.stdout
    listing = cli(tmp_path, "account", "list", "--json")
    assert "synthetic-refresh-token" not in listing.stdout
    assert json.loads(listing.stdout)[0]["status"] == "ready"
    assert cli(tmp_path, "account", "disable", "0").returncode == 0
    assert json.loads(cli(tmp_path, "account", "list", "--json").stdout)[0]["status"] == "disabled"
    assert cli(tmp_path, "account", "enable", "0").returncode == 0
    assert cli(tmp_path, "account", "clear-holds", "0").returncode == 0
    assert cli(tmp_path, "account", "remove", "0").returncode == 0
    assert json.loads(cli(tmp_path, "account", "list", "--json").stdout) == []


def test_cli_help_and_static_models(tmp_path):
    for command in ("serve", "probe", "models", "account", "config"):
        result = cli(tmp_path, command, "--help")
        assert result.returncode == 0, result.stderr
    result = cli(tmp_path, "models")
    assert result.returncode == 0 and "gemini-3.8-flash-high" in result.stdout
    # A missing selector fails before OAuth or upstream access.
    result = cli(tmp_path, "probe", "--account", "missing")
    assert result.returncode == 1 and "account selector" in result.stderr


def test_service_url_wildcards_and_ipv6():
    config = Config()
    config.server.host = "0.0.0.0"
    assert service_url(config) == "http://127.0.0.1:18081"
    config.server.host = "::"
    assert service_url(config) == "http://[::1]:18081"
    assert parser().parse_args(["account", "recheck", "0"]).selector == "0"
