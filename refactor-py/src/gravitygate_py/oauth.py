"""OAuth PKCE login, token single-flight and account project discovery."""

import asyncio
import base64
import hashlib
import hmac
import logging
import secrets
import time
import webbrowser
from collections import OrderedDict
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from .errors import GatewayError
from .storage import Account, Database, now_ms
from .upstream import Transport

# Public installed-app client identity, also distributed by the existing CLI.
CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
CLIENT_SECRET = "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"
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo"
SCOPES = " ".join(
    "https://www.googleapis.com/auth/" + scope
    for scope in (
        "cloud-platform",
        "userinfo.email",
        "userinfo.profile",
        "cclog",
        "experimentsandconfigs",
    )
)
FALLBACK_PROJECT = "rising-fact-p41fc"


def pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    return verifier, challenge


def authorize_url(redirect: str, challenge: str, state: str) -> str:
    return (
        AUTHORIZE_URL
        + "?"
        + urlencode(
            {
                "client_id": CLIENT_ID,
                "redirect_uri": redirect,
                "response_type": "code",
                "scope": SCOPES,
                "access_type": "offline",
                "prompt": "consent",
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
    )


def extract_code(value: str, state: str) -> str:
    value = value.strip()
    if value.startswith(("http://", "https://")) or value.startswith("?"):
        query = parse_qs(urlsplit(value).query if not value.startswith("?") else value[1:])
        if not hmac.compare_digest(query.get("state", [""])[0], state):
            raise ValueError("OAuth state does not match this login attempt")
        if query.get("error"):
            raise ValueError("OAuth authorization was denied")
        value = query.get("code", [""])[0]
    if not value:
        raise ValueError("authorization code is empty")
    return value


class OAuth:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def tokens(self, **form) -> dict:
        response = await self.client.post(
            TOKEN_URL,
            data={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, **form},
            headers={"User-Agent": "google-api-nodejs-client/9.15.1"},
            timeout=30,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GatewayError(
                502, "oauth_error", "OAuth token endpoint returned invalid JSON"
            ) from exc
        if response.status_code >= 400 or not payload.get("access_token"):
            code = payload.get("error", "oauth_error")
            terminal = code in {"invalid_grant", "invalid_client", "unauthorized_client"}
            raise GatewayError(
                401 if terminal else 502, "oauth_" + code, f"OAuth token request failed ({code})"
            )
        return payload

    async def refresh(self, token: str) -> dict:
        return await self.tokens(grant_type="refresh_token", refresh_token=token)

    async def userinfo(self, token: str) -> dict:
        response = await self.client.get(
            USERINFO_URL,
            params={"alt": "json"},
            headers={"Authorization": "Bearer " + token},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()


def extract_id(value) -> str | None:
    return (
        value
        if isinstance(value, str) and value
        else value.get("id")
        if isinstance(value, dict)
        else None
    )


class Credentials:
    def __init__(self, transport: Transport, database: Database, endpoints: list[str]):
        self.transport, self.database, self.endpoints = transport, database, endpoints
        self.oauth = OAuth(transport.client)
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.locks: dict[str, asyncio.Lock] = {}
        self.epoch = 0

    def clear(self) -> None:
        self.cache.clear()
        self.epoch += 1

    def invalidate(self, identity: str) -> None:
        self.cache.pop(identity, None)

    async def prepare(self, account: Account) -> tuple[str, str]:
        lock = self.locks.setdefault(account.id, asyncio.Lock())
        try:
            async with lock:
                cached = self.cache.get(account.id)
                token_fresh = cached and cached["expires"] > time.monotonic() + 60
                if token_fresh and cached["project_until"] > time.monotonic():
                    self.cache.move_to_end(account.id)
                    return cached["token"], cached["project"]
                epoch = self.epoch
                tokens = None if token_fresh else await self.oauth.refresh(account.refresh_token)
                token = cached["token"] if token_fresh else tokens["access_token"]
                project, tier, paid = await self.discover(token, account)

                def persist(a: Account):
                    if project != FALLBACK_PROJECT:
                        a.project_id = project
                    a.state.update(tier=tier, paid_tier=paid, tier_at=now_ms())

                await asyncio.to_thread(self.database.update, account.id, persist)
                lifetime = tokens.get("expires_in") if tokens else None
                lifetime = lifetime if isinstance(lifetime, (int, float)) and lifetime > 0 else 3600
                if self.epoch == epoch:
                    self.cache[account.id] = {
                        "token": token,
                        "project": project,
                        "expires": cached["expires"]
                        if token_fresh
                        else time.monotonic() + lifetime,
                        "project_until": time.monotonic() + 1800,
                    }
                    while len(self.cache) > 512:
                        self.cache.popitem(last=False)
                return token, project
        finally:
            # Other waiters may still reference this lock. Only remove idle, unreferenced keys
            # through bounded cache cleanup; never replace a lock while waiters exist.
            if len(self.locks) > 1024:
                for key, item in list(self.locks.items()):
                    if (
                        key not in self.cache
                        and not item.locked()
                        and not getattr(item, "_waiters", None)
                    ):
                        self.locks.pop(key, None)

    async def discover(self, token: str, account: Account) -> tuple[str, str | None, str | None]:
        last_error = None
        for endpoint in self.endpoints:
            try:
                payload = await self.transport.json(
                    endpoint + "/v1internal:loadCodeAssist",
                    token,
                    {"metadata": {"ideType": "ANTIGRAVITY"}},
                )
                project = extract_id(payload.get("cloudaicompanionProject")) or account.project_id
                tier, paid = (
                    extract_id(payload.get("currentTier")),
                    extract_id(payload.get("paidTier")),
                )
                if project:
                    return project, tier, paid
                tiers = payload.get("allowedTiers", [])
                chosen = next(
                    (t for t in tiers if t.get("isDefault")),
                    tiers[0] if tiers else {"id": "free-tier"},
                )
                tier = chosen.get("id", "free-tier")
                for attempt in range(11):
                    if attempt:
                        await asyncio.sleep(5)
                    onboard = await self.transport.json(
                        endpoint + "/v1internal:onboardUser", token, {"tierId": tier}
                    )
                    managed = extract_id(
                        onboard.get("response", {}).get("cloudaicompanionProject")
                    ) or extract_id(onboard.get("cloudaicompanionProject"))
                    if managed:
                        return managed, tier, paid
                raise GatewayError(502, "project_error", "project onboarding did not complete")
            except GatewayError as exc:
                if exc.code in {
                    "verification_required",
                    "banned",
                    "account_ineligible",
                    "auth_invalid",
                }:
                    raise
                last_error = exc
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
        if account.project_id:
            return account.project_id, account.state.get("tier"), account.state.get("paid_tier")
        logging.getLogger(__name__).warning(
            "project discovery failed (%s); using the shared fallback project",
            type(last_error).__name__,
        )
        return FALLBACK_PROJECT, None, None


async def login(
    oauth: OAuth,
    *,
    no_browser: bool = False,
    no_open: bool = False,
    timeout: float = 300,  # noqa: ASYNC109 -- bounds the interactive login callback below
    ports: tuple[int, ...] = (51121, 51122, 0),
    prompt=print,
) -> tuple[str, str | None]:
    verifier, challenge = pkce()
    state = secrets.token_urlsafe(32)
    future = asyncio.get_running_loop().create_future()
    clients: set[asyncio.Task] = set()

    async def callback(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        task = asyncio.current_task()
        clients.add(task)
        try:
            async with asyncio.timeout(5):
                head = await reader.readuntil(b"\r\n\r\n")
            line = head.split(b"\r\n", 1)[0].decode("ascii")
            method, target, _ = line.split(" ", 2)
            parsed = urlsplit(target)
            code, status, text = None, "400 Bad Request", "Invalid OAuth callback."
            if method == "GET" and parsed.path == "/oauth-callback":
                query = parse_qs(parsed.query)
                if hmac.compare_digest(query.get("state", [""])[0], state):
                    if query.get("error"):
                        if not future.done():
                            future.set_exception(ValueError("OAuth authorization was denied"))
                    elif query.get("code"):
                        code = query["code"][0]
                        status, text = "200 OK", "Login completed. You may close this window."
            body = text.encode()
            writer.write(
                f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            if code and not future.done():
                future.set_result(code)
        except (
            ValueError,
            UnicodeError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            TimeoutError,
            ConnectionError,
        ):
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            clients.discard(task)

    server = None
    try:
        if not no_browser:
            for port in ports:
                try:
                    server = await asyncio.start_server(callback, "127.0.0.1", port, limit=8192)
                    break
                except OSError:
                    continue
            if server is None:
                raise ValueError("cannot bind an OAuth callback port; use --no-browser")
            port = server.sockets[0].getsockname()[1]
        else:
            port = ports[0]
        redirect = f"http://localhost:{port}/oauth-callback"
        url = authorize_url(redirect, challenge, state)
        prompt(url)
        if not no_open and not no_browser:
            await asyncio.to_thread(webbrowser.open, url)
        if no_browser:
            pasted = await asyncio.to_thread(input, "Paste the callback URL or code: ")
            code = extract_code(pasted, state)
        else:
            async with asyncio.timeout(timeout):
                code = await future
        tokens = await oauth.tokens(
            grant_type="authorization_code",
            code=code,
            code_verifier=verifier,
            redirect_uri=redirect,
        )
        if not tokens.get("refresh_token"):
            raise ValueError("Google returned no refresh token; authorize again with consent")
        try:
            identity = await oauth.userinfo(tokens["access_token"])
        except (httpx.HTTPError, ValueError):
            identity = {}
        return tokens["refresh_token"], identity.get("email")
    finally:
        if server:
            server.close()
            await server.wait_closed()
        for task in clients.copy():
            task.cancel()
        if clients:
            await asyncio.gather(*clients, return_exceptions=True)
