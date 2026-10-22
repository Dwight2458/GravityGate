"""Operator CLI. No credential or live service is touched at import time."""

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from pathlib import Path
from urllib.parse import quote

import httpx
import uvicorn
from pydantic import ValidationError

from . import __version__
from .config import Config, config_path, data_dir
from .engine import Engine
from .errors import GatewayError
from .oauth import login
from .request import text_content
from .server import create_app
from .storage import Account, Database, StoreError, now_ms


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="gravitygate-py", description="OpenAI-compatible Antigravity gateway (Python)"
    )
    root.add_argument("--version", action="version", version=f"gravitygate-py {__version__}")
    root.add_argument("--config", type=Path, help="TOML configuration file")
    root.add_argument("--data-dir", type=Path, help="isolated directory containing gravitygate.db")
    root.add_argument("-v", "--verbose", action="count", default=0)
    commands = root.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the gateway")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    config = commands.add_parser("config", help="inspect or initialize configuration")
    config_actions = config.add_subparsers(dest="operation", required=True)
    config_actions.add_parser("path")
    config_actions.add_parser("show")
    config_actions.add_parser("init")
    account = commands.add_parser("account", help="manage accounts")
    actions = account.add_subparsers(dest="operation", required=True)
    listing = actions.add_parser("list")
    listing.add_argument("--json", action="store_true")
    add = actions.add_parser("add-token")
    add.add_argument("token", nargs="?")
    add.add_argument("--stdin", action="store_true", help="read the refresh token from stdin")
    add.add_argument("--email")
    add.add_argument("--project")
    for operation in ("remove", "enable", "disable", "clear-holds", "recheck"):
        action = actions.add_parser(operation)
        action.add_argument("selector", help="email, credential ID prefix or zero-based index")
    verify = actions.add_parser(
        "verify", help="check OAuth/project discovery; does not prove generation recovery"
    )
    verify.add_argument("selector", nargs="?")
    auth = actions.add_parser("login")
    auth.add_argument("--no-browser", action="store_true")
    auth.add_argument("--no-open", action="store_true")
    auth.add_argument("--timeout", type=float, default=300)
    auth.add_argument("--email")
    auth.add_argument("--project")
    auth.add_argument("--no-verify", action="store_true")
    migrate = actions.add_parser(
        "import-rust", help="explicitly import a Rust accounts.json into the new schema"
    )
    migrate.add_argument("file", type=Path)
    probe = commands.add_parser(
        "probe", help="make a real upstream request through the production engine"
    )
    probe.add_argument("--model", default="gemini-3.8-flash")
    probe.add_argument("--account")
    probe.add_argument("--prompt", default="Reply with exactly: ok")
    probe.add_argument("--repeat", type=int, default=1)
    probe.add_argument("--tool", action="store_true")
    probe.add_argument("--raw", action="store_true")
    models = commands.add_parser("models", help="list static or live models")
    models.add_argument(
        "--live", action="store_true", help="query account model availability over the network"
    )
    return root


def service_url(config: Config) -> str:
    host = config.server.host
    host = "127.0.0.1" if host in {"0.0.0.0", "*"} else "::1" if host == "::" else host
    if ":" in host and not host.startswith("["):
        host = "[" + host + "]"
    return f"http://{host}:{config.server.port}"


def import_rust(database: Database, source: Path) -> int:
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    rows = payload.get("accounts") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("accounts.json must contain an accounts array")
    if any(
        not isinstance(r, dict)
        or not isinstance(r.get("refresh_token"), str)
        or not r["refresh_token"].split("|")[0]
        for r in rows
    ):
        raise ValueError("every imported account must have a nonempty refresh_token")
    # Validate everything before mutation. Import all accounts in one transaction.
    import hashlib

    with database.connect(write=True) as db:
        for row in rows:
            token, *parts = row["refresh_token"].split("|")
            identity = hashlib.sha256(token.encode()).hexdigest()[:24]
            if db.execute("SELECT 1 FROM accounts WHERE id=?", (identity,)).fetchone():
                continue
            state = {
                "added_at": row.get("added_at", now_ms()),
                "last_used": row.get("last_used", 0),
                "holds": row.get("rate_limit_reset_times", {}),
                "verification_required": bool(row.get("verification_required")),
                "verification_url": row.get("verification_url"),
                "verification_reason": row.get("verification_required_reason"),
                "recheck_at": min(
                    row.get("verification_recheck_at") or now_ms(), now_ms() + 300000
                ),
                "ineligible": bool(row.get("account_ineligible")),
                "ineligible_reason": row.get("account_ineligible_reason"),
                "tier": row.get("captured_tier_id"),
                "paid_tier": row.get("captured_paid_tier_id"),
                "cooldown_until": row.get("cooling_down_until") or 0,
                "cooldown_reason": row.get("cooldown_reason"),
                "revision": 0,
            }
            account = Account(
                identity,
                token,
                row.get("email"),
                row.get("project_id") or (parts[0] if parts else None),
                row.get("managed_project_id") or (parts[1] if len(parts) > 1 else None),
                bool(row.get("enabled", True)),
                state,
            )
            database._save(db, account)
    return len(rows)


async def execute(args, config: Config, database: Database) -> int:
    if args.command == "account":
        operation = args.operation
        if operation == "list":
            accounts = database.list()
            if args.json:
                print(json.dumps([a.public() for a in accounts], ensure_ascii=False, indent=2))
            else:
                print(f"{'#':<4} {'ACCOUNT':<32} {'ID':<10} {'STATUS':<12} DETAIL")
                for i, account in enumerate(accounts):
                    public = account.public()
                    view = public["verification"]
                    detail = public.get("detail") or ""
                    if view["state"] == "waiting":
                        seconds = max(0, ((view["next_check_at"] or now_ms()) - now_ms()) // 1000)
                        detail += f"; recheck in {seconds}s" if seconds else "; recheck due"
                    if view.get("last_result"):
                        detail += f"; last={view['last_result']}"
                    if view.get("recovered_at"):
                        detail += (
                            f"; verified {max(0, now_ms() - view['recovered_at']) // 1000}s ago"
                        )
                    if public.get("verification_url"):
                        detail += "; " + public["verification_url"]
                    print(
                        f"{i:<4} {account.label:<32} {account.id[:8]:<10} {public['status']:<12} {detail.lstrip('; ')}"
                    )
            return 0
        if operation == "add-token":
            token = sys.stdin.read().strip() if args.stdin else args.token
            if not token:
                raise ValueError("supply a token or --stdin")
            account = database.add(token, args.email, args.project)
            print(f"Added {account.label} ({account.id[:8]}).")
            return 0
        if operation == "import-rust":
            count = import_rust(database, args.file)
            print(
                f"Processed {count} account(s); existing accounts were preserved. Database: {database.path}"
            )
            return 0
        if operation == "remove":
            account = database.get(args.selector)
            database.remove(account.id)
            print(f"Removed {account.label}.")
            return 0
        if operation in {"enable", "disable", "clear-holds"}:

            def update(a: Account):
                if operation == "clear-holds":
                    a.reset_holds()
                else:
                    a.enabled = operation == "enable"
                    a.state["revision"] = a.state.get("revision", 0) + 1

            account = database.get(args.selector)
            database.update(account.id, update)
            print(f"{operation}: {account.label}.")
            return 0
        if operation == "recheck":
            account = database.get(args.selector)
            headers = (
                {"Authorization": "Bearer " + config.server.api_keys[0]}
                if config.server.api_keys
                else {}
            )
            async with httpx.AsyncClient(
                timeout=config.verification.timeout_secs + 15, trust_env=False
            ) as client:
                try:
                    response = await client.post(
                        service_url(config)
                        + "/api/accounts/"
                        + quote(account.id, safe="")
                        + "/recheck",
                        headers=headers,
                    )
                except httpx.HTTPError as exc:
                    raise ValueError(
                        "could not reach the Python service; start gravitygate-py serve with the same config/data directory"
                    ) from exc
                payload = response.json()
                if response.status_code >= 400:
                    raise ValueError(
                        payload.get("error", {}).get("message", "recheck request failed")
                    )
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 0 if payload.get("verification", {}).get("state") == "recovered" else 1
    if args.command == "models" and not args.live:
        from .registry import model_list

        print("\n".join(row["id"] for row in model_list()))
        return 0
    engine = Engine(config, database)
    try:
        if args.command == "account" and args.operation == "login":
            if args.timeout <= 0:
                raise ValueError("timeout must be positive")
            token, email = await login(
                engine.credentials.oauth,
                no_browser=args.no_browser,
                no_open=args.no_open,
                timeout=args.timeout,
            )
            account = database.add(token, args.email or email, args.project)
            print(f"Added {account.label} ({account.id[:8]}).")
            if not args.no_verify:
                await engine.credentials.prepare(account)
                print("OAuth and project discovery succeeded.")
            return 0
        if args.command == "account" and args.operation == "verify":
            accounts = [database.get(args.selector)] if args.selector else database.list()
            failed = False
            for account in accounts:
                try:
                    _, project = await engine.credentials.prepare(account)
                    print(
                        f"{account.label}: OAuth/project OK ({project}); use account recheck to confirm generation."
                    )
                except GatewayError as exc:
                    failed = True
                    print(f"{account.label}: {exc.code}", file=sys.stderr)
            return int(failed)
        if args.command == "models":
            from .registry import model_list

            pools = await engine.available_models()
            print("\n".join(row["id"] for row in model_list(next(iter(pools.values()), {}))))
            return 0
        if args.command == "probe":
            if not 1 <= args.repeat <= 100:
                raise ValueError("repeat must be between 1 and 100")
            for _ in range(args.repeat):
                request = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": args.prompt}],
                }
                if args.tool:
                    request.update(
                        tools=[
                            {
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "description": "Echo a message",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"message": {"type": "string"}},
                                        "required": ["message"],
                                    },
                                },
                            }
                        ],
                        tool_choice="required",
                    )
                run = engine.execution(request, args.account)
                run.capture_raw = args.raw
                try:
                    await run.drain()
                finally:
                    await run.close()
                print(
                    f"model     : {args.model}\nresolved  : {run.resolved.wire_model} (tier {run.resolved.tier})"
                )
                print(
                    f"served by : {run.account.label} ({run.account.id[:8]})\nproject   : {run.project}"
                )
                print(
                    f"endpoint  : {run.endpoint}\ntraceId   : {run.trace_id or 'unavailable'}\nresult    : 200 OK ({run.attempts} attempt(s))"
                )
                print(
                    f"answer    : {text_content(run.acc.completion()['choices'][0]['message'].get('content'))}"
                )
                print(
                    f"finish    : {run.acc.finish_reason}\nusage     : {json.dumps(run.acc.usage)}\nsignatures: {len(engine.signatures)} cached"
                )
                if run.acc.reasoning:
                    print("reasoning : " + "".join(run.acc.reasoning))
                if run.acc.tools:
                    print("tool_calls: " + json.dumps(run.acc.tools, ensure_ascii=False))
                if args.raw:
                    print("\n--- raw response ---")
                    for event in run.raw_events:
                        print("data: " + json.dumps(event, ensure_ascii=False))
            return 0
    finally:
        await engine.close()
    return 0


def main() -> None:
    args = parser().parse_args()
    if args.data_dir:
        import os

        os.environ["GRAVITYGATE_PY_DATA_DIR"] = str(args.data_dir)
    try:
        config = Config.load(args.config)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else config.logging.level.upper(),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        if args.command == "config":
            path = config_path(args.config)
            if args.operation == "path":
                print(
                    f"config: {path}\ndata: {data_dir()}\ndatabase: {data_dir() / 'gravitygate.db'}"
                )
            elif args.operation == "show":
                print(config.toml(redact=True), end="")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as output:
                    output.write(config.toml())
                print(f"Created {path}")
            return
        database = Database(data_dir() / "gravitygate.db")
        if args.command == "serve":
            if args.host:
                config.server.host = args.host
            if args.port is not None:
                config.server.port = args.port
            uvicorn.run(
                create_app(config, database),
                host=config.server.host,
                port=config.server.port,
                log_level=config.logging.level,
                server_header=False,
            )
            return
        code = asyncio.run(execute(args, config, database))
    except ValidationError as exc:
        details = "; ".join(
            ".".join(str(k) for k in e["loc"]) + ": " + e["msg"]
            for e in exc.errors(include_input=False, include_context=False)
        )
        print(f"error: invalid configuration or request: {details}", file=sys.stderr)
        code = 1
    except (
        ValueError,
        OSError,
        StoreError,
        GatewayError,
        httpx.HTTPError,
        TimeoutError,
        sqlite3.Error,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)
