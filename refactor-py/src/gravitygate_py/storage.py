"""Transactional account state. Connections never span a network await."""

import hashlib
import json
import math
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Accounts


def now_ms() -> int:
    return time.time_ns() // 1_000_000


class StoreError(Exception):
    pass


class NotFound(StoreError):
    pass


class Busy(StoreError):
    pass


@dataclass
class Account:
    id: str
    refresh_token: str = field(repr=False)
    email: str | None = None
    project_id: str | None = None
    managed_project_id: str | None = None
    enabled: bool = True
    state: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.email or f"<{self.id[:8]}>"

    def running(self, now: int) -> bool:
        return bool(self.state.get("lease_id") and self.state.get("lease_until", 0) > now)

    def reset_holds(self) -> None:
        for key in (
            "verification_required",
            "verification_url",
            "verification_reason",
            "recheck_at",
            "cooldown_until",
            "cooldown_reason",
            "last_checked_at",
            "recovered_at",
        ):
            self.state.pop(key, None)
        self.state.update(
            holds={},
            revision=self.state.get("revision", 0) + 1,
            verification_failures=0,
            last_result="operator_reset",
            failures=0,
            health=100,
        )
        # Keep an active lease until the old check exits; its revision no longer matches.

    def verification(self, now: int | None = None) -> dict:
        now = now_ms() if now is None else now
        if not self.enabled or self.state.get("ineligible"):
            state = "blocked"
        elif self.running(now):
            state = "checking"
        elif self.state.get("verification_required"):
            state = "waiting"
        elif self.state.get("recovered_at"):
            state = "recovered"
        else:
            state = "unchecked"
        return {
            "state": state,
            "next_check_at": self.state.get("recheck_at"),
            "last_checked_at": self.state.get("last_checked_at"),
            "last_result": self.state.get("last_result"),
            "recovered_at": self.state.get("recovered_at"),
        }

    def public(self, now: int | None = None) -> dict:
        now = now_ms() if now is None else now
        detail, wait = None, None
        if not self.enabled:
            status, detail = "disabled", "removed from rotation by an operator"
        elif self.state.get("ineligible"):
            status, detail = "ineligible", self.state.get("ineligible_reason")
        elif self.running(now):
            status = "checking"
        elif self.state.get("verification_required"):
            status, detail = "verify", self.state.get("verification_reason")
        elif self.state.get("cooldown_until", 0) > now:
            status, detail = "cooling", self.state.get("cooldown_reason")
            wait = math.ceil((self.state["cooldown_until"] - now) / 1000)
        elif resets := [v for v in self.state.get("holds", {}).values() if v > now]:
            status, detail = "limited", "model-scoped limits; other model groups may be usable"
            wait = math.ceil((min(resets) - now) / 1000)
        else:
            status = "ready"
        return {
            "label": self.label,
            "credential_id": self.id[:8],
            "enabled": self.enabled,
            "status": status,
            "detail": detail,
            "verification_url": self.state.get("verification_url"),
            "verification": self.verification(now),
            "tier": self.state.get("tier"),
            "project_id": self.project_id,
            "available_in_seconds": wait,
        }


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise StoreError(f"unsupported database schema version: {version}")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY, refresh_token TEXT NOT NULL,
                    email TEXT, project_id TEXT, managed_project_id TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1, state TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS routing (pool TEXT PRIMARY KEY, account_id TEXT);
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY, ts INTEGER NOT NULL, account_id TEXT NOT NULL,
                    requested_model TEXT NOT NULL, wire_model TEXT NOT NULL, stream INTEGER NOT NULL,
                    status INTEGER NOT NULL, outcome TEXT NOT NULL, attempts INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL,
                    cached_tokens INTEGER NOT NULL, reasoning_tokens INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts);
                PRAGMA user_version=1;
            """)
        if os.name != "nt":
            os.chmod(self.path, 0o600)

    @contextmanager
    def connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _account(row: sqlite3.Row) -> Account:
        values = dict(row)
        values["enabled"] = bool(values["enabled"])
        values["state"] = json.loads(values["state"])
        return Account(**values)

    @staticmethod
    def _save(db: sqlite3.Connection, a: Account) -> None:
        db.execute(
            """INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET refresh_token=excluded.refresh_token,
            email=excluded.email, project_id=excluded.project_id,
            managed_project_id=excluded.managed_project_id, enabled=excluded.enabled,
            state=excluded.state""",
            (
                a.id,
                a.refresh_token,
                a.email,
                a.project_id,
                a.managed_project_id,
                a.enabled,
                json.dumps(a.state, separators=(",", ":")),
            ),
        )

    @staticmethod
    def _resolve(accounts: list[Account], selector: str) -> Account:
        if selector.isdecimal() and len(selector) < 8:
            index = int(selector)
            if index < len(accounts):
                return accounts[index]
        matches = [a for a in accounts if a.email == selector or a.id.startswith(selector)]
        if len(matches) != 1:
            raise NotFound("account selector does not identify exactly one account")
        return matches[0]

    def list(self) -> list[Account]:
        with self.connect() as db:
            return [
                self._account(row) for row in db.execute("SELECT * FROM accounts ORDER BY rowid")
            ]

    def get(self, selector: str) -> Account:
        return self._resolve(self.list(), selector)

    def update(self, selector: str, change: Callable[[Account], Any]) -> Any:
        with self.connect(write=True) as db:
            accounts = [
                self._account(r) for r in db.execute("SELECT * FROM accounts ORDER BY rowid")
            ]
            account = self._resolve(accounts, selector)
            result = change(account)
            self._save(db, account)
            return result

    def add(self, token: str, email: str | None = None, project: str | None = None) -> Account:
        parts = token.strip().split("|")
        if not parts[0]:
            raise StoreError("refresh token is empty")
        identity = hashlib.sha256(parts[0].encode()).hexdigest()[:24]
        with self.connect(write=True) as db:
            row = db.execute("SELECT * FROM accounts WHERE id=?", (identity,)).fetchone()
            account = self._account(row) if row else Account(identity, parts[0])
            account.email = email or account.email
            account.project_id = project or (
                parts[1] if len(parts) > 1 and parts[1] else account.project_id
            )
            account.managed_project_id = (
                parts[2] if len(parts) > 2 and parts[2] else account.managed_project_id
            )
            account.state.setdefault("added_at", now_ms())
            self._save(db, account)
            return account

    def remove(self, selector: str) -> None:
        with self.connect(write=True) as db:
            account = self._resolve(
                [self._account(r) for r in db.execute("SELECT * FROM accounts ORDER BY rowid")],
                selector,
            )
            db.execute("DELETE FROM accounts WHERE id=?", (account.id,))

    @staticmethod
    def blocked_until(
        a: Account, quota: str, now: int, *, recovery: bool = False, monitor: bool = True
    ) -> int | None:
        """None means an indefinite blocker; zero means eligible."""
        s = a.state
        if not a.enabled or s.get("ineligible") or a.running(now):
            return None
        if s.get("verification_required") and not recovery:
            if monitor:
                return None
            if s.get("recheck_at", 0) > now:
                return s["recheck_at"]
        holds = s.get("holds", {})
        return max(
            0, s.get("cooldown_until", 0), holds.get(quota, 0), holds.get(quota.split(":")[0], 0)
        )

    @staticmethod
    def bucket(a: Account, config: Accounts, now: int) -> float:
        s = a.state
        return min(
            config.token_bucket_max,
            s.get("tokens", config.token_bucket_max)
            + max(0, now - s.get("tokens_at", now)) * config.token_bucket_refill_per_min / 60000,
        )

    @staticmethod
    def health(a: Account, now: int) -> float:
        elapsed = max(0, now - a.state.get("health_at", now)) / 3600000
        return min(100, a.state.get("health", 70) + math.floor(elapsed * 2))

    def select(
        self,
        quota: str,
        config: Accounts,
        exclude: set[str],
        *,
        selector: str | None = None,
        monitor: bool = True,
    ) -> tuple[Account | None, int | None]:
        now = now_ms()
        with self.connect(write=True) as db:
            accounts = [
                self._account(r) for r in db.execute("SELECT * FROM accounts ORDER BY rowid")
            ]
            if selector:
                accounts = [self._resolve(accounts, selector)]
            candidates, waits = [], []
            for a in accounts:
                until = self.blocked_until(a, quota, now, monitor=monitor)
                if until is None:
                    continue
                tokens = self.bucket(a, config, now)
                if tokens < 1:
                    until = max(
                        until,
                        now + math.ceil((1 - tokens) * 60000 / config.token_bucket_refill_per_min),
                    )
                if until > now:
                    waits.append(until)
                elif a.id not in exclude:
                    candidates.append(a)
            if not candidates:
                return None, min(waits) if waits else None
            cursor = db.execute("SELECT account_id FROM routing WHERE pool=?", (quota,)).fetchone()
            previous = cursor[0] if cursor else None
            strategy = config.strategy
            if strategy == "round-robin":
                ids = [a.id for a in accounts]
                offset = (ids.index(previous) + 1) % len(ids) if previous in ids else 0
                chosen = min(candidates, key=lambda a: (ids.index(a.id) - offset) % len(ids))
            elif strategy == "sticky" and any(a.id == previous for a in candidates):
                chosen = next(a for a in candidates if a.id == previous)
            elif strategy in {"sticky", "least-recently-used"}:
                chosen = min(candidates, key=lambda a: a.state.get("last_used", 0))
            else:

                def score(a: Account) -> float:
                    idle = min(3600, max(0, now - a.state.get("last_used", 0)) / 1000)
                    return (
                        self.health(a, now) * 2
                        + self.bucket(a, config, now) / config.token_bucket_max * 500
                        + idle * 0.1
                    )

                chosen = max(candidates, key=score)
                incumbent = next((a for a in candidates if a.id == previous), None)
                if incumbent and score(chosen) - score(incumbent) < 100:
                    chosen = incumbent
            chosen.state.update(
                tokens=self.bucket(chosen, config, now) - 1, tokens_at=now, last_used=now
            )
            self._save(db, chosen)
            db.execute("INSERT OR REPLACE INTO routing VALUES (?, ?)", (quota, chosen.id))
            return chosen, None
