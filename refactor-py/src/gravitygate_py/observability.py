"""Bounded audit queue and low-cardinality Prometheus counters."""

import asyncio
import logging
import sqlite3
import time

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .config import Config
from .registry import base_for_wire
from .storage import Database, now_ms

log = logging.getLogger(__name__)
COLUMNS = (
    "ts",
    "account_id",
    "requested_model",
    "wire_model",
    "stream",
    "status",
    "outcome",
    "attempts",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "latency_ms",
)


class Observability:
    def __init__(self, database: Database, config: Config):
        self.database, self.config = database, config
        self.queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=4096)
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "gravitygate_requests_total",
            "Requests by model and outcome",
            ["model", "outcome"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "gravitygate_request_duration_seconds",
            "End-to-end client request duration",
            ["model"],
            registry=self.registry,
        )
        self.attempts = Histogram(
            "gravitygate_upstream_attempts",
            "Upstream attempts per request",
            ["model"],
            buckets=(0, 1, 2, 3, 5, 10, 20),
            registry=self.registry,
        )
        self.selected = Counter(
            "gravitygate_account_selected_total",
            "Requests dispatched per credential ID",
            ["account"],
            registry=self.registry,
        )
        self.adverse_events = Counter(
            "gravitygate_adverse_events_total",
            "Account and upstream problems",
            ["kind"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "gravitygate_tokens_total",
            "Reported tokens by kind",
            ["model", "kind"],
            registry=self.registry,
        )
        self.account_counts = Gauge(
            "gravitygate_accounts",
            "Accounts currently in each condition",
            ["condition"],
            registry=self.registry,
        )
        self.signature_count = Gauge(
            "gravitygate_signatures_cached",
            "Thinking signatures currently cached",
            registry=self.registry,
        )
        self.dropped_count = Counter(
            "gravitygate_audit_dropped_total", "Audit records dropped", registry=self.registry
        )
        self.dropped = 0
        self.writer: asyncio.Task | None = None

    def start(self) -> None:
        if self.config.audit.enabled:
            self.writer = asyncio.create_task(self._write(), name="gravitygate-audit")

    async def close(self) -> None:
        if self.writer:
            await self.queue.put(None)
            await self.writer

    async def flush(self) -> None:
        if self.writer:
            await self.queue.join()

    def record(self, *, run=None, status: int = 200, outcome: str = "ok") -> None:
        usage = run.acc.usage if run else {}
        record = {
            "ts": now_ms(),
            "account_id": run.account.id[:8] if run and run.account else "",
            "requested_model": run.request["model"] if run else "",
            "wire_model": run.resolved.wire_model if run else "",
            "stream": bool(run and run.request.get("stream")),
            "status": status,
            "outcome": outcome,
            "attempts": run.attempts if run else 0,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            "reasoning_tokens": usage.get("completion_tokens_details", {}).get(
                "reasoning_tokens", 0
            ),
            "latency_ms": int((time.monotonic() - run.started_at) * 1000) if run else 0,
        }
        spec = base_for_wire(record["wire_model"])
        model = spec.id if spec else "other"
        result = "ok" if status == 200 else "upstream_error" if status >= 500 else "client_error"
        self.requests_total.labels(model=model, outcome=result).inc()
        self.duration.labels(model=model).observe(record["latency_ms"] / 1000)
        self.attempts.labels(model=model).observe(record["attempts"])
        for kind in ("prompt", "completion", "cached", "reasoning"):
            self.tokens.labels(model=model, kind=kind).inc(record[kind + "_tokens"])
        if self.config.audit.enabled and self.writer:
            try:
                self.queue.put_nowait(record)
            except asyncio.QueueFull:
                self.dropped += 1
                self.dropped_count.inc()
                log.warning("audit queue full; dropping one record")

    def _insert(self, record: dict) -> None:
        with self.database.connect(write=True) as db:
            db.execute(
                f"INSERT INTO requests ({','.join(COLUMNS)}) VALUES ({','.join('?' for _ in COLUMNS)})",
                tuple(record[k] for k in COLUMNS),
            )

    async def _write(self) -> None:
        last_prune = 0
        while True:
            record = await self.queue.get()
            try:
                if record is None:
                    return
                await asyncio.to_thread(self._insert, record)
                if now_ms() - last_prune > 3600000:

                    def prune():
                        with self.database.connect(write=True) as db:
                            db.execute(
                                "DELETE FROM requests WHERE ts < ?",
                                (now_ms() - self.config.audit.retention_days * 86400000,),
                            )

                    await asyncio.to_thread(prune)
                    last_prune = now_ms()
            except sqlite3.Error:
                self.dropped += 1
                self.dropped_count.inc()
                log.exception("could not write an audit record")
            finally:
                self.queue.task_done()

    def recent(self, limit: int = 40) -> list[dict]:
        with self.database.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT " + ",".join(COLUMNS) + " FROM requests ORDER BY id DESC LIMIT ?",
                    (max(1, min(1000, limit)),),
                )
            ]

    def stats(self, *, accounts: bool = False) -> dict:
        with self.database.connect() as db:
            if accounts:
                rows = db.execute(
                    """SELECT account_id, count(*) AS requests,
                    sum(status != 200) AS errors, CAST(avg(latency_ms) AS INTEGER) AS mean_latency_ms
                    FROM requests WHERE ts >= ? GROUP BY account_id""",
                    (now_ms() - 3600000,),
                )
                return {"accounts": [dict(row) for row in rows]}
            row = dict(
                db.execute(
                    """SELECT count(*) AS requests, sum(status=200) AS ok,
                sum(status!=200) AS errors, sum(prompt_tokens) AS prompt_tokens,
                sum(completion_tokens) AS completion_tokens, sum(cached_tokens) AS cached_tokens,
                sum(reasoning_tokens) AS reasoning_tokens, CAST(avg(latency_ms) AS INTEGER) AS mean_latency_ms
                FROM requests WHERE ts >= ?""",
                    (now_ms() - 3600000,),
                ).fetchone()
            )
            return {key: value or 0 for key, value in row.items()}

    def account_selected(self, account) -> None:
        self.selected.labels(account=account.id[:8]).inc()

    def adverse(self, code: str) -> None:
        kind = {
            "quota_exhausted": "quota",
            "rate_limit_exceeded": "rate",
            "rate_limit_unknown": "rate",
            "model_capacity_exhausted": "capacity",
            "verification_required": "verification",
            "account_ineligible": "ineligibility",
            "banned": "ban",
            "network_error": "network",
        }.get(code, "other")
        self.adverse_events.labels(kind=kind).inc()

    def metrics(self, accounts: list[dict], signatures: int = 0) -> str:
        for state in (
            "ready",
            "disabled",
            "ineligible",
            "verify",
            "checking",
            "cooling",
            "limited",
        ):
            self.account_counts.labels(condition=state).set(
                sum(a["status"] == state for a in accounts)
            )
        self.signature_count.set(signatures)
        return generate_latest(self.registry).decode()
