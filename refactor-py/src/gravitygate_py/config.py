"""Validated configuration; Python state is deliberately isolated from Rust."""

import os
import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, field_validator

ENDPOINTS = ["https://daily-cloudcode-pa.googleapis.com", "https://cloudcode-pa.googleapis.com"]


def data_dir() -> Path:
    if value := os.getenv("GRAVITYGATE_PY_DATA_DIR"):
        return Path(value).expanduser()
    if os.name == "nt" and (value := os.getenv("APPDATA")):
        return Path(value) / "gravitygate-py"
    return Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "gravitygate-py"


def config_path(path: Path | None = None) -> Path:
    return path or Path(os.getenv("GRAVITYGATE_PY_CONFIG_FILE", data_dir() / "config.toml"))


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Server(Section):
    host: str = "127.0.0.1"
    port: int = Field(default=18081, ge=1, le=65535)
    max_request_body_bytes: int = Field(default=16 * 1024 * 1024, gt=0)
    api_keys: list[str] = Field(default_factory=list, repr=False)


class Upstream(Section):
    endpoints: list[str] = Field(default_factory=lambda: ENDPOINTS.copy(), min_length=1)
    connect_timeout_secs: float = Field(default=10, gt=0)
    read_timeout_secs: float = Field(default=180, gt=0)
    request_jitter_max_ms: int = Field(default=0, ge=0)
    inject_agent_system_prompt: bool = False

    @field_validator("endpoints")
    @classmethod
    def validate_urls(cls, values: list[str]) -> list[str]:
        from urllib.parse import urlsplit

        for value in values:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("endpoints must be absolute HTTP(S) URLs")
        return [value.rstrip("/") for value in values]


class Accounts(Section):
    strategy: Literal["hybrid", "sticky", "round-robin", "least-recently-used"] = "hybrid"
    cooldown_secs: int = Field(default=60, ge=1)
    max_consecutive_failures: int = Field(default=3, ge=1)
    soft_quota_threshold: float = Field(default=0.2, ge=0, le=1)
    quota_refresh_secs: int = Field(default=1800, ge=1)
    token_bucket_max: float = Field(default=10, ge=1)
    token_bucket_refill_per_min: float = Field(default=6, gt=0)


class Routing(Section):
    max_wait_before_error_secs: float = Field(default=120, ge=0)
    max_account_attempts: int = Field(default=10, ge=1, le=100)
    max_capacity_retries: int = Field(default=5, ge=0, le=20)
    max_empty_response_retries: int = Field(default=2, ge=0, le=20)
    exhausted_error_mode: Literal["too-many-requests", "bad-request"] = "too-many-requests"


class Reasoning(Section):
    output_field: Literal["reasoning_content", "reasoning-content", "reasoning", "both"] = (
        "reasoning_content"
    )
    default_tier: Literal["minimal", "low", "medium", "high"] = "medium"


class Verification(Section):
    enabled: bool = True
    model: str = Field(default="gemini-3.8-flash-low", min_length=1)
    scan_interval_secs: float = Field(default=15, gt=0)
    timeout_secs: float = Field(default=60, gt=0, le=300)


class Audit(Section):
    enabled: bool = True
    retention_days: int = Field(default=30, ge=1)


class Metrics(Section):
    enabled: bool = True


class Logging(Section):
    level: Literal["debug", "info", "warning", "error"] = "info"


class Config(Section):
    server: Server = Field(default_factory=Server)
    upstream: Upstream = Field(default_factory=Upstream)
    accounts: Accounts = Field(default_factory=Accounts)
    routing: Routing = Field(default_factory=Routing)
    reasoning: Reasoning = Field(default_factory=Reasoning)
    verification: Verification = Field(default_factory=Verification)
    audit: Audit = Field(default_factory=Audit)
    metrics: Metrics = Field(default_factory=Metrics)
    logging: Logging = Field(default_factory=Logging)

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = config_path(path)
        raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        overrides = {
            "GG_HOST": ("server", "host"),
            "GG_PORT": ("server", "port"),
            "GG_LOG_LEVEL": ("logging", "level"),
            "GG_ACCOUNT_STRATEGY": ("accounts", "strategy"),
        }
        for env, (section, key) in overrides.items():
            if env in os.environ:
                raw.setdefault(section, {})[key] = os.environ[env]
        if "GG_API_KEYS" in os.environ:
            raw.setdefault("server", {})["api_keys"] = [
                key.strip() for key in os.environ["GG_API_KEYS"].split(",") if key.strip()
            ]
        return cls.model_validate(raw)

    def toml(self, *, redact: bool = False) -> str:
        raw = self.model_dump()
        if redact:
            raw["server"]["api_keys"] = ["<redacted>" for _ in self.server.api_keys]
        return tomli_w.dumps(raw)
