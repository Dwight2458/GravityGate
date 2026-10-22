import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime


@dataclass
class GatewayError(Exception):
    status: int
    code: str
    message: str
    retry_after: float | None = None
    verification_url: str | None = None

    def __str__(self) -> str:
        return self.message

    def envelope(self) -> dict:
        kind = (
            "invalid_request_error"
            if self.status in {400, 413, 415, 422}
            else "rate_limit_error"
            if self.status == 429
            else "server_error"
        )
        if self.status == 401:
            kind = "authentication_error"
        return {"error": {"message": self.message, "type": kind, "param": None, "code": self.code}}

    @property
    def headers(self) -> dict:
        return (
            {"Retry-After": str(max(1, math.ceil(self.retry_after)))}
            if self.retry_after is not None
            else {}
        )


def duration(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) and value > 0 else None
    if not isinstance(value, str):
        return None
    try:
        parsed = float(value.removesuffix("s"))
        return parsed if math.isfinite(parsed) and parsed > 0 else None
    except ValueError:
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", value)
        seconds = sum(
            float(v) * {"ms": 0.001, "h": 3600, "m": 60, "s": 1}[unit] for v, unit in matches
        )
        return seconds or None


def error_details(error: dict) -> list[dict]:
    details = error.get("details")
    return [item for item in details if isinstance(item, dict)] if isinstance(details, list) else []


def retry_hint(headers, payload: dict) -> float | None:
    if hint := duration(headers.get("retry-after-ms")):
        return max(2, hint / 1000)
    value = headers.get("retry-after")
    if hint := duration(value):
        return max(2, hint)
    if value:
        try:
            return max(2, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError):
            pass
    error = payload.get("error", payload)
    if not isinstance(error, dict):
        return None
    for item in error_details(error):
        metadata = item.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if hint := duration(item.get("retryDelay")):
            return max(2, hint)
        for key in ("quotaResetDelay", "retryDelay"):
            if hint := duration(metadata.get(key)):
                return max(2, hint)
        reset = metadata.get("quotaResetTime")
        if reset:
            try:
                return max(
                    2,
                    (
                        datetime.fromisoformat(reset.replace("Z", "+00:00")) - datetime.now(UTC)
                    ).total_seconds(),
                )
            except ValueError:
                pass
    return None


def classify(status: int, headers, body: str) -> GatewayError:
    try:
        payload = json.loads(body)
        payload = payload if isinstance(payload, dict) else {}
    except ValueError:
        payload = {}
    error = payload.get("error", {})
    error = error if isinstance(error, dict) else {}
    message = str(error.get("message") or f"upstream returned HTTP {status}")[:1000]
    lower = body.lower()
    hint = retry_hint(headers, payload)
    if status == 401:
        return GatewayError(401, "auth_invalid", message)
    if status == 403:
        if "has been disabled" in lower and "violation of terms of service" in lower:
            return GatewayError(403, "banned", message)
        if re.search(r"\baccount_ineligible\b", lower):
            return GatewayError(403, "account_ineligible", message)
        if any(
            marker in lower
            for marker in ("validation_required", "account_disabled", "user_disabled")
        ):
            urls = re.findall(r'https://[^\s"<>]+', body.replace("\\/", "/"))
            url = next((u for u in urls if u.startswith("https://accounts.google.com/")), None)
            return GatewayError(
                403,
                "verification_required",
                "account holder verification required",
                verification_url=url,
            )
        return GatewayError(403, "permission_denied", message)
    if status in {503, 529}:
        return GatewayError(503, "model_capacity_exhausted", message, hint)
    reasons = [str(d.get("reason", "")).upper() for d in error_details(error)]
    reasons.append(str(error.get("status", "")).upper())
    reason_map = {
        "QUOTA_EXHAUSTED": "quota_exhausted",
        "RATE_LIMIT_EXCEEDED": "rate_limit_exceeded",
        "MODEL_CAPACITY_EXHAUSTED": "model_capacity_exhausted",
        "INTERNAL": "upstream_server_error",
        "SERVER_ERROR": "upstream_server_error",
    }
    reason = next((reason_map[r] for r in reasons if r in reason_map), None)
    if not reason:
        for detail in error_details(error):
            if str(detail.get("@type", "")).endswith("QuotaFailure") and detail.get("violations"):
                text = json.dumps(detail["violations"]).lower()
                reason = (
                    "rate_limit_exceeded"
                    if any(s in text for s in ("perminute", "per minute", "per_minute"))
                    else "quota_exhausted"
                )
                break
    text = str(error.get("message", "")).lower()
    if not reason:
        if "capacity" in text or "overloaded" in text:
            reason = "model_capacity_exhausted"
        elif any(
            marker in text
            for marker in ("per minute", "rate limit", "rate_limit", "too many requests")
        ):
            reason = "rate_limit_exceeded"
        elif (
            "quota" in text
            and ("exhausted" in text or "exceeded" in text)
            and "e.g. check quota" not in text
        ):
            reason = "quota_exhausted"
    if status == 429 or (status == 400 and reason):
        return GatewayError(429, reason or "rate_limit_unknown", message, hint)
    if status == 404:
        return GatewayError(404, "model_not_found", message)
    if status >= 500:
        return GatewayError(502, "upstream_server_error", message, hint)
    return GatewayError(status if 400 <= status < 500 else 502, "upstream_bad_request", message)


def backoff(error: GatewayError, failures: int = 1) -> float:
    if error.retry_after:
        return max(2, error.retry_after)
    if error.code == "quota_exhausted":
        return (60, 300, 1800, 7200)[min(max(0, failures - 1), 3)]
    return {
        "rate_limit_exceeded": 30,
        "rate_limit_unknown": 60,
        "model_capacity_exhausted": 45,
        "upstream_server_error": 20,
        "network_error": 20,
    }.get(error.code, 60)
