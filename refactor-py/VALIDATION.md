# Validation

Validated on Windows on 2026-10-22. The source and dependency lockfile are in
`refactor-py`; no real account files or Google quota were used.

| Check | Result |
| --- | --- |
| CPython 3.12.12 | 107 tests passed |
| CPython 3.14.0 | 107 tests passed |
| Offline Rust contracts | All 100 model/request/schema cases match |
| Ruff lint | Passed |
| Ruff format | All Python files formatted |
| Real loopback HTTP | Uvicorn → httpx → TCP upstream; incremental SSE verified |
| Distribution | Source archive and platform-independent Python wheel built |
| Wheel isolation | Imported directly from wheel; dashboard resource present; no Rust runtime files |

Coverage includes both OpenAI protocol routes, tool/signature round trips,
reasoning and token accounting, routing strategies, concurrent SQLite writes
and token consumption, OAuth refresh/PKCE callback/state/cleanup, upstream
classification and fallback, quota scoping, JSON errors and body limits,
authenticated administrative operations, metrics/audit, and verification
recovery under timeout, cancellation, duplicate checks, removal, resets and
new challenges.

The Rust contracts were exported from the existing implementation at revision
`c55a600`. Regeneration is optional and needs Rust; ordinary Python tests and
runtime do not. The contract exporter only invokes pure transformation code.

The verification timeout test warms the authentication cache before asserting
stream cleanup, so filesystem scheduling cannot turn that assertion into a
test of whether the upstream stream had started yet. OAuth cleanup assertions
inspect the listener directly instead of assuming Windows sends a TCP refusal
for closed ports.

Actual Google acceptance remains unverified for this Python transport. An
operator can explicitly import/login and run `gravitygate-py probe` for that
check. No live service or background process from validation is left running.
