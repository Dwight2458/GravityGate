# Python migration contract

The Python package is independent of the Rust runtime. All new source, tests,
tools, packaging and documentation live in this directory. The existing Rust
implementation and the user's existing account/config files are not modified.

Decisions confirmed by the user:

- Full functional migration, including CLI/OAuth, account routing and retry,
  both OpenAI protocols (streamed and buffered), verification recovery,
  audit/metrics and dashboard.
- New storage schema; direct file-format compatibility is not required.
- Python 3.12+, FastAPI, httpx, uv; command `gravitygate-py`.

Architecture:

- `config`: validated TOML and environment overrides, independent data directory.
- `storage`: SQLite WAL, short transactions, durable model-scoped holds,
  token buckets and verification leases; no network I/O inside transactions.
- `registry`: captured Rust model routes/limits, exact unknown-name passthrough.
- `protocol`: request normalization/schema sanitation and incremental stream
  translation with tool IDs, reasoning, signatures and usage.
- `upstream` / `oauth`: HTTP/1.1, ordered headers, session envelopes, PKCE,
  token single-flight, project discovery and live catalogue.
- `engine`: bounded account/endpoint retries, classification and cancellation.
- `verification`: generation-backed recovery, 1/2/5 minute backoff, persisted
  leases and revision checks to discard stale completions.
- `server` / `cli`: authenticated API, 16 MiB ingress budget, administrative
  operations, diagnostic probes and packaged dashboard.

Validation uses local fake upstreams and synthetic accounts. Test runs never
load real user credentials or consume Google quota. Browser login and live
Google calls require an explicit operator command. Python's transport is
verified locally; successful local tests do not establish Google acceptance.
