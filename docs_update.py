p = "docs/progress.md"
s = open(p, encoding="utf-8").read()

def sub(old, new):
    global s
    if old not in s:
        raise SystemExit("not found: " + old[:80])
    s = s.replace(old, new, 1)

sub("Last updated: after bringing up the HTTP surface (M5).",
    "Last updated: after adding observability (M6).")

sub("| M6 | Observability | not started |",
    "| M6 | Observability | **done**, live-validated in a browser |")

sub('''gravitygate serve                    # the gateway itself, on 127.0.0.1:8080
```''',
'''gravitygate serve                    # the gateway itself, on 127.0.0.1:8080
```

Open <http://127.0.0.1:8080/> for the dashboard.''')

sub('''| `POST /refresh-token` | drops the in-memory caches |''',
'''| `POST /refresh-token` | drops the in-memory caches |
| `GET /metrics` | Prometheus text exposition |
| `GET /api/stats`, `/api/stats/accounts` | aggregates over the last hour |
| `GET /api/requests?limit=N` | the recent request log |
| `GET /` | the dashboard |''')

sub('''## Next, in order

1. **M6 — observability.** Prometheus metrics, the SQLite audit log, and the
   dashboard. All four were confirmed in scope at kickoff; only the status JSON
   endpoints exist.
2. **Live `fetchAvailableModels`.** `/v1/models` is built from the static
   catalogue. Merging the live list would make it authoritative about what an
   account can actually reach; the catalogue already supplies what the upstream
   does not report.
3. **A Claude tool loop.** Blocked on the thinking/tool-choice constraint above,
   not on the gateway.''',
'''## M6 — Observability

Added `src/observ/`: `metrics.rs` (Prometheus), `audit.rs` (SQLite), and the
dashboard in `src/server/dashboard.html`. All four observability items confirmed
at kickoff are now built.

### Three decisions

**Writes never block a request.** SQLite is synchronous, so audit records go
through a channel to a thread that owns the write connection. A request path that
waited on an fsync would trade latency for telemetry, which is the wrong trade.

**Losing a record is acceptable; failing a request is not.** A broken database or
a full channel degrades to a logged warning. Nor can either component stop the
gateway starting: one that refuses to boot because it cannot write a log file is
a worse outcome than one running without telemetry.

**The account is recorded by credential id, not by email.** Stable across a
rename, and it correlates against `/health` without scattering addresses through
a table.

### Metric cardinality

The `model` label is bounded to the catalogue, with anything unknown collapsing
to `other`. This matters more than it sounds: the model name is client-supplied,
so labelling with it directly would let one client mint unbounded series and take
down the metrics endpoint. The wire model is also mapped back to its catalogue
entry, so the four tiered spellings of one Flash model are one series rather than
four — which is what a dashboard wants anyway.

### What the browser caught

`curl` returning 200 with 10 KB of HTML proved nothing. Loading the page showed
`perAccount.map is not a function`: `/api/stats/accounts` returns
`{"accounts": [...]}` and the script treated it as an array. The stat cards
rendered, so the page looked alive, while both tables sat on "loading…" forever.

Fixed, and the renderers are now isolated per section so one malformed payload
cannot blank the rest of the page without saying which part failed.

### Verified live

- `/metrics` reports request counts by model and outcome, account conditions, and
  the signature cache size.
- `/api/stats` and `/api/stats/accounts` return real aggregates from SQLite:
  requests, error counts, token sums, and mean latency, per account.
- `/api/requests` returns the recent log with account, model, status, latency,
  token breakdown, and attempt count.
- The dashboard renders both accounts, the traffic cards, and the request table,
  confirmed by screenshot rather than by status code.

### Two bugs found, one by the compiler and one by the user asking

**A deadlock in `AuditLog::drop`.** It joined the writer thread before the sender
had been dropped, so the writer's `recv` never returned and the join blocked
forever. Every audit test hung rather than failed — the suite simply stopped
producing output, which is what prompted the question that found it. Fixed by
closing the channel before joining.

**A stale binary masked a fix.** After correcting the metric labels, live
verification still showed `other`, because `cargo test` had not refreshed
`target/debug/gravitygate.exe`. Worth remembering: `cargo build` before
believing a live result.

## Next, in order

1. **Live `fetchAvailableModels`.** `/v1/models` is built from the static
   catalogue. Merging the live list would make it authoritative about what an
   account can actually reach; the catalogue already supplies what the upstream
   does not report.
2. **A Claude tool loop.** Blocked on the thinking/tool-choice constraint above,
   not on the gateway.
3. **M7 — the OpenAI Responses API.** Phase two of the approved plan, and the
   only client protocol still missing.''')

open(p, "w", encoding="utf-8").write(s)
print("progress updated")

p = "README.md"
s = open(p, encoding="utf-8").read()
sub("| Metrics, audit log, dashboard | **not started** |",
    "| Metrics, audit log, dashboard | done, live-validated |")
sub('''gravitygate serve                    # the gateway, on 127.0.0.1:8080
```

Then point any OpenAI client at `http://127.0.0.1:8080/v1`.''',
'''gravitygate serve                    # the gateway, on 127.0.0.1:8080
```

Then point any OpenAI client at `http://127.0.0.1:8080/v1`, and open
<http://127.0.0.1:8080/> for the dashboard. Prometheus metrics are at `/metrics`.''')
sub('''  server/
    mod.rs                router, authentication, graceful shutdown''',
'''  observ/
    metrics.rs            Prometheus metrics and their label bounds
    audit.rs              the SQLite request log
  server/
    mod.rs                router, authentication, graceful shutdown
    dashboard.html        the operator UI, embedded''')
open(p, "w", encoding="utf-8").write(s)
print("readme updated")
