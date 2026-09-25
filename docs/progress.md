# Development progress

Running record of what is built, what was verified, and what is next. Updated as
each milestone lands. The requirements plan this executes against is the approved
plan from the kickoff; this file is the state of play.

Last updated: after diagnosing a 404 that turned out to be a stale binary.

## At a glance

| Milestone | Scope | State |
|---|---|---|
| M0 | Transport feasibility | **done and closed against a live account** |
| M1 | Upstream connectivity backbone | **done**, verified against a live account |
| M2 | Account pool and routing | **done**, live-validated across two accounts |
| M3 | Streaming, thinking, tool calls | **done** — both paths, live-validated |
| M4 | OAuth browser login | **done** |
| M5 | HTTP surface (`serve`, `/v1/chat/completions`) | **done**, live-validated |
| M6 | Observability | **done**, live-validated in a browser |
| M7 | OpenAI Responses (phase two) | **done**, live-validated |

## What runs today

```sh
cargo build --release

gravitygate account login            # browser sign-in
gravitygate account add-token        # from an existing refresh token
gravitygate account list             # status table
gravitygate account verify           # authenticate + resolve project
gravitygate probe                    # one real upstream request, fully reported
gravitygate probe --repeat 4         # routing across several requests in one process
gravitygate probe --tool             # force a tool call, exercising the tool path
gravitygate config path | show | init

gravitygate serve                    # the gateway itself, on 127.0.0.1:8080
```

Open <http://127.0.0.1:8080/> for the dashboard.

## What the gateway serves

| Route | |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible, streaming and buffered |
| `POST /v1/responses` | the Responses API, streaming and buffered |
| `GET /v1/models` | the static catalogue merged with the live list |
| `GET /health` | per-account condition, with reset timers and verification URLs |
| `GET /account-limits` | quota matrix |
| `POST /refresh-token` | drops the in-memory caches |
| `GET /metrics` | Prometheus text exposition |
| `GET /api/stats`, `/api/stats/accounts` | aggregates over the last hour |
| `GET /api/requests?limit=N` | the recent request log |
| `GET /` | the dashboard |

Client auth is optional and off by default, which suits a loopback deployment.
Set `GG_API_KEYS` or `server.api_keys` and the gateway requires
`Authorization: Bearer` or `x-api-key`; comparison is constant-time.

## M4 — OAuth login

Added `src/oauth/{pkce,callback,login}.rs` and `gravitygate account login`.

The flow is: bind the loopback listener, generate PKCE and state, build the
authorization URL, open a browser, wait for the redirect, exchange the code, read
the account identity, save the account, then optionally check it against the
upstream.

### Decisions worth remembering

**The listener binds before the URL is built.** The redirect URI carries the port,
so the port has to be settled first. The alternative — assume 51121, discover it
is taken, rebuild — means the URL already handed to the user is wrong.

**Ports 51121 through 51126 are tried in order.** 51121 is sometimes inside a
range reserved by Hyper-V, WSL2, or Docker on Windows. Google's loopback redirect
handling permits a varying port, so falling back works rather than merely being
optimistic.

**Both loopback families are bound.** On Windows `localhost` frequently resolves
to `::1` first. A listener on `127.0.0.1` alone leaves the browser connecting to a
closed port, which the user experiences as a hang, not an error. The listener binds
`127.0.0.1` and `::1` and succeeds if either works.

**Non-callback requests are answered and ignored.** Browsers request
`/favicon.ico`. Treating that as the callback would consume the one wait.

**`access_type=offline` and `prompt=consent` together.** Offline is what asks for a
refresh token at all; consent is what makes Google reissue one for an account that
has already granted access, so re-running login still produces a usable token.

**Browser launching avoids the shell.** On Windows it uses
`rundll32 url.dll,FileProtocolHandler`, not `cmd /C start`. Routing through `cmd`
would let it reinterpret the `&` separators in the query string as command
separators and silently truncate the URL.

**A failed post-login check does not roll back the account.** A working refresh
token is worth more than a clean exit code.

### Verified

- `--no-browser` prints a well-formed authorization URL and fails cleanly on empty
  input, exit code 1.
- Listener mode binds 51121, waits, and times out cleanly.
- A callback delivered over real HTTP to the running listener returns the success
  page and the flow proceeds to the token exchange, where Google correctly rejects
  a fabricated code with `invalid_grant`.
- 292 tests, clippy clean.

### Not verified

No real Google account has been authorized through this flow yet. Everything up to
the token exchange is proven; the exchange itself is proven only to the point of
Google rejecting a fake code.

## M0 — Transport feasibility — CLOSED

Question: `antigravity-auth` hand-rolls HTTP/1.1 over `tls.connect` instead of
using the platform client, citing header ordering. Does Rust need the same?

Answer: **no.** `http::HeaderMap` iterates in insertion order and hyper serialises
in iteration order, so header order is fully controllable. Streamed bodies produce
`Transfer-Encoding: chunked` as the capture shows. `Accept-Encoding` can be pinned
to exactly `gzip`.

The three deviations from the captured CLI request — lowercase header names, an
injected `accept: */*`, and `host` plus framing headers last — are all tolerated.
Resolved against a live account:

- `oauth2.googleapis.com` accepted the token exchange.
- `cloudcode-pa.googleapis.com` accepted `loadCodeAssist` and returned a real
  project (`aicode-consumers`, not the fallback).
- `daily-cloudcode-pa.googleapis.com` accepted `:streamGenerateContent` and
  returned 200 with a well-formed SSE body.

No hand-rolled HTTP/1.1 writer is needed. Detail in `docs/m0-transport-findings.md`.

### What live traffic revealed

Beyond closing the header question, real responses corrected four assumptions.
All four are things a mocked fixture would not have surfaced, which is the
argument for having done this early.

**Thinking is charged against `maxOutputTokens`, on Gemini too.** A probe with a
64-token limit came back with `thoughtsTokenCount: 61`, an empty text part, and
`finishReason: MAX_TOKENS`. The guard for this existed in the Claude transform
only, because that is where the reference implementations noticed it. It is not
family-specific. `build_output_budget` now applies to every family, and the probe
uses 1024 rather than 64.

**A signature can arrive on a part with empty text and no `thought` flag.**
The observed part was `{"thoughtSignature": "EusCCugCARFN...", "text": ""}` — no
thinking text, no `thought: true`, but a 488-character signature. The signature
cache must therefore not assume a signature implies thinking content, nor that
thinking content carries the signature. Signatures need collecting from any part
that has one.

**`candidatesTokenCount` can be absent.** One usage block carried only
`promptTokenCount`, `totalTokenCount`, and `thoughtsTokenCount` — with the
candidate count missing entirely. Since `totalTokenCount` includes thinking,
`usage.completion_tokens` can be neither read directly nor naively derived. It
has to be handled explicitly when the OpenAI response translation is written.

**A single response arrives as multiple SSE events, split by content kind.**
This is the most consequential finding so far, and it was invisible until the
text actually came through. A live `gemini-3.8-flash` response was:

```text
data: {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "usageMetadata": {...}}}

data: {"response": {"candidates": [{"content": {"parts": [{"thoughtSignature": "EvYDCvMDARFN...", "text": ""}]}, "finishReason": "STOP"}], "usageMetadata": {...}}}
```

The answer text is in the first event; the signature and the finish reason are in
the second, attached to an *empty* text part. Reading only the first event loses
the signature and the finish reason; reading only the last loses the answer. Both
mistakes are silent — you get a plausible-looking response with something
important missing.

The probe originally read only the last event, on the reasoning that usage
metadata arrives last. It does, but so does the signature, and the text does not.
`merge_sse_events` now accumulates parts across events while taking usage, finish
reason, and model version from the newest.

This directly shapes the streaming translation that has not been written yet. A
streaming translator cannot assume one event maps to one content block: it must
hold the preceding block open until it knows whether a trailing signature-only
event belongs to it, and it must not treat an empty-text part as "nothing to
emit" — that part is where the signature lives.

The envelope also carried a `metadata: {}` field that neither reference project
documents. Harmless, but it means the wrapper has more fields than we knew about.

Latency across the first three requests was 19.8s, 25.2s, and 32.9s. Rising, but
still too few samples to call it a trend, and the account is on free tier where
queuing is plausible. Worth watching before it is mistaken for a gateway problem.

## M1 — Upstream connectivity

`src/upstream/` (constants, envelope, metadata, transport), `src/oauth/token.rs`,
`src/accounts/project.rs`, `src/config.rs`, `src/cli.rs`, `src/engine/`.

- Wire identity, envelope field order, and session metadata reproduced from the
  captured `agy` CLI 1.1.24 request.
- Errors the reference implementations get wrong were avoided deliberately: no
  random per-request project id (fragments prompt caching and quota accounting),
  no synthetic account state.
- Project discovery: `loadCodeAssist` then `onboardUser` polling, cached 30
  minutes, falling back to the shared project rather than inventing one.

## M2 — Account pool

**Done:** the account model and its persistence. `src/accounts/account.rs` carries
the full state vocabulary — per-pool rate-limit expiries, cooldowns with reasons,
verification holds with URLs, ineligibility, cached tier. `src/accounts/store.rs`
does atomic writes with an advisory lock file, staleness detection, and fail-closed
parsing.

Plaintext storage with `0700`/`0600` permissions, per the approved plan. The
refresh token is redacted from `Debug` so it cannot reach a log line.

**Not done:** the hybrid router itself — token bucket, health score, LRU
combination, and the retry state machine. This is the largest remaining piece of
M2 and blocks multi-account operation.

## M3 — Translation

**Done:** the model registry with tier routing, the OpenAI request types, the
OpenAI → IR translation, the JSON Schema sanitiser, and the thinking-aware output
budget.

Three real bugs were caught and fixed, all worth remembering:

- `allOf` merging used first-wins for `properties`, silently dropping every
  property after the first member.
- Model resolution split the tier suffix before looking up the catalogue, so
  `gpt-oss-120b-medium` lost its `-medium` and with it its thinking budget. The fix
  is exact-match-first, then suffix-split.
- The thinking-aware output budget was applied to Claude only. Found by the first
  live request, not by a test — see the M0 section.

**Not done:** upstream SSE parsing, IR → OpenAI response and chunk emission, and
the thinking signature cache. The signature cache is the piece with no prior art
to copy — OpenAI's protocol has nowhere to carry a signature, so the gateway must
maintain `tool_call_id → signature` itself.

## M3 — Response path

Added `src/upstream/sse.rs`, `src/transform/signature_cache.rs`,
`src/transform/response.rs`, and rewrote the probe to run the production pipeline
rather than a simplified one. A healthy `probe` is now evidence that the real
path is healthy, not just that the transport works.

`probe` also gained `--tool`, which sends a dummy tool with
`tool_choice: "required"`. That makes the tool-call path deterministic to
exercise instead of depending on whether the model felt like calling something.

### The signature cache

The subsystem with no prior art. The upstream requires a conversation's prior
thinking to be returned with its signature; Anthropic's protocol has
`signature_delta` and `thinking.signature` to carry one, and OpenAI's has
nothing — `tool_calls` is an id, a name, and a JSON string, and clients drop
unknown fields when echoing a turn back. So the gateway keeps the signature
itself, keyed by the tool call id it handed the client.

Two maps, both TTL'd at two hours: tool call id, and session. Family-scoped,
because a signature from the wrong family is rejected upstream with an opaque
error. Signatures under 50 characters are treated as absent — real ones run to
hundreds.

### What live traffic corrected, again

Three findings, all of which were wrong in code that passed its tests.

**The `thought` flag is not a reliable discriminator.** The Gemini signature
arrived on `{"text": "", "thoughtSignature": "..."}` with no `thought` flag, so
it was classified as a signature carrier. The Claude signature arrived on
`{"thought": true, "text": "", "thoughtSignature": "..."}` *with* one — and the
classifier, testing `thought` first, called it reasoning and emitted an empty
reasoning delta for every signature. The symptom was a chunk count of 5 where 3
was correct, on a six-event Claude stream. Fixed by testing emptiness before the
`thought` flag: empty text is never content, whatever else the part carries. The
regression test uses the captured six-event sequence verbatim.

**The upstream does supply a `functionCall.id`.** The assumption was that it
usually omits one, so ids were always generated. Live traffic sent
`{"name": "get_weather", "args": {...}, "id": "call_9680"}`. The upstream id is
now preferred, because the backend may correlate it with the signature issued in
the same part and substituting our own would break a link we cannot see. Reusing
it is safe precisely because the client echoes back whichever id it was given.

**That id is a sibling of `name` and `args`, not an entry inside `args`.** The
Claude path had been inserting the id into `args`, which would have produced a
tool call whose arguments contained a stray `id` key. It is now a proper field on
`FunctionCall`.

A fourth finding is about ordering rather than schema: a Claude stream for a
one-word answer was six events — an empty text part, the thinking text, an empty
thinking part, the signature, the answer, then an empty part carrying the finish
reason. Usage advanced across them (`candidatesTokenCount` 1 then 13), which
confirms newest-wins accumulation is the right rule and not just a Gemini quirk.

### Verified

- Live `gemini-3.8-flash`: 200, answer `ok`, signature captured, usage derived
  (`prompt=6 completion=1`), two-event split as documented.
- Live `claude-opus-4-6-thinking`: 200, reasoning and answer both `ok`, 448-char
  signature captured, snake_case thinking config accepted.
- Live tool call via `--tool`: `get_weather({"city":"Paris"})`, upstream id
  preserved, exactly one signature captured, `finish_reason: tool_calls`
  downstream against `STOP` upstream.
- 414 tests, clippy clean.

### Not verified

The signature cache has not been exercised across two turns on a live account.
Capture is proven; replay is proven only against tests. That needs a second turn
carrying the first turn's tool call back, which the CLI cannot yet do.

## M2 — Routing and retry

Added `src/accounts/ratelimit.rs`, `src/accounts/router.rs`,
`src/engine/retry.rs`, and `src/engine/dispatch.rs`.

### The shape

The decision is a pure function and the I/O is a thin driver around it. `decide`
maps a classified failure plus the attempt history to a `Step` — try the next
endpoint, retry the same one after a delay, rotate accounts, or give up — and the
dispatch loop only executes steps. Everything easy to get wrong is in the half
that can be tested without a network.

The distinction the policy is built on, because getting it wrong is expensive in
both directions: **capacity exhaustion is the model's problem, quota exhaustion
is the account's.** A busy model means retry the same account shortly; rotating
for it spends another account for nothing. A spent quota means the same account
cannot succeed, so rotating is the only useful move.

### What was built

`ratelimit.rs` classifies any non-2xx into a rate limit (four kinds, each with
its own backoff shape), an account problem (verification, ineligibility, ban,
dead credential), or a transport-class fault. It reads reset hints from
`Retry-After`, `retry-after-ms`, the structured `RetryInfo.retryDelay`, and
prose, in that order of trustworthiness.

`router.rs` holds the two pieces of per-account runtime state: a token bucket
that is the client-side throttle, and a health score that decays on failure and
recovers with rest. Both are keyed by credential id, so removing an account
cannot hand its history to whatever slides into its index. Four strategies:
hybrid (score, with the incumbent protected), sticky, round-robin, LRU.

`dispatch.rs` runs the loop. Translation happens once, outside it, because the IR
is account-independent; only the envelope is rebuilt per attempt, since the
project id is not. Retries happen on the response head and never mid-body,
because a stream that has already delivered bytes cannot be retried — which is
why the loop returns a `StreamingResponse` rather than a decoded result.

### Verified live

Two accounts, four requests each, one process:

| Strategy | Accounts used | Expected |
|---|---|---|
| hybrid | 1 | 1 — stickiness protects the prompt cache |
| sticky | 1 | 1 |
| round-robin | 2, alternating | 2 |
| least-recently-used | 2, alternating | 2 |

Also confirmed live: the resolved project is written back after a successful
call, so later requests skip discovery.

### Two bugs this shook out

**The project was never persisted.** The probe reported `tier: unknown` and an
empty project even though discovery had just succeeded, because it read the
account's stored state rather than what discovery returned. Fixed by carrying the
project and tier on the successful attempt and writing them back — except when
the project came from the shared fallback, which is deliberately *not* recorded,
since caching it would make a wrong project permanent.

**A formatted string was stored where a raw id belongs.** `captured_tier_id` got
`"free-tier (paid: g1-pro-tier)"` instead of `"free-tier"`. The display
formatting belongs at the edges. Also changed to update on every discovery rather
than only when absent: tiers change, and an account that upgrades would otherwise
be described by its old tier forever.

A third was found by the live run refusing to behave: `GG_ACCOUNT_STRATEGY=lru`
was accepted while the TOML took `least-recently-used`, so the same setting had
two vocabularies and `least-recently-used` silently did nothing. Parsing is now
one function with a round-trip test.

### Not verified

No live rate limit was induced, so the 429 path is covered by unit tests only.
Forcing one would mean deliberately exhausting an account, which is not worth
doing to a working account.

## M5 — HTTP surface

Added `src/server/`: `mod.rs` (router, auth, shutdown), `chat.rs`, `models.rs`,
`admin.rs`, `error.rs`.

### The prelude

The interesting part is that the handler reads upstream events *before*
committing to a response. That buys two properties that are otherwise
unobtainable:

- **An empty response can be retried.** A stream that produces only signatures
  and finish reasons is a failure, and catching it is only possible while the
  response head is still unwritten. Commit first and the client gets a 200 with
  nothing in it — indistinguishable from a model that had nothing to say.
- **A transport failure is still a real HTTP status.** Because the head is
  inspected first, an upstream 429 becomes a client 429 with `Retry-After`
  rather than a stream that breaks after its headers.

The cost is a short delay before the first byte, bounded by an event count and a
deadline. Both bounds are checked *after* absorbing each chunk rather than at the
top of the loop, because one chunk can carry many events.

### Verified live, over HTTP

| | |
|---|---|
| buffered completion | 200, `content: "pong"`, usage with cached and reasoning token detail |
| streaming | role announced once, content delta, finish chunk, usage chunk, `[DONE]` |
| tool call | `get_weather({"city":"Paris"})` with the upstream id preserved, `finish_reason: tool_calls` |
| two-turn tool loop | turn 1 returns the call, turn 2 replays it with its result and answers |
| auth | 401 without a key, 401 with the wrong key, 200 with either configured key via either header |
| error paths | 404 fallback and an unresolvable model both return the OpenAI envelope |

### What live traffic corrected, once more

**Claude refuses a forced tool choice while thinking is enabled.** A request with
`tool_choice: "required"` came back as *"Thinking may not be enabled when
tool_choice forces tool use."* — a constraint none of the reference projects
document. The gateway now catches the combination before spending a round trip
and returns a message naming both ways out.

The first attempt at that check was wrong and a test caught it: it rejected
*any* non-`AUTO` mode, but Claude's default is `VALIDATED`, so it would have
refused nearly all Claude traffic. The conflict is specifically about *forcing*.

**Upstream errors nested a second envelope.** The Anthropic-shaped error arrived
inside the Google error's `message` field, so clients saw an escaped JSON blob
instead of a sentence. `first_message` now unwraps one level.

### The signature cache is not load-bearing — measured, not assumed

This closes the question left open since M3. The tool-call replay was run twice:
once with the cache warm (a real signature reattached) and once cold (an id the
process had never issued, so the `skip_thought_signature_validator` sentinel was
sent instead). **Both were accepted and both produced identical answers.**

So for Gemini, the sentinel is a full substitute for a captured signature, and
the cache is fidelity insurance rather than a correctness requirement. The
reference projects' claim that signatures *must* be preserved does not hold for
this path. The Claude path has no sentinel, so there the cache remains the only
mechanism — but Claude cannot currently exercise it, because a forced tool choice
is incompatible with thinking and a non-forced one may not call a tool at all.

## M6 — Observability

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

## The live model list

`/v1/models` now merges the static catalogue with `fetchAvailableModels`, cached
for five minutes and shared across accounts — membership is a deployment
property, not an account one. The static table wins on anything it knows, because
it carries the metadata the upstream does not report; the live list is the
authority on membership and adds models this build has never heard of.

Three decisions:

- **Fail open.** A refresh that fails serves what was cached before, and a cold
  cache leaves the static catalogue. A model list that empties on a network
  hiccup is worse than a stale one, because clients fetch it once at startup.
- **An exhausted model is still listed.** Removing a model that clears in an hour
  would mean restarting the client to get it back. Exhaustion belongs on the
  quota endpoints.
- **Unknown models carry no limits.** A wrong context window is worse than none,
  because a client budgets against whatever it is told.

### The response is not a list of models

It contains entries that are not models at all:

```text
chat_20706                       an internal routing identifier
chat_23310                       likewise
tab_flash_lite_preview           a feature flag
tab_jump_flash_lite_preview      likewise
```

A client that picks one from a picker gets a failure it cannot interpret. The
filter is an explicit family prefix. The reference implementation instead asks
"is this Claude or Gemini", which removes the internal entries but also discards
`gpt-oss-120b-medium` as collateral — a real model, confirmed callable.

The `gemini-*-tiered` entries look equally suspicious and were tested rather than
assumed: both are callable, so they stay.

## M7 — the OpenAI Responses API

Added `src/transform/responses.rs` and `src/server/responses.rs`, plus
`src/server/execute.rs` so both protocol handlers share one dispatch, one prelude,
and one recording path. An empty-response retry that existed on one route and not
the other is a bug nobody would think to look for.

The Responses API is a different protocol, not a renamed one. Three differences
shaped the work:

- **Input items are typed.** A conversation is `message`, `function_call`, and
  `function_call_output` items, and a tool call is a *sibling* of the message that
  produced it rather than a field on it.
- **Output is a list of items** with their own ids, which a client tracks.
- **Streaming uses named SSE events** with a monotonic `sequence_number`. The
  Chat Completions route sends bare `data:` lines; this one sends
  `event: <type>` as well.

Requests are converted into the Chat Completions shape and dispatched through the
same pipeline, so signature replay, account rotation, and the empty-response
retry all behave identically on both routes. Only the rendering differs.

### Where this improves on the reference

The reference implementation's event state machine provided the event set, and
three of its choices were worth not copying:

- Its `response.output_text.done` carries an empty string, making the event
  useless to a client that relies on it. The real accumulated text is sent here.
- Its `response.completed` carries an empty `output` array. The full output is
  sent here, so a client that only handles the terminal event still has
  everything.
- Its reasoning delta uses `response.reasoning.delta`, which is not one of the
  protocol's event names. `response.reasoning_summary_text.delta` is.

One gap of my own, found by writing the index test: the reasoning item was added
to the output but never announced with `output_item.added`, leaving a client no
id to associate its deltas with. Every item is announced now.

### Verified live

| | |
|---|---|
| buffered | `resp_*`, `status: completed`, message item with `output_text`, usage with cached and reasoning detail |
| streaming | `created` → `in_progress` → `output_item.added` → `content_part.added` → `output_text.delta` → `done` → `output_item.done` → `completed`, sequence numbers monotonic |
| tool call | `function_call` item with `fc_*` id and `call_*` call_id, argument delta and done carrying the same JSON |
| two-turn loop | turn 1 returns the call, turn 2 replays it as typed input items and answers |
| audit | both routes are recorded; the log captured the 400 from the bug below, which is what it is for |

### The bug the live test caught

`function_call_output` does not repeat the tool name — the Chat Completions spec
puts it on the call, and the Responses item has no name field at all. The upstream
requires a non-empty name on every `functionResponse` and rejected the whole
request:

```text
GenerateContentRequest.contents[2].parts[0].function_response.name: Name cannot be empty.
```

The converter now recovers the name from the call it answers, in the shared
translation layer so both routes benefit. A name on the result still wins; an
orphaned result falls back to a placeholder, because the upstream rejects an empty
name and a placeholder beats a failure.

### Not verified

The Responses route has since been driven by the official OpenAI Python SDK —
`responses.create` non-streaming and streaming, a tool call, and a
`function_call_output` replay — so the hand-written-request stage is over. Codex
CLI remains untried, and it is the client that would exercise an agent loop rather
than one call at a time.

## The 404 that was a stale binary

A `gravitygate probe` failed with `404 Not Found` / `Requested entity was not
found`, from the same account that had worked days earlier. It was not the
gateway.

`gravitygate` on `PATH` resolved to `~/.cargo/bin/gravitygate.exe`, installed five
days earlier, and the probe output gave it away on inspection: it printed the
pre-M5 field set (`request: 553 bytes`, `requestId:` rather than `traceId:`).
That build predated two things, and either would have explained it.

The mechanism was confirmed rather than assumed. `examples/raw_probe.rs` sends a
model name verbatim, with no tier resolution, which is what the old build did:

| Model sent | Result |
|---|---|
| `gemini-3.8-flash` | **404 Not Found** |
| `gemini-3.8-flash-medium` | 200 OK |
| `gemini-3.8-flash-low` | 200 OK |

The upstream has no model called `gemini-3.8-flash`. Tier resolution exists
precisely to turn that base name into a wire name, and the old build had none, so
it sent a name the upstream does not have. The current build resolves it and
returns 200 on both accounts.

`raw_probe` is kept: asking the upstream directly about a spelling is something
the normal routes cannot do by design, and that is exactly the question worth
answering when a 404 appears.

### The lesson worth keeping

A stale binary is indistinguishable from a bug when the version string never
changes. `--version` reported `0.1.0` for every build ever made, so nothing in the
output said the binary was five days old.

`build.rs` now stamps the build time and revision, and `--version` reports them:

```text
gravitygate 0.1.0 (built 2026-09-22T16:06:52Z, rev d0b42e1-dirty)
```

The timestamp is formatted at build time by arithmetic, so the binary carries no
date handling, and `SOURCE_DATE_EPOCH` is honoured for reproducible builds. This
is what should have caught the problem: the first thing to check when a command
misbehaves is whether the binary is the one you think it is.

## The probe reported an empty answer

`gravitygate probe` reported an empty answer for responses that plainly had one,
and the streaming translator had produced the whole thing. The report simply was
not reading it.

`drain_stream` decoded every byte and fed every event to the translator, but kept
only the first 4 KiB of the body for the report, and the report then reconstructed
the answer by folding that copy. With thinking enabled the reasoning prose arrives
first, so a long reasoning phase filled the cap before the first answer token was
ever streamed, and the fold found nothing but thoughts. `text()` filters thoughts
out by design — which is what keeps reasoning from being printed as the reply — so
the result was an empty answer where a long one belonged.

The cap is a display concern and now stays one. The run keeps every decoded event
and the report folds those; `raw` is still what `--raw` prints and what an attempt
records, and the report still says when it truncated. Fixing it also moved two
smaller things off the capped copy:

- The trace id now comes from the decoded `{response, traceId}` envelope instead
  of being re-parsed out of the truncated text.
- The string-scanning fold is gone. It was a second implementation of the merge,
  operating on text the decoder had already parsed, and the tests that covered it
  now drive the real path instead: bytes through the decoder into the fold, with a
  case where the answer lies past the cap and the copy cannot be its source.

One promise the cap had quietly broken: `--raw` is documented as printing the
whole body, but it inherited the same 4 KiB limit, and the truncation notice was
suppressed *because* `--raw` was set — so the flag printed a truncated body with
no indication that anything was missing. The limit is now a parameter of
`probe_request`, the default call keeps 4 KiB, and `--raw` passes no limit at all.
Measured after the change: the default body comes back at 4173 bytes with the
notice, `--raw` at 18 690 bytes with no notice and a final event that parses.

### The thinking text was never withheld

The earlier note that Gemini "returns a signature and a `thoughtsTokenCount` but
no thinking text" does not hold up as stated. Thinking text comes back for most
configs; the config being probed was part of the problem. Measured with
`raw_probe`, which never had the cap:

| Wire model | `thinkingConfig` | Prompt | `thoughtsTokenCount` | thinking text |
|---|---|---|---|---|
| `gemini-3.8-flash-high` | `{"includeThoughts":true}` | reasoning task | 270 | **none** |
| `gemini-3.8-flash-high` | `+{"thinkingBudget":1000}` | reasoning task | 147 | **none** |
| `gemini-3.8-flash-high` | `+{"thinkingBudget":10000}` | reasoning task | 343 | **none** |
| `gemini-3.8-flash-high` | `+{"thinkingBudget":-1}` | reasoning task | 473 | 192 chars |
| `gemini-3.8-flash-high` | `+{"thinkingBudget":-1}` | trivial | 396 | 175 chars |
| `gemini-3.8-flash-high` | `+{"thinkingLevel":"high"}` | reasoning task | 541 | 141 chars |
| `gemini-3.8-flash-medium` | `+{"thinkingBudget":4000}` | design task | 1596 | 1950 chars |
| `gemini-3.8-flash-medium` | `+{"thinkingBudget":4000}` | trivial | 210–237 | **none** (twice) |
| `gemini-3.6-flash-high` | `+{"thinkingBudget":10000}` | reasoning task | 753 | 352 chars |
| `gemini-pro-agent` | `+{"thinkingBudget":10001}` | reasoning task | 835 | 1113 chars |

Two things follow, and only the first is tidy:

- A positive `thinkingBudget` on a `gemini-3.8-flash-high` wire name never
  returned text, across three budgets, while `-1` on the same model did. Every
  tier route the gateway resolves reaches a config that can return text —
  including the `-1` the catalogue records for the high tier of 3.7 and 3.8
  Flash, which is now explained rather than merely copied from a capture.
- Beyond that, text is intermittent rather than guaranteed. A trivial prompt on
  the default `-medium` route returned 210–237 thinking tokens and no text, twice,
  on a config that returned 1950 characters for a substantial one. So the token
  count really is no evidence of anything, exactly as the diagnostic tool's own
  doc comment warns.

Nothing in the translation layer needed to change: a response with
`reasoning_tokens` and no reasoning deltas is a shape the gateway already
produces, and the chat route's `reasoning` line and the Responses route's
reasoning items simply stay empty when the upstream sends none.

## The usage numbers follow OpenAI's convention

The first live SDK pass reported `completion_tokens: 1` next to
`reasoning_tokens: 84`, with `total_tokens: 7`. The identity
`total = prompt + completion` held, but `reasoning_tokens` exceeded
`completion_tokens` — a shape no OpenAI client expects, because OpenAI's
convention is that reasoning tokens are a *subset* of completion tokens:
`completion = answer + thinking`.

The mapping had treated completion as the visible answer only, with thinking
reported solely under `completion_tokens_details`. That was a deliberate call at
the time — the derivation lives in `UsageMetadata::candidates_tokens` — but it
lost to the convention the clients actually code against, so it is now the other
way: `UsageMetadata::completion_tokens()` returns candidates **plus** thinking,
`to_usage` and the Responses `ResponsesUsage::from_ir` both use it for
`completion_tokens` / `output_tokens`, and the audit record and the metrics use
it too. `reasoning_tokens` is unchanged: it stays the thinking-only breakdown
inside the details object.

Measured after the change, all three shapes satisfy both identities:

```text
chat:        completion=350 reasoning=276 total=374
responses:   output=330     reasoning=269 total=354
chat stream: completion=274 reasoning=211 total=298
```

The dashboard's "Completion" card now counts thinking as well, which is what a
reader comparing it against the provider's own billing expects.

## Two operator traps: duplicate accounts and a hold that never cleared

Both were found the same evening, from the outside: an `account list` that showed
one Google account three times, and an account stuck in `verify` even after its
holder had completed the challenge in the browser.

### A re-login used to become a second pool entry

`upsert_account` deduplicated on the credential id, which is derived from the
refresh token. That guard is airtight against adding the same token twice, and
useless against its most common near-miss: every `account login` mints a *fresh*
refresh token, so a second login of one Google account arrived as a brand-new
credential and became a brand-new pool entry. Three logins produced three
entries with one email, and since all of them draw on the *same* upstream quota,
rotation between them gained nothing while every quota exhaustion cost two extra
guaranteed-429 requests — each of which marked another entry as limited. The
list showed one account limited three times with countdowns seconds apart.

The match now falls back to identity: an add whose email matches an existing
entry, and whose project does not contradict it, replaces the stored credential
in place. A missing project on either side is compatible, because `login` learns
the project later, at first use. The same account with a *different* project
still becomes a separate entry — that is the packed
`refresh|project|managedProject` form doing its job. Replacement drops
verification holds and cooldowns — the fresh credential deserves a fresh
judgment — but keeps recorded rate limits: same account, same quota pool, and a
re-login must not launder away a limit the upstream is still enforcing.

### The verification hold had no way to end

`verification_required` blocked dispatch with no expiry, and the doc comment said
so: blocked "until an operator clears this". Worse, nothing cleared it — not a
served request, not `account verify` (whose discovery path never touched the
persisted flag). An operator who completed the challenge saw the status stay
`verify` forever, and remove-plus-re-login was the only exit.

The hold now expires into a recheck window: 1 minute, then 5, 15, and 60, one
rung per re-assertion. When the window opens the account becomes dispatchable
again, and the next real request settles the question — a success clears the
hold (`record_success` now does this, as does `account verify` on a successful
discovery), while a fresh 403 re-arms it one rung further out. The status line
says where the account stands: `verify, recheck in 2m, <url>`, then `recheck
due`. Accounts marked by older builds carry no window at all and count as due
immediately, so pre-existing holds self-heal too. `clear-holds` remains for
impatience.

The demand itself — a 403 carrying `validation_required` — is Google's
risk-based challenge, and it is common for accounts enabling the service for the
first time. That part is upstream. The part that kept a verified account benched
was ours.

### Two smaller calibrations

- The "every account is rate limited; waiting" log printed whole seconds, so a
  400 ms token-bucket refill wait appeared as `waiting seconds=0` — correct
  behavior dressed as a bug. It now logs milliseconds.
- The token bucket's default ceiling dropped from 50 to 10 per account. The
  bucket exists so a burst stops before the upstream's rate limiter sees it, and
  fifty let twenty thinking-heavy requests through in two minutes, which is how
  the free-tier quota got exhausted in the first place. Refill stays at six per
  minute; both are config knobs.

## Next, in order

Everything in the approved plan is now built. What remains is validation and
polish rather than features:

1. **Codex CLI against `/v1/responses`.** The official OpenAI Python SDK now
   drives both routes end to end: chat non-streaming and streaming, a tool call
   with its result replayed on a second turn, Responses non-streaming and
   streaming, a `function_call_output` replay, `stream_options.include_usage`,
   and `/v1/models`. Codex would add what an SDK does not — an agent loop
   choosing its own requests.
2. **A Claude tool loop.** Blocked on the thinking/tool-choice constraint above,
   not on the gateway.
3. **Live rate-limit handling.** Still covered by unit tests only.
