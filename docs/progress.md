# Development progress

Running record of what is built, what was verified, and what is next. Updated as
each milestone lands. The requirements plan this executes against is the approved
plan from the kickoff; this file is the state of play.

Last updated: after closing M2 against live accounts.

## At a glance

| Milestone | Scope | State |
|---|---|---|
| M0 | Transport feasibility | **done and closed against a live account** |
| M1 | Upstream connectivity backbone | **done**, verified against a live account |
| M2 | Account pool and routing | **done**, live-validated across two accounts |
| M3 | Streaming, thinking, tool calls | **done** — both paths, live-validated |
| M4 | OAuth browser login | **done** |
| M5 | HTTP surface (`serve`, `/v1/chat/completions`) | not started |
| M6 | Observability | not started |
| M7 | OpenAI Responses (phase two) | not started |

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
```

`serve` is a deliberate stub that explains what is missing.

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

## Next, in order

1. **`serve` and `/v1/chat/completions`** — turns the library into a gateway.
   Everything under it is now built.
2. **A two-turn live tool loop** — closes the replay half of the signature cache.
   Natural to do as a script against `serve`.
