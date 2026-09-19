# GravityGate

An OpenAI-compatible gateway in front of Google Antigravity's Cloud Code Assist
backend, written in Rust on Axum and Tokio.

> **Read this first.** Using this software violates Google's Terms of Service.
> Account holders have reported suspensions, bans, and shadow-bans for exactly
> this kind of proxying. The account pool exists so that one lost account does
> not take the gateway down, but it does not make the practice safe. Do not
> configure an account you cannot afford to lose.

## Status

Under construction. The upstream connectivity path is complete and verified; the
HTTP surface is not.

| Area | State |
|---|---|
| Wire identity, envelope, metadata | done |
| Transport (`reqwest`, HTTP/1.1) | done, validated offline |
| OAuth refresh, manual token intake | done |
| Browser OAuth login (PKCE) | done |
| Project discovery (`loadCodeAssist` / `onboardUser`) | done |
| Account pool: model, storage, locking | done |
| CLI: `account`, `probe`, `config` | done |
| Model registry, tier routing | done |
| OpenAI → IR request translation | done |
| JSON Schema sanitisation | done |
| Upstream SSE parsing → OpenAI chunks | **not started** |
| Thinking signature cache | **not started** |
| Account routing, retry, backoff | done, live-validated across two accounts |
| `serve`, `/v1/chat/completions` | **not started** |
| Metrics, audit log, dashboard | **not started** |

Per-milestone detail, decisions, and what was verified when is in
[docs/progress.md](docs/progress.md).

## Quick start

```sh
cargo build --release

# Check where configuration lives.
./gravitygate config path

# Sign in with a browser. Runs the Google PKCE flow against a loopback listener.
./gravitygate account login

# Or add an account from a refresh token / packed refresh|project|managed token.
./gravitygate account add-token '1//0g...' --email you@example.com

# Confirm it works before putting anything in front of it.
./gravitygate probe --model gemini-3.8-flash
```

`probe` runs the production pipeline — model resolution, request translation,
signature replay, upstream call, SSE decoding, response translation — and then
prints what happened: the resolved wire model and tier, the project, the
endpoint and status, the answer, any thinking, the signatures captured, what the
client would have received, and the raw response body. Because it runs the real
path, a healthy probe is evidence about the gateway rather than about the
transport. It is the tool to reach for when something is wrong.

## Design decisions

Three choices shape everything else.

**The IR is the Google GenAI shape, not Anthropic's Messages shape.** The
upstream is the only fixed constraint, so shaping the normalised representation
around it means a client request crosses exactly one translation boundary.
Thinking has a native home on the parts that carry it
(`{text, thought, thoughtSignature}`), rather than needing a side channel.

**Thinking signatures are the gateway's problem, and the hardest part.** The
upstream requires a signature to be returned with a conversation's prior
thinking. OpenAI's protocol has nowhere to put one — `tool_calls` has no
signature field, and clients drop unknown fields when they echo a turn back. So
the gateway must maintain a `tool_call_id → signature` cache of its own. This is
the sharpest difference from the reference implementations, which speak the
Anthropic protocol where `signature_delta` and `thinking.signature` carry
signatures natively.

**Header order is reproduced, not approximated.** The upstream is calibrated
against captured CLI traffic. `http::HeaderMap` iterates in insertion order and
hyper serialises in iteration order, so the sequence of `.header()` calls in
`src/upstream/transport.rs` is the sequence on the wire. Reordering them is a
wire-format change. See `docs/m0-transport-findings.md` for what was verified and
what remains unverified.

## Layout

```
src/
  main.rs, cli.rs         CLI: serve, account, probe, config
  config.rs               TOML + GG_* environment overrides
  upstream/               Google wire protocol
    constants.rs          endpoints, User-Agent, OAuth client
    envelope.rs           request envelope, field order fixed by capture
    metadata.rs           session context, requestId, labels
    transport.rs          reqwest client, header block, framing
  oauth/
    pkce.rs               verifier and challenge generation (RFC 7636)
    callback.rs           loopback listener, port fallback, pasted-code parsing
    login.rs              end-to-end authorization-code flow
    token.rs              refresh, code exchange, error classification
  accounts/
    account.rs            account model and on-disk schema
    store.rs              atomic writes, advisory locking, migration
    project.rs            loadCodeAssist / onboardUser
    ratelimit.rs          failure classification and backoff
    router.rs             selection strategies, token bucket, health
  registry/               model catalogue and tier routing
  upstream/sse.rs         incremental SSE decoder
  transform/
    ir.rs                 canonical intermediate representation
    openai.rs             OpenAI wire types
    request.rs            OpenAI -> IR
    response.rs           IR -> OpenAI, buffered and streamed
    schema.rs             JSON Schema -> Gemini schema
    signature_cache.rs    thinking signature storage and replay
  engine/
    credentials.rs        in-memory access token cache
    dispatch.rs           the account/endpoint retry loop
    retry.rs              retry decisions, as a pure function
    mod.rs                request execution and probe
```

## Configuring an account

Two intake paths, both already implemented:

```sh
# A bare refresh token.
gravitygate account add-token '1//0g...' --email you@example.com

# The packed form, which carries its own project ids.
gravitygate account add-token '1//0g...|my-project|managed-project'
```

Adding the same token twice updates in place rather than creating a duplicate
that would split traffic between two entries.

Inspect and manage:

```sh
gravitygate account list              # table, with status and time remaining
gravitygate account list --json       # the raw store
gravitygate account verify            # authenticate and resolve a project
gravitygate account disable <sel>     # take out of rotation without removing
gravitygate account clear-holds <sel> # drop rate limits, cooldowns, holds
```

`login` flags worth knowing:

```sh
gravitygate account login --no-browser   # SSH/container: print URL, paste the code back
gravitygate account login --no-open      # do not launch a browser
gravitygate account login --no-verify    # skip the post-login connectivity check
```

Post-humans. Selectors accept an index, an email, or a credential-id prefix —
whatever `account list` printed.

## Storage and its limits

Credentials live in `accounts.json` in plaintext, matching every reference
implementation. Protection is filesystem permissions: `0700` on the directory,
`0600` on the file. On Windows those bits are not applied and the file relies on
the profile's ACL, which is why it lives under `%APPDATA%`.

Writes are atomic (temp file plus `rename`) and serialised behind an advisory
lock file with staleness detection, so a running gateway and a concurrent
`account add` cannot clobber each other. A store that fails to parse fails
closed rather than being replaced with an empty one, because an empty pool and a
corrupted pool look identical from the outside.

## Development

```sh
cargo test          # unit tests
cargo clippy --all-targets
cargo run --example spike_http    # re-run the transport feasibility probe
```

`examples/spike_http.rs` captures the raw bytes reqwest puts on the wire and
compares them against the captured CLI request. It needs no credentials.

## Reference material

`reference/` holds four open-source projects this was designed against. The
analysis is in the requirements plan; the short version:

- `antigravity-auth` — the only implementation calibrated against captured
  traffic. Source of the wire identity, envelope field order, account schema,
  and rate-limit taxonomy.
- `antigravity-claude-proxy` — the cleanest translation layer. Source of the
  schema sanitiser's approach and the signature-cache design.
- `antigravity-gateway` — the closest to this project's target shape. Source of
  the OpenAI adapter's field choices.
- `opencode-antigravity-auth` — the oldest and simplest. Useful mainly as a
  counter-example; it generates a random project id per request, which fragments
  prompt caching and quota accounting.
