# M0 — HTTP transport feasibility

**Question.** `antigravity-auth` hand-rolls HTTP/1.1 over `tls.connect` instead of using
the platform HTTP client, citing header ordering and connection behaviour. Does
`reqwest`/`hyper` need the same treatment in Rust, or can we use the normal stack?

**Method.** `examples/spike_http.rs` runs entirely against a local TCP listener that
captures the raw request bytes. No credentials required. It answers four things:
`http::HeaderMap` iteration order, streamed vs buffered body framing, and which headers
reqwest injects that we did not request.

## Results

### 1. Header order IS controllable

```
inserted : ["user-agent", "authorization", "content-type", "accept-encoding"]
iterated : ["user-agent", "authorization", "content-type", "accept-encoding"]
verdict  : PRESERVED
```

`http::HeaderMap` iterates in insertion order (`Vec`-backed `entries`, not the lookup
hash table), and hyper serialises in iteration order. **This was the main risk and it
does not materialise.** No hand-rolled HTTP/1.1 writer is needed for ordering reasons.

### 2. Streamed bodies produce `Transfer-Encoding: chunked`

A `reqwest::Body::wrap_stream` body yields `transfer-encoding: chunked`; a buffered body
yields `content-length`. This matches the capture, which uses chunked only for
`*:streamGenerateContent`.

### 3. `Accept-Encoding` is fully controllable

reqwest's decompression features would otherwise send `accept-encoding: gzip,deflate,br`.
Setting the header explicitly overrides that and emits exactly `gzip`, matching the capture.

### 4. Three deviations from the captured block remain

Captured `agy` CLI 1.1.24 request, for reference:

```text
POST /v1internal:streamGenerateContent?alt=sse HTTP/1.1
Host: daily-cloudcode-pa.googleapis.com
User-Agent: antigravity/cli/1.1.24 (aidev_client; os_type=darwin; arch=arm64; cl=974782877; auth_method=consumer)
Transfer-Encoding: chunked
Authorization: <redacted>
Content-Type: application/json
Accept-Encoding: gzip
```

What reqwest actually emits:

```text
POST /v1internal:streamGenerateContent?alt=sse HTTP/1.1
user-agent: antigravity/cli/1.1.24 (aidev_client; os_type=darwin; arch=arm64; cl=974782877; auth_method=consumer)
authorization: Bearer REDACTED
content-type: application/json
accept-encoding: gzip
accept: */*
host: 127.0.0.1:11500
transfer-encoding: chunked
```

| Deviation | Cause | Fixable? |
|---|---|---|
| Header names are lowercase, CLI sends title case | `http::HeaderName` normalises to lowercase; hyper writes as stored | No, not via the hyper API. Would require a hand-rolled writer. |
| `accept: */*` present, CLI sends no `Accept` | Injected by reqwest | Not removable. Setting `Accept: ""` emits `accept: ` (empty), which is worse. Only overridable, not suppressible. |
| `host` and framing headers come last, CLI puts `Host` first | hyper appends its own framing/host headers after the user's map | No. |

Header *names* being case-insensitive is guaranteed by RFC 9110, and header *order* carries
no semantics in HTTP. Google's frontend terminates TLS and normalises both. The realistic
reading is that these three deviations are cosmetic.

It is also worth noting what the hand-rolled transport in `antigravity-auth` is most
plausibly *for*. In Bun/Node, `fetch` gives you no control over chunked-vs-length framing,
no fine-grained timeout control, and no idle-body watchdog. All three of those are
available in Rust — the spike demonstrates exact control over framing, and reqwest exposes
per-phase timeouts. Header ordering was likely the stated reason rather than the binding one.

## Decision

**Proceed with `reqwest` (HTTP/1.1 only) for M1.** The transport is isolated behind a
single module (`src/upstream/transport.rs`) so that swapping in a hand-rolled HTTP/1.1
writer over `rustls` is a contained change if real-upstream validation demands it.

## Open item — cannot be closed without credentials

The spike proves what goes on the wire, not whether the upstream accepts it. Validating
the three deviations above requires a real refresh token. This is the first thing to check
once an account is configured; `gravitygate probe` (M1) exists for exactly that.
