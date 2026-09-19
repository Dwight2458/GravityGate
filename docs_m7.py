p = "docs/progress.md"
s = open(p, encoding="utf-8").read()

def sub(old, new):
    global s
    if old not in s:
        raise SystemExit("not found: " + old[:80])
    s = s.replace(old, new, 1)

sub("Last updated: after adding observability (M6).",
    "Last updated: after the live model list and the Responses API.")

sub("| M7 | OpenAI Responses (phase two) | not started |",
    "| M7 | OpenAI Responses (phase two) | **done**, live-validated |")

sub('''| `POST /v1/chat/completions` | OpenAI-compatible, streaming and buffered |
| `GET /v1/models` | 18 entries: base names plus genuinely distinct tier variants |''',
'''| `POST /v1/chat/completions` | OpenAI-compatible, streaming and buffered |
| `POST /v1/responses` | the Responses API, streaming and buffered |
| `GET /v1/models` | the static catalogue merged with the live list |''')

sub('''## Next, in order

1. **Live `fetchAvailableModels`.** `/v1/models` is built from the static
   catalogue. Merging the live list would make it authoritative about what an
   account can actually reach; the catalogue already supplies what the upstream
   does not report.
2. **A Claude tool loop.** Blocked on the thinking/tool-choice constraint above,
   not on the gateway.
3. **M7 — the OpenAI Responses API.** Phase two of the approved plan, and the
   only client protocol still missing.''',
'''## The live model list

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

The Responses route has not been driven by a real client — only by hand-written
requests. Codex CLI is the obvious candidate and would be the next thing to try.

## Next, in order

Everything in the approved plan is now built. What remains is validation and
polish rather than features:

1. **A real client on each route.** Codex CLI against `/v1/responses`, and any
   OpenAI SDK against `/v1/chat/completions`, would exercise the parts a
   hand-written request does not.
2. **A Claude tool loop.** Blocked on the thinking/tool-choice constraint above,
   not on the gateway.
3. **Live rate-limit handling.** Still covered by unit tests only.''')

open(p, "w", encoding="utf-8").write(s)
print("progress updated")

p = "README.md"
s = open(p, encoding="utf-8").read()
sub("| `serve`, `/v1/chat/completions` | done, live-validated |",
    "| `serve`, `/v1/chat/completions` | done, live-validated |\n"
    "| OpenAI Responses API (`/v1/responses`) | done, live-validated |")
sub("| OpenAI Responses API (`/v1/responses`) | **not started** |", "")
sub('''The gateway works end to end. Everything in the table below has been exercised
against live accounts over HTTP; the one remaining piece is the OpenAI Responses
API, which is phase two of the plan.''',
'''The gateway works end to end, on both protocol routes. Everything in the table
below has been exercised against live accounts over HTTP.''')
sub('''| `GET /v1/models` | the model list |''',
    '''| `GET /v1/models` | the static catalogue merged with the upstream's live list |
| `POST /v1/responses` | the OpenAI Responses API, streaming and buffered |''')
open(p, "w", encoding="utf-8").write(s)
print("readme updated")
