"""One accumulator shared by buffered, Chat and Responses streaming paths."""

import json
import time
import uuid

from .cache import SignatureCache
from .sse import StreamError


def new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:24]


def usage_for(raw: dict) -> dict:
    prompt = max(0, raw.get("promptTokenCount", 0))
    cached = max(0, raw.get("cachedContentTokenCount", 0))
    thinking = max(0, raw.get("thoughtsTokenCount", 0))
    answer = raw.get("candidatesTokenCount")
    if answer is None:
        answer = max(0, raw.get("totalTokenCount", prompt + thinking) - prompt - thinking)
    completion = max(0, answer) + thinking
    prompt = max(0, prompt - cached)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": thinking},
    }


class Accumulator:
    def __init__(
        self,
        model: str,
        family: str,
        session: str,
        cache: SignatureCache,
        reasoning_field: str = "reasoning_content",
    ):
        self.model, self.family, self.session, self.cache = model, family, session, cache
        self.reasoning_field = reasoning_field
        self.id, self.created = new_id("chatcmpl-"), int(time.time())
        self.text, self.reasoning, self.tools, self.images = [], [], [], []
        self.raw_usage: dict = {}
        self.finish_reason: str | None = None
        self.model_version = self.response_id = None
        self.finished = False

    @property
    def visible(self) -> bool:
        return bool(self.text or self.tools)

    @property
    def has_output(self) -> bool:
        return self.visible or bool(self.reasoning)

    @property
    def usage(self) -> dict:
        return usage_for(self.raw_usage)

    @property
    def finish(self) -> str:
        if self.tools:
            return "tool_calls"
        if self.finish_reason == "MAX_TOKENS":
            return "length"
        if self.finish_reason in {
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "IMAGE_SAFETY",
        }:
            return "content_filter"
        return "stop"

    def reasoning_delta(self, text: str) -> dict:
        out = {}
        if self.reasoning_field in {"reasoning_content", "reasoning-content", "both"}:
            out["reasoning_content"] = text
        if self.reasoning_field in {"reasoning", "both"}:
            out["reasoning"] = text
        return out

    def absorb(self, payload: dict) -> list[dict]:
        if payload.get("error"):
            raise StreamError("upstream returned an error inside its stream")
        response = payload.get("response", payload)
        if not isinstance(response, dict):
            raise StreamError("upstream response is not an object")
        if response.get("error"):
            raise StreamError("upstream returned an error inside its response")
        if response.get("usageMetadata"):
            usage = response["usageMetadata"]
            if not isinstance(usage, dict) or any(
                not isinstance(usage.get(key, 0), (int, float))
                for key in (
                    "promptTokenCount",
                    "cachedContentTokenCount",
                    "thoughtsTokenCount",
                    "totalTokenCount",
                    "candidatesTokenCount",
                )
                if usage.get(key) is not None
            ):
                raise StreamError("upstream returned invalid token usage")
            self.raw_usage = response["usageMetadata"]
        self.model_version = response.get("modelVersion", self.model_version)
        self.response_id = response.get("responseId", self.response_id)
        deltas = []
        for candidate in response.get("candidates", []):
            if candidate.get("index", 0) != 0:
                continue
            if candidate.get("finishReason"):
                self.finish_reason = candidate["finishReason"]
            for part in candidate.get("content", {}).get("parts", []):
                signature = part.get("thoughtSignature", "")
                if fn := part.get("functionCall"):
                    if (
                        not isinstance(fn, dict)
                        or not isinstance(fn.get("name", ""), str)
                        or not isinstance(fn.get("args", {}), dict)
                    ):
                        raise StreamError("upstream returned an invalid tool call")
                    call = {
                        "id": fn.get("id") or new_id("call_"),
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": json.dumps(
                                fn.get("args", {}), ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    }
                    self.cache.put("tool", call["id"], signature, self.family)
                    index = len(self.tools)
                    self.tools.append(call)
                    deltas.append({"tool_calls": [{"index": index, **call}]})
                else:
                    self.cache.put("thinking", self.session, signature, self.family)
                    if text := part.get("text"):
                        if not isinstance(text, str):
                            raise StreamError("upstream returned a non-text answer")
                        if part.get("thought"):
                            self.reasoning.append(text)
                            deltas.append(self.reasoning_delta(text))
                        else:
                            self.text.append(text)
                            deltas.append({"content": text})
                    if image := part.get("inlineData"):
                        self.images.append(image)
        return deltas

    def completion(self) -> dict:
        message = {"role": "assistant", "content": "".join(self.text) or None}
        if self.reasoning:
            message.update(self.reasoning_delta("".join(self.reasoning)))
        if self.tools:
            message["tool_calls"] = self.tools
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "message": message, "finish_reason": self.finish}],
            "usage": self.usage,
        }

    def chunk(self, delta: dict, reason: str | None = None) -> dict:
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
        }

    def final_chunks(self, include_usage: bool) -> list[dict]:
        if self.finished:
            return []
        self.finished = True
        out = [self.chunk({}, self.finish)]
        if include_usage:
            out.append(
                {
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [],
                    "usage": self.usage,
                }
            )
        return out


class ResponsesStream:
    def __init__(self, accumulator: Accumulator, request: dict | None = None):
        self.acc = accumulator
        self.request = request or {}
        self.id = new_id("resp_")
        self.items: list[dict] = []
        self.active: tuple[str, int] | None = None
        self.sequence = 0
        self.done = False
        self.started = False

    def event(self, kind: str, **fields) -> dict:
        out = {"type": kind, "sequence_number": self.sequence, **fields}
        self.sequence += 1
        return out

    def response(self, status: str = "in_progress") -> dict:
        usage = self.acc.usage
        return {
            "id": self.id,
            "object": "response",
            "created_at": self.acc.created,
            "status": status,
            "model": self.acc.model,
            "output": self.items,
            "parallel_tool_calls": self.request.get("parallel_tool_calls", True),
            "tool_choice": self.request.get("tool_choice", "auto"),
            "tools": [],
            "error": None,
            "incomplete_details": {"reason": "max_output_tokens"}
            if status == "incomplete"
            else None,
            "usage": {
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "input_tokens_details": usage["prompt_tokens_details"],
                "output_tokens_details": usage["completion_tokens_details"],
            },
        }

    def start(self) -> list[dict]:
        if self.started:
            return []
        self.started = True
        opening = {**self.response(), "output": [], "usage": None}
        return [
            self.event("response.created", response=opening.copy()),
            self.event("response.in_progress", response=opening.copy()),
        ]

    def close_active(self) -> list[dict]:
        if self.active is None:
            return []
        kind, index = self.active
        item = self.items[index]
        self.active = None
        events = []
        if kind == "text":
            part = item["content"][0]
            events.append(
                self.event(
                    "response.output_text.done",
                    item_id=item["id"],
                    output_index=index,
                    content_index=0,
                    text=part["text"],
                )
            )
            events.append(
                self.event(
                    "response.content_part.done",
                    item_id=item["id"],
                    output_index=index,
                    content_index=0,
                    part=part.copy(),
                )
            )
        elif kind == "reasoning":
            part = item["summary"][0]
            events.append(
                self.event(
                    "response.reasoning_summary_text.done",
                    item_id=item["id"],
                    output_index=index,
                    summary_index=0,
                    text=part["text"],
                )
            )
            events.append(
                self.event(
                    "response.reasoning_summary_part.done",
                    item_id=item["id"],
                    output_index=index,
                    summary_index=0,
                    part=part.copy(),
                )
            )
        else:
            events.append(
                self.event(
                    "response.function_call_arguments.done",
                    item_id=item["id"],
                    output_index=index,
                    arguments=item["arguments"],
                    name=item["name"],
                    call_id=item["call_id"],
                )
            )
        item["status"] = "completed"
        events.append(
            self.event(
                "response.output_item.done", output_index=index, item=json.loads(json.dumps(item))
            )
        )
        return events

    def on_deltas(self, deltas: list[dict]) -> list[dict]:
        events = []
        for delta in deltas:
            if calls := delta.get("tool_calls"):
                kind, value = "tool", calls[0]
            elif "content" in delta:
                kind, value = "text", delta["content"]
            else:
                kind, value = (
                    "reasoning",
                    delta.get("reasoning_content", delta.get("reasoning", "")),
                )
            if self.active is not None and (self.active[0] != kind or kind == "tool"):
                events.extend(self.close_active())
            if self.active is None:
                index = len(self.items)
                if kind == "text":
                    item = {
                        "id": new_id("msg_"),
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                elif kind == "reasoning":
                    item = {
                        "id": new_id("rs_"),
                        "type": "reasoning",
                        "status": "in_progress",
                        "summary": [],
                    }
                else:
                    item = {
                        "id": new_id("fc_"),
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": value["id"],
                        "name": value["function"]["name"],
                        "arguments": "",
                    }
                self.items.append(item)
                self.active = kind, index
                events.append(
                    self.event(
                        "response.output_item.added",
                        output_index=index,
                        item=json.loads(json.dumps(item)),
                    )
                )
                if kind == "text":
                    part = {"type": "output_text", "text": "", "annotations": []}
                    item["content"].append(part)
                    events.append(
                        self.event(
                            "response.content_part.added",
                            item_id=item["id"],
                            output_index=index,
                            content_index=0,
                            part=part.copy(),
                        )
                    )
                elif kind == "reasoning":
                    part = {"type": "summary_text", "text": ""}
                    item["summary"].append(part)
                    events.append(
                        self.event(
                            "response.reasoning_summary_part.added",
                            item_id=item["id"],
                            output_index=index,
                            summary_index=0,
                            part=part.copy(),
                        )
                    )
            _, index = self.active
            item = self.items[index]
            if kind == "text":
                item["content"][0]["text"] += value
                events.append(
                    self.event(
                        "response.output_text.delta",
                        item_id=item["id"],
                        output_index=index,
                        content_index=0,
                        delta=value,
                    )
                )
            elif kind == "reasoning":
                item["summary"][0]["text"] += value
                events.append(
                    self.event(
                        "response.reasoning_summary_text.delta",
                        item_id=item["id"],
                        output_index=index,
                        summary_index=0,
                        delta=value,
                    )
                )
            else:
                item["arguments"] += value["function"]["arguments"]
                events.append(
                    self.event(
                        "response.function_call_arguments.delta",
                        item_id=item["id"],
                        output_index=index,
                        delta=value["function"]["arguments"],
                    )
                )
        return events

    def finish(self) -> list[dict]:
        if self.done:
            return []
        self.done = True
        events = self.close_active()
        status = "incomplete" if self.acc.finish_reason == "MAX_TOKENS" else "completed"
        events.append(self.event("response." + status, response=self.response(status)))
        return events
