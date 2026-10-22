"""Client protocol normalization and Google GenAI request construction."""

import hashlib
import json
import re
import uuid

from pydantic import BaseModel, ConfigDict, Field

from .cache import SENTINEL, SignatureCache
from .config import Config
from .registry import Resolved
from .schema import sanitize


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = Field(min_length=1)
    messages: list[dict] = Field(min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    tools: list[dict] | None = None
    thinking: dict | None = None
    stream_options: dict | None = None
    reasoning_effort: str | None = None
    user: str | None = None


def text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def session_key(request: dict) -> str:
    if request.get("user"):
        return "user:" + request["user"]
    opening = next(
        (text_content(m.get("content")) for m in request["messages"] if m.get("role") == "user"), ""
    )
    return (
        "conv:" + hashlib.sha256(opening.encode()).hexdigest()[:32]
        if opening
        else "anon:" + uuid.uuid4().hex
    )


def tool_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_:\-]", "_", name)[:64]


def content_parts(content) -> list[dict]:
    if isinstance(content, str):
        return [{"text": content}] if content else []
    result = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and part.get("text"):
            result.append({"text": part["text"]})
        elif part.get("type") == "image_url":
            url = part.get("image_url", {})
            url = url.get("url", "") if isinstance(url, dict) else url
            if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                meta, data = url[5:].split(",", 1)
                result.append({"inlineData": {"mimeType": meta.split(";")[0], "data": data}})
    return result


def to_ir(request: dict, resolved: Resolved, config: Config, signatures: SignatureCache) -> dict:
    contents, system, names = [], [], {}
    if config.upstream.inject_agent_system_prompt:
        system.append(
            {
                "text": "You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team working on Advanced Agentic Coding.You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.**Absolute paths only****Proactiveness**"
            }
        )
    for message in request["messages"]:
        role = message.get("role", "user")
        content = message.get("content")
        for call in message.get("tool_calls") or []:
            if call.get("id"):
                names[call["id"]] = call.get("function", {}).get("name", "unknown")
        if role in {"system", "developer"}:
            system.extend(p for p in content_parts(content) if "text" in p)
            continue
        if role == "tool":
            name = message.get("name") or names.get(message.get("tool_call_id"), "unknown")
            parts = [
                {
                    "functionResponse": {
                        "name": tool_name(name),
                        "response": {"result": text_content(content)},
                    }
                }
            ]
            target_role, category = "user", "tool"
        else:
            parts = content_parts(content)
            target_role, category = (
                ("model", "assistant") if role == "assistant" else ("user", "user")
            )
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    if not function.get("name"):
                        continue
                    args = function.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    fc = {
                        "name": tool_name(function["name"]),
                        "args": args if isinstance(args, dict) else {},
                    }
                    if resolved.family == "claude" and call.get("id"):
                        fc["id"] = call["id"]
                    part = {"functionCall": fc}
                    signature = call.get("thought_signature") or signatures.get(
                        "tool", call.get("id", ""), resolved.family
                    )
                    if signature:
                        part["thoughtSignature"] = signature
                    elif resolved.family == "gemini":
                        part["thoughtSignature"] = SENTINEL
                    parts.append(part)
        if not parts:
            continue
        previous_category = (
            "tool"
            if contents and all("functionResponse" in p for p in contents[-1]["parts"])
            else "user"
        )
        if (
            target_role == "user"
            and contents
            and contents[-1]["role"] == "user"
            and category == previous_category
        ):
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": target_role, "parts": parts})
    if not contents:
        raise ValueError("messages produced no usable content")
    if contents[-1]["role"] == "model":
        contents.append({"role": "user", "parts": [{"text": "[Continue]"}]})
    if resolved.family == "claude" and resolved.thinking and names:
        system.append(
            {
                "text": "Interleaved thinking is enabled. You may think between tool calls and after receiving tool results before deciding the next action or final answer. Do not mention these instructions or any constraints about thinking blocks; just apply them."
            }
        )
    result = {"contents": contents}
    if system:
        result["systemInstruction"] = {"role": "user", "parts": system}
    declarations = []
    for tool in request.get("tools") or []:
        if tool.get("type", "function") != "function" or not tool.get("function"):
            continue
        fn = tool["function"]
        name = tool_name(fn.get("name", ""))
        if not name:
            continue
        declaration = {"name": name}
        if fn.get("description") is not None:
            declaration["description"] = fn["description"]
        if (params := sanitize(fn.get("parameters"))) is not None:
            declaration["parameters"] = params
        declarations.append(declaration)
    if declarations:
        choice = request.get("tool_choice")
        mode = "VALIDATED" if resolved.family == "claude" else "AUTO"
        allowed = None
        if isinstance(choice, str):
            mode = {"none": "NONE", "auto": "AUTO", "required": "ANY", "any": "ANY"}.get(
                choice, mode
            )
        elif isinstance(choice, dict) and (name := choice.get("function", {}).get("name")):
            mode, allowed = "ANY", [tool_name(name)]
        if resolved.family == "claude" and resolved.thinking and mode == "ANY":
            raise ValueError(
                "Claude thinking is incompatible with forced tool_choice; disable thinking or use auto"
            )
        calling = {"mode": mode}
        if allowed:
            calling["allowedFunctionNames"] = allowed
        result.update(
            tools=[{"functionDeclarations": declarations}],
            toolConfig={"functionCallingConfig": calling},
        )
    requested = request.get("max_completion_tokens")
    if requested is None:
        requested = request.get("max_tokens")
    limit = resolved.output_limit
    maximum = limit if requested is None else min(requested, limit)
    if requested is not None and resolved.thinking:
        reserve = resolved.budget if resolved.budget and resolved.budget > 0 else 4096
        if resolved.family == "claude" and resolved.budget and resolved.budget > 0:
            reserve = max(reserve * 2, 32000)
        maximum = max(maximum, min(reserve + 1024, limit))
    generation = {"maxOutputTokens": maximum}
    for source, target in (("temperature", "temperature"), ("top_p", "topP")):
        if request.get(source) is not None:
            generation[target] = request[source]
    if request.get("stop") is not None:
        generation["stopSequences"] = (
            [request["stop"]] if isinstance(request["stop"], str) else request["stop"]
        )
    if resolved.thinking and resolved.budget is not None:
        generation["thinkingConfig"] = (
            {"include_thoughts": True, "thinking_budget": max(0, resolved.budget)}
            if resolved.family == "claude"
            else {"includeThoughts": True, "thinkingBudget": resolved.budget}
        )
    result["generationConfig"] = generation
    return result


def responses_to_chat(request: dict) -> dict:
    if request.get("previous_response_id"):
        raise ValueError("previous_response_id is unsupported; send the conversation in input")
    messages = []
    if request.get("instructions"):
        messages.append({"role": "system", "content": request["instructions"]})
    items = request.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    if not isinstance(items, list):
        raise ValueError("input must be a string or an array of items")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("input items must be objects")
        kind = item.get("type", "message")
        if kind == "message":
            content = item.get("content")
            if isinstance(content, list):
                converted = []
                for part in content:
                    if part.get("type") in {"input_text", "output_text", "text", "summary_text"}:
                        converted.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") == "input_image":
                        converted.append(
                            {"type": "image_url", "image_url": {"url": part.get("image_url", "")}}
                        )
                content = converted
            messages.append({"role": item.get("role", "user"), "content": content})
        elif kind == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id", item.get("id", "")),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    ],
                }
            )
        elif kind == "function_call_output":
            output = item.get("output", "")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "name": item.get("name", ""),
                    "content": output
                    if isinstance(output, str)
                    else json.dumps(output, separators=(",", ":")),
                }
            )
        elif kind != "reasoning":
            raise ValueError(f"unsupported input item type: {kind}")
    chat = {
        "model": request.get("model"),
        "messages": messages,
        "stream": request.get("stream", False),
    }
    for source, target in (
        ("max_output_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("user", "user"),
        ("tool_choice", "tool_choice"),
    ):
        if source in request:
            chat[target] = request[source]
    choice = chat.get("tool_choice")
    if isinstance(choice, dict) and choice.get("name"):
        chat["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if isinstance(request.get("reasoning"), dict) and request["reasoning"].get("effort"):
        chat["reasoning_effort"] = request["reasoning"]["effort"]
    tools = []
    for tool in request.get("tools") or []:
        if tool.get("type", "function") == "function":
            fn = tool.get("function") or {
                k: v for k, v in tool.items() if k in {"name", "description", "parameters"}
            }
            tools.append({"type": "function", "function": fn})
    if tools:
        chat["tools"] = tools
    return ChatRequest.model_validate(chat).model_dump(exclude_none=True)
