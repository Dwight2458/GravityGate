"""Incremental UTF-8 SSE framing with bounded event size and strict JSON."""

import codecs
import json
from collections.abc import AsyncIterator


class StreamError(Exception):
    pass


class SSEDecoder:
    def __init__(self, maximum: int = 1024 * 1024):
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.buffer = ""
        self.data: list[str] = []
        self.size = 0
        self.maximum = maximum
        self.done = False

    def feed(self, chunk: bytes, *, final: bool = False) -> list[dict]:
        try:
            self.buffer += self.decoder.decode(chunk, final=final)
        except UnicodeError as exc:
            raise StreamError("upstream stream contained invalid UTF-8") from exc
        events = []
        while True:
            positions = [p for p in (self.buffer.find("\r"), self.buffer.find("\n")) if p >= 0]
            if not positions:
                break
            end = min(positions)
            if self.buffer[end] == "\r" and end == len(self.buffer) - 1 and not final:
                break
            consumed = end + (2 if self.buffer[end : end + 2] == "\r\n" else 1)
            line, self.buffer = self.buffer[:end], self.buffer[consumed:]
            events.extend(self._line(line))
        if len(self.buffer.encode()) + self.size > self.maximum:
            raise StreamError("upstream SSE event exceeds byte limit")
        if final:
            if self.buffer:
                events.extend(self._line(self.buffer))
                self.buffer = ""
            events.extend(self._line(""))
        return events

    def _line(self, line: str) -> list[dict]:
        if not line:
            if not self.data:
                return []
            text = "\n".join(self.data)
            self.data, self.size = [], 0
            if text == "[DONE]":
                self.done = True
                return []
            if self.done:
                raise StreamError("data received after upstream DONE")
            try:
                value = json.loads(text)
            except ValueError as exc:
                raise StreamError("upstream SSE event contained invalid JSON") from exc
            if not isinstance(value, dict):
                raise StreamError("upstream SSE event must contain an object")
            return [value]
        if line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            self.data.append(value)
            self.size += len(value.encode()) + 1
            if self.size > self.maximum:
                raise StreamError("upstream SSE event exceeds byte limit")
        return []


async def decode_stream(
    chunks: AsyncIterator[bytes], content_type: str = "text/event-stream"
) -> AsyncIterator[dict]:
    if "application/json" in content_type:
        data = bytearray()
        async for chunk in chunks:
            data.extend(chunk)
            if len(data) > 16 * 1024 * 1024:
                raise StreamError("upstream JSON response exceeds byte limit")
        try:
            value = json.loads(data)
        except ValueError as exc:
            raise StreamError("invalid upstream JSON") from exc
        if not isinstance(value, dict):
            raise StreamError("upstream response must be an object")
        yield value
        return
    decoder = SSEDecoder()
    async for chunk in chunks:
        for event in decoder.feed(chunk):
            yield event
    for event in decoder.feed(b"", final=True):
        yield event


def encode_event(value: dict | str, event: str | None = None) -> bytes:
    data = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    )
    return ((f"event: {event}\n" if event else "") + f"data: {data}\n\n").encode()
