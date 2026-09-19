//! Incremental decoder for the upstream event stream.
//!
//! The upstream speaks a deliberately plain dialect of SSE:
//!
//! - every useful line is `data: {json}`
//! - there are no `event:` names
//! - there is no `[DONE]` sentinel; end of stream is the terminator
//! - each payload wraps in `{"response": {...}, "traceId": "..."}`
//!
//! Two properties are load-bearing and were both learned the hard way.
//!
//! **One response spans several events.** A live `gemini-3.8-flash` reply came
//! back as one event carrying the answer text and a second carrying a
//! `thoughtSignature` on an *empty* text part alongside the finish reason. Any
//! consumer that treats one event as one response is wrong in a way that looks
//! fine: it produces a plausible answer with the signature and finish reason
//! quietly missing.
//!
//! **Chunk boundaries do not respect UTF-8 or line boundaries.** A multi-byte
//! character or a JSON payload can straddle two network chunks. Decoding each
//! chunk independently would corrupt both. Bytes are buffered and only complete
//! lines are decoded.

use serde_json::Value;

/// Cap on the undecoded buffer. A stream that produces this much data without a
/// newline is malformed or hostile; either way there is nothing useful to do
/// with it, and growing without bound would be a memory leak.
const MAX_BUFFER_BYTES: usize = 8 * 1024 * 1024;

/// One decoded event.
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// The `response` payload, unwrapped from its envelope.
    pub payload: Value,
    /// `traceId` from the envelope, when present. Useful in support requests.
    pub trace_id: Option<String>,
}

/// What one `push` produced.
#[derive(Debug, Default)]
pub struct Decoded {
    pub events: Vec<SseEvent>,
    /// Lines that began with `data:` but did not parse as JSON. Counted rather
    /// than fatal: one bad line should not discard a stream that is otherwise
    /// delivering content.
    pub malformed: usize,
}

impl Decoded {
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SseError {
    #[error("event stream exceeded {MAX_BUFFER_BYTES} bytes without a complete line")]
    BufferOverflow,
}

#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of bytes, returning whatever complete events it completed.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Decoded, SseError> {
        self.buffer.extend_from_slice(chunk);

        if self.buffer.len() > MAX_BUFFER_BYTES {
            return Err(SseError::BufferOverflow);
        }

        let mut decoded = Decoded::default();
        // Drain every complete line, leaving any partial tail in the buffer.
        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.buffer.drain(..=newline).collect();
            // Drop the trailing `\n`, and a `\r` if the upstream used CRLF.
            let line = &line[..line.len() - 1];
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            self.decode_line(line, &mut decoded);
        }

        Ok(decoded)
    }

    /// Flush a trailing line with no terminating newline.
    ///
    /// A stream that ends mid-line is unusual but not impossible, and a final
    /// event carrying only the finish reason is exactly the kind of thing that
    /// would be lost by discarding it.
    pub fn finish(&mut self) -> Result<Decoded, SseError> {
        let mut decoded = Decoded::default();
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            self.decode_line(&line, &mut decoded);
        }
        Ok(decoded)
    }

    /// Whether any partial line is held. Used to decide if `finish` will do
    /// anything.
    pub fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    fn decode_line(&self, line: &[u8], decoded: &mut Decoded) {
        // Lines that are not `data:` are comments, blank separators, or fields
        // this dialect does not use. All are ignored by design.
        let Some(rest) = line.strip_prefix(b"data:") else {
            return;
        };

        let Ok(text) = std::str::from_utf8(rest) else {
            // Invalid UTF-8 in a payload we have to hand to a JSON parser is
            // indistinguishable from a malformed line.
            decoded.malformed += 1;
            return;
        };

        let text = text.trim();
        if text.is_empty() {
            return;
        }

        let Ok(value) = serde_json::from_str::<Value>(text) else {
            decoded.malformed += 1;
            return;
        };

        decoded.events.push(unwrap_event(value));
    }
}

/// Split the `{response, traceId}` envelope.
///
/// A non-streaming body carries the payload at the top level, so the unwrap is
/// conditional on the `response` key actually being present.
pub fn unwrap_event(value: Value) -> SseEvent {
    let trace_id = value
        .get("traceId")
        .and_then(Value::as_str)
        .map(str::to_string);

    match value {
        Value::Object(mut object) => match object.remove("response") {
            Some(payload) => SseEvent { payload, trace_id },
            None => SseEvent {
                payload: Value::Object(object),
                trace_id,
            },
        },
        other => SseEvent {
            payload: other,
            trace_id,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn collect(decoder: &mut SseDecoder, chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(decoder.push(chunk).unwrap().events);
        }
        events.extend(decoder.finish().unwrap().events);
        events
    }

    #[test]
    fn a_single_event_is_decoded() {
        let mut decoder = SseDecoder::new();
        let events = collect(&mut decoder, &[b"data: {\"response\":{\"a\":1}}\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].payload["a"], 1);
    }

    #[test]
    fn the_response_envelope_is_unwrapped() {
        let mut decoder = SseDecoder::new();
        let events = collect(
            &mut decoder,
            &[b"data: {\"response\":{\"candidates\":[]},\"traceId\":\"abc\"}\n"],
        );
        assert_eq!(events[0].payload["candidates"], json!([]));
        assert_eq!(events[0].trace_id.as_deref(), Some("abc"));
        assert!(
            events[0].payload.get("response").is_none(),
            "the envelope must not survive the unwrap"
        );
    }

    #[test]
    fn an_unwrapped_payload_is_passed_through() {
        // A non-streaming body has no envelope; unwrapping must not eat it.
        let mut decoder = SseDecoder::new();
        let events = collect(&mut decoder, &[b"data: {\"candidates\":[]}\n"]);
        assert_eq!(events[0].payload["candidates"], json!([]));
        assert!(events[0].trace_id.is_none());
    }

    #[test]
    fn a_payload_split_across_chunks_is_reassembled() {
        // The property that matters: the network boundary is not the event
        // boundary.
        let mut decoder = SseDecoder::new();

        let first = decoder.push(b"data: {\"respo").unwrap();
        assert!(first.is_empty(), "nothing is complete yet");

        let second = decoder.push(b"nse\":{\"a\":1}}\n").unwrap();
        assert_eq!(second.events.len(), 1);
        assert_eq!(second.events[0].payload["a"], 1);
    }

    #[test]
    fn multiple_events_in_one_chunk_are_all_decoded() {
        let mut decoder = SseDecoder::new();
        let events = collect(
            &mut decoder,
            &[b"data: {\"response\":{\"n\":1}}\n\ndata: {\"response\":{\"n\":2}}\n\n"],
        );
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].payload["n"], 1);
        assert_eq!(events[1].payload["n"], 2);
    }

    #[test]
    fn a_multibyte_character_split_across_chunks_survives() {
        // The failure this guards: decoding each chunk independently turns a
        // split code point into replacement characters.
        let payload = "data: {\"response\":{\"text\":\"日本語\"}}\n";
        let bytes = payload.as_bytes();
        let split = bytes
            .windows(3)
            .position(|window| window == "日".as_bytes())
            .expect("multi-byte character present")
            + 1; // split inside the first character

        let mut decoder = SseDecoder::new();
        let first = decoder.push(&bytes[..split]).unwrap();
        assert!(first.is_empty());

        let second = decoder.push(&bytes[split..]).unwrap();
        assert_eq!(second.events.len(), 1);
        assert_eq!(second.events[0].payload["text"], "日本語");
    }

    #[test]
    fn crlf_line_endings_are_tolerated() {
        let mut decoder = SseDecoder::new();
        let events = collect(&mut decoder, &[b"data: {\"response\":{\"a\":1}}\r\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].payload["a"], 1);
    }

    #[test]
    fn a_trailing_line_without_a_newline_is_flushed() {
        // Exactly the shape of a final event carrying only the finish reason.
        let mut decoder = SseDecoder::new();
        let events = collect(
            &mut decoder,
            &[b"data: {\"response\":{\"candidates\":[{\"finishReason\":\"STOP\"}]}}"],
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].payload["candidates"][0]["finishReason"], "STOP");
    }

    #[test]
    fn comments_and_unknown_fields_are_ignored() {
        let mut decoder = SseDecoder::new();
        let events = collect(
            &mut decoder,
            &[
                b": keep-alive\n",
                b"event: something\n",
                b"id: 42\n",
                b"retry: 1000\n",
                b"\n",
                b"data: {\"response\":{\"ok\":true}}\n",
            ],
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].payload["ok"], true);
    }

    #[test]
    fn malformed_lines_are_counted_not_fatal() {
        let mut decoder = SseDecoder::new();
        let first = decoder.push(b"data: {not json\n").unwrap();
        assert_eq!(first.malformed, 1);
        assert!(first.events.is_empty());

        // The stream keeps working afterwards.
        let second = decoder.push(b"data: {\"response\":{\"ok\":1}}\n").unwrap();
        assert_eq!(second.events.len(), 1);
        assert_eq!(second.malformed, 0);
    }

    #[test]
    fn invalid_utf8_is_counted_not_fatal() {
        let mut decoder = SseDecoder::new();
        let decoded = decoder.push(b"data: \xff\xfe\xfd\n").unwrap();
        assert_eq!(decoded.malformed, 1);
        assert!(decoded.events.is_empty());
    }

    #[test]
    fn empty_data_lines_are_skipped_silently() {
        let mut decoder = SseDecoder::new();
        let decoded = decoder.push(b"data:\n").unwrap();
        assert!(decoded.events.is_empty());
        assert_eq!(decoded.malformed, 0, "an empty payload is not an error");
    }

    #[test]
    fn leading_and_trailing_whitespace_in_a_payload_is_trimmed() {
        let mut decoder = SseDecoder::new();
        let events = collect(&mut decoder, &[b"data:   {\"response\":{\"a\":1}}  \n"]);
        assert_eq!(events[0].payload["a"], 1);
    }

    #[test]
    fn a_realistic_two_event_response_is_fully_decoded() {
        // The shape a live gemini-3.8-flash reply actually took: text first,
        // then a signature on an empty text part with the finish reason.
        let stream = concat!(
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]}}],",
            "\"usageMetadata\":{\"promptTokenCount\":9,\"thoughtsTokenCount\":0}},\"traceId\":\"t1\"}\n\n",
            "data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"thoughtSignature\":\"EvYDCvMDARFN\",\"text\":\"\"}]},",
            "\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":9,\"candidatesTokenCount\":2}},\"traceId\":\"t2\"}\n\n",
        );

        let mut decoder = SseDecoder::new();
        let events = collect(&mut decoder, &[stream.as_bytes()]);
        assert_eq!(events.len(), 2, "both events must be delivered");

        // Answer text lives in the first.
        assert_eq!(
            events[0].payload["candidates"][0]["content"]["parts"][0]["text"],
            "ok"
        );
        // Signature and finish reason live in the second, on an empty text part.
        let second_part = &events[1].payload["candidates"][0]["content"]["parts"][0];
        assert_eq!(second_part["text"], "");
        assert_eq!(second_part["thoughtSignature"], "EvYDCvMDARFN");
        assert_eq!(events[1].payload["candidates"][0]["finishReason"], "STOP");
    }

    #[test]
    fn no_done_sentinel_is_expected_or_produced() {
        // `[DONE]` is an OpenAI convention the upstream does not use; it would
        // fail JSON parsing and must not be treated as a terminator.
        let mut decoder = SseDecoder::new();
        let decoded = decoder.push(b"data: [DONE]\n").unwrap();
        assert!(decoded.events.is_empty());
        assert_eq!(decoded.malformed, 1, "it is counted as unparseable");
    }

    #[test]
    fn has_pending_tracks_partial_state() {
        let mut decoder = SseDecoder::new();
        assert!(!decoder.has_pending());
        decoder.push(b"data: {\"partial").unwrap();
        assert!(decoder.has_pending());
        decoder.push(b"\":1}\n").unwrap();
        assert!(!decoder.has_pending());
    }

    #[test]
    fn byte_at_a_time_delivery_still_produces_the_event() {
        // The pathological framing case: every byte its own chunk.
        let payload = "data: {\"response\":{\"text\":\"héllo\"}}\n";
        let mut decoder = SseDecoder::new();
        let mut events = Vec::new();
        for byte in payload.as_bytes() {
            events.extend(decoder.push(&[*byte]).unwrap().events);
        }
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].payload["text"], "héllo");
    }
}
