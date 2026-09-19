//! M0 feasibility spike: can `reqwest`/`hyper` reproduce the wire identity that
//! the Antigravity CLI emits?
//!
//! The reference project `antigravity-auth` hand-rolls HTTP/1.1 over `tls.connect`
//! instead of using the platform HTTP client, citing header ordering and connection
//! handling. The captured `agy` CLI 1.1.24 request (see
//! `reference/antigravity-auth/test-fixtures/agy-cli-1.1.24-stream-request.json`)
//! records this exact header block:
//!
//! ```text
//! Host: daily-cloudcode-pa.googleapis.com
//! User-Agent: antigravity/cli/1.1.24 (aidev_client; os_type=darwin; arch=arm64; cl=974782877; auth_method=consumer)
//! Transfer-Encoding: chunked
//! Authorization: <redacted>
//! Content-Type: application/json
//! Accept-Encoding: gzip
//! ```
//!
//! This spike runs entirely against a local listener and answers three questions
//! without needing credentials:
//!
//!   1. Does `http::HeaderMap` iterate in insertion order (hyper serializes in
//!      iteration order, so this decides whether ordering is controllable at all)?
//!   2. What does the raw request block actually look like on the wire — header
//!      order, casing, and which headers reqwest injects that the CLI never sent?
//!   3. Does a streamed body produce `Transfer-Encoding: chunked` rather than a
//!      buffered `Content-Length`?
//!
//! Run with `cargo run --example spike_http`.

use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Byte-for-byte the User-Agent captured from `agy` CLI 1.1.24 on 2026-09-02.
const AGY_CLI_UA: &str = "antigravity/cli/1.1.24 (aidev_client; os_type=darwin; arch=arm64; cl=974782877; auth_method=consumer)";

/// Read one HTTP request off `socket` and return its raw bytes, then answer `200 {}`
/// so the client does not report a transport error.
async fn capture_request(socket: &mut tokio::net::TcpStream) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];

    // Phase 1: everything up to the blank line that ends the header block.
    loop {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }

    // Phase 2: the body, if any. A short idle timeout marks the end of a chunked
    // body without needing to parse chunk framing.
    loop {
        match tokio::time::timeout(Duration::from_millis(400), socket.read(&mut tmp)).await {
            Ok(Ok(0)) | Ok(Err(_)) | Err(_) => break,
            Ok(Ok(n)) => buf.extend_from_slice(&tmp[..n]),
        }
    }

    let body = b"{}";
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    socket.write_all(response.as_bytes()).await?;
    socket.write_all(body).await?;
    socket.flush().await?;
    Ok(buf)
}

/// Render a raw request block with CRLF made visible, so ordering is unambiguous.
fn show_block(raw: &[u8]) -> String {
    let text = String::from_utf8_lossy(raw);
    let (head, rest) = match text.split_once("\r\n\r\n") {
        Some((h, r)) => (h, r),
        None => (text.as_ref(), ""),
    };
    let mut out = String::new();
    for line in head.split("\r\n") {
        out.push_str("    ");
        out.push_str(line);
        out.push('\n');
    }
    if !rest.is_empty() {
        out.push_str("    -- body --\n");
        for line in rest.split("\r\n").take(8) {
            if line.is_empty() {
                continue;
            }
            out.push_str("    ");
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Question 1: is `http::HeaderMap` iteration order insertion order?
fn probe_headermap_order() -> bool {
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

    let names = [
        "user-agent",
        "authorization",
        "content-type",
        "accept-encoding",
    ];

    let mut map = HeaderMap::new();
    for name in names {
        map.insert(
            HeaderName::from_static(name),
            HeaderValue::from_static("x"),
        );
    }

    let observed: Vec<&str> = map.keys().map(|k| k.as_str()).collect();
    let preserved = observed == names;

    println!("  inserted : {:?}", names);
    println!("  iterated : {:?}", observed);
    println!("  verdict  : {}", if preserved { "PRESERVED" } else { "SCRAMBLED" });
    preserved
}

/// Question 2 + 3: what actually reaches the socket for a streamed request?
async fn probe_wire(streaming: bool) -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await?;
        capture_request(&mut socket).await
    });

    let client = reqwest::Client::builder().http1_only().build()?;

    // Insert in exactly the captured order, skipping Host (hyper owns that).
    let mut request = client
        .post(format!("http://{addr}/v1internal:streamGenerateContent?alt=sse"))
        .header(reqwest::header::USER_AGENT, AGY_CLI_UA)
        .header(reqwest::header::AUTHORIZATION, "Bearer REDACTED")
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::ACCEPT_ENCODING, "gzip");

    if streaming {
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from_static(b"{\"project\":\"p\",")),
            Ok(Bytes::from_static(b"\"model\":\"m\"}")),
        ];
        let stream = futures::stream::iter(chunks);
        request = request.body(reqwest::Body::wrap_stream(stream));
    } else {
        request = request.body(r#"{"project":"p","model":"m"}"#);
    }

    let response = request.send().await?;
    let status = response.status();

    // Drain so the connection closes cleanly.
    let _ = response.bytes().await;

    let raw = server.await??;
    println!("  status   : {status}");
    println!("  raw request block:");
    print!("{}", show_block(&raw));

    Ok(())
}

/// Question 4: which headers does reqwest/hyper inject that we did not ask for?
async fn probe_injected(extra_accept: bool) -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await?;
        capture_request(&mut socket).await
    });

    let client = reqwest::Client::builder().http1_only().build()?;

    // Deliberately minimal: only User-Agent, so anything else on the wire is injected.
    let mut request = client
        .post(format!("http://{addr}/v1internal:generateContent"))
        .header(reqwest::header::USER_AGENT, AGY_CLI_UA);

    if extra_accept {
        // Try to occupy the Accept slot explicitly to see if that suppresses the default.
        request = request.header(reqwest::header::ACCEPT, "");
    }

    let response = request.body(r#"{"project":"p","model":"m"}"#).send().await?;
    let status = response.status();
    let _ = response.bytes().await;

    let raw = server.await??;
    println!(
        "  status   : {status}   (explicit empty Accept: {extra_accept})"
    );
    println!("  raw request block:");
    print!("{}", show_block(&raw));

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("== M0 spike: reqwest/hyper wire fidelity ==\n");

    println!("[1] http::HeaderMap iteration order");
    let order_ok = probe_headermap_order();

    println!("\n[2] streamed body (expect Transfer-Encoding: chunked)");
    probe_wire(true).await?;

    println!("\n[3] buffered body (expect Content-Length)");
    probe_wire(false).await?;

    println!("\n[4a] injected headers, no explicit Accept");
    probe_injected(false).await?;

    println!("\n[4b] injected headers, explicit empty Accept");
    probe_injected(true).await?;

    println!("\n== conclusions ==");
    println!(
        "  header order controllable: {}",
        if order_ok {
            "yes — HeaderMap preserves insertion order"
        } else {
            "NO — hyper would reorder; need a hand-rolled HTTP/1.1 writer"
        }
    );

    Ok(())
}
