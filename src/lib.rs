//! GravityGate — an OpenAI-compatible gateway in front of Google Antigravity's
//! Cloud Code Assist backend.
//!
//! The crate is split into two halves that meet at a single boundary:
//!
//! - [`upstream`] speaks the Google wire protocol: identity, envelope
//!   construction, transport, and the metadata that makes requests look like
//!   real CLI traffic.
//! - [`transform`] holds the canonical IR and the translations between it and
//!   the client-facing protocols.
//!
//! The IR is the Google Generative AI shape rather than Anthropic's Messages
//! shape, because the upstream is the only fixed constraint: a client request
//! crosses exactly one translation boundary, and thinking signatures have a
//! native home on the parts that carry them.

pub mod accounts;
pub mod config;
pub mod engine;
pub mod oauth;
pub mod registry;
pub mod transform;
pub mod upstream;
