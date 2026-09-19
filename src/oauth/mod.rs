//! OAuth 2.0: the interactive login flow and the token operations behind it.
//!
//! - [`pkce`] — verifier and challenge generation (RFC 7636).
//! - [`callback`] — the local redirect listener, plus parsing for a pasted
//!   callback when no listener can be used.
//! - [`token`] — token exchange, refresh, and failure classification.
//! - [`login`] — the end-to-end flow that ties the three together.

pub mod callback;
pub mod login;
pub mod pkce;
pub mod token;
