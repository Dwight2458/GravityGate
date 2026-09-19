//! PKCE (RFC 7636) primitives for the authorization-code flow.
//!
//! Google's authorization server for this client requires `S256`, so the
//! challenge is always a SHA-256 of the verifier rather than the verifier itself.
//! The verifier never leaves the process except in the token exchange, and the
//! `state` value is what protects against a callback arriving from somewhere
//! other than the flow we started.

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand::Rng as _;
use sha2::{Digest, Sha256};

/// Bytes of entropy behind a verifier.
///
/// 48 bytes render to 64 base64url characters, comfortably inside RFC 7636's
/// 43–128 character range. Overshooting matters: a 96-byte draw renders to 128
/// characters *plus* what some encoders add, which lands outside the range and
/// is rejected by the authorization server.
const VERIFIER_BYTES: usize = 48;

/// Bytes of entropy behind `state`. Hex-encoded, so 32 characters.
const STATE_BYTES: usize = 16;

/// A verifier and its derived challenge.
#[derive(Clone)]
pub struct Pkce {
    verifier: String,
    challenge: String,
}

impl std::fmt::Debug for Pkce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The verifier is a secret for the lifetime of the flow; only its shape
        // is useful in a log line.
        f.debug_struct("Pkce")
            .field("verifier_len", &self.verifier.len())
            .field("challenge", &self.challenge)
            .finish()
    }
}

impl Pkce {
    /// Generate a fresh verifier and challenge.
    pub fn generate() -> Self {
        let mut bytes = [0u8; VERIFIER_BYTES];
        rand::rng().fill_bytes(&mut bytes);
        let verifier = URL_SAFE_NO_PAD.encode(bytes);

        // S256: BASE64URL(SHA256(ASCII(verifier)))
        let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));

        Self {
            verifier,
            challenge,
        }
    }

    /// The unhashed verifier, sent only in the token exchange.
    pub fn verifier(&self) -> &str {
        &self.verifier
    }

    /// The hashed challenge, sent in the authorization URL.
    pub fn challenge(&self) -> &str {
        &self.challenge
    }

    /// Rebuild from a known verifier, recomputing the challenge.
    ///
    /// Used by the resume path, where a flow started in an earlier process is
    /// completed from a callback URL plus the saved verifier.
    pub fn from_verifier(verifier: impl Into<String>) -> Self {
        let verifier = verifier.into();
        let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        Self {
            verifier,
            challenge,
        }
    }

    /// Whether the verifier satisfies RFC 7636's length bounds.
    pub fn is_valid_verifier(verifier: &str) -> bool {
        let len = verifier.len();
        (43..=128).contains(&len) && verifier.chars().all(is_unreserved)
    }
}

/// RFC 7636's `unreserved` character set for a verifier.
fn is_unreserved(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '-' | '.' | '_' | '~')
}

/// Generate an opaque `state` value.
pub fn generate_state() -> String {
    let mut bytes = [0u8; STATE_BYTES];
    rand::rng().fill_bytes(&mut bytes);
    to_hex(&bytes)
}

fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn verifier_length_is_within_spec() {
        // The lower bound is 43; the upper bound of 128 is what a naive
        // `rand -base64 96` overruns.
        for _ in 0..32 {
            let pkce = Pkce::generate();
            let len = pkce.verifier().len();
            assert!(
                (43..=128).contains(&len),
                "verifier length {len} is outside RFC 7636's range"
            );
        }
    }

    #[test]
    fn verifier_uses_only_unreserved_characters() {
        for _ in 0..32 {
            let pkce = Pkce::generate();
            assert!(
                pkce.verifier().chars().all(is_unreserved),
                "verifier contains a reserved character: {}",
                pkce.verifier()
            );
            assert!(pkce.challenge().chars().all(is_unreserved));
        }
    }

    #[test]
    fn challenge_is_the_base64url_sha256_of_the_verifier() {
        // Pin the exact S256 construction; a plain base64 (with padding, or the
        // standard alphabet) is rejected by Google with `invalid_request`.
        let pkce = Pkce::generate();
        let expected = URL_SAFE_NO_PAD.encode(Sha256::digest(pkce.verifier().as_bytes()));
        assert_eq!(pkce.challenge(), expected);
        assert!(!pkce.challenge().contains('='), "padding must be stripped");
        assert!(!pkce.challenge().contains('+'), "standard alphabet was used");
        assert!(!pkce.challenge().contains('/'), "standard alphabet was used");
    }

    #[test]
    fn challenge_length_matches_sha256_output() {
        let pkce = Pkce::generate();
        // 32 bytes of digest render to 43 unpadded base64url characters.
        assert_eq!(pkce.challenge().len(), 43);
    }

    #[test]
    fn each_generation_is_distinct() {
        let a = Pkce::generate();
        let b = Pkce::generate();
        assert_ne!(a.verifier(), b.verifier());
        assert_ne!(a.challenge(), b.challenge());
    }

    #[test]
    fn from_verifier_reproduces_the_challenge() {
        let original = Pkce::generate();
        let rebuilt = Pkce::from_verifier(original.verifier().to_string());
        assert_eq!(rebuilt.challenge(), original.challenge());
    }

    #[test]
    fn verifier_validation_enforces_the_length_bounds() {
        assert!(!Pkce::is_valid_verifier(&"a".repeat(42)));
        assert!(Pkce::is_valid_verifier(&"a".repeat(43)));
        assert!(Pkce::is_valid_verifier(&"a".repeat(128)));
        assert!(!Pkce::is_valid_verifier(&"a".repeat(129)));
    }

    #[test]
    fn verifier_validation_rejects_reserved_characters() {
        let mut verifier = "a".repeat(50);
        verifier.push('+');
        assert!(!Pkce::is_valid_verifier(&verifier));

        let mut verifier = "a".repeat(50);
        verifier.push('=');
        assert!(!Pkce::is_valid_verifier(&verifier));
    }

    #[test]
    fn generated_verifiers_pass_validation() {
        for _ in 0..16 {
            let pkce = Pkce::generate();
            assert!(Pkce::is_valid_verifier(pkce.verifier()));
        }
    }

    #[test]
    fn state_is_hex_and_the_right_length() {
        let state = generate_state();
        assert_eq!(state.len(), STATE_BYTES * 2);
        assert!(state.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn state_values_are_distinct() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..64 {
            assert!(seen.insert(generate_state()), "state collided");
        }
    }

    #[test]
    fn debug_does_not_expose_the_verifier() {
        let pkce = Pkce::generate();
        let rendered = format!("{pkce:?}");
        assert!(
            !rendered.contains(pkce.verifier()),
            "the verifier leaked into Debug output"
        );
    }
}
