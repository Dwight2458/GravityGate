//! Model catalogue and name resolution.

pub mod models;
pub mod resolve;

pub use models::{ModelFamily, ModelSpec, ResolvedModel, ThinkingTier};
pub use resolve::{ResolveError, ResolveInput, resolve};
