//! Model catalogue and name resolution.

pub mod live;
pub mod models;
pub mod resolve;

pub use live::{LiveCatalogue, LiveModel};
pub use models::{ModelFamily, ModelSpec, ResolvedModel, ThinkingTier};
pub use resolve::{ResolveError, ResolveInput, resolve};
