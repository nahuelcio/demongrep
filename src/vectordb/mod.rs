#![allow(unused_imports)]

use std::env;

mod store;

pub use store::{SearchResult, StoreStats, VectorStore};

/// Supported vector backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
	Arroy,
	#[cfg(feature = "zvec")]
	Zvec,
}

impl BackendKind {
	pub fn as_str(&self) -> &'static str {
		match self {
			BackendKind::Arroy => "arroy",
			#[cfg(feature = "zvec")]
			BackendKind::Zvec => "zvec",
		}
	}
}

/// Resolve the active backend from env/config.
///
/// `DEMONGREP_VECTOR_BACKEND=zvec` requires the binary to be built with
/// `--features zvec`; otherwise demongrep falls back to `arroy`.
pub fn selected_backend() -> BackendKind {
	let requested = env::var("DEMONGREP_VECTOR_BACKEND")
		.unwrap_or_else(|_| "arroy".to_string())
		.to_lowercase();

	if requested == "zvec" {
		#[cfg(feature = "zvec")]
		{
			return BackendKind::Zvec;
		}
	}

	BackendKind::Arroy
}

pub fn requested_backend() -> Option<String> {
	env::var("DEMONGREP_VECTOR_BACKEND").ok()
}
