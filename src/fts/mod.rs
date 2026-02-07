//! Full-text search module using Tantivy
//!
//! Provides BM25-based full-text search to complement vector similarity search.
//! Used in hybrid search mode with RRF (Reciprocal Rank Fusion).

mod code_tokenizer;
mod tantivy_store;

pub use tantivy_store::{FtsResult, FtsStore};
