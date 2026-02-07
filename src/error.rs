//! Typed error definitions for demongrep
//!
//! Provides structured error types for pattern matching and differentiated
//! error recovery, replacing raw `anyhow::Error` in critical paths.

use thiserror::Error;

/// Core error type for demongrep operations
#[derive(Error, Debug)]
pub enum DemongrepError {
    // === Embedding errors ===
    #[error("Failed to load embedding model '{model}': {source}")]
    ModelLoadError {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("Embedding failed: {source}")]
    EmbeddingError {
        #[source]
        source: anyhow::Error,
    },

    // === Database errors ===
    #[error("Database not found: {path}")]
    DatabaseNotFound { path: String },

    #[error("Database error: {source}")]
    DatabaseError {
        #[source]
        source: anyhow::Error,
    },

    // === Search errors ===
    #[error("Search failed: {reason}")]
    SearchError { reason: String },

    #[error("No databases available for search")]
    NoDatabases,

    // === Chunking errors ===
    #[error("Chunking failed for {path}: {source}")]
    ChunkingError {
        path: String,
        #[source]
        source: anyhow::Error,
    },

    // === Configuration errors ===
    #[error("Invalid configuration: {details}")]
    ConfigError { details: String },

    // === Concurrency errors ===
    #[error("Mutex lock failed (poisoned): {context}")]
    LockError { context: String },

    // === Reranking errors ===
    #[error("Reranking failed: {source}")]
    RerankError {
        #[source]
        source: anyhow::Error,
    },

    // === I/O errors ===
    #[error(transparent)]
    Io(#[from] std::io::Error),

    // === Fallback for gradual migration ===
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
