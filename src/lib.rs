// Allow dead code - this is a library with public APIs for future use
#![allow(dead_code)]

pub mod bench;
pub mod cache;
pub mod chunker;
pub mod config;
pub mod database;
pub mod embed;
pub mod error;
pub mod file;
pub mod fts;
pub mod index;
pub mod mcp;
pub mod output;
pub mod rerank;
pub mod search;
pub mod server;
pub mod vectordb;
pub mod watch;

// Re-export commonly used types
pub use chunker::{Chunk, ChunkKind, Chunker};
pub use config::ProjectConfig;
pub use database::{CombinedStats, Database, DatabaseManager, DatabaseType};
pub use embed::{
    CacheStats, EmbeddedChunk, EmbeddingService, FastEmbedder, ModelType, PersistentEmbeddingCache,
};
pub use error::DemongrepError;
pub use file::{FileInfo, FileWalker, Language, WalkStats};
pub use fts::{FtsResult, FtsStore};
pub use vectordb::{SearchResult, StoreStats, VectorStore};
