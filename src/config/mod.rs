//! Configuration management for demongrep
//!
//! Supports loading configuration from `.demongrep.toml` (project-local)
//! or `~/.demongrep/config.toml` (global), with sensible defaults.
//!
//! Priority: CLI flags > env vars > config file > defaults

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Project-level configuration loaded from .demongrep.toml
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ProjectConfig {
    pub embedding: EmbeddingConfig,
    pub chunking: ChunkingConfig,
    pub search: SearchConfig,
    pub database: DatabaseConfig,
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// Model name (e.g., "bge-small", "minilm-l6-q", "jina-code")
    pub model: String,
    /// Batch size for embedding
    pub batch_size: usize,
    /// Cache size in MB
    pub cache_size_mb: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "mxbai-embed-xsmall-v1".to_string(),
            batch_size: 32,
            cache_size_mb: 512,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkingConfig {
    /// Maximum chunk size in lines
    pub max_lines: usize,
    /// Maximum chunk size in characters
    pub max_chars: usize,
    /// Overlap between chunks in lines
    pub overlap_lines: usize,
    /// Lines of surrounding context to include
    pub context_lines: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_lines: 75,
            max_chars: 2000,
            overlap_lines: 10,
            context_lines: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// RRF k parameter for score fusion
    pub rrf_k: f32,
    /// Rerank weight for neural reranking blend
    pub rerank_weight: f32,
    /// Default maximum results
    pub default_limit: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            rrf_k: 20.0,
            rerank_weight: 0.575,
            default_limit: 25,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DatabaseConfig {
    /// Maximum database size in GB
    pub max_size_gb: usize,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self { max_size_gb: 10 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Default port for HTTP server
    pub port: u16,
    /// File watcher debounce in milliseconds
    pub debounce_ms: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 4444,
            debounce_ms: 300,
        }
    }
}

impl ProjectConfig {
    /// Load config with priority: project-local > CWD > global > defaults
    pub fn load(project_path: Option<&Path>) -> Self {
        // 1. Try project-local .demongrep.toml
        if let Some(path) = project_path {
            let config_path = path.join(".demongrep.toml");
            if let Ok(config) = Self::load_from_file(&config_path) {
                return config;
            }
        }

        // 2. Try CWD
        if let Ok(cwd) = std::env::current_dir() {
            let config_path = cwd.join(".demongrep.toml");
            if let Ok(config) = Self::load_from_file(&config_path) {
                return config;
            }
        }

        // 3. Try global ~/.demongrep/config.toml
        if let Some(home) = dirs::home_dir() {
            let config_path = home.join(".demongrep").join("config.toml");
            if let Ok(config) = Self::load_from_file(&config_path) {
                return config;
            }
        }

        // 4. Defaults
        Self::default()
    }

    fn load_from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: ProjectConfig = toml::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ProjectConfig::default();
        assert_eq!(config.embedding.model, "mxbai-embed-xsmall-v1");
        assert_eq!(config.search.rrf_k, 20.0);
        assert_eq!(config.server.port, 4444);
        assert_eq!(config.chunking.max_lines, 75);
        assert_eq!(config.database.max_size_gb, 10);
    }

    #[test]
    fn test_parse_partial_toml() {
        let toml_str = r#"
[search]
rrf_k = 30.0
default_limit = 15
"#;
        let config: ProjectConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.search.rrf_k, 30.0);
        assert_eq!(config.search.default_limit, 15);
        // Other fields should be defaults
        assert_eq!(config.embedding.model, "mxbai-embed-xsmall-v1");
        assert_eq!(config.server.port, 4444);
    }

    #[test]
    fn test_parse_full_toml() {
        let toml_str = r#"
[embedding]
model = "bge-small"
batch_size = 64
cache_size_mb = 1024

[chunking]
max_lines = 100
max_chars = 3000
overlap_lines = 15
context_lines = 5

[search]
rrf_k = 25.0
rerank_weight = 0.6
default_limit = 50

[database]
max_size_gb = 20

[server]
port = 8080
debounce_ms = 500
"#;
        let config: ProjectConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.embedding.model, "bge-small");
        assert_eq!(config.embedding.batch_size, 64);
        assert_eq!(config.chunking.max_lines, 100);
        assert_eq!(config.search.rrf_k, 25.0);
        assert_eq!(config.database.max_size_gb, 20);
        assert_eq!(config.server.port, 8080);
    }

    #[test]
    fn test_load_nonexistent_returns_defaults() {
        let config = ProjectConfig::load(Some(std::path::Path::new("/nonexistent/path")));
        assert_eq!(config.embedding.model, "mxbai-embed-xsmall-v1");
    }
}
