//! Centralized database management for dual-database support
//!
//! This module provides a unified interface for working with both local and global
//! databases, eliminating code duplication across search, server, MCP, and index modules.

use anyhow::{anyhow, Result};
use std::path::PathBuf;

use crate::embed::ModelType;
use crate::index::get_search_db_paths;
use crate::vectordb::{SearchResult, VectorStore};

/// Type of database (local or global)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatabaseType {
    /// Local database in project directory (.demongrep.db)
    Local,
    /// Global database in home directory
    Global,
}

impl DatabaseType {
    pub fn name(&self) -> &str {
        match self {
            DatabaseType::Local => "Local",
            DatabaseType::Global => "Global",
        }
    }
}

/// A single database entry with metadata
pub struct Database {
    pub path: PathBuf,
    pub db_type: DatabaseType,
    store: VectorStore,
}

impl Database {
    /// Create a new database instance
    pub fn new(path: PathBuf, db_type: DatabaseType, dimensions: usize) -> Result<Self> {
        let store = VectorStore::new(&path, dimensions)?;
        Ok(Self {
            path,
            db_type,
            store,
        })
    }

    /// Get the vector store
    pub fn store(&self) -> &VectorStore {
        &self.store
    }

    /// Get mutable vector store
    pub fn store_mut(&mut self) -> &mut VectorStore {
        &mut self.store
    }
}

/// Combined statistics from all databases
#[derive(Debug, Clone, Default)]
pub struct CombinedStats {
    pub total_chunks: usize,
    pub total_files: usize,
    pub local_chunks: usize,
    pub local_files: usize,
    pub global_chunks: usize,
    pub global_files: usize,
    pub indexed: bool,
    pub dimensions: usize,
}

/// Centralized database manager for handling multiple databases
pub struct DatabaseManager {
    databases: Vec<Database>,
    model_type: ModelType,
    dimensions: usize,
}

impl DatabaseManager {
    /// Load all available databases for a given path
    pub fn load(path: Option<PathBuf>) -> Result<Self> {
        let db_paths = get_search_db_paths(path)?;

        if db_paths.is_empty() {
            return Err(anyhow!("No databases found"));
        }

        // Read metadata from first database
        let (model_type, dimensions) = Self::read_metadata(&db_paths[0])
            .unwrap_or_else(|| (ModelType::default(), 384));

        // Load all databases
        let mut databases = Vec::new();
        for db_path in db_paths {
            let db_type = if db_path.ends_with(".demongrep.db") {
                DatabaseType::Local
            } else {
                DatabaseType::Global
            };

            match Database::new(db_path.clone(), db_type, dimensions) {
                Ok(db) => databases.push(db),
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to load {} database at {}: {}",
                        db_type.name(),
                        db_path.display(),
                        e
                    );
                }
            }
        }

        if databases.is_empty() {
            return Err(anyhow!("Failed to load any databases"));
        }

        Ok(Self {
            databases,
            model_type,
            dimensions,
        })
    }

    /// Get model type
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get all database paths
    pub fn database_paths(&self) -> Vec<&PathBuf> {
        self.databases.iter().map(|db| &db.path).collect()
    }

    /// Get number of databases
    pub fn database_count(&self) -> usize {
        self.databases.len()
    }

    /// Check if a local database exists
    pub fn has_local(&self) -> bool {
        self.databases.iter().any(|db| db.db_type == DatabaseType::Local)
    }

    /// Check if a global database exists
    pub fn has_global(&self) -> bool {
        self.databases.iter().any(|db| db.db_type == DatabaseType::Global)
    }

    /// Get local database if it exists
    pub fn local_database(&self) -> Option<&Database> {
        self.databases.iter().find(|db| db.db_type == DatabaseType::Local)
    }

    /// Get local database mutably if it exists
    pub fn local_database_mut(&mut self) -> Option<&mut Database> {
        self.databases.iter_mut().find(|db| db.db_type == DatabaseType::Local)
    }

    /// Get all databases
    pub fn databases(&self) -> &[Database] {
        &self.databases
    }

    /// Get all databases mutably
    pub fn databases_mut(&mut self) -> &mut [Database] {
        &mut self.databases
    }

    /// Search across all databases
    pub fn search_all(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        let mut all_results = Vec::new();

        for database in &self.databases {
            match database.store.search(query_embedding, limit) {
                Ok(mut results) => {
                    all_results.append(&mut results);
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Search failed in {} database: {}",
                        database.db_type.name(),
                        e
                    );
                }
            }
        }

        // Sort by score descending
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit total results
        all_results.truncate(limit);

        Ok(all_results)
    }

    /// Get combined statistics from all databases
    pub fn combined_stats(&self) -> Result<CombinedStats> {
        let mut stats = CombinedStats::default();

        for database in &self.databases {
            let db_stats = database.store.stats()?;

            stats.total_chunks += db_stats.total_chunks;
            stats.total_files += db_stats.total_files;
            stats.indexed = stats.indexed || db_stats.indexed;
            stats.dimensions = db_stats.dimensions;

            match database.db_type {
                DatabaseType::Local => {
                    stats.local_chunks += db_stats.total_chunks;
                    stats.local_files += db_stats.total_files;
                }
                DatabaseType::Global => {
                    stats.global_chunks += db_stats.total_chunks;
                    stats.global_files += db_stats.total_files;
                }
            }
        }

        Ok(stats)
    }

    /// Read metadata from a database
    fn read_metadata(db_path: &PathBuf) -> Option<(ModelType, usize)> {
        let metadata_path = db_path.join("metadata.json");
        
        if !metadata_path.exists() {
            return None;
        }

        let content = std::fs::read_to_string(&metadata_path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;

        let model_name = json
            .get("model_short_name")
            .and_then(|v| v.as_str())
            .unwrap_or("minilm-l6");

        let dimensions = json
            .get("dimensions")
            .and_then(|v| v.as_u64())
            .unwrap_or(384) as usize;

        let model_type = ModelType::from_str(model_name).unwrap_or_default();

        Some((model_type, dimensions))
    }

    /// Print database information
    pub fn print_info(&self) {
        use colored::Colorize;

        println!("{}", "📚 Available Databases:".bright_green());
        for database in &self.databases {
            println!(
                "   {} {}",
                match database.db_type {
                    DatabaseType::Local => "📍",
                    DatabaseType::Global => "🌍",
                },
                database.path.display()
            );
        }
    }
}

/// Builder for creating a DatabaseManager with specific databases
pub struct DatabaseManagerBuilder {
    db_paths: Vec<PathBuf>,
    model_type: Option<ModelType>,
    dimensions: Option<usize>,
}

impl DatabaseManagerBuilder {
    pub fn new() -> Self {
        Self {
            db_paths: Vec::new(),
            model_type: None,
            dimensions: None,
        }
    }

    pub fn add_database(mut self, path: PathBuf) -> Self {
        self.db_paths.push(path);
        self
    }

    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = Some(model_type);
        self
    }

    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    pub fn build(self) -> Result<DatabaseManager> {
        if self.db_paths.is_empty() {
            return Err(anyhow!("No database paths specified"));
        }

        // Determine model and dimensions
        let (model_type, dimensions) = if let (Some(mt), Some(dims)) = (self.model_type, self.dimensions) {
            (mt, dims)
        } else {
            DatabaseManager::read_metadata(&self.db_paths[0])
                .unwrap_or_else(|| (ModelType::default(), 384))
        };

        // Load all databases
        let mut databases = Vec::new();
        for db_path in self.db_paths {
            let db_type = if db_path.ends_with(".demongrep.db") {
                DatabaseType::Local
            } else {
                DatabaseType::Global
            };

            match Database::new(db_path.clone(), db_type, dimensions) {
                Ok(db) => databases.push(db),
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to load {} database at {}: {}",
                        db_type.name(),
                        db_path.display(),
                        e
                    );
                }
            }
        }

        if databases.is_empty() {
            return Err(anyhow!("Failed to load any databases"));
        }

        Ok(DatabaseManager {
            databases,
            model_type,
            dimensions,
        })
    }
}

impl Default for DatabaseManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_type() {
        assert_eq!(DatabaseType::Local.name(), "Local");
        assert_eq!(DatabaseType::Global.name(), "Global");
    }

    #[test]
    fn test_combined_stats_default() {
        let stats = CombinedStats::default();
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_files, 0);
    }
}
