use anyhow::Result;
use heed::types::{Bytes, Str};
use heed::{Database, EnvOpenOptions};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Persistent embedding cache backed by LMDB
///
/// Stores embeddings on disk so they survive process restarts.
/// Key: SHA-256(content_hash + model_short_name)
/// Value: bincode-serialized Vec<f32>
pub struct PersistentEmbeddingCache {
    env: heed::Env,
    db: Database<Str, Bytes>,
    model_key: String,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl PersistentEmbeddingCache {
    /// Open or create persistent cache at `db_path/embedding_cache/`
    pub fn new(db_path: &Path, model_short_name: &str) -> Result<Self> {
        let cache_path = db_path.join("embedding_cache");
        std::fs::create_dir_all(&cache_path)?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(2 * 1024 * 1024 * 1024) // 2GB max
                .max_dbs(1)
                .open(&cache_path)?
        };

        let mut wtxn = env.write_txn()?;
        let db: Database<Str, Bytes> = env.create_database(&mut wtxn, Some("embeddings"))?;
        wtxn.commit()?;

        Ok(Self {
            env,
            db,
            model_key: model_short_name.to_string(),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        })
    }

    /// Generate cache key from content hash and model
    fn cache_key(&self, content_hash: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content_hash.as_bytes());
        hasher.update(b":");
        hasher.update(self.model_key.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Get embedding from persistent cache
    pub fn get(&self, content_hash: &str) -> Option<Vec<f32>> {
        let key = self.cache_key(content_hash);
        let rtxn = self.env.read_txn().ok()?;
        match self.db.get(&rtxn, &key) {
            Ok(Some(bytes)) => match bincode::deserialize::<Vec<f32>>(bytes) {
                Ok(embedding) => {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    Some(embedding)
                }
                Err(_) => {
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    None
                }
            },
            _ => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Store embedding in persistent cache
    pub fn put(&self, content_hash: &str, embedding: &[f32]) -> Result<()> {
        let key = self.cache_key(content_hash);
        let bytes = bincode::serialize(embedding)?;
        let mut wtxn = self.env.write_txn()?;
        self.db.put(&mut wtxn, &key, &bytes)?;
        wtxn.commit()?;
        Ok(())
    }

    /// Batch put for efficiency (single transaction)
    pub fn put_batch(&self, items: &[(&str, &[f32])]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let mut wtxn = self.env.write_txn()?;
        for (content_hash, embedding) in items {
            let key = self.cache_key(content_hash);
            let bytes = bincode::serialize(embedding)?;
            self.db.put(&mut wtxn, &key, &bytes)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Get number of cached embeddings
    pub fn len(&self) -> Result<usize> {
        let rtxn = self.env.read_txn()?;
        Ok(self.db.len(&rtxn)? as usize)
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Get cache hit/miss statistics
    pub fn hits(&self) -> usize {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn misses(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_persistent_cache_put_get() {
        let temp = tempdir().unwrap();
        let cache = PersistentEmbeddingCache::new(temp.path(), "test-model").unwrap();

        let hash = "abc123";
        let embedding = vec![1.0_f32, 2.0, 3.0, 4.0];

        // Not in cache yet
        assert!(cache.get(hash).is_none());

        // Put and retrieve
        cache.put(hash, &embedding).unwrap();
        let retrieved = cache.get(hash).unwrap();
        assert_eq!(retrieved, embedding);
    }

    #[test]
    fn test_persistent_cache_model_isolation() {
        let temp = tempdir().unwrap();
        let cache1 = PersistentEmbeddingCache::new(temp.path(), "model-a").unwrap();
        let cache2 = PersistentEmbeddingCache::new(temp.path(), "model-b").unwrap();

        let hash = "same_hash";
        cache1.put(hash, &[1.0, 2.0]).unwrap();

        // Different model should not see the embedding
        assert!(cache2.get(hash).is_none());
        // Same model should
        assert!(cache1.get(hash).is_some());
    }

    #[test]
    fn test_persistent_cache_batch() {
        let temp = tempdir().unwrap();
        let cache = PersistentEmbeddingCache::new(temp.path(), "test").unwrap();

        let items: Vec<(&str, &[f32])> = vec![
            ("h1", &[1.0, 2.0]),
            ("h2", &[3.0, 4.0]),
            ("h3", &[5.0, 6.0]),
        ];

        cache.put_batch(&items).unwrap();
        assert_eq!(cache.len().unwrap(), 3);

        assert_eq!(cache.get("h1").unwrap(), vec![1.0, 2.0]);
        assert_eq!(cache.get("h2").unwrap(), vec![3.0, 4.0]);
        assert_eq!(cache.get("h3").unwrap(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_persistent_cache_persistence() {
        let temp = tempdir().unwrap();
        let path = temp.path().to_path_buf();

        // Write data
        {
            let cache = PersistentEmbeddingCache::new(&path, "test").unwrap();
            cache.put("hash1", &[1.0, 2.0, 3.0]).unwrap();
        }

        // Reopen and verify
        {
            let cache = PersistentEmbeddingCache::new(&path, "test").unwrap();
            let emb = cache.get("hash1").unwrap();
            assert_eq!(emb, vec![1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn test_persistent_cache_stats() {
        let temp = tempdir().unwrap();
        let cache = PersistentEmbeddingCache::new(temp.path(), "test").unwrap();

        cache.put("h1", &[1.0]).unwrap();

        cache.get("h1"); // hit
        cache.get("h2"); // miss
        cache.get("h1"); // hit

        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
    }
}
