use anyhow::{anyhow, Result};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use notify_debouncer_full::{new_debouncer, DebounceEventResult, Debouncer, FileIdMap};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

/// Types of file system events we care about
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileEvent {
    /// File was created or modified
    Modified(PathBuf),
    /// File was deleted
    Deleted(PathBuf),
    /// File was renamed (from, to)
    Renamed(PathBuf, PathBuf),
}

/// File watcher for incremental indexing
///
/// Uses notify-debouncer-full for efficient debounced file watching.
/// Improvements over osgrep:
/// 1. Native Rust implementation (faster than Node.js chokidar)
/// 2. Built-in debouncing (configurable)
/// 3. Batched events for efficient processing
pub struct FileWatcher {
    root: PathBuf,
    debouncer: Option<Debouncer<RecommendedWatcher, FileIdMap>>,
    receiver: Option<Receiver<DebounceEventResult>>,
    ignore_patterns: Vec<String>,
}

impl FileWatcher {
    /// Create a new file watcher for the given root directory
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            debouncer: None,
            receiver: None,
            ignore_patterns: vec![
                ".git".to_string(),
                ".demongrep.db".to_string(),
                "node_modules".to_string(),
                "target".to_string(),
                ".venv".to_string(),
                "__pycache__".to_string(),
                "*.lock".to_string(),
                "*.pyc".to_string(),
            ],
        }
    }

    /// Add patterns to ignore
    pub fn with_ignore_patterns(mut self, patterns: Vec<String>) -> Self {
        self.ignore_patterns.extend(patterns);
        self
    }

    /// Start watching for file changes
    pub fn start(&mut self, debounce_ms: u64) -> Result<()> {
        let (tx, rx) = channel();

        let debouncer = new_debouncer(
            Duration::from_millis(debounce_ms),
            None, // No tick rate
            tx,
        ).map_err(|e| anyhow!("Failed to create file watcher: {}", e))?;

        self.receiver = Some(rx);
        self.debouncer = Some(debouncer);

        // Start watching the root directory
        if let Some(ref mut debouncer) = self.debouncer {
            debouncer.watcher().watch(&self.root, RecursiveMode::Recursive)
                .map_err(|e| anyhow!("Failed to watch directory: {}", e))?;

            // Also watch with the cache (for file ID tracking)
            debouncer.cache().add_root(&self.root, RecursiveMode::Recursive);
        }

        Ok(())
    }

    /// Stop watching
    pub fn stop(&mut self) {
        if let Some(ref mut debouncer) = self.debouncer {
            let _ = debouncer.watcher().unwatch(&self.root);
        }
        self.debouncer = None;
        self.receiver = None;
    }

    /// Check if a path should be ignored
    fn should_ignore(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        // Special case: allow .gitignore files even if they match other patterns
        if let Some(name) = path.file_name() {
            let name_str = name.to_string_lossy();
            if name_str == ".gitignore" {
                return false;
            }
        }

        for pattern in &self.ignore_patterns {
            if pattern.starts_with('*') {
                // Extension pattern like "*.lock"
                let ext = &pattern[1..];
                if path_str.ends_with(ext) {
                    return true;
                }
            } else {
                // Directory/file name pattern
                if path_str.contains(pattern) {
                    return true;
                }
            }
        }

        // Ignore hidden files (except .gitignore which was handled above)
        if let Some(name) = path.file_name() {
            let name_str = name.to_string_lossy();
            if name_str.starts_with('.') {
                return true;
            }
        }

        false
    }

    /// Poll for file events (non-blocking)
    /// Returns a batch of deduplicated events
    pub fn poll_events(&self) -> Vec<FileEvent> {
        let Some(ref receiver) = self.receiver else {
            return vec![];
        };

        let mut events = Vec::new();
        let mut seen_paths = HashSet::new();

        // Drain all available events
        while let Ok(result) = receiver.try_recv() {
            match result {
                Ok(debounced_events) => {
                    for event in debounced_events {
                        for path in &event.paths {
                            // Skip ignored paths
                            if self.should_ignore(path) {
                                continue;
                            }

                            // Skip duplicates
                            if seen_paths.contains(path) {
                                continue;
                            }
                            seen_paths.insert(path.clone());

                            // Convert to our event type
                            use notify::EventKind;
                            match event.kind {
                                EventKind::Create(_) | EventKind::Modify(_) => {
                                    if path.exists() {
                                        events.push(FileEvent::Modified(path.clone()));
                                    }
                                }
                                EventKind::Remove(_) => {
                                    events.push(FileEvent::Deleted(path.clone()));
                                }
                                _ => {}
                            }
                        }
                    }
                }
                Err(errors) => {
                    for error in errors {
                        tracing::warn!("File watch error: {:?}", error);
                    }
                }
            }
        }

        events
    }

    /// Block and wait for events (with timeout)
    pub fn wait_for_events(&self, timeout: Duration) -> Vec<FileEvent> {
        let Some(ref receiver) = self.receiver else {
            return vec![];
        };

        let mut events = Vec::new();
        let mut seen_paths = HashSet::new();

        // Wait for first event
        match receiver.recv_timeout(timeout) {
            Ok(result) => {
                self.process_debounce_result(result, &mut events, &mut seen_paths);
            }
            Err(_) => return events, // Timeout or disconnected
        }

        // Drain any additional events that came in
        while let Ok(result) = receiver.try_recv() {
            self.process_debounce_result(result, &mut events, &mut seen_paths);
        }

        events
    }

    fn process_debounce_result(
        &self,
        result: DebounceEventResult,
        events: &mut Vec<FileEvent>,
        seen_paths: &mut HashSet<PathBuf>,
    ) {
        match result {
            Ok(debounced_events) => {
                for event in debounced_events {
                    for path in &event.paths {
                        if self.should_ignore(path) || seen_paths.contains(path) {
                            continue;
                        }
                        seen_paths.insert(path.clone());

                        use notify::EventKind;
                        match event.kind {
                            EventKind::Create(_) | EventKind::Modify(_) => {
                                if path.exists() {
                                    events.push(FileEvent::Modified(path.clone()));
                                }
                            }
                            EventKind::Remove(_) => {
                                events.push(FileEvent::Deleted(path.clone()));
                            }
                            _ => {}
                        }
                    }
                }
            }
            Err(errors) => {
                for error in errors {
                    tracing::warn!("File watch error: {:?}", error);
                }
            }
        }
    }
}

impl Drop for FileWatcher {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_should_ignore() {
        let watcher = FileWatcher::new(PathBuf::from("/tmp"));

        assert!(watcher.should_ignore(Path::new("/tmp/.git/config")));
        assert!(watcher.should_ignore(Path::new("/tmp/node_modules/foo")));
        assert!(watcher.should_ignore(Path::new("/tmp/Cargo.lock")));
        assert!(watcher.should_ignore(Path::new("/tmp/.hidden_file")));

        assert!(!watcher.should_ignore(Path::new("/tmp/src/main.rs")));
        assert!(!watcher.should_ignore(Path::new("/tmp/.gitignore")));
    }

    #[test]
    #[ignore] // Requires actual filesystem events
    fn test_file_watcher() {
        let dir = tempdir().unwrap();
        let mut watcher = FileWatcher::new(dir.path().to_path_buf());

        watcher.start(100).unwrap();

        // Create a file
        let test_file = dir.path().join("test.rs");
        fs::write(&test_file, "fn main() {}").unwrap();

        // Wait for events
        std::thread::sleep(Duration::from_millis(200));
        let events = watcher.poll_events();

        assert!(!events.is_empty());
    }
}
