use anyhow::{anyhow, Result};
use ignore::gitignore::{Gitignore, GitignoreBuilder};
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
/// 4. Respects .gitignore, .demongrepignore, and .osgrepignore
pub struct FileWatcher {
    root: PathBuf,
    debouncer: Option<Debouncer<RecommendedWatcher, FileIdMap>>,
    receiver: Option<Receiver<DebounceEventResult>>,
    gitignore: Option<Gitignore>,
}

impl FileWatcher {
    /// Create a new file watcher for the given root directory
    pub fn new(root: PathBuf) -> Self {
        // Build gitignore matcher
        let gitignore = Self::build_gitignore(&root);

        Self {
            root,
            debouncer: None,
            receiver: None,
            gitignore,
        }
    }

    /// Build gitignore matcher from .gitignore, .demongrepignore, and .osgrepignore
    fn build_gitignore(root: &Path) -> Option<Gitignore> {
        let mut builder = GitignoreBuilder::new(root);

        // Add .gitignore
        let gitignore_path = root.join(".gitignore");
        if gitignore_path.exists() {
            let _ = builder.add(gitignore_path);
        }

        // Add .demongrepignore
        let demongrepignore_path = root.join(".demongrepignore");
        if demongrepignore_path.exists() {
            let _ = builder.add(demongrepignore_path);
        }

        // Add .osgrepignore (for compatibility)
        let osgrepignore_path = root.join(".osgrepignore");
        if osgrepignore_path.exists() {
            let _ = builder.add(osgrepignore_path);
        }

        // Add common ignore patterns
        let _ = builder.add_line(None, ".git");
        let _ = builder.add_line(None, ".demongrep.db");
        let _ = builder.add_line(None, "node_modules");
        let _ = builder.add_line(None, "target");
        let _ = builder.add_line(None, ".venv");
        let _ = builder.add_line(None, "__pycache__");
        let _ = builder.add_line(None, "*.dll");
        let _ = builder.add_line(None, "*.exe");
        let _ = builder.add_line(None, "*.so");
        let _ = builder.add_line(None, "*.dylib");
        let _ = builder.add_line(None, "*.pdb");
        let _ = builder.add_line(None, "*.lock");
        let _ = builder.add_line(None, "*.pyc");

        builder.build().ok()
    }

    /// Add custom ignore patterns (deprecated - use .demongrepignore instead)
    #[deprecated(note = "Use .demongrepignore file instead")]
    pub fn with_ignore_patterns(self, _patterns: Vec<String>) -> Self {
        self
    }

    /// Start watching for file changes
    pub fn start(&mut self, debounce_ms: u64) -> Result<()> {
        let (tx, rx) = channel();

        let debouncer = new_debouncer(
            Duration::from_millis(debounce_ms),
            None, // No tick rate
            tx,
        )
        .map_err(|e| anyhow!("Failed to create file watcher: {}", e))?;

        self.receiver = Some(rx);
        self.debouncer = Some(debouncer);

        // Start watching the root directory
        if let Some(ref mut debouncer) = self.debouncer {
            debouncer
                .watcher()
                .watch(&self.root, RecursiveMode::Recursive)
                .map_err(|e| anyhow!("Failed to watch directory: {}", e))?;

            // Also watch with the cache (for file ID tracking)
            debouncer
                .cache()
                .add_root(&self.root, RecursiveMode::Recursive);
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
        // Use gitignore matcher if available
        if let Some(ref gitignore) = self.gitignore {
            // Make path relative to root for gitignore matching
            let relative_path = if path.starts_with(&self.root) {
                path.strip_prefix(&self.root).unwrap_or(path)
            } else {
                path
            };

            // Check if path itself is ignored
            let is_dir = path.is_dir();
            match gitignore.matched(relative_path, is_dir) {
                ignore::Match::Ignore(_) => return true,
                ignore::Match::Whitelist(_) => return false,
                ignore::Match::None => {}
            }

            // Also check if any parent directory is ignored
            // This handles cases like .git/config where the file is inside an ignored directory
            let mut current = relative_path;
            while let Some(parent) = current.parent() {
                if !parent.as_os_str().is_empty() {
                    match gitignore.matched(parent, true) {
                        ignore::Match::Ignore(_) => return true,
                        ignore::Match::Whitelist(_) => return false,
                        ignore::Match::None => {}
                    }
                }
                current = parent;
            }
        }

        // Additional check: skip if file is binary (common binary extensions not in gitignore)
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            if matches!(
                ext_str.as_str(),
                "dll"
                    | "exe"
                    | "so"
                    | "dylib"
                    | "bin"
                    | "pdb"
                    | "obj"
                    | "o"
                    | "a"
                    | "lib"
                    | "class"
                    | "jar"
                    | "zip"
                    | "tar"
                    | "gz"
                    | "bz2"
                    | "xz"
                    | "7z"
                    | "png"
                    | "jpg"
                    | "jpeg"
                    | "gif"
                    | "ico"
                    | "svg"
                    | "pdf"
                    | "doc"
                    | "docx"
                    | "xls"
                    | "xlsx"
                    | "ppt"
                    | "pptx"
            ) {
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
        let dir = tempdir().unwrap();
        let watcher = FileWatcher::new(dir.path().to_path_buf());

        // Create some test paths
        let git_path = dir.path().join(".git/config");
        let node_modules_path = dir.path().join("node_modules/foo");
        let dll_path = dir.path().join("test.dll");
        let exe_path = dir.path().join("test.exe");
        let lock_path = dir.path().join("Cargo.lock");
        let rs_path = dir.path().join("src/main.rs");

        assert!(watcher.should_ignore(&git_path));
        assert!(watcher.should_ignore(&node_modules_path));
        assert!(watcher.should_ignore(&dll_path));
        assert!(watcher.should_ignore(&exe_path));
        assert!(watcher.should_ignore(&lock_path));
        assert!(!watcher.should_ignore(&rs_path));
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
