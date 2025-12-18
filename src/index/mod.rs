use anyhow::Result;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::chunker::SemanticChunker;
use crate::embed::{EmbeddingService, ModelType};
use crate::file::FileWalker;
use crate::fts::FtsStore;
use crate::vectordb::VectorStore;

/// Get the database path for a given project directory
fn get_db_path(path: Option<PathBuf>) -> Result<PathBuf> {
    let project_path = path.unwrap_or_else(|| PathBuf::from("."));
    let canonical_path = project_path.canonicalize()?;

    // Create database in the project directory
    Ok(canonical_path.join(".demongrep.db"))
}

/// Index a repository
pub async fn index(path: Option<PathBuf>, dry_run: bool, force: bool, model: Option<ModelType>) -> Result<()> {
    let project_path = path.clone().unwrap_or_else(|| PathBuf::from("."));
    let db_path = get_db_path(path)?;
    let model_type = model.unwrap_or_default();

    println!("{}", "🚀 Demongrep Indexer".bright_cyan().bold());
    println!("{}", "=".repeat(60));
    println!("📂 Project: {}", project_path.display());
    println!("💾 Database: {}", db_path.display());
    println!("🧠 Model: {} ({} dims)", model_type.name(), model_type.dimensions());

    if dry_run {
        println!("\n{}", "🔍 DRY RUN MODE".bright_yellow());
    }

    // Phase 1: File Discovery
    println!("\n{}", "Phase 1: File Discovery".bright_cyan());
    println!("{}", "-".repeat(60));

    let start = Instant::now();
    let walker = FileWalker::new(project_path.clone());
    let (files, stats) = walker.walk()?;
    let discovery_duration = start.elapsed();

    println!("✅ Found {} indexable files in {:?}", files.len(), discovery_duration);
    println!("   Total files scanned: {}", stats.total_files);
    println!("   Binary/skipped: {}", stats.skipped_binary);
    println!("   Total size: {:.2} MB", stats.total_size_mb());

    if files.is_empty() {
        println!("\n{}", "No files to index!".yellow());
        return Ok(());
    }

    if dry_run {
        println!("\n{}", "Dry run complete!".green());
        return Ok(());
    }

    // Check if database exists and handle force flag
    if db_path.exists() && !force {
        println!("\n{}", "⚠️  Database already exists!".yellow());
        println!("   Use --force to re-index");
        return Ok(());
    }

    // Clear existing database if forcing
    if db_path.exists() && force {
        println!("\n{}", "🗑️  Clearing existing database...".yellow());
        std::fs::remove_dir_all(&db_path)?;
    }

    // Phase 2: Semantic Chunking
    println!("\n{}", "Phase 2: Semantic Chunking".bright_cyan());
    println!("{}", "-".repeat(60));

    let start = Instant::now();
    let mut chunker = SemanticChunker::new(100, 2000, 10);
    let mut all_chunks = Vec::new();

    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("█▓▒░ "),
    );

    let mut skipped_files = 0;
    for file in &files {
        pb.set_message(format!("{}", file.path.file_name().unwrap().to_string_lossy()));

        // Skip files that aren't valid UTF-8
        let source_code = match std::fs::read_to_string(&file.path) {
            Ok(content) => content,
            Err(_) => {
                skipped_files += 1;
                pb.inc(1);
                continue;
            }
        };

        let chunks = chunker.chunk_semantic(file.language, &file.path, &source_code)?;
        all_chunks.extend(chunks);

        pb.inc(1);
    }

    if skipped_files > 0 {
        println!("   ⚠️  Skipped {} files (invalid UTF-8)", skipped_files);
    }

    pb.finish_with_message("Done!");
    let chunking_duration = start.elapsed();

    println!("✅ Created {} chunks in {:?}", all_chunks.len(), chunking_duration);

    if all_chunks.is_empty() {
        println!("\n{}", "No chunks created!".yellow());
        return Ok(());
    }

    // Phase 3: Embedding Generation
    println!("\n{}", "Phase 3: Embedding Generation".bright_cyan());
    println!("{}", "-".repeat(60));

    let start = Instant::now();
    println!("🔄 Initializing embedding model...");

    let mut embedding_service = EmbeddingService::with_model(model_type)?;
    println!("✅ Model loaded: {} ({} dims)", embedding_service.model_name(), embedding_service.dimensions());

    println!("\n🔄 Generating embeddings for {} chunks...", all_chunks.len());
    let embedded_chunks = embedding_service.embed_chunks(all_chunks)?;
    let embedding_duration = start.elapsed();

    println!("✅ Generated {} embeddings in {:?}", embedded_chunks.len(), embedding_duration);
    println!("   Average: {:?} per chunk", embedding_duration / embedded_chunks.len() as u32);

    // Show cache stats
    let cache_stats = embedding_service.cache_stats();
    println!("   Cache hit rate: {:.1}%", cache_stats.hit_rate() * 100.0);

    // Phase 4: Vector Storage
    println!("\n{}", "Phase 4: Vector Storage".bright_cyan());
    println!("{}", "-".repeat(60));

    let start = Instant::now();
    println!("🔄 Creating vector database...");

    let mut store = VectorStore::new(&db_path, embedding_service.dimensions())?;
    println!("✅ Database created");

    println!("\n🔄 Inserting {} chunks...", embedded_chunks.len());
    let chunk_ids = store.insert_chunks_with_ids(embedded_chunks.clone())?;
    println!("✅ Inserted {} chunks into vector store", chunk_ids.len());

    println!("\n🔄 Building vector index...");
    store.build_index()?;

    // Phase 4b: FTS Index
    println!("\n🔄 Building full-text search index...");
    let mut fts_store = FtsStore::new(&db_path)?;

    for (chunk, chunk_id) in embedded_chunks.iter().zip(chunk_ids.iter()) {
        fts_store.add_chunk(
            *chunk_id,
            &chunk.chunk.content,
            &chunk.chunk.path,
            chunk.chunk.signature.as_deref(),
            &format!("{:?}", chunk.chunk.kind),
        )?;
    }
    fts_store.commit()?;

    let fts_stats = fts_store.stats()?;
    println!("✅ FTS index built ({} documents)", fts_stats.num_documents);

    let storage_duration = start.elapsed();

    println!("✅ Index built in {:?}", storage_duration);

    // Save model metadata
    let metadata = serde_json::json!({
        "model_short_name": embedding_service.model_short_name(),
        "model_name": embedding_service.model_name(),
        "dimensions": embedding_service.dimensions(),
        "indexed_at": chrono::Utc::now().to_rfc3339(),
    });
    std::fs::write(
        db_path.join("metadata.json"),
        serde_json::to_string_pretty(&metadata)?
    )?;
    println!("✅ Metadata saved");

    // Show final stats
    let db_stats = store.stats()?;
    println!("\n{}", "📊 Final Statistics".bright_green().bold());
    println!("{}", "=".repeat(60));
    println!("   Total chunks: {}", db_stats.total_chunks);
    println!("   Total files: {}", db_stats.total_files);
    println!("   Indexed: {}", if db_stats.indexed { "✅ Yes" } else { "❌ No" });
    println!("   Dimensions: {}", db_stats.dimensions);

    // Calculate database size
    let mut total_size = 0u64;
    for entry in std::fs::read_dir(&db_path)? {
        let entry = entry?;
        total_size += entry.metadata()?.len();
    }
    println!("   Database size: {:.2} MB", total_size as f64 / (1024.0 * 1024.0));

    // Total time
    let total_duration = discovery_duration + chunking_duration + embedding_duration + storage_duration;
    println!("\n{}", "⏱️  Timing Breakdown".bright_green());
    println!("{}", "-".repeat(60));
    println!("   File discovery:      {:?}", discovery_duration);
    println!("   Semantic chunking:   {:?}", chunking_duration);
    println!("   Embedding generation:{:?}", embedding_duration);
    println!("   Vector storage:      {:?}", storage_duration);
    println!("   {}", format!("Total:               {:?}", total_duration).bold());

    println!("\n{}", "✨ Indexing complete!".bright_green().bold());
    println!("   Run {} to search your codebase", "demongrep search <query>".bright_cyan());

    Ok(())
}

/// List all indexed repositories
pub async fn list() -> Result<()> {
    println!("{}", "📚 Indexed Repositories".bright_cyan().bold());
    println!("{}", "=".repeat(60));

    // TODO: Scan all repositories in ~/.demongrep/repos.json
    // For now just check current directory

    // Check current directory
    let current_dir = std::env::current_dir()?;
    let current_db = current_dir.join(".demongrep.db");

    if current_db.exists() {
        println!("\n{}", "Current Directory:".bright_green());
        print_repo_stats(&current_dir, &current_db)?;
    }

    // TODO: Track indexed repositories globally in ~/.demongrep/repos.json
    // For now, just show current directory

    Ok(())
}

/// Show statistics about the vector database
pub async fn stats(path: Option<PathBuf>) -> Result<()> {
    let db_path = get_db_path(path)?;

    if !db_path.exists() {
        println!("{}", "❌ No database found!".red());
        println!("   Run {} first", "demongrep index".bright_cyan());
        return Ok(());
    }

    println!("{}", "📊 Database Statistics".bright_cyan().bold());
    println!("{}", "=".repeat(60));
    println!("💾 Database: {}", db_path.display());

    let store = VectorStore::new(&db_path, 384)?; // We'll need to store dimensions in metadata
    let stats = store.stats()?;

    println!("\n{}", "Vector Store:".bright_green());
    println!("   Total chunks: {}", stats.total_chunks);
    println!("   Total files: {}", stats.total_files);
    println!("   Indexed: {}", if stats.indexed { "✅ Yes" } else { "❌ No" });
    println!("   Dimensions: {}", stats.dimensions);

    // Calculate database size
    let mut total_size = 0u64;
    for entry in std::fs::read_dir(&db_path)? {
        let entry = entry?;
        total_size += entry.metadata()?.len();
    }

    println!("\n{}", "Storage:".bright_green());
    println!("   Database size: {:.2} MB", total_size as f64 / (1024.0 * 1024.0));
    println!("   Avg per chunk: {:.2} KB", (total_size as f64 / stats.total_chunks as f64) / 1024.0);

    Ok(())
}

/// Clear the vector database
pub async fn clear(path: Option<PathBuf>, yes: bool) -> Result<()> {
    let db_path = get_db_path(path)?;

    if !db_path.exists() {
        println!("{}", "❌ No database found!".red());
        return Ok(());
    }

    println!("{}", "🗑️  Clear Database".bright_yellow().bold());
    println!("{}", "=".repeat(60));
    println!("💾 Database: {}", db_path.display());

    if !yes {
        println!("\n{}", "⚠️  This will delete all indexed data!".yellow());
        print!("Are you sure? (y/N): ");
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("{}", "Cancelled.".dimmed());
            return Ok(());
        }
    }

    println!("\n🔄 Removing database...");
    std::fs::remove_dir_all(&db_path)?;

    println!("{}", "✅ Database cleared!".green());

    Ok(())
}

/// Helper to print repository stats
fn print_repo_stats(repo_path: &Path, db_path: &Path) -> Result<()> {
    println!("   📂 {}", repo_path.display());

    // Try to load stats
    match VectorStore::new(db_path, 384) {
        Ok(store) => {
            match store.stats() {
                Ok(stats) => {
                    println!("      {} chunks in {} files", stats.total_chunks, stats.total_files);
                }
                Err(_) => {
                    println!("      {}", "Could not load stats".dimmed());
                }
            }
        }
        Err(_) => {
            println!("      {}", "Could not open database".dimmed());
        }
    }

    Ok(())
}
