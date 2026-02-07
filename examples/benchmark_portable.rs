//! Portable benchmark tool for evaluating embedding models in any repository
//!
//! Usage: cargo run --release --example benchmark_portable -- [OPTIONS]
//!
//! This tool can be run in any repository to determine which embedding model
//! works best for that specific codebase.

use anyhow::Result;
use clap::Parser;
use demongrep::chunker::{Chunk, SemanticChunker};
use demongrep::embed::{FastEmbedder, ModelType};
use demongrep::file::FileWalker;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkConfig {
    /// Test queries with expected file patterns
    queries: Vec<TestQuery>,
    /// False positive test query (should have low scores)
    false_positive_query: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestQuery {
    /// Natural language query
    query: String,
    /// Expected file path pattern (substring match)
    expected_file: String,
    /// Description of what we're looking for
    description: String,
}

#[derive(Debug)]
struct BenchmarkResult {
    model: ModelType,
    model_load_time: Duration,
    index_time: Duration,
    chunks_created: usize,
    avg_query_time: Duration,
    accuracy: f32,
    avg_score: f32,
    false_positive_score: f32,
    query_results: Vec<QueryResult>,
}

#[derive(Debug, Clone)]
struct QueryResult {
    query: String,
    expected: String,
    found: String,
    score: f32,
    correct: bool,
}

#[derive(Parser)]
#[command(name = "benchmark-portable")]
#[command(about = "Portable benchmark tool for embedding models")]
struct Cli {
    /// Path to benchmark config file (JSON)
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Comma-separated list of models to test
    #[arg(short, long, default_value = "minilm-l6-q,bge-small,jina-code")]
    models: String,

    /// Output directory for results
    #[arg(short, long, default_value = "./demongrep-benchmarks")]
    output: PathBuf,

    /// Path to analyze
    #[arg(short, long, default_value = ".")]
    path: PathBuf,

    /// Auto-generate config if not exists
    #[arg(long, default_value = "true")]
    auto_config: bool,

    /// Number of chunks to limit (for faster testing)
    #[arg(long)]
    limit_chunks: Option<usize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    print_banner();

    // Load or generate config
    let config = load_or_create_config(&cli)?;

    println!("📋 Loaded {} test queries", config.queries.len());
    println!();

    // Parse models to test
    let models = parse_models(&cli.models)?;
    println!("🧪 Models to benchmark:");
    for model in &models {
        println!("   - {} ({} dims)", model.name(), model.dimensions());
    }
    println!();

    // Collect files and chunks
    println!("📂 Analyzing directory: {}", cli.path.display());
    let chunks = collect_chunks(&cli.path, cli.limit_chunks)?;
    println!("   Created {} chunks", chunks.len());
    println!();

    // Run benchmarks
    let mut results = Vec::new();

    for model_type in models {
        println!("═══════════════════════════════════════════════════════════════");
        println!("🧪 Testing: {}", model_type.name());
        println!("   Dimensions: {} | Quantized: {}",
            model_type.dimensions(),
            model_type.is_quantized()
        );
        println!("───────────────────────────────────────────────────────────────");

        match benchmark_model(model_type, &chunks, &config) {
            Ok(result) => {
                print_result(&result);
                results.push(result);
            }
            Err(e) => {
                println!("   ❌ Error: {}", e);
            }
        }
        println!();
    }

    // Generate reports
    print_summary(&results);
    save_reports(&results, &cli.output, &config)?;

    Ok(())
}

fn print_banner() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     DEMONGREP PORTABLE BENCHMARK TOOL                        ║");
    println!("║     Find the best embedding model for YOUR codebase          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

fn load_or_create_config(cli: &Cli) -> Result<BenchmarkConfig> {
    if let Some(config_path) = &cli.config {
        if config_path.exists() {
            println!("📄 Loading config from: {}", config_path.display());
            let content = fs::read_to_string(config_path)?;
            return Ok(serde_json::from_str(&content)?);
        }
    }

    if cli.auto_config {
        println!("🔧 Auto-generating benchmark config...");
        let config = auto_generate_config(&cli.path)?;

        // Save generated config
        let config_path = cli.output.join("benchmark-config.json");
        fs::create_dir_all(&cli.output)?;
        fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;
        println!("💾 Config saved to: {}", config_path.display());

        Ok(config)
    } else {
        anyhow::bail!("Config file not found and auto-config disabled")
    }
}

fn auto_generate_config(path: &Path) -> Result<BenchmarkConfig> {
    let walker = FileWalker::new(path.to_path_buf());
    let (files, _stats) = walker.walk()?;

    // Analyze file structure to generate intelligent queries
    let mut queries = Vec::new();
    let mut seen_patterns = std::collections::HashSet::new();

    // Look for common patterns
    for file in files.iter().take(100) {
        let path_str = file.path.to_string_lossy();

        // Main/entry point
        if (path_str.contains("main") || path_str.contains("entry") || path_str.contains("index"))
            && !seen_patterns.contains("entry") {
            queries.push(TestQuery {
                query: "main entry point where execution starts".to_string(),
                expected_file: "main".to_string(),
                description: "Application entry point".to_string(),
            });
            seen_patterns.insert("entry");
        }

        // Config
        if (path_str.contains("config") || path_str.contains("settings"))
            && !seen_patterns.contains("config") {
            queries.push(TestQuery {
                query: "configuration settings and parameters".to_string(),
                expected_file: "config".to_string(),
                description: "Configuration files".to_string(),
            });
            seen_patterns.insert("config");
        }

        // Error handling
        if path_str.contains("error") && !seen_patterns.contains("error") {
            queries.push(TestQuery {
                query: "error handling and exceptions".to_string(),
                expected_file: "error".to_string(),
                description: "Error handling".to_string(),
            });
            seen_patterns.insert("error");
        }

        // Database/models
        if (path_str.contains("model") || path_str.contains("db") || path_str.contains("database"))
            && !seen_patterns.contains("models") {
            queries.push(TestQuery {
                query: "database models and data structures".to_string(),
                expected_file: "model".to_string(),
                description: "Data models".to_string(),
            });
            seen_patterns.insert("models");
        }

        // API/Routes
        if (path_str.contains("api") || path_str.contains("route") || path_str.contains("endpoint"))
            && !seen_patterns.contains("api") {
            queries.push(TestQuery {
                query: "API endpoints and routes".to_string(),
                expected_file: "api".to_string(),
                description: "API endpoints".to_string(),
            });
            seen_patterns.insert("api");
        }

        // Utils/helpers
        if (path_str.contains("util") || path_str.contains("helper"))
            && !seen_patterns.contains("utils") {
            queries.push(TestQuery {
                query: "utility functions and helpers".to_string(),
                expected_file: "util".to_string(),
                description: "Utility functions".to_string(),
            });
            seen_patterns.insert("utils");
        }

        // Auth/security
        if (path_str.contains("auth") || path_str.contains("security"))
            && !seen_patterns.contains("auth") {
            queries.push(TestQuery {
                query: "authentication and security".to_string(),
                expected_file: "auth".to_string(),
                description: "Authentication".to_string(),
            });
            seen_patterns.insert("auth");
        }

        // Tests
        if path_str.contains("test") && !seen_patterns.contains("tests") {
            queries.push(TestQuery {
                query: "test cases and unit tests".to_string(),
                expected_file: "test".to_string(),
                description: "Test files".to_string(),
            });
            seen_patterns.insert("tests");
        }
    }

    // Ensure minimum queries
    if queries.is_empty() {
        queries = vec![
            TestQuery {
                query: "main function entry point".to_string(),
                expected_file: "main".to_string(),
                description: "Entry point".to_string(),
            },
            TestQuery {
                query: "configuration and settings".to_string(),
                expected_file: "config".to_string(),
                description: "Configuration".to_string(),
            },
            TestQuery {
                query: "error handling".to_string(),
                expected_file: "error".to_string(),
                description: "Error handling".to_string(),
            },
        ];
    }

    Ok(BenchmarkConfig {
        queries,
        false_positive_query: "kubernetes deployment yaml docker container orchestration".to_string(),
    })
}

fn parse_models(models_str: &str) -> Result<Vec<ModelType>> {
    let mut models = Vec::new();

    for model_str in models_str.split(',') {
        let model_str = model_str.trim();
        if let Some(model) = ModelType::from_str(model_str) {
            models.push(model);
        } else {
            println!("⚠️  Unknown model: {}, skipping", model_str);
        }
    }

    if models.is_empty() {
        models.push(ModelType::default());
    }

    Ok(models)
}

fn collect_chunks(path: &Path, limit: Option<usize>) -> Result<Vec<Chunk>> {
    let walker = FileWalker::new(path.to_path_buf());
    let (files, _stats) = walker.walk()?;

    let mut chunker = SemanticChunker::new(100, 4000, 5);
    let mut all_chunks = Vec::new();

    for file in files {
        if let Ok(content) = fs::read_to_string(&file.path) {
            if let Ok(chunks) = chunker.chunk_semantic(file.language, &file.path, &content) {
                all_chunks.extend(chunks);
                if let Some(limit) = limit {
                    if all_chunks.len() >= limit {
                        all_chunks.truncate(limit);
                        break;
                    }
                }
            }
        }
    }

    Ok(all_chunks)
}

fn benchmark_model(
    model_type: ModelType,
    chunks: &[Chunk],
    config: &BenchmarkConfig,
) -> Result<BenchmarkResult> {
    // 1. Load model
    let start = Instant::now();
    let mut embedder = FastEmbedder::with_model(model_type)?;
    let model_load_time = start.elapsed();
    println!("   ⏱️  Model load: {:?}", model_load_time);

    // 2. Create embeddings
    let start = Instant::now();
    let texts: Vec<String> = chunks
        .iter()
        .map(|c| {
            let context_str = c.context.join(" > ");
            format!(
                "{}\n{}\n{}",
                context_str,
                c.signature.as_deref().unwrap_or(""),
                c.content
            )
        })
        .collect();

    let embeddings = embedder.embed_batch(texts)?;
    let index_time = start.elapsed();
    println!("   ⏱️  Indexed {} chunks: {:?}", chunks.len(), index_time);

    // 3. Run accuracy tests
    let mut query_results = Vec::new();
    let mut query_times = Vec::new();

    for test_query in &config.queries {
        let start = Instant::now();
        let query_embedding = embedder.embed_one(&test_query.query)?;
        query_times.push(start.elapsed());

        // Find best match
        let mut best_score = 0.0f32;
        let mut best_idx = 0;

        for (i, emb) in embeddings.iter().enumerate() {
            let score = cosine_similarity(&query_embedding, emb);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        let best_chunk = &chunks[best_idx];
        let is_correct = best_chunk.path.contains(&test_query.expected_file);

        query_results.push(QueryResult {
            query: test_query.query.clone(),
            expected: test_query.expected_file.clone(),
            found: best_chunk.path.clone(),
            score: best_score,
            correct: is_correct,
        });

        println!(
            "   {} \"{}\" -> {} (score: {:.3})",
            if is_correct { "✅" } else { "❌" },
            &test_query.query[..test_query.query.len().min(40)],
            best_chunk.path.split('/').last().unwrap_or(&best_chunk.path),
            best_score
        );
    }

    // 4. Test false positive
    let query_embedding = embedder.embed_one(&config.false_positive_query)?;
    let mut false_positive_score = 0.0f32;
    for emb in &embeddings {
        let score = cosine_similarity(&query_embedding, emb);
        if score > false_positive_score {
            false_positive_score = score;
        }
    }
    println!(
        "   ⚠️  False positive: {:.3} (should be < 0.85)",
        false_positive_score
    );

    let correct_count = query_results.iter().filter(|r| r.correct).count();
    let accuracy = correct_count as f32 / query_results.len() as f32;
    let avg_score = query_results.iter().map(|r| r.score).sum::<f32>() / query_results.len() as f32;
    let avg_query_time = query_times.iter().sum::<Duration>() / query_times.len().max(1) as u32;

    Ok(BenchmarkResult {
        model: model_type,
        model_load_time,
        index_time,
        chunks_created: chunks.len(),
        avg_query_time,
        accuracy,
        avg_score,
        false_positive_score,
        query_results,
    })
}

fn print_result(result: &BenchmarkResult) {
    println!("───────────────────────────────────────────────────────────────");
    println!(
        "   📊 Accuracy: {:.0}% ({}/{}) | Avg Score: {:.3}",
        result.accuracy * 100.0,
        (result.accuracy * result.query_results.len() as f32) as usize,
        result.query_results.len(),
        result.avg_score
    );
    println!("   📊 Query time: {:?} | False positive: {:.3}",
        result.avg_query_time,
        result.false_positive_score
    );
}

fn print_summary(results: &[BenchmarkResult]) {
    if results.is_empty() {
        return;
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK SUMMARY                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Sort by accuracy, then by avg_score
    let mut sorted: Vec<_> = results.iter().collect();
    sorted.sort_by(|a, b| {
        b.accuracy
            .partial_cmp(&a.accuracy)
            .unwrap()
            .then(b.avg_score.partial_cmp(&a.avg_score).unwrap())
    });

    println!(
        "{:<20} {:>6} {:>8} {:>10} {:>12} {:>10}",
        "Model", "Dims", "Acc", "Score", "Index Time", "Query"
    );
    println!("{}", "─".repeat(75));

    for r in sorted {
        println!(
            "{:<20} {:>6} {:>7.0}% {:>10.3} {:>12.2?} {:>10.2?}",
            r.model.short_name(),
            r.model.dimensions(),
            r.accuracy * 100.0,
            r.avg_score,
            r.index_time,
            r.avg_query_time
        );
    }

    println!();

    // Recommendation
    if let Some(best) = sorted.first() {
        println!("🏆 RECOMMENDATION: {}", best.model.name());
        println!("   Accuracy: {:.0}% | Dimensions: {} | Speed: {:?}",
            best.accuracy * 100.0,
            best.model.dimensions(),
            best.index_time
        );
        println!();
        println!("   To use this model in your project:");
        println!("   demongrep index --model {}", best.model.short_name());
    }
}

fn save_reports(
    results: &[BenchmarkResult],
    output_dir: &Path,
    config: &BenchmarkConfig,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;

    // Save JSON results
    let json_path = output_dir.join("benchmark-results.json");
    let json_data: Vec<_> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "model": r.model.short_name(),
                "dimensions": r.model.dimensions(),
                "accuracy": r.accuracy,
                "avg_score": r.avg_score,
                "false_positive_score": r.false_positive_score,
                "model_load_time_ms": r.model_load_time.as_millis(),
                "index_time_ms": r.index_time.as_millis(),
                "avg_query_time_ms": r.avg_query_time.as_millis(),
                "chunks": r.chunks_created,
                "query_results": r.query_results.iter().map(|qr| {
                    serde_json::json!({
                        "query": qr.query,
                        "expected": qr.expected,
                        "found": qr.found,
                        "score": qr.score,
                        "correct": qr.correct
                    )
                }).collect::<Vec<_>>()
            })
        })
        .collect();
    fs::write(&json_path, serde_json::to_string_pretty(&json_data)?)?;
    println!("📄 JSON results: {}", json_path.display());

    // Save markdown report
    let md_path = output_dir.join("benchmark-report.md");
    let mut md = String::new();

    md.push_str("# Demongrep Model Benchmark Report\n\n");
    md.push_str(&format!("**Date**: {}\n\n", chrono::Local::now().format("%Y-%m-%d %H:%M")));

    // Summary table
    md.push_str("## Summary\n\n");
    md.push_str("| Model | Dims | Accuracy | Avg Score | Index Time | Query Time |\n");
    md.push_str("|-------|------|----------|-----------|------------|------------|\n");

    let mut sorted: Vec<_> = results.iter().collect();
    sorted.sort_by(|a, b| {
        b.accuracy
            .partial_cmp(&a.accuracy)
            .unwrap()
            .then(b.avg_score.partial_cmp(&a.avg_score).unwrap())
    });

    for r in sorted {
        md.push_str(&format!(
            "| {} | {} | {:.0}% | {:.3} | {:.2?} | {:.2?} |\n",
            r.model.short_name(),
            r.model.dimensions(),
            r.accuracy * 100.0,
            r.avg_score,
            r.index_time,
            r.avg_query_time
        ));
    }

    // Recommendation
    if let Some(best) = sorted.first() {
        md.push_str("\n## 🏆 Recommendation\n\n");
        md.push_str(&format!("**Best Model**: `{}`\n\n", best.model.name()));
        md.push_str(&format!("- **Accuracy**: {:.0}%\n", best.accuracy * 100.0));
        md.push_str(&format!("- **Dimensions**: {}\n", best.model.dimensions()));
        md.push_str(&format!("- **Quantized**: {}\n", best.model.is_quantized()));
        md.push_str(&format!("- **Indexing Time**: {:.2?}\n", best.index_time));
        md.push_str(&format!("- **Avg Query Time**: {:.2?}\n\n", best.avg_query_time));
        md.push_str("### Usage\n\n");
        md.push_str(&format!("```bash\ndemongrep index --model {}\n```\n\n", best.model.short_name()));
    }

    // Detailed results per model
    md.push_str("## Detailed Results\n\n");

    for r in results {
        md.push_str(&format!("### {}\n\n", r.model.name()));
        md.push_str(&format!("- **Dimensions**: {}\n", r.model.dimensions()));
        md.push_str(&format!("- **Accuracy**: {:.0}% ({}/{})\n",
            r.accuracy * 100.0,
            r.query_results.iter().filter(|qr| qr.correct).count(),
            r.query_results.len()
        ));
        md.push_str(&format!("- **Avg Score**: {:.3}\n", r.avg_score));
        md.push_str(&format!("- **False Positive Score**: {:.3}\n\n", r.false_positive_score));

        md.push_str("#### Query Results\n\n");
        md.push_str("| Query | Expected | Found | Score | Status |\n");
        md.push_str("|-------|----------|-------|-------|--------|\n");

        for qr in &r.query_results {
            md.push_str(&format!(
                "| {} | {} | {} | {:.3} | {} |\n",
                &qr.query[..qr.query.len().min(30)],
                qr.expected,
                qr.found.split('/').last().unwrap_or(&qr.found),
                qr.score,
                if qr.correct { "✅" } else { "❌" }
            ));
        }
        md.push_str("\n");
    }

    // Test queries used
    md.push_str("## Test Queries Used\n\n");
    for (i, q) in config.queries.iter().enumerate() {
        md.push_str(&format!("{}. **{}**\n", i + 1, q.description));
        md.push_str(&format!("   - Query: `{}`\n", q.query));
        md.push_str(&format!("   - Expected pattern: `{}`\n\n", q.expected_file));
    }

    fs::write(&md_path, md)?;
    println!("📄 Markdown report: {}", md_path.display());

    println!();
    println!("✅ Benchmark complete! Check the reports above.");

    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (mag_a * mag_b)
}
