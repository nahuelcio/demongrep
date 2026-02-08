//! Benchmark framework for comparing embedding models.
//!
//! Measures performance (throughput, latency), quality (accuracy, false positives),
//! and memory (RSS delta, estimated DB size) across all supported models.

use anyhow::Result;
use colored::Colorize;
use rayon::prelude::*;
use serde::Serialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::chunker::{Chunk, SemanticChunker};
use crate::embed::{BatchEmbedder, FastEmbedder, ModelType};
use crate::file::FileWalker;

/// Built-in test queries with expected file path substrings
const TEST_QUERIES: &[(&str, &str)] = &[
    ("SemanticChunker struct", "src/chunker/semantic.rs"),
    ("VectorStore insert chunks", "src/vectordb/store.rs"),
    ("tree-sitter grammar loading", "src/chunker/parser.rs"),
    (
        "extract function signature from AST",
        "src/chunker/extractor.rs",
    ),
    ("how do we detect binary files", "src/file/binary.rs"),
    ("where is the main entry point", "src/main.rs"),
    ("CLI argument parsing clap", "src/cli/mod.rs"),
    ("FileWalker walk directory", "file_walker"),
    ("RRF fusion reranking", "src/rerank/mod.rs"),
];

const FALSE_POSITIVE_QUERY: &str = "kubernetes deployment yaml helm chart";

#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub model_name: String,
    pub short_name: String,
    pub dimensions: usize,
    pub quantized: bool,
    // Performance
    pub model_load_ms: u64,
    pub embed_throughput: f32,
    pub avg_query_ms: f64,
    pub total_index_ms: u64,
    // Quality
    pub accuracy: f32,
    pub avg_score: f32,
    pub false_positive_score: f32,
    // Memory
    pub rss_delta_mb: f64,
    pub estimated_db_mb: f64,
    pub chunks_count: usize,
}

/// Get current process RSS in MB (macOS + Linux)
fn get_rss_mb() -> f64 {
    let pid = std::process::id().to_string();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok();
    output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

/// Benchmark a single model against pre-chunked data
fn benchmark_model(
    model_type: ModelType,
    chunks: &[Chunk],
    prepared_texts: &[String],
) -> Result<BenchResult> {
    let chunks_count = chunks.len();

    // 1. Measure RSS before model load
    let rss_before = get_rss_mb();

    // 2. Load model
    let start = Instant::now();
    let mut embedder = FastEmbedder::with_model(model_type)?;
    let model_load_ms = start.elapsed().as_millis() as u64;

    // 3. Measure RSS after model load
    let rss_after = get_rss_mb();
    let rss_delta_mb = rss_after - rss_before;

    println!(
        "   Model loaded in {}ms (RSS: +{:.0} MB)",
        model_load_ms, rss_delta_mb
    );

    // 4. Embed all chunks
    let start = Instant::now();
    let embeddings = embedder.embed_batch(prepared_texts.to_vec())?;
    let total_index_ms = start.elapsed().as_millis() as u64;
    let embed_throughput = if total_index_ms > 0 {
        chunks_count as f32 / (total_index_ms as f32 / 1000.0)
    } else {
        0.0
    };

    println!(
        "   Embedded {} chunks in {}ms ({:.0} ch/sec)",
        chunks_count, total_index_ms, embed_throughput
    );

    // 5. Run accuracy tests
    let mut correct = 0;
    let mut total_score = 0.0f32;
    let mut query_times = Vec::new();

    for (query, expected_file) in TEST_QUERIES {
        let start = Instant::now();
        let query_embedding = embedder.embed_one(query)?;
        query_times.push(start.elapsed());

        // Find best match via brute-force cosine similarity
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
        let is_correct = best_chunk.path.contains(expected_file);

        if is_correct {
            correct += 1;
        }
        total_score += best_score;

        let query_short = if query.len() > 35 {
            format!("{}...", &query[..32])
        } else {
            query.to_string()
        };
        let file_short = best_chunk
            .path
            .split('/')
            .last()
            .unwrap_or(&best_chunk.path);
        println!(
            "   {} \"{}\" -> {} ({:.3})",
            if is_correct { "✅" } else { "❌" },
            query_short,
            file_short,
            best_score
        );
    }

    // 6. False positive test
    let fp_embedding = embedder.embed_one(FALSE_POSITIVE_QUERY)?;
    let false_positive_score = embeddings
        .iter()
        .map(|emb| cosine_similarity(&fp_embedding, emb))
        .fold(0.0f32, f32::max);
    println!("   FP score: {:.3}", false_positive_score);

    let accuracy = correct as f32 / TEST_QUERIES.len() as f32;
    let avg_score = total_score / TEST_QUERIES.len() as f32;
    let avg_query_ms = if query_times.is_empty() {
        0.0
    } else {
        query_times.iter().sum::<Duration>().as_secs_f64() * 1000.0 / query_times.len() as f64
    };

    // 7. Estimated DB size (vectors only: dims * chunks * 4 bytes)
    let estimated_db_mb =
        (model_type.dimensions() * chunks_count * 4) as f64 / (1024.0 * 1024.0);

    // Drop embedder to free memory before next model
    drop(embedder);
    drop(embeddings);

    Ok(BenchResult {
        model_name: model_type.name().to_string(),
        short_name: model_type.short_name().to_string(),
        dimensions: model_type.dimensions(),
        quantized: model_type.is_quantized(),
        model_load_ms,
        embed_throughput,
        avg_query_ms,
        total_index_ms,
        accuracy,
        avg_score,
        false_positive_score,
        rss_delta_mb,
        estimated_db_mb,
        chunks_count,
    })
}

fn print_summary_table(results: &[BenchResult]) {
    println!();
    println!(
        "{}",
        "╔════════════════════════════════════════════════════════════════════════════════════════╗"
            .bright_cyan()
    );
    println!(
        "{}",
        "║                           BENCHMARK SUMMARY                                          ║"
            .bright_cyan()
    );
    println!(
        "{}",
        "╚════════════════════════════════════════════════════════════════════════════════════════╝"
            .bright_cyan()
    );
    println!();

    // Header
    println!(
        "{:<18} {:>5} {:>5} {:>8} {:>8} {:>8} {:>5} {:>6} {:>7}",
        "Model".bold(),
        "Dims".bold(),
        "Quant".bold(),
        "Load".bold(),
        "ch/sec".bold(),
        "qry(ms)".bold(),
        "Acc".bold(),
        "FP".bold(),
        "RSS".bold(),
    );
    println!("{}", "─".repeat(88));

    // Sort by accuracy desc, then throughput desc
    let mut sorted: Vec<&BenchResult> = results.iter().collect();
    sorted.sort_by(|a, b| {
        b.accuracy
            .partial_cmp(&a.accuracy)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.embed_throughput
                    .partial_cmp(&a.embed_throughput)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    for r in &sorted {
        let acc_str = format!("{:.0}%", r.accuracy * 100.0);
        let acc_colored = if r.accuracy >= 0.8 {
            acc_str.green()
        } else if r.accuracy >= 0.6 {
            acc_str.yellow()
        } else {
            acc_str.red()
        };

        let fp_str = format!("{:.2}", r.false_positive_score);
        let fp_colored = if r.false_positive_score < 0.5 {
            fp_str.green()
        } else if r.false_positive_score < 0.7 {
            fp_str.yellow()
        } else {
            fp_str.red()
        };

        println!(
            "{:<18} {:>5} {:>5} {:>6}ms {:>8.0} {:>8.1} {:>5} {:>6} {:>5.0}MB",
            r.short_name,
            r.dimensions,
            if r.quantized { "yes" } else { "no" },
            r.model_load_ms,
            r.embed_throughput,
            r.avg_query_ms,
            acc_colored,
            fp_colored,
            r.rss_delta_mb,
        );
    }

    println!("{}", "─".repeat(88));

    // Winners
    if let Some(best_acc) = sorted.iter().max_by(|a, b| {
        a.accuracy
            .partial_cmp(&b.accuracy)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!(
            "  {} Best accuracy:  {} ({:.0}%)",
            "🏆",
            best_acc.short_name.bright_green(),
            best_acc.accuracy * 100.0
        );
    }
    if let Some(fastest) = sorted.iter().max_by(|a, b| {
        a.embed_throughput
            .partial_cmp(&b.embed_throughput)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!(
            "  {} Fastest:        {} ({:.0} ch/sec)",
            "⚡",
            fastest.short_name.bright_cyan(),
            fastest.embed_throughput
        );
    }
    if let Some(smallest) = sorted
        .iter()
        .filter(|r| r.rss_delta_mb > 0.0)
        .min_by(|a, b| {
            a.rss_delta_mb
                .partial_cmp(&b.rss_delta_mb)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        println!(
            "  {} Smallest RSS:   {} (+{:.0} MB)",
            "💾",
            smallest.short_name.bright_yellow(),
            smallest.rss_delta_mb
        );
    }
    println!();
}

fn save_markdown_report(results: &[BenchResult], path: &std::path::Path) -> Result<()> {
    let mut md = String::new();

    md.push_str("# Demongrep Model Benchmark Results\n\n");
    md.push_str(&format!(
        "**Date**: {}  \n",
        chrono::Local::now().format("%Y-%m-%d %H:%M")
    ));
    if let Some(first) = results.first() {
        md.push_str(&format!("**Chunks**: {}  \n\n", first.chunks_count));
    }

    // Summary table
    md.push_str("## Summary\n\n");
    md.push_str("| Model | Dims | Quant | Load(ms) | ch/sec | qry(ms) | Accuracy | Avg Score | FP Score | RSS(MB) | Est DB(MB) |\n");
    md.push_str("|-------|------|-------|----------|--------|---------|----------|-----------|----------|---------|------------|\n");

    for r in results {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:.0} | {:.1} | {:.0}% | {:.3} | {:.3} | {:.0} | {:.1} |\n",
            r.short_name,
            r.dimensions,
            if r.quantized { "yes" } else { "no" },
            r.model_load_ms,
            r.embed_throughput,
            r.avg_query_ms,
            r.accuracy * 100.0,
            r.avg_score,
            r.false_positive_score,
            r.rss_delta_mb,
            r.estimated_db_mb,
        ));
    }

    // Per-model details
    md.push_str("\n## Per-Model Details\n\n");
    for r in results {
        md.push_str(&format!("### {}\n\n", r.model_name));
        md.push_str(&format!("- **Short name**: `{}`\n", r.short_name));
        md.push_str(&format!("- **Dimensions**: {}\n", r.dimensions));
        md.push_str(&format!("- **Quantized**: {}\n", r.quantized));
        md.push_str(&format!("- **Model load**: {} ms\n", r.model_load_ms));
        md.push_str(&format!(
            "- **Embedding throughput**: {:.0} chunks/sec\n",
            r.embed_throughput
        ));
        md.push_str(&format!("- **Avg query time**: {:.1} ms\n", r.avg_query_ms));
        md.push_str(&format!(
            "- **Accuracy**: {:.0}%\n",
            r.accuracy * 100.0
        ));
        md.push_str(&format!("- **Avg score**: {:.3}\n", r.avg_score));
        md.push_str(&format!(
            "- **False positive score**: {:.3}\n",
            r.false_positive_score
        ));
        md.push_str(&format!("- **RSS delta**: {:.0} MB\n", r.rss_delta_mb));
        md.push_str(&format!(
            "- **Estimated DB size**: {:.1} MB\n\n",
            r.estimated_db_mb
        ));
    }

    std::fs::write(path, md)?;
    println!("📄 Report saved to: {}", path.display());

    Ok(())
}

/// Run benchmark across selected (or all) models
pub async fn bench(
    models_filter: Option<String>,
    limit: Option<usize>,
    path: Option<PathBuf>,
    output: Option<PathBuf>,
    json_output: bool,
) -> Result<()> {
    // Parse model list
    let models: Vec<ModelType> = if let Some(filter) = &models_filter {
        let mut parsed = Vec::new();
        for name in filter.split(',') {
            let name = name.trim();
            match ModelType::from_str(name) {
                Some(m) => parsed.push(m),
                None => {
                    eprintln!("Unknown model: '{}', skipping", name);
                }
            }
        }
        if parsed.is_empty() {
            return Err(anyhow::anyhow!(
                "No valid models specified. Available: {}",
                ModelType::all()
                    .iter()
                    .map(|m| m.short_name())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        parsed
    } else {
        ModelType::all().to_vec()
    };

    if !json_output {
        println!(
            "{}",
            "╔════════════════════════════════════════════════════════════════╗".bright_cyan()
        );
        println!(
            "{}",
            "║              DEMONGREP MODEL BENCHMARK                        ║".bright_cyan()
        );
        println!(
            "{}",
            "╚════════════════════════════════════════════════════════════════╝".bright_cyan()
        );
        println!();
        println!("Models to benchmark: {}", models.len());
        for m in &models {
            println!(
                "   - {} ({} dims{})",
                m.short_name(),
                m.dimensions(),
                if m.is_quantized() { ", quantized" } else { "" }
            );
        }
        println!();
    }

    // Phase 1: Collect files and create chunks (shared across all models)
    let project_path = path.unwrap_or_else(|| PathBuf::from("."));
    if !json_output {
        println!("📂 Project: {}", project_path.display());
    }

    let walker = FileWalker::new(project_path.clone());
    let (mut files, _stats) = walker.walk()?;

    if let Some(max_files) = limit {
        files.truncate(max_files);
    }

    if !json_output {
        println!("🔍 Discovered {} files", files.len());
        println!("🔪 Chunking...");
    }

    let mut chunker = SemanticChunker::new(100, 2000, 10);
    let mut all_chunks = Vec::new();
    for file in &files {
        if let Ok(content) = std::fs::read_to_string(&file.path) {
            if let Ok(chunks) = chunker.chunk_semantic(file.language, &file.path, &content) {
                all_chunks.extend(chunks);
            }
        }
    }

    if !json_output {
        println!("   {} chunks from {} files", all_chunks.len(), files.len());
        println!();
    }

    if all_chunks.is_empty() {
        return Err(anyhow::anyhow!("No chunks created. Is this a code project?"));
    }

    // Prepare texts once (uses same logic as real indexing)
    let prepared_texts: Vec<String> = all_chunks
        .par_iter()
        .map(|chunk| BatchEmbedder::prepare_text(chunk))
        .collect();

    // Phase 2: Benchmark each model
    let mut results = Vec::new();

    for (i, model_type) in models.iter().enumerate() {
        if !json_output {
            println!(
                "{}",
                format!(
                    "━━━ [{}/{}] {} ━━━",
                    i + 1,
                    models.len(),
                    model_type.name()
                )
                .bright_cyan()
            );
        }

        match benchmark_model(*model_type, &all_chunks, &prepared_texts) {
            Ok(result) => results.push(result),
            Err(e) => {
                if !json_output {
                    eprintln!("   ❌ Failed: {}", e);
                }
            }
        }

        if !json_output {
            println!();
        }
    }

    if results.is_empty() {
        return Err(anyhow::anyhow!("All models failed"));
    }

    // Phase 3: Output results
    if json_output {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        print_summary_table(&results);
    }

    // Save markdown report if requested
    if let Some(output_path) = output {
        save_markdown_report(&results, &output_path)?;
    }

    Ok(())
}
