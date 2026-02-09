//! Benchmark framework for comparing embedding models.
//!
//! Measures performance (throughput, latency), quality (accuracy, false positives),
//! and memory (RSS delta, estimated DB size) across benchmark model profiles.

use anyhow::Result;
use colored::Colorize;
// NOTE: Rayon intentionally not used here - see comment below about ONNX thread pool conflict
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchProfile {
    Smoke,
    Standard,
    Full,
}

impl BenchProfile {
    pub fn from_str(profile: &str) -> Result<Self> {
        match profile.to_lowercase().as_str() {
            "smoke" => Ok(Self::Smoke),
            "standard" => Ok(Self::Standard),
            "full" => Ok(Self::Full),
            _ => Err(anyhow::anyhow!(
                "Invalid profile '{}'. Available: smoke, standard, full",
                profile
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Standard => "standard",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub model_name: String,
    pub short_name: String,
    pub dimensions: usize,
    pub quantized: bool,
    // Performance
    pub model_load_ms: u64,
    pub embed_total_ms: u64,
    pub query_eval_ms: u64,
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

fn models_for_profile(profile: BenchProfile) -> Vec<ModelType> {
    match profile {
        BenchProfile::Smoke => vec![ModelType::AllMiniLML6V2Q, ModelType::BGESmallENV15Q],
        BenchProfile::Standard => vec![
            ModelType::AllMiniLML6V2Q,
            ModelType::BGESmallENV15Q,
            ModelType::JinaEmbeddingsV2BaseCode,
        ],
        BenchProfile::Full => ModelType::all().to_vec(),
    }
}

fn select_models(models_filter: Option<&str>, profile: BenchProfile) -> Result<Vec<ModelType>> {
    if let Some(filter) = models_filter {
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

        Ok(parsed)
    } else {
        Ok(models_for_profile(profile))
    }
}

/// Benchmark a single model against pre-chunked data
fn benchmark_model(
    model_type: ModelType,
    chunks: &[Chunk],
    base_prepared_texts: &[String],
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

    if model_type.dimensions() >= 1024 && !model_type.is_quantized() {
        println!(
            "   ℹ️  Tip: {} is a heavy model ({} dims). Use `--profile standard` for faster runs.",
            model_type.short_name(),
            model_type.dimensions()
        );
    }

    let model_prepared_storage = if model_type.has_special_passage_format() {
        Some(
            base_prepared_texts
                .iter()
                .map(|text| model_type.format_passage(text))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let prepared_texts = model_prepared_storage
        .as_deref()
        .unwrap_or(base_prepared_texts);

    // 4. Embed all chunks
    let start = Instant::now();
    let embeddings = embedder.embed_batch_refs(prepared_texts)?;
    let embed_total_ms = start.elapsed().as_millis() as u64;
    let embed_throughput = if embed_total_ms > 0 {
        chunks_count as f32 / (embed_total_ms as f32 / 1000.0)
    } else {
        0.0
    };

    println!(
        "   Embedded {} chunks in {}ms ({:.0} ch/sec)",
        chunks_count, embed_total_ms, embed_throughput
    );

    // 5. Run accuracy tests
    let mut correct = 0;
    let mut total_score = 0.0f32;
    let mut query_times = Vec::new();

    for (query, expected_file) in TEST_QUERIES {
        let start = Instant::now();
        let formatted_query = model_type.format_query(query);
        let query_embedding = embedder.embed_one(&formatted_query)?;
        let query_elapsed = start.elapsed();
        query_times.push(query_elapsed);

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
    let fp_start = Instant::now();
    let fp_query = model_type.format_query(FALSE_POSITIVE_QUERY);
    let fp_embedding = embedder.embed_one(&fp_query)?;
    let fp_elapsed = fp_start.elapsed();
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
    let query_eval_ms = (query_times.iter().sum::<Duration>() + fp_elapsed).as_millis() as u64;
    let total_index_ms = embed_total_ms;

    println!(
        "   Query eval in {}ms ({:.1} ms/query avg)",
        query_eval_ms, avg_query_ms
    );

    // 7. Estimated DB size (vectors only: dims * chunks * 4 bytes)
    let estimated_db_mb = (model_type.dimensions() * chunks_count * 4) as f64 / (1024.0 * 1024.0);

    // Drop embedder to free memory before next model
    drop(embedder);
    drop(embeddings);

    Ok(BenchResult {
        model_name: model_type.name().to_string(),
        short_name: model_type.short_name().to_string(),
        dimensions: model_type.dimensions(),
        quantized: model_type.is_quantized(),
        model_load_ms,
        embed_total_ms,
        query_eval_ms,
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
        "{:<18} {:>5} {:>5} {:>8} {:>8} {:>8} {:>8} {:>5} {:>6} {:>7}",
        "Model".bold(),
        "Dims".bold(),
        "Quant".bold(),
        "Load".bold(),
        "Embed".bold(),
        "ch/sec".bold(),
        "qeval".bold(),
        "Acc".bold(),
        "FP".bold(),
        "RSS".bold(),
    );
    println!("{}", "─".repeat(103));

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
            "{:<18} {:>5} {:>5} {:>6}ms {:>6}ms {:>8.0} {:>6}ms {:>5} {:>6} {:>5.0}MB",
            r.short_name,
            r.dimensions,
            if r.quantized { "yes" } else { "no" },
            r.model_load_ms,
            r.embed_total_ms,
            r.embed_throughput,
            r.query_eval_ms,
            acc_colored,
            fp_colored,
            r.rss_delta_mb,
        );
    }

    println!("{}", "─".repeat(103));

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
    md.push_str("| Model | Dims | Quant | Load(ms) | Embed(ms) | QueryEval(ms) | ch/sec | qry(ms) | Accuracy | Avg Score | FP Score | RSS(MB) | Est DB(MB) |\n");
    md.push_str("|-------|------|-------|----------|-----------|---------------|--------|---------|----------|-----------|----------|---------|------------|\n");

    for r in results {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.0} | {:.1} | {:.0}% | {:.3} | {:.3} | {:.0} | {:.1} |\n",
            r.short_name,
            r.dimensions,
            if r.quantized { "yes" } else { "no" },
            r.model_load_ms,
            r.embed_total_ms,
            r.query_eval_ms,
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
        md.push_str(&format!("- **Embed total**: {} ms\n", r.embed_total_ms));
        md.push_str(&format!("- **Query eval total**: {} ms\n", r.query_eval_ms));
        md.push_str(&format!(
            "- **Embedding throughput**: {:.0} chunks/sec\n",
            r.embed_throughput
        ));
        md.push_str(&format!("- **Avg query time**: {:.1} ms\n", r.avg_query_ms));
        md.push_str(&format!("- **Accuracy**: {:.0}%\n", r.accuracy * 100.0));
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

/// Run benchmark across selected models or a predefined benchmark profile.
pub async fn bench(
    models_filter: Option<String>,
    profile: String,
    limit: Option<usize>,
    path: Option<PathBuf>,
    output: Option<PathBuf>,
    json_output: bool,
) -> Result<()> {
    let profile = BenchProfile::from_str(&profile)?;
    let models = select_models(models_filter.as_deref(), profile)?;
    let using_custom_model_list = models_filter.is_some();

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
        if using_custom_model_list {
            println!("Profile: custom (--models takes precedence over --profile)");
        } else {
            println!("Profile: {}", profile.as_str());
        }
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
        return Err(anyhow::anyhow!(
            "No chunks created. Is this a code project?"
        ));
    }

    // Prepare texts once (uses same logic as real indexing)
    // CRITICAL: Use iter() instead of par_iter() here. ONNX Runtime creates its own
    // thread pool based on available parallelism. If Rayon threads are active when
    // ONNX tries to create/use its threads, it can cause a deadlock due to thread
    // pool exhaustion. This is a known issue when mixing Rayon with ONNX Runtime.
    let base_prepared_texts: Vec<String> = all_chunks
        .iter()
        .map(|chunk| BatchEmbedder::prepare_text(chunk))
        .collect();

    // Phase 2: Benchmark each model
    let mut results = Vec::new();

    for (i, model_type) in models.iter().enumerate() {
        if !json_output {
            println!(
                "{}",
                format!("━━━ [{}/{}] {} ━━━", i + 1, models.len(), model_type.name()).bright_cyan()
            );
        }

        // Run benchmark directly - ONNX Runtime has its own thread management
        let model_type = *model_type;
        match benchmark_model(model_type, &all_chunks, &base_prepared_texts) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_model_selection() {
        let smoke = select_models(None, BenchProfile::Smoke).unwrap();
        assert_eq!(
            smoke,
            vec![ModelType::AllMiniLML6V2Q, ModelType::BGESmallENV15Q]
        );

        let standard = select_models(None, BenchProfile::Standard).unwrap();
        assert!(standard.contains(&ModelType::AllMiniLML6V2Q));
        assert!(standard.contains(&ModelType::JinaEmbeddingsV2BaseCode));
        assert!(standard.contains(&ModelType::BGESmallENV15Q));
        assert_eq!(standard.len(), 3);

        let full = select_models(None, BenchProfile::Full).unwrap();
        assert_eq!(full.len(), ModelType::all().len());
    }

    #[test]
    fn test_models_filter_precedence() {
        let selected = select_models(Some("minilm-l6-q,jina-code"), BenchProfile::Full).unwrap();
        assert_eq!(
            selected,
            vec![
                ModelType::AllMiniLML6V2Q,
                ModelType::JinaEmbeddingsV2BaseCode
            ]
        );
    }

    #[test]
    fn test_invalid_profile() {
        let err = BenchProfile::from_str("fastest").unwrap_err().to_string();
        assert!(err.contains("Invalid profile"));
    }
}
