use crate::info_print;
use anyhow::{anyhow, Context, Result};
use fastembed::{
    EmbeddingModel as FastEmbedModel, InitOptions, InitOptionsUserDefined, Pooling,
    QuantizationMode, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use std::io::Read;
use std::path::PathBuf;

/// Available embedding models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Quantized All-MiniLM-L6-v2 - 384 dimensions, faster
    AllMiniLML6V2Q,
    /// Quantized BGE Small EN v1.5 - 384 dimensions, faster
    BGESmallENV15Q,
    /// Jina Embeddings v2 Base Code - 768 dimensions, optimized for code
    JinaEmbeddingsV2BaseCode,
    /// Mixedbread Embed Large v1 - 1024 dimensions, high quality general retrieval
    MxbaiEmbedLargeV1,
    /// Mixedbread Embed XSmall v1 - 384 dimensions, lightweight mixedbread model
    MxbaiEmbedXSmallV1,
}

impl ModelType {
    pub fn to_fastembed_model(&self) -> Option<FastEmbedModel> {
        match self {
            Self::AllMiniLML6V2Q => Some(FastEmbedModel::AllMiniLML6V2Q),
            Self::BGESmallENV15Q => Some(FastEmbedModel::BGESmallENV15Q),
            Self::JinaEmbeddingsV2BaseCode => Some(FastEmbedModel::JinaEmbeddingsV2BaseCode),
            Self::MxbaiEmbedLargeV1 => Some(FastEmbedModel::MxbaiEmbedLargeV1),
            // Not included in fastembed's built-in enum in v5.8.1; loaded via user-defined path.
            Self::MxbaiEmbedXSmallV1 => None,
        }
    }

    pub fn dimensions(&self) -> usize {
        match self {
            // 384 dimensions
            Self::AllMiniLML6V2Q | Self::BGESmallENV15Q | Self::MxbaiEmbedXSmallV1 => 384,
            // 768 dimensions
            Self::JinaEmbeddingsV2BaseCode => 768,
            // 1024 dimensions
            Self::MxbaiEmbedLargeV1 => 1024,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::AllMiniLML6V2Q => "sentence-transformers/all-MiniLM-L6-v2 (quantized)",
            Self::BGESmallENV15Q => "BAAI/bge-small-en-v1.5 (quantized)",
            Self::JinaEmbeddingsV2BaseCode => "jinaai/jina-embeddings-v2-base-code",
            Self::MxbaiEmbedLargeV1 => "mixedbread-ai/mxbai-embed-large-v1",
            Self::MxbaiEmbedXSmallV1 => "mixedbread-ai/mxbai-embed-xsmall-v1",
        }
    }

    /// Check if model is quantized (faster but slightly less accurate)
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::AllMiniLML6V2Q | Self::BGESmallENV15Q)
    }

    /// Format user query according to model-specific recommendations.
    pub fn format_query(&self, query: &str) -> String {
        match self {
            // BGE family benefits from an instruction-style query prefix.
            Self::BGESmallENV15Q => {
                format!(
                    "Represent this sentence for searching relevant code: {}",
                    query
                )
            }
            // Mixedbread recommends this retrieval query prefix.
            Self::MxbaiEmbedLargeV1 | Self::MxbaiEmbedXSmallV1 => {
                format!(
                    "Represent this sentence for searching relevant passages: {}",
                    query
                )
            }
            _ => query.to_string(),
        }
    }

    /// Format indexed passages according to model-specific recommendations.
    pub fn format_passage(&self, passage: &str) -> String {
        passage.to_string()
    }

    /// Whether this model applies special passage formatting.
    pub fn has_special_passage_format(&self) -> bool {
        false
    }

    /// Get a short identifier for the model (for filenames, etc.)
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::AllMiniLML6V2Q => "minilm-l6-q",
            Self::BGESmallENV15Q => "bge-small-q",
            Self::JinaEmbeddingsV2BaseCode => "jina-code",
            Self::MxbaiEmbedLargeV1 => "mxbai-large",
            Self::MxbaiEmbedXSmallV1 => "mxbai-xsmall",
        }
    }

    /// List all available models
    pub fn all() -> &'static [ModelType] {
        &[
            Self::AllMiniLML6V2Q,
            Self::BGESmallENV15Q,
            Self::JinaEmbeddingsV2BaseCode,
            Self::MxbaiEmbedLargeV1,
            Self::MxbaiEmbedXSmallV1,
        ]
    }

    /// Parse model from string (for CLI)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "minilm-l6" | "allminiml6v2" => Some(Self::AllMiniLML6V2Q),
            "minilm-l6-q" | "allminiml6v2q" => Some(Self::AllMiniLML6V2Q),
            "minilm-l12" | "allminiml12v2" => Some(Self::BGESmallENV15Q),
            "minilm-l12-q" | "allminiml12v2q" => Some(Self::BGESmallENV15Q),
            "paraphrase-minilm" => Some(Self::BGESmallENV15Q),
            "bge-small" | "bgesmallenv15" => Some(Self::BGESmallENV15Q),
            "bge-small-q" | "bgesmallenv15q" => Some(Self::BGESmallENV15Q),
            "bge-base" | "bgebaseenv15" => Some(Self::JinaEmbeddingsV2BaseCode),
            "bge-large" | "bgelargeenv15" => Some(Self::JinaEmbeddingsV2BaseCode),
            "nomic-v1" | "nomicembedtextv1" => Some(Self::JinaEmbeddingsV2BaseCode),
            "nomic-v1.5" | "nomicembedtextv15" => Some(Self::JinaEmbeddingsV2BaseCode),
            "nomic-v1.5-q" | "nomicembedtextv15q" => Some(Self::JinaEmbeddingsV2BaseCode),
            "jina-code" | "jinaembeddingsv2basecode" => Some(Self::JinaEmbeddingsV2BaseCode),
            "e5-multilingual" | "multilinguale5small" => Some(Self::BGESmallENV15Q),
            "mxbai-large" | "mxbaiembedlargev1" => Some(Self::MxbaiEmbedLargeV1),
            "mxbai-xsmall"
            | "mxbaiembedxsmallv1"
            | "mixedbread-ai/mxbai-embed-xsmall-v1"
            | "mxbai-embed-xsmall-v1" => Some(Self::MxbaiEmbedXSmallV1),
            "modernbert-large" | "modernbertembedlarge" => Some(Self::JinaEmbeddingsV2BaseCode),
            _ => None,
        }
    }
}

impl Default for ModelType {
    fn default() -> Self {
        // Default to the code-specialized Jina model for best code understanding.
        Self::JinaEmbeddingsV2BaseCode
    }
}

/// Fast embedding model using fastembed library
pub struct FastEmbedder {
    model: TextEmbedding,
    model_type: ModelType,
}

impl FastEmbedder {
    /// Create a new embedder with default model
    pub fn new() -> Result<Self> {
        Self::with_model(ModelType::default())
    }

    /// Create a new embedder with specified model
    pub fn with_model(model_type: ModelType) -> Result<Self> {
        info_print!("📦 Loading embedding model: {}", model_type.name());
        info_print!("   Dimensions: {}", model_type.dimensions());

        let model = match model_type.to_fastembed_model() {
            Some(fast_model) => TextEmbedding::try_new(
                InitOptions::new(fast_model).with_show_download_progress(true),
            )
            .map_err(|e| anyhow!("Failed to initialize embedding model: {}", e))?,
            None => Self::load_mxbai_xsmall_user_defined()?,
        };

        info_print!("✅ Model loaded successfully!");

        Ok(Self { model, model_type })
    }

    fn load_mxbai_xsmall_user_defined() -> Result<TextEmbedding> {
        const MODEL_REPO: &str = "mixedbread-ai/mxbai-embed-xsmall-v1";
        const ONNX_FILE: &str = "onnx/model.onnx";
        const TOKENIZER_JSON: &str = "tokenizer.json";
        const CONFIG_JSON: &str = "config.json";
        const SPECIAL_TOKENS_MAP_JSON: &str = "special_tokens_map.json";
        const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";

        let endpoint =
            std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
        let endpoint = endpoint.trim_end_matches('/');

        let cache_root = std::env::var("FASTEMBED_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(fastembed::get_cache_dir()));
        let model_cache = cache_root
            .join("user-defined")
            .join("mixedbread-ai")
            .join("mxbai-embed-xsmall-v1");

        let read_repo_file = |name: &str| -> Result<Vec<u8>> {
            let local_path = model_cache.join(name);
            if local_path.exists() {
                return std::fs::read(&local_path).with_context(|| {
                    format!("Failed to read cached file {}", local_path.display())
                });
            }

            if let Some(parent) = local_path.parent() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("Failed to create cache directory {}", parent.display())
                })?;
            }

            let url = format!("{}/{}/resolve/main/{}", endpoint, MODEL_REPO, name);
            let mut request = ureq::get(&url);
            if let Ok(token) = std::env::var("HF_TOKEN") {
                if !token.trim().is_empty() {
                    request = request.set("Authorization", &format!("Bearer {}", token));
                }
            }

            let response = request
                .call()
                .map_err(|e| anyhow!("Failed HTTP request for {}: {}", url, e))?;
            let mut reader = response.into_reader();
            let mut bytes = Vec::new();
            reader
                .read_to_end(&mut bytes)
                .with_context(|| format!("Failed to read response body for {}", url))?;

            std::fs::write(&local_path, &bytes)
                .with_context(|| format!("Failed to write cached file {}", local_path.display()))?;
            Ok(bytes)
        };

        let tokenizer_files = TokenizerFiles {
            tokenizer_file: read_repo_file(TOKENIZER_JSON)?,
            config_file: read_repo_file(CONFIG_JSON)?,
            special_tokens_map_file: read_repo_file(SPECIAL_TOKENS_MAP_JSON)?,
            tokenizer_config_file: read_repo_file(TOKENIZER_CONFIG_JSON)?,
        };
        let onnx_file = read_repo_file(ONNX_FILE)?;

        let model = UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files)
            .with_pooling(Pooling::Mean)
            .with_quantization(QuantizationMode::None);

        TextEmbedding::try_new_from_user_defined(model, InitOptionsUserDefined::new()).map_err(
            |e| {
                anyhow!(
                    "Failed to initialize custom model {} with fastembed user-defined loader: {}",
                    MODEL_REPO,
                    e
                )
            },
        )
    }

    fn resolve_batch_size(&self) -> usize {
        // Check for env var override (tune with DEMONGREP_BATCH_SIZE=N)
        if let Ok(env_size) = std::env::var("DEMONGREP_BATCH_SIZE") {
            env_size.parse().unwrap_or(256)
        } else {
            // Adaptive batch size: smaller batches for larger models to avoid OOM
            // Benchmarked on 12-core/24-thread CPU - batch size has minimal impact
            // when CPU is saturated, but larger batches slightly more efficient
            match self.model_type.dimensions() {
                d if d <= 384 => 256, // Small models: larger batches OK
                d if d <= 768 => 128, // Medium models
                _ => 64,              // Large models: smaller to avoid OOM
            }
        }
    }

    /// Embed a batch of texts (processes in mini-batches to avoid OOM)
    /// Uses adaptive batch size based on model dimensions
    /// Can be overridden with DEMONGREP_BATCH_SIZE environment variable
    pub fn embed_batch(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_refs(&texts)
    }

    /// Embed a batch from borrowed text references (avoids cloning at call sites).
    pub fn embed_batch_refs(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let batch_size = self.resolve_batch_size();
        self.embed_batch_refs_chunked(texts, batch_size)
    }

    /// Embed a batch of texts with configurable mini-batch size
    pub fn embed_batch_chunked(
        &mut self,
        texts: Vec<String>,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_refs_chunked(&texts, batch_size)
    }

    /// Embed a borrowed batch with configurable mini-batch size.
    pub fn embed_batch_refs_chunked(
        &mut self,
        texts: &[String],
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in mini-batches to avoid OOM with large models
        for chunk in texts.chunks(batch_size) {
            let text_refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();

            let embeddings = self
                .model
                .embed(text_refs, None)
                .map_err(|e| anyhow!("Failed to generate embeddings: {}", e))?;

            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    /// Embed a single text
    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(vec![text.to_string()])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding generated"))
    }

    /// Get the dimensionality of embeddings
    pub fn dimensions(&self) -> usize {
        self.model_type.dimensions()
    }

    /// Get the model name
    pub fn model_name(&self) -> &str {
        self.model_type.name()
    }

    /// Get the model type
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }
}

// NOTE: Default impl removed - FastEmbedder::new() returns Result and must not
// panic on model load failure. Use FastEmbedder::new() or ::with_model() instead.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_dimensions() {
        // 384 dimension models
        assert_eq!(ModelType::BGESmallENV15Q.dimensions(), 384);
        assert_eq!(ModelType::AllMiniLML6V2Q.dimensions(), 384);
        assert_eq!(ModelType::MxbaiEmbedXSmallV1.dimensions(), 384);
        // 768 dimension models
        assert_eq!(ModelType::JinaEmbeddingsV2BaseCode.dimensions(), 768);
        // 1024 dimension models
        assert_eq!(ModelType::MxbaiEmbedLargeV1.dimensions(), 1024);
    }

    #[test]
    fn test_model_type_names() {
        assert_eq!(
            ModelType::BGESmallENV15Q.name(),
            "BAAI/bge-small-en-v1.5 (quantized)"
        );
        assert_eq!(
            ModelType::AllMiniLML6V2Q.name(),
            "sentence-transformers/all-MiniLM-L6-v2 (quantized)"
        );
    }

    #[test]
    fn test_default_model() {
        let model = ModelType::default();
        assert_eq!(model, ModelType::JinaEmbeddingsV2BaseCode);
        assert_eq!(model.dimensions(), 768);
    }

    #[test]
    fn test_all_models() {
        let all = ModelType::all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_from_str() {
        assert_eq!(
            ModelType::from_str("bge-small"),
            Some(ModelType::BGESmallENV15Q)
        );
        assert_eq!(
            ModelType::from_str("jina-code"),
            Some(ModelType::JinaEmbeddingsV2BaseCode)
        );
        assert_eq!(
            ModelType::from_str("minilm-l6-q"),
            Some(ModelType::AllMiniLML6V2Q)
        );
        assert_eq!(
            ModelType::from_str("e5-multilingual"),
            Some(ModelType::BGESmallENV15Q)
        );
        assert_eq!(
            ModelType::from_str("mxbai-large"),
            Some(ModelType::MxbaiEmbedLargeV1)
        );
        assert_eq!(
            ModelType::from_str("mxbai-xsmall"),
            Some(ModelType::MxbaiEmbedXSmallV1)
        );
        assert_eq!(ModelType::from_str("unknown"), None);
    }

    #[test]
    fn test_is_quantized() {
        assert!(ModelType::AllMiniLML6V2Q.is_quantized());
        assert!(ModelType::BGESmallENV15Q.is_quantized());
        assert!(!ModelType::JinaEmbeddingsV2BaseCode.is_quantized());
        assert!(!ModelType::MxbaiEmbedLargeV1.is_quantized());
        assert!(!ModelType::MxbaiEmbedXSmallV1.is_quantized());
    }

    #[test]
    fn test_model_specific_formatting() {
        let query = "find auth";
        let passage = "Code:\nfn auth() {}";

        let bge_query = ModelType::BGESmallENV15Q.format_query(query);
        assert!(bge_query.starts_with("Represent this sentence for searching relevant code: "));
        assert_eq!(ModelType::BGESmallENV15Q.format_passage(passage), passage);

        let mxbai_query = ModelType::MxbaiEmbedLargeV1.format_query(query);
        assert!(
            mxbai_query.starts_with("Represent this sentence for searching relevant passages: ")
        );
        assert_eq!(
            ModelType::MxbaiEmbedLargeV1.format_passage(passage),
            passage
        );
        let mxbai_xs_query = ModelType::MxbaiEmbedXSmallV1.format_query(query);
        assert!(
            mxbai_xs_query.starts_with("Represent this sentence for searching relevant passages: ")
        );
        assert_eq!(
            ModelType::MxbaiEmbedXSmallV1.format_passage(passage),
            passage
        );

        assert_eq!(ModelType::AllMiniLML6V2Q.format_query(query), query);
        assert_eq!(ModelType::AllMiniLML6V2Q.format_passage(passage), passage);
    }

    #[test]
    #[ignore] // Requires downloading model
    fn test_embedder_creation() {
        let embedder = FastEmbedder::new();
        assert!(embedder.is_ok());

        let embedder = embedder.unwrap();
        assert_eq!(embedder.dimensions(), 768);
    }

    #[test]
    #[ignore] // Requires model
    fn test_embed_single_text() {
        let mut embedder = FastEmbedder::new().unwrap();
        let embedding = embedder.embed_one("Hello, world!").unwrap();

        assert_eq!(embedding.len(), 768);
        // Check embedding is normalized (roughly unit length)
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.1);
    }

    #[test]
    #[ignore] // Requires model
    fn test_embed_batch() {
        let mut embedder = FastEmbedder::new().unwrap();
        let texts = vec![
            "Hello, world!".to_string(),
            "Rust is awesome".to_string(),
            "Code search with AI".to_string(),
        ];

        let embeddings = embedder.embed_batch(texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for embedding in embeddings {
            assert_eq!(embedding.len(), 768);
        }
    }

    #[test]
    #[ignore] // Requires model
    fn test_semantic_similarity() {
        let mut embedder = FastEmbedder::new().unwrap();

        let text1 = "The quick brown fox jumps over the lazy dog";
        let text2 = "A fast auburn fox leaps over a sleepy canine";
        let text3 = "Python is a programming language";

        let emb1 = embedder.embed_one(text1).unwrap();
        let emb2 = embedder.embed_one(text2).unwrap();
        let emb3 = embedder.embed_one(text3).unwrap();

        // Cosine similarity
        let sim_1_2 = cosine_similarity(&emb1, &emb2);
        let sim_1_3 = cosine_similarity(&emb1, &emb3);

        // Similar texts should have higher similarity
        assert!(sim_1_2 > sim_1_3);
        assert!(sim_1_2 > 0.7); // Should be quite similar
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (mag_a * mag_b)
    }
}
