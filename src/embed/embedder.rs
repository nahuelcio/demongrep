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
    /// Jina Embeddings v5 Text Nano - 768 dimensions, requires public ONNX export
    JinaEmbeddingsV5TextNano,
    /// Jina Code Embeddings 1.5B - 1536 dimensions, loaded from a public ONNX export
    JinaCodeEmbeddings15B,
    /// Mixedbread Embed XSmall v1 - 384 dimensions, lightweight mixedbread model
    MxbaiEmbedXSmallV1,
}

impl ModelType {
    pub fn to_fastembed_model(&self) -> Option<FastEmbedModel> {
        match self {
            Self::AllMiniLML6V2Q => Some(FastEmbedModel::AllMiniLML6V2Q),
            // Not included in fastembed's built-in enum; loaded via user-defined path.
            Self::MxbaiEmbedXSmallV1
            | Self::JinaEmbeddingsV5TextNano
            | Self::JinaCodeEmbeddings15B => None,
        }
    }

    pub fn dimensions(&self) -> usize {
        match self {
            // 384 dimensions
            Self::AllMiniLML6V2Q | Self::MxbaiEmbedXSmallV1 => 384,
            // 768 dimensions
            Self::JinaEmbeddingsV5TextNano => 768,
            // 1536 dimensions
            Self::JinaCodeEmbeddings15B => 1536,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::AllMiniLML6V2Q => "sentence-transformers/all-MiniLM-L6-v2 (quantized)",
            Self::JinaEmbeddingsV5TextNano => "jinaai/jina-embeddings-v5-text-nano",
            Self::JinaCodeEmbeddings15B => "jinaai/jina-code-embeddings-1.5b",
            Self::MxbaiEmbedXSmallV1 => "mixedbread-ai/mxbai-embed-xsmall-v1",
        }
    }

    /// Check if model is quantized (faster but slightly less accurate)
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::AllMiniLML6V2Q)
    }

    /// Format user query according to model-specific recommendations.
    pub fn format_query(&self, query: &str) -> String {
        match self {
            Self::JinaEmbeddingsV5TextNano => format!("Query: {}", query),
            // Mixedbread recommends this retrieval query prefix.
            Self::MxbaiEmbedXSmallV1 => {
                format!(
                    "Represent this sentence for searching relevant passages: {}",
                    query
                )
            }
            // Jina Code 1.5B exposes task-specific prompts for natural-language-to-code retrieval.
            Self::JinaCodeEmbeddings15B => {
                format!(
                    "Find the most relevant code snippet given the following query:\n{}",
                    query
                )
            }
            _ => query.to_string(),
        }
    }

    /// Format indexed passages according to model-specific recommendations.
    pub fn format_passage(&self, passage: &str) -> String {
        match self {
            Self::JinaEmbeddingsV5TextNano => format!("Document: {}", passage),
            Self::JinaCodeEmbeddings15B => {
                format!("Candidate code snippet:\n{}", passage)
            }
            _ => passage.to_string(),
        }
    }

    /// Whether this model applies special passage formatting.
    pub fn has_special_passage_format(&self) -> bool {
        matches!(
            self,
            Self::JinaEmbeddingsV5TextNano | Self::JinaCodeEmbeddings15B
        )
    }

    /// Get a short identifier for the model (for filenames, etc.)
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::AllMiniLML6V2Q => "minilm-l6-q",
            Self::JinaEmbeddingsV5TextNano => "jina-v5-nano",
            Self::JinaCodeEmbeddings15B => "jina-code-1.5b",
            Self::MxbaiEmbedXSmallV1 => "mxbai-xsmall",
        }
    }

    /// List all available models
    pub fn all() -> &'static [ModelType] {
        &[
            Self::AllMiniLML6V2Q,
            Self::JinaEmbeddingsV5TextNano,
            Self::JinaCodeEmbeddings15B,
            Self::MxbaiEmbedXSmallV1,
        ]
    }

    /// Parse model from string (for CLI)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "minilm-l6" | "allminiml6v2" => Some(Self::AllMiniLML6V2Q),
            "minilm-l6-q" | "allminiml6v2q" => Some(Self::AllMiniLML6V2Q),
            "jina-v5-nano"
            | "jinav5nano"
            | "jina-embeddings-v5-text-nano"
            | "jinaai/jina-embeddings-v5-text-nano" => Some(Self::JinaEmbeddingsV5TextNano),
            "jina-code-1.5b"
            | "jina-code-embeddings-1.5b"
            | "jinacodeembeddings15b"
            | "jinaai/jina-code-embeddings-1.5b"
            | "hermaster/jina-code-embeddings-1.5b-onnx" => Some(Self::JinaCodeEmbeddings15B),
            "mxbai-xsmall"
            | "mxbaiembedxsmallv1"
            | "mixedbread-ai/mxbai-embed-xsmall-v1"
            | "mxbai-embed-xsmall-v1" => Some(Self::MxbaiEmbedXSmallV1),
            _ => None,
        }
    }
}

impl Default for ModelType {
    fn default() -> Self {
        // Default to MiniLM-L6-Q: built-in fastembed model, lightweight, and stable.
        Self::AllMiniLML6V2Q
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
            None => match model_type {
                ModelType::MxbaiEmbedXSmallV1 => Self::load_mxbai_xsmall_user_defined()?,
                ModelType::JinaEmbeddingsV5TextNano => Self::load_jina_v5_text_nano_user_defined()
                    .map_err(|e| anyhow!("Failed to initialize '{}': {}", model_type.name(), e))?,
                ModelType::JinaCodeEmbeddings15B => {
                    Self::load_jina_code_embeddings_15b_user_defined().map_err(|e| {
                        anyhow!("Failed to initialize '{}': {}", model_type.name(), e)
                    })?
                }
                _ => {
                    return Err(anyhow!(
                        "Model {} requires a user-defined loader that is not implemented",
                        model_type.name()
                    ))
                }
            },
        };

        info_print!("✅ Model loaded successfully!");

        Ok(Self { model, model_type })
    }

    fn huggingface_endpoint() -> String {
        std::env::var("HF_ENDPOINT")
            .unwrap_or_else(|_| "https://huggingface.co".to_string())
            .trim_end_matches('/')
            .to_string()
    }

    fn user_defined_cache_dir(path_segments: &[&str]) -> PathBuf {
        let cache_root = std::env::var("FASTEMBED_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(fastembed::get_cache_dir()));

        path_segments
            .iter()
            .fold(cache_root.join("user-defined"), |path, segment| {
                path.join(segment)
            })
    }

    fn read_hf_repo_file(
        model_cache: &std::path::Path,
        model_repo: &str,
        endpoint: &str,
        name: &str,
    ) -> Result<Vec<u8>> {
        let local_path = model_cache.join(name);
        if local_path.exists() {
            return std::fs::read(&local_path)
                .with_context(|| format!("Failed to read cached file {}", local_path.display()));
        }

        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create cache directory {}", parent.display())
            })?;
        }

        let url = format!("{}/{}/resolve/main/{}", endpoint, model_repo, name);
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
    }

    fn synthesize_special_tokens_map(tokenizer_config_file: &[u8]) -> Result<Vec<u8>> {
        let tokenizer_config: serde_json::Value = serde_json::from_slice(tokenizer_config_file)
            .context("Failed to parse tokenizer_config.json for synthetic special tokens map")?;

        let mut special_tokens = serde_json::Map::new();
        for key in [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "cls_token",
            "pad_token",
            "mask_token",
        ] {
            if let Some(value) = tokenizer_config.get(key) {
                if value.is_string() || value.is_object() {
                    special_tokens.insert(key.to_string(), value.clone());
                }
            }
        }

        serde_json::to_vec(&serde_json::Value::Object(special_tokens))
            .context("Failed to serialize synthetic special_tokens_map.json")
    }

    fn load_mxbai_xsmall_user_defined() -> Result<TextEmbedding> {
        const MODEL_REPO: &str = "mixedbread-ai/mxbai-embed-xsmall-v1";
        const ONNX_FILE: &str = "onnx/model.onnx";
        const TOKENIZER_JSON: &str = "tokenizer.json";
        const CONFIG_JSON: &str = "config.json";
        const SPECIAL_TOKENS_MAP_JSON: &str = "special_tokens_map.json";
        const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";

        let endpoint = Self::huggingface_endpoint();
        let model_cache = Self::user_defined_cache_dir(&["mixedbread-ai", "mxbai-embed-xsmall-v1"]);

        let tokenizer_files = TokenizerFiles {
            tokenizer_file: Self::read_hf_repo_file(
                &model_cache,
                MODEL_REPO,
                &endpoint,
                TOKENIZER_JSON,
            )?,
            config_file: Self::read_hf_repo_file(&model_cache, MODEL_REPO, &endpoint, CONFIG_JSON)?,
            special_tokens_map_file: Self::read_hf_repo_file(
                &model_cache,
                MODEL_REPO,
                &endpoint,
                SPECIAL_TOKENS_MAP_JSON,
            )?,
            tokenizer_config_file: Self::read_hf_repo_file(
                &model_cache,
                MODEL_REPO,
                &endpoint,
                TOKENIZER_CONFIG_JSON,
            )?,
        };
        let onnx_file = Self::read_hf_repo_file(&model_cache, MODEL_REPO, &endpoint, ONNX_FILE)?;

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

    fn load_jina_v5_text_nano_user_defined() -> Result<TextEmbedding> {
        const MODEL_REPO: &str = "jinaai/jina-embeddings-v5-text-nano-retrieval";
        const TOKENIZER_JSON: &str = "tokenizer.json";
        const CONFIG_JSON: &str = "config.json";
        const SPECIAL_TOKENS_MAP_JSON: &str = "special_tokens_map.json";
        const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";
        const ONNX_CANDIDATES: &[(&str, &str, &str)] = &[
            ("onnx/model.onnx", "onnx/model.onnx_data", "model.onnx_data"),
            (
                "onnx/model_fp16.onnx",
                "onnx/model_fp16.onnx_data",
                "model_fp16.onnx_data",
            ),
            (
                "onnx/model_quantized.onnx",
                "onnx/model_quantized.onnx_data",
                "model_quantized.onnx_data",
            ),
        ];

        let endpoint = Self::huggingface_endpoint();
        let model_cache =
            Self::user_defined_cache_dir(&["jinaai", "jina-embeddings-v5-text-nano-retrieval"]);

        let tokenizer_config_file =
            Self::read_hf_repo_file(&model_cache, MODEL_REPO, &endpoint, TOKENIZER_CONFIG_JSON)?;
        let special_tokens_map_file = match Self::read_hf_repo_file(
            &model_cache,
            MODEL_REPO,
            &endpoint,
            SPECIAL_TOKENS_MAP_JSON,
        ) {
            Ok(bytes) => bytes,
            Err(_) => Self::synthesize_special_tokens_map(&tokenizer_config_file)?,
        };

        let tokenizer_files = TokenizerFiles {
            tokenizer_file: Self::read_hf_repo_file(
                &model_cache,
                MODEL_REPO,
                &endpoint,
                TOKENIZER_JSON,
            )?,
            config_file: Self::read_hf_repo_file(&model_cache, MODEL_REPO, &endpoint, CONFIG_JSON)?,
            special_tokens_map_file,
            tokenizer_config_file,
        };

        let mut onnx_file: Option<(Vec<u8>, String, Vec<u8>)> = None;
        let mut onnx_errors = Vec::new();
        for (onnx_candidate, data_candidate, initializer_name) in ONNX_CANDIDATES {
            match (
                Self::read_hf_repo_file(&model_cache, MODEL_REPO, &endpoint, onnx_candidate),
                Self::read_hf_repo_file(&model_cache, MODEL_REPO, &endpoint, data_candidate),
            ) {
                (Ok(onnx_bytes), Ok(data_bytes)) => {
                    onnx_file = Some((onnx_bytes, (*initializer_name).to_string(), data_bytes));
                    break;
                }
                (Err(e), _) => onnx_errors.push(format!("{} -> {}", onnx_candidate, e)),
                (_, Err(e)) => onnx_errors.push(format!("{} -> {}", data_candidate, e)),
            }
        }

        let (onnx_file, initializer_name, onnx_data_file) = onnx_file.ok_or_else(|| {
            anyhow!(
                "No usable ONNX export found for {} via {}. Tried: {}. Use 'jina-code-1.5b' instead.",
                ModelType::JinaEmbeddingsV5TextNano.name(),
                MODEL_REPO,
                onnx_errors.join(" | ")
            )
        })?;

        let model = UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files)
            .with_external_initializer(initializer_name, onnx_data_file)
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

    fn load_jina_code_embeddings_15b_user_defined() -> Result<TextEmbedding> {
        const MODEL_REPO: &str = "herMaster/jina-code-embeddings-1.5b-ONNX";
        const TOKENIZER_JSON: &str = "tokenizer.json";
        const CONFIG_JSON: &str = "config.json";
        const SPECIAL_TOKENS_MAP_JSON: &str = "special_tokens_map.json";
        const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";
        const ONNX_FILE: &str = "model.onnx";
        const EXTERNAL_DATA_FILE: &str = "model.onnx_data";

        let endpoint = Self::huggingface_endpoint();
        let model_cache =
            Self::user_defined_cache_dir(&["herMaster", "jina-code-embeddings-1.5b-ONNX"]);

        let tokenizer_files = TokenizerFiles {
            tokenizer_file: Self::read_hf_repo_file(
                &model_cache,
                MODEL_REPO,
                &endpoint,
                TOKENIZER_JSON,
            )?,
            config_file: Self::read_hf_repo_file(&model_cache, MODEL_REPO, &endpoint, CONFIG_JSON)?,
            special_tokens_map_file: Self::read_hf_repo_file(
                &model_cache,
                MODEL_REPO,
                &endpoint,
                SPECIAL_TOKENS_MAP_JSON,
            )?,
            tokenizer_config_file: Self::read_hf_repo_file(
                &model_cache,
                MODEL_REPO,
                &endpoint,
                TOKENIZER_CONFIG_JSON,
            )?,
        };

        let onnx_file = Self::read_hf_repo_file(&model_cache, MODEL_REPO, &endpoint, ONNX_FILE)?;
        let external_data =
            Self::read_hf_repo_file(&model_cache, MODEL_REPO, &endpoint, EXTERNAL_DATA_FILE)?;

        let model = UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files)
            .with_external_initializer(EXTERNAL_DATA_FILE.to_string(), external_data)
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
        assert_eq!(ModelType::AllMiniLML6V2Q.dimensions(), 384);
        assert_eq!(ModelType::MxbaiEmbedXSmallV1.dimensions(), 384);
        // 768 dimension models
        assert_eq!(ModelType::JinaEmbeddingsV5TextNano.dimensions(), 768);
        // 1536 dimension models
        assert_eq!(ModelType::JinaCodeEmbeddings15B.dimensions(), 1536);
    }

    #[test]
    fn test_model_type_names() {
        assert_eq!(
            ModelType::AllMiniLML6V2Q.name(),
            "sentence-transformers/all-MiniLM-L6-v2 (quantized)"
        );
        assert_eq!(
            ModelType::MxbaiEmbedXSmallV1.name(),
            "mixedbread-ai/mxbai-embed-xsmall-v1"
        );
    }

    #[test]
    fn test_default_model() {
        let model = ModelType::default();
        assert_eq!(model, ModelType::AllMiniLML6V2Q);
        assert_eq!(model.dimensions(), 384);
    }

    #[test]
    fn test_all_models() {
        let all = ModelType::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_from_str() {
        assert_eq!(
            ModelType::from_str("jina-v5-nano"),
            Some(ModelType::JinaEmbeddingsV5TextNano)
        );
        assert_eq!(
            ModelType::from_str("jina-code-1.5b"),
            Some(ModelType::JinaCodeEmbeddings15B)
        );
        assert_eq!(
            ModelType::from_str("jinaai/jina-code-embeddings-1.5b"),
            Some(ModelType::JinaCodeEmbeddings15B)
        );
        assert_eq!(
            ModelType::from_str("minilm-l6-q"),
            Some(ModelType::AllMiniLML6V2Q)
        );
        assert_eq!(
            ModelType::from_str("mxbai-xsmall"),
            Some(ModelType::MxbaiEmbedXSmallV1)
        );
        assert_eq!(ModelType::from_str("jina-code"), None);
        assert_eq!(ModelType::from_str("bge-small-q"), None);
        assert_eq!(ModelType::from_str("mxbai-large"), None);
        assert_eq!(ModelType::from_str("unknown"), None);
    }

    #[test]
    fn test_is_quantized() {
        assert!(ModelType::AllMiniLML6V2Q.is_quantized());
        assert!(!ModelType::JinaEmbeddingsV5TextNano.is_quantized());
        assert!(!ModelType::JinaCodeEmbeddings15B.is_quantized());
        assert!(!ModelType::MxbaiEmbedXSmallV1.is_quantized());
    }

    #[test]
    fn test_model_specific_formatting() {
        let query = "find auth";
        let passage = "Code:\nfn auth() {}";

        let mxbai_xs_query = ModelType::MxbaiEmbedXSmallV1.format_query(query);
        assert!(
            mxbai_xs_query.starts_with("Represent this sentence for searching relevant passages: ")
        );
        assert_eq!(
            ModelType::MxbaiEmbedXSmallV1.format_passage(passage),
            passage
        );

        let jina_v5_query = ModelType::JinaEmbeddingsV5TextNano.format_query(query);
        assert_eq!(jina_v5_query, "Query: find auth");
        assert_eq!(
            ModelType::JinaEmbeddingsV5TextNano.format_passage(passage),
            format!("Document: {}", passage)
        );
        assert!(ModelType::JinaEmbeddingsV5TextNano.has_special_passage_format());

        let jina_15b_query = ModelType::JinaCodeEmbeddings15B.format_query(query);
        assert!(jina_15b_query
            .starts_with("Find the most relevant code snippet given the following query:\n"));
        assert_eq!(
            ModelType::JinaCodeEmbeddings15B.format_passage(passage),
            format!("Candidate code snippet:\n{}", passage)
        );
        assert!(ModelType::JinaCodeEmbeddings15B.has_special_passage_format());

        assert_eq!(ModelType::AllMiniLML6V2Q.format_query(query), query);
        assert_eq!(ModelType::AllMiniLML6V2Q.format_passage(passage), passage);
    }

    #[test]
    #[ignore] // Requires downloading model
    fn test_embedder_creation() {
        let embedder = FastEmbedder::new();
        assert!(embedder.is_ok());

        let embedder = embedder.unwrap();
        assert_eq!(embedder.dimensions(), 384);
    }

    #[test]
    #[ignore] // Requires model
    fn test_embed_single_text() {
        let mut embedder = FastEmbedder::new().unwrap();
        let embedding = embedder.embed_one("Hello, world!").unwrap();

        assert_eq!(embedding.len(), 384);
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
            assert_eq!(embedding.len(), 384);
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
