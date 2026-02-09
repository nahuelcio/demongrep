use crate::embed::{EmbeddingService, ModelType};
use anyhow::{anyhow, Result};

pub async fn run(model: Option<String>) -> Result<()> {
    let model_type = match model {
        Some(name) => ModelType::from_str(&name).ok_or_else(|| {
            anyhow!(
                "Unknown model '{}'. Use --model with one of: minilm-l6-q, bge-small-q, jina-code, mxbai-large, mxbai-xsmall",
                name
            )
        })?,
        None => ModelType::default(),
    };

    println!("Setting up demongrep model cache...");
    println!("  Model: {}", model_type.name());
    println!("  Dimensions: {}", model_type.dimensions());
    println!("  This will download model files if they are not present.");

    // Loading the embedding service forces model initialization and local caching.
    let service = EmbeddingService::with_model(model_type).map_err(|e| {
        anyhow!(
            "Failed to initialize/download model '{}': {}.\nIf this is a missing ONNX Runtime library issue, install demongrep using the release installer so runtime libraries are bundled.",
            model_type.name(),
            e
        )
    })?;

    println!("Setup complete.");
    println!("  Ready model: {}", service.model_name());
    println!("  Next steps:");
    println!("    1) demongrep index");
    println!("    2) demongrep search \"where do we handle authentication?\"");
    Ok(())
}
