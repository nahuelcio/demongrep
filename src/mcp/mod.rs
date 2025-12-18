//! MCP (Model Context Protocol) server for Claude Code integration
//!
//! Exposes demongrep's semantic search capabilities via the MCP protocol,
//! allowing AI assistants like Claude to search codebases during conversations.

use anyhow::Result;
use rmcp::{
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    schemars::JsonSchema,
    tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::embed::{EmbeddingService, ModelType};
use crate::vectordb::VectorStore;

/// Demongrep MCP service
pub struct DemongrepService {
    tool_router: ToolRouter<DemongrepService>,
    db_path: PathBuf,
    model_type: ModelType,
    dimensions: usize,
    // Lazily initialized on first search
    embedding_service: Mutex<Option<EmbeddingService>>,
}

impl std::fmt::Debug for DemongrepService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DemongrepService")
            .field("db_path", &self.db_path)
            .field("model_type", &self.model_type)
            .field("dimensions", &self.dimensions)
            .finish()
    }
}

// === Tool Request/Response Types ===

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SemanticSearchRequest {
    /// The search query (natural language or code snippet)
    pub query: String,

    /// Maximum number of results to return (default: 10)
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetFileChunksRequest {
    /// Path to the file (relative to project root)
    pub path: String,
}

#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub kind: String,
    pub content: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_prev: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_next: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct IndexStatusResponse {
    pub indexed: bool,
    pub total_chunks: usize,
    pub total_files: usize,
    pub model: String,
    pub dimensions: usize,
    pub db_path: String,
}

// === Tool Router Implementation ===

#[tool_router]
impl DemongrepService {
    /// Create a new DemongrepService
    pub fn new(db_path: PathBuf) -> Result<Self> {
        // Read model metadata from database
        let metadata_path = db_path.join("metadata.json");
        let (model_type, dimensions) = if metadata_path.exists() {
            let content = std::fs::read_to_string(&metadata_path)?;
            let json: serde_json::Value = serde_json::from_str(&content)?;
            let model_name = json
                .get("model_short_name")
                .and_then(|v| v.as_str())
                .unwrap_or("minilm-l6");
            let dims = json
                .get("dimensions")
                .and_then(|v| v.as_u64())
                .unwrap_or(384) as usize;
            let mt = ModelType::from_str(model_name).unwrap_or_default();
            (mt, dims)
        } else {
            (ModelType::default(), 384)
        };

        Ok(Self {
            tool_router: Self::tool_router(),
            db_path,
            model_type,
            dimensions,
            embedding_service: Mutex::new(None),
        })
    }

    /// Get or initialize the embedding service
    fn get_embedding_service(&self) -> Result<std::sync::MutexGuard<'_, Option<EmbeddingService>>> {
        let mut guard = self.embedding_service.lock().unwrap();
        if guard.is_none() {
            *guard = Some(EmbeddingService::with_model(self.model_type)?);
        }
        Ok(guard)
    }

    #[tool(description = "Search the codebase using semantic similarity. Returns code chunks that are semantically similar to the query.")]
    async fn semantic_search(
        &self,
        Parameters(request): Parameters<SemanticSearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let limit = request.limit.unwrap_or(10);

        // Check if database exists
        if !self.db_path.exists() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Error: No index found. Run 'demongrep index' first to index the codebase.",
            )]));
        }

        // Open vector store
        let store = match VectorStore::new(&self.db_path, self.dimensions) {
            Ok(s) => s,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error opening database: {}",
                    e
                ))]));
            }
        };

        // Get embedding service and embed query
        let mut service_guard = match self.get_embedding_service() {
            Ok(g) => g,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error initializing embedding service: {}",
                    e
                ))]));
            }
        };

        let service = service_guard.as_mut().unwrap();
        let query_embedding = match service.embed_query(&request.query) {
            Ok(e) => e,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error embedding query: {}",
                    e
                ))]));
            }
        };

        // Search
        let results = match store.search(&query_embedding, limit) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error searching: {}",
                    e
                ))]));
            }
        };

        if results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No results found for the query.",
            )]));
        }

        // Convert to response format
        let items: Vec<SearchResultItem> = results
            .into_iter()
            .map(|r| SearchResultItem {
                path: r.path,
                start_line: r.start_line,
                end_line: r.end_line,
                kind: r.kind,
                content: r.content,
                score: r.score,
                signature: r.signature,
                context_prev: r.context_prev,
                context_next: r.context_next,
            })
            .collect();

        let json = serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string());
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Get all indexed chunks from a specific file. Useful for understanding the structure of a file.")]
    async fn get_file_chunks(
        &self,
        Parameters(request): Parameters<GetFileChunksRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Check if database exists
        if !self.db_path.exists() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Error: No index found. Run 'demongrep index' first to index the codebase.",
            )]));
        }

        // Open vector store
        let store = match VectorStore::new(&self.db_path, self.dimensions) {
            Ok(s) => s,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error opening database: {}",
                    e
                ))]));
            }
        };

        // Get all chunks and filter by path
        // We need to iterate through all chunks since there's no path index
        // For now, we'll use a simple approach - search with empty query would require embedding
        // Instead, let's iterate the metadata database

        let stats = match store.stats() {
            Ok(s) => s,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error getting stats: {}",
                    e
                ))]));
            }
        };

        // Collect chunks for the requested file
        let mut file_chunks: Vec<SearchResultItem> = Vec::new();
        for id in 0..stats.total_chunks as u32 {
            if let Ok(Some(chunk)) = store.get_chunk(id) {
                // Normalize paths for comparison
                let chunk_path = chunk.path.trim_start_matches("./");
                let req_path = request.path.trim_start_matches("./");

                if chunk_path == req_path || chunk.path == request.path {
                    file_chunks.push(SearchResultItem {
                        path: chunk.path,
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        kind: chunk.kind,
                        content: chunk.content,
                        score: 1.0,
                        signature: chunk.signature,
                        context_prev: chunk.context_prev,
                        context_next: chunk.context_next,
                    });
                }
            }
        }

        // Sort by start line
        file_chunks.sort_by_key(|c| c.start_line);

        if file_chunks.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "No chunks found for file: {}",
                request.path
            ))]));
        }

        let json =
            serde_json::to_string_pretty(&file_chunks).unwrap_or_else(|_| "[]".to_string());
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Get the status of the semantic search index including model info and statistics.")]
    async fn index_status(&self) -> Result<CallToolResult, McpError> {
        // Check if database exists
        if !self.db_path.exists() {
            let response = IndexStatusResponse {
                indexed: false,
                total_chunks: 0,
                total_files: 0,
                model: "none".to_string(),
                dimensions: 0,
                db_path: self.db_path.display().to_string(),
            };
            let json =
                serde_json::to_string_pretty(&response).unwrap_or_else(|_| "{}".to_string());
            return Ok(CallToolResult::success(vec![Content::text(json)]));
        }

        // Open vector store and get stats
        let store = match VectorStore::new(&self.db_path, self.dimensions) {
            Ok(s) => s,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error opening database: {}",
                    e
                ))]));
            }
        };

        let stats = match store.stats() {
            Ok(s) => s,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error getting stats: {}",
                    e
                ))]));
            }
        };

        let response = IndexStatusResponse {
            indexed: stats.indexed,
            total_chunks: stats.total_chunks,
            total_files: stats.total_files,
            model: self.model_type.short_name().to_string(),
            dimensions: stats.dimensions,
            db_path: self.db_path.display().to_string(),
        };

        let json = serde_json::to_string_pretty(&response).unwrap_or_else(|_| "{}".to_string());
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

// === Server Handler Implementation ===

#[tool_handler]
impl ServerHandler for DemongrepService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: rmcp::model::Implementation {
                name: "demongrep".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                title: None,
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Demongrep is a semantic code search tool. Use semantic_search to find code \
                 by meaning, get_file_chunks to see all chunks in a file, and index_status \
                 to check if the index is ready."
                    .to_string(),
            ),
            ..Default::default()
        }
    }
}

/// Run the MCP server using stdio transport
pub async fn run_mcp_server(path: Option<PathBuf>) -> Result<()> {
    use rmcp::{transport::stdio, ServiceExt};

    // Determine database path
    let project_path = path.unwrap_or_else(|| PathBuf::from("."));
    let db_path = project_path.canonicalize()?.join(".demongrep.db");

    eprintln!("Starting demongrep MCP server...");
    eprintln!("Database path: {}", db_path.display());

    let service = DemongrepService::new(db_path)?;

    // Serve using stdio transport
    let server = service.serve(stdio()).await?;

    eprintln!("MCP server ready. Waiting for requests...");

    // Wait for shutdown
    server.waiting().await?;

    Ok(())
}
