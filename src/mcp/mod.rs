//! MCP (Model Context Protocol) server for Claude Code integration
//!
//! Exposes demongrep's semantic search capabilities via the MCP protocol,
//! allowing AI assistants like Claude to search codebases during conversations.
//!
//! **Now supports dual-database search**: Searches both local and global databases automatically.

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

use crate::database::DatabaseManager;  // NEW: Use DatabaseManager
use crate::embed::EmbeddingService;


/// Demongrep MCP service with dual-database support via DatabaseManager
pub struct DemongrepService {
    tool_router: ToolRouter<DemongrepService>,
    db_manager: DatabaseManager,  // NEW: Replaced db_paths with DatabaseManager
    // Lazily initialized on first search
    embedding_service: Mutex<Option<EmbeddingService>>,
}

impl std::fmt::Debug for DemongrepService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DemongrepService")
            .field("db_manager", &"<DatabaseManager>")
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub database: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct IndexStatusResponse {
    pub indexed: bool,
    pub total_chunks: usize,
    pub total_files: usize,
    pub local_chunks: usize,
    pub local_files: usize,
    pub global_chunks: usize,
    pub global_files: usize,
    pub model: String,
    pub dimensions: usize,
    pub databases: Vec<String>,
    pub databases_available: usize,
}

// === Tool Router Implementation ===

#[tool_router]
impl DemongrepService {
    /// Create a new DemongrepService with DatabaseManager
    pub fn new(db_manager: DatabaseManager) -> Result<Self> {
        Ok(Self {
            tool_router: Self::tool_router(),
            db_manager,
            embedding_service: Mutex::new(None),
        })
    }

    /// Get or initialize the embedding service
    fn get_embedding_service(&self) -> Result<std::sync::MutexGuard<'_, Option<EmbeddingService>>> {
        let mut guard = self.embedding_service.lock()
            .map_err(|e| anyhow::anyhow!("MCP embedding mutex poisoned: {}", e))?;
        if guard.is_none() {
            *guard = Some(EmbeddingService::with_model(self.db_manager.model_type())?);
        }
        Ok(guard)
    }

    #[tool(description = "Search the codebase using semantic similarity. Searches both local and global databases. Returns code chunks that are semantically similar to the query.")]
    async fn semantic_search(
        &self,
        Parameters(request): Parameters<SemanticSearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let limit = request.limit.unwrap_or(10);

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

        // Search across all databases using DatabaseManager
        let results = match self.db_manager.search_all(&query_embedding, limit) {
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
            .map(|r| {
                // Determine which database this came from based on path
                let database = self.db_manager.databases()
                    .iter()
                    .find(|db| r.path.starts_with(db.path.to_str().unwrap_or("")))
                    .map(|db| match db.db_type {
                        crate::database::DatabaseType::Local => "local".to_string(),
                        crate::database::DatabaseType::Global => "global".to_string(),
                    });

                SearchResultItem {
                    path: r.path,
                    start_line: r.start_line,
                    end_line: r.end_line,
                    kind: r.kind,
                    content: r.content,
                    score: r.score,
                    signature: r.signature,
                    context_prev: r.context_prev,
                    context_next: r.context_next,
                    database,
                }
            })
            .collect();

        let json = serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string());
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Get all indexed chunks from a specific file. Searches across all databases. Useful for understanding the structure of a file.")]
    async fn get_file_chunks(
        &self,
        Parameters(request): Parameters<GetFileChunksRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut all_file_chunks: Vec<SearchResultItem> = Vec::new();

        // Search across all databases
        for database in self.db_manager.databases() {
            let store = database.store();
            
            let stats = match store.stats() {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Collect chunks for the requested file
            for id in 0..stats.total_chunks as u32 {
                if let Ok(Some(chunk)) = store.get_chunk(id) {
                    // Normalize paths for comparison
                    let chunk_path = chunk.path.trim_start_matches("./");
                    let req_path = request.path.trim_start_matches("./");

                    if chunk_path == req_path || chunk.path == request.path {
                        let db_type = match database.db_type {
                            crate::database::DatabaseType::Local => "local",
                            crate::database::DatabaseType::Global => "global",
                        };

                        all_file_chunks.push(SearchResultItem {
                            path: chunk.path,
                            start_line: chunk.start_line,
                            end_line: chunk.end_line,
                            kind: chunk.kind,
                            content: chunk.content,
                            score: 1.0,
                            signature: chunk.signature,
                            context_prev: chunk.context_prev,
                            context_next: chunk.context_next,
                            database: Some(db_type.to_string()),
                        });
                    }
                }
            }
        }

        // Sort by start line
        all_file_chunks.sort_by_key(|c| c.start_line);

        if all_file_chunks.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "No chunks found for file: {}",
                request.path
            ))]));
        }

        let json =
            serde_json::to_string_pretty(&all_file_chunks).unwrap_or_else(|_| "[]".to_string());
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Get the status of the semantic search index including model info and statistics from all databases.")]
    async fn index_status(&self) -> Result<CallToolResult, McpError> {
        // Use DatabaseManager for stats - MUCH SIMPLER!
        let stats = match self.db_manager.combined_stats() {
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
            local_chunks: stats.local_chunks,
            local_files: stats.local_files,
            global_chunks: stats.global_chunks,
            global_files: stats.global_files,
            model: self.db_manager.model_type().short_name().to_string(),
            dimensions: stats.dimensions,
            databases: self.db_manager.database_paths().iter().map(|p| p.display().to_string()).collect(),
            databases_available: self.db_manager.database_count(),
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
                "Demongrep is a semantic code search tool with dual-database support. \
                 Use semantic_search to find code by meaning (searches both local and global databases), \
                 get_file_chunks to see all chunks in a file, and index_status \
                 to check if the index is ready and see stats from all databases."
                    .to_string(),
            ),
            ..Default::default()
        }
    }
}

/// Run the MCP server using stdio transport with DatabaseManager
pub async fn run_mcp_server(path: Option<PathBuf>) -> Result<()> {
    use rmcp::{transport::stdio, ServiceExt};

    // Use DatabaseManager to load all databases
    let db_manager = match DatabaseManager::load(path) {
        Ok(manager) => manager,
        Err(_) => {
            eprintln!("Error: No databases found!");
            eprintln!("Run 'demongrep index' or 'demongrep index --global' first.");
            return Err(anyhow::anyhow!("No databases found"));
        }
    };

    eprintln!("Starting demongrep MCP server...");
    eprintln!("Databases loaded:");
    for database in db_manager.databases() {
        eprintln!("  {} {}", 
            match database.db_type {
                crate::database::DatabaseType::Local => "📍 Local: ",
                crate::database::DatabaseType::Global => "🌍 Global:",
            },
            database.path.display()
        );
    }

    let service = DemongrepService::new(db_manager)?;

    // Serve using stdio transport
    let server = service.serve(stdio()).await?;

    eprintln!("MCP server ready. Waiting for requests...");

    // Wait for shutdown
    server.waiting().await?;

    Ok(())
}
