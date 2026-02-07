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

use crate::database::DatabaseManager; // NEW: Use DatabaseManager
use crate::embed::EmbeddingService;

/// Demongrep MCP service with dual-database support via DatabaseManager
pub struct DemongrepService {
    tool_router: ToolRouter<DemongrepService>,
    db_manager: DatabaseManager, // NEW: Replaced db_paths with DatabaseManager
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
    /// Offset for pagination (default: 0)
    pub offset: Option<usize>,
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

#[derive(Debug, Deserialize, JsonSchema)]
pub struct HybridSearchRequest {
    /// The search query (natural language or code snippet)
    pub query: String,
    /// Maximum number of results to return (default: 10)
    pub limit: Option<usize>,
    /// Offset for pagination (default: 0)
    pub offset: Option<usize>,
    /// Filter results to a specific path prefix (e.g., "src/")
    pub filter_path: Option<String>,
    /// RRF k parameter for score fusion (default: 20)
    pub rrf_k: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReindexRequest {
    /// Optional path to reindex (defaults to project root)
    pub path: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindDefinitionsRequest {
    /// Optional name pattern to search for (e.g., "auth", "parse")
    pub pattern: Option<String>,
    /// Filter by kind: "Function", "Struct", "Trait", "Method", "Enum", "Impl", "Class", "Interface"
    pub kind: Option<String>,
    /// Maximum results (default: 20)
    pub limit: Option<usize>,
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
        let mut guard = self
            .embedding_service
            .lock()
            .map_err(|e| anyhow::anyhow!("MCP embedding mutex poisoned: {}", e))?;
        if guard.is_none() {
            *guard = Some(EmbeddingService::with_model(self.db_manager.model_type())?);
        }
        Ok(guard)
    }

    #[tool(
        description = "Search the codebase using semantic similarity. Searches both local and global databases. Returns code chunks that are semantically similar to the query."
    )]
    async fn semantic_search(
        &self,
        Parameters(request): Parameters<SemanticSearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let limit = request.limit.unwrap_or(10);
        let offset = request.offset.unwrap_or(0);

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
        let results = match self.db_manager.search_all(&query_embedding, limit, offset) {
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
                let database = self
                    .db_manager
                    .databases()
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

    #[tool(
        description = "Get all indexed chunks from a specific file. Searches across all databases. Useful for understanding the structure of a file."
    )]
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

    #[tool(
        description = "Search the codebase using hybrid search (vector similarity + BM25 full-text + RRF fusion). More accurate than semantic_search alone. Searches both local and global databases."
    )]
    async fn hybrid_search(
        &self,
        Parameters(request): Parameters<HybridSearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let limit = request.limit.unwrap_or(10);
        let offset = request.offset.unwrap_or(0);
        let rrf_k = request.rrf_k.unwrap_or(20.0);

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

        // Use hybrid search (vector + FTS + RRF)
        let mut results = match self.db_manager.hybrid_search_all(
            &request.query,
            &query_embedding,
            limit,
            offset,
            rrf_k,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error searching: {}",
                    e
                ))]));
            }
        };

        // Filter by path if specified
        if let Some(ref filter) = request.filter_path {
            let filter_normalized = filter.trim_start_matches("./");
            results.retain(|r| {
                let path_normalized = r.path.trim_start_matches("./");
                path_normalized.starts_with(filter_normalized)
            });
        }

        if results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No results found for the query.",
            )]));
        }

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
                database: None,
            })
            .collect();

        let json = serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string());
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Re-index changed files in all databases. Detects modified, new, and deleted files and updates the index incrementally."
    )]
    async fn reindex(
        &self,
        Parameters(_request): Parameters<ReindexRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut total_changes = 0;
        let model_type = self.db_manager.model_type();

        for database in self.db_manager.databases() {
            match crate::search::sync_database(&database.path, model_type) {
                Ok(()) => {
                    total_changes += 1; // At minimum we checked
                }
                Err(e) => {
                    return Ok(CallToolResult::success(vec![Content::text(format!(
                        "Error reindexing {} database: {}",
                        database.db_type.name(),
                        e
                    ))]));
                }
            }
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Reindex complete. Checked {} database(s).",
            total_changes
        ))]))
    }

    #[tool(
        description = "Find code definitions (functions, structs, traits, methods, etc.) across all databases. Useful for navigating the codebase structure."
    )]
    async fn find_definitions(
        &self,
        Parameters(request): Parameters<FindDefinitionsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let limit = request.limit.unwrap_or(20);
        let mut definitions: Vec<SearchResultItem> = Vec::new();

        for database in self.db_manager.databases() {
            let store = database.store();

            let stats = match store.stats() {
                Ok(s) => s,
                Err(_) => continue,
            };

            for id in 0..stats.total_chunks as u32 {
                if let Ok(Some(chunk)) = store.get_chunk(id) {
                    // Skip non-definition kinds (Block, Anchor, Other)
                    let kind_lower = chunk.kind.to_lowercase();
                    if kind_lower == "block" || kind_lower == "anchor" || kind_lower == "other" {
                        continue;
                    }

                    // Filter by kind if specified
                    if let Some(ref kind_filter) = request.kind {
                        if !kind_lower.contains(&kind_filter.to_lowercase()) {
                            continue;
                        }
                    }

                    // Filter by pattern if specified (matches signature or content)
                    if let Some(ref pattern) = request.pattern {
                        let pattern_lower = pattern.to_lowercase();
                        let matches_signature = chunk
                            .signature
                            .as_ref()
                            .map(|s| s.to_lowercase().contains(&pattern_lower))
                            .unwrap_or(false);
                        let matches_content = chunk.content.to_lowercase().contains(&pattern_lower);

                        if !matches_signature && !matches_content {
                            continue;
                        }
                    }

                    let db_type = match database.db_type {
                        crate::database::DatabaseType::Local => "local",
                        crate::database::DatabaseType::Global => "global",
                    };

                    definitions.push(SearchResultItem {
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

                    if definitions.len() >= limit {
                        break;
                    }
                }
            }

            if definitions.len() >= limit {
                break;
            }
        }

        // Sort by path and line
        definitions.sort_by(|a, b| a.path.cmp(&b.path).then(a.start_line.cmp(&b.start_line)));

        if definitions.is_empty() {
            let msg = match (&request.kind, &request.pattern) {
                (Some(k), Some(p)) => format!("No {} definitions matching '{}' found.", k, p),
                (Some(k), None) => format!("No {} definitions found.", k),
                (None, Some(p)) => format!("No definitions matching '{}' found.", p),
                (None, None) => "No definitions found.".to_string(),
            };
            return Ok(CallToolResult::success(vec![Content::text(msg)]));
        }

        let json = serde_json::to_string_pretty(&definitions).unwrap_or_else(|_| "[]".to_string());
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Get the status of the semantic search index including model info and statistics from all databases."
    )]
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
            databases: self
                .db_manager
                .database_paths()
                .iter()
                .map(|p| p.display().to_string())
                .collect(),
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
                 Tools: semantic_search (vector similarity), hybrid_search (vector + BM25 + RRF fusion, most accurate), \
                 find_definitions (browse functions/structs/traits by kind and name), \
                 get_file_chunks (see all chunks in a file), reindex (update index with file changes), \
                 index_status (check index readiness and stats)."
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
        eprintln!(
            "  {} {}",
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
