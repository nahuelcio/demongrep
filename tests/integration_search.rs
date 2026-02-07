use anyhow::Result;
use demongrep::chunker::{Chunk, ChunkKind};
use demongrep::database::DatabaseManagerBuilder;
use demongrep::embed::{EmbeddedChunk, ModelType};
use demongrep::fts::FtsStore;
use demongrep::vectordb::VectorStore;
use tempfile::Builder;

fn workspace_tempdir(prefix: &str) -> Result<tempfile::TempDir> {
    let cwd = std::env::current_dir()?;
    Ok(Builder::new().prefix(prefix).tempdir_in(cwd)?)
}

fn make_embedded_chunk(
    path: &str,
    content: &str,
    signature: Option<&str>,
    kind: ChunkKind,
    embedding: Vec<f32>,
) -> EmbeddedChunk {
    let mut chunk = Chunk::new(content.to_string(), 1, 3, kind, path.to_string());
    chunk.signature = signature.map(|s| s.to_string());
    chunk.string_literals = Chunk::extract_string_literals(content);
    EmbeddedChunk::new(chunk, embedding)
}

#[test]
fn integration_vector_search_pagination_across_databases() -> Result<()> {
    let root = workspace_tempdir("itest-pag-")?;
    let local_db = root.path().join(".demongrep.db");
    let global_db = root.path().join("global_store");

    let mut local_store = VectorStore::new(&local_db, 4)?;
    local_store.insert_chunks_with_ids(vec![
        make_embedded_chunk(
            "a.rs",
            "fn alpha() {}",
            Some("alpha"),
            ChunkKind::Function,
            vec![1.0, 0.0, 0.0, 0.0],
        ),
        make_embedded_chunk(
            "c.rs",
            "fn gamma() {}",
            Some("gamma"),
            ChunkKind::Function,
            vec![0.6, 0.8, 0.0, 0.0],
        ),
    ])?;
    local_store.build_index()?;

    let mut global_store = VectorStore::new(&global_db, 4)?;
    global_store.insert_chunks_with_ids(vec![
        make_embedded_chunk(
            "b.rs",
            "fn beta() {}",
            Some("beta"),
            ChunkKind::Function,
            vec![0.8, 0.6, 0.0, 0.0],
        ),
        make_embedded_chunk(
            "d.rs",
            "fn delta() {}",
            Some("delta"),
            ChunkKind::Function,
            vec![0.0, 1.0, 0.0, 0.0],
        ),
    ])?;
    global_store.build_index()?;

    let manager = DatabaseManagerBuilder::new()
        .add_database(local_db)
        .add_database(global_db)
        .with_model_type(ModelType::default())
        .with_dimensions(4)
        .build()?;

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let page = manager.search_all(&query, 2, 1)?;
    let paths: Vec<String> = page.iter().map(|r| r.path.clone()).collect();

    assert_eq!(paths, vec!["b.rs".to_string(), "c.rs".to_string()]);
    Ok(())
}

#[test]
fn integration_hybrid_search_respects_offset_and_code_tokenization() -> Result<()> {
    let root = workspace_tempdir("itest-hybrid-")?;
    let db_path = root.path().join(".demongrep.db");

    let mut store = VectorStore::new(&db_path, 4)?;
    let chunks = vec![
        make_embedded_chunk(
            "config.rs",
            "struct UserConfig { name: String }",
            Some("UserConfig"),
            ChunkKind::Struct,
            vec![1.0, 0.0, 0.0, 0.0],
        ),
        make_embedded_chunk(
            "process.rs",
            "fn process_data(input: &[u8]) {}",
            Some("process_data"),
            ChunkKind::Function,
            vec![0.8, 0.6, 0.0, 0.0],
        ),
        make_embedded_chunk(
            "other.rs",
            "fn helper() {}",
            Some("helper"),
            ChunkKind::Function,
            vec![0.2, 0.98, 0.0, 0.0],
        ),
    ];

    let ids = store.insert_chunks_with_ids(chunks.clone())?;
    store.build_index()?;

    let mut fts = FtsStore::new(&db_path)?;
    for (id, embedded) in ids.iter().zip(chunks.iter()) {
        fts.add_chunk(
            *id,
            &embedded.chunk.content,
            &embedded.chunk.path,
            embedded.chunk.signature.as_deref(),
            &format!("{:?}", embedded.chunk.kind),
            &embedded.chunk.string_literals,
        )?;
    }
    fts.commit()?;

    let manager = DatabaseManagerBuilder::new()
        .add_database(db_path)
        .with_model_type(ModelType::default())
        .with_dimensions(4)
        .build()?;

    let query_embedding = vec![1.0, 0.0, 0.0, 0.0];
    let first_page = manager.hybrid_search_all("user config", &query_embedding, 1, 0, 20.0)?;
    assert_eq!(first_page.len(), 1);
    assert_eq!(first_page[0].path, "config.rs");

    let second_page = manager.hybrid_search_all("user config", &query_embedding, 1, 1, 20.0)?;
    assert_eq!(second_page.len(), 1);
    assert_ne!(second_page[0].path, "config.rs");

    Ok(())
}
