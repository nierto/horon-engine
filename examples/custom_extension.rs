//! # Custom Extensions Example
//!
//! This example demonstrates engine storage operations with metadata.
//! It shows how to:
//!
//! - Store documents as byte data
//! - Attach and retrieve metadata
//! - List and search stored documents

use horon_engine::{HTTStorage, HTTStorageConfig};
use serde::{Serialize, Deserialize};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// Define a custom data type for documents
#[derive(Clone, Debug, Serialize, Deserialize)]
struct DocumentMetadata {
    title: String,
    author: String,
    keywords: Vec<String>,
    created_at: String,
    version: u32,
}

fn main() -> Result<()> {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    // Create sample documents
    let doc1 = DocumentMetadata {
        title: "Introduction to HTT".to_string(),
        author: "Geometric Team".to_string(),
        keywords: vec!["hyperbolic".to_string(), "tensor".to_string(), "tree".to_string()],
        created_at: "2023-05-15".to_string(),
        version: 1,
    };

    let doc2 = DocumentMetadata {
        title: "Advanced HTT Applications".to_string(),
        author: "Tensor Network Specialists".to_string(),
        keywords: vec!["hyperbolic".to_string(), "application".to_string(), "performance".to_string()],
        created_at: "2023-06-20".to_string(),
        version: 2,
    };

    // Serialize and store documents
    let doc1_data = serde_json::to_vec(&doc1).unwrap();
    let doc2_data = serde_json::to_vec(&doc2).unwrap();

    storage.store("/documents/intro.json", &doc1_data, Some("application/json".to_string()))?;
    storage.store("/documents/advanced.json", &doc2_data, Some("application/json".to_string()))?;

    // Add searchable metadata
    for keyword in &doc1.keywords {
        storage.set_metadata("/documents/intro.json", &format!("keyword_{}", keyword), "true")?;
    }
    for keyword in &doc2.keywords {
        storage.set_metadata("/documents/advanced.json", &format!("keyword_{}", keyword), "true")?;
    }

    // List all documents
    let stored_docs = storage.list("/documents")?;
    println!("Stored documents: {:?}", stored_docs);

    // Retrieve and verify
    let doc1_bytes = storage.retrieve("/documents/intro.json")?;
    let doc1_recovered: DocumentMetadata = serde_json::from_slice(&doc1_bytes).unwrap();
    println!("Document 1: {:?}", doc1_recovered);

    // Check metadata
    let metadata = storage.get_metadata("/documents/intro.json")?;
    println!("Document 1 metadata: {:?}", metadata);

    // Find documents by keyword
    let tensor_docs = find_docs_by_keyword(&storage, "tensor")?;
    println!("Documents with 'tensor' keyword: {:?}", tensor_docs);

    Ok(())
}

fn find_docs_by_keyword(storage: &HTTStorage, keyword: &str) -> Result<Vec<String>> {
    let mut matching = Vec::new();
    let all_docs = storage.list("/documents")?;

    let meta_key = format!("keyword_{}", keyword);
    for doc_path in all_docs {
        let metadata = storage.get_metadata(&doc_path)?;
        if metadata.contains_key(&meta_key) {
            matching.push(doc_path);
        }
    }

    Ok(matching)
}
