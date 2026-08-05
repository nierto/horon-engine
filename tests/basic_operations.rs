//! =============================================================================
//! Engine Test Suite: Basic Operations
//! =============================================================================
//!
//! Integration tests for the core functionality of the Hyperbolic Tree Tensor
//! (HTT) Rust implementation.

use horon_engine::{HTTStorage, HTTStorageConfig};

#[test]
fn test_basic_operations() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    // Test data
    let path = "/test/basic";
    let data = b"Hello, HTT!";

    // Store data
    storage.store(path, data, None).unwrap();

    // Check existence
    assert!(storage.exists(path));

    // Retrieve data
    let retrieved = storage.retrieve(path).unwrap();
    assert_eq!(retrieved, data);

    // List keys (includes /test directory node created by ensure_parent_directories)
    let keys = storage.list("/test").unwrap();
    assert!(keys.contains(&path.to_string()));

    // Delete data
    storage.delete(path).unwrap();

    // Verify deletion
    assert!(!storage.exists(path));
}

#[test]
fn test_metadata_operations() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    // Test data
    let path = "/test/metadata";
    let data = b"Hello, HTT!";

    // Store data with content type
    storage.store(path, data, Some("text/plain".to_string())).unwrap();

    // Retrieve and check metadata
    let metadata = storage.get_metadata(path).unwrap();
    assert_eq!(metadata.get("content_type").unwrap(), "text/plain");

    // Set additional metadata
    storage.set_metadata(path, "author", "test").unwrap();

    // Verify updated metadata
    let updated_metadata = storage.get_metadata(path).unwrap();
    assert_eq!(updated_metadata.get("author").unwrap(), "test");

    // Clean up
    storage.delete(path).unwrap();
}
