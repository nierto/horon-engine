//! basic_usage.rs - Example of using the engine as a standalone module
//!
//! This example demonstrates how the engine can be used as a standalone library.
//! It illustrates:
//! - Configuring the HTT storage system
//! - Storing and retrieving data with metadata
//! - Working with nested paths
//! - Basic operations like listing keys and deleting data

use horon_engine::{HTTStorage, HTTStorageConfig};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Starting standalone engine example");

    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    println!("Storage initialized successfully");

    // Store some data
    println!("Storing data...");
    let data1 = b"Hello, HTT world!";
    let data2 = b"This is a test of HTT storage";

    // Store data with content type
    storage.store("/example/hello", data1, Some("text/plain".to_string()))?;

    // Store data without content type
    storage.store("/example/test", data2, None)?;

    // Add metadata
    storage.set_metadata("/example/hello", "author", "HTT Team")?;

    // Create nested data
    for i in 1..5 {
        let path = format!("/example/nested/level{}/item", i);
        let data = format!("Nested data at level {}", i);
        storage.store(&path, data.as_bytes(), None)?;
    }

    // Retrieve data
    println!("Retrieving data...");
    let retrieved1 = storage.retrieve("/example/hello")?;
    let retrieved2 = storage.retrieve("/example/test")?;

    assert_eq!(retrieved1, data1);
    assert_eq!(retrieved2, data2);

    println!("Data retrieval successful");

    // Get metadata
    let metadata = storage.get_metadata("/example/hello")?;
    println!("Metadata: {:?}", metadata);

    // List keys
    println!("Listing keys with prefix '/example'...");
    let keys = storage.list("/example")?;
    println!("Found {} keys", keys.len());
    for key in &keys {
        println!("  - {}", key);
    }

    // List keys with nested prefix
    println!("Listing keys with prefix '/example/nested'...");
    let nested_keys = storage.list("/example/nested")?;
    println!("Found {} nested keys", nested_keys.len());
    for key in &nested_keys {
        println!("  - {}", key);
    }

    // Delete data
    println!("Deleting '/example/hello'...");
    storage.delete("/example/hello")?;

    // Verify deletion
    let exists = storage.exists("/example/hello");
    println!("'/example/hello' exists: {}", exists);

    println!("Standalone engine example completed successfully");
    Ok(())
}
