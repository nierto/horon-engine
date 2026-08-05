//! store_wrapper.rs - Using the Store wrapper API
//!
//! Demonstrates the ergonomic `Store` API that hides all internal types
//! behind standard Rust types. This is the recommended entry point for
//! most users.

use horon_engine::{Store, StoreConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Create a store (default: 10,000 node capacity) ---
    let store = Store::new();
    assert!(store.is_empty());

    // --- Basic CRUD ---
    store.put("/config/db", b"postgres://localhost:5432")?;
    store.put("/config/cache", b"redis://localhost:6379")?;
    store.put("/config/cache/ttl", b"3600")?;

    let db = store.get("/config/db")?;
    println!("db = {}", String::from_utf8_lossy(&db));

    assert!(store.exists("/config/cache"));
    println!("entries: {}", store.len());

    // --- Metadata ---
    store.set_meta("/config/db", "env", "production")?;
    store.set_meta("/config/db", "owner", "platform-team")?;

    let meta = store.get_meta("/config/db")?;
    println!("db metadata: {:?}", meta);

    // --- Hierarchy traversal ---
    let children = store.children("/config")?;
    println!("children of /config: {:?}", children);

    let all = store.list("/config")?;
    println!("all under /config: {:?}", all);

    // --- Spatial queries ---
    // Find the nearest stored node to an arbitrary point in hyperbolic space
    let (path, distance) = store.nearest(&fp(&[0.0, 0.0, 0.0, 0.0]))?;
    println!("nearest to origin: '{}' (distance {:.6})", path, distance);

    // Find k nearest neighbors of an existing node
    let neighbors = store.neighbors("/config/cache", 2)?;
    println!("2 nearest neighbors of /config/cache: {:?}", neighbors);

    // Find all nodes within a hyperbolic distance
    let nearby = store.find_within("/config", g_math::fixed_point::FixedPoint::from_f64(5.0))?;
    println!("nodes within radius 5.0 of /config: {:?}", nearby);

    // --- Delete ---
    store.remove("/config/cache/ttl")?;
    assert!(!store.exists("/config/cache/ttl"));
    println!("entries after delete: {}", store.len());

    // --- Custom capacity ---
    let small = Store::with_config(StoreConfig::new().capacity(100));
    println!("small store empty: {}", small.is_empty());

    // --- Escape hatch to HTTStorage ---
    let stats = store.inner().stats();
    println!("internal stats: {:?}", stats);

    println!("\nDone.");
    Ok(())
}

/// Exact fixed-point coordinates from decimal literals — the public API takes
/// `FixedPoint`, so tests convert explicitly rather than through a float API.
fn fp(vals: &[f64]) -> Vec<g_math::fixed_point::FixedPoint> {
    vals.iter().map(|&v| g_math::fixed_point::FixedPoint::from_f64(v)).collect()
}
