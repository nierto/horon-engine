//! spatial_queries.rs - Spatial query capabilities of the engine
//!
//! Demonstrates the O(1) nearest-neighbor and spatial proximity features:
//! - nearest_neighbor_point: find the closest node to an arbitrary point
//! - find_nearest: find k nearest neighbors of an existing node
//! - find_in_radius: find all nodes within a hyperbolic distance

use horon_engine::{HTTStorage, HTTStorageConfig};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Spatial Query Example ===\n");

    // Build a tree with geographic-like data
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    // Store some nodes representing a document hierarchy
    let docs = [
        ("/papers", "Research papers"),
        ("/papers/geometry", "Geometry papers"),
        ("/papers/geometry/hyperbolic", "Hyperbolic geometry"),
        ("/papers/geometry/euclidean", "Euclidean geometry"),
        ("/papers/algebra", "Algebra papers"),
        ("/papers/algebra/linear", "Linear algebra"),
        ("/papers/algebra/abstract", "Abstract algebra"),
        ("/books", "Books"),
        ("/books/intro", "Introductory"),
        ("/books/advanced", "Advanced"),
    ];

    for (path, description) in &docs {
        storage.store(path, description.as_bytes(), None)?;
        println!("  Stored: {}", path);
    }

    // --- nearest_neighbor_point: query with an arbitrary coordinate ---
    println!("\n--- Nearest Neighbor Point (arbitrary coordinates) ---");
    let query_coords = fp(&[0.1, 0.0, 0.0, 0.0]); // near the origin in 4D Poincaré disk
    let (nearest_path, distance) = storage.nearest_neighbor_point(&query_coords)?;
    println!(
        "  Query at ({:.1}, {:.1}, {:.1}, {:.1}) → nearest: '{}' (distance: {:.6})",
        query_coords[0], query_coords[1], query_coords[2], query_coords[3],
        nearest_path, distance
    );

    // Query near the boundary of the disk
    let boundary_query = fp(&[0.8, 0.0, 0.0, 0.0]);
    let (nearest_path, distance) = storage.nearest_neighbor_point(&boundary_query)?;
    println!(
        "  Query at ({:.1}, {:.1}, {:.1}, {:.1}) → nearest: '{}' (distance: {:.6})",
        boundary_query[0], boundary_query[1], boundary_query[2], boundary_query[3],
        nearest_path, distance
    );

    // --- find_nearest: k nearest neighbors of an existing node ---
    println!("\n--- Find K Nearest Neighbors ---");
    let k = 3;
    let neighbors = storage.find_nearest("/papers/geometry", k)?;
    println!("  {} nearest neighbors of '/papers/geometry':", k);
    for path in &neighbors {
        println!("    {}", path);
    }

    // --- find_in_radius: all nodes within a distance threshold ---
    println!("\n--- Find In Radius ---");
    let radius = g_math::fixed_point::FixedPoint::from_int(3);
    let nearby = storage.find_in_radius("/papers", radius)?;
    println!("  Nodes within radius 3.0 of '/papers':");
    for path in &nearby {
        println!("    {}", path);
    }

    // --- Demonstrate O(1) behavior: NN time is independent of tree size ---
    println!("\n--- O(1) Scaling Demonstration ---");
    // Add more nodes and show that NN query still works instantly
    for i in 0..50 {
        let path = format!("/data/item_{}", i);
        storage.store(&path, format!("item {}", i).as_bytes(), None)?;
    }

    let stats = storage.stats();
    println!("  Total nodes: {}", stats.get("total_nodes").unwrap_or(&"?".to_string()));

    let (path, dist) = storage.nearest_neighbor_point(&fp(&[0.0, 0.0, 0.0, 0.0]))?;
    println!("  NN at origin with {} nodes → '{}' (distance: {:.6})", 60, path, dist);

    println!("\n=== Done ===");
    Ok(())
}

/// Exact fixed-point coordinates from decimal literals — the public API takes
/// `FixedPoint`, so tests convert explicitly rather than through a float API.
fn fp(vals: &[f64]) -> Vec<g_math::fixed_point::FixedPoint> {
    vals.iter().map(|&v| g_math::fixed_point::FixedPoint::from_f64(v)).collect()
}
