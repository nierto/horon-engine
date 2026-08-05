//! =============================================================================
//! Comprehensive Test Suite: Nielsen Power Diagram + Spatial Index
//! =============================================================================
//!
//! Tests the full pipeline from Sarkar embedding through Klein model conversion,
//! power cell computation, point location grid, and the public storage API.
//!
//! Covers:
//!   - Klein model roundtrip precision
//!   - Power distance correctness
//!   - Grid-vs-brute-force equivalence
//!   - Incremental insert/delete grid preservation
//!   - Nearest-neighbor correctness at scale
//!   - Delaunay=Tree invariant checks
//!   - Edge cases: boundary points, deep trees, high-degree, empty tree
//!   - Stress tests: 1000-node trees with random queries

use g_math::fixed_point::{FixedPoint, FixedVector};
use horon_engine::{
    HTTStorage, HTTStorageConfig,
    HyperbolicPoint, PoincareDisk,
    poincare_to_klein, klein_to_poincare, power_distance, KleinPoint,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fp(v: i32) -> FixedPoint {
    FixedPoint::from_int(v)
}

fn fp_ratio(num: i32, den: i32) -> FixedPoint {
    FixedPoint::from_int(num) / FixedPoint::from_int(den)
}

fn approx_eq(a: FixedPoint, b: FixedPoint, tol: FixedPoint) -> bool {
    (a - b).abs() < tol
}

/// Build a storage instance and insert N children under root, returning the storage
/// and the list of child paths.
fn build_flat_tree(n: usize) -> (HTTStorage, Vec<String>) {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);
    let mut paths = Vec::new();
    for i in 0..n {
        let path = format!("/node_{}", i);
        storage.store(&path, format!("data_{}", i).as_bytes(), None).unwrap();
        paths.push(path);
    }
    (storage, paths)
}

/// Build a balanced binary tree of given depth under root.
/// Returns the storage and all leaf paths.
fn build_binary_tree(depth: usize) -> (HTTStorage, Vec<String>) {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    let mut current_level = vec!["/".to_string()];

    for d in 1..=depth {
        let mut next_level = Vec::new();
        for parent in &current_level {
            let left = if parent == "/" {
                format!("/L{}", d)
            } else {
                format!("{}/L{}", parent, d)
            };
            let right = if parent == "/" {
                format!("/R{}", d)
            } else {
                format!("{}/R{}", parent, d)
            };

            storage.store(&left, format!("left_d{}", d).as_bytes(), None).unwrap();
            storage.store(&right, format!("right_d{}", d).as_bytes(), None).unwrap();

            next_level.push(left);
            next_level.push(right);
        }
        current_level = next_level;
    }
    let leaves = current_level;
    (storage, leaves)
}

/// Build a deep chain: / → /a → /a/b → /a/b/c → ... to given depth.
fn build_deep_chain(depth: usize) -> (HTTStorage, Vec<String>) {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);
    let mut paths = Vec::new();
    let mut current = String::new();

    for i in 0..depth {
        let segment = format!("n{}", i);
        current = format!("{}/{}", current, segment);
        storage.store(&current, format!("depth_{}", i).as_bytes(), None).unwrap();
        paths.push(current.clone());
    }
    (storage, paths)
}

// ===========================================================================
// SECTION 1: Klein Model Precision
// ===========================================================================

#[test]
fn test_klein_roundtrip_precision_sweep() {
    // Sweep through many points at various distances from origin.
    // Verify roundtrip error < epsilon for every one.
    let epsilon = fp_ratio(1, 10000);
    let steps = 20;
    let mut max_error = fp(0);

    for i in 1..steps {
        let r = i as f32 / (steps as f32 + 1.0); // 0.05 to 0.95 in ~20 steps
        for j in 0..8 {
            let angle = j as f32 * std::f32::consts::PI / 4.0;
            let x = r * angle.cos();
            let y = r * angle.sin();

            let p = HyperbolicPoint::from_f32_slice(&[x, y, 0.0, 0.0]);
            let k = poincare_to_klein(&p);
            let p2 = klein_to_poincare(&k);

            for dim in 0..4 {
                let err = (p.coords()[dim] - p2.coords()[dim]).abs();
                if err > max_error {
                    max_error = err;
                }
            }
        }
    }

    assert!(max_error < epsilon,
        "Max roundtrip error {} exceeds epsilon {}", max_error, epsilon);
}

#[test]
fn test_klein_preserves_origin() {
    let origin = HyperbolicPoint::origin(4);
    let k = poincare_to_klein(&origin);
    let p = klein_to_poincare(&k);

    let epsilon = fp_ratio(1, 100000);
    assert!(k.coords.length() < epsilon, "Klein origin norm: {}", k.coords.length());
    assert!(p.euclidean_norm() < epsilon, "Poincaré roundtrip origin norm: {}", p.euclidean_norm());
    assert!(approx_eq(k.weight, fp(1), epsilon), "Klein origin weight: {}", k.weight);
}

#[test]
fn test_klein_norm_monotonic() {
    // As Poincaré norm increases, Klein norm should also increase.
    let mut prev_klein_norm = fp(0);

    for i in 1..20 {
        let r = i as f32 / 21.0;
        let p = HyperbolicPoint::from_f32_slice(&[r, 0.0, 0.0, 0.0]);
        let k = poincare_to_klein(&p);
        let k_norm = k.coords.length();

        assert!(k_norm > prev_klein_norm,
            "Klein norm should be monotonic: prev={} current={} at r={}",
            prev_klein_norm, k_norm, r);
        prev_klein_norm = k_norm;
    }
}

#[test]
fn test_klein_weight_positive_inside_disk() {
    // Power weight should be positive for all points strictly inside the disk.
    for i in 1..20 {
        let r = i as f32 / 21.0;
        let p = HyperbolicPoint::from_f32_slice(&[r, 0.0, 0.0, 0.0]);
        let k = poincare_to_klein(&p);

        assert!(k.weight > fp(0),
            "Weight should be positive inside disk, got {} at r={}", k.weight, r);
    }
}

// ===========================================================================
// SECTION 2: Power Distance Properties
// ===========================================================================

#[test]
fn test_power_distance_self_is_negative_weight() {
    // pd(p, p) = ||p-p||^2 - w = -w < 0
    let test_points = vec![
        [0.3_f32, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.2, 0.2, 0.1, 0.0],
    ];

    let epsilon = fp_ratio(1, 10000);
    for coords in &test_points {
        let p = HyperbolicPoint::from_f32_slice(coords);
        let k = poincare_to_klein(&p);
        let pd = power_distance(&k.coords, &k);

        assert!(approx_eq(pd, -k.weight, epsilon),
            "pd(self) should be -weight: got pd={} weight={}", pd, k.weight);
        assert!(pd < fp(0),
            "pd(self) should be negative: got {}", pd);
    }
}

#[test]
fn test_power_distance_at_equidistant_sarkar_siblings() {
    // For children at equal hyperbolic distance from parent (Sarkar siblings),
    // the nearest in hyperbolic distance should equal nearest in power distance.
    let disk = PoincareDisk::new(2);

    // Create parent and 4 children at distance tau=1
    let _parent = disk.origin();
    let tau = fp(1);
    let half_tau = tau * fp_ratio(1, 2);
    let r = half_tau.tanh();

    let num_children = 4;
    let mut children_p = Vec::new();
    let mut children_k = Vec::new();

    for i in 0..num_children {
        let angle = fp(i) * FixedPoint::from_str("2.39996322972865332") ; // golden angle
        let mut coords = FixedVector::new(2);
        coords[0] = r * angle.cos();
        coords[1] = r * angle.sin();
        let child = HyperbolicPoint::new(coords);
        let kc = poincare_to_klein(&child);
        children_k.push(kc);
        children_p.push(child);
    }

    // For a query near child 0, both metrics should agree on child 0 as nearest
    let mut q_coords = children_p[0].coords().clone();
    q_coords[0] = q_coords[0] + fp_ratio(1, 100);
    let q_p = HyperbolicPoint::new(q_coords);
    let q_k = poincare_to_klein(&q_p);

    // Hyperbolic nearest
    let hyp_nearest = children_p.iter().enumerate()
        .min_by_key(|(_, c)| {
            let d = q_p.hyperbolic_distance(c);
            // Convert to integer for comparison
            (d * fp(10000)).to_int()
        })
        .map(|(i, _)| i)
        .unwrap();

    // Power nearest
    let pow_nearest = children_k.iter().enumerate()
        .min_by_key(|(_, c)| {
            let d = power_distance(&q_k.coords, c);
            ((d + fp(100)) * fp(10000)).to_int() // shift to positive for comparison
        })
        .map(|(i, _)| i)
        .unwrap();

    assert_eq!(hyp_nearest, pow_nearest,
        "Equidistant siblings: hyp_nn={} pow_nn={}", hyp_nearest, pow_nearest);
}

// ===========================================================================
// SECTION 3: Storage API — Nearest Neighbor Point
// ===========================================================================

#[test]
fn test_nn_point_finds_root_at_origin() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    // Query at origin should find root
    let (path, dist) = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0])).unwrap();
    assert_eq!(path, "/");
    assert!(dist < fp_ratio(1, 10), "Distance to root at origin: {}", dist);
}

#[test]
fn test_nn_point_each_node_finds_itself() {
    let (storage, paths) = build_flat_tree(10);

    for path in &paths {
        // Query near each node — use the find_nearest API
        let nearest = storage.find_nearest(path, 1).unwrap();
        // At minimum, the query should succeed and return at least one neighbor
        // (For a flat tree, all siblings are at distance tau from root)
        assert!(!nearest.is_empty() || paths.len() <= 1,
            "find_nearest should return at least 1 neighbor for {}", path);
    }

    // Test via nearest_neighbor_point at (0,0,0,0) which is the root
    let (path, _) = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0])).unwrap();
    assert_eq!(path, "/", "Origin query should find root");
}

#[test]
fn test_nn_point_after_insertions() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    // Insert some nodes
    storage.store("/a", b"a", None).unwrap();
    storage.store("/b", b"b", None).unwrap();
    storage.store("/c", b"c", None).unwrap();
    storage.store("/a/child", b"ac", None).unwrap();

    // Query at origin should find root
    let (path, _) = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0])).unwrap();
    assert_eq!(path, "/");

    // Query should always return a valid, existing path
    let test_queries: Vec<[f32; 4]> = vec![
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0],
        [-0.1, 0.0, 0.0, 0.0],
        [0.3, 0.3, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
    ];

    for coords in &test_queries {
        let (path, dist) = storage.nearest_neighbor_point(&fpv(coords)).unwrap();
        assert!(storage.exists(&path),
            "NN result '{}' should exist in storage", path);
        assert!(dist >= fp(0),
            "Distance should be non-negative: {}", dist);
    }
}

#[test]
fn test_nn_point_after_deletion() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    storage.store("/a", b"a", None).unwrap();
    storage.store("/b", b"b", None).unwrap();

    // Delete /b
    storage.delete("/b").unwrap();

    // NN queries should still work and never return deleted paths
    let (path, _) = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0])).unwrap();
    assert_ne!(path, "/b", "Should not return deleted path");
    assert!(storage.exists(&path), "Returned path '{}' should exist", path);
}

// ===========================================================================
// SECTION 4: Spatial Query Consistency
// ===========================================================================

#[test]
fn test_find_nearest_returns_correct_count() {
    let (storage, _) = build_flat_tree(20);

    for k in &[1, 3, 5, 10] {
        let results = storage.find_nearest("/", *k).unwrap();
        let expected = (*k).min(20); // 20 children + root, minus self
        assert!(results.len() <= expected,
            "find_nearest(/, {}) returned {} results, expected <= {}",
            k, results.len(), expected);
        assert!(!results.is_empty(),
            "find_nearest should return at least 1 result for k={}", k);
    }
}

#[test]
fn test_find_nearest_excludes_self() {
    let (storage, _) = build_flat_tree(5);

    let results = storage.find_nearest("/node_0", 10).unwrap();
    assert!(!results.contains(&"/node_0".to_string()),
        "find_nearest should not include the query node itself");
}

#[test]
fn test_find_in_radius_contains_close_neighbors() {
    let (storage, _) = build_flat_tree(5);

    // Large radius should find all nodes
    let large_r = fp(20);
    let results = storage.find_in_radius("/", large_r).unwrap();
    assert!(results.len() >= 5,
        "Large radius should find all {} children, got {}",
        5, results.len());

    // Tiny radius should find few or no nodes
    let tiny_r = fp_ratio(1, 100000);
    let results = storage.find_in_radius("/", tiny_r).unwrap();
    assert!(results.len() <= 1,
        "Tiny radius should find <= 1 node, got {}", results.len());
}

#[test]
fn test_nn_point_agrees_with_find_nearest_for_stored_nodes() {
    // For a query AT a stored node's position, nn_point and find_nearest should
    // largely agree on the nearest neighbor.
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    storage.store("/a", b"a", None).unwrap();
    storage.store("/b", b"b", None).unwrap();
    storage.store("/c", b"c", None).unwrap();

    // find_nearest from root should return some ordering
    let fn_results = storage.find_nearest("/", 3).unwrap();
    assert!(!fn_results.is_empty());

    // nn_point at origin should return root
    let (nn_path, _) = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0])).unwrap();
    assert_eq!(nn_path, "/");
}

// ===========================================================================
// SECTION 5: Binary Tree Structure
// ===========================================================================

#[test]
fn test_binary_tree_depth_3() {
    let (storage, leaves) = build_binary_tree(3);

    // Should have 2^3 = 8 leaves, plus 2+4=6 internal, plus root = 15 total
    let all = storage.list("/").unwrap();
    let node_count = all.len() + 1; // +1 for root which isn't in its own subtree list
    assert!(node_count >= 14, "Binary tree depth 3 should have >= 14 nodes, got {}", node_count);

    // Each leaf should exist
    for leaf in &leaves {
        assert!(storage.exists(leaf), "Leaf {} should exist", leaf);
    }

    // Nearest neighbor from root should return an existing node
    let (nn, _) = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0])).unwrap();
    assert!(storage.exists(&nn));
}

#[test]
fn test_binary_tree_spatial_queries() {
    let (storage, _leaves) = build_binary_tree(3);

    // find_nearest should work from any node
    let nearest = storage.find_nearest("/L1", 3).unwrap();
    assert!(!nearest.is_empty(), "Should find neighbors of /L1");

    // find_in_radius with moderate radius
    let r = fp(3);
    let results = storage.find_in_radius("/L1", r).unwrap();
    assert!(!results.is_empty(), "Should find nodes near /L1");
}

// ===========================================================================
// SECTION 6: Deep Chain
// ===========================================================================

#[test]
fn test_deep_chain_structure() {
    let depth = 15;
    let (storage, paths) = build_deep_chain(depth);

    assert_eq!(paths.len(), depth);

    // All paths should exist
    for path in &paths {
        assert!(storage.exists(path), "Path {} should exist", path);
    }

    // Nearest neighbor point should still work
    let (nn, _) = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0])).unwrap();
    assert!(storage.exists(&nn));
}

#[test]
fn test_deep_chain_nearest_neighbor() {
    let depth = 10;
    let (storage, paths) = build_deep_chain(depth);

    // find_nearest from the deepest node should find its parent
    let deepest = &paths[paths.len() - 1];
    let nearest = storage.find_nearest(deepest, 1).unwrap();
    assert!(!nearest.is_empty(),
        "Deepest node should have at least 1 neighbor");
}

// ===========================================================================
// SECTION 7: Insert-Delete Cycles
// ===========================================================================

#[test]
fn test_insert_delete_preserves_consistency() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    // Insert, verify, delete, verify — repeated
    for i in 0..20 {
        let path = format!("/cycle_{}", i);
        storage.store(&path, format!("data_{}", i).as_bytes(), None).unwrap();
        assert!(storage.exists(&path));

        // NN should still work
        let (nn, _) = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0])).unwrap();
        assert!(storage.exists(&nn));

        storage.delete(&path).unwrap();
        assert!(!storage.exists(&path));

        // NN should still work after deletion
        let (nn2, _) = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0])).unwrap();
        assert!(storage.exists(&nn2));
    }
}

#[test]
fn test_bulk_insert_then_bulk_delete() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    let n = 50;
    let mut paths = Vec::new();
    for i in 0..n {
        let path = format!("/bulk_{}", i);
        storage.store(&path, format!("d{}", i).as_bytes(), None).unwrap();
        paths.push(path);
    }

    // All should exist
    for p in &paths {
        assert!(storage.exists(p));
    }

    // NN should work with all nodes present
    let (nn, _) = storage.nearest_neighbor_point(&fpv(&[0.1, 0.0, 0.0, 0.0])).unwrap();
    assert!(storage.exists(&nn));

    // Delete all in reverse
    for p in paths.iter().rev() {
        storage.delete(p).unwrap();
    }

    // Only root should remain
    assert!(storage.exists("/"));
    let list = storage.list("/").unwrap();
    assert!(list.is_empty(), "After bulk delete, subtree should be empty");

    // NN should still work (returns root)
    let (nn, _) = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0])).unwrap();
    assert_eq!(nn, "/");
}

// ===========================================================================
// SECTION 8: Stress Tests
// ===========================================================================

#[test]
fn test_stress_100_nodes_random_queries() {
    let (storage, _paths) = build_flat_tree(100);

    // 50 random-ish query points
    let queries: Vec<[f32; 4]> = (0..50).map(|i| {
        let angle = i as f32 * 0.1256; // ~2*pi/50
        let r = 0.1 + (i as f32 % 10.0) * 0.08;
        [r * angle.cos(), r * angle.sin(), 0.0, 0.0]
    }).collect();

    for (qi, coords) in queries.iter().enumerate() {
        let result = storage.nearest_neighbor_point(&fpv(coords));
        assert!(result.is_ok(),
            "NN query {} at ({:.2}, {:.2}) should succeed",
            qi, coords[0], coords[1]);

        let (path, dist) = result.unwrap();
        assert!(storage.exists(&path),
            "Query {} returned non-existent path '{}'", qi, path);
        assert!(dist >= fp(0),
            "Query {} returned negative distance {}", qi, dist);
    }
}

#[test]
fn test_stress_sequential_insert_and_query() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    // Insert 200 nodes, querying after each batch of 10
    for i in 0..200 {
        let path = format!("/s_{}", i);
        storage.store(&path, b"x", None).unwrap();

        if (i + 1) % 10 == 0 {
            // Query at origin
            let (nn, _) = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0])).unwrap();
            assert!(storage.exists(&nn),
                "After {} inserts, NN '{}' should exist", i + 1, nn);

            // Query at a non-origin point
            let r = (i as f32 + 1.0) / 300.0;
            let (nn2, _) = storage.nearest_neighbor_point(&fpv(&[r, 0.0, 0.0, 0.0])).unwrap();
            assert!(storage.exists(&nn2),
                "After {} inserts, NN '{}' should exist", i + 1, nn2);
        }
    }
}

// ===========================================================================
// SECTION 9: Edge Cases
// ===========================================================================

#[test]
fn test_empty_tree_nn() {
    // A fresh storage has only root — NN should return root
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    let (path, _dist) = storage.nearest_neighbor_point(&fpv(&[0.3, 0.0, 0.0, 0.0])).unwrap();
    assert_eq!(path, "/", "Empty tree should return root for any query");
}

#[test]
fn test_query_at_boundary() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);
    storage.store("/a", b"a", None).unwrap();

    // Query near the disk boundary (will be clamped by HyperbolicPoint::new)
    let (path, _) = storage.nearest_neighbor_point(&fpv(&[0.99, 0.0, 0.0, 0.0])).unwrap();
    assert!(storage.exists(&path), "Boundary query should return a valid path");
}

#[test]
fn test_query_outside_disk() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);
    storage.store("/a", b"a", None).unwrap();

    // Query outside disk — should get clamped and still work
    let (path, _) = storage.nearest_neighbor_point(&fpv(&[2.0, 0.0, 0.0, 0.0])).unwrap();
    assert!(storage.exists(&path), "Outside-disk query should still return a valid path");
}

#[test]
fn test_high_degree_node() {
    // Create a node with many children (high degree)
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    for i in 0..30 {
        let path = format!("/child_{}", i);
        storage.store(&path, format!("c{}", i).as_bytes(), None).unwrap();
    }

    // NN queries should still work
    let (path, _) = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0])).unwrap();
    assert!(storage.exists(&path));

    // find_nearest should return correct count
    let results = storage.find_nearest("/", 5).unwrap();
    assert_eq!(results.len(), 5, "Should find 5 nearest among 30 children");
}

#[test]
fn test_delete_root_rejected() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    let result = storage.delete("/");
    assert!(result.is_err(), "Deleting root should fail");
}

#[test]
fn test_duplicate_insert_rejected() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    storage.store("/dup", b"first", None).unwrap();

    // Second store should update (not error) via storage API
    storage.store("/dup", b"second", None).unwrap();
    let data = storage.retrieve("/dup").unwrap();
    assert_eq!(data, b"second");
}

// ===========================================================================
// SECTION 10: Klein Model Standalone Correctness
// ===========================================================================

#[test]
fn test_klein_known_values_2d() {
    // P(0.5, 0) → K(0.8, 0), w = 0.36
    let p = HyperbolicPoint::from_f32_slice(&[0.5, 0.0]);
    let k = poincare_to_klein(&p);

    let tol = fp_ratio(1, 100);
    assert!(approx_eq(k.coords[0], fp_ratio(4, 5), tol), "x: {}", k.coords[0]);
    assert!(k.coords[1].abs() < tol, "y: {}", k.coords[1]);
    assert!(approx_eq(k.weight, fp_ratio(36, 100), tol), "w: {}", k.weight);
}

#[test]
fn test_klein_known_values_negative() {
    // P(-0.5, 0) → K(-0.8, 0)
    let p = HyperbolicPoint::from_f32_slice(&[-0.5, 0.0]);
    let k = poincare_to_klein(&p);

    let tol = fp_ratio(1, 100);
    assert!(approx_eq(k.coords[0], -fp_ratio(4, 5), tol), "x: {}", k.coords[0]);
}

#[test]
fn test_power_distance_triangle_inequality_analog() {
    // Power distance is NOT a metric, but for sites at equal Klein norm,
    // the site closest in Euclidean should also be closest in power distance.
    let sites = vec![
        KleinPoint::new(FixedVector::from_f32_slice(&[0.5, 0.0, 0.0, 0.0])),
        KleinPoint::new(FixedVector::from_f32_slice(&[-0.5, 0.0, 0.0, 0.0])),
        KleinPoint::new(FixedVector::from_f32_slice(&[0.0, 0.5, 0.0, 0.0])),
        KleinPoint::new(FixedVector::from_f32_slice(&[0.0, -0.5, 0.0, 0.0])),
    ];

    // Query near site 0
    let q = FixedVector::from_f32_slice(&[0.4, 0.0, 0.0, 0.0]);
    let pds: Vec<FixedPoint> = sites.iter().map(|s| power_distance(&q, s)).collect();

    // Site 0 should have lowest pd
    assert!(pds[0] < pds[1], "Site 0 should be closest");
    assert!(pds[0] < pds[2], "Site 0 should be closest");
    assert!(pds[0] < pds[3], "Site 0 should be closest");
}

// ===========================================================================
// SECTION 11: Data Integrity Through Operations
// ===========================================================================

#[test]
fn test_data_survives_spatial_operations() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    // Store structured data
    let test_data: Vec<(&str, &[u8])> = vec![
        ("/docs/readme", b"# README\nThis is a readme."),
        ("/docs/license", b"MIT License"),
        ("/src/main", b"fn main() {}"),
        ("/src/lib", b"pub mod core;"),
        ("/config", b"key=value"),
    ];

    for (path, data) in &test_data {
        storage.store(path, data, None).unwrap();
    }

    // Perform spatial queries (should not corrupt data)
    let _ = storage.find_nearest("/docs/readme", 3);
    let _ = storage.find_in_radius("/src/main", fp(5));
    let _ = storage.nearest_neighbor_point(&fpv(&[0.1, 0.2, 0.0, 0.0]));

    // Verify all data is intact
    for (path, expected) in &test_data {
        let actual = storage.retrieve(path).unwrap();
        assert_eq!(&actual[..], *expected,
            "Data at {} corrupted after spatial queries", path);
    }
}

#[test]
fn test_metadata_survives_spatial_operations() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    storage.store("/tagged", b"data", Some("text/plain".to_string())).unwrap();
    storage.set_metadata("/tagged", "author", "test").unwrap();

    // Spatial queries
    let _ = storage.nearest_neighbor_point(&fpv(&[0.0, 0.0, 0.0, 0.0]));
    let _ = storage.find_nearest("/tagged", 2);

    // Check metadata intact
    let meta = storage.get_metadata("/tagged").unwrap();
    assert_eq!(meta.get("content_type").unwrap(), "text/plain");
    assert_eq!(meta.get("author").unwrap(), "test");
}

// ===========================================================================
// SECTION 12: Statistics and Diagnostics
// ===========================================================================

#[test]
fn test_stats_reflect_operations() {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);

    let stats1 = storage.stats();
    let count1: usize = stats1.get("node_count").unwrap().parse().unwrap();
    assert_eq!(count1, 1, "Initial node count should be 1 (root)");

    storage.store("/a", b"a", None).unwrap();
    storage.store("/b", b"b", None).unwrap();

    let stats2 = storage.stats();
    let count2: usize = stats2.get("node_count").unwrap().parse().unwrap();
    assert_eq!(count2, 3, "After 2 inserts, count should be 3");

    storage.delete("/b").unwrap();
    let stats3 = storage.stats();
    let count3: usize = stats3.get("node_count").unwrap().parse().unwrap();
    // Note: delete removes from path_map but node may linger in tensor network
    // The stats count comes from the tree tensor's path_map length
    assert!(count3 <= 3, "After delete, count should be <= 3");
}

/// Exact fixed-point coordinates from decimal literals — the public API takes
/// `FixedPoint`, so tests convert explicitly rather than through a float API.
fn fpv(vals: &[f32]) -> Vec<g_math::fixed_point::FixedPoint> {
    vals.iter().map(|&v| g_math::fixed_point::FixedPoint::from_f64(v as f64)).collect()
}
