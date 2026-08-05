//! =============================================================================
//! Engine Benchmarks: Semantic Queries (scan baseline → semantic index)
//! =============================================================================
//!
//! Measures the semantic query path at 10k / 50k / 100k nodes:
//!   - nearest_semantic (Store API — routes through the lazy per-slice
//!     VP-tree above the node floor; criterion's iterations reuse the store,
//!     so these numbers are WARM-index queries, directly comparable to the
//!     pre-index scan numbers in BENCHMARKS.md)
//!   - scan_vs_indexed (network level — the two paths side by side)
//!   - index_rebuild (the honest cost of the first query after a semantic
//!     write: epoch bump → full O(n log n) rebuild → query)
//!   - semantic_distance (single-pair primitive baseline)
//!
//! Run with: GMATH_PROFILE=embedded cargo bench --bench semantic

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use horon_engine::tensor_network::NodeMetadata;
use horon_engine::{HyperbolicTensorNetwork, Store};
use g_math::fixed_point::FixedPoint;

const DIMS: usize = 8;

/// Encode f64 values as raw Q64.64 little-endian bytes (16 bytes per dim).
fn coords(vals: &[f64]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| FixedPoint::from_f64(*v).raw().to_le_bytes())
        .collect()
}

/// Deterministic pseudo-random value in [0, 1) from an index pair.
fn val(i: usize, d: usize) -> f64 {
    let x = (i.wrapping_mul(2654435761) ^ d.wrapping_mul(40503)) as u32;
    (x as f64) / (u32::MAX as f64)
}

/// Build a store with `n` data-only nodes, each carrying DIMS semantic dims.
fn build_semantic_store(n: usize) -> Store {
    let store = Store::new();
    for i in 0..n {
        let key = format!("/n{}", i);
        store.put_data_only(&key, b"x").expect("put");
        let vals: Vec<f64> = (0..DIMS).map(|d| val(i, d)).collect();
        store.set_semantic(&key, coords(&vals)).expect("semantic");
    }
    store
}

fn bench_nearest_semantic(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_semantic");
    group.sample_size(10);

    for &n in &[10_000usize, 50_000, 100_000] {
        let store = build_semantic_store(n);
        let query = coords(&(0..DIMS).map(|d| val(n / 2, d)).collect::<Vec<_>>());

        // Full dimension slice, top-10.
        group.bench_with_input(BenchmarkId::new("k10_d8", n), &n, |b, _| {
            b.iter(|| black_box(store.nearest_semantic(&query, 10, 0..DIMS).unwrap()))
        });

        // Narrow 2-dim slice, top-10 (d matters: O(n × d)).
        group.bench_with_input(BenchmarkId::new("k10_d2", n), &n, |b, _| {
            b.iter(|| black_box(store.nearest_semantic(&query, 10, 0..2).unwrap()))
        });
    }

    group.finish();
}

/// Build a bare network with `n` data-only nodes (network-level benches).
fn build_semantic_network(n: usize) -> HyperbolicTensorNetwork {
    let network = HyperbolicTensorNetwork::new(4, FixedPoint::from_int(1));
    for i in 0..n {
        let key = format!("/n{}", i);
        let uid = network.add_node_data_only(NodeMetadata::new(key, None), b"x".to_vec(), 1);
        let vals: Vec<f64> = (0..DIMS).map(|d| val(i, d)).collect();
        network.set_node_semantic(&uid, coords(&vals));
    }
    network
}

/// The two query paths side by side on identical data (10k nodes, k=10, d=8).
fn bench_scan_vs_indexed(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan_vs_indexed_10k");
    group.sample_size(10);

    let n = 10_000usize;
    let network = build_semantic_network(n);
    let query = coords(&(0..DIMS).map(|d| val(n / 2, d)).collect::<Vec<_>>());
    let range = 0..DIMS;

    group.bench_function("scan_k10_d8", |b| {
        b.iter(|| black_box(network.nearest_semantic_scan(&query, 10, &range)))
    });
    // Warm the index once so the measured iterations are pure lookups.
    let _ = network.nearest_semantic(&query, 10, &range);
    group.bench_function("indexed_warm_k10_d8", |b| {
        b.iter(|| black_box(network.nearest_semantic(&query, 10, &range)))
    });

    group.finish();
}

/// First-query-after-write cost: every iteration bumps the semantic epoch
/// (a real set_semantic) and pays the full rebuild plus the query.
fn bench_index_rebuild(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_rebuild_10k");
    group.sample_size(10);

    let n = 10_000usize;
    let network = build_semantic_network(n);
    let query = coords(&(0..DIMS).map(|d| val(n / 2, d)).collect::<Vec<_>>());
    let range = 0..DIMS;
    let uid = network.add_node_data_only(NodeMetadata::new("/churn".into(), None), b"x".to_vec(), 1);
    let churn = coords(&(0..DIMS).map(|d| val(7, d)).collect::<Vec<_>>());

    group.bench_function("write_then_query_k10_d8", |b| {
        b.iter(|| {
            network.set_node_semantic(&uid, churn.clone()); // epoch bump
            black_box(network.nearest_semantic(&query, 10, &range))
        })
    });

    group.finish();
}

/// Semantic-disk queries at 10k nodes: warm hyperbolic k-NN over
/// derived positions, and constant-time concept classification.
fn bench_semantic_disk(c: &mut Criterion) {
    use horon_engine::SemanticDisk;

    let mut group = c.benchmark_group("semantic_disk_10k");
    group.sample_size(10);

    let n = 10_000usize;
    let store = Store::new();
    store.put_data_only("/c", b"parent").expect("put");
    for i in 0..n {
        let key = format!("/c/n{:05}", i);
        store.put_data_only(&key, b"x").expect("put");
        let vals: Vec<f64> = (0..5).map(|d| val(i, d)).collect();
        store.set_semantic(&key, coords(&vals)).expect("semantic");
    }
    let disk = SemanticDisk::build(&[
        ("/a", 0),
        ("/b", 1),
        ("/c", 2),
        ("/d", 3),
        ("/e", 4),
    ])
    .expect("disk");

    // Warm the derived-position index once.
    let _ = disk.nearest(&store, "/c/n05000", 10).expect("warm");

    group.bench_function("nearest_warm_k10", |b| {
        b.iter(|| black_box(disk.nearest(&store, "/c/n05000", 10).unwrap()))
    });
    group.bench_function("concept_of", |b| {
        b.iter(|| black_box(disk.concept_of(&store, "/c/n05000").unwrap()))
    });

    group.finish();
}

fn bench_semantic_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic_distance");

    let a = coords(&(0..DIMS).map(|d| val(1, d)).collect::<Vec<_>>());
    let b_ = coords(&(0..DIMS).map(|d| val(2, d)).collect::<Vec<_>>());

    group.bench_function("single_pair_d8", |b| {
        b.iter(|| black_box(Store::semantic_distance(black_box(&a), black_box(&b_), 0..DIMS)))
    });

    // Proxy diagnostics: where does the hyperbolic pair cost actually live?
    let pa = horon_engine::HyperbolicPoint::from_f32_slice(&[0.31, -0.42]);
    let pb = horon_engine::HyperbolicPoint::from_f32_slice(&[-0.15, 0.27]);
    group.bench_function("hyperbolic_distance_pair", |b| {
        b.iter(|| black_box(black_box(&pa).hyperbolic_distance(black_box(&pb))))
    });
    group.bench_function("hyperbolic_ratio_pair", |b| {
        b.iter(|| black_box(black_box(&pa).hyperbolic_ratio(black_box(&pb))))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_nearest_semantic,
    bench_scan_vs_indexed,
    bench_index_rebuild,
    bench_semantic_disk,
    bench_semantic_distance
);
criterion_main!(benches);
