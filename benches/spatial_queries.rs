//! =============================================================================
//! Engine Benchmarks: spatial queries and the Klein model
//! =============================================================================
//!
//! Measures:
//!   - Klein model conversions (Poincaré ↔ Klein)
//!   - Power distance computation
//!   - Nearest neighbour queries through the full storage API
//!   - Insert throughput at various tree sizes
//!   - Insert+delete churn
//!
//! Run with: GMATH_PROFILE=embedded cargo bench --bench spatial_queries

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use g_math::fixed_point::{FixedPoint, FixedVector};
use horon_engine::{
    HTTStorage, HTTStorageConfig,
    HyperbolicPoint,
    poincare_to_klein, klein_to_poincare, power_distance, KleinPoint,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a flat tree with N children under root.
fn build_storage(n: usize) -> HTTStorage {
    let config = HTTStorageConfig::default();
    let storage = HTTStorage::new(config);
    for i in 0..n {
        let path = format!("/node_{}", i);
        storage.store(&path, format!("d{}", i).as_bytes(), None).unwrap();
    }
    storage
}

/// Generate N test points inside the Poincaré disk at various radii and angles.
///
/// Fixed point end to end. The geometry under measurement is fixed point, so
/// the fixture that drives it is built the same way: f32 trig here would feed
/// the benchmark coordinates the engine itself would never produce.
fn generate_query_points(n: usize, dim: usize) -> Vec<Vec<FixedPoint>> {
    let two_pi = FixedPoint::from_f64(std::f64::consts::TAU);
    let n_fp = FixedPoint::from_int(n as i32);
    let base_r = FixedPoint::from_f64(0.1);
    let step_r = FixedPoint::from_f64(0.09);
    (0..n).map(|i| {
        let angle = FixedPoint::from_int(i as i32) * two_pi / n_fp;
        let r = base_r + FixedPoint::from_int((i % 9) as i32) * step_r; // 0.1 to 0.91
        let (sin, cos) = angle.sincos();
        let mut coords = vec![FixedPoint::ZERO; dim];
        coords[0] = r * cos;
        if dim >= 2 { coords[1] = r * sin; }
        coords
    }).collect()
}

// ===========================================================================
// BENCHMARK GROUP 1: Klein Model Conversions
// ===========================================================================

fn bench_poincare_to_klein(c: &mut Criterion) {
    let mut group = c.benchmark_group("klein_conversion");

    // 2D conversion
    let p2d = HyperbolicPoint::from_f32_slice(&[0.5, 0.3]);
    group.bench_function("poincare_to_klein_2d", |b| {
        b.iter(|| poincare_to_klein(black_box(&p2d)))
    });

    // 4D conversion (default HTT dimension)
    let p4d = HyperbolicPoint::from_f32_slice(&[0.3, 0.2, -0.1, 0.15]);
    group.bench_function("poincare_to_klein_4d", |b| {
        b.iter(|| poincare_to_klein(black_box(&p4d)))
    });

    // Roundtrip 4D
    group.bench_function("roundtrip_4d", |b| {
        b.iter(|| {
            let k = poincare_to_klein(black_box(&p4d));
            klein_to_poincare(black_box(&k))
        })
    });

    // Batch of 100 conversions
    let points: Vec<HyperbolicPoint> = (0..100).map(|i| {
        let angle = i as f32 * 0.0628; // ~2π/100
        let r = 0.1 + (i as f32 % 9.0) * 0.09;
        HyperbolicPoint::from_f32_slice(&[r * angle.cos(), r * angle.sin(), 0.0, 0.0])
    }).collect();

    group.bench_function("poincare_to_klein_batch_100", |b| {
        b.iter(|| {
            for p in &points {
                black_box(poincare_to_klein(p));
            }
        })
    });

    group.finish();
}

// ===========================================================================
// BENCHMARK GROUP 2: Power Distance
// ===========================================================================

fn bench_power_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("power_distance");

    let site = poincare_to_klein(&HyperbolicPoint::from_f32_slice(&[0.5, 0.3, 0.0, 0.0]));
    let query = FixedVector::from_f32_slice(&[0.2, 0.1, 0.0, 0.0]);

    // Single power distance computation
    group.bench_function("single_4d", |b| {
        b.iter(|| power_distance(black_box(&query), black_box(&site)))
    });

    // Power distance against 10 sites (brute-force NN)
    let sites: Vec<KleinPoint> = (0..10).map(|i| {
        let angle = i as f32 * 0.628;
        let r = 0.3 + (i as f32 % 5.0) * 0.1;
        poincare_to_klein(&HyperbolicPoint::from_f32_slice(&[
            r * angle.cos(), r * angle.sin(), 0.0, 0.0
        ]))
    }).collect();

    group.bench_function("brute_force_nn_10_sites", |b| {
        b.iter(|| {
            let mut best_pd = power_distance(&query, &sites[0]);
            for site in sites.iter().skip(1) {
                let pd = power_distance(black_box(&query), site);
                if pd < best_pd { best_pd = pd; }
            }
            black_box(best_pd)
        })
    });

    // Power distance against 100 sites
    let sites_100: Vec<KleinPoint> = (0..100).map(|i| {
        let angle = i as f32 * 0.0628;
        let r = 0.1 + (i as f32 % 9.0) * 0.09;
        poincare_to_klein(&HyperbolicPoint::from_f32_slice(&[
            r * angle.cos(), r * angle.sin(), 0.0, 0.0
        ]))
    }).collect();

    group.bench_function("brute_force_nn_100_sites", |b| {
        b.iter(|| {
            let mut best_pd = power_distance(&query, &sites_100[0]);
            for site in sites_100.iter().skip(1) {
                let pd = power_distance(black_box(&query), site);
                if pd < best_pd { best_pd = pd; }
            }
            black_box(best_pd)
        })
    });

    group.finish();
}

// ===========================================================================
// BENCHMARK GROUP 4: Storage API — Nearest Neighbor Point
// ===========================================================================

fn bench_nearest_neighbor_api(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_neighbor_api");
    group.sample_size(20); // fewer samples for slower benchmarks

    // Vary tree size
    for &n in &[10, 50, 100, 200] {
        let storage = build_storage(n);
        let queries = generate_query_points(10, 4);

        group.bench_with_input(
            BenchmarkId::new("nn_point", n),
            &n,
            |b, _| {
                b.iter(|| {
                    for q in &queries {
                        black_box(storage.nearest_neighbor_point(q).unwrap());
                    }
                })
            },
        );
    }

    // NN at the origin, where the root sits: the query is exactly on a node
    {
        let storage = build_storage(50);
        group.bench_function("nn_point_at_origin_50nodes", |b| {
            b.iter(|| {
                black_box(storage.nearest_neighbor_point(&[FixedPoint::ZERO; 4]).unwrap());
            })
        });
    }

    group.finish();
}

// ===========================================================================
// BENCHMARK GROUP 5: find_nearest (VP-tree path)
// ===========================================================================

fn bench_find_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_nearest_vptree");
    group.sample_size(20);

    for &n in &[10, 50, 100, 200] {
        let storage = build_storage(n);

        group.bench_with_input(
            BenchmarkId::new("k1", n),
            &n,
            |b, _| {
                b.iter(|| {
                    black_box(storage.find_nearest("/", 1).unwrap());
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("k5", n),
            &n,
            |b, _| {
                b.iter(|| {
                    black_box(storage.find_nearest("/", 5).unwrap());
                })
            },
        );
    }

    group.finish();
}

// ===========================================================================
// BENCHMARK GROUP 6: Insert Throughput
// ===========================================================================

fn bench_insert_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput");
    group.sample_size(10);

    // Measure time to insert N nodes into an empty tree
    for &n in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("flat_tree", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    let config = HTTStorageConfig::default();
                    let storage = HTTStorage::new(config);
                    for i in 0..n {
                        let path = format!("/node_{}", i);
                        storage.store(&path, b"x", None).unwrap();
                    }
                    black_box(&storage);
                })
            },
        );
    }

    // Measure per-node insert cost at different tree sizes (marginal cost)
    for &base_size in &[10, 50, 100] {
        let config = HTTStorageConfig::default();
        let storage = HTTStorage::new(config);
        for i in 0..base_size {
            storage.store(&format!("/pre_{}", i), b"x", None).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("marginal_insert_at", base_size),
            &base_size,
            |b, _| {
                let mut idx = base_size;
                b.iter(|| {
                    let path = format!("/marginal_{}", idx);
                    storage.store(&path, b"x", None).unwrap();
                    idx += 1;
                    black_box(&storage);
                })
            },
        );
    }

    group.finish();
}

// ===========================================================================
// BENCHMARK GROUP 7: Delete Throughput
// ===========================================================================

fn bench_delete_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("delete_throughput");
    group.sample_size(10);

    for &n in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("delete_all", n),
            &n,
            |b, &n| {
                b.iter_batched(
                    || {
                        // Setup: build tree
                        let config = HTTStorageConfig::default();
                        let storage = HTTStorage::new(config);
                        let mut paths = Vec::new();
                        for i in 0..n {
                            let path = format!("/node_{}", i);
                            storage.store(&path, b"x", None).unwrap();
                            paths.push(path);
                        }
                        (storage, paths)
                    },
                    |(storage, paths)| {
                        // Measured: delete all
                        for p in &paths {
                            storage.delete(p).unwrap();
                        }
                        black_box(&storage);
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

// ===========================================================================
// BENCHMARK GROUP 8: Hyperbolic Distance (baseline for comparison)
// ===========================================================================

fn bench_hyperbolic_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperbolic_distance_baseline");

    let p1 = HyperbolicPoint::from_f32_slice(&[0.3, 0.2, -0.1, 0.15]);
    let p2 = HyperbolicPoint::from_f32_slice(&[0.5, -0.1, 0.2, 0.0]);

    group.bench_function("single_4d", |b| {
        b.iter(|| black_box(p1.hyperbolic_distance(black_box(&p2))))
    });

    // Batch 100 distance computations
    let points: Vec<HyperbolicPoint> = (0..100).map(|i| {
        let angle = i as f32 * 0.0628;
        let r = 0.1 + (i as f32 % 9.0) * 0.09;
        HyperbolicPoint::from_f32_slice(&[r * angle.cos(), r * angle.sin(), 0.0, 0.0])
    }).collect();

    group.bench_function("batch_100_distances", |b| {
        b.iter(|| {
            for p in &points {
                black_box(p1.hyperbolic_distance(black_box(p)));
            }
        })
    });

    group.finish();
}

// ===========================================================================
// BENCHMARK GROUP 9: End-to-End Store+Query Cycle
// ===========================================================================

fn bench_store_then_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_then_query");
    group.sample_size(10);

    // Simulate a realistic workload: store 50 items, then issue 20 mixed queries
    group.bench_function("50_store_20_query", |b| {
        b.iter(|| {
            let config = HTTStorageConfig::default();
            let storage = HTTStorage::new(config);

            // Store phase
            for i in 0..50 {
                let path = format!("/item_{}", i);
                storage.store(&path, format!("value_{}", i).as_bytes(), None).unwrap();
            }

            // Query phase: mix of NN point queries and find_nearest
            for i in 0..10 {
                let r = FixedPoint::from_int(i + 1) / FixedPoint::from_int(15);
                let angle = FixedPoint::from_int(i) * FixedPoint::from_f64(0.628);
                let (sin, cos) = angle.sincos();
                black_box(storage.nearest_neighbor_point(&[
                    r * cos, r * sin, FixedPoint::ZERO, FixedPoint::ZERO
                ]).unwrap());
            }
            for i in 0..10 {
                let path = format!("/item_{}", i * 5);
                black_box(storage.find_nearest(&path, 3).unwrap());
            }

            black_box(&storage);
        })
    });

    group.finish();
}

// ===========================================================================
// Register all benchmark groups
// ===========================================================================

criterion_group!(
    benches,
    bench_poincare_to_klein,
    bench_power_distance,
    bench_nearest_neighbor_api,
    bench_find_nearest,
    bench_insert_throughput,
    bench_delete_throughput,
    bench_hyperbolic_distance,
    bench_store_then_query,
);
criterion_main!(benches);
