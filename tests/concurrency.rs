//! Concurrency correctness tests — the CI-gated version of the invariants
//! demonstrated in examples/concurrency_bench.rs and
//! examples/concurrency_contention.rs.
//!
//! These test *correctness under parallelism*, not throughput:
//! zero lost children, zero lost writes, no deadlocks, and spatial/semantic
//! queries staying coherent while writers run.
//!
//! Run: GMATH_PROFILE=embedded cargo test --test concurrency

use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

use horon_engine::Store;

const THREADS: usize = 8;

/// N threads writing to independent subtrees: every write must land.
#[test]
fn parallel_writes_independent_subtrees_lose_nothing() {
    let store = Arc::new(Store::new());
    let per_thread = 100;

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                for i in 0..per_thread {
                    store
                        .put(&format!("/t{}/item_{}", tid, i), format!("v{}_{}", tid, i).as_bytes())
                        .unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    for tid in 0..THREADS {
        let kids = store.children(&format!("/t{}", tid)).unwrap();
        assert_eq!(
            kids.len(),
            per_thread,
            "thread {} lost children: expected {}, found {}",
            tid,
            per_thread,
            kids.len()
        );
        for i in 0..per_thread {
            let data = store.get(&format!("/t{}/item_{}", tid, i)).unwrap();
            assert_eq!(data, format!("v{}_{}", tid, i).as_bytes());
        }
    }
}

/// N threads writing children under the SAME parent: the striped parent
/// lock must serialize sibling registration without losing any child.
#[test]
fn parallel_writes_same_parent_lose_nothing() {
    let store = Arc::new(Store::new());
    let per_thread = 50;

    let handles: Vec<_> = (0..THREADS)
        .map(|tid| {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                for i in 0..per_thread {
                    store
                        .put(&format!("/shared/t{}_i{}", tid, i), b"x")
                        .unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    let kids: HashSet<String> = store.children("/shared").unwrap().into_iter().collect();
    assert_eq!(
        kids.len(),
        THREADS * per_thread,
        "lost children under contended parent"
    );
}

/// Readers and writers running simultaneously: reads never observe
/// torn state and writes all land.
#[test]
fn mixed_readers_and_writers_stay_coherent() {
    let store = Arc::new(Store::new());
    for i in 0..100 {
        store.put(&format!("/warm/item_{}", i), b"warm").unwrap();
    }

    let writers: Vec<_> = (0..THREADS / 2)
        .map(|tid| {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                for i in 0..50 {
                    store.put(&format!("/hot/t{}_i{}", tid, i), b"new").unwrap();
                }
            })
        })
        .collect();
    let readers: Vec<_> = (0..THREADS / 2)
        .map(|_| {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                for i in 0..500 {
                    let data = store.get(&format!("/warm/item_{}", i % 100)).unwrap();
                    assert_eq!(data, b"warm");
                }
            })
        })
        .collect();

    for h in writers.into_iter().chain(readers) {
        h.join().unwrap();
    }
    assert_eq!(store.children("/hot").unwrap().len(), (THREADS / 2) * 50);
}

/// Spatial queries running concurrently with inserts must not panic or
/// deadlock, and must only ever return keys that exist.
#[test]
fn spatial_queries_during_writes_return_valid_keys() {
    let store = Arc::new(Store::new());
    for i in 0..50 {
        store.put(&format!("/base/item_{}", i), b"seed").unwrap();
    }

    let writer = {
        let store = Arc::clone(&store);
        thread::spawn(move || {
            for i in 0..100 {
                store.put(&format!("/grow/item_{}", i), b"grown").unwrap();
            }
        })
    };
    let queriers: Vec<_> = (0..4)
        .map(|_| {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                for i in 0..100 {
                    let neighbors = store
                        .neighbors(&format!("/base/item_{}", i % 50), 3)
                        .unwrap();
                    for key in neighbors {
                        assert!(store.exists(&key), "query returned ghost key {}", key);
                    }
                }
            })
        })
        .collect();

    writer.join().unwrap();
    for h in queriers {
        h.join().unwrap();
    }
}

/// Concurrent removes and puts on disjoint keys: removed keys are gone,
/// inserted keys are present.
#[test]
fn concurrent_put_and_remove_disjoint_keys() {
    let store = Arc::new(Store::new());
    for i in 0..200 {
        store.put(&format!("/churn/old_{}", i), b"old").unwrap();
    }

    let remover = {
        let store = Arc::clone(&store);
        thread::spawn(move || {
            for i in 0..200 {
                store.remove(&format!("/churn/old_{}", i)).unwrap();
            }
        })
    };
    let inserter = {
        let store = Arc::clone(&store);
        thread::spawn(move || {
            for i in 0..200 {
                store.put(&format!("/churn/new_{}", i), b"new").unwrap();
            }
        })
    };
    remover.join().unwrap();
    inserter.join().unwrap();

    for i in 0..200 {
        assert!(!store.exists(&format!("/churn/old_{}", i)));
        assert!(store.exists(&format!("/churn/new_{}", i)));
    }
}

/// Semantic writes and semantic queries in parallel: results only contain
/// live keys and distances are finite.
#[test]
fn semantic_queries_during_semantic_writes() {
    let store = Arc::new(Store::new());
    let encode = |vals: &[f64]| -> Vec<u8> {
        use g_math::fixed_point::FixedPoint;
        vals.iter()
            .flat_map(|v| FixedPoint::from_f64(*v).raw().to_le_bytes())
            .collect()
    };

    for i in 0..50 {
        let key = format!("/sem/item_{}", i);
        store.put(&key, b"s").unwrap();
        store.set_semantic(&key, encode(&[i as f64 * 0.01, 1.0 - i as f64 * 0.01])).unwrap();
    }

    let writer = {
        let store = Arc::clone(&store);
        let encode = encode;
        thread::spawn(move || {
            for i in 50..150 {
                let key = format!("/sem/item_{}", i);
                store.put(&key, b"s").unwrap();
                store
                    .set_semantic(&key, encode(&[i as f64 * 0.01, 0.5]))
                    .unwrap();
            }
        })
    };
    let queriers: Vec<_> = (0..4)
        .map(|_| {
            let store = Arc::clone(&store);
            let query = encode(&[0.25, 0.75]);
            thread::spawn(move || {
                for _ in 0..50 {
                    let results = store.nearest_semantic(&query, 5, 0..2).unwrap();
                    for (key, dist) in results {
                        assert!(dist.to_f64().is_finite(), "non-finite semantic distance");
                        assert!(key.starts_with("/sem/"), "unexpected key {}", key);
                    }
                }
            })
        })
        .collect();

    writer.join().unwrap();
    for h in queriers {
        h.join().unwrap();
    }
}
