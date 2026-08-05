//! concurrency_bench.rs - Measure what the concurrent spatial engine unlocked
//!
//! Compares single-threaded vs multi-threaded throughput for:
//!   1. Concurrent reads (many readers, no writers)
//!   2. Concurrent writes to independent subtrees (parallel inserts)
//!   3. Mixed read/write workload (readers + writers simultaneously)
//!   4. Concurrent nearest-neighbor queries
//!
//! Run: GMATH_PROFILE=embedded cargo run --release --example concurrency_bench

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use horon_engine::Store;

fn main() {
    println!("=== Concurrency Benchmark ===\n");
    println!("All methods take &self — no outer RwLock, fine-grained interior mutability.\n");

    let num_cpus = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    println!("Available parallelism: {} threads\n", num_cpus);

    bench_concurrent_reads(num_cpus);
    bench_concurrent_writes(num_cpus);
    bench_concurrent_mixed(num_cpus);
    bench_concurrent_nn_queries(num_cpus);
    bench_concurrent_insert_scaling(num_cpus);

    println!("\n=== Done ===");
}

/// Measure: N threads all reading from a pre-populated store simultaneously.
fn bench_concurrent_reads(num_threads: usize) {
    println!("--- 1. Concurrent Reads ({} threads) ---", num_threads);

    let store = Arc::new(Store::new());

    // Pre-populate with 200 nodes
    let num_nodes = 200;
    for i in 0..num_nodes {
        store.put(&format!("/data/item_{}", i), format!("value_{}", i).as_bytes()).unwrap();
    }

    let ops_per_thread = 5000;

    // Single-threaded baseline
    let start = Instant::now();
    for t in 0..ops_per_thread * num_threads {
        let key = format!("/data/item_{}", t % num_nodes);
        let _ = store.get(&key).unwrap();
    }
    let single = start.elapsed();

    // Multi-threaded
    let start = Instant::now();
    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let key = format!("/data/item_{}", (tid * ops_per_thread + i) % num_nodes);
                    let _ = store.get(&key).unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let multi = start.elapsed();

    let total_ops = ops_per_thread * num_threads;
    report("reads", total_ops, single, multi, num_threads);
}

/// Measure: N threads writing to independent subtrees simultaneously.
fn bench_concurrent_writes(num_threads: usize) {
    println!("\n--- 2. Concurrent Writes to Independent Subtrees ({} threads) ---", num_threads);

    let ops_per_thread = 200;

    // Single-threaded baseline
    let store = Arc::new(Store::new());
    let start = Instant::now();
    for t in 0..num_threads {
        for i in 0..ops_per_thread {
            let key = format!("/tree_{}/node_{}", t, i);
            store.put(&key, b"x").unwrap();
        }
    }
    let single = start.elapsed();

    // Multi-threaded (each thread writes to its own subtree — different parents)
    let store = Arc::new(Store::new());
    let start = Instant::now();
    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let key = format!("/tree_{}/node_{}", tid, i);
                    store.put(&key, b"x").unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let multi = start.elapsed();

    let total_ops = ops_per_thread * num_threads;
    report("writes (independent subtrees)", total_ops, single, multi, num_threads);
}

/// Measure: readers and writers running simultaneously.
fn bench_concurrent_mixed(num_threads: usize) {
    println!("\n--- 3. Mixed Read/Write Workload ({} threads) ---", num_threads);

    let store = Arc::new(Store::new());

    // Pre-populate
    for i in 0..100 {
        store.put(&format!("/pre/item_{}", i), b"seed").unwrap();
    }

    let ops_per_thread = 1000;
    let writers = num_threads / 2;
    let readers = num_threads - writers;

    // Single-threaded baseline: interleaved reads and writes
    let store_st = Arc::new(Store::new());
    for i in 0..100 {
        store_st.put(&format!("/pre/item_{}", i), b"seed").unwrap();
    }
    let start = Instant::now();
    for t in 0..num_threads {
        for i in 0..ops_per_thread {
            if t < writers {
                let key = format!("/new_{}/item_{}", t, i);
                store_st.put(&key, b"w").unwrap();
            } else {
                let key = format!("/pre/item_{}", i % 100);
                let _ = store_st.get(&key).unwrap();
            }
        }
    }
    let single = start.elapsed();

    // Multi-threaded
    let start = Instant::now();
    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                for i in 0..ops_per_thread {
                    if tid < writers {
                        let key = format!("/new_{}/item_{}", tid, i);
                        store.put(&key, b"w").unwrap();
                    } else {
                        let key = format!("/pre/item_{}", i % 100);
                        let _ = store.get(&key).unwrap();
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let multi = start.elapsed();

    let total_ops = ops_per_thread * num_threads;
    report(&format!("mixed ({} writers + {} readers)", writers, readers),
           total_ops, single, multi, num_threads);
}

/// Measure: N threads doing nearest-neighbor queries simultaneously.
fn bench_concurrent_nn_queries(num_threads: usize) {
    println!("\n--- 4. Concurrent Nearest-Neighbor Queries ({} threads) ---", num_threads);

    let store = Arc::new(Store::new());
    for i in 0..100 {
        store.put(&format!("/node_{}", i), format!("d{}", i).as_bytes()).unwrap();
    }

    let ops_per_thread = 200;

    // Single-threaded
    let start = Instant::now();
    for t in 0..num_threads {
        for i in 0..ops_per_thread {
            let r = ((t * ops_per_thread + i) as f32) / (num_threads * ops_per_thread) as f32 * 0.8;
            let angle = (t * ops_per_thread + i) as f32 * 0.1;
            let _ = store.nearest(&fp(&[(r * angle.cos()) as f64, (r * angle.sin()) as f64, 0.0, 0.0])).unwrap();
        }
    }
    let single = start.elapsed();

    // Multi-threaded
    let start = Instant::now();
    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let r = ((tid * ops_per_thread + i) as f32) / (num_threads * ops_per_thread) as f32 * 0.8;
                    let angle = (tid * ops_per_thread + i) as f32 * 0.1;
                    let _ = store.nearest(&fp(&[(r * angle.cos()) as f64, (r * angle.sin()) as f64, 0.0, 0.0])).unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let multi = start.elapsed();

    let total_ops = ops_per_thread * num_threads;
    report("nearest-neighbor queries", total_ops, single, multi, num_threads);
}

/// Measure: how insert throughput scales as we add threads.
fn bench_concurrent_insert_scaling(max_threads: usize) {
    println!("\n--- 5. Insert Scaling: 1 to {} threads ---", max_threads);

    let ops_per_thread = 200;
    let mut results = Vec::new();

    for n in [1, 2, 4, max_threads.min(8), max_threads] {
        if n > max_threads { continue; }
        if results.iter().any(|(t, _, _): &(usize, Duration, f64)| *t == n) { continue; }

        let store = Arc::new(Store::new());
        let start = Instant::now();
        let handles: Vec<_> = (0..n)
            .map(|tid| {
                let store = Arc::clone(&store);
                thread::spawn(move || {
                    for i in 0..ops_per_thread {
                        let key = format!("/t{}/n{}", tid, i);
                        store.put(&key, b"x").unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let elapsed = start.elapsed();
        let total = n * ops_per_thread;
        let throughput = total as f64 / elapsed.as_secs_f64();
        results.push((n, elapsed, throughput));
    }

    let baseline = results[0].2;
    println!("  {:>7} | {:>12} | {:>12} | {:>10}",
             "Threads", "Total ops", "Wall time", "Speedup");
    println!("  {}", "-".repeat(53));
    for (n, elapsed, throughput) in &results {
        println!("  {:>7} | {:>12} | {:>10.2}ms | {:>9.2}x",
                 n, n * ops_per_thread, elapsed.as_secs_f64() * 1000.0, throughput / baseline);
    }
}

fn report(label: &str, total_ops: usize, single: Duration, multi: Duration, threads: usize) {
    let st_ops = total_ops as f64 / single.as_secs_f64();
    let mt_ops = total_ops as f64 / multi.as_secs_f64();
    let speedup = single.as_secs_f64() / multi.as_secs_f64();

    println!("  Single-threaded:  {:>10.2}ms  ({:.0} ops/sec)", single.as_secs_f64() * 1000.0, st_ops);
    println!("  {}-thread:       {:>10.2}ms  ({:.0} ops/sec)", threads, multi.as_secs_f64() * 1000.0, mt_ops);
    println!("  Speedup:          {:.2}x for {}", speedup, label);
}

/// Exact fixed-point coordinates from decimal literals.
fn fp(vals: &[f64]) -> Vec<g_math::fixed_point::FixedPoint> {
    vals.iter().map(|&v| g_math::fixed_point::FixedPoint::from_f64(v)).collect()
}
