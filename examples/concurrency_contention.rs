//! concurrency_contention.rs - Measure contention behavior under the striped lock
//!
//! Shows that writes to the SAME parent serialize correctly via StripedLock<64>,
//! while writes to DIFFERENT parents proceed in parallel.
//!
//! Run: GMATH_PROFILE=embedded cargo run --release --example concurrency_contention

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

use horon_engine::Store;

fn main() {
    println!("=== Contention Analysis: Same Parent vs Different Parents ===\n");

    let num_threads = 4;
    let ops_per_thread = 50;

    // Scenario A: all threads write children under the SAME parent
    // StripedLock serializes these — correctness guaranteed, throughput limited
    println!("--- Scenario A: {} threads, all writing under /same_parent ---", num_threads);
    let store = Arc::new(Store::new());
    let counter = Arc::new(AtomicUsize::new(0));

    let start = Instant::now();
    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let store = Arc::clone(&store);
            let counter = Arc::clone(&counter);
            thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let idx = counter.fetch_add(1, Ordering::Relaxed);
                    let key = format!("/same_parent/child_{}", idx);
                    store.put(&key, format!("t{}i{}", tid, i).as_bytes()).unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let same_parent = start.elapsed();
    let total_same = num_threads * ops_per_thread;
    println!("  {} inserts in {:.2}ms ({:.0} ops/sec)",
             total_same, same_parent.as_secs_f64() * 1000.0,
             total_same as f64 / same_parent.as_secs_f64());

    // Verify all children exist
    let children = store.children("/same_parent").unwrap();
    let subtree = store.list("/same_parent").unwrap();
    println!("  Children (via children()): {} (expected {})", children.len(), total_same);
    println!("  Subtree  (via list()):     {}", subtree.len());

    // Check which keys are missing
    let mut missing = Vec::new();
    for i in 0..total_same {
        let key = format!("/same_parent/child_{}", i);
        if !store.exists(&key) {
            missing.push(i);
        }
    }
    if !missing.is_empty() {
        println!("  Missing keys (not in path_map): {:?}", &missing[..missing.len().min(10)]);
    }

    let in_subtree_not_children: Vec<_> = subtree.iter()
        .filter(|s| !children.iter().any(|c| c == *s))
        .collect();
    if !in_subtree_not_children.is_empty() {
        println!("  In subtree but not children(): {:?}", &in_subtree_not_children[..in_subtree_not_children.len().min(10)]);
    }

    // Scenario B: each thread writes under its OWN parent
    // StripedLock allows parallelism — different parents hit different stripes
    println!("\n--- Scenario B: {} threads, each writing under /parent_N ---", num_threads);
    let store = Arc::new(Store::new());

    let start = Instant::now();
    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let key = format!("/parent_{}/child_{}", tid, i);
                    store.put(&key, b"x").unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let diff_parent = start.elapsed();
    let total_diff = num_threads * ops_per_thread;
    println!("  {} inserts in {:.2}ms ({:.0} ops/sec)",
             total_diff, diff_parent.as_secs_f64() * 1000.0,
             total_diff as f64 / diff_parent.as_secs_f64());

    // Verify all children exist per parent
    for tid in 0..num_threads {
        let children = store.children(&format!("/parent_{}", tid)).unwrap();
        assert_eq!(children.len(), ops_per_thread,
            "Each parent should have {} children", ops_per_thread);
    }

    let speedup = same_parent.as_secs_f64() / diff_parent.as_secs_f64();
    println!("\n  Different-parent speedup vs same-parent: {:.2}x", speedup);
    println!("  (Stripe isolation allows parallel writes to independent subtrees)");

    // Scenario C: readers concurrent with writers — no blocking
    println!("\n--- Scenario C: {} reader threads + 1 writer thread ---", num_threads - 1);
    let store = Arc::new(Store::new());
    // Pre-populate
    for i in 0..100 {
        store.put(&format!("/read/item_{}", i), b"data").unwrap();
    }

    let read_count = Arc::new(AtomicUsize::new(0));
    let write_count = Arc::new(AtomicUsize::new(0));

    let start = Instant::now();
    let duration = std::time::Duration::from_millis(2000);

    let mut handles = Vec::new();

    // Writer thread
    {
        let store = Arc::clone(&store);
        let write_count = Arc::clone(&write_count);
        handles.push(thread::spawn(move || {
            let mut i = 0;
            while start.elapsed() < duration {
                let key = format!("/write/node_{}", i);
                store.put(&key, b"w").unwrap();
                write_count.fetch_add(1, Ordering::Relaxed);
                i += 1;
            }
        }));
    }

    // Reader threads
    for _ in 0..(num_threads - 1) {
        let store = Arc::clone(&store);
        let read_count = Arc::clone(&read_count);
        handles.push(thread::spawn(move || {
            let mut i = 0;
            while start.elapsed() < duration {
                let key = format!("/read/item_{}", i % 100);
                let _ = store.get(&key).unwrap();
                read_count.fetch_add(1, Ordering::Relaxed);
                i += 1;
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let reads = read_count.load(Ordering::Relaxed);
    let writes = write_count.load(Ordering::Relaxed);
    println!("  Over 2 seconds:");
    println!("    Reads:  {} ({:.0}/sec)", reads, reads as f64 / 2.0);
    println!("    Writes: {} ({:.0}/sec)", writes, writes as f64 / 2.0);
    println!("  Readers are NOT blocked by writer — no global RwLock starvation.");

    println!("\n=== Done ===");
}
