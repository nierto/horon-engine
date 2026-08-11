//! Full-mode (in-memory) semantic k-NN cost — the Tier 3 measurement.
//!
//!     GMATH_PROFILE=embedded cargo run --release --example semantic_query_cost -- 12000

use g_math::fixed_point::FixedPoint;
use horon_engine::Store;
use std::time::Instant;

const SEM_DIMS: usize = 24;

fn coords(vals: &[f64]) -> Vec<u8> {
    let mut out = vec![0u8; SEM_DIMS * 16];
    for (d, v) in vals.iter().enumerate() {
        let off = (16 + d) * 16;
        out[off..off + 16].copy_from_slice(&FixedPoint::from_f64(*v).raw().to_le_bytes());
    }
    out
}

fn prng(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / (u32::MAX as f64 / 2.0)
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12_000);
    let reps = 200usize;

    let store = Store::new();
    let mut seed = 0x5EED_u64;
    for i in 0..n {
        let key = format!("/d/{i:06}");
        store.put(&key, b"x").unwrap();
        let v: Vec<f64> = (0..8).map(|_| prng(&mut seed).fract()).collect();
        store.set_semantic(&key, coords(&v)).unwrap();
    }

    let mut qseed = 0xC0FFEE_u64;
    let queries: Vec<Vec<u8>> = (0..reps)
        .map(|_| coords(&(0..8).map(|_| prng(&mut qseed).fract()).collect::<Vec<_>>()))
        .collect();

    // First query builds the lazy index; time it separately.
    let t = Instant::now();
    let _ = store.nearest_semantic(&queries[0], 10, 16..24).unwrap();
    println!("index build + first query: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);

    let t = Instant::now();
    for q in &queries {
        let _ = store.nearest_semantic(q, 10, 16..24).unwrap();
    }
    println!(
        "full-mode nearest_semantic(k=10), {n} nodes: {:.0} us/query",
        t.elapsed().as_secs_f64() * 1e6 / reps as f64
    );

    // Structural neighbors — the sites-371/532/570 consumer.
    let t = Instant::now();
    for i in 0..reps {
        let _ = store.neighbors(&format!("/d/{:06}", (i * 61) % n), 5).unwrap();
    }
    println!(
        "structural neighbors(k=5), {n} nodes: {:.0} us/query",
        t.elapsed().as_secs_f64() * 1e6 / reps as f64
    );
}
