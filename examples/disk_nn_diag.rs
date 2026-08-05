//! Disk-NN diagnostic: where do the milliseconds in `disk.nearest` live?
//! Counts metric calls (visits) and times each phase separately.
//! Run: GMATH_PROFILE=embedded cargo run --release --example disk_nn_diag

use std::cell::Cell;
use std::time::Instant;

use horon_engine::metric_tree::{CachedNormPoint, HyperbolicMetric, Metric, MetricVpTree};
use horon_engine::{SemanticDisk, Store};
use g_math::fixed_point::FixedPoint;

struct CountingMetric {
    inner: HyperbolicMetric,
    proxies: Cell<u64>,
    exacts: Cell<u64>,
}

impl Metric<CachedNormPoint> for CountingMetric {
    fn distance(&self, a: &CachedNormPoint, b: &CachedNormPoint) -> FixedPoint {
        self.exacts.set(self.exacts.get() + 1);
        self.inner.distance(a, b)
    }
    fn has_proxy(&self) -> bool {
        true
    }
    fn proxy(&self, a: &CachedNormPoint, b: &CachedNormPoint) -> FixedPoint {
        self.proxies.set(self.proxies.get() + 1);
        self.inner.proxy(a, b)
    }
    fn prune_left(&self, s: FixedPoint, m: FixedPoint, w: FixedPoint) -> bool {
        self.inner.prune_left(s, m, w)
    }
    fn prune_right(&self, s: FixedPoint, m: FixedPoint, w: FixedPoint) -> bool {
        self.inner.prune_right(s, m, w)
    }
    fn left_first(&self, s: FixedPoint, m: FixedPoint) -> bool {
        self.inner.left_first(s, m)
    }
}

fn coords(vals: &[f64]) -> Vec<u8> {
    let mut out = vec![0u8; 0];
    for &v in vals {
        out.extend_from_slice(&FixedPoint::from_f64(v).raw().to_le_bytes());
    }
    out
}

fn val(i: usize, d: usize) -> f64 {
    let x = (i.wrapping_mul(2654435761) ^ d.wrapping_mul(40503)) as u32;
    (x as f64) / (u32::MAX as f64)
}

fn main() {
    let n = 10_000usize;
    let store = Store::new();
    store.put_data_only("/c", b"parent").unwrap();
    for i in 0..n {
        let key = format!("/c/n{:05}", i);
        store.put_data_only(&key, b"x").unwrap();
        let vals: Vec<f64> = (0..5).map(|d| val(i, d)).collect();
        store.set_semantic(&key, coords(&vals)).unwrap();
    }
    let disk = SemanticDisk::build(&[("/a", 0), ("/b", 1), ("/c", 2), ("/d", 3), ("/e", 4)]).unwrap();

    // Phase 1: derive positions (what index() does before building).
    let t = Instant::now();
    let mut entries = Vec::new();
    for i in 0..n {
        let key = format!("/c/n{:05}", i);
        if let Ok(Some(p)) = disk.position_of(&store, &key) {
            let fp: Vec<f32> = p.iter().map(|v| *v as f32).collect();
            entries.push((
                key,
                CachedNormPoint::new(horon_engine::HyperbolicPoint::from_f32_slice(&fp)),
            ));
        }
    }
    println!("derive {} positions: {:?}", entries.len(), t.elapsed());

    // Phase 2: build with counting metric.
    let m = CountingMetric {
        inner: HyperbolicMetric,
        proxies: Cell::new(0),
        exacts: Cell::new(0),
    };
    let t = Instant::now();
    let tree = MetricVpTree::build(entries.clone(), &m);
    println!(
        "build: {:?}  (proxy calls: {}, exact calls: {})",
        t.elapsed(),
        m.proxies.get(),
        m.exacts.get()
    );

    // Phase 3: one warm query, counted.
    m.proxies.set(0);
    m.exacts.set(0);
    let query = entries[n / 2].1.clone();
    let t = Instant::now();
    let hits = tree.knn(&query, 10, &m);
    println!(
        "knn k=10: {:?}  (proxy calls/visits: {}, exact calls: {}), top: {}",
        t.elapsed(),
        m.proxies.get(),
        m.exacts.get(),
        hits[0].0
    );

    // Phase 4: primitive costs in this exact context.
    let a = &entries[10].1;
    let b = &entries[9000].1;
    let t = Instant::now();
    let mut acc = FixedPoint::from_int(0);
    for _ in 0..1000 {
        acc = acc + m.inner.proxy(a, b);
    }
    println!("proxy ×1000: {:?} (acc {:?})", t.elapsed(), acc.to_f64());
    let t = Instant::now();
    let mut acc = FixedPoint::from_int(0);
    for _ in 0..100 {
        acc = acc + m.inner.distance(a, b);
    }
    println!("exact ×100: {:?} (acc {:?})", t.elapsed(), acc.to_f64());

    // Phase 5: full disk.nearest for comparison.
    let _ = disk.nearest(&store, "/c/n05000", 10).unwrap(); // warm
    let t = Instant::now();
    let _ = disk.nearest(&store, "/c/n05000", 10).unwrap();
    println!("disk.nearest warm total: {:?}", t.elapsed());
}
