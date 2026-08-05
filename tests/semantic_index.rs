//! Semantic-index regression tests: the lazy per-slice semantic VP-tree must be
//! result-identical to a brute-force reference, invalidate on every
//! semantic mutation, and break distance ties deterministically.
//!
//! `SEMANTIC_INDEX_MIN_NODES` (256) is the routing floor: stores below it
//! scan, stores above it use the index. Tests exercise both sides through
//! the public `Store` API.

use std::collections::BTreeMap;
use std::ops::Range;

use horon_engine::constants::SEMANTIC_INDEX_MIN_NODES;
use horon_engine::Store;
use g_math::fixed_point::FixedPoint;

/// Encode values for dims 16..16+vals.len() as raw Q64.64 bytes
/// (dims 0..16 zeroed, GACL-reserved).
fn coords(vals: &[i32]) -> Vec<u8> {
    let mut out = vec![0u8; 16 * 16];
    for &v in vals {
        out.extend_from_slice(&FixedPoint::from_int(v).raw().to_le_bytes());
    }
    out
}

/// Deterministic pseudo-random values (no rand dependency).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> i32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % 201) as i32 - 100 // range [-100, 100]
    }
}

/// Build a store with `n` data-only nodes carrying semantic coords, plus a
/// local map of what was written (the reference input).
fn build_store(n: usize, dims: usize, seed: u64) -> (Store, BTreeMap<String, Vec<u8>>) {
    let store = Store::new();
    store.put_data_only("/n", b"parent").unwrap();

    let mut lcg = Lcg(seed);
    let mut written = BTreeMap::new();
    for i in 0..n {
        let key = format!("/n/{:05}", i);
        store.put_data_only(&key, b"x").unwrap();
        // Every third node gets a SHORT vector (one dim fewer): dims past
        // its end must decode as zero on both paths.
        let d = if i % 3 == 0 { dims - 1 } else { dims };
        let vals: Vec<i32> = (0..d).map(|_| lcg.next()).collect();
        let c = coords(&vals);
        store.set_semantic(&key, c.clone()).unwrap();
        written.insert(key, c);
    }
    (store, written)
}

/// Brute-force reference over the locally recorded coords, using the same
/// public distance function and the same (distance, key) total order.
fn reference_knn(
    written: &BTreeMap<String, Vec<u8>>,
    query: &[u8],
    k: usize,
    range: Range<usize>,
) -> Vec<(String, g_math::fixed_point::FixedPoint)> {
    let mut all: Vec<(String, g_math::fixed_point::FixedPoint)> = written
        .iter()
        .map(|(key, c)| (key.clone(), Store::semantic_distance(query, c, range.clone())))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then_with(|| a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

/// Indexed path (above the floor) and scan path (below it) must both match
/// the reference exactly — keys AND distances — across slices and k values.
#[test]
fn nearest_semantic_matches_reference_on_both_paths() {
    for &(n, seed) in &[
        (60usize, 11u64),                          // below the floor: scan path
        (SEMANTIC_INDEX_MIN_NODES + 60, 4242u64),  // above the floor: indexed path
    ] {
        let (store, written) = build_store(n, 4, seed);
        let mut lcg = Lcg(seed ^ 0x5eed);

        for ref_range in [16..20usize, 16..18, 17..19] {
            for _ in 0..5 {
                let qvals: Vec<i32> = (0..4).map(|_| lcg.next()).collect();
                let query = coords(&qvals);
                for &k in &[1usize, 7, 25] {
                    let got = store
                        .nearest_semantic(&query, k, ref_range.clone())
                        .unwrap();
                    let want = reference_knn(&written, &query, k, ref_range.clone());
                    assert_eq!(
                        got, want,
                        "mismatch: n={} slice={:?} k={}",
                        n, ref_range, k
                    );
                }
            }
        }
    }
}

/// A semantic write after the index is warm must be visible to the very
/// next query (epoch invalidation), and a delete must remove the node.
#[test]
fn index_invalidates_on_write_and_delete() {
    let n = SEMANTIC_INDEX_MIN_NODES + 20;
    let (store, _) = build_store(n, 4, 99);

    let query = coords(&[500, 500, 500, 500]); // far from every node
    let range = 16..20usize;

    // Warm the index.
    let before = store.nearest_semantic(&query, 3, range.clone()).unwrap();
    assert_eq!(before.len(), 3);
    assert!(before[0].1.to_f64() > 0.0, "no node sits at the query point yet");

    // Move one node exactly onto the query point.
    let moved = "/n/00007";
    store.set_semantic(moved, query.clone()).unwrap();
    let after = store.nearest_semantic(&query, 3, range.clone()).unwrap();
    assert_eq!(after[0].0, moved, "stale index: write not visible");
    assert_eq!(after[0].1, g_math::fixed_point::FixedPoint::from_int(0));

    // Delete it again: it must vanish from the results.
    store.remove(moved).unwrap();
    let gone = store.nearest_semantic(&query, 3, range.clone()).unwrap();
    assert!(
        gone.iter().all(|(key, _)| key != moved),
        "deleted node still served by the index: {:?}",
        gone
    );
}

/// Distance ties must break by key, identically on repeated queries.
#[test]
fn ties_break_deterministically_by_key() {
    let n = SEMANTIC_INDEX_MIN_NODES + 10;
    let (store, _) = build_store(n, 4, 7);

    // Place five nodes at the exact same coordinates.
    let spot = coords(&[77, 77, 77, 77]);
    let tied = ["/n/00050", "/n/00010", "/n/00030", "/n/00020", "/n/00040"];
    for key in &tied {
        store.set_semantic(key, spot.clone()).unwrap();
    }

    let r1 = store.nearest_semantic(&spot, 5, 16..20).unwrap();
    let r2 = store.nearest_semantic(&spot, 5, 16..20).unwrap();
    assert_eq!(r1, r2, "repeated identical queries must be byte-identical");

    let keys: Vec<&str> = r1.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys,
        vec!["/n/00010", "/n/00020", "/n/00030", "/n/00040", "/n/00050"],
        "distance-zero ties must be sorted by key"
    );
    assert!(r1.iter().all(|(_, d)| *d == g_math::fixed_point::FixedPoint::from_int(0)));
}
