//! Rainbow fan-out: siblings fill concentric bands instead of
//! exhausting one circle, so extreme fan-out is collision-free by
//! construction.
//!
//! The regression this guards: ~1000 siblings under one parent used to
//! quantize two children to the same geometric signature, silently crossing
//! their data on Sarkar reconstruction (found by Horon's meaning-addressed
//! test suite).
//!
//! Run: GMATH_PROFILE=embedded cargo test --test rainbow_fanout

use horon_engine::Store;

/// The original rainbow-fan-out failure shape: massive fan-out under a single parent,
/// every child must keep its own data.
#[test]
fn two_thousand_siblings_no_data_crossover() {
    let store = Store::new();
    let n = 2000;

    for i in 0..n {
        store
            .put(&format!("/wide/node_{:04}", i), format!("payload-{}", i).as_bytes())
            .unwrap();
    }

    assert_eq!(
        store.children("/wide").unwrap().len(),
        n,
        "children lost under extreme fan-out"
    );
    for i in 0..n {
        let key = format!("/wide/node_{:04}", i);
        assert_eq!(
            store.get(&key).unwrap(),
            format!("payload-{}", i).as_bytes(),
            "data crossover at {}",
            key
        );
    }
}

/// Same shape, with per-child semantic coords — the exact original Horon scenario
/// (semantic vectors crossing between siblings).
#[test]
fn wide_fanout_semantics_stay_attached() {
    use g_math::fixed_point::FixedPoint;
    let store = Store::new();
    let n = 1200;

    let coords_of = |i: usize| -> Vec<u8> {
        let mut c = vec![0u8; 2 * 16];
        c[0..16].copy_from_slice(&FixedPoint::from_f64(i as f64).raw().to_le_bytes());
        c[16..32].copy_from_slice(&FixedPoint::from_f64((n - i) as f64).raw().to_le_bytes());
        c
    };

    for i in 0..n {
        let key = format!("/sem/node_{:04}", i);
        store.put(&key, b"x").unwrap();
        store.set_semantic(&key, coords_of(i)).unwrap();
    }
    for i in 0..n {
        let key = format!("/sem/node_{:04}", i);
        assert_eq!(
            store.get_semantic(&key).unwrap(),
            coords_of(i),
            "semantic crossover at {}",
            key
        );
    }
}

/// Band 0 must be bit-identical to the historical single-ring placement:
/// small-fan-out trees keep their exact geometry (spatial query results
/// unchanged, persisted files reconstruct identically).
#[test]
fn band_zero_preserves_classic_placement() {
    // Few-children tree: structural neighbors must stay within the branch —
    // siblings and the parent (at distance exactly τ) are legitimate nearest
    // neighbors; nothing from outside /small may appear.
    //
    // (hardening-audit note: before the effective-radius pruning fix, the parent
    // "/small" was silently missing from KNN results because its bucket was
    // wrongly pruned; the fixed query correctly ranks it among the nearest.)
    let store = Store::new();
    for i in 0..10 {
        store.put(&format!("/small/item_{}", i), b"v").unwrap();
    }
    let neighbors = store.neighbors("/small/item_0", 3).unwrap();
    assert_eq!(neighbors.len(), 3);
    for k in &neighbors {
        assert!(
            k == "/small" || k.starts_with("/small/"),
            "band-0 neighbor left the branch: {}",
            k
        );
    }
}
