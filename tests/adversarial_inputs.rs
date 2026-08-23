//! Every public entry point, fed values a caller can legally construct but the
//! internal maths may not survive.
//!
//! Written after `find_within` was found to panic on `radius = 1000`: the
//! radius became a `cosh` ceiling, and Q64.64 `cosh` panics rather than
//! saturating. That defect was in the engine for the whole dual-write period
//! and no engine test caught it — `horon`'s API-surface suite did, because
//! `horon` passes 1000 to mean "everything". The class is: a caller-supplied
//! value crossing into a bounded-domain computation.
//!
//! These tests assert only that the engine does not panic and does not lie.
//! Returning an error is fine. Returning an empty result is fine where that is
//! the truth. Aborting the process is not.

use horon_engine::Store;
use g_math::fixed_point::FixedPoint;

fn populated() -> Store {
    let store = Store::new();
    for i in 0..40 {
        let p = format!("/n{i}");
        store.put(&p, b"x").unwrap();
        for j in 0..3 {
            store.put(&format!("{p}/c{j}"), b"x").unwrap();
        }
    }
    let mut deep = String::new();
    for d in 0..12 {
        deep = format!("{deep}/d{d}");
        store.put(&deep, b"x").unwrap();
    }
    store
}

fn fp(v: f64) -> FixedPoint {
    FixedPoint::from_f64(v)
}

/// `find_within` takes a caller radius straight into `cosh` space.
#[test]
fn find_within_survives_every_radius() {
    let store = populated();
    for r in [0.0f64, 1e-9, 0.5, 21.0, 28.4, 44.0, 45.0, 1e3, 1e9, 1e15] {
        let got = store.find_within("/n0", fp(r));
        assert!(got.is_ok(), "find_within panicked or errored at radius {r}");
    }
    // Negative: a radius below zero can contain nothing.
    let neg = store.find_within("/n0", fp(-1.0));
    assert!(neg.is_ok(), "negative radius must not blow up");
    assert!(neg.unwrap().is_empty(), "negative radius must match nothing");
}

/// `nearest` / `nearest_k` accept arbitrary coordinates. Points on or outside
/// the unit disk are not valid positions, but nothing stops a caller passing
/// them, and `1 - ||q||^2` is a divisor on the query path.
#[test]
fn nearest_survives_coordinates_on_and_outside_the_disk() {
    let store = populated();
    let cases: Vec<(&str, Vec<f64>)> = vec![
        ("origin", vec![0.0, 0.0, 0.0, 0.0]),
        ("just inside", vec![0.999_999, 0.0, 0.0, 0.0]),
        ("on the boundary", vec![1.0, 0.0, 0.0, 0.0]),
        ("outside", vec![2.0, 0.0, 0.0, 0.0]),
        ("far outside", vec![1e6, 0.0, 0.0, 0.0]),
        ("negative outside", vec![-3.0, -3.0, 0.0, 0.0]),
        ("norm 1 diagonally", vec![0.707_106_78, 0.707_106_78, 0.0, 0.0]),
    ];
    for (label, coords) in cases {
        let c: Vec<FixedPoint> = coords.iter().copied().map(fp).collect();
        let n = store.nearest(&c);
        assert!(n.is_ok() || n.is_err(), "unreachable");
        let k = store.nearest_k(&c, 5);
        assert!(k.is_ok() || k.is_err(), "unreachable");
        // The real assertion is that we got here at all.
        let _ = (label, n, k);
    }
}

/// k is caller-supplied and unbounded.
///
/// `k = 0` is the interesting one: it is a request for nothing, and the honest
/// answer is nothing. `nearest_k` used to report `No nodes in tree` against a
/// 200-node store, because it read an empty result set as an empty index —
/// two different causes, one error. `neighbors` always got this right, so the
/// two sibling APIs disagreed.
#[test]
fn extreme_k_is_answered_honestly() {
    let store = populated();
    let n = store.len();
    let q: Vec<FixedPoint> = vec![fp(0.1), fp(0.1), fp(0.0), fp(0.0)];

    assert_eq!(
        store.nearest_k(&q, 0).expect("k=0 against a populated store is not an error"),
        vec![],
        "k=0 must return nothing, not claim the tree is empty",
    );
    assert_eq!(store.neighbors("/n0", 0).unwrap(), Vec::<String>::new());

    for k in [1usize, 10_000, 1_000_000] {
        let got = store.nearest_k(&q, k).unwrap_or_else(|e| panic!("k={k}: {e:?}"));
        assert!(got.len() <= n + 1, "k={k} returned more nodes than exist");
        assert!(!got.is_empty(), "k={k} must find something in a populated store");
        assert!(store.neighbors("/n0", k).is_ok(), "neighbors failed at k={k}");
    }

    // A fresh store is not empty — it holds the root at the origin, which
    // `len()` excludes but the index does not. So k>0 always finds something
    // through the public API, and the "no nodes in tree" branch is a
    // defensive guard rather than a reachable state.
    let fresh = Store::new();
    let (_, d) = fresh.nearest(&q).expect("a fresh store still holds its root");
    assert!(d > FixedPoint::from_int(0), "root is not at the query point");
    assert_eq!(fresh.nearest_k(&q, 4).unwrap().len(), 1, "only the root exists");
    assert_eq!(fresh.nearest_k(&q, 0).unwrap().len(), 0, "k=0 still means none");
}

/// `tau` is caller-configurable and divides into the depth budget.
#[test]
fn extreme_tau_is_rejected_or_survived() {
    use horon_engine::StoreConfig;
    for t in [1e-9f64, 0.01, 1.0, 21.0, 100.0] {
        let store = Store::with_config(StoreConfig::new().tau(fp(t)));
        let root = store.put("/a", b"x");
        assert!(root.is_ok() || root.is_err(), "unreachable");
        let _ = store.put("/a/b", b"x");
        let _ = store.max_depth();
        let _ = store.nearest(&[fp(0.0), fp(0.0), fp(0.0), fp(0.0)]);
    }
}

/// Semantic queries take a caller-chosen dimension range and a z threshold.
#[test]
fn semantic_entry_points_survive_hostile_ranges_and_thresholds() {
    let store = Store::new();
    for i in 0..20 {
        let key = format!("/s{i}");
        store.put(&key, b"x").unwrap();
        let mut coords = vec![0u8; 4 * 16];
        for d in 0..4 {
            let v = FixedPoint::from_f64((i as f64 + d as f64) / 32.0).raw().to_le_bytes();
            coords[d * 16..(d + 1) * 16].copy_from_slice(&v);
        }
        store.set_semantic(&key, coords).unwrap();
    }
    let probe = store.get_semantic("/s3").unwrap();

    // A slice that cannot carry information is now rejected rather than
    // answered. `decode_semantic_slice` zero-extends by design, so an empty
    // range — or one starting past the query's own width — would make every
    // node tie at distance zero and let the key tie-break pick the "nearest".
    // A confident, reproducible, information-free answer is the one thing this
    // engine refuses to give.
    for r in [0..0, 2..2, 900..1000] {
        assert!(
            store.nearest_semantic(&probe, 3, r.clone()).is_err(),
            "vacuous range {r:?} must be rejected, not answered",
        );
        assert!(
            store.find_similar("/s3", 3, r.clone()).is_err(),
            "vacuous range {r:?} must be rejected for find_similar",
        );
    }
    assert!(
        store.find_outliers("/", fp(1.5), 0..0).is_err(),
        "an empty range leaves nothing for a node to be an outlier in",
    );

    // Zero-extension itself is legitimate and must keep working: a range that
    // starts inside the data and runs past it compares the overlap.
    for r in [0..4, 0..1000, 3..64] {
        assert!(
            store.nearest_semantic(&probe, 3, r.clone()).is_ok(),
            "range {r:?} overlaps the data and must still be answered",
        );
    }
    assert!(store.find_outliers("/", fp(1.5), 0..1000).is_ok());

    for z in [0.5f64, 1.5, 1e6] {
        assert!(store.find_outliers("/", fp(z), 0..4).is_ok(), "z={z} rejected");
    }
}
