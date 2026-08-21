//! `Store::nearest` must return the *true* nearest node, not the node the
//! point-location grid happens to name.
//!
//! The grid stores one owner per tile. Sarkar placement pushes nodes toward
//! the boundary exponentially with depth, so a node's power cell falls below
//! tile size within a few levels and the grid can no longer name it — in a
//! 43-node tree only depth ≤ 2 owns any tile at all. A grid hit is therefore
//! a *proposal*: the true nearest need not be the tile's owner, nor one of
//! that owner's tree neighbours. `nearest` must consult the VP-tree on every
//! query, not only when the grid misses, or it answers confidently wrong.
//!
//! Regression: before that change, querying at a node's own exact stored
//! position returned a different node for 25 of 42 nodes in the deep tree
//! below and 4 of 30 in the flat one.

use horon_engine::Store;
use g_math::fixed_point::FixedPoint;

/// Deep, unbalanced: branch depths 2/5/8/11, plus three shallow siblings per
/// branch — a wide spread of Klein norms, which is what breaks a uniform grid.
fn deep_tree() -> (Store, Vec<String>) {
    let store = Store::new();
    let mut keys = vec!["/".to_string()];
    for a in 0..4 {
        let mut path = format!("/n{}", a);
        store.put(&path, b"x").unwrap();
        keys.push(path.clone());
        for d in 0..(2 + a * 3) {
            path = format!("{}/c{}", path, d);
            store.put(&path, b"x").unwrap();
            keys.push(path.clone());
        }
        for s in 0..3 {
            let sp = format!("/n{}/s{}", a, s);
            store.put(&sp, b"x").unwrap();
            keys.push(sp);
        }
    }
    (store, keys)
}

fn norm_sq(v: &[f64]) -> f64 {
    v.iter().map(|c| c * c).sum()
}

/// Monotone in hyperbolic distance — enough to rank, and it needs no atanh.
fn cosh_dist(a: &[f64], b: &[f64]) -> f64 {
    let d2: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    1.0 + 2.0 * d2 / ((1.0 - norm_sq(a)) * (1.0 - norm_sq(b)))
}

/// A query at a node's own exact position: distance 0 is the global minimum,
/// so the node itself must come back. No oracle needed, and using the stored
/// `FixedPoint` coordinates directly rules out any f64 round-trip.
#[test]
fn a_node_is_its_own_nearest_neighbour() {
    let (store, keys) = deep_tree();
    for key in &keys {
        let Ok(position) = store.position(key) else { continue };
        let (found, _) = store.nearest(&position).unwrap();
        assert_eq!(
            found, *key,
            "querying at {}'s own position returned {}",
            key, found
        );
    }

    // Flat tree: all leaves share one Klein norm, so the grid's *diagram* is
    // right here — the failures were purely tile-ownership, a different route
    // to the same wrong answer.
    let store = Store::new();
    for i in 0..30 {
        store.put(&format!("/leaf{}", i), b"x").unwrap();
    }
    for i in 0..30 {
        let key = format!("/leaf{}", i);
        let position = store.position(&key).unwrap();
        assert_eq!(store.nearest(&position).unwrap().0, key);
    }
}

/// Arbitrary query points, checked against a brute-force scan.
#[test]
fn matches_brute_force_for_arbitrary_queries() {
    let (store, keys) = deep_tree();
    let sites: Vec<(String, Vec<f64>)> = keys
        .iter()
        .filter_map(|k| {
            store
                .position(k)
                .ok()
                .map(|p| (k.clone(), p.iter().map(|c| c.to_f64()).collect()))
        })
        .collect();

    // xorshift: deterministic, so a failure is reproducible.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rand = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };

    for _ in 0..300 {
        // sqrt for area-uniform sampling rather than centre-heavy
        let r = rand().sqrt() * 0.98;
        let theta = rand() * std::f64::consts::TAU;
        let query = vec![r * theta.cos(), r * theta.sin(), 0.0, 0.0];

        let expected = &sites
            .iter()
            .min_by(|a, b| {
                cosh_dist(&query, &a.1)
                    .partial_cmp(&cosh_dist(&query, &b.1))
                    .expect("positions are finite")
            })
            .unwrap()
            .0;

        let fixed: Vec<FixedPoint> = query.iter().map(|c| FixedPoint::from_f64(*c)).collect();
        assert_eq!(
            store.nearest(&fixed).unwrap().0,
            **expected,
            "query {:?} took the grid's answer over the true nearest",
            &query[..2]
        );
    }
}
