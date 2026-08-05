//! Usable nesting depth.
//!
//! Deep paths are the ordinary case for hierarchical data, and the failure
//! mode here is silent: a too-conservative guard makes the distance kernel
//! return its saturation value for legitimate geometry, at which point every
//! node is equidistant and nearest-neighbour ranking becomes arbitrary. No
//! error is raised — queries simply return the wrong nodes.
//!
//! Two properties are pinned:
//!   1. Siblings sharing a maximal prefix stay mutually discoverable to at
//!      least depth 20.
//!   2. The kernel keeps returning true distances rather than saturating —
//!      the underlying invariant, tested directly so a regression is
//!      diagnosed rather than merely detected.

use horon_engine::constants;
use horon_engine::Store;
use g_math::fixed_point::FixedPoint;

/// Minimum depth at which two siblings must still find each other.
/// Measured ceiling is 21 with the current constants; 20 leaves one level of
/// slack so an unrelated change does not fail this test spuriously.
const REQUIRED_DEPTH: usize = 20;

fn build_spine(depth: usize) -> (Store, String, String) {
    let store = Store::new();
    let mut prefix = String::new();
    for level in 0..depth {
        prefix.push_str(&format!("/l{}", level));
        store.put(&prefix, b"x").unwrap();
    }
    let a = format!("{}/sibling_a", prefix);
    let b = format!("{}/sibling_b", prefix);
    store.put(&a, b"a").unwrap();
    store.put(&b, b"b").unwrap();
    (store, a, b)
}

#[test]
fn siblings_stay_discoverable_to_required_depth() {
    for depth in 1..=REQUIRED_DEPTH {
        let (store, a, b) = build_spine(depth);
        let neighbors = store.neighbors(&a, 3).unwrap();
        assert!(
            neighbors.contains(&b),
            "at depth {} the sibling is not among the 3 nearest neighbours of {} \
             (got {:?}) — the distance kernel is probably saturating; see \
             constants::min_safe_denominator",
            depth,
            a,
            neighbors
        );
    }
}

#[test]
fn norms_grow_monotonically_with_depth() {
    // The old boundary margin rescaled deep points back onto near_boundary,
    // so norms cycled (0.99 → 0.9963 → 0.9986 → 0.9995 → 0.99) instead of
    // approaching 1. Monotonicity is the observable that catches it.
    let mut previous = 0.0f64;
    for depth in 1..=REQUIRED_DEPTH {
        let (store, a, _) = build_spine(depth);
        let coords = store.position(&a).unwrap();
        let norm: f64 = coords.iter().map(|c| c.to_f64() * c.to_f64()).sum::<f64>().sqrt();
        assert!(
            norm > previous,
            "‖p‖ must grow with depth, but depth {} gave {:.12} after {:.12} \
             — points are being rescaled back toward the origin",
            depth,
            norm,
            previous
        );
        assert!(norm < 1.0, "‖p‖ must stay strictly inside the disk at depth {}", depth);
        previous = norm;
    }
}

#[test]
fn kernel_does_not_saturate_at_depth() {
    // Sibling separation is a property of the local fan-out, not of position
    // in the disk, so it must stay a usable distance at every depth. It is
    // *not* perfectly constant: it steps from 1.8944 to 1.4557 at depth 16
    // and holds there, a discrete change in the placement geometry rather
    // than precision decay. Both values discriminate fine, so this test
    // bounds the separation instead of pinning it — see the open question in
    // docs/HYPERBOLIC_INDEX.md.
    let saturation =
        (FixedPoint::from_int(2) * constants::near_boundary().atanh()).to_f64();

    for depth in 4..=REQUIRED_DEPTH {
        let (store, a, b) = build_spine(depth);
        let d = distance_between(&store, &a, &b);
        assert!(
            (d - saturation).abs() > 1.0,
            "at depth {} the kernel returned {:.8}, which is the saturation \
             value {:.8} — the degenerate-denominator guard is firing on valid \
             geometry",
            depth,
            d,
            saturation
        );
        assert!(
            d > 1.0 && d < 3.0,
            "sibling separation at depth {} is {:.8}, outside the usable band \
             — placement geometry has changed materially",
            depth,
            d
        );
    }
}

#[test]
fn nearest_neighbours_stay_exact_at_depth() {
    // Discoverability is weaker than correctness: the sibling could appear in
    // the results while the *ranking* is wrong. This compares the index
    // against brute-force ranking over every node — the index may prune, but
    // never approximate.
    const K: usize = 5;

    for depth in [4usize, 12, 18, REQUIRED_DEPTH] {
        // Fan-out gives each neighbour slot real competitors rather than one
        // uncontested sibling.
        let store = Store::new();
        let mut prefix = String::new();
        let mut leaves = Vec::new();
        for level in 0..depth {
            prefix.push_str(&format!("/l{}", level));
            store.put(&prefix, b"spine").unwrap();
            for child in 0..3 {
                let key = format!("{}/c{}", prefix, child);
                store.put(&key, b"leaf").unwrap();
                leaves.push(key);
            }
        }

        // The root is not a positioned node, so it cannot be ranked by brute
        // force; it is excluded from both sides of the comparison.
        let positioned: Vec<(String, Vec<f64>)> = store
            .list("/")
            .unwrap()
            .into_iter()
            .filter(|k| k != "/")
            .filter_map(|k| {
                store.position(&k).ok().map(|p| {
                    (k, p.iter().map(|c| c.to_f64()).collect::<Vec<f64>>())
                })
            })
            .collect();

        for query in &leaves {
            let qp: Vec<f64> =
                store.position(query).unwrap().iter().map(|c| c.to_f64()).collect();
            let mut truth: Vec<(String, f64)> = positioned
                .iter()
                .filter(|(k, _)| k != query)
                .map(|(k, p)| (k.clone(), hyperbolic(&qp, p)))
                .collect();
            truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let got: Vec<String> = store
                .neighbors(query, K + 1)
                .unwrap()
                .into_iter()
                .filter(|k| k != "/")
                .take(K)
                .collect();

            // Symmetric trees produce exact ties, so any node at or inside the
            // k-th distance is an acceptable answer for a slot.
            let cutoff = truth.get(K - 1).map(|(_, d)| *d).unwrap_or(f64::MAX);
            let acceptable: std::collections::HashSet<&String> = truth
                .iter()
                .filter(|(_, d)| *d <= cutoff + 1e-9)
                .map(|(k, _)| k)
                .collect();

            assert_eq!(
                got.len(),
                K.min(truth.len()),
                "depth {}: expected {} neighbours for {}, got {:?}",
                depth,
                K,
                query,
                got
            );
            for candidate in &got {
                assert!(
                    acceptable.contains(candidate),
                    "depth {}: {} returned {} as a nearest neighbour, but it is \
                     outside the true k-NN set (cutoff {:.8}); true top {}: {:?}",
                    depth,
                    query,
                    candidate,
                    cutoff,
                    K,
                    truth.iter().take(K).collect::<Vec<_>>()
                );
            }
        }
    }
}

/// Hyperbolic distance between two coordinate vectors.
fn hyperbolic(pa: &[f64], pb: &[f64]) -> f64 {
    let na: f64 = pa.iter().map(|x| x * x).sum();
    let nb: f64 = pb.iter().map(|x| x * x).sum();
    let d2: f64 = pa.iter().zip(pb.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
    (1.0 + 2.0 * d2 / ((1.0 - na) * (1.0 - nb))).acosh()
}

/// Hyperbolic distance between two stored keys, via the store's own geometry.
fn distance_between(store: &Store, a: &str, b: &str) -> f64 {
    let pa: Vec<f64> = store.position(a).unwrap().iter().map(|c| c.to_f64()).collect();
    let pb: Vec<f64> = store.position(b).unwrap().iter().map(|c| c.to_f64()).collect();
    let na: f64 = pa.iter().map(|x| x * x).sum();
    let nb: f64 = pb.iter().map(|x| x * x).sum();
    let d2: f64 = pa.iter().zip(pb.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
    (1.0 + 2.0 * d2 / ((1.0 - na) * (1.0 - nb))).acosh()
}
