//! Regression tests for hardening fixes.
//!
//! Each test pins a bug that reached a release before being found: user-input
//! panics in `nearest`, range/KNN queries silently skipping deep nodes, and
//! deleted nodes ghosting in the semantic index. All three failed quietly —
//! wrong results, no error — which is why they get dedicated tests rather
//! than living inside a broader suite.

use horon_engine::Store;
use g_math::fixed_point::FixedPoint;

fn coords1(v: f64) -> Vec<u8> {
    FixedPoint::from_f64(v).raw().to_le_bytes().to_vec()
}

/// `nearest`/`nearest_k` must reject wrong-dimension input with an
/// error, not panic in the geometry kernels.
#[test]
fn nearest_rejects_wrong_dimension_input() {
    let store = Store::new();
    store.put("/a", b"x").unwrap();

    assert!(store.nearest(&fp(&[0.1, 0.1])).is_err(), "2 coords vs dim=4 must error");
    assert!(store.nearest(&[]).is_err(), "empty coords must error");
    assert!(store.nearest_k(&fp(&[0.1; 7]), 3).is_err(), "7 coords vs dim=4 must error");
    // Correct dimension still works.
    assert!(store.nearest(&fp(&[0.1, 0.0, 0.0, 0.0])).is_ok());
}

/// `find_within` must see nodes at depth >= 2. Depth pushes nodes toward the
/// boundary exponentially, so a range query has to keep widening through the
/// outer bands rather than stopping at the query's own neighbourhood. The
/// bucket layer that originally failed this is gone; the property is not.
#[test]
fn find_within_sees_deep_nodes() {
    let store = Store::new();
    store.put("/n1", b"x").unwrap();
    store.put("/n1/n2", b"x").unwrap();
    store.put("/n1/n2/n3", b"x").unwrap();
    store.put("/n1/n2/n3/n4", b"x").unwrap();
    store.put("/n1/sib", b"x").unwrap();
    store.put("/n1/n2/sib", b"x").unwrap();

    // Every non-root node sits at hyperbolic distance tau (=1.0) from its
    // parent, so a radius-1.5 query from any node must at least find it.
    for key in ["/n1", "/n1/n2", "/n1/n2/n3", "/n1/n2/n3/n4"] {
        let hits = store.find_within(key, g_math::fixed_point::FixedPoint::from_f64(1.5)).unwrap();
        assert!(
            !hits.is_empty(),
            "find_within({}, 1.5) returned nothing — parent is at distance 1.0",
            key
        );
    }

    // The depth-2 node must see both its parent and its sibling-branch child.
    let hits = store.find_within("/n1/n2", g_math::fixed_point::FixedPoint::from_f64(1.5)).unwrap();
    assert!(hits.contains(&"/n1".to_string()), "parent missing: {:?}", hits);
    assert!(hits.contains(&"/n1/n2/n3".to_string()), "child missing: {:?}", hits);
}

/// `neighbors` (KNN) must also reach deep nodes.
#[test]
fn neighbors_sees_deep_nodes() {
    let store = Store::new();
    store.put("/n1", b"x").unwrap();
    store.put("/n1/n2", b"x").unwrap();
    store.put("/n1/n2/n3", b"x").unwrap();
    store.put("/n1/n2/n3/n4", b"x").unwrap();

    let n = store.neighbors("/n1/n2/n3", 2).unwrap();
    assert!(
        n.iter().any(|p| p == "/n1/n2" || p == "/n1/n2/n3/n4"),
        "KNN from a depth-3 node found neither parent nor child: {:?}",
        n
    );
}

/// deleting a node must actually remove it — its semantic coordinates
/// must not crowd out live results in `nearest_semantic`.
#[test]
fn deleted_node_does_not_ghost_semantic_results() {
    let store = Store::new();
    store.put("/a", b"x").unwrap();
    store.put("/b", b"x").unwrap();
    store.put("/c", b"x").unwrap();
    store.set_semantic("/a", coords1(1.0)).unwrap();
    store.set_semantic("/b", coords1(1.1)).unwrap();
    store.set_semantic("/c", coords1(5.0)).unwrap();

    store.remove("/b").unwrap();

    let res = store.nearest_semantic(&coords1(1.0), 2, 0..1).unwrap();
    assert_eq!(res.len(), 2, "ghost of /b crowded out a live result: {:?}", res);
    assert!(res.iter().all(|(p, _)| p != "/b"), "deleted /b returned: {:?}", res);
}

/// delete must keep the node count and store-level bookkeeping
/// consistent (len() uses the typed node count).
#[test]
fn delete_keeps_len_consistent() {
    let store = Store::new();
    store.put("/a", b"x").unwrap();
    store.put("/b", b"x").unwrap();
    assert_eq!(store.len(), 2);
    store.remove("/b").unwrap();
    assert_eq!(store.len(), 1);
    store.put("/b2", b"y").unwrap();
    assert_eq!(store.len(), 2);
}

/// Inserting then deleting a deep node must leave later spatial queries
/// correct. Originally this caught a bucket whose pruning radius stayed wide
/// after the deep node left, so every unrelated query over-scanned forever.
/// The cell index has no such per-region state — a deleted node leaves its
/// cell and that is all — but the observable property is worth keeping:
/// churn at depth must not perturb the answers.
#[test]
fn deep_node_churn_leaves_queries_correct() {
    let store = Store::new();
    store.put("/n1", b"x").unwrap();
    store.put("/n1/n2", b"x").unwrap();
    store.put("/n1/n2/n3", b"x").unwrap();
    // A transient deep branch, then removed.
    store.put("/n1/n2/n3/deep", b"x").unwrap();
    assert!(store.exists("/n1/n2/n3/deep"));
    store.remove("/n1/n2/n3/deep").unwrap();
    assert!(!store.exists("/n1/n2/n3/deep"));

    // The surviving nodes are still found at their true distances.
    let hits = store.find_within("/n1/n2", g_math::fixed_point::FixedPoint::from_f64(1.5)).unwrap();
    assert!(hits.contains(&"/n1".to_string()), "parent missing after churn: {:?}", hits);
    assert!(hits.contains(&"/n1/n2/n3".to_string()), "child missing after churn: {:?}", hits);
    assert!(!hits.contains(&"/n1/n2/n3/deep".to_string()), "deleted node ghosted: {:?}", hits);
}

/// concurrent inserts and deletes of many distinct children under a
/// single shared parent stress the parent-keyed write serialization
/// (`insert_data_only`'s stripe lock and `delete`'s two-stripe hold). After
/// the storm, every key that was left inserted must exist, every key that was
/// removed must be gone, and the node count must match exactly — no lost
/// updates, no double-frees, no torn parent child-lists.
#[test]
fn concurrent_sibling_churn_stays_consistent() {
    use std::thread;

    let store = Store::new();
    store.put("/p", b"x").unwrap();

    const THREADS: usize = 8;
    const PER_THREAD: usize = 40;

    thread::scope(|s| {
        for t in 0..THREADS {
            let store = &store;
            s.spawn(move || {
                for i in 0..PER_THREAD {
                    let key = format!("/p/t{}_n{}", t, i);
                    store.put(&key, b"v").unwrap();
                    // Delete the odd-indexed ones back out; keep the even ones.
                    if i % 2 == 1 {
                        store.remove(&key).unwrap();
                    }
                }
            });
        }
    });

    // Even indices survive, odd indices were removed.
    let mut expected_survivors = 0;
    for t in 0..THREADS {
        for i in 0..PER_THREAD {
            let key = format!("/p/t{}_n{}", t, i);
            if i % 2 == 0 {
                assert!(store.exists(&key), "survivor missing: {}", key);
                expected_survivors += 1;
            } else {
                assert!(!store.exists(&key), "removed key still present: {}", key);
            }
        }
    }

    // len() counts every live node except the root: parent /p + survivors.
    assert_eq!(
        store.len(),
        expected_survivors + 1,
        "node count diverged from the survivor set after concurrent churn"
    );
}

/// Exact fixed-point coordinates from decimal literals.
fn fp(vals: &[f64]) -> Vec<g_math::fixed_point::FixedPoint> {
    vals.iter().map(|&v| g_math::fixed_point::FixedPoint::from_f64(v)).collect()
}
