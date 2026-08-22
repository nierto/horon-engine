//! Node identity must be unique from position and level alone.
//!
//! `GeometricSignature::unique_id` digests `level` plus `position_signature`
//! (coordinates quantised at 2^-20). It deliberately does *not* include the
//! geometric bucket hash: that would make a node's identity depend on which
//! bucket the spatial index chose, so the index could not be replaced without
//! renaming every node. The bucket hash is a function of the same position
//! anyway — nodes sharing a level and a position signature already shared a
//! bucket — so it contributed no discrimination.
//!
//! Uniqueness therefore rests entirely on the quantisation resolution, and a
//! collision is not a subtle failure: two paths would resolve to one node and
//! silently return each other's data. That is exactly what the hardening audit
//! found when identity used `quantize_1000` — depth-2 cousins in different
//! branches collided at a few hundred nodes.
//!
//! These are black-box crossover tests: every path stores data unique to it,
//! and every path must read back exactly what it stored.

use horon_engine::Store;

/// Store a distinctive payload at `path`, derived from the path itself so a
/// crossover is self-evident in the assertion message.
fn payload(path: &str) -> Vec<u8> {
    format!("payload-for::{}", path).into_bytes()
}

fn assert_no_crossover(store: &Store, paths: &[String], label: &str) {
    for path in paths {
        let got = store.get(path).unwrap_or_else(|e| {
            panic!("{label}: {path} vanished after insert: {e:?}")
        });
        assert_eq!(
            got,
            payload(path),
            "{label}: {path} returned another node's data — identity collision",
        );
    }
}

/// A deep chain stresses *radial* resolution: consecutive depths differ only
/// in how far along one ray they sit, and that gap shrinks exponentially.
/// Parent and child share no level, which is part of why they stay distinct.
#[test]
fn deep_chain_keeps_identities_distinct() {
    let store = Store::new();
    let mut paths = Vec::new();
    let mut path = String::new();
    for level in 0..17 {
        path = format!("{}/d{}", path, level);
        store.put(&path, &payload(&path)).unwrap();
        paths.push(path.clone());
    }
    assert_no_crossover(&store, &paths, "deep chain");
}

/// Same-level cousins in *different branches* are the case that broke under
/// the coarser quantisation: nothing about their level or branch distinguishes
/// them, only their coordinates.
#[test]
fn same_level_cousins_across_branches_stay_distinct() {
    let store = Store::new();
    let mut paths = Vec::new();
    let mut frontier = vec![String::new()];
    for _ in 0..5 {
        let mut next = Vec::new();
        for parent in &frontier {
            for child in 0..4 {
                let path = format!("{}/b{}", parent, child);
                if store.put(&path, &payload(&path)).is_ok() {
                    paths.push(path.clone());
                    next.push(path);
                }
            }
        }
        frontier = next;
        if paths.len() > 1_000 {
            break;
        }
    }
    assert!(paths.len() > 900, "fixture should be large enough to matter");
    assert_no_crossover(&store, &paths, "cousins");
}

/// Wide fan-out stresses *angular* resolution, and crosses rainbow band 0 at
/// 256 siblings so later children are placed on an outer ring.
#[test]
fn wide_fanout_keeps_identities_distinct() {
    let store = Store::new();
    store.put("/wide", &payload("/wide")).unwrap();
    let mut paths = vec!["/wide".to_string()];
    for i in 0..600 {
        let path = format!("/wide/s{:04}", i);
        store.put(&path, &payload(&path)).unwrap();
        paths.push(path);
    }
    assert_no_crossover(&store, &paths, "wide fan-out");
}

/// Identity must not depend on insertion order: the same tree built in a
/// different order must produce the same node count and the same data, since
/// position is a pure function of the path and the child index.
#[test]
fn identity_is_independent_of_sibling_insertion_order() {
    let build = |reverse: bool| {
        let store = Store::new();
        store.put("/root", &payload("/root")).unwrap();
        let mut order: Vec<usize> = (0..64).collect();
        if reverse {
            order.reverse();
        }
        // Insert under a fixed parent; child_index follows insertion order, so
        // the two stores place a given *path* at different sites. What must
        // hold is that neither store loses or merges a node.
        for i in order {
            let path = format!("/root/c{:03}", i);
            store.put(&path, &payload(&path)).unwrap();
        }
        store
    };

    for (label, store) in [("forward", build(false)), ("reverse", build(true))] {
        let paths: Vec<String> = (0..64).map(|i| format!("/root/c{:03}", i)).collect();
        assert_no_crossover(&store, &paths, label);
        assert_eq!(
            store.list("/root").unwrap().len(),
            64,
            "{label}: every sibling must survive as its own node"
        );
    }
}

/// The functional-integrity check must be *live* — a check that always returns
/// true is worse than none, because it looks like coverage.
///
/// `verify_index_locates_all_nodes` asks the spatial index to find every node
/// at its own stored position. Every other integrity check in the engine is
/// referential (do these maps point at things that exist?), and the bucket
/// layer passed all of them while `nearest` returned the wrong node for 25 of
/// 42 nodes.
#[test]
fn the_index_can_locate_every_node_it_holds() {
    let store = Store::new();
    // Deep and wide together: the shapes that broke the layer this replaced.
    let mut path = String::new();
    for level in 0..16 {
        path = format!("{}/d{}", path, level);
        store.put(&path, &payload(&path)).unwrap();
    }
    for i in 0..300 {
        let p = format!("/wide{:04}", i);
        store.put(&p, &payload(&p)).unwrap();
    }

    let network = store.inner().shared_htt().tensor_network();
    assert!(
        network.verify_index_locates_all_nodes(),
        "the spatial index cannot find a node it indexed"
    );
    // And it is reachable through the composite check.
    assert!(network.validate_network(), "network integrity failed");
}
