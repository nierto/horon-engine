//! Embed-on-demand regression tests: upgrading data-only nodes to full embeddings.
//!
//! The contract: after `embed_existing`, a `put_data_only` node behaves
//! exactly like a `put` node — spatial queries find it, it has a position,
//! and its identity (value, metadata, semantic coords) is untouched.
//! Ancestors embed automatically; the operation is idempotent; readers
//! racing an embed never observe the key missing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use horon_engine::Store;
use g_math::fixed_point::FixedPoint;

fn coords(vals: &[f64]) -> Vec<u8> {
    let mut out = vec![0u8; 16 * 16];
    for &v in vals {
        out.extend_from_slice(&FixedPoint::from_f64(v).raw().to_le_bytes());
    }
    out
}

#[test]
fn embed_upgrades_a_chain_and_preserves_identity() {
    let store = Store::new();
    store.put_data_only("/a", b"a-data").unwrap();
    store.put_data_only("/a/b", b"b-data").unwrap();
    store.put_data_only("/a/b/c", b"c-data").unwrap();
    store.set_meta("/a/b/c", "author", "alice").unwrap();
    store.set_semantic("/a/b/c", coords(&[0.7, 0.2])).unwrap();

    // Data-only: no position, invisible to spatial queries.
    assert!(store.position("/a/b/c").is_err());
    assert!(store.neighbors("/a/b/c", 1).is_err() || store.neighbors("/a/b/c", 1).unwrap().is_empty());

    // Embedding the leaf pulls the whole ancestor chain in.
    assert!(store.embed_existing("/a/b/c").unwrap(), "first embed does work");

    for key in ["/a", "/a/b", "/a/b/c"] {
        let pos = store.position(key).unwrap();
        assert!(!pos.is_empty(), "{} must have a position", key);
    }

    // Identity preserved through the upgrade.
    assert_eq!(store.get("/a/b/c").unwrap(), b"c-data");
    assert_eq!(store.get_meta("/a/b/c").unwrap().get("author").unwrap(), "alice");
    let sem = store.get_semantic("/a/b/c").unwrap();
    assert!(!sem.is_empty(), "semantic coords must survive the embed");
    let hits = store.nearest_semantic(&coords(&[0.7, 0.2]), 1, 16..18).unwrap();
    assert_eq!(hits[0].0, "/a/b/c");
    assert_eq!(hits[0].1.to_f64(), 0.0);

    // Spatial world now sees the chain: the parent is within reach.
    let near = store.find_within("/a/b/c", g_math::fixed_point::FixedPoint::from_f64(1.5)).unwrap();
    assert!(near.contains(&"/a/b".to_string()), "parent not spatially visible: {:?}", near);
}

#[test]
fn embed_is_idempotent_and_full_nodes_are_noops() {
    let store = Store::new();
    store.put("/full", b"x").unwrap();
    store.put_data_only("/full/lazy", b"y").unwrap();

    assert!(!store.embed_existing("/full").unwrap(), "already-embedded → no-op");
    assert!(store.embed_existing("/full/lazy").unwrap());
    assert!(!store.embed_existing("/full/lazy").unwrap(), "second embed → no-op");
    assert!(store.embed_existing("/missing").is_err());
}

#[test]
fn embedded_child_joins_its_full_parent_geometrically() {
    let store = Store::new();
    store.put("/p", b"parent").unwrap();
    store.put_data_only("/p/lazy", b"child").unwrap();
    store.put("/p/full", b"sibling").unwrap();

    store.embed_existing("/p/lazy").unwrap();

    // The upgraded child must be reachable from its embedded sibling.
    let near = store.find_within("/p/full", g_math::fixed_point::FixedPoint::from_f64(3.0)).unwrap();
    assert!(near.contains(&"/p/lazy".to_string()), "embedded child unreachable: {:?}", near);

    // And deleting it afterwards must work through the full path.
    store.remove("/p/lazy").unwrap();
    assert!(!store.exists("/p/lazy"));
    let near = store.find_within("/p/full", g_math::fixed_point::FixedPoint::from_f64(3.0)).unwrap();
    assert!(!near.contains(&"/p/lazy".to_string()), "deleted node still visible");
}

#[test]
fn embed_all_reports_upgrades_and_is_deterministic() {
    let build = || {
        let store = Store::new();
        store.put_data_only("/n", b"p").unwrap();
        for i in 0..20 {
            let key = format!("/n/{:02}", i);
            store.put_data_only(&key, b"x").unwrap();
            store.set_semantic(&key, coords(&[i as f64 * 0.05, 0.5])).unwrap();
        }
        store.embed_all("/n").unwrap();
        store
    };

    let a = build();
    let b = build();
    assert_eq!(a.embed_all("/n").unwrap(), 0, "second embed_all does nothing");

    // Same operation sequence → identical positions and spatial answers.
    for i in 0..20 {
        let key = format!("/n/{:02}", i);
        assert_eq!(a.position(&key).unwrap(), b.position(&key).unwrap(), "{}", key);
    }
    assert_eq!(
        a.neighbors("/n/07", 5).unwrap(),
        b.neighbors("/n/07", 5).unwrap()
    );
}

/// Readers racing an embed must never observe the key missing, and the
/// data must never change under them.
#[test]
fn readers_never_lose_the_key_during_embed() {
    let store = Arc::new(Store::new());
    store.put_data_only("/r", b"p").unwrap();
    for i in 0..50 {
        let key = format!("/r/{:02}", i);
        store.put_data_only(&key, b"payload").unwrap();
        store.set_semantic(&key, coords(&[i as f64 * 0.01, 0.3])).unwrap();
    }

    let stop = Arc::new(AtomicBool::new(false));
    let mut readers = Vec::new();
    for t in 0..4 {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        readers.push(std::thread::spawn(move || {
            let mut checks = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let key = format!("/r/{:02}", (checks as usize * 7 + t * 13) % 50);
                let data = store.get(&key).expect("key vanished during embed");
                assert_eq!(data, b"payload");
                let sem = store.get_semantic(&key).expect("semantic read failed");
                assert!(!sem.is_empty(), "coords vanished during embed");
                checks += 1;
            }
            checks
        }));
    }

    let upgraded = store.embed_all("/r").unwrap();
    assert_eq!(upgraded, 51); // parent + 50 children

    stop.store(true, Ordering::Relaxed);
    for r in readers {
        assert!(r.join().unwrap() > 0, "reader did no work");
    }

    // Post-embed: fully spatial.
    for i in 0..50 {
        assert!(store.position(&format!("/r/{:02}", i)).is_ok());
    }
}
