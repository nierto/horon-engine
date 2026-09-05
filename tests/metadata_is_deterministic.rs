//! `get_meta` must return the same map on every read of an unmodified store.
//!
//! Reported by a downstream determinism audit (2026-09-05): folding
//! `get_meta` into a content hash made two reads of ONE byte-identical file
//! disagree, because the map carried `created_at`/`updated_at` taken from the
//! wall clock when the node was materialised. Two reads twenty seconds apart
//! produced timestamps twenty seconds apart.
//!
//! The fields were never durable — a node is rebuilt on every load, and the
//! file layer stripped them before persisting in five separate places — so
//! they answered a question nobody could act on while breaking one people did.

use std::collections::HashMap;
use std::thread::sleep;
use std::time::Duration;

use horon_engine::Store;

fn populated() -> Store {
    let store = Store::new();
    for i in 0..20 {
        let key = format!("/n{i}");
        store.put(&key, format!("payload-{i}").as_bytes()).unwrap();
        store.set_meta(&key, "author", "nierto").unwrap();
        store.set_meta(&key, "stage", "q1").unwrap();
    }
    store
}

/// The defect exactly as reported: read, wait, read again, compare.
#[test]
fn two_reads_of_one_store_agree() {
    let store = populated();

    let first: Vec<HashMap<String, String>> =
        (0..20).map(|i| store.get_meta(&format!("/n{i}")).unwrap()).collect();
    // The original report saw 20 seconds between reads. One is enough: the
    // timestamps had one-second resolution, so a single tick reproduces it.
    sleep(Duration::from_millis(1100));
    let second: Vec<HashMap<String, String>> =
        (0..20).map(|i| store.get_meta(&format!("/n{i}")).unwrap()).collect();

    assert_eq!(first, second, "get_meta changed between two reads of an unmodified store");
}

/// The map contains what the caller stored, and not the engine's bookkeeping.
#[test]
fn get_meta_returns_only_what_was_set() {
    let store = populated();
    let meta = store.get_meta("/n7").unwrap();

    assert_eq!(meta.get("author").map(String::as_str), Some("nierto"));
    assert_eq!(meta.get("stage").map(String::as_str), Some("q1"));

    for leaked in ["created_at", "updated_at", "key", "size"] {
        assert!(
            !meta.contains_key(leaked),
            "`{leaked}` is engine bookkeeping and must not appear in get_meta; \
             it belongs to get_node_info",
        );
    }
    // `_child_index` stays. Unlike the four above it is durable, deterministic
    // and load-bearing: replay re-derives a node's position from it, so a
    // consumer that dropped it would move every node on reload. See the
    // persist-derived-position rule.
    assert_eq!(meta.get("_child_index").map(String::as_str), Some("7"));
    assert_eq!(meta.len(), 3, "unexpected extra keys: {meta:?}");
}

/// The bookkeeping is still reachable — moved, not deleted — and the parts of
/// it that are deterministic stay deterministic.
#[test]
fn node_info_carries_the_engine_view() {
    let store = populated();
    let a = store.get_node_info("/n7").unwrap();
    assert_eq!(a.key, "/n7");
    assert_eq!(a.size, "payload-7".len());

    let b = store.get_node_info("/n7").unwrap();
    assert_eq!(a.key, b.key, "key must not vary between reads");
    assert_eq!(a.size, b.size, "size must not vary between reads");
    // The timestamps are documented as process-local, so they are *allowed* to
    // differ across reloads — but within one process, an untouched node must
    // not spontaneously change when it was materialised.
    assert_eq!(a.materialized_at, b.materialized_at);
}

/// A mutation advances `touched_at` without disturbing `materialized_at`.
#[test]
fn touching_a_node_moves_only_the_touch_time() {
    let store = populated();
    let before = store.get_node_info("/n3").unwrap();
    sleep(Duration::from_millis(1100));
    store.set_meta("/n3", "stage", "q2").unwrap();
    let after = store.get_node_info("/n3").unwrap();

    assert_eq!(before.materialized_at, after.materialized_at);
    assert!(
        after.touched_at > before.touched_at,
        "a mutation must advance touched_at ({} -> {})",
        before.touched_at, after.touched_at,
    );
}
