//! Every answer the cell index gives is checked against **brute force** — the
//! only oracle that cannot itself be wrong.
//!
//! This began as the dual-write changeover gate, comparing the cell index and
//! the bucket layer it replaced. It never compared them to *each other*: the
//! bucket layer is exactly what 0.5.2 had to stop believing. The bucket layer
//! is gone as of 0.6.0; the brute-force comparison it was measured against
//! stays, because that is the part that had value.
//!
//! Queries are half self-nearest-neighbour (the answer is known a priori:
//! distance zero) and half arbitrary points, over a deep unbalanced tree and a
//! 400-sibling wide one — the two shapes that broke the layer this replaced.

use horon_engine::{HyperbolicPoint, Store};
use g_math::fixed_point::FixedPoint;

/// Deep and unbalanced: branch depths 2/5/8/11 plus shallow siblings, which is
/// the spread of Klein norms that broke the fixed buckets.
fn deep_tree() -> (Store, Vec<String>) {
    let store = Store::new();
    let mut keys = vec!["/".to_string()];
    for branch in 0..4 {
        let mut path = format!("/n{}", branch);
        store.put(&path, b"x").unwrap();
        keys.push(path.clone());
        for depth in 0..(2 + branch * 3) {
            path = format!("{}/c{}", path, depth);
            store.put(&path, b"x").unwrap();
            keys.push(path.clone());
        }
        for sibling in 0..3 {
            let leaf = format!("/n{}/s{}", branch, sibling);
            store.put(&leaf, b"x").unwrap();
            keys.push(leaf);
        }
    }
    (store, keys)
}

fn wide_tree(n: usize) -> (Store, Vec<String>) {
    let store = Store::new();
    let mut keys = vec!["/".to_string()];
    for i in 0..n {
        let path = format!("/leaf{:04}", i);
        store.put(&path, b"x").unwrap();
        keys.push(path);
    }
    (store, keys)
}

fn embedded(store: &Store, keys: &[String]) -> Vec<(String, HyperbolicPoint)> {
    keys.iter()
        .filter_map(|k| {
            store
                .position(k)
                .ok()
                .map(|p| (k.clone(), HyperbolicPoint::from_slice(&p)))
        })
        .collect()
}

/// Brute-force distances to the k nearest, ascending.
///
/// The index is keyed by `unique_id` (a hash) while the store is keyed by path,
/// and there is no public mapping between them — so the comparison is on
/// *distances*, using the ones the index itself reports. That is the right
/// comparison regardless: nodes genuinely tied at the same distance make
/// several answers correct, and asserting identity would report tie-breaks as
/// defects.
fn nearest_distances(nodes: &[(String, HyperbolicPoint)], q: &HyperbolicPoint, k: usize) -> Vec<f64> {
    let mut all: Vec<f64> = nodes
        .iter()
        .map(|(_, p)| q.hyperbolic_distance(p).to_f64())
        .collect();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all.truncate(k);
    all
}

fn agrees(want: &[f64], got: &[f64]) -> bool {
    want.len() == got.len() && want.iter().zip(got).all(|(a, b)| (a - b).abs() < 1e-9)
}

struct Tally {
    cell_wrong: usize,
    total: usize,
}

fn compare(label: &str, store: &Store, keys: &[String], ks: &[usize]) -> Tally {
    let nodes = embedded(store, keys);
    let cell = store.inner().shared_htt().tensor_network().cell_index();

    // Half the queries sit exactly on stored nodes (the self-nearest-neighbour
    // case, where the answer is known a priori); half are arbitrary points.
    let mut queries: Vec<HyperbolicPoint> = nodes.iter().map(|(_, p)| p.clone()).collect();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rand = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..nodes.len() {
        let r = rand().sqrt() * 0.97;
        let a = rand() * std::f64::consts::TAU;
        let mut coords = vec![FixedPoint::from_int(0); nodes[0].1.dimension()];
        coords[0] = FixedPoint::from_f64(r * a.cos());
        coords[1] = FixedPoint::from_f64(r * a.sin());
        queries.push(HyperbolicPoint::from_slice(&coords));
    }

    let mut tally = Tally { cell_wrong: 0, total: 0 };
    for k in ks {
        for q in &queries {
            tally.total += 1;
            let want = nearest_distances(&nodes, q, *k);
            let mut got: Vec<f64> = cell.knn(q, *k).into_iter().map(|(_, d)| d.to_f64()).collect();
            got.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if !agrees(&want, &got) {
                tally.cell_wrong += 1;
            }
        }
    }
    println!(
        "{:<28} nodes {:<5} cells {:<5} queries {:<5} cell-index disagreements {}",
        label,
        nodes.len(),
        cell.cell_count(),
        tally.total,
        tally.cell_wrong
    );
    tally
}

#[test]
fn cell_index_matches_brute_force_on_a_deep_tree() {
    let (store, keys) = deep_tree();
    let t = compare("deep unbalanced", &store, &keys, &[1, 3, 8]);
    assert_eq!(t.cell_wrong, 0, "cell index disagreed with brute force");
}

#[test]
fn cell_index_matches_brute_force_on_a_wide_tree() {
    let (store, keys) = wide_tree(400);
    let t = compare("wide, 400 siblings", &store, &keys, &[1, 5]);
    assert_eq!(t.cell_wrong, 0, "cell index disagreed with brute force");
}

/// The index must hold every embedded node and nothing else. A drift here
/// means an insert reached the store but not the index.
#[test]
fn the_index_holds_every_embedded_node() {
    let (store, keys) = deep_tree();
    let embedded_count = keys.iter().filter(|k| store.position(k).is_ok()).count();
    assert_eq!(
        store.inner().shared_htt().tensor_network().cell_index().len(),
        embedded_count,
        "cell index population drifted from the embedded node count"
    );
}

/// A node deleted from the store must not survive in the index, or a later
/// query would resurrect it.
#[test]
fn removal_reaches_the_cell_index() {
    let (store, keys) = deep_tree();
    let before = store.inner().shared_htt().tensor_network().cell_index().len();

    let victim = keys
        .iter()
        .find(|k| k.matches('/').count() > 2 && store.position(k).is_ok())
        .expect("a deep leaf to delete")
        .clone();
    let at = HyperbolicPoint::from_slice(&store.position(&victim).unwrap());

    store.remove(&victim).unwrap();

    let cell = store.inner().shared_htt().tensor_network().cell_index();
    assert_eq!(cell.len(), before - 1, "cell index did not shrink on delete");
    // The index is keyed by unique_id, so identity is not comparable here.
    // Distance is: nothing should still sit exactly on the deleted position.
    let nearest = cell.knn(&at, 1);
    assert!(
        nearest.first().map(|(_, d)| d.to_f64()).unwrap_or(1.0) > 1e-12,
        "a node still answers at the deleted position: {nearest:?}"
    );
}
