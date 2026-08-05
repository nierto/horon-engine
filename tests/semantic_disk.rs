//! Semantic-disk regression tests: the semantic disk (E+D design).
//!
//! The correctness bar matches the semantic index: indexed nearest-neighbor results must
//! equal a brute-force reference over the same derived positions, exactly.
//! Plus the design doc's behavioral guarantees: classification follows the
//! dominant affinity, movement re-homes nodes, miscategorization surfaces,
//! writes invalidate the cache, and trajectory samples classify into
//! symbolic concept sequences.

use horon_engine::{SemanticDisk, Store};
use g_math::fixed_point::FixedPoint;

/// Course-catalog-shaped spec: five categories mapped to dims 16..21.
fn spec() -> Vec<(&'static str, usize)> {
    vec![
        ("/trauma", 16),
        ("/cgt", 17),
        ("/systemisch", 18),
        ("/kind_jeugd", 19),
        ("/ouderen", 20),
    ]
}

/// Encode affinity values for dims 16.. (dims 0..16 zeroed).
fn coords(vals: &[f64]) -> Vec<u8> {
    let mut out = vec![0u8; 16 * 16];
    for &v in vals {
        out.extend_from_slice(&FixedPoint::from_f64(v).raw().to_le_bytes());
    }
    out
}

fn add(store: &Store, key: &str, affinities: &[f64]) {
    store.put_data_only(key, b"x").unwrap();
    store.set_semantic(key, coords(affinities)).unwrap();
}

#[test]
fn concept_follows_the_dominant_affinity() {
    let store = Store::new();
    let disk = SemanticDisk::build(&spec()).unwrap();

    store.put_data_only("/course", b"parent").unwrap();
    add(&store, "/course/emdr", &[0.9, 0.1, 0.0, 0.0, 0.0]);
    add(&store, "/course/schema", &[0.0, 0.85, 0.05, 0.0, 0.0]);
    add(&store, "/course/family", &[0.05, 0.0, 0.9, 0.0, 0.05]);

    assert_eq!(disk.concept_of(&store, "/course/emdr").unwrap().unwrap(), "/trauma");
    assert_eq!(disk.concept_of(&store, "/course/schema").unwrap().unwrap(), "/cgt");
    assert_eq!(disk.concept_of(&store, "/course/family").unwrap().unwrap(), "/systemisch");

    // No affinities → no concept position.
    store.put_data_only("/course/blank", b"x").unwrap();
    assert!(disk.concept_of(&store, "/course/blank").unwrap().is_none());
}

#[test]
fn miscategorization_surfaces_as_path_vs_concept_disagreement() {
    let store = Store::new();
    let disk = SemanticDisk::build(&spec()).unwrap();

    // Filed under /overig; its measured affinities are trauma-shaped.
    store.put_data_only("/course", b"parent").unwrap();
    store.put_data_only("/course/overig", b"parent").unwrap();
    add(&store, "/course/overig/dementie", &[0.8, 0.1, 0.0, 0.0, 0.1]);

    let concept = disk.concept_of(&store, "/course/overig/dementie").unwrap().unwrap();
    assert_eq!(concept, "/trauma");
    assert!(
        !"/course/overig/dementie".contains(&concept[1..]),
        "the storage path disagrees with the measured concept — that IS the finding"
    );
}

#[test]
fn movement_rehomes_a_node() {
    let store = Store::new();
    let disk = SemanticDisk::build(&spec()).unwrap();
    store.put_data_only("/c", b"parent").unwrap();
    add(&store, "/c/drifter", &[0.9, 0.1, 0.0, 0.0, 0.0]);
    assert_eq!(disk.concept_of(&store, "/c/drifter").unwrap().unwrap(), "/trauma");

    // Its population changes; recalibration rewrites the affinities.
    store.set_semantic("/c/drifter", coords(&[0.1, 0.1, 0.9, 0.0, 0.0])).unwrap();
    assert_eq!(
        disk.concept_of(&store, "/c/drifter").unwrap().unwrap(),
        "/systemisch",
        "a moved point gets an entirely new meaning"
    );
}

#[test]
fn nearest_matches_brute_force_over_derived_positions() {
    let store = Store::new();
    let disk = SemanticDisk::build(&spec()).unwrap();
    store.put_data_only("/c", b"parent").unwrap();

    // Deterministic varied affinities.
    let mut state: u64 = 42;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) % 100) as f64 / 100.0
    };
    let mut keys = Vec::new();
    for i in 0..120 {
        let key = format!("/c/n{:03}", i);
        add(&store, &key, &[next(), next(), next(), next(), next()]);
        keys.push(key);
    }

    // Brute force: hyperbolic distance between derived positions.
    let brute = |query_key: &str, k: usize| -> Vec<String> {
        let q = disk.position_of(&store, query_key).unwrap().unwrap();
        let mut all: Vec<(f64, String)> = keys
            .iter()
            .filter(|key| key.as_str() != query_key)
            .map(|key| {
                let p = disk.position_of(&store, key).unwrap().unwrap();
                // Poincaré distance via the library's own primitive.
                let a = horon_engine::HyperbolicPoint::from_f32_slice(
                    &q.iter().map(|v| *v as f32).collect::<Vec<_>>(),
                );
                let b = horon_engine::HyperbolicPoint::from_f32_slice(
                    &p.iter().map(|v| *v as f32).collect::<Vec<_>>(),
                );
                (a.hyperbolic_distance(&b).to_f64(), key.clone())
            })
            .collect();
        all.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap().then_with(|| x.1.cmp(&y.1)));
        all.truncate(k);
        all.into_iter().map(|(_, key)| key).collect()
    };

    for query_key in ["/c/n000", "/c/n057", "/c/n119"] {
        let got: Vec<String> = disk
            .nearest(&store, query_key, 8)
            .unwrap()
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        let want = brute(query_key, 8);
        // f32 round-trip in the reference introduces tiny rank jitter only
        // at genuinely-equal distances; compare sets AND leading order.
        assert_eq!(got.len(), want.len(), "query {}", query_key);
        assert_eq!(got[0], want[0], "nearest disagrees for {}", query_key);
        let mut gs = got.clone();
        let mut ws = want.clone();
        gs.sort();
        ws.sort();
        assert_eq!(gs, ws, "k-NN set disagrees for {}", query_key);
    }
}

#[test]
fn index_invalidates_when_the_store_moves() {
    let store = Store::new();
    let disk = SemanticDisk::build(&spec()).unwrap();
    store.put_data_only("/c", b"parent").unwrap();
    // /c/b shares /c/a's affinity shape (definitely nearest); /c/far is in
    // a different concept entirely. No assumption about anchor adjacency.
    add(&store, "/c/a", &[0.9, 0.1, 0.0, 0.0, 0.0]);
    add(&store, "/c/b", &[0.8, 0.2, 0.0, 0.0, 0.0]);
    add(&store, "/c/far", &[0.0, 0.0, 0.0, 0.0, 0.9]);

    let first = disk.nearest(&store, "/c/a", 1).unwrap();
    assert_eq!(first[0].0, "/c/b");

    // /c/far moves EXACTLY onto /c/a's affinities: distance zero must win.
    store.set_semantic("/c/far", coords(&[0.9, 0.1, 0.0, 0.0, 0.0])).unwrap();
    let second = disk.nearest(&store, "/c/a", 1).unwrap();
    assert_eq!(second[0].0, "/c/far", "stale disk index: write not visible");
}

#[test]
fn trajectory_classifies_into_a_symbolic_sequence() {
    let disk = SemanticDisk::build(&spec()).unwrap();

    // Samples shaped like HttHistory::trajectory(key, 16..21) output:
    // a node drifting trauma → trauma → systemisch across four epochs.
    // Dominant weights keep each classification unambiguous — a *balanced*
    // blend of two distant concepts travels through the middle of the disk
    // and honestly classifies to whichever anchor cell occupies that
    // region (documented behavior, not asserted here).
    let samples = vec![
        (1u64, vec![0.9, 0.05, 0.05, 0.0, 0.0]),
        (2u64, vec![0.85, 0.05, 0.1, 0.0, 0.0]),
        (3u64, vec![0.05, 0.05, 0.9, 0.0, 0.0]),
        (4u64, vec![0.0, 0.0, 0.0, 0.0, 0.0]), // no position this epoch
    ];
    let symbolic = disk.classify_trajectory(16, &samples);
    assert_eq!(
        symbolic,
        vec![
            (1, "/trauma".to_string()),
            (2, "/trauma".to_string()),
            (3, "/systemisch".to_string()),
        ],
        "the moving point's meaning-states, in causal order"
    );
}

#[test]
fn explicit_weight_queries_work() {
    let store = Store::new();
    let disk = SemanticDisk::build(&spec()).unwrap();
    store.put_data_only("/c", b"parent").unwrap();
    add(&store, "/c/pure_trauma", &[1.0, 0.0, 0.0, 0.0, 0.0]);
    add(&store, "/c/pure_cgt", &[0.0, 1.0, 0.0, 0.0, 0.0]);

    // Weights are positional in concepts() (sorted) order — look the
    // index up instead of assuming spec order.
    let concepts = disk.concepts();
    let trauma_idx = concepts.iter().position(|c| *c == "/trauma").unwrap();
    let mut w = vec![0.0; concepts.len()];
    w[trauma_idx] = 1.0;
    let hits = disk.nearest_to_weights(&store, &w, 1).unwrap();
    assert_eq!(hits[0].0, "/c/pure_trauma");

    assert!(disk.nearest_to_weights(&store, &[1.0], 1).is_err(), "wrong arity");
    assert!(
        disk.nearest_to_weights(&store, &[0.0; 5], 1).is_err(),
        "all-zero weights have no position"
    );
}
