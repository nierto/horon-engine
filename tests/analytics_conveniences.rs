//! Regression tests: `find_similar` and `find_outliers`.
//!
//! `find_similar` must equal `neighbors_semantic` exactly. `find_outliers`
//! must flag a planted anomaly, respect the prefix population boundary,
//! refuse nonsense thresholds, and stay silent on tiny or uniform
//! populations.

use horon_engine::{SemanticOutlier, Store};
use g_math::fixed_point::FixedPoint;

/// Encode values for dims 16.. as raw Q64.64 bytes (dims 0..16 zeroed).
fn coords(vals: &[f64]) -> Vec<u8> {
    let mut out = vec![0u8; 16 * 16];
    for &v in vals {
        out.extend_from_slice(&FixedPoint::from_f64(v).raw().to_le_bytes());
    }
    out
}

/// A tight cluster under /courses plus one planted far-away node.
fn store_with_planted_outlier() -> Store {
    let store = Store::new();
    store.put_data_only("/courses", b"parent").unwrap();
    for i in 0..30 {
        let key = format!("/courses/c{:02}", i);
        store.put_data_only(&key, b"x").unwrap();
        // Cluster: coordinates in a small disc around (0.5, 0.5).
        let dx = (i % 5) as f64 * 0.01;
        let dy = (i / 5) as f64 * 0.01;
        store.set_semantic(&key, coords(&[0.5 + dx, 0.5 + dy])).unwrap();
    }
    store.put_data_only("/courses/dementie", b"x").unwrap();
    store.set_semantic("/courses/dementie", coords(&[9.0, 9.0])).unwrap();
    store
}

#[test]
fn find_similar_equals_neighbors_semantic() {
    let store = store_with_planted_outlier();
    let a = store.find_similar("/courses/c07", 5, 16..18).unwrap();
    let b = store.neighbors_semantic("/courses/c07", 5, 16..18).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 5);
    assert!(a.iter().all(|(k, _)| k != "/courses/c07"), "self must be excluded");
}

#[test]
fn find_outliers_flags_the_planted_anomaly() {
    let store = store_with_planted_outlier();
    let outliers = store.find_outliers("/courses", FixedPoint::from_f64(2.0), 16..18).unwrap();

    assert!(!outliers.is_empty(), "the planted outlier must be found");
    let top: &SemanticOutlier = &outliers[0];
    assert_eq!(top.key, "/courses/dementie");
    assert!(top.z_score.to_f64() > 2.0);
    assert!(
        top.nearest_peer.starts_with("/courses/c"),
        "nearest peer must come from the cluster: {}",
        top.nearest_peer
    );
    assert!(top.nearest_distance.to_f64() > 5.0, "even its best match is far away");
    // No cluster member should be flagged at this threshold.
    assert!(
        outliers.iter().all(|o| o.key == "/courses/dementie"),
        "cluster members wrongly flagged: {:?}",
        outliers
    );
}

#[test]
fn outlier_statistics_respect_the_prefix_boundary() {
    let store = store_with_planted_outlier();
    // A second population, far from /courses in coordinate space. If the
    // prefix boundary leaked, these would dominate every /courses baseline.
    store.put_data_only("/venues", b"parent").unwrap();
    for i in 0..20 {
        let key = format!("/venues/v{:02}", i);
        store.put_data_only(&key, b"x").unwrap();
        store.set_semantic(&key, coords(&[100.0 + i as f64, 100.0])).unwrap();
    }

    // /courses analysis is unchanged by the /venues population.
    let outliers = store.find_outliers("/courses", FixedPoint::from_f64(2.0), 16..18).unwrap();
    assert_eq!(outliers.len(), 1);
    assert_eq!(outliers[0].key, "/courses/dementie");
    assert!(
        outliers[0].nearest_peer.starts_with("/courses/"),
        "peer from outside the prefix population: {}",
        outliers[0].nearest_peer
    );
}

#[test]
fn small_and_uniform_populations_yield_no_outliers() {
    let store = Store::new();
    store.put_data_only("/few", b"parent").unwrap();
    for i in 0..3 {
        let key = format!("/few/n{}", i);
        store.put_data_only(&key, b"x").unwrap();
        store.set_semantic(&key, coords(&[i as f64 * 50.0, 0.0])).unwrap();
    }
    // Population 3 < OUTLIER_MIN_POPULATION: silence, even with a spread.
    assert!(store.find_outliers("/few", FixedPoint::from_f64(2.0), 16..18).unwrap().is_empty());

    // Perfectly uniform population: stdev 0 → no outliers, no NaN.
    store.put_data_only("/same", b"parent").unwrap();
    for i in 0..10 {
        let key = format!("/same/n{}", i);
        store.put_data_only(&key, b"x").unwrap();
        store.set_semantic(&key, coords(&[1.0, 1.0])).unwrap();
    }
    assert!(store.find_outliers("/same", FixedPoint::from_f64(2.0), 16..18).unwrap().is_empty());
}

#[test]
fn find_outliers_rejects_bad_thresholds() {
    let store = store_with_planted_outlier();
    // Q64.64 has no NaN, so the old NaN case is unrepresentable by
    // construction — the remaining rejections are zero and negatives,
    // including the smallest negative the type can express.
    assert!(store.find_outliers("/courses", FixedPoint::from_int(0), 16..18).is_err());
    assert!(store.find_outliers("/courses", FixedPoint::from_int(-1), 16..18).is_err());
    assert!(store.find_outliers("/courses", FixedPoint::from_raw(-1), 16..18).is_err());
}
