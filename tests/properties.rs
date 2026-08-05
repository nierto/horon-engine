//! Property-based tests for the engine's geometric and store invariants.
//!
//! Run: GMATH_PROFILE=embedded cargo test --test properties

use horon_engine::{klein_to_poincare, poincare_to_klein, HyperbolicPoint, PoincareDisk, Store};
use g_math::fixed_point::FixedPoint;
use proptest::prelude::*;

/// Encode f64 values as raw Q64.64 semantic bytes (16 bytes per dimension).
fn encode(vals: &[f64]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| FixedPoint::from_f64(*v).raw().to_le_bytes())
        .collect()
}

/// Strategy: a well-inside-the-disk 2D point (|coord| ≤ 0.7 keeps the
/// Euclidean norm below the disk boundary for 2 dimensions).
fn disk_point() -> impl Strategy<Value = (f64, f64)> {
    (-0.7f64..0.7, -0.7f64..0.7)
}

/// Strategy: a path segment safe for Store keys.
fn segment() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_]{0,10}"
}

proptest! {
    /// poincare→klein→poincare is the identity up to fixed-point tolerance.
    #[test]
    fn klein_roundtrip_is_identity((x, y) in disk_point()) {
        let p = HyperbolicPoint::from_f32_slice(&[x as f32, y as f32]);
        let k = poincare_to_klein(&p);
        let back = klein_to_poincare(&k);

        let tolerance = FixedPoint::from_f64(1e-6);
        for (a, b) in p.coords().iter().zip(back.coords().iter()) {
            let diff = (*a - *b).abs();
            prop_assert!(diff < tolerance, "roundtrip drift: {} vs {}", a.to_f64(), b.to_f64());
        }
    }

    /// Hyperbolic distance is symmetric, non-negative, and zero on the diagonal.
    #[test]
    fn hyperbolic_distance_is_a_metric((x1, y1) in disk_point(), (x2, y2) in disk_point()) {
        let disk = PoincareDisk::new(2);
        let p = HyperbolicPoint::from_f32_slice(&[x1 as f32, y1 as f32]);
        let q = HyperbolicPoint::from_f32_slice(&[x2 as f32, y2 as f32]);

        let d_pq = disk.distance(&p, &q);
        let d_qp = disk.distance(&q, &p);
        let d_pp = disk.distance(&p, &p);

        let tolerance = FixedPoint::from_f64(1e-9);
        prop_assert!((d_pq - d_qp).abs() < tolerance, "asymmetric: {} vs {}", d_pq.to_f64(), d_qp.to_f64());
        prop_assert!(d_pq >= FixedPoint::from_int(0), "negative distance");
        prop_assert!(d_pp < tolerance, "d(p,p) = {}", d_pp.to_f64());
    }

    /// Semantic distance is symmetric, non-negative, zero on identical
    /// vectors, and invariant to which argument carries trailing zeros.
    #[test]
    fn semantic_distance_is_a_metric(
        a in prop::collection::vec(-100.0f64..100.0, 1..8),
        b in prop::collection::vec(-100.0f64..100.0, 1..8),
    ) {
        let dims = a.len().max(b.len());
        let ea = encode(&a);
        let eb = encode(&b);

        let d_ab = Store::semantic_distance(&ea, &eb, 0..dims);
        let d_ba = Store::semantic_distance(&eb, &ea, 0..dims);
        let d_aa = Store::semantic_distance(&ea, &ea, 0..dims);

        // Exact, not approximate: the distances are Q64.64, so the metric
        // axioms hold bit-for-bit rather than within a tolerance. The old
        // 1e-9 slack was an artifact of the API returning f64.
        prop_assert!(d_ab == d_ba, "asymmetric: {} vs {}", d_ab.to_f64(), d_ba.to_f64());
        prop_assert!(d_ab >= FixedPoint::from_int(0), "negative distance {}", d_ab.to_f64());
        prop_assert!(d_aa == FixedPoint::from_int(0), "d(a,a) = {}", d_aa.to_f64());
    }

    /// put/get roundtrip: stored bytes come back exactly; children and
    /// exists agree with what was inserted.
    #[test]
    fn put_get_roundtrip(
        segs in prop::collection::vec(segment(), 1..4),
        payload in prop::collection::vec(any::<u8>(), 0..256),
    ) {
        let store = Store::new();
        let key = format!("/{}", segs.join("/"));

        store.put(&key, &payload).unwrap();
        prop_assert!(store.exists(&key));
        prop_assert_eq!(store.get(&key).unwrap(), payload);

        // Parents are auto-created and enumerable.
        if segs.len() > 1 {
            let parent = format!("/{}", segs[..segs.len() - 1].join("/"));
            let kids = store.children(&parent).unwrap();
            prop_assert!(kids.contains(&key), "child {} missing under {}", key, parent);
        }
    }

    /// remove really removes: key gone, get errors, re-put works.
    #[test]
    fn remove_then_reput(seg in segment(), payload in prop::collection::vec(any::<u8>(), 1..64)) {
        let store = Store::new();
        let key = format!("/{}", seg);

        store.put(&key, &payload).unwrap();
        store.remove(&key).unwrap();
        prop_assert!(!store.exists(&key));
        prop_assert!(store.get(&key).is_err());

        store.put(&key, &payload).unwrap();
        prop_assert_eq!(store.get(&key).unwrap(), payload);
    }

    /// Misaligned semantic vectors are rejected, aligned ones accepted.
    #[test]
    fn set_semantic_alignment_enforced(extra in 1usize..16, dims in 1usize..4) {
        let store = Store::new();
        store.put("/n", b"x").unwrap();

        let aligned = vec![0u8; dims * 16];
        prop_assert!(store.set_semantic("/n", aligned.clone()).is_ok());

        let mut misaligned = aligned;
        misaligned.extend(std::iter::repeat(0u8).take(extra));
        prop_assert!(store.set_semantic("/n", misaligned).is_err());
    }
}
