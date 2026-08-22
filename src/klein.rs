//! klein.rs — Klein Projective Model + Nielsen Power Diagram
//!
//! Implements the Klein model of hyperbolic space and the Nielsen reduction
//! (hyperbolic Voronoi = Euclidean power diagram).
//!
//! One limit is load-bearing for callers of this module:
//! [`power_distance`] is the *Euclidean* power distance. It coincides with the
//! hyperbolic Voronoi diagram only when every site shares one Klein norm; for
//! sites at differing norms the exact reduction is
//! `argmin_i (1 - <x, k_i>) * gamma_i` (see `semantic_disk::classify_point`).
//!
//! Spatial queries do not go through this module. They are answered by
//! `cell_index`, which works in the Poincaré disk and decides with exact
//! hyperbolic distance. An earlier uniform grid over this model held one owner
//! per tile and was removed in 0.6.0: Sarkar placement drives cells below tile
//! size within a few levels, so most nodes owned no tile at any affordable
//! resolution.
//!
//! Mathematical foundation:
//! - Klein map: x_K = 2·x_P / (1 + ||x_P||²)
//! - Inverse:   x_P = x_K / (1 + √(1 - ||x_K||²))
//! - Power weight: w_i = 1 - ||x_K_i||²
//! - Power distance: pd(q, p_i) = ||q - p_i||² - w_i
//! - Nielsen 2009: hyperbolic Voronoi cells = Euclidean power cells in Klein model

use g_math::fixed_point::{FixedPoint, FixedVector};
use crate::constants;
use crate::hyperbolic_geometry::HyperbolicPoint;

// ---------------------------------------------------------------------------
// KleinPoint
// ---------------------------------------------------------------------------

/// A point in the Klein projective model of hyperbolic space.
///
/// In the Klein model, geodesics are Euclidean straight lines (chords),
/// which makes power diagram bisectors into Euclidean hyperplanes.
#[derive(Clone, Debug)]
pub struct KleinPoint {
    /// Klein disk coordinates (||coords|| < 1)
    pub coords: FixedVector,
    /// Power weight: w = 1 - ||coords||²
    pub weight: FixedPoint,
}

impl KleinPoint {
    /// Create a KleinPoint from raw coordinates, computing the weight.
    pub fn new(coords: FixedVector) -> Self {
        let weight = FixedPoint::from_int(1) - coords.length_squared();
        Self { coords, weight }
    }

    /// Dimension of the point.
    pub fn dimension(&self) -> usize {
        self.coords.len()
    }
}

// ---------------------------------------------------------------------------
// Poincaré ↔ Klein conversions
// ---------------------------------------------------------------------------

/// Convert a Poincaré disk point to a Klein disk point.
///
/// Formula: x_K = 2·x_P / (1 + ||x_P||²)
/// Weight:  w = 1 - ||x_K||²
pub fn poincare_to_klein(p: &HyperbolicPoint) -> KleinPoint {
    let dim = p.dimension();
    let norm_sq = p.coords().length_squared();
    let one = FixedPoint::from_int(1);
    let two = FixedPoint::from_int(2);

    let denom = one + norm_sq; // 1 + ||x_P||²
    let scale = two / denom;   // 2 / (1 + ||x_P||²)

    let mut klein_coords = FixedVector::new(dim);
    for i in 0..dim {
        klein_coords[i] = p.coords()[i] * scale;
    }

    KleinPoint::new(klein_coords)
}

/// Convert a Klein disk point back to a Poincaré disk point.
///
/// Formula: x_P = x_K / (1 + √(1 - ||x_K||²))
pub fn klein_to_poincare(k: &KleinPoint) -> HyperbolicPoint {
    let dim = k.dimension();
    let one = FixedPoint::from_int(1);
    let norm_sq = k.coords.length_squared();

    // Handle origin specially
    if norm_sq < constants::small_epsilon() {
        return HyperbolicPoint::origin(dim);
    }

    let sqrt_term = (one - norm_sq).sqrt(); // √(1 - ||x_K||²)
    let denom = one + sqrt_term;
    let inv_denom = one / denom;

    let mut poincare_coords = FixedVector::new(dim);
    for i in 0..dim {
        poincare_coords[i] = k.coords[i] * inv_denom;
    }

    HyperbolicPoint::new(poincare_coords)
}

// ---------------------------------------------------------------------------
// Weighted barycenter (Einstein midpoint) — semantic disk
// ---------------------------------------------------------------------------

/// Weighted hyperbolic barycenter of Klein-model sites (Einstein midpoint):
///
/// ```text
/// m = Σ wᵢ·γᵢ·kᵢ / Σ wᵢ·γᵢ,   γᵢ = 1/√(1 − ||kᵢ||²)
/// ```
///
/// Note `1 − ||kᵢ||²` is exactly [`KleinPoint::weight`], so γᵢ = 1/√weightᵢ.
/// The result is a convex combination of points strictly inside the unit
/// ball, hence itself strictly inside — no clamping needed.
///
/// Sites with non-positive weight are ignored. Returns `None` when no site
/// carries positive weight (the caller's "no concept position" case).
/// Scale-invariant in the weights: `(w₁..wₙ)` and `(c·w₁..c·wₙ)` produce the
/// same point. Deterministic: fixed-point arithmetic, input-order defined
/// accumulation (callers pass sites in a canonical order).
pub fn weighted_barycenter(sites: &[(KleinPoint, FixedPoint)]) -> Option<KleinPoint> {
    let zero = FixedPoint::from_int(0);
    let one = FixedPoint::from_int(1);

    let mut dim = 0;
    let mut denom = zero;
    let mut numer: Option<FixedVector> = None;

    for (site, w) in sites {
        if *w <= zero {
            continue;
        }
        // γ = 1/√(1 − ||k||²); clamp the radicand away from zero so
        // boundary-adjacent sites yield a large-but-finite factor instead
        // of a division blow-up.
        let radicand = if site.weight > constants::small_epsilon() {
            site.weight
        } else {
            constants::small_epsilon()
        };
        let gamma = one / radicand.sqrt();
        let coeff = *w * gamma;

        if numer.is_none() {
            dim = site.dimension();
            numer = Some(FixedVector::new(dim));
        }
        let acc = numer.as_mut().unwrap();
        for i in 0..dim {
            acc[i] += site.coords[i] * coeff;
        }
        denom += coeff;
    }

    let numer = numer?;
    if denom <= zero {
        return None;
    }
    let inv = one / denom;
    let mut coords = FixedVector::new(dim);
    for i in 0..dim {
        coords[i] = numer[i] * inv;
    }
    Some(KleinPoint::new(coords))
}

// ---------------------------------------------------------------------------
// Power distance
// ---------------------------------------------------------------------------

/// Compute the power distance from a query point (in Klein coords) to a site.
///
/// pd(q, p_i) = ||q - p_i||² - w_i
///
/// The nearest neighbor in hyperbolic Voronoi = argmin of power distance.
pub fn power_distance(query: &FixedVector, site: &KleinPoint) -> FixedPoint {
    let dim = query.len();
    assert_eq!(dim, site.dimension(), "Dimension mismatch");

    // Deliberately storage-tier (fused-kernel adoption evaluated 2026-07-11 and
    // rejected by measurement: 23.8 → 75.9 ns, 3.2×): Klein-interior
    // inputs (|k| < 1, d ≤ 4) bound the accumulator below 4 — wrap is
    // impossible — and this is the O(1) grid/nearest hot path where the
    // fused kernel's upscale/downscale overhead dominates.
    let mut dist_sq = FixedPoint::from_int(0);
    for i in 0..dim {
        let d = query[i] - site.coords[i];
        dist_sq = dist_sq + d * d;
    }

    dist_sq - site.weight
}

/// Find the nearest neighbor by minimum power distance (brute force).
///
/// Returns (index, power_distance) of the nearest site.
pub fn nearest_by_power_distance(query: &FixedVector, sites: &[KleinPoint]) -> Option<(usize, FixedPoint)> {
    if sites.is_empty() {
        return None;
    }

    let mut best_idx = 0;
    let mut best_pd = power_distance(query, &sites[0]);

    for (i, site) in sites.iter().enumerate().skip(1) {
        let pd = power_distance(query, site);
        if pd < best_pd {
            best_pd = pd;
            best_idx = i;
        }
    }

    Some((best_idx, best_pd))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;

    fn fp(v: i32) -> FixedPoint {
        FixedPoint::from_int(v)
    }

    fn fp_approx_eq(a: FixedPoint, b: FixedPoint, tol: FixedPoint) -> bool {
        (a - b).abs() < tol
    }

    // ---- Weighted barycenter (semantic disk) ----

    fn klein_at(x: f32, y: f32) -> KleinPoint {
        poincare_to_klein(&HyperbolicPoint::from_f32_slice(&[x, y]))
    }

    #[test]
    fn barycenter_single_site_is_identity() {
        let site = klein_at(0.4, -0.2);
        let m = weighted_barycenter(&[(site.clone(), fp(3))]).unwrap();
        assert!(fp_approx_eq(m.coords[0], site.coords[0], constants::epsilon()));
        assert!(fp_approx_eq(m.coords[1], site.coords[1], constants::epsilon()));
    }

    #[test]
    fn barycenter_equal_weights_matches_verified_midpoint() {
        // Two sites, equal weights: the Einstein midpoint must agree with
        // the independently verified gyro hyperbolic_midpoint (independent oracle).
        let pa = HyperbolicPoint::from_f32_slice(&[0.5, 0.1]);
        let pb = HyperbolicPoint::from_f32_slice(&[-0.2, 0.4]);
        let expected = pa.hyperbolic_midpoint(&pb);

        let m = weighted_barycenter(&[
            (poincare_to_klein(&pa), fp(1)),
            (poincare_to_klein(&pb), fp(1)),
        ])
        .unwrap();
        let got = klein_to_poincare(&m);

        let tol = FixedPoint::from_int(1) / FixedPoint::from_int(1000);
        assert!(
            fp_approx_eq(got.coords()[0], expected.coords()[0], tol)
                && fp_approx_eq(got.coords()[1], expected.coords()[1], tol),
            "einstein midpoint {:?} != gyro midpoint {:?}",
            got, expected
        );
    }

    #[test]
    fn barycenter_is_weight_scale_invariant() {
        let sites = [klein_at(0.3, 0.3), klein_at(-0.4, 0.1), klein_at(0.0, -0.5)];
        let a = weighted_barycenter(&[
            (sites[0].clone(), fp(1)),
            (sites[1].clone(), fp(2)),
            (sites[2].clone(), fp(3)),
        ])
        .unwrap();
        let b = weighted_barycenter(&[
            (sites[0].clone(), fp(7)),
            (sites[1].clone(), fp(14)),
            (sites[2].clone(), fp(21)),
        ])
        .unwrap();
        let tol = FixedPoint::from_int(1) / FixedPoint::from_int(100000);
        assert!(fp_approx_eq(a.coords[0], b.coords[0], tol));
        assert!(fp_approx_eq(a.coords[1], b.coords[1], tol));
    }

    #[test]
    fn barycenter_stays_inside_disk_and_handles_zero_weights() {
        // Skewed weights over spread-out sites: result strictly inside.
        let m = weighted_barycenter(&[
            (klein_at(0.9, 0.0), fp(100)),
            (klein_at(-0.9, 0.0), fp(1)),
        ])
        .unwrap();
        assert!(m.coords.length_squared() < FixedPoint::from_int(1));

        // Non-positive weights are ignored; all-zero → None.
        assert!(weighted_barycenter(&[(klein_at(0.5, 0.0), fp(0))]).is_none());
        assert!(weighted_barycenter(&[]).is_none());
        let only_positive = weighted_barycenter(&[
            (klein_at(0.5, 0.0), fp(0)),
            (klein_at(0.2, 0.2), fp(1)),
            (klein_at(0.7, 0.0), fp(-2)),
        ])
        .unwrap();
        let expected = klein_at(0.2, 0.2);
        assert!(fp_approx_eq(only_positive.coords[0], expected.coords[0], constants::epsilon()));
        assert!(fp_approx_eq(only_positive.coords[1], expected.coords[1], constants::epsilon()));
    }

    // ---- Step 1 tests: Klein conversions + power distance ----

    #[test]
    fn test_klein_origin_maps_to_origin() {
        let origin = HyperbolicPoint::origin(2);
        let k = poincare_to_klein(&origin);

        assert!(k.coords[0].abs() < constants::epsilon());
        assert!(k.coords[1].abs() < constants::epsilon());
        // Weight at origin should be 1
        assert!(fp_approx_eq(k.weight, fp(1), constants::epsilon()));
    }

    #[test]
    fn test_klein_roundtrip() {
        // P(0.5, 0) → K → P should return (0.5, 0) within ε
        let p = HyperbolicPoint::from_f32_slice(&[0.5, 0.0]);
        let k = poincare_to_klein(&p);
        let p2 = klein_to_poincare(&k);

        let tol = constants::epsilon();
        assert!(fp_approx_eq(p.coords()[0], p2.coords()[0], tol),
            "x roundtrip: {} vs {}", p.coords()[0], p2.coords()[0]);
        assert!(fp_approx_eq(p.coords()[1], p2.coords()[1], tol),
            "y roundtrip: {} vs {}", p.coords()[1], p2.coords()[1]);
    }

    #[test]
    fn test_klein_roundtrip_multiple() {
        // Test roundtrip for several points
        let test_points: Vec<[f32; 2]> = vec![
            [0.3, 0.2],
            [-0.4, 0.1],
            [0.0, 0.7],
            [0.1, -0.5],
            [0.8, 0.0],
        ];

        let tol = constants::epsilon();
        for coords in &test_points {
            let p = HyperbolicPoint::from_f32_slice(coords);
            let k = poincare_to_klein(&p);
            let p2 = klein_to_poincare(&k);

            assert!(fp_approx_eq(p.coords()[0], p2.coords()[0], tol),
                "Roundtrip failed for ({}, {})", coords[0], coords[1]);
            assert!(fp_approx_eq(p.coords()[1], p2.coords()[1], tol),
                "Roundtrip failed for ({}, {})", coords[0], coords[1]);
        }
    }

    #[test]
    fn test_klein_known_example() {
        // P(0.5, 0) → K should give (0.8, 0), w = 0.36
        // 2·0.5/(1+0.25) = 1.0/1.25 = 0.8
        // w = 1 - 0.64 = 0.36
        let p = HyperbolicPoint::from_f32_slice(&[0.5, 0.0]);
        let k = poincare_to_klein(&p);

        let tol = FixedPoint::from_int(1) / FixedPoint::from_int(100);
        let expected_x = FixedPoint::from_int(4) / FixedPoint::from_int(5); // 0.8
        let expected_w = FixedPoint::from_int(36) / FixedPoint::from_int(100); // 0.36

        assert!(fp_approx_eq(k.coords[0], expected_x, tol),
            "Klein x: expected 0.8, got {}", k.coords[0]);
        assert!(k.coords[1].abs() < tol,
            "Klein y: expected 0, got {}", k.coords[1]);
        assert!(fp_approx_eq(k.weight, expected_w, tol),
            "Klein weight: expected 0.36, got {}", k.weight);
    }

    #[test]
    fn test_klein_boundary_behavior() {
        // As ||x_P|| → 1, ||x_K|| → 1
        let near_boundary = HyperbolicPoint::from_f32_slice(&[0.95, 0.0]);
        let k = poincare_to_klein(&near_boundary);

        // ||x_K|| should be close to 1 (and closer than ||x_P||)
        let k_norm = k.coords.length();
        assert!(k_norm > FixedPoint::from_int(9) / FixedPoint::from_int(10),
            "Klein norm should be near 1 for boundary point, got {}", k_norm);
        assert!(k_norm < FixedPoint::from_int(1),
            "Klein norm should be < 1, got {}", k_norm);
    }

    #[test]
    fn test_power_distance_at_site_center() {
        // pd(p_i, p_i) = ||p_i - p_i||² - w_i = -w_i = ||x_K||² - 1 < 0
        let p = HyperbolicPoint::from_f32_slice(&[0.5, 0.0]);
        let k = poincare_to_klein(&p);

        let pd = power_distance(&k.coords, &k);
        let expected = -k.weight; // = ||x_K||² - 1

        let tol = constants::epsilon();
        assert!(fp_approx_eq(pd, expected, tol),
            "Power distance at site center should be -weight: {} vs {}", pd, expected);
        assert!(pd < FixedPoint::from_int(0),
            "Power distance at own site should be negative");
    }

    #[test]
    fn test_power_distance_ordering_matches_hyperbolic() {
        // For a query point and two sites, power distance ordering should
        // match hyperbolic distance ordering
        let query_p = HyperbolicPoint::from_f32_slice(&[0.1, 0.1]);
        let site1_p = HyperbolicPoint::from_f32_slice(&[0.2, 0.0]);
        let site2_p = HyperbolicPoint::from_f32_slice(&[0.6, 0.3]);

        let query_k = poincare_to_klein(&query_p);
        let site1_k = poincare_to_klein(&site1_p);
        let site2_k = poincare_to_klein(&site2_p);

        let pd1 = power_distance(&query_k.coords, &site1_k);
        let pd2 = power_distance(&query_k.coords, &site2_k);

        let hd1 = query_p.hyperbolic_distance(&site1_p);
        let hd2 = query_p.hyperbolic_distance(&site2_p);

        // If hd1 < hd2, then pd1 should be < pd2
        if hd1 < hd2 {
            assert!(pd1 < pd2,
                "Power distance ordering should match hyperbolic: pd1={} pd2={}, hd1={} hd2={}",
                pd1, pd2, hd1, hd2);
        } else {
            assert!(pd2 <= pd1,
                "Power distance ordering should match hyperbolic: pd1={} pd2={}, hd1={} hd2={}",
                pd1, pd2, hd1, hd2);
        }
    }

    #[test]
    fn test_nearest_by_power_distance() {
        let sites = vec![
            KleinPoint::new(FixedVector::from_f32_slice(&[0.2, 0.0])),
            KleinPoint::new(FixedVector::from_f32_slice(&[0.8, 0.0])),
            KleinPoint::new(FixedVector::from_f32_slice(&[0.0, 0.5])),
        ];

        let query = FixedVector::from_f32_slice(&[0.1, 0.0]);

        let (idx, _pd) = nearest_by_power_distance(&query, &sites).unwrap();

        // Closest to (0.2, 0) should be site 0
        assert_eq!(idx, 0, "Nearest should be site 0");
    }

    #[test]
    fn test_klein_roundtrip_4d() {
        // Test roundtrip in 4D (default HTT dimension)
        let p = HyperbolicPoint::from_f32_slice(&[0.3, 0.2, -0.1, 0.15]);
        let k = poincare_to_klein(&p);
        let p2 = klein_to_poincare(&k);

        let tol = constants::epsilon();
        for i in 0..4 {
            assert!(fp_approx_eq(p.coords()[i], p2.coords()[i], tol),
                "4D roundtrip failed at dim {}: {} vs {}", i, p.coords()[i], p2.coords()[i]);
        }
    }

    // ---- Step 5 validation tests ----

    #[test]
    fn test_power_distance_ordering_equidistant_sites() {
        // The Nielsen reduction gives exact Voronoi equivalence for sites at
        // equal Klein norm (common in Sarkar embeddings at same tree depth).
        // For siblings at equal distance from parent, pd ordering = d_H ordering.
        let tau = constants::default_tau();
        let half_tau = tau * constants::half();
        let r = half_tau.tanh(); // Sarkar radius in Poincaré disk

        // Create 4 equidistant siblings (children of origin at distance τ)
        let angles: Vec<FixedPoint> = vec![
            FixedPoint::from_int(0),
            FixedPoint::from_int(3) / FixedPoint::from_int(2),
            FixedPoint::from_int(3),
            FixedPoint::from_int(9) / FixedPoint::from_int(2),
        ];
        let sites_p: Vec<HyperbolicPoint> = angles.iter().map(|a| {
            let mut v = FixedVector::new(2);
            let (sin_a, cos_a) = a.sincos();
            v[0] = r * cos_a;
            v[1] = r * sin_a;
            HyperbolicPoint::new(v)
        }).collect();

        let sites_k: Vec<KleinPoint> = sites_p.iter().map(|p| poincare_to_klein(p)).collect();

        // Query points near each site — power NN should match hyperbolic NN
        for (qi, site) in sites_p.iter().enumerate() {
            // Query slightly perturbed from the site
            let mut q_coords = site.coords().clone();
            q_coords[0] = q_coords[0] + constants::epsilon();
            let q_p = HyperbolicPoint::new(q_coords.clone());
            let q_k = poincare_to_klein(&q_p);

            let (pd_nn, _) = nearest_by_power_distance(&q_k.coords, &sites_k).unwrap();

            let mut hyp_dists: Vec<(usize, FixedPoint)> = sites_p.iter().enumerate()
                .map(|(i, s)| (i, q_p.hyperbolic_distance(s)))
                .collect();
            hyp_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            assert_eq!(pd_nn, hyp_dists[0].0,
                "Power NN should match hyperbolic NN near site {}", qi);
        }
    }
}
