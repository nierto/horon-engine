//! hyperbolic_geometry.rs - Poincaré Disk Model Implementation for horon-engine
//! # Hyperbolic Geometry Implementation
//!
//! This module implements the Poincaré disk model of hyperbolic geometry,
//! providing the mathematical foundation for Hyperbolic Tree Tensors (HTT).
//!
//! ## Key Features:
//!
//! - **Poincaré Disk Model**: Represents hyperbolic space within a unit disk
//! - **Fixed-Point Arithmetic**: Ensures deterministic results across platforms
//! - **Möbius Transformations**: Efficient operations for manipulating points
//! - **Hyperbolic Distance Calculations**: Accurate measurement in hyperbolic space
//!
//! The Poincaré disk is ideal for representing hierarchical tree structures
//! because it visually emphasizes the exponential growth characteristic of
//! hyperbolic space, making it perfect for HTT's hierarchical representations.

use std::fmt::{self, Debug, Formatter};
use g_math::fixed_point::{FixedPoint, FixedVector};
use crate::constants;

/// A point in the Poincaré disk model of hyperbolic space.
///
/// The Poincaré disk model represents hyperbolic space within a unit disk,
/// where geodesics are represented by arcs of circles orthogonal to the boundary,
/// and hyperbolic distance is distorted in a way that makes the model ideal
/// for representing hierarchical tree structures.
#[derive(Clone)]
pub struct HyperbolicPoint {
    /// Coordinates in the Poincaré disk using fixed-point representation
    /// The disk has radius 1, so all valid points must have norm < 1
    coords: FixedVector,
}

impl HyperbolicPoint {
    /// Create a new point in the Poincaré disk with the given coordinates.
    ///
    /// Coordinates are automatically projected into the disk if they're outside.
    pub fn new(coords: FixedVector) -> Self {
        let mut point = Self { coords };
        point.ensure_in_disk();
        point
    }

    /// Create a new point from a slice of f32 values (user-facing boundary API).
    pub fn from_f32_slice(values: &[f32]) -> Self {
        Self::new(FixedVector::from_f32_slice(values))
    }

    /// Create a point from exact fixed-point coordinates — the lossless
    /// entry point, and the one the public API uses.
    pub fn from_slice(values: &[FixedPoint]) -> Self {
        let mut v = FixedVector::new(values.len());
        for (i, &c) in values.iter().enumerate() {
            v[i] = c;
        }
        Self::new(v)
    }

    /// Create a new point at the origin of the disk.
    pub fn origin(dimension: usize) -> Self {
        Self {
            coords: FixedVector::new(dimension),
        }
    }

    /// Ensure the point is inside the Poincaré disk.
    ///
    /// Projects the point onto the disk if it's outside.
    fn ensure_in_disk(&mut self) {
        let squared_norm = self.coords.length_squared();
        let one = FixedPoint::from_int(1);

        // If point is outside the disk or very close to the boundary, project it in
        if squared_norm >= one || (one - squared_norm) < constants::boundary_margin() {
            let norm = self.coords.length_fused();
            let scale_factor = constants::near_boundary() / norm;

            for i in 0..self.coords.len() {
                self.coords[i] = self.coords[i] * scale_factor;
            }
        }
    }

    /// Get a reference to the underlying coordinates.
    pub fn coords(&self) -> &FixedVector {
        &self.coords
    }

    /// Get a mutable reference to the underlying coordinates.
    pub fn coords_mut(&mut self) -> &mut FixedVector {
        &mut self.coords
    }

    /// Get the dimension of the point.
    pub fn dimension(&self) -> usize {
        self.coords.len()
    }

    /// Calculate the Euclidean norm of the point.
    pub fn euclidean_norm(&self) -> FixedPoint {
        self.coords.length_fused()
    }

    /// Calculate the hyperbolic distance between this point and another.
    ///
    /// The hyperbolic distance in the Poincaré disk model is given by:
    /// d(p, q) = 2 * atanh(|p-q| / |1-p̄q|)
    /// where p̄ is the complex conjugate of p and |x| is the Euclidean norm.
    ///
    /// Computed in squared space (one-sqrt form): `r = √(|p−q|² / |1−p̄q|²)` with
    /// `|p−q|² = |p|² + |q|² − 2⟨p,q⟩`, so the kernel pays **one** sqrt
    /// instead of the previous four (|p−q|, |p|, |q|, and the denominator —
    /// two of which were norms immediately squared back). Measured: the
    /// four-sqrt form cost ~78 µs/pair; fixed-point sqrt is ~15 µs each.
    /// Same guards, same clamps, same saturation semantics; outputs may
    /// differ from the old form in the last ULPs (different-but-still-
    /// deterministic rounding sequence).
    pub fn hyperbolic_distance(&self, other: &Self) -> FixedPoint {
        ratio_to_distance(self.hyperbolic_ratio(other))
    }

    /// Compute the Möbius ratio |p-q| / |1-p̄q| without the atanh transcendental.
    ///
    /// Since atanh is strictly monotonic on [0,1), comparing ratios is
    /// equivalent to comparing hyperbolic distances:
    ///   d(p,q) < d(p,r)  ⟺  ratio(p,q) < ratio(p,r)
    ///
    /// Computed in squared space with a single sqrt (one-sqrt form):
    /// `r = √( (|p|² + |q|² − 2⟨p,q⟩) / (1 − 2⟨p,q⟩ + |p|²·|q|²) )` — one
    /// dot product, two squared norms (no norm→square round-trips), one
    /// division, one sqrt. The algebra now matches the proxy scorer
    /// (`metric_tree::HyperbolicMetric::proxy` is exactly the pre-sqrt
    /// value), so proxy and exact orderings share one computation path.
    /// Guards are the squared-space equivalents of the previous ones:
    /// origin when |p|² < small_epsilon², degenerate when den² < epsilon².
    pub fn hyperbolic_ratio(&self, other: &Self) -> FixedPoint {
        assert_eq!(self.dimension(), other.dimension(),
                  "Points must have the same dimension for ratio calculation");

        let zero = FixedPoint::from_int(0);
        let one = FixedPoint::from_int(1);
        let two = FixedPoint::from_int(2);

        let self_norm_sq = self.coords.length_squared();
        let other_norm_sq = other.coords.length_squared();

        // Origin special cases: ratio(0, q) = |q|, ratio(p, 0) = |p| —
        // one sqrt, same as the general path below.
        let eps_sq = constants::small_epsilon() * constants::small_epsilon();
        if self_norm_sq < eps_sq {
            return clamp_ratio(other_norm_sq.sqrt());
        }
        if other_norm_sq < eps_sq {
            return clamp_ratio(self_norm_sq.sqrt());
        }

        // Deliberately storage-tier with a shared dot product (fused-kernel adoption
        // evaluated 2026-07-11 and rejected by measurement: the fused
        // kernels cost +22% here — 35.4 → 43.1 µs — recomputing the dot
        // and norms per kernel at compute tier, for ULPs the interior-
        // bounded inputs (|x| < 1, sums < 4, wrap impossible) never need).
        let dot_product = self.coords.dot(&other.coords);

        // |p−q|² expanded through the dot product the denominator needs
        // anyway; rounding can nudge a coincident pair slightly negative.
        let mut dist_sq = self_norm_sq + other_norm_sq - two * dot_product;
        if dist_sq < zero {
            dist_sq = zero;
        }

        // Division safety only. For points at equal radius this term is
        // `(1 − ‖p‖²)² + ‖p − q‖²`, which shrinks quadratically with depth —
        // ordinary sibling geometry reaches 1e-9 around depth 11 and 1e-17
        // around depth 21. The threshold must therefore sit at the
        // representation floor, not at a "small number": anything higher
        // returns the saturation value for legitimate points, making every
        // node equidistant and nearest-neighbour ranking arbitrary.
        let denominator_sq = one - two * dot_product + self_norm_sq * other_norm_sq;
        if denominator_sq < constants::min_safe_denominator() {
            return constants::near_boundary();
        }

        clamp_ratio((dist_sq / denominator_sq).sqrt())
    }

    /// Apply the disk isometry that sends `a` to `b` (through the origin) to
    /// this point.
    ///
    /// Concretely this composes two gyro-translations:
    ///   `T(z) = b ⊕ ((−a) ⊕ z)`
    /// The first factor maps `a` to the origin, the second maps the origin to
    /// `b`; each is a Poincaré-disk isometry, so their composition is one too,
    /// in **any** dimension. In particular `T(a) = b` and distances are
    /// preserved: `d(T(x), T(y)) = d(x, y)`.
    ///
    /// This replaces an earlier real-scalar formula `(z − a)/(1 − z·a)` that
    /// only reduced to a valid Möbius map when the disk was treated as the 1-D
    /// complex plane; for d ≥ 2 it distorted distances. Prefer [`mobius_add`],
    /// [`reflect_to_origin`], and [`reflect_from_origin`] directly when you
    /// need just one of these factors.
    ///
    /// [`mobius_add`]: Self::mobius_add
    /// [`reflect_to_origin`]: Self::reflect_to_origin
    /// [`reflect_from_origin`]: Self::reflect_from_origin
    pub fn mobius_transform(&self, a: &Self, b: &Self) -> Self {
        // z ↦ (−a) ⊕ z  (maps a to origin), then w ↦ b ⊕ w (maps origin to b).
        let centered = self.reflect_to_origin(a);
        centered.reflect_from_origin(b)
    }

    /// Möbius addition in the Poincaré ball model: a ⊕ z.
    ///
    /// The correct d-dimensional formula (Ungar's gyroaddition):
    ///   a ⊕ z = ((1 + 2⟨a,z⟩ + ‖z‖²)·a + (1 − ‖a‖²)·z) / (1 + 2⟨a,z⟩ + ‖a‖²·‖z‖²)
    ///
    /// Key properties:
    /// - Left identity: 0 ⊕ z = z
    /// - Right identity: a ⊕ 0 = a
    /// - Left inverse: (−a) ⊕ a = 0
    /// - Left cancellation: (−a) ⊕ (a ⊕ b) = b
    /// - Isometry: d(a⊕x, a⊕y) = d(x, y)
    pub fn mobius_add(a: &Self, z: &Self) -> Self {
        let dimension = a.dimension();
        assert_eq!(dimension, z.dimension(), "Points must have the same dimension");

        let a_dot_z = a.coords.dot(&z.coords);
        let z_norm_sq = z.coords.length_squared();
        let a_norm_sq = a.coords.length_squared();

        let one = FixedPoint::from_int(1);
        let two = FixedPoint::from_int(2);

        let coeff_a = one + two * a_dot_z + z_norm_sq;
        let coeff_z = one - a_norm_sq;
        let denom = one + two * a_dot_z + a_norm_sq * z_norm_sq;

        if denom.abs() < constants::epsilon() {
            return Self::origin(dimension);
        }

        let inv_denom = one / denom;
        let mut result = FixedVector::new(dimension);
        for i in 0..dimension {
            result[i] = (coeff_a * a.coords[i] + coeff_z * z.coords[i]) * inv_denom;
        }

        Self::new(result)
    }

    /// Reflect a point to the origin frame via Möbius addition: (−center) ⊕ self.
    /// Maps center to the origin while preserving hyperbolic distances.
    pub fn reflect_to_origin(&self, center: &Self) -> Self {
        let dimension = center.dimension();
        let mut neg_coords = FixedVector::new(dimension);
        for i in 0..dimension {
            neg_coords[i] = -center.coords[i];
        }
        // neg_center has same norm as center (already in disk); skip ensure_in_disk
        let neg_center = Self { coords: neg_coords };
        Self::mobius_add(&neg_center, self)
    }

    /// Reflect a point from the origin frame to center's frame: center ⊕ self.
    /// Maps the origin to center's position while preserving hyperbolic distances.
    pub fn reflect_from_origin(&self, center: &Self) -> Self {
        Self::mobius_add(center, self)
    }

    /// Calculate the hyperbolic midpoint between this point and another.
    pub fn hyperbolic_midpoint(&self, other: &Self) -> Self {
        let dimension = self.dimension();
        assert_eq!(dimension, other.dimension(), "Points must have the same dimension");

        // Shortcut: if one point is the origin, the midpoint lies on the line to the other point
        // In the Poincaré disk: midpoint at |m| = tanh(atanh(|p|)/2) in direction of p
        if self.euclidean_norm() < constants::small_epsilon() {
            let r = other.euclidean_norm();
            if r < constants::small_epsilon() {
                return Self::origin(dimension);
            }
            let half_hyp_dist = constants::safe_atanh(r) / FixedPoint::from_int(2);
            let m_norm = half_hyp_dist.tanh();
            let scale = m_norm / r;
            let mut midpoint = FixedVector::new(dimension);
            for i in 0..dimension {
                midpoint[i] = other.coords[i] * scale;
            }
            return Self::new(midpoint);
        }

        if other.euclidean_norm() < constants::small_epsilon() {
            let r = self.euclidean_norm();
            if r < constants::small_epsilon() {
                return Self::origin(dimension);
            }
            let half_hyp_dist = constants::safe_atanh(r) / FixedPoint::from_int(2);
            let m_norm = half_hyp_dist.tanh();
            let scale = m_norm / r;
            let mut midpoint = FixedVector::new(dimension);
            for i in 0..dimension {
                midpoint[i] = self.coords[i] * scale;
            }
            return Self::new(midpoint);
        }

        // General case: follow the geodesic between the two points using the
        // gyrovector reflection pair (Ungar), which is a true isometry in any
        // dimension — unlike `mobius_transform`, whose real-scalar denominator
        // `1 − z·a` is only the 1-D complex Möbius map and distorts distances
        // for d ≥ 2.
        //
        // 1. Reflect `other` into the frame where `self` is the origin:
        //        q0 = (−self) ⊕ other
        // 2. In that frame the midpoint is radial: the point at half the
        //    hyperbolic distance to q0, i.e. |m0| = tanh(atanh(|q0|)/2) along
        //    the direction of q0.
        // 3. Reflect the radial midpoint back out of `self`'s frame:
        //        m = self ⊕ m0
        let q0 = other.reflect_to_origin(self);

        let r = q0.euclidean_norm();
        if r < constants::small_epsilon() {
            // The points coincide — the midpoint is the point itself.
            return self.clone();
        }

        let half_hyp_dist = constants::safe_atanh(r) / FixedPoint::from_int(2);
        let m0_norm = half_hyp_dist.tanh();
        let scale = m0_norm / r;

        let mut m0 = FixedVector::new(dimension);
        for i in 0..dimension {
            m0[i] = q0.coords[i] * scale;
        }

        Self::new(m0).reflect_from_origin(self)
    }

    /// Create a point at a specified distance and direction from this point.
    ///
    /// The direction is specified as a Euclidean vector that gets normalized.
    pub fn point_at_distance(&self, direction: &FixedVector, distance: FixedPoint) -> Self {
        let dimension = self.dimension();
        assert_eq!(dimension, direction.len(), "Direction vector must have the same dimension");

        // Normalize direction vector
        let mut normalized = direction.clone();
        normalized.normalize();

        // Special case: starting from origin
        if self.euclidean_norm() < constants::epsilon() {
            // From the origin: |p| = tanh(d/2)
            let half_dist = distance / FixedPoint::from_int(2);
            let tanh_half_dist = half_dist.tanh();

            let mut new_coords = FixedVector::new(dimension);
            for i in 0..dimension {
                new_coords[i] = normalized[i] * tanh_half_dist;
            }

            return Self::new(new_coords);
        }

        // General case: Möbius-reflect from self to origin, place point, reflect back
        let _origin = Self::origin(dimension);

        let half_dist = distance / FixedPoint::from_int(2);
        let tanh_half_dist = half_dist.tanh();

        let mut new_coords = FixedVector::new(dimension);
        for i in 0..dimension {
            new_coords[i] = normalized[i] * tanh_half_dist;
        }

        let new_point = Self::new(new_coords);

        // Möbius-reflect from origin frame back to self's position
        new_point.reflect_from_origin(self)
    }
}

impl Debug for HyperbolicPoint {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "HyperbolicPoint(dim={}, norm={})",
               self.dimension(), self.euclidean_norm())
    }
}

/// Convert a hyperbolic distance to its corresponding ratio threshold.
///
/// ratio = tanh(distance / 2)
///
/// Useful for converting a radius value to a ratio for comparison
/// with `hyperbolic_ratio` results.
pub fn distance_to_ratio(distance: FixedPoint) -> FixedPoint {
    let half = constants::half();
    (distance * half).tanh()
}

/// Clamp a Möbius ratio at the boundary safety margin (boundary saturation).
#[inline]
fn clamp_ratio(ratio: FixedPoint) -> FixedPoint {
    if ratio > constants::near_boundary() {
        constants::near_boundary()
    } else {
        ratio
    }
}

/// Convert a Möbius ratio back to its hyperbolic distance.
///
/// Inverse of [`distance_to_ratio`]: `distance = 2 · atanh(ratio)`. Since
/// `hyperbolic_ratio` and `hyperbolic_distance` share the same ratio term,
/// this recovers the exact distance a `hyperbolic_ratio` result stands for —
/// no second point lookup or full distance recomputation needed. As of the one-sqrt kernel
/// it IS the exact kernel's final step: `hyperbolic_distance ≡
/// ratio_to_distance(hyperbolic_ratio)` by construction.
pub fn ratio_to_distance(ratio: FixedPoint) -> FixedPoint {
    FixedPoint::from_int(2) * constants::safe_atanh(ratio)
}

/// The Poincaré disk model of hyperbolic space.
///
/// This structure represents the hyperbolic space itself and provides
/// operations for working with hyperbolic points.
#[derive(Clone, Debug)]
pub struct PoincareDisk {
    /// Dimension of the hyperbolic space
    dimension: usize,
    /// Curvature of the hyperbolic space (standard Poincaré disk has -1)
    curvature: FixedPoint,
}

impl PoincareDisk {
    /// Create a new Poincaré disk with the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            curvature: FixedPoint::from_int(-1),
        }
    }

    /// Get the dimension of the hyperbolic space.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the curvature of the hyperbolic space.
    pub fn curvature(&self) -> FixedPoint {
        self.curvature
    }

    /// Create a point at the origin of the disk.
    pub fn origin(&self) -> HyperbolicPoint {
        HyperbolicPoint::origin(self.dimension)
    }

    /// Create a new point from Euclidean coordinates, projecting to the disk.
    pub fn point_from_euclidean(&self, coords: FixedVector) -> HyperbolicPoint {
        assert_eq!(coords.len(), self.dimension, "Coordinates must match disk dimension");
        HyperbolicPoint::new(coords)
    }

    /// Create a new point from FixedPoint coordinates, projecting to the disk.
    pub fn point_from_coords(&self, coords: FixedVector) -> HyperbolicPoint {
        assert_eq!(coords.len(), self.dimension, "Coordinates must match disk dimension");
        HyperbolicPoint::new(coords)
    }

    /// Create a new point from f32 coordinates, projecting to the disk (boundary API).
    pub fn point_from_f32_slice(&self, coords: &[f32]) -> HyperbolicPoint {
        assert_eq!(coords.len(), self.dimension, "Coordinates must match disk dimension");
        HyperbolicPoint::from_f32_slice(coords)
    }

    /// Project a Euclidean point to the hyperbolic space.
    pub fn project(&self, point: &FixedVector) -> HyperbolicPoint {
        self.point_from_euclidean(point.clone())
    }

    /// Compute the hyperbolic distance between two points.
    pub fn distance(&self, p1: &HyperbolicPoint, p2: &HyperbolicPoint) -> FixedPoint {
        p1.hyperbolic_distance(p2)
    }

    /// Find the hyperbolic midpoint between two points.
    pub fn midpoint(&self, p1: &HyperbolicPoint, p2: &HyperbolicPoint) -> HyperbolicPoint {
        p1.hyperbolic_midpoint(p2)
    }

    /// Create a point at the specified radial distance from the origin.
    pub fn point_at_distance_from_origin(&self, direction: &FixedVector, distance: FixedPoint) -> HyperbolicPoint {
        self.origin().point_at_distance(direction, distance)
    }

    /// Create a point at the specified hyperbolic coordinates.
    ///
    /// Hyperbolic coordinates are specified as (r, θ₁, θ₂, ..., θₙ₋₁) where:
    /// - r is the hyperbolic distance from the origin
    /// - θᵢ are angular coordinates (similar to spherical coordinates)
    pub fn point_from_hyperbolic_coords(&self, r: FixedPoint, angles: &[FixedPoint]) -> HyperbolicPoint {
        assert_eq!(angles.len(), self.dimension - 1,
                  "Need exactly dimension-1 angles for hyperbolic coordinates");

        let mut coords = FixedVector::new(self.dimension);

        // Shortcut: r=0 is the origin
        if r < constants::epsilon() {
            return self.origin();
        }

        // |p| = tanh(r/2)
        let half_r = r / FixedPoint::from_int(2);
        let rho = half_r.tanh();

        // Convert from spherical to Cartesian coordinates
        let (sin_0, cos_0) = angles[0].sincos();
        coords[0] = rho * cos_0;

        let mut sin_product = sin_0;

        for i in 1..self.dimension - 1 {
            let (sin_i, cos_i) = angles[i].sincos();
            coords[i] = rho * sin_product * cos_i;
            sin_product = sin_product * sin_i;
        }

        if self.dimension > 1 {
            coords[self.dimension - 1] = rho * sin_product;
        }

        HyperbolicPoint::new(coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;

    fn fp_approx_eq(a: FixedPoint, b: FixedPoint, tolerance: FixedPoint) -> bool {
        (a - b).abs() < tolerance
    }

    #[test]
    fn test_poincare_disk_creation() {
        let disk = PoincareDisk::new(2);
        assert_eq!(disk.dimension(), 2);
        assert_eq!(disk.curvature().to_int(), -1);
    }

    #[test]
    fn test_origin_creation() {
        let disk = PoincareDisk::new(2);
        let origin = disk.origin();

        assert_eq!(origin.dimension(), 2);
        assert!(origin.euclidean_norm() < constants::epsilon());
    }

    #[test]
    fn test_point_creation() {
        let disk = PoincareDisk::new(2);
        let point = disk.point_from_f32_slice(&[0.5, 0.0]);

        assert_eq!(point.dimension(), 2);
        assert!(fp_approx_eq(point.coords()[0], constants::half(), constants::epsilon()));
        assert!(point.coords()[1].abs() < constants::epsilon());
    }

    #[test]
    fn test_boundary_projection() {
        let disk = PoincareDisk::new(2);

        // Try to create a point outside the disk
        let point = disk.point_from_f32_slice(&[1.5, 0.0]);

        // Point should be projected to inside the disk
        assert!(point.euclidean_norm() < FixedPoint::from_int(1));
    }

    #[test]
    fn test_hyperbolic_distance() {
        let disk = PoincareDisk::new(2);
        let origin = disk.origin();
        let point = disk.point_from_f32_slice(&[0.5, 0.0]);

        // Distance from origin in Poincaré disk: d(0,p) = 2*atanh(|p|)
        let expected = FixedPoint::from_int(2) * constants::safe_atanh(constants::half());
        let actual = disk.distance(&origin, &point);
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);

        assert!(fp_approx_eq(actual, expected, tolerance));
    }

    #[test]
    fn test_point_at_distance() {
        let disk = PoincareDisk::new(2);
        let origin = disk.origin();

        // Create a direction vector along the x-axis
        let direction = FixedVector::from_f32_slice(&[1.0, 0.0]);

        // Create a point at distance 1.0 from origin in the x direction
        let distance = FixedPoint::from_int(1);
        let point = origin.point_at_distance(&direction, distance);

        // Check the distance is as expected
        let actual_distance = disk.distance(&origin, &point);
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);
        assert!(fp_approx_eq(actual_distance, FixedPoint::from_int(1), tolerance));

        // Check the direction is along the x-axis
        let expected_x = (distance / FixedPoint::from_int(2)).tanh();
        assert!(fp_approx_eq(point.coords()[0], expected_x, tolerance));
        assert!(point.coords()[1].abs() < tolerance);
    }

    #[test]
    fn test_hyperbolic_midpoint() {
        let disk = PoincareDisk::new(2);
        let origin = disk.origin();
        let point = disk.point_from_f32_slice(&[0.5, 0.0]);

        let midpoint = disk.midpoint(&origin, &point);

        let d1 = disk.distance(&origin, &midpoint);
        let d2 = disk.distance(&midpoint, &point);
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);

        // The distances should be approximately equal
        assert!(fp_approx_eq(d1, d2, tolerance));

        // The sum of distances should equal the total distance
        let total_distance = disk.distance(&origin, &point);
        assert!(fp_approx_eq(d1 + d2, total_distance, tolerance));
    }

    #[test]
    fn test_hyperbolic_midpoint_general_case() {
        // Neither point is the origin — exercises the general geodesic path,
        // not the radial origin shortcut. The midpoint must be equidistant
        // from both endpoints and split the total distance exactly in half.
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);

        let cases: [(&[f32], &[f32]); 3] = [
            (&[0.5, 0.0], &[0.0, 0.5]),
            (&[0.3, 0.2], &[-0.4, 0.1]),
            (&[0.1, -0.3], &[0.25, 0.35]),
        ];

        for (pa, pb) in cases {
            let p = HyperbolicPoint::from_f32_slice(pa);
            let q = HyperbolicPoint::from_f32_slice(pb);
            let m = p.hyperbolic_midpoint(&q);

            let d_pm = p.hyperbolic_distance(&m);
            let d_mq = m.hyperbolic_distance(&q);
            let d_pq = p.hyperbolic_distance(&q);

            assert!(
                fp_approx_eq(d_pm, d_mq, tolerance),
                "midpoint not equidistant for {:?}/{:?}: d(p,m)={} d(m,q)={}",
                pa, pb, d_pm, d_mq
            );
            assert!(
                fp_approx_eq(d_pm + d_mq, d_pq, tolerance),
                "midpoint off the geodesic for {:?}/{:?}: d(p,m)+d(m,q)={} vs d(p,q)={}",
                pa, pb, d_pm + d_mq, d_pq
            );
        }
    }

    #[test]
    fn test_hyperbolic_midpoint_4d_general_case() {
        // Same invariants in 4D (the default HTT dimension), where the old
        // real-scalar Möbius formula was most wrong.
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);
        let p = HyperbolicPoint::from_f32_slice(&[0.3, 0.2, -0.1, 0.15]);
        let q = HyperbolicPoint::from_f32_slice(&[-0.2, 0.1, 0.25, -0.05]);
        let m = p.hyperbolic_midpoint(&q);

        let d_pm = p.hyperbolic_distance(&m);
        let d_mq = m.hyperbolic_distance(&q);
        let d_pq = p.hyperbolic_distance(&q);
        assert!(fp_approx_eq(d_pm, d_mq, tolerance),
            "4D midpoint not equidistant: {} vs {}", d_pm, d_mq);
        assert!(fp_approx_eq(d_pm + d_mq, d_pq, tolerance),
            "4D midpoint off geodesic: {} vs {}", d_pm + d_mq, d_pq);
    }

    #[test]
    fn test_mobius_transformation() {
        let disk = PoincareDisk::new(2);
        let origin = disk.origin();
        let point = disk.point_from_f32_slice(&[0.5, 0.0]);
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);

        // Identity transformation (a=0, b=0)
        let transformed = point.mobius_transform(&origin, &origin);
        assert!(fp_approx_eq(transformed.coords()[0], point.coords()[0], tolerance));
        assert!(fp_approx_eq(transformed.coords()[1], point.coords()[1], tolerance));

        let a = disk.point_from_f32_slice(&[0.3, 0.2]);
        let b = disk.point_from_f32_slice(&[-0.1, 0.4]);

        // Stays inside the disk.
        let boundary_point = disk.point_from_f32_slice(&[0.95, 0.0]);
        let transformed_boundary = boundary_point.mobius_transform(&a, &b);
        assert!(transformed_boundary.euclidean_norm() < FixedPoint::from_int(1));

        // Maps a to b.
        let a_image = a.mobius_transform(&a, &b);
        assert!(fp_approx_eq(a_image.coords()[0], b.coords()[0], tolerance));
        assert!(fp_approx_eq(a_image.coords()[1], b.coords()[1], tolerance));

        // Is an isometry: distances are preserved under the transform.
        let x = disk.point_from_f32_slice(&[0.1, -0.25]);
        let y = disk.point_from_f32_slice(&[0.4, 0.15]);
        let d_before = x.hyperbolic_distance(&y);
        let d_after = x.mobius_transform(&a, &b).hyperbolic_distance(&y.mobius_transform(&a, &b));
        assert!(
            fp_approx_eq(d_before, d_after, tolerance),
            "mobius_transform not an isometry: d_before={} d_after={}",
            d_before, d_after
        );
    }

    #[test]
    fn test_mobius_add_properties() {
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);

        // Property 1: 0 ⊕ z = z (left identity)
        let origin = HyperbolicPoint::origin(2);
        let p = HyperbolicPoint::from_f32_slice(&[0.3, 0.2]);
        let result = HyperbolicPoint::mobius_add(&origin, &p);
        assert!(fp_approx_eq(result.coords()[0], p.coords()[0], tolerance));
        assert!(fp_approx_eq(result.coords()[1], p.coords()[1], tolerance));

        // Property 2: a ⊕ 0 = a (right identity)
        let a = HyperbolicPoint::from_f32_slice(&[0.4, -0.3]);
        let result2 = HyperbolicPoint::mobius_add(&a, &origin);
        assert!(fp_approx_eq(result2.coords()[0], a.coords()[0], tolerance));
        assert!(fp_approx_eq(result2.coords()[1], a.coords()[1], tolerance));

        // Property 3: (-a) ⊕ a = 0 (left inverse)
        let neg_a = HyperbolicPoint::from_f32_slice(&[-0.4, 0.3]);
        let result3 = HyperbolicPoint::mobius_add(&neg_a, &a);
        assert!(result3.euclidean_norm() < tolerance,
            "(-a) ⊕ a should be origin, got norm {}", result3.euclidean_norm());

        // Property 4: round-trip reflect (left cancellation)
        let child = HyperbolicPoint::from_f32_slice(&[0.2, 0.1]);
        let center = HyperbolicPoint::from_f32_slice(&[0.5, 0.0]);
        let reflected = child.reflect_from_origin(&center);
        let back = reflected.reflect_to_origin(&center);
        assert!(fp_approx_eq(back.coords()[0], child.coords()[0], tolerance));
        assert!(fp_approx_eq(back.coords()[1], child.coords()[1], tolerance));

        // Property 5: isometry d(0, z) = d(a, a ⊕ z)
        let disk = PoincareDisk::new(2);
        let z = HyperbolicPoint::from_f32_slice(&[0.2, -0.1]);
        let a2 = HyperbolicPoint::from_f32_slice(&[0.3, 0.2]);
        let a_plus_z = HyperbolicPoint::mobius_add(&a2, &z);
        let d_origin_z = disk.distance(&origin, &z);
        let d_a_az = disk.distance(&a2, &a_plus_z);
        assert!(fp_approx_eq(d_origin_z, d_a_az, tolerance),
            "Isometry violated: d(0,z)={} vs d(a,a⊕z)={}", d_origin_z, d_a_az);
    }

    #[test]
    fn test_mobius_add_higher_dimensions() {
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);

        // Test in 4D — the default HTT dimension
        let origin = HyperbolicPoint::origin(4);
        let a = HyperbolicPoint::from_f32_slice(&[0.3, 0.2, -0.1, 0.15]);
        let z = HyperbolicPoint::from_f32_slice(&[0.1, -0.2, 0.15, -0.05]);

        // Round-trip: reflect out then back
        let reflected = z.reflect_from_origin(&a);
        let back = reflected.reflect_to_origin(&a);
        for i in 0..4 {
            assert!(fp_approx_eq(back.coords()[i], z.coords()[i], tolerance),
                "4D round-trip failed at dim {}: {} vs {}", i, back.coords()[i], z.coords()[i]);
        }

        // Isometry in 4D
        let a_plus_z = HyperbolicPoint::mobius_add(&a, &z);
        let d_oz = origin.hyperbolic_distance(&z);
        let d_a_az = a.hyperbolic_distance(&a_plus_z);
        assert!(fp_approx_eq(d_oz, d_a_az, tolerance),
            "4D isometry violated: d(0,z)={} vs d(a,a⊕z)={}", d_oz, d_a_az);
    }

    #[test]
    fn test_hyperbolic_coordinates() {
        let disk = PoincareDisk::new(2);

        let r = FixedPoint::from_int(1);
        let theta = FixedPoint::from_int(0); // Along the x-axis

        let point = disk.point_from_hyperbolic_coords(r, &[theta]);

        let distance = disk.distance(&disk.origin(), &point);
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);
        assert!(fp_approx_eq(distance, FixedPoint::from_int(1), tolerance));

        // Direction should be along x-axis
        assert!(point.coords()[0] > FixedPoint::from_int(0));
        assert!(point.coords()[1].abs() < tolerance);
    }

    #[test]
    fn test_ratio_ordering_matches_distance() {
        let query = HyperbolicPoint::from_f32_slice(&[0.1, 0.1, 0.0, 0.0]);
        let points = vec![
            HyperbolicPoint::from_f32_slice(&[0.2, 0.0, 0.0, 0.0]),
            HyperbolicPoint::from_f32_slice(&[0.5, 0.3, 0.0, 0.0]),
            HyperbolicPoint::from_f32_slice(&[-0.3, 0.1, 0.0, 0.0]),
            HyperbolicPoint::from_f32_slice(&[0.0, 0.6, 0.0, 0.0]),
            HyperbolicPoint::from_f32_slice(&[0.7, -0.2, 0.0, 0.0]),
        ];

        let mut by_dist: Vec<(usize, FixedPoint)> = points.iter().enumerate()
            .map(|(i, p)| (i, query.hyperbolic_distance(p)))
            .collect();
        by_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut by_ratio: Vec<(usize, FixedPoint)> = points.iter().enumerate()
            .map(|(i, p)| (i, query.hyperbolic_ratio(p)))
            .collect();
        by_ratio.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let dist_order: Vec<usize> = by_dist.iter().map(|(i, _)| *i).collect();
        let ratio_order: Vec<usize> = by_ratio.iter().map(|(i, _)| *i).collect();
        assert_eq!(dist_order, ratio_order,
            "Ratio ordering must match distance ordering");
    }

    #[test]
    fn test_ratio_origin_cases() {
        let origin = HyperbolicPoint::origin(4);
        let p = HyperbolicPoint::from_f32_slice(&[0.5, 0.3, 0.0, 0.0]);
        let tol = FixedPoint::from_int(1) / FixedPoint::from_int(1000);

        let p_norm = p.euclidean_norm();
        let r1 = origin.hyperbolic_ratio(&p);
        let r2 = p.hyperbolic_ratio(&origin);

        assert!(fp_approx_eq(r1, p_norm, tol),
            "ratio(origin, p) should equal |p|: {} vs {}", r1, p_norm);
        assert!(fp_approx_eq(r2, p_norm, tol),
            "ratio(p, origin) should equal |p|: {} vs {}", r2, p_norm);
    }

    #[test]
    fn test_ratio_symmetry() {
        let p = HyperbolicPoint::from_f32_slice(&[0.3, 0.2, -0.1, 0.0]);
        let q = HyperbolicPoint::from_f32_slice(&[0.5, -0.1, 0.2, 0.0]);
        let tol = FixedPoint::from_int(1) / FixedPoint::from_int(10000);

        let r_pq = p.hyperbolic_ratio(&q);
        let r_qp = q.hyperbolic_ratio(&p);
        assert!(fp_approx_eq(r_pq, r_qp, tol),
            "Ratio should be symmetric: {} vs {}", r_pq, r_qp);
    }

    #[test]
    fn test_ratio_self_is_zero() {
        let p = HyperbolicPoint::from_f32_slice(&[0.4, 0.3, 0.0, 0.0]);
        let r = p.hyperbolic_ratio(&p);
        assert!(r < constants::epsilon(),
            "ratio(p, p) should be ~0: got {}", r);
    }

    #[test]
    fn test_distance_to_ratio_roundtrip() {
        let p = HyperbolicPoint::from_f32_slice(&[0.3, 0.1, 0.0, 0.0]);
        let q = HyperbolicPoint::from_f32_slice(&[0.5, -0.2, 0.0, 0.0]);
        let tol = FixedPoint::from_int(1) / FixedPoint::from_int(1000);

        let dist = p.hyperbolic_distance(&q);
        let ratio_from_dist = super::distance_to_ratio(dist);
        let ratio_direct = p.hyperbolic_ratio(&q);

        assert!(fp_approx_eq(ratio_from_dist, ratio_direct, tol),
            "distance_to_ratio(d(p,q)) should equal ratio(p,q): {} vs {}",
            ratio_from_dist, ratio_direct);
    }
}
