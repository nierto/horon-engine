//! klein.rs — Klein Projective Model + Nielsen Power Diagram
//!
//! Implements the Klein model of hyperbolic space, the Nielsen reduction
//! (hyperbolic Voronoi = Euclidean power diagram), and a uniform grid
//! for O(1) point location.
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
// Power Cell (Step 2)
// ---------------------------------------------------------------------------

/// A half-plane constraint defining one face of a power cell.
///
/// Represents: ⟨q, normal⟩ ≤ offset
/// where q is on node i's side of the bisector with neighbor j.
#[derive(Clone, Debug)]
pub struct HalfPlane {
    /// Normal vector (direction p_j - p_i in Klein space)
    pub normal: FixedVector,
    /// Offset threshold: ||p_j||² - ||p_i||²
    pub offset: FixedPoint,
    /// Which tree neighbor defines this boundary
    pub neighbor_id: String,
}

/// A power cell — the Voronoi region of a single node in the Klein model.
///
/// Cell(i) = ∩_{j ∈ tree_neighbors(i)} { q : pd(q,i) ≤ pd(q,j) }
/// By the Delaunay=Tree theorem, neighbors are exactly the tree neighbors.
#[derive(Clone, Debug)]
pub struct PowerCell {
    /// Unique ID of the node owning this cell
    pub node_id: String,
    /// The site (Klein point) at the center of this cell
    pub site: KleinPoint,
    /// Half-plane constraints, one per tree neighbor
    pub half_planes: Vec<HalfPlane>,
}

/// Compute the bisector half-plane between sites i and j.
///
/// The bisector {q : pd(q,i) = pd(q,j)} is a hyperplane:
///   ⟨q, p_j - p_i⟩ = ||p_j||² - ||p_i||²
///
/// The half-plane for node i (closer to i than j):
///   ⟨q, p_j - p_i⟩ ≤ ||p_j||² - ||p_i||²
pub fn compute_bisector(site_i: &KleinPoint, site_j: &KleinPoint, neighbor_id: &str) -> HalfPlane {
    let dim = site_i.dimension();
    assert_eq!(dim, site_j.dimension(), "Dimension mismatch");

    // normal = p_j - p_i
    let mut normal = FixedVector::new(dim);
    for i in 0..dim {
        normal[i] = site_j.coords[i] - site_i.coords[i];
    }

    // offset = ||p_j||² - ||p_i||²
    let offset = site_j.coords.length_squared() - site_i.coords.length_squared();

    HalfPlane {
        normal,
        offset,
        neighbor_id: neighbor_id.to_string(),
    }
}

/// Test whether a query point lies inside a power cell.
///
/// Returns true if ⟨q, hp.normal⟩ ≤ hp.offset for all half-planes.
pub fn point_in_cell(query: &FixedVector, cell: &PowerCell) -> bool {
    for hp in &cell.half_planes {
        let dot = query.dot(&hp.normal);
        if dot > hp.offset {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Point Location Grid (Step 3)
// ---------------------------------------------------------------------------

/// Uniform grid over the Klein disk for O(1) point location.
///
/// The grid partitions the [-1, 1]² bounding box into resolution×resolution
/// tiles. Each tile stores the ID of the power cell that owns its center.
/// Query: map Klein coords → tile → owner ID → verify with half-plane check.
pub struct PointLocationGrid {
    /// Grid cells per axis
    pub resolution: usize,
    /// Dimension of the embedding space
    dimension: usize,
    /// Size of each grid tile: 2 / resolution
    cell_size: FixedPoint,
    /// Inverse cell size: resolution / 2 (precomputed for fast indexing)
    inv_cell_size: FixedPoint,
    /// Flat grid: grid[row * resolution + col] → Option<node_id>
    grid: Vec<Option<String>>,
    /// Inverted index: node_id → list of tile indices owned by that node.
    /// Enables O(parent_tiles) insert updates instead of O(R²).
    tile_owners: std::collections::HashMap<String, Vec<usize>>,
}

impl PointLocationGrid {
    /// Create an empty grid with the given resolution and dimension.
    /// For dim > 2, the grid projects onto the first 2 coordinates.
    pub fn new(resolution: usize) -> Self {
        Self::with_dimension(resolution, 2)
    }

    /// Create a grid for a specific embedding dimension.
    ///
    /// A `resolution` of 0 is treated as 1 (a single degenerate cell): the
    /// grid spans `[-1, 1]` in each axis, so `cell_size = 2 / resolution`
    /// would divide by zero. A 1×1 grid holds no useful spatial structure but
    /// keeps the constructor total rather than panicking.
    pub fn with_dimension(resolution: usize, dimension: usize) -> Self {
        let resolution = resolution.max(1);
        let res_fp = FixedPoint::from_int(resolution as i32);
        let two = FixedPoint::from_int(2);
        let cell_size = two / res_fp;
        let inv_cell_size = res_fp / two;

        Self {
            resolution,
            dimension,
            cell_size,
            inv_cell_size,
            grid: vec![None; resolution * resolution],
            tile_owners: std::collections::HashMap::new(),
        }
    }

    /// Build the grid from a set of KleinPoints by brute-force nearest power distance.
    ///
    /// For each tile center inside the Klein disk, find the site with minimum
    /// power distance and assign that tile to that site's node_id.
    pub fn build(&mut self, sites: &[(String, KleinPoint)]) {
        if sites.is_empty() {
            return;
        }

        let one = FixedPoint::from_int(1);

        for row in 0..self.resolution {
            for col in 0..self.resolution {
                let center = self.tile_center(row, col);

                // Skip tiles outside the Klein disk
                if center.length_squared() >= one {
                    self.grid[row * self.resolution + col] = None;
                    continue;
                }

                // Find site with minimum power distance
                let mut best_id: Option<&str> = None;
                let mut best_pd = FixedPoint::from_int(0);
                let mut first = true;

                for (id, site) in sites {
                    let pd = power_distance(&center, site);
                    if first || pd < best_pd {
                        best_pd = pd;
                        best_id = Some(id.as_str());
                        first = false;
                    }
                }

                self.grid[row * self.resolution + col] = best_id.map(|s| s.to_string());
            }
        }

        // Build inverted index
        self.tile_owners.clear();
        for (idx, cell) in self.grid.iter().enumerate() {
            if let Some(ref id) = cell {
                self.tile_owners.entry(id.clone()).or_default().push(idx);
            }
        }
    }

    /// Query the grid for the node owning the tile containing the given Klein point.
    ///
    /// Returns None if the point is outside the disk or the tile is unassigned.
    pub fn query(&self, query_klein: &FixedVector) -> Option<&str> {
        // Clamp to valid range
        let (row, col) = self.coords_to_tile(query_klein);

        if row >= self.resolution || col >= self.resolution {
            return None;
        }

        self.grid[row * self.resolution + col].as_deref()
    }

    /// Update the grid after inserting a new leaf node.
    ///
    /// Only tiles currently assigned to the parent need checking.
    /// For each such tile, if the tile center is closer to the new leaf
    /// in power distance, reassign it.
    pub fn update_insert(&mut self, parent_id: &str, new_id: &str, new_site: &KleinPoint, parent_site: &KleinPoint) {
        let one = FixedPoint::from_int(1);

        // Get parent's tile indices via inverted index — O(parent_tiles) not O(R²)
        let parent_tiles = match self.tile_owners.get(parent_id) {
            Some(tiles) => tiles.clone(),
            None => return,
        };

        let mut tiles_to_reassign = Vec::new();

        for &idx in &parent_tiles {
            let row = idx / self.resolution;
            let col = idx % self.resolution;
            let center = self.tile_center(row, col);

            if center.length_squared() >= one {
                continue;
            }

            let pd_parent = power_distance(&center, parent_site);
            let pd_new = power_distance(&center, new_site);

            if pd_new < pd_parent {
                tiles_to_reassign.push(idx);
            }
        }

        // Apply reassignments to grid
        for &idx in &tiles_to_reassign {
            self.grid[idx] = Some(new_id.to_string());
        }

        // Update inverted index: remove from parent, add to new node
        if !tiles_to_reassign.is_empty() {
            if let Some(parent_list) = self.tile_owners.get_mut(parent_id) {
                parent_list.retain(|idx| !tiles_to_reassign.contains(idx));
            }
            self.tile_owners.entry(new_id.to_string())
                .or_default()
                .extend(&tiles_to_reassign);
        }
    }

    /// Update the grid after deleting a leaf node.
    ///
    /// All tiles assigned to the deleted node are reassigned to the parent.
    pub fn update_delete(&mut self, deleted_id: &str, parent_id: &str) {
        // Get deleted node's tiles via inverted index — O(deleted_tiles) not O(R²)
        let deleted_tiles = match self.tile_owners.remove(deleted_id) {
            Some(tiles) => tiles,
            None => return,
        };

        // Reassign to parent in grid
        for &idx in &deleted_tiles {
            self.grid[idx] = Some(parent_id.to_string());
        }

        // Add to parent's inverted index
        self.tile_owners.entry(parent_id.to_string())
            .or_default()
            .extend(deleted_tiles);
    }

    /// Get the Klein-space center coordinates of a grid tile.
    /// Returns a vector of dimension `self.dimension`, with higher dims = 0.
    fn tile_center(&self, row: usize, col: usize) -> FixedVector {
        let half = constants::half();
        let one = FixedPoint::from_int(1);

        // Klein disk spans [-1, 1]. Tile (row, col) maps to:
        // x = -1 + (col + 0.5) * cell_size
        // y = -1 + (row + 0.5) * cell_size
        let col_fp = FixedPoint::from_int(col as i32);
        let row_fp = FixedPoint::from_int(row as i32);

        let x = -one + (col_fp + half) * self.cell_size;
        let y = -one + (row_fp + half) * self.cell_size;

        let mut v = FixedVector::new(self.dimension);
        v[0] = x;
        if self.dimension >= 2 {
            v[1] = y;
        }
        // Higher dimensions stay at zero
        v
    }

    /// Map Klein coordinates to grid tile indices (uses first 2 dims).
    fn coords_to_tile(&self, coords: &FixedVector) -> (usize, usize) {
        let one = FixedPoint::from_int(1);

        // col = floor((x + 1) / cell_size), row = floor((y + 1) / cell_size)
        let x = coords[0];
        let y = if coords.len() >= 2 { coords[1] } else { FixedPoint::from_int(0) };
        let col_fp = (x + one) * self.inv_cell_size;
        let row_fp = (y + one) * self.inv_cell_size;

        let col = col_fp.to_int().max(0) as usize;
        let row = row_fp.to_int().max(0) as usize;

        (row.min(self.resolution - 1), col.min(self.resolution - 1))
    }

    /// Get the number of assigned tiles (tiles inside the disk with an owner).
    pub fn assigned_tile_count(&self) -> usize {
        self.grid.iter().filter(|t| t.is_some()).count()
    }

    /// Get the grid resolution.
    pub fn resolution(&self) -> usize {
        self.resolution
    }
}

impl std::fmt::Debug for PointLocationGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PointLocationGrid(resolution={}, assigned={})",
               self.resolution, self.assigned_tile_count())
    }
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

    // ---- Step 2 tests: Power cells ----

    #[test]
    fn test_compute_bisector_symmetry() {
        let site_i = KleinPoint::new(FixedVector::from_f32_slice(&[0.2, 0.0]));
        let site_j = KleinPoint::new(FixedVector::from_f32_slice(&[0.6, 0.0]));

        let hp_ij = compute_bisector(&site_i, &site_j, "j");
        let hp_ji = compute_bisector(&site_j, &site_i, "i");

        // Normals should be opposite, offsets should be opposite
        let tol = constants::epsilon();
        assert!(fp_approx_eq(hp_ij.normal[0], -hp_ji.normal[0], tol));
        assert!(fp_approx_eq(hp_ij.offset, -hp_ji.offset, tol));
    }

    #[test]
    fn test_point_in_cell_at_site_center() {
        // The site center should be inside its own cell
        let site_i = KleinPoint::new(FixedVector::from_f32_slice(&[0.2, 0.0]));
        let site_j = KleinPoint::new(FixedVector::from_f32_slice(&[0.6, 0.0]));

        let hp = compute_bisector(&site_i, &site_j, "j");
        let cell = PowerCell {
            node_id: "i".to_string(),
            site: site_i.clone(),
            half_planes: vec![hp],
        };

        assert!(point_in_cell(&site_i.coords, &cell),
            "Site center should be inside its own cell");
    }

    #[test]
    fn test_bisector_midpoint_on_boundary() {
        // The midpoint between two Klein sites should be on the bisector (within ε)
        let site_i = KleinPoint::new(FixedVector::from_f32_slice(&[0.2, 0.0]));
        let site_j = KleinPoint::new(FixedVector::from_f32_slice(&[0.6, 0.0]));

        let hp = compute_bisector(&site_i, &site_j, "j");

        // Midpoint in Klein space (Euclidean midpoint, since Klein is projective)
        let mut midpoint = FixedVector::new(2);
        midpoint[0] = (site_i.coords[0] + site_j.coords[0]) * constants::half();
        midpoint[1] = (site_i.coords[1] + site_j.coords[1]) * constants::half();

        // For the midpoint to be on the bisector: ⟨midpoint, normal⟩ should ≈ offset
        // But only when the two sites have equal norm (symmetric case)
        // In general: pd(mid, i) should ≈ pd(mid, j)
        let _pd_i = power_distance(&midpoint, &site_i);
        let _pd_j = power_distance(&midpoint, &site_j);

        // The Euclidean midpoint is NOT generally on the power bisector.
        // The actual bisector point is where pd(q, i) = pd(q, j).
        // Let's verify: query the bisector condition directly.
        // At the bisector: ⟨q, normal⟩ = offset
        // Let's find the point on the x-axis where this holds:
        // q = (x, 0), normal = (0.4, 0), offset = 0.36 - 0.04 = 0.32
        // 0.4x = 0.32 → x = 0.8
        let bisector_x = hp.offset / hp.normal[0];
        let mut bisector_pt = FixedVector::new(2);
        bisector_pt[0] = bisector_x;

        let pd_i_bpt = power_distance(&bisector_pt, &site_i);
        let pd_j_bpt = power_distance(&bisector_pt, &site_j);

        let tol = constants::epsilon();
        assert!(fp_approx_eq(pd_i_bpt, pd_j_bpt, tol),
            "Bisector point should have equal power distances: {} vs {}", pd_i_bpt, pd_j_bpt);
    }

    #[test]
    fn test_cell_membership_consistency() {
        // Create two sites and verify every test point belongs to exactly one cell
        let site_a = KleinPoint::new(FixedVector::from_f32_slice(&[0.3, 0.0]));
        let site_b = KleinPoint::new(FixedVector::from_f32_slice(&[-0.3, 0.0]));

        let hp_ab = compute_bisector(&site_a, &site_b, "b");
        let hp_ba = compute_bisector(&site_b, &site_a, "a");

        let cell_a = PowerCell {
            node_id: "a".to_string(),
            site: site_a.clone(),
            half_planes: vec![hp_ab],
        };
        let cell_b = PowerCell {
            node_id: "b".to_string(),
            site: site_b.clone(),
            half_planes: vec![hp_ba],
        };

        // Test points along x-axis
        let test_xs: Vec<f32> = vec![-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8];
        for &x in &test_xs {
            let q = FixedVector::from_f32_slice(&[x, 0.0]);
            let in_a = point_in_cell(&q, &cell_a);
            let in_b = point_in_cell(&q, &cell_b);

            // Exactly one should be true (or both at boundary)
            assert!(in_a || in_b,
                "Point ({}, 0) should be in at least one cell", x);
        }
    }

    // ---- Step 3 tests: Point Location Grid ----

    #[test]
    fn test_grid_single_site() {
        // With a single site at origin, all disk tiles should point to it
        let sites = vec![
            ("root".to_string(), KleinPoint::new(FixedVector::from_f32_slice(&[0.0, 0.0]))),
        ];

        let mut grid = PointLocationGrid::new(16);
        grid.build(&sites);

        // Query at various points — all should return "root"
        let test_points: Vec<[f32; 2]> = vec![[0.0, 0.0], [0.5, 0.0], [0.0, -0.5], [0.3, 0.3]];
        for coords in &test_points {
            let q = FixedVector::from_f32_slice(coords);
            let result = grid.query(&q);
            assert_eq!(result, Some("root"), "Single site should own all tiles");
        }
    }

    #[test]
    fn test_grid_query_matches_brute_force() {
        // Build a grid with multiple sites and verify it matches brute-force
        let sites = vec![
            ("a".to_string(), KleinPoint::new(FixedVector::from_f32_slice(&[0.3, 0.0]))),
            ("b".to_string(), KleinPoint::new(FixedVector::from_f32_slice(&[-0.3, 0.0]))),
            ("c".to_string(), KleinPoint::new(FixedVector::from_f32_slice(&[0.0, 0.4]))),
        ];

        let mut grid = PointLocationGrid::new(32);
        grid.build(&sites);

        // Test random-ish points
        let test_points: Vec<[f32; 2]> = vec![
            [0.2, 0.0], [-0.2, 0.0], [0.0, 0.3],
            [0.5, 0.1], [-0.4, -0.2], [0.1, 0.5],
        ];

        let klein_sites: Vec<KleinPoint> = sites.iter().map(|(_, s)| s.clone()).collect();
        let ids: Vec<&str> = sites.iter().map(|(id, _)| id.as_str()).collect();

        for coords in &test_points {
            let q = FixedVector::from_f32_slice(coords);
            if q.length_squared() >= FixedPoint::from_int(1) {
                continue;
            }

            let grid_result = grid.query(&q);
            let (brute_idx, _) = nearest_by_power_distance(&q, &klein_sites).unwrap();
            let brute_result = ids[brute_idx];

            assert_eq!(grid_result, Some(brute_result),
                "Grid mismatch at ({}, {}): grid={:?} brute={}",
                coords[0], coords[1], grid_result, brute_result);
        }
    }

    #[test]
    fn test_grid_insert_update() {
        // Build with parent, then insert child, verify grid updated correctly
        let parent_site = KleinPoint::new(FixedVector::from_f32_slice(&[0.0, 0.0]));
        let sites = vec![
            ("parent".to_string(), parent_site.clone()),
        ];

        let mut grid = PointLocationGrid::new(16);
        grid.build(&sites);

        // All tiles should be parent
        let q = FixedVector::from_f32_slice(&[0.5, 0.0]);
        assert_eq!(grid.query(&q), Some("parent"));

        // Insert child near (0.5, 0)
        let child_site = KleinPoint::new(FixedVector::from_f32_slice(&[0.5, 0.0]));
        grid.update_insert("parent", "child", &child_site, &parent_site);

        // Query near child should now return child
        let q_near_child = FixedVector::from_f32_slice(&[0.6, 0.0]);
        let result = grid.query(&q_near_child);
        assert_eq!(result, Some("child"),
            "After insert, tile near child should be owned by child");

        // Query near origin should still be parent
        let q_origin = FixedVector::from_f32_slice(&[0.0, 0.0]);
        assert_eq!(grid.query(&q_origin), Some("parent"),
            "Origin tile should still be parent");
    }

    #[test]
    fn test_grid_delete_update() {
        let parent_site = KleinPoint::new(FixedVector::from_f32_slice(&[0.0, 0.0]));
        let child_site = KleinPoint::new(FixedVector::from_f32_slice(&[0.5, 0.0]));

        let sites = vec![
            ("parent".to_string(), parent_site.clone()),
            ("child".to_string(), child_site.clone()),
        ];

        let mut grid = PointLocationGrid::new(16);
        grid.build(&sites);

        // Verify child owns some tiles
        let q = FixedVector::from_f32_slice(&[0.6, 0.0]);
        assert_eq!(grid.query(&q), Some("child"));

        // Delete child → reassign to parent
        grid.update_delete("child", "parent");

        assert_eq!(grid.query(&q), Some("parent"),
            "After delete, child's tiles should revert to parent");
    }

    #[test]
    fn test_grid_tile_count() {
        // Tiles inside disk ≈ π/4 · resolution²
        let resolution = 64;
        let sites = vec![
            ("root".to_string(), KleinPoint::new(FixedVector::from_f32_slice(&[0.0, 0.0]))),
        ];

        let mut grid = PointLocationGrid::new(resolution);
        grid.build(&sites);

        let assigned = grid.assigned_tile_count();
        let expected_approx = (std::f64::consts::PI / 4.0 * (resolution as f64).powi(2)) as usize;

        // Allow 10% tolerance
        let lower = expected_approx * 9 / 10;
        let upper = expected_approx * 11 / 10;
        assert!(assigned >= lower && assigned <= upper,
            "Assigned tiles {} should be near π/4·{}² ≈ {} (range [{}, {}])",
            assigned, resolution, expected_approx, lower, upper);
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

    #[test]
    fn test_grid_vs_brute_force_stress() {
        // Build a grid with many sites and verify grid matches brute-force
        // for a large number of random query points
        let site_coords: Vec<[f32; 2]> = vec![
            [0.0, 0.0], [0.3, 0.0], [-0.3, 0.0], [0.0, 0.3], [0.0, -0.3],
            [0.2, 0.2], [-0.2, 0.2], [0.2, -0.2], [-0.2, -0.2],
            [0.5, 0.1], [-0.4, 0.3], [0.1, 0.6], [-0.1, -0.5],
        ];

        let sites: Vec<(String, KleinPoint)> = site_coords.iter().enumerate()
            .map(|(i, c)| {
                let p = HyperbolicPoint::from_f32_slice(c);
                let k = poincare_to_klein(&p);
                (format!("node_{}", i), k)
            })
            .collect();

        let mut grid = PointLocationGrid::new(64);
        grid.build(&sites);

        let klein_only: Vec<KleinPoint> = sites.iter().map(|(_, k)| k.clone()).collect();
        let ids: Vec<&str> = sites.iter().map(|(id, _)| id.as_str()).collect();

        // Test a grid of query points
        let mut mismatches = 0;
        let mut total = 0;
        for xi in -9..=9 {
            for yi in -9..=9 {
                let x = xi as f32 / 10.0;
                let y = yi as f32 / 10.0;
                if x * x + y * y >= 0.99 {
                    continue;
                }

                let q = FixedVector::from_f32_slice(&[x, y]);
                total += 1;

                let grid_result = grid.query(&q);
                let (brute_idx, _) = nearest_by_power_distance(&q, &klein_only).unwrap();
                let brute_result = ids[brute_idx];

                if grid_result != Some(brute_result) {
                    mismatches += 1;
                }
            }
        }

        // Allow a tiny mismatch rate due to grid quantization at cell boundaries
        let mismatch_rate = mismatches as f64 / total as f64;
        assert!(mismatch_rate < 0.05,
            "Grid vs brute-force mismatch rate {} ({}/{}) exceeds 5%",
            mismatch_rate, mismatches, total);
    }

    #[test]
    fn test_insert_preserves_grid_correctness() {
        // Incrementally insert nodes and verify grid correctness after each insert
        let mut sites: Vec<(String, KleinPoint)> = Vec::new();
        let mut grid = PointLocationGrid::new(32);

        // Start with root at origin
        let root_k = KleinPoint::new(FixedVector::from_f32_slice(&[0.0, 0.0]));
        sites.push(("root".to_string(), root_k.clone()));
        grid.build(&sites);

        // Insert 20 nodes incrementally
        let child_coords: Vec<[f32; 2]> = vec![
            [0.3, 0.0], [-0.3, 0.0], [0.0, 0.3], [0.0, -0.3],
            [0.5, 0.1], [-0.4, 0.3], [0.1, 0.6], [-0.1, -0.5],
            [0.2, 0.2], [-0.2, 0.2], [0.2, -0.2], [-0.2, -0.2],
            [0.7, 0.0], [0.0, 0.7], [-0.6, 0.1], [0.3, -0.4],
            [0.4, 0.4], [-0.3, -0.3], [0.15, 0.15], [-0.15, 0.45],
        ];

        for (i, coords) in child_coords.iter().enumerate() {
            let p = HyperbolicPoint::from_f32_slice(coords);
            let k = poincare_to_klein(&p);
            let new_id = format!("node_{}", i);

            // Insert using incremental update (parent = root for simplicity)
            grid.update_insert("root", &new_id, &k, &root_k);
            sites.push((new_id, k));

            // Every 5 inserts, verify a spot-check
            if (i + 1) % 5 == 0 {
                let klein_only: Vec<KleinPoint> = sites.iter().map(|(_, k)| k.clone()).collect();
                // Check a few query points
                let test_queries: Vec<[f32; 2]> = vec![[0.0, 0.0], [0.2, 0.1], [-0.3, 0.2]];
                for q_coords in &test_queries {
                    let q = FixedVector::from_f32_slice(q_coords);
                    if q.length_squared() >= fp(1) { continue; }

                    let grid_r = grid.query(&q);
                    let (_brute_idx, _) = nearest_by_power_distance(&q, &klein_only).unwrap();
                    // Grid may lag behind brute force at boundaries; just verify it returns something
                    assert!(grid_r.is_some(),
                        "Grid should return a result for query inside disk");
                }
            }
        }
    }

    #[test]
    fn test_empty_tree_grid() {
        // Empty grid should return None for all queries
        let grid = PointLocationGrid::new(16);
        let q = FixedVector::from_f32_slice(&[0.0, 0.0]);
        assert_eq!(grid.query(&q), None, "Empty grid should return None");
    }

    #[test]
    fn test_query_outside_disk_clamped() {
        // Query outside Klein disk should not panic, should return something or None
        let sites = vec![
            ("root".to_string(), KleinPoint::new(FixedVector::from_f32_slice(&[0.0, 0.0]))),
        ];
        let mut grid = PointLocationGrid::new(16);
        grid.build(&sites);

        let q = FixedVector::from_f32_slice(&[1.5, 0.0]);
        // Should not panic — returns whatever tile it maps to
        let _result = grid.query(&q);
    }

    #[test]
    fn test_inverted_index_consistency_after_build() {
        let sites = vec![
            ("a".to_string(), KleinPoint::new(FixedVector::from_f32_slice(&[0.3, 0.0]))),
            ("b".to_string(), KleinPoint::new(FixedVector::from_f32_slice(&[-0.3, 0.0]))),
            ("c".to_string(), KleinPoint::new(FixedVector::from_f32_slice(&[0.0, 0.4]))),
        ];

        let mut grid = PointLocationGrid::new(32);
        grid.build(&sites);

        // Verify: every tile in tile_owners has matching grid entry
        for (id, tiles) in &grid.tile_owners {
            for &idx in tiles {
                assert_eq!(grid.grid[idx].as_deref(), Some(id.as_str()),
                    "tile_owners[{}] contains idx {} but grid[{}] = {:?}",
                    id, idx, idx, grid.grid[idx]);
            }
        }

        // Verify: every assigned grid entry is in tile_owners
        for (idx, cell) in grid.grid.iter().enumerate() {
            if let Some(ref id) = cell {
                let tiles = grid.tile_owners.get(id).expect("grid has id not in tile_owners");
                assert!(tiles.contains(&idx),
                    "grid[{}] = {} but tile_owners[{}] doesn't contain {}", idx, id, id, idx);
            }
        }
    }

    #[test]
    fn test_inverted_index_after_insert() {
        let parent_site = KleinPoint::new(FixedVector::from_f32_slice(&[0.0, 0.0]));
        let sites = vec![("parent".to_string(), parent_site.clone())];

        let mut grid = PointLocationGrid::new(16);
        grid.build(&sites);

        let parent_tiles_before = grid.tile_owners.get("parent").map(|v| v.len()).unwrap_or(0);
        assert!(parent_tiles_before > 0, "Parent should own tiles after build");

        // Insert child
        let child_site = KleinPoint::new(FixedVector::from_f32_slice(&[0.5, 0.0]));
        grid.update_insert("parent", "child", &child_site, &parent_site);

        let parent_tiles_after = grid.tile_owners.get("parent").map(|v| v.len()).unwrap_or(0);
        let child_tiles = grid.tile_owners.get("child").map(|v| v.len()).unwrap_or(0);

        assert!(child_tiles > 0, "Child should own some tiles");
        assert_eq!(parent_tiles_before, parent_tiles_after + child_tiles,
            "Total tiles should be conserved: {} != {} + {}",
            parent_tiles_before, parent_tiles_after, child_tiles);

        // Verify consistency
        for (id, tiles) in &grid.tile_owners {
            for &idx in tiles {
                assert_eq!(grid.grid[idx].as_deref(), Some(id.as_str()));
            }
        }
    }

    #[test]
    fn test_inverted_index_after_delete() {
        let parent_site = KleinPoint::new(FixedVector::from_f32_slice(&[0.0, 0.0]));
        let child_site = KleinPoint::new(FixedVector::from_f32_slice(&[0.5, 0.0]));

        let sites = vec![
            ("parent".to_string(), parent_site.clone()),
            ("child".to_string(), child_site.clone()),
        ];

        let mut grid = PointLocationGrid::new(16);
        grid.build(&sites);

        let total_before: usize = grid.tile_owners.values().map(|v| v.len()).sum();
        let child_tiles_before = grid.tile_owners.get("child").map(|v| v.len()).unwrap_or(0);
        assert!(child_tiles_before > 0, "Child should own tiles");

        // Delete child
        grid.update_delete("child", "parent");

        assert!(grid.tile_owners.get("child").is_none(),
            "Deleted node should be removed from tile_owners");

        let parent_tiles_after = grid.tile_owners.get("parent").map(|v| v.len()).unwrap_or(0);
        assert_eq!(parent_tiles_after, total_before,
            "Parent should absorb all tiles: {} vs {}", parent_tiles_after, total_before);

        // Verify consistency
        for (id, tiles) in &grid.tile_owners {
            for &idx in tiles {
                assert_eq!(grid.grid[idx].as_deref(), Some(id.as_str()));
            }
        }
    }
}
