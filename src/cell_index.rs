//! Hyperbolic cell index — a node's cell is *computed* from its coordinates.
//!
//! Replaces the fixed-bucket layer, which never located anything: its ~61
//! regions were seeded on a `dimension`-D golden spiral while the Sarkar
//! embedding is provably planar, so no bucket could contain any node and every
//! point fell through to a half-space sign test. Queries then sorted all
//! buckets with the exact 37.5 µs kernel — and the bucket count was
//! `1 + 14 × dimension`, so the cost scaled with a configuration knob rather
//! than with the data.
//!
//! # Why a computed cell
//!
//! Because the cell is a pure function of the point, it is O(1) with no search
//! and no global structure, and `shard = f(CellId)` is computable on any node
//! with no coordination — which is what a distributed deployment needs and
//! what a globally-structured tree cannot offer.
//!
//! # The cost model this is designed to
//!
//! Measured in Q64.64: `mul` 5.2 ns, `div` 32 ns, `cosh` 96 ns, `sincos`
//! 318 ns, `tanh` 2.8 µs, `atanh` 16.1 µs, `sqrt` 17.7 µs, **`atan2` 33.2 µs**,
//! `hyperbolic_distance` 37.5 µs, squared-ratio proxy ≈150 ns.
//!
//! So: no transcendental on any per-node or per-cell path. `tanh`/`sinh`/`cosh`
//! appear only in build-time tables, `atanh` and `atan2` appear nowhere, and
//! `sqrt` is paid at most once per query.
//!
//! # Exactness
//!
//! The ring expands until the per-cell lower bound proves nothing closer
//! remains. There is no window, no candidate cap and no count-based stopping
//! rule; if a bound cannot be proven the search widens. Degradation is toward
//! more work, never toward a plausible answer.

use dashmap::DashMap;
use g_math::fixed_point::FixedPoint;

use crate::hyperbolic_geometry::HyperbolicPoint;
use crate::metric_tree::{CachedNormPoint, HyperbolicMetric, Metric};

/// Bands past this are unreachable: the Q64.64 distance kernel saturates
/// around hyperbolic radius 22, and `MAX_BANDS × W` covers well beyond it.
const MAX_BANDS: usize = 64;

/// Ceiling on sectors per band.
///
/// `k(b)` grows like `sinh`, passing `i32::MAX` around band 39 (3.9e9 at band
/// 40) — which silently wrapped the sector arithmetic negative and scattered
/// deep nodes to unrelated sectors, so a query at depth 19 found shallow
/// ancestors instead of its own sibling. Capping keeps every sector value
/// inside the range the conversions can carry.
///
/// Nothing is lost by capping: sectors exist to keep cells small, and a band
/// this far out holds at most a handful of nodes, so extra angular resolution
/// there subdivides emptiness.
const MAX_SECTORS: i64 = 1 << 28;

/// A cell: radial band and angular sector.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct CellId {
    /// Radial band — hyperbolic radius `[band·W, (band+1)·W)`.
    pub band: i32,
    /// Angular sector within the band, `0..sectors(band)`.
    pub sector: i64,
}

/// One indexed node.
#[derive(Clone)]
struct Entry {
    unique_id: String,
    point: CachedNormPoint,
}

/// Per-band constants, computed once. Every transcendental in the query path
/// is hoisted into here.
#[derive(Clone)]
struct Band {
    cosh_lo: FixedPoint,
    sinh_lo: FixedPoint,
    tanh_lo: FixedPoint,
    cosh_hi: FixedPoint,
    sinh_hi: FixedPoint,
    tanh_hi: FixedPoint,
    /// Sector count, chosen so each sector subtends roughly constant
    /// hyperbolic arc length — which is what keeps occupancy flat as the
    /// space expands exponentially.
    sectors: i64,
}

/// The index.
pub struct CellIndex {
    /// Squared-norm thresholds. Band `b` covers `[thresholds[b], thresholds[b+1])`.
    ///
    /// This table *is* the definition of a band, not an approximation of
    /// `2·artanh(‖p‖)/W`. Cross-checking the two on 5 461 real positions gave
    /// 7 disagreements at boundaries where they round differently — harmless
    /// individually, fatal if insert used one route and query the other.
    thresholds: Vec<FixedPoint>,
    bands: Vec<Band>,
    cells: DashMap<CellId, Vec<Entry>>,
    /// `unique_id → CellId`, so removal touches one cell.
    node_cell: DashMap<String, CellId>,
    /// Live node count per band, so a query enumerates only occupied bands.
    band_load: DashMap<i32, usize>,
    /// Occupied sectors per band. A query walks *these*, never the sector
    /// space: `sectors(b)` grows like `sinh`, reaching ~477 000 by band 22
    /// (a depth-11 node at tau=1), so iterating the space costs ~150 ms per
    /// band whenever nothing prunes — which is exactly the case when fewer
    /// than k results have been found yet.
    band_sectors: DashMap<i32, std::collections::BTreeSet<i64>>,
    band_width: f64,
    arc: f64,
}

impl Default for CellIndex {
    fn default() -> Self {
        Self::new(0.5, 0.5)
    }
}

impl CellIndex {
    /// Build an index with band width `w` and target sector arc `arc`, both in
    /// hyperbolic units. Measured on 5 461 nodes: `0.5 / 0.5` gives 1 184 cells
    /// with a largest cell of 43 (the fixed buckets gave 26 cells and 1 462).
    /// Both are stored so cells stay reproducible.
    pub fn new(w: f64, arc: f64) -> Self {
        let thresholds = (0..=MAX_BANDS)
            .map(|b| {
                // ‖p‖ = tanh(r/2) at the band edge; compare squared norms so
                // the query path needs neither artanh nor sqrt.
                let edge = FixedPoint::from_f64((b as f64) * w / 2.0).tanh();
                edge * edge
            })
            .collect();
        let bands = (0..MAX_BANDS)
            .map(|b| {
                let lo = FixedPoint::from_f64((b as f64) * w);
                let hi = FixedPoint::from_f64(((b + 1) as f64) * w);
                let mid = ((b as f64) + 0.5) * w;
                // sectors ≈ circumference / arc; sinh grows exponentially, but
                // only *occupied* cells are ever materialised, so this is an
                // index range rather than an allocation.
                let ideal = 2.0 * std::f64::consts::PI * mid.sinh() / arc;
                let sectors = if ideal >= MAX_SECTORS as f64 {
                    MAX_SECTORS
                } else {
                    (ideal.ceil() as i64).clamp(1, MAX_SECTORS)
                };
                Band {
                    cosh_lo: lo.cosh(),
                    sinh_lo: lo.sinh(),
                    tanh_lo: lo.tanh(),
                    cosh_hi: hi.cosh(),
                    sinh_hi: hi.sinh(),
                    tanh_hi: hi.tanh(),
                    sectors,
                }
            })
            .collect();
        Self {
            thresholds,
            bands,
            cells: DashMap::new(),
            node_cell: DashMap::new(),
            band_load: DashMap::new(),
            band_sectors: DashMap::new(),
            band_width: w,
            arc,
        }
    }

    /// Band width and target arc this index was built with.
    pub fn parameters(&self) -> (f64, f64) {
        (self.band_width, self.arc)
    }

    /// Live node count.
    pub fn len(&self) -> usize {
        self.node_cell.len()
    }

    /// Whether the index holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.node_cell.is_empty()
    }

    /// Number of materialised cells — occupancy diagnostics.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// The cell a point belongs to. Comparisons and one division; no
    /// transcendental.
    pub fn cell_of(&self, point: &HyperbolicPoint) -> CellId {
        let norm_sq = planar_norm_sq(point);
        let band = self.band_of(norm_sq);
        let sectors = self.bands[band as usize].sectors;
        CellId { band, sector: self.sector_of(point, sectors) }
    }

    fn band_of(&self, norm_sq: FixedPoint) -> i32 {
        // `partition_point` over the threshold table; the last band absorbs
        // anything at or beyond the representable radius.
        let idx = self.thresholds.partition_point(|t| *t <= norm_sq);
        (idx.max(1) - 1).min(MAX_BANDS - 1) as i32
    }

    fn sector_of(&self, point: &HyperbolicPoint, sectors: i64) -> i64 {
        let pseudo = pseudo_angle(point.coords()[0], point.coords()[1]);
        sector_from_pseudo(pseudo.to_f64(), sectors)
    }

    /// Register a node. Re-registering an id moves it to its new cell.
    pub fn insert(&self, unique_id: &str, point: &HyperbolicPoint) {
        let cell = self.cell_of(point);
        if let Some(previous) = self.node_cell.get(unique_id).map(|r| *r.value()) {
            if previous == cell {
                return;
            }
            self.detach(unique_id, previous);
        }
        self.cells.entry(cell).or_default().push(Entry {
            unique_id: unique_id.to_string(),
            point: CachedNormPoint::new(point.clone()),
        });
        self.node_cell.insert(unique_id.to_string(), cell);
        *self.band_load.entry(cell.band).or_insert(0) += 1;
        self.band_sectors.entry(cell.band).or_default().insert(cell.sector);
    }

    /// Drop a node.
    pub fn remove(&self, unique_id: &str) {
        if let Some((_, cell)) = self.node_cell.remove(unique_id) {
            self.detach(unique_id, cell);
        }
    }

    fn detach(&self, unique_id: &str, cell: CellId) {
        let emptied = match self.cells.get_mut(&cell) {
            Some(mut members) => {
                members.retain(|e| e.unique_id != unique_id);
                members.is_empty()
            }
            None => false,
        };
        if emptied {
            self.cells.remove(&cell);
            if let Some(mut sectors) = self.band_sectors.get_mut(&cell.band) {
                sectors.remove(&cell.sector);
            }
        }
        if let Some(mut load) = self.band_load.get_mut(&cell.band) {
            *load = load.saturating_sub(1);
        }
    }

    /// The k nearest nodes to `query`, ascending by `(distance, unique_id)`.
    pub fn knn(&self, query: &HyperbolicPoint, k: usize) -> Vec<(String, FixedPoint)> {
        if k == 0 {
            return Vec::new();
        }
        let probe = CachedNormPoint::new(query.clone());
        let mut best: Vec<(FixedPoint, String)> = Vec::with_capacity(k + 1);
        // Threshold in cosh space, matching the bounds. `None` until k found.
        // A `Cell` so the visitor can raise it while the walker reads it.
        let ceiling = std::cell::Cell::new(None::<FixedPoint>);

        self.expand(query, |cell| {
            let Some(members) = self.cells.get(&cell) else { return };
            for entry in members.iter() {
                // Rank in squared-ratio proxy space; the exact kernel is paid
                // only for what survives into the result.
                let score = HyperbolicMetric.proxy(&probe, &entry.point);
                best.push((score, entry.unique_id.clone()));
            }
            best.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            best.dedup_by(|a, b| a.1 == b.1);
            best.truncate(k);
            if best.len() == k {
                ceiling.set(Some(cosh_from_proxy(best[k - 1].0)));
            }
        }, || ceiling.get());

        best.into_iter()
            .map(|(score, id)| (id, ratio_sq_to_distance(score)))
            .collect()
    }

    /// Every node within `radius` (hyperbolic) of `centre`.
    pub fn within_radius(&self, centre: &HyperbolicPoint, radius: FixedPoint) -> Vec<(String, FixedPoint)> {
        let probe = CachedNormPoint::new(centre.clone());
        // `radius` is caller-supplied and routinely means "everything" — horon
        // passes 1000. `cosh(1000)` is not representable in Q64.64 and the
        // infallible `cosh` panics on it, so ask for the ceiling and accept
        // that it may not exist: a radius too large to express in cosh space
        // is a radius that prunes nothing, and `None` is exactly how `expand`
        // spells "no cell can be ruled out". Degrades to a full scan, never to
        // a wrong answer.
        let ceiling = radius.try_cosh().ok();
        let mut found: Vec<(String, FixedPoint)> = Vec::new();
        let limit = move || ceiling;
        self.expand(centre, |cell| {
            let Some(members) = self.cells.get(&cell) else { return };
            for entry in members.iter() {
                let distance = HyperbolicMetric.distance(&probe, &entry.point);
                if distance <= radius {
                    found.push((entry.unique_id.clone(), distance));
                }
            }
        }, limit);
        found.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        found
    }

    /// Walk cells outward from the query, calling `visit` on each that the
    /// bound cannot rule out.
    ///
    /// Bands are ordered by their radial lower bound `cosh(|r_q − nearest band
    /// edge|)`, which follows from `d ≥ |r_q − r_p|` through the origin. That
    /// ordering is what makes the early break sound. Within a band, sectors are
    /// walked outward from the query's own; both bounds grow monotonically with
    /// their expansion index, so the first failure ends that direction.
    fn expand<F, C>(&self, query: &HyperbolicPoint, mut visit: F, ceiling: C)
    where
        F: FnMut(CellId),
        C: Fn() -> Option<FixedPoint>,
    {
        // Geometry is taken from the first two coordinates only. Structural
        // placement is provably planar, so for stored nodes this *is* the norm.
        // A caller may still hand `nearest` an off-plane query point; the plane
        // through the origin is totally geodesic, so projection onto it is
        // distance-decreasing and `d(q, p) >= d(proj(q), p)`. Bounding against
        // the projection therefore stays a valid lower bound on the true
        // distance, while scoring below still uses every coordinate.
        let norm_sq = planar_norm_sq(query);
        let pseudo_q = pseudo_angle(query.coords()[0], query.coords()[1]).to_f64();

        let one = FixedPoint::from_int(1);
        let outside = one - norm_sq;
        if outside <= FixedPoint::from_int(0) {
            return;
        }
        let cosh_q = (one + norm_sq) / outside;
        // sinh r_q = 2‖q‖/(1−‖q‖²), computed WITHOUT squaring the denominator.
        //
        // The closed form 4‖q‖²/(1−‖q‖²)² looks cheaper — no sqrt — but
        // squaring `outside` destroys it: at hyperbolic radius 20 that term is
        // 5e-9, and its square, 2.5e-17, sits only ~460 units above Q64.64's
        // 5.4e-20 resolution, leaving under three significant digits. The
        // resulting `sinh_q` was wrong by ~6e5, so `cosh_q − sinh_q` (which
        // must equal e^-r_q ≈ 1e-9) came out as 6e5 and every band bound was
        // nonsense — a query at depth 19 ranked shallow ancestors first.
        //
        // `norm_sq` is near 1, so its sqrt is accurate, and dividing by
        // `outside` once keeps the small quantity to the first power.
        let sinh_q = FixedPoint::from_int(2) * norm_sq.sqrt() / outside;
        let sinh_q_sq = sinh_q * sinh_q;

        // Occupied bands, ordered by radial bound then index (deterministic).
        let mut order: Vec<(FixedPoint, i32)> = self
            .band_load
            .iter()
            .filter(|r| *r.value() > 0)
            .map(|r| (self.radial_bound(*r.key(), cosh_q, sinh_q), *r.key()))
            .collect();
        order.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        for (radial, band) in order {
            if let Some(limit) = ceiling() {
                if radial > limit {
                    break;
                }
            }
            let data = &self.bands[band as usize];
            let sectors = data.sectors;
            let centre = sector_from_pseudo(pseudo_q, sectors);

            // Occupied sectors only, ordered by circular distance from the
            // query's sector. The bound grows with that distance, so the first
            // failure ends the band; the sector index breaks ties so the walk
            // stays deterministic. Iterating the *space* instead would be
            // unbounded work in sparse outer bands, where nothing prunes
            // because fewer than k results have been found.
            let Some(occupied) = self.band_sectors.get(&band) else { continue };
            let mut candidates: Vec<(i64, i64)> = occupied
                .iter()
                .map(|s| {
                    let raw = (s - centre).rem_euclid(sectors);
                    (raw.min(sectors - raw), *s)
                })
                .collect();
            drop(occupied);
            candidates.sort_unstable();
            let sector_width = 4.0 / (sectors as f64);
            for (offset, sector) in candidates {
                if let Some(limit) = ceiling() {
                    // Envelope: the smallest gap any sector at this offset can
                    // have. Monotone in offset, so once it fails, every later
                    // candidate fails too — unlike the per-sector bound, which
                    // is not monotone and must only `continue`.
                    let envelope_gap = ((offset - 1).max(0) as f64) * sector_width;
                    let envelope =
                        self.bound_for_gap(data, envelope_gap, cosh_q, sinh_q, sinh_q_sq);
                    if envelope.exceeds(limit) {
                        break;
                    }
                    let bound =
                        self.cell_bound(data, sector, pseudo_q, cosh_q, sinh_q, sinh_q_sq);
                    if bound.exceeds(limit) {
                        continue;
                    }
                }
                visit(CellId { band, sector });
            }
        }
    }

    /// `cosh` of the smallest possible distance from a query at `cosh_q` to
    /// anything in `band` — from `d ≥ |r_q − r_p|`, so it needs no angle.
    fn radial_bound(&self, band: i32, cosh_q: FixedPoint, sinh_q: FixedPoint) -> FixedPoint {
        let data = &self.bands[band as usize];
        let one = FixedPoint::from_int(1);
        // cosh(r_q − r_edge) = cosh r_q · cosh r_edge − sinh r_q · sinh r_edge
        let below = cosh_q * data.cosh_lo - sinh_q * data.sinh_lo;
        let above = cosh_q * data.cosh_hi - sinh_q * data.sinh_hi;
        if below < one && above < one {
            one
        } else if below >= one && above >= one {
            if below < above { below } else { above }
        } else {
            one
        }
    }

    /// Lower bound on `cosh d(query, cell)`.
    ///
    /// From the hyperbolic law of cosines with `A = cosh r_q`,
    /// `B = sinh r_q · cos Δθ_min`, minimised over the band's radius range.
    /// The interior case is `√(A² − B²)`, which is returned **squared** so no
    /// `sqrt` is taken — and is computed as `1 + sinh²r_q · sin²Δθ`, an
    /// identity (`cosh² − sinh² = 1`) that turns a difference of near-equal
    /// terms into a sum of positive ones, so it cannot cancel in fixed point.
    fn cell_bound(
        &self,
        data: &Band,
        sector: i64,
        pseudo_q: f64,
        cosh_q: FixedPoint,
        sinh_q: FixedPoint,
        sinh_q_sq: FixedPoint,
    ) -> Bound {
        let sectors = data.sectors as f64;
        let lo = (sector as f64) * 4.0 / sectors;
        let hi = ((sector + 1) as f64) * 4.0 / sectors;
        // Circular gap in pseudo-angle. Each end needs its own wrap: taking the
        // linear gap and wrapping afterwards scores a query at 0.1 against
        // sector [3,4) as 1.1 rather than 0.1.
        let circular = |a: f64, b: f64| {
            let d = (a - b).abs();
            d.min(4.0 - d)
        };
        let gap = if pseudo_q >= lo && pseudo_q < hi {
            0.0
        } else {
            circular(pseudo_q, lo).min(circular(pseudo_q, hi))
        };
        self.bound_for_gap(data, gap, cosh_q, sinh_q, sinh_q_sq)
    }

    /// The bound for a given pseudo-angle gap. Split out so the walk can also
    /// ask for the *best possible* bound at an offset — an envelope that is
    /// monotone in the offset even though individual sectors at the same
    /// offset are not, because the query sits somewhere inside its own sector
    /// rather than at its centre.
    fn bound_for_gap(
        &self,
        data: &Band,
        gap: f64,
        cosh_q: FixedPoint,
        sinh_q: FixedPoint,
        sinh_q_sq: FixedPoint,
    ) -> Bound {
        // The diamond pseudo-angle has dp/dθ = 1/(cos θ + sin θ)², which is 1
        // at θ = 0 and π/2 and ½ at π/4 — so max slope is exactly 1 and
        // Δθ ≥ Δp. Using the *average* slope here instead would overestimate
        // Δθ and break the bound.
        let delta_theta = gap.min(std::f64::consts::PI);
        let (sin_dt, cos_dt) = FixedPoint::from_f64(delta_theta).sincos();

        let zero = FixedPoint::from_int(0);
        if cos_dt <= zero {
            // B ≤ 0: the expression increases with radius, so the inner edge.
            return Bound::Plain(cosh_q * data.cosh_lo - sinh_q * cos_dt * data.sinh_lo);
        }
        let b_term = sinh_q * cos_dt;
        // Locate the minimising radius by comparing against precomputed tanh of
        // the band edges rather than taking atanh of B/A.
        if b_term >= cosh_q * data.tanh_lo && b_term <= cosh_q * data.tanh_hi {
            Bound::Squared(FixedPoint::from_int(1) + sinh_q_sq * sin_dt * sin_dt)
        } else if b_term < cosh_q * data.tanh_lo {
            Bound::Plain(cosh_q * data.cosh_lo - b_term * data.sinh_lo)
        } else {
            Bound::Plain(cosh_q * data.cosh_hi - b_term * data.sinh_hi)
        }
    }
}

/// A bound on `cosh d`, either directly or squared (the interior case, kept
/// squared so the query path never takes a `sqrt`).
enum Bound {
    Plain(FixedPoint),
    Squared(FixedPoint),
}

impl Bound {
    /// Whether this bound rules out everything within `limit` (a `cosh d`).
    ///
    /// The squared case divides rather than squaring the limit. `v > limit²`
    /// is the natural form, but `limit` is a `cosh d` and passes 3e9 around
    /// hyperbolic radius 22, where `limit²` overflows Q64.64 and wraps to
    /// nonsense — precisely the depth at which the engine is already at its
    /// precision limit, so the failure would land where it is least visible.
    /// `limit ≥ 1` always, so the division is safe, and it costs 32 ns against
    /// the 17.7 µs a `sqrt` would.
    fn exceeds(&self, limit: FixedPoint) -> bool {
        match self {
            Bound::Plain(v) => *v > limit,
            Bound::Squared(v) => *v / limit > limit,
        }
    }
}

/// Sector index for a pseudo-angle in `[0, 4)`.
///
/// Done in `f64` deliberately: `sectors` reaches `MAX_SECTORS` (2.7e8), beyond
/// what `FixedPoint::from_int`'s `i32` argument can carry, and an `f64` mantissa
/// represents every value in that range exactly. Insert and query both route
/// through here so they can never disagree about a node's sector.
fn sector_from_pseudo(pseudo: f64, sectors: i64) -> i64 {
    let scaled = (pseudo / 4.0) * sectors as f64;
    (scaled.floor() as i64).rem_euclid(sectors)
}

/// Squared norm of a point's first two coordinates — the radius the cell
/// geometry works in. Equal to the full norm for every stored node, since
/// structural placement never leaves the plane.
fn planar_norm_sq(point: &HyperbolicPoint) -> FixedPoint {
    let x = point.coords()[0];
    let y = point.coords()[1];
    x * x + y * y
}

/// Diamond pseudo-angle in `[0, 4)` — monotone in `atan2` at one division
/// instead of 33 µs. Not linear in θ; callers must use the exact max slope of
/// 1 when converting a pseudo-gap to an angular gap.
fn pseudo_angle(x: FixedPoint, y: FixedPoint) -> FixedPoint {
    let zero = FixedPoint::from_int(0);
    if x == zero && y == zero {
        return zero;
    }
    let one = FixedPoint::from_int(1);
    if y >= zero {
        if x >= zero {
            y / (x + y)
        } else {
            one - x / (y - x)
        }
    } else if x < zero {
        FixedPoint::from_int(2) - y / (-x - y)
    } else {
        FixedPoint::from_int(3) + x / (x - y)
    }
}

/// `cosh d` from a squared Möbius ratio `s = tanh²(d/2)`: `(1 + s)/(1 − s)`.
fn cosh_from_proxy(s: FixedPoint) -> FixedPoint {
    let one = FixedPoint::from_int(1);
    let denominator = one - s;
    if denominator <= FixedPoint::from_int(0) {
        return crate::constants::near_boundary().cosh();
    }
    (one + s) / denominator
}

/// Exact distance from a squared ratio: `d = 2·atanh(√s)`.
fn ratio_sq_to_distance(s: FixedPoint) -> FixedPoint {
    crate::hyperbolic_geometry::ratio_to_distance(s.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f64, y: f64) -> HyperbolicPoint {
        HyperbolicPoint::from_slice(&[FixedPoint::from_f64(x), FixedPoint::from_f64(y)])
    }

    fn populated() -> (CellIndex, Vec<(String, HyperbolicPoint)>) {
        let index = CellIndex::default();
        let mut nodes = Vec::new();
        for i in 0..400 {
            let radius = 0.05 + 0.9 * ((i % 20) as f64) / 20.0;
            let angle = 0.37 * i as f64;
            let p = point(radius * angle.cos(), radius * angle.sin());
            let id = format!("n{i}");
            index.insert(&id, &p);
            nodes.push((id, p));
        }
        (index, nodes)
    }

    fn brute_force(nodes: &[(String, HyperbolicPoint)], q: &HyperbolicPoint, k: usize) -> Vec<String> {
        let mut all: Vec<(FixedPoint, String)> = nodes
            .iter()
            .map(|(id, p)| (q.hyperbolic_distance(p), id.clone()))
            .collect();
        all.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        all.into_iter().take(k).map(|(_, id)| id).collect()
    }

    #[test]
    fn cell_assignment_is_stable_and_reproducible() {
        let index = CellIndex::default();
        let p = point(0.3, -0.4);
        assert_eq!(index.cell_of(&p), index.cell_of(&p));
        assert_eq!(index.parameters(), (0.5, 0.5));
    }

    #[test]
    fn a_node_is_its_own_nearest_neighbour() {
        let (index, nodes) = populated();
        for (id, p) in &nodes {
            let got = index.knn(p, 1);
            assert_eq!(&got[0].0, id, "{id} did not find itself");
        }
    }

    /// The aliasing defect found during design was invisible at k=1: at small
    /// sector counts, offsets +m and −m name the same sector.
    #[test]
    fn matches_brute_force_for_k_greater_than_one() {
        let (index, nodes) = populated();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for k in [1usize, 5, 20] {
            for _ in 0..40 {
                let r = rand().sqrt() * 0.95;
                let a = rand() * std::f64::consts::TAU;
                let q = point(r * a.cos(), r * a.sin());
                let got: Vec<String> = index.knn(&q, k).into_iter().map(|(id, _)| id).collect();
                let want = brute_force(&nodes, &q, k);
                assert_eq!(got, want, "k={k} disagreed with brute force");
            }
        }
    }

    #[test]
    fn within_radius_matches_brute_force() {
        let (index, nodes) = populated();
        let q = point(0.2, 0.1);
        for r in [0.5f64, 1.3, 2.7] {
            let radius = FixedPoint::from_f64(r);
            let mut got: Vec<String> =
                index.within_radius(&q, radius).into_iter().map(|(id, _)| id).collect();
            let mut want: Vec<String> = nodes
                .iter()
                .filter(|(_, p)| q.hyperbolic_distance(p) <= radius)
                .map(|(id, _)| id.clone())
                .collect();
            got.sort();
            want.sort();
            assert_eq!(got, want, "radius {r} disagreed with brute force");
        }
    }

    /// A radius so large it cannot be expressed in `cosh` space must return
    /// everything, not panic.
    ///
    /// `radius` is caller-supplied and "give me everything" is a normal way to
    /// call this — `horon`'s own API-surface test passes 1000. `cosh(1000)`
    /// overflows Q64.64 and the infallible `cosh` panics, which took the whole
    /// query down. The bound's job is to prune; when it cannot be computed the
    /// answer is "prune nothing", never "give up".
    #[test]
    fn a_radius_too_large_for_cosh_returns_everything() {
        let (index, nodes) = populated();
        let q = point(0.2, 0.1);
        for r in [30, 100, 1_000, 100_000] {
            let got = index.within_radius(&q, FixedPoint::from_int(r));
            assert_eq!(
                got.len(),
                nodes.len(),
                "radius {r} should sweep the whole index",
            );
        }
    }

    #[test]
    fn removal_takes_a_node_out_of_results() {
        let (index, nodes) = populated();
        let (victim, at) = nodes[17].clone();
        assert_eq!(index.knn(&at, 1)[0].0, victim);
        index.remove(&victim);
        assert_eq!(index.len(), nodes.len() - 1);
        let ids: Vec<String> = index.knn(&at, 5).into_iter().map(|(id, _)| id).collect();
        assert!(!ids.contains(&victim), "removed node came back: {ids:?}");
    }

    #[test]
    fn reinsert_moves_a_node_between_cells() {
        let index = CellIndex::default();
        let start = point(0.1, 0.1);
        let end = point(-0.8, 0.2);
        index.insert("drifter", &start);
        let first = index.cell_of(&start);
        index.insert("drifter", &end);
        assert_ne!(first, index.cell_of(&end));
        assert_eq!(index.len(), 1, "moving a node must not duplicate it");
        assert_eq!(index.knn(&end, 1)[0].0, "drifter");
    }

    /// `Store::nearest` accepts arbitrary coordinates, which may carry
    /// components outside the plane every stored node lives in. The bound is
    /// taken against the projection (distance-decreasing, so still a lower
    /// bound) while scoring uses all coordinates — the answer must stay exact.
    #[test]
    fn off_plane_queries_stay_exact() {
        // Mirror the engine: every point is `dimension`-wide, and structural
        // placement leaves the extra coordinates at zero.
        let index = CellIndex::default();
        let mut nodes = Vec::new();
        for i in 0..400 {
            let radius = 0.05 + 0.9 * ((i % 20) as f64) / 20.0;
            let angle = 0.37 * i as f64;
            let p = HyperbolicPoint::from_slice(&[
                FixedPoint::from_f64(radius * angle.cos()),
                FixedPoint::from_f64(radius * angle.sin()),
                FixedPoint::from_int(0),
            ]);
            let id = format!("n{i}");
            index.insert(&id, &p);
            nodes.push((id, p));
        }
        for (dx, dy, dz) in [(0.2, -0.1, 0.3), (-0.5, 0.25, 0.15), (0.0, 0.0, 0.6)] {
            let q = HyperbolicPoint::from_slice(&[
                FixedPoint::from_f64(dx),
                FixedPoint::from_f64(dy),
                FixedPoint::from_f64(dz),
            ]);
            let got: Vec<String> = index.knn(&q, 5).into_iter().map(|(id, _)| id).collect();
            let want = brute_force(&nodes, &q, 5);
            assert_eq!(got, want, "off-plane query ({dx},{dy},{dz}) was not exact");
        }
    }

    /// **The correctness argument, asserted directly.**
    ///
    /// Ring expansion is exact only if a cell's bound is a true *lower* bound
    /// on the distance to everything in that cell. Everything else is
    /// bookkeeping; if this is ever wrong, queries return confidently
    /// incorrect answers — the failure 0.5.2 had to fix.
    ///
    /// This assertion found three real defects during design that the
    /// end-to-end query tests showed only as rare wrong answers, or not at all:
    /// the pseudo-angle's *average* slope used where its maximum was required;
    /// a circular gap computed linearly and wrapped afterwards; and sector
    /// aliasing at small sector counts.
    #[test]
    fn every_cell_bound_is_a_true_lower_bound() {
        let (index, nodes) = populated();
        let mut state: u64 = 0x243F_6A88_85A3_08D3;
        let mut rand = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };

        // Deep nodes as well as shallow: the shallow-only fixture missed a
        // defect where `sinh r_q` was computed by squaring `1 − ‖q‖²`, which
        // loses most of its significant digits as that term shrinks and made
        // every band bound nonsense.
        //
        // These sit at hyperbolic radius ≈16, ≈11 and ≈7 — deep enough that
        // the squaring defect shows, but inside the ≈17.5 radius where the
        // distance kernel is still faithful. **Beyond that there is no
        // trustworthy oracle**: `hyperbolic_distance` saturates at 28.324
        // (cosh ≈ 1e12), so a bound cannot be validated against it at all.
        // That is a limit of the verification, not of the bound.
        const DEEP_NORMS: [f64; 3] = [1.0 - 2.5e-7, 1.0 - 1e-5, 1.0 - 1e-3];
        for (i, depth_norm) in DEEP_NORMS.iter().enumerate() {
            let angle = 0.9 * i as f64;
            index.insert(
                &format!("deep{i}"),
                &point(depth_norm * angle.cos(), depth_norm * angle.sin()),
            );
        }
        let nodes: Vec<(String, HyperbolicPoint)> = nodes
            .into_iter()
            .chain((0..3).map(|i| {
                let angle = 0.9 * i as f64;
                let n = DEEP_NORMS[i];
                (format!("deep{i}"), point(n * angle.cos(), n * angle.sin()))
            }))
            .collect();

        let one = FixedPoint::from_int(1);
        let mut checked = 0usize;
        let mut saturated = 0usize;
        for round in 0..120 {
            // Every fourth query sits in the deep regime.
            let r = if round % 4 == 0 {
                1.0 - 2.5e-7 * (1.0 + rand())
            } else {
                rand().sqrt() * 0.96
            };
            let a = rand() * std::f64::consts::TAU;
            let q = point(r * a.cos(), r * a.sin());

            let norm_sq = planar_norm_sq(&q);
            let outside = one - norm_sq;
            let cosh_q = (one + norm_sq) / outside;
            let sinh_q_sq = FixedPoint::from_int(4) * norm_sq / (outside * outside);
            let sinh_q = sinh_q_sq.sqrt();
            let pseudo_q = pseudo_angle(q.coords()[0], q.coords()[1]).to_f64();

            for cell in index.cells.iter() {
                let id = *cell.key();
                let data = &index.bands[id.band as usize];
                let bound = index.cell_bound(
                    data, id.sector, pseudo_q, cosh_q, sinh_q, sinh_q_sq,
                );
                for entry in cell.value().iter() {
                    let actual = &nodes
                        .iter()
                        .find(|(n, _)| *n == entry.unique_id)
                        .expect("indexed node must exist in the fixture")
                        .1;
                    // cosh d, the unit the bounds are expressed in.
                    let d = q.hyperbolic_distance(actual);
                    let cosh_d = d.cosh();
                    // The kernel saturates at 28.324; a saturated pair carries
                    // no information, so it is skipped rather than compared
                    // against a value that is already wrong.
                    if d.to_f64() > 27.0 {
                        saturated += 1;
                        continue;
                    }
                    checked += 1;
                    let holds = match bound {
                        Bound::Plain(v) => cosh_d >= v,
                        // Divided, not squared — `cosh_d²` overflows Q64.64
                        // in the deep regime this test now covers.
                        Bound::Squared(v) => cosh_d >= v / cosh_d,
                    };
                    assert!(
                        holds,
                        "cell {id:?} claimed a bound that exceeds a member's true \
                         distance (cosh d = {}) — the ring expansion would prune \
                         a genuine neighbour",
                        cosh_d.to_f64()
                    );
                }
            }
        }
        assert!(checked > 10_000, "too few (query, point) pairs checked: {checked}");
        assert!(
            saturated * 4 < checked,
            "{saturated} of {} pairs saturated the distance kernel — the fixture \
             has drifted past the usable radius and is no longer testing anything",
            saturated + checked
        );
    }

    #[test]
    fn occupancy_is_spread_not_concentrated() {
        let (index, nodes) = populated();
        assert!(
            index.cell_count() > nodes.len() / 20,
            "cells {} for {} nodes — occupancy collapsed",
            index.cell_count(),
            nodes.len()
        );
    }
}
