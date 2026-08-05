//! Generic static VP-tree over a pluggable metric (the semantic-index core).
//!
//! A build-once metric tree: no incremental insert/delete, no buffer, no lazy
//! deletion. The dynamic per-bucket [`crate::hash_table::VPTree`] serves the
//! hyperbolic spatial index, where entries churn; this static core serves the
//! epoch model of the semantic index (`docs/SEMANTIC_INDEX.md`), where an
//! index over a frozen snapshot is built in one shot and *discarded* on
//! invalidation, never mutated.
//!
//! Same proven algorithm as the bucket VPTree (Yianilos 1993): first-entry
//! vantage point, median partition, tau-shrinking KNN with closer-subtree-
//! first descent and inclusive pruning bounds.
//!
//! # Correctness constraint
//!
//! VP-tree pruning relies on the triangle inequality of the metric used at
//! **build** time. A tree is only valid for queries under the *same* metric
//! (for semantic slices: the same `dim_range`). Callers own that pairing;
//! the per-slice cache in [`crate::semantic_index`] enforces it by keying
//! trees on the dimension range they were built over.
//!
//! # Proxy scoring
//!
//! A metric may offer a cheap **proxy** — a value monotone in the distance.
//! The tree then builds, searches, and prunes entirely in proxy space,
//! paying the exact kernel ONLY for the k returned winners (whose
//! distances are recomputed exactly, so results match a brute-force exact
//! scan). Motivating case: the hyperbolic metric's exact kernel costs
//! ~78 µs/pair (fixed-point sqrt + atanh); its squared-Möbius-ratio proxy
//! costs a dot product and a division — no sqrt, no atanh — and its
//! pruning runs at exact strength in ratio space (tanh subtraction
//! identity; see `HyperbolicMetric::prune_left`).
//!
//! Pruning decisions are owned by the metric (`prune_left`/`prune_right`),
//! each proving its own conservative bound — looser bounds cost extra node
//! visits, never missed neighbors. The hyperbolic pruning bound is
//! one-sided by construction (integer Newton upper bound — no floats
//! anywhere in the compute path). Honest caveat: proxy and exact kernels
//! round independently in their last ULPs, so at rounding knife edges the
//! *retained* tie among equal-distance candidates may differ from the
//! exact-path tie; output ordering is always `(distance, unique_id)`
//! with exact distances.
//!
//! # Determinism
//!
//! - Entries are sorted by `unique_id` before building, so the tree shape is
//!   independent of input (e.g. `DashMap` iteration) order.
//! - KNN candidates are totally ordered by `(score, unique_id)`; identical
//!   inputs produce byte-identical results.

use g_math::fixed_point::FixedPoint;

use crate::constants;
use crate::hyperbolic_geometry::HyperbolicPoint;

/// A distance function over points of type `P`, with optional cheap proxy
/// scoring (see module docs).
///
/// Implementations must satisfy the metric axioms (in particular the
/// triangle inequality) — VP-tree pruning is unsound otherwise.
pub trait Metric<P> {
    /// Exact distance between two points. Symmetric, non-negative.
    fn distance(&self, a: &P, b: &P) -> FixedPoint;

    /// Whether this metric scores with a proxy. When `false` (default),
    /// the tree uses exact distances throughout — zero overhead.
    fn has_proxy(&self) -> bool {
        false
    }

    /// Cheap score, monotone in `distance`. Only called when
    /// [`Self::has_proxy`] is true; the default (exact distance) keeps
    /// proxy-less metrics correct if it is ever called anyway.
    fn proxy(&self, a: &P, b: &P) -> FixedPoint {
        self.distance(a, b)
    }

    /// May the LEFT subtree (entries with `d(vp,·) ≤ median`) be pruned?
    /// `s_query` is the query→vantage score, `s_worst` the k-th-best score.
    /// Must return true only when provably no left entry can beat the
    /// k-th best (a false negative costs a visit, never correctness).
    /// Default (score == distance): `d − τ > median`.
    fn prune_left(&self, s_query: FixedPoint, median: FixedPoint, s_worst: FixedPoint) -> bool {
        s_query - s_worst > median
    }

    /// May the RIGHT subtree (entries with `d(vp,·) ≥ median`) be pruned?
    /// Default (score == distance): `d + τ < median`.
    fn prune_right(&self, s_query: FixedPoint, median: FixedPoint, s_worst: FixedPoint) -> bool {
        s_query + s_worst < median
    }

    /// Which subtree to visit first (heuristic — correctness never depends
    /// on it). Default: left when the query sits inside the median ball.
    fn left_first(&self, s_query: FixedPoint, median: FixedPoint) -> bool {
        s_query < median
    }
}

/// Euclidean distance over decoded semantic coordinate slices.
///
/// Uses gMath's fused kernel: differences, squares, and the accumulator all
/// live at the compute tier, so the sum cannot wrap the way a storage-tier
/// Q64.64 accumulator would for large coordinates or many dimensions.
pub struct EuclideanMetric;

impl Metric<Vec<FixedPoint>> for EuclideanMetric {
    fn distance(&self, a: &Vec<FixedPoint>, b: &Vec<FixedPoint>) -> FixedPoint {
        g_math::fixed_point::imperative::fused::euclidean_distance(a, b)
    }
}

/// A Poincaré-disk point with its squared Euclidean norm cached — the
/// point type of the hyperbolic metric tree. Caching the norm turns each
/// proxy evaluation into one dot product plus a handful of arithmetic ops
/// (no sqrt at all): `|a−b|² = |a|² + |b|² − 2⟨a,b⟩`.
#[derive(Clone, Debug)]
pub struct CachedNormPoint {
    /// The point itself.
    pub point: HyperbolicPoint,
    /// `‖point‖²`, computed once at construction.
    pub norm_sq: FixedPoint,
}

impl CachedNormPoint {
    /// Wrap a point, caching its squared norm.
    pub fn new(point: HyperbolicPoint) -> Self {
        let norm_sq = point.coords().length_squared();
        Self { point, norm_sq }
    }
}

/// Hyperbolic (Poincaré disk) distance — the metric of the semantic disk semantic
/// disk (`docs/SEMANTIC_DISK.md`).
///
/// A true metric on the open disk; callers must supply strictly interior
/// points (derived barycenters always are), keeping distances out of the
/// saturating boundary regime.
///
/// # Proxy: the squared Möbius ratio
///
/// `d(a,b) = 2·atanh(r)` with `r = |a−b| / |1−āb|`. The exact kernel costs
/// ~78 µs/pair (four fixed-point sqrts + atanh, measured); `r²` needs none
/// of them:
///
/// ```text
/// r² = (|a|² + |b|² − 2⟨a,b⟩) / (1 − 2⟨a,b⟩ + |a|²·|b|²)
/// ```
///
/// `r²` is monotone in `r`, hence in `d`. Bounds (both sqrt-free, both
/// provable from the atanh series, both wide-margin under the `r ≤ 0.99`
/// clamp): `2r² ≤ 2r ≤ d` and `d ≤ 2r/(1−r²) ≤ (1+r²)/(1−r²)`.
pub struct HyperbolicMetric;

/// `near_boundary()²` — the proxy-space cap matching the exact kernel's
/// ratio clamp (keeps `1 − r²` away from zero for the upper bound).
fn near_boundary_sq() -> FixedPoint {
    constants::near_boundary() * constants::near_boundary()
}

impl Metric<CachedNormPoint> for HyperbolicMetric {
    fn distance(&self, a: &CachedNormPoint, b: &CachedNormPoint) -> FixedPoint {
        a.point.hyperbolic_distance(&b.point)
    }

    fn has_proxy(&self) -> bool {
        true
    }

    fn proxy(&self, a: &CachedNormPoint, b: &CachedNormPoint) -> FixedPoint {
        let zero = FixedPoint::from_int(0);
        let one = FixedPoint::from_int(1);
        let two = FixedPoint::from_int(2);
        let cap = near_boundary_sq();

        // Origin special cases mirror hyperbolic_ratio: ratio(0, q) = |q|,
        // so the squared proxy is the squared norm (capped like the main
        // branch, so ties beyond the cap stay consistent).
        let eps_sq = constants::small_epsilon() * constants::small_epsilon();
        if a.norm_sq < eps_sq {
            return if b.norm_sq > cap { cap } else { b.norm_sq };
        }
        if b.norm_sq < eps_sq {
            return if a.norm_sq > cap { cap } else { a.norm_sq };
        }

        let dot = a.point.coords().dot(b.point.coords());
        let mut dist_sq = a.norm_sq + b.norm_sq - two * dot;
        if dist_sq < zero {
            dist_sq = zero; // coincident points + rounding
        }
        let den_sq = one - two * dot + a.norm_sq * b.norm_sq;

        // Degenerate |1−āb| ≈ 0 guard, squared-space threshold of the
        // exact kernel's ε check; also catches rounding-negative values.
        if den_sq < constants::epsilon() * constants::epsilon() {
            return cap;
        }

        let r_sq = dist_sq / den_sq;
        if r_sq > cap {
            cap
        } else {
            r_sq
        }
    }

    /// Prune left ⟺ provably `d_q − m > τ` — the tanh subtraction identity
    /// (`r(x) = tanh(x/2)` is monotone, `tanh(a−b) = (rₐ−r_b)/(1−rₐr_b)`):
    ///
    /// ```text
    /// d_q − m > τ  ⟺  (r_q − r_m) > r_τ · (1 − r_q·r_m)     [r_q > r_m]
    /// ```
    ///
    /// Exact pruning power — no polynomial-bound loss (bounds like
    /// `d ≥ 2r` saturate at 2 while hyperbolic medians grow without
    /// bound, which collapses pruning at the top of the tree).
    ///
    /// Squaring both (non-negative) sides leaves ONE irrational term,
    /// `x = r_q·r_m = √(s_q·s_m)`:
    ///
    /// ```text
    /// s_q − 2x + s_m > s_τ · (1 − x)²
    /// ```
    ///
    /// The left side falls and the right side rises as x grows (both
    /// factors of d/dx: −2 vs 2·s_τ·(1−x) < 2), so substituting a
    /// GUARANTEED UPPER bound for x — [`sqrt_upper_bound`], pure
    /// integer/fixed-point, no floats, no gMath sqrt — keeps the test
    /// strictly conservative: it may descend a few percent more than
    /// exact, it can never prune a true neighbor.
    fn prune_left(&self, s_query: FixedPoint, median: FixedPoint, s_worst: FixedPoint) -> bool {
        let one = FixedPoint::from_int(1);
        let two = FixedPoint::from_int(2);
        if s_query <= median {
            return false; // r_q ≤ r_m — the left ball may contain neighbors
        }
        let x_ub = sqrt_upper_bound(s_query * median);
        let one_minus_x = one - x_ub;
        s_query + median - two * x_ub > s_worst * (one_minus_x * one_minus_x)
    }

    /// Prune right ⟺ provably `m − d_q > τ` (same identity, mirrored:
    /// requires `r_m > r_q`, i.e. `median > s_query`).
    fn prune_right(&self, s_query: FixedPoint, median: FixedPoint, s_worst: FixedPoint) -> bool {
        let one = FixedPoint::from_int(1);
        let two = FixedPoint::from_int(2);
        if median <= s_query {
            return false;
        }
        let x_ub = sqrt_upper_bound(s_query * median);
        let one_minus_x = one - x_ub;
        s_query + median - two * x_ub > s_worst * (one_minus_x * one_minus_x)
    }

    /// Heuristic order (squares are monotone — sqrt-free and exact).
    fn left_first(&self, s_query: FixedPoint, median: FixedPoint) -> bool {
        s_query < median
    }
}

/// A guaranteed UPPER bound on √p in pure integer/fixed-point arithmetic —
/// no floats, no full-precision sqrt. Seed: the power of two above √p,
/// read off the raw bit length (shifts only). One Newton step
/// `x₁ = (x₀ + p/x₀)/2` is ≥ √p for ANY positive x₀ (AM-GM) and lands
/// within ~6.1% from a power-of-two seed. One-sided by construction, so
/// pruning needs no error pad; deterministic by construction, so the
/// no-floats determinism claim stays verifiable.
fn sqrt_upper_bound(p: FixedPoint) -> FixedPoint {
    let raw = p.raw();
    if raw <= 0 {
        return FixedPoint::from_int(0);
    }
    // p = raw / 2^64, so √p = √raw / 2^32. With b = bit_length(raw):
    // √raw < 2^⌈b/2⌉, hence seed_raw = 2^(⌈b/2⌉ + 32) ≥ √p in Q64.64
    // (b ≤ 65 for the r²-products here, so the shift never overflows).
    let b = 128 - (raw as u128).leading_zeros();
    let seed = FixedPoint::from_raw(1i128 << (b.div_ceil(2) + 32));
    // Two Newton steps (each preserves the upper bound): ~6.1% error after
    // the first, ~0.18% after the second — tight enough that pruning
    // visits stay within a few percent of the exact-sqrt count, for the
    // price of two fixed-point divisions.
    let x1 = FixedPoint::from_raw((seed.raw() + (p / seed).raw()) >> 1);
    FixedPoint::from_raw((x1.raw() + (p / x1).raw()) >> 1)
}

/// One tree node: an entry plus the median SCORE that partitions its
/// descendants (left: `score ≤ median`, right: `score ≥ median` —
/// inclusive boundaries; scores are exact distances for proxy-less
/// metrics, monotone surrogates otherwise).
struct TreeNode<P> {
    unique_id: String,
    point: P,
    median: FixedPoint,
    left: Option<Box<TreeNode<P>>>,
    right: Option<Box<TreeNode<P>>>,
}

/// Static VP-tree over `(unique_id, point)` entries under a pluggable metric.
pub struct MetricVpTree<P> {
    root: Option<Box<TreeNode<P>>>,
    len: usize,
}

impl<P> MetricVpTree<P> {
    /// Build a tree from entries. Consumes and sorts them by `unique_id`
    /// first so the tree shape is deterministic regardless of input order.
    pub fn build<M: Metric<P>>(mut entries: Vec<(String, P)>, metric: &M) -> Self {
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let len = entries.len();
        Self {
            root: Self::build_node(entries, metric),
            len,
        }
    }

    /// Number of entries in the tree.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Proxy score when the metric offers one, exact distance otherwise.
    fn score<M: Metric<P>>(metric: &M, a: &P, b: &P) -> FixedPoint {
        if metric.has_proxy() {
            metric.proxy(a, b)
        } else {
            metric.distance(a, b)
        }
    }

    fn build_node<M: Metric<P>>(
        mut entries: Vec<(String, P)>,
        metric: &M,
    ) -> Option<Box<TreeNode<P>>> {
        if entries.is_empty() {
            return None;
        }
        if entries.len() == 1 {
            let (unique_id, point) = entries.remove(0);
            return Some(Box::new(TreeNode {
                unique_id,
                point,
                median: FixedPoint::from_int(0),
                left: None,
                right: None,
            }));
        }

        // Vantage point: first entry (deterministic after the uid sort).
        let (vp_id, vp_point) = entries.swap_remove(0);

        // Score from the vantage point to every remaining entry — with a
        // proxy, ONE exact-kernel call per tree node (the median, below)
        // instead of one per entry.
        let mut with_scores: Vec<((String, P), FixedPoint)> = entries
            .into_iter()
            .map(|e| {
                let s = Self::score(metric, &vp_point, &e.1);
                (e, s)
            })
            .collect();

        // Median by (score, uid) — monotone in (distance, uid), so equal
        // scores partition deterministically too. The median is stored in
        // SCORE space: pruning decisions are metric-owned and score-space
        // native, so the build never touches the exact kernel at all.
        with_scores.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| (a.0).0.cmp(&(b.0).0)));
        let median = with_scores[with_scores.len() / 2].1;

        // Strictly less than the median score -> left, rest -> right.
        // Monotonicity gives left ⟹ d ≤ d(median), right ⟹ d ≥ d(median).
        let (left_vec, right_vec): (Vec<_>, Vec<_>) =
            with_scores.into_iter().partition(|(_, s)| *s < median);

        let left = Self::build_node(left_vec.into_iter().map(|(e, _)| e).collect(), metric);
        let right = Self::build_node(right_vec.into_iter().map(|(e, _)| e).collect(), metric);

        Some(Box::new(TreeNode {
            unique_id: vp_id,
            point: vp_point,
            median,
            left,
            right,
        }))
    }

    /// Find the k nearest entries to `query`, sorted ascending by
    /// `(distance, unique_id)` with **exact** distances.
    ///
    /// `metric` must be the same metric the tree was built with — pruning is
    /// unsound otherwise (see module docs).
    pub fn knn<M: Metric<P>>(&self, query: &P, k: usize, metric: &M) -> Vec<(String, FixedPoint)> {
        if k == 0 {
            return Vec::new();
        }

        // Candidates kept sorted ascending by (score, uid); scores are
        // exact distances for proxy-less metrics.
        let mut candidates: Vec<(FixedPoint, &TreeNode<P>)> = Vec::with_capacity(k + 1);
        if let Some(ref root) = self.root {
            Self::search_knn(root, query, k, metric, &mut candidates);
        }

        // Winners get exact distances (recomputed for proxy metrics — k
        // exact calls total), and the output order is (distance, uid).
        let mut results: Vec<(String, FixedPoint)> = candidates
            .into_iter()
            .map(|(score, node)| {
                let d = if metric.has_proxy() {
                    metric.distance(query, &node.point)
                } else {
                    score
                };
                (node.unique_id.clone(), d)
            })
            .collect();
        results.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        results
    }

    /// The k-th-best score, or `None` while fewer than k candidates are
    /// known (pruning is disabled until the ring is full).
    fn worst_score<P2>(candidates: &[(FixedPoint, P2)], k: usize) -> Option<FixedPoint> {
        if candidates.len() < k {
            return None;
        }
        Some(candidates.last().unwrap().0)
    }

    fn search_knn<'a, M: Metric<P>>(
        node: &'a TreeNode<P>,
        query: &P,
        k: usize,
        metric: &M,
        candidates: &mut Vec<(FixedPoint, &'a TreeNode<P>)>,
    ) {
        let s = Self::score(metric, query, &node.point);

        // Admit the vantage point under the (score, uid) total order —
        // monotone in (distance, uid).
        let full = candidates.len() == k;
        let admit = !full || {
            let worst = candidates.last().unwrap();
            (s, node.unique_id.as_str()) < (worst.0, worst.1.unique_id.as_str())
        };
        if admit {
            candidates.push((s, node));
            candidates.sort_by(|a, b| {
                a.0.cmp(&b.0)
                    .then_with(|| a.1.unique_id.cmp(&b.1.unique_id))
            });
            if candidates.len() > k {
                candidates.truncate(k);
            }
        }

        // Pruning decisions belong to the metric (each proves its own
        // bounds; see prune_left/prune_right docs). For proxy-less metrics
        // the defaults are the exact pre-proxy conditions, bit for bit.
        // Pruning stays disabled until k candidates are known.
        let worst = Self::worst_score(candidates, k);
        let descend_left = |worst: Option<FixedPoint>| {
            worst.is_none_or(|w| !metric.prune_left(s, node.median, w))
        };
        let descend_right = |worst: Option<FixedPoint>| {
            worst.is_none_or(|w| !metric.prune_right(s, node.median, w))
        };

        // Closer subtree first for tighter pruning (heuristic — any order
        // is correct); re-read the worst score between the two descents —
        // the first may have shrunk it.
        if metric.left_first(s, node.median) {
            if let Some(ref left) = node.left {
                if descend_left(worst) {
                    Self::search_knn(left, query, k, metric, candidates);
                }
            }
            let worst = Self::worst_score(candidates, k);
            if let Some(ref right) = node.right {
                if descend_right(worst) {
                    Self::search_knn(right, query, k, metric, candidates);
                }
            }
        } else {
            if let Some(ref right) = node.right {
                if descend_right(worst) {
                    Self::search_knn(right, query, k, metric, candidates);
                }
            }
            let worst = Self::worst_score(candidates, k);
            if let Some(ref left) = node.left {
                if descend_left(worst) {
                    Self::search_knn(left, query, k, metric, candidates);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(v: i32) -> FixedPoint {
        FixedPoint::from_int(v)
    }

    fn point(coords: &[i32]) -> Vec<FixedPoint> {
        coords.iter().map(|&c| fp(c)).collect()
    }

    /// Brute-force reference: full scan, same (distance, uid) ordering.
    fn brute_knn(
        entries: &[(String, Vec<FixedPoint>)],
        query: &Vec<FixedPoint>,
        k: usize,
    ) -> Vec<(String, FixedPoint)> {
        let mut all: Vec<(FixedPoint, String)> = entries
            .iter()
            .map(|(uid, p)| (EuclideanMetric.distance(query, p), uid.clone()))
            .collect();
        all.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        all.truncate(k);
        all.into_iter().map(|(d, uid)| (uid, d)).collect()
    }

    /// Deterministic pseudo-random coordinates (no rand dependency): a
    /// simple LCG over a fixed seed.
    fn lcg_entries(n: usize, dims: usize, seed: u64) -> Vec<(String, Vec<FixedPoint>)> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) % 2001) as i32 - 1000 // range [-1000, 1000]
        };
        (0..n)
            .map(|i| {
                let coords: Vec<FixedPoint> = (0..dims).map(|_| fp(next())).collect();
                (format!("node_{:05}", i), coords)
            })
            .collect()
    }

    #[test]
    fn knn_matches_brute_force() {
        for &(n, dims, seed) in &[(50usize, 2usize, 7u64), (200, 4, 42), (500, 8, 1234)] {
            let entries = lcg_entries(n, dims, seed);
            let tree = MetricVpTree::build(entries.clone(), &EuclideanMetric);
            let mut qstate = seed ^ 0xdead_beef;
            let mut next = || {
                qstate = qstate
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((qstate >> 33) % 2001) as i32 - 1000
            };
            for _ in 0..20 {
                let query: Vec<FixedPoint> = (0..dims).map(|_| fp(next())).collect();
                for &k in &[1usize, 5, 17, n + 10] {
                    let got = tree.knn(&query, k, &EuclideanMetric);
                    let want = brute_knn(&entries, &query, k);
                    assert_eq!(got, want, "n={} dims={} k={}", n, dims, k);
                }
            }
        }
    }

    #[test]
    fn ties_break_by_uid() {
        // Four entries at the same point: ties must resolve by uid, and the
        // result must be identical no matter the insertion order.
        let p = point(&[3, 4]);
        let mut entries: Vec<(String, Vec<FixedPoint>)> = ["d", "b", "a", "c"]
            .iter()
            .map(|s| (s.to_string(), p.clone()))
            .collect();
        let tree1 = MetricVpTree::build(entries.clone(), &EuclideanMetric);
        entries.reverse();
        let tree2 = MetricVpTree::build(entries, &EuclideanMetric);

        let query = point(&[0, 0]);
        let r1 = tree1.knn(&query, 2, &EuclideanMetric);
        let r2 = tree2.knn(&query, 2, &EuclideanMetric);
        assert_eq!(r1, r2);
        assert_eq!(r1[0].0, "a");
        assert_eq!(r1[1].0, "b");
    }

    fn hpoint(x: f32, y: f32) -> CachedNormPoint {
        CachedNormPoint::new(HyperbolicPoint::from_f32_slice(&[x, y]))
    }

    /// The load-bearing proxy assumptions: the squared-ratio proxy must be
    /// bracketed by its bounds against the EXACT kernel, on every code
    /// path (main, near-origin, boundary-clamped, coincident), and must
    /// order pairs the same way the exact distance does.
    #[test]
    fn hyperbolic_proxy_bounds_and_monotonicity() {
        let m = HyperbolicMetric;
        let cases = [
            (hpoint(0.3, 0.4), hpoint(-0.2, 0.5)),
            (hpoint(0.0, 0.0), hpoint(0.6, -0.3)),
            (hpoint(0.55, 0.0), hpoint(0.0, 0.0)),
            (hpoint(0.95, 0.0), hpoint(-0.95, 0.0)),
            (hpoint(0.98, 0.01), hpoint(0.97, 0.02)),
            (hpoint(0.1, 0.1), hpoint(0.1, 0.1)),
            (hpoint(0.001, 0.0), hpoint(0.0, 0.001)),
        ];
        let mut scored: Vec<(FixedPoint, FixedPoint)> = Vec::new();
        let two = FixedPoint::from_int(2);
        let one = FixedPoint::from_int(1);
        for (a, b) in &cases {
            let exact = m.distance(a, b);
            let s = m.proxy(a, b);
            // The bound facts the pruning proofs rest on:
            // 2s ≤ d (since s = r² ≤ r) and d ≤ 2r/(1−r²) ⟹ d² ≤ 4s/(1−s)².
            assert!(two * s <= exact, "lower bound violated: s={:?} d={:?}", s, exact);
            let denom = one - s;
            let d_sq_ub = FixedPoint::from_int(4) * s / (denom * denom);
            assert!(
                exact * exact <= d_sq_ub + constants::epsilon(),
                "upper bound violated: d²={:?} ub={:?}",
                exact * exact, d_sq_ub
            );
            scored.push((s, exact));
        }
        // Monotone: sorting by proxy and by exact distance agree.
        let mut by_proxy = scored.clone();
        by_proxy.sort_by(|a, b| a.0.cmp(&b.0));
        let mut by_exact = scored;
        by_exact.sort_by(|a, b| a.1.cmp(&b.1));
        let d_order: Vec<_> = by_proxy.iter().map(|(_, d)| *d).collect();
        let d_expected: Vec<_> = by_exact.iter().map(|(_, d)| *d).collect();
        assert_eq!(d_order, d_expected, "proxy ordering diverged from distance ordering");
    }

    /// Hyperbolic-metric tree results must equal a brute-force exact scan —
    /// the same bar the Euclidean path is held to. Returned distances are
    /// exact (winners are recomputed with the exact kernel).
    #[test]
    fn hyperbolic_knn_matches_brute_force() {
        let m = HyperbolicMetric;
        // Deterministic interior points (norm < ~0.9).
        let mut state = 7u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((state >> 33) % 1800) as f32 / 1000.0) - 0.9 // [-0.9, 0.9)
        };
        let entries: Vec<(String, CachedNormPoint)> = (0..300)
            .map(|i| {
                let (x, y) = (next() * 0.7, next() * 0.7); // keep |p| < 0.9
                (format!("p{:03}", i), hpoint(x, y))
            })
            .collect();
        let tree = MetricVpTree::build(entries.clone(), &m);

        for qi in [0usize, 111, 222] {
            let query = entries[qi].1.clone();
            for k in [1usize, 5, 20] {
                let got = tree.knn(&query, k, &m);
                let mut want: Vec<(FixedPoint, String)> = entries
                    .iter()
                    .map(|(uid, p)| (m.distance(&query, p), uid.clone()))
                    .collect();
                want.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                want.truncate(k);
                let want: Vec<(String, FixedPoint)> =
                    want.into_iter().map(|(d, uid)| (uid, d)).collect();
                assert_eq!(got, want, "qi={} k={}", qi, k);
            }
        }
    }

    #[test]
    fn empty_and_degenerate() {
        let tree: MetricVpTree<Vec<FixedPoint>> =
            MetricVpTree::build(Vec::new(), &EuclideanMetric);
        assert!(tree.is_empty());
        assert!(tree.knn(&point(&[0]), 5, &EuclideanMetric).is_empty());

        let tree = MetricVpTree::build(vec![("only".to_string(), point(&[1]))], &EuclideanMetric);
        assert_eq!(tree.len(), 1);
        let r = tree.knn(&point(&[0]), 3, &EuclideanMetric);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, "only");
        assert!(tree.knn(&point(&[0]), 0, &EuclideanMetric).is_empty());
    }
}
