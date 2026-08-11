//! hash_table.rs - O(1) Hyperbolic Space Lookup with VP-tree Spatial Index
//! # Hyperbolic Hash Table
//!
//! Efficient O(1) lookups in hyperbolic space using geometric hashing techniques,
//! with O(log n) spatial queries via per-bucket Vantage Point trees.
//!
//! This module implements a specialized hash table that leverages the Poincaré disk model
//! of hyperbolic geometry to enable constant-time operations on hierarchical data structures.
//!
//! ## Key Features:
//!
//! - Geometric hashing for fast point-location in hyperbolic space
//! - Locality-sensitive buckets for efficient similarity-based retrieval
//! - VP-tree per bucket for O(log n) range and nearest-neighbor queries
//! - Hierarchical organization supporting tree-like data structures
//! - Fixed-point arithmetic for numerical stability

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug, Formatter};
use std::sync::Mutex;
use dashmap::DashMap;
use sha3::{Sha3_512, Digest};
use g_math::fixed_point::{FixedPoint, FixedVector};
use super::hyperbolic_geometry::{PoincareDisk, HyperbolicPoint, distance_to_ratio};
use crate::metric_tree::{hyperbolic_ratio_sq, sq_ratio_separation_exceeds};
use crate::constants;

// ---------------------------------------------------------------------------
// GeometricSignature
// ---------------------------------------------------------------------------

/// A geometric signature for a node in hyperbolic space.
///
/// This signature uniquely identifies a point or region in the
/// hyperbolic space, enabling O(1) lookups.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct GeometricSignature {
    /// Hash value for O(1) lookup
    hash: String,
    /// Tree level for hierarchical navigation
    level: u32,
    /// Position signature in hyperbolic space
    position_signature: Vec<i32>,
}

impl GeometricSignature {
    /// Create a new geometric signature.
    pub fn new(hash: String, level: u32, position_signature: Vec<i32>) -> Self {
        Self {
            hash,
            level,
            position_signature,
        }
    }

    /// Get the hash value.
    pub fn hash(&self) -> &str {
        &self.hash
    }

    /// Get the tree level.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// Get the position signature.
    pub fn position_signature(&self) -> &[i32] {
        &self.position_signature
    }

    /// Create a stub signature for data-only nodes (no geometric meaning).
    pub fn stub(unique_id: &str) -> Self {
        Self {
            hash: unique_id.to_string(),
            level: 0,
            position_signature: Vec::new(),
        }
    }

    /// Whether this is a stub signature (a data-only node with no geometric
    /// embedding — upgradeable via `embed_existing`).
    pub fn is_stub(&self) -> bool {
        self.position_signature.is_empty()
    }

    /// Get a unique node identifier that includes level and position.
    /// Unlike `hash()` which identifies a geometric bucket (shared by nearby points),
    /// this produces a unique key suitable for node storage.
    ///
    /// For stub signatures (data-only nodes), returns the hash directly.
    pub fn unique_id(&self) -> String {
        if self.position_signature.is_empty() {
            // Stub signature — hash IS the unique_id
            return self.hash.clone();
        }
        use sha3::{Sha3_256, Digest as _};
        let mut hasher = Sha3_256::new();
        hasher.update(self.hash.as_bytes());
        hasher.update(self.level.to_le_bytes());
        for &v in &self.position_signature {
            hasher.update(v.to_le_bytes());
        }
        hex::encode(&hasher.finalize()[..16])
    }
}

impl Debug for GeometricSignature {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "GeometricSignature(hash={}, level={})",
               &self.hash[0..8], self.level)
    }
}

// ---------------------------------------------------------------------------
// HyperbolicRegion
// ---------------------------------------------------------------------------

/// Hyperbolic region in the Poincaré disk.
///
/// A region in hyperbolic space, used for locality-sensitive hashing.
#[derive(Clone, Debug)]
pub struct HyperbolicRegion {
    /// Center point of the region
    center: HyperbolicPoint,
    /// Hyperbolic radius of the region
    radius: FixedPoint,
    /// Validation mask for fast verification
    validation_mask: FixedVector,
}

impl HyperbolicRegion {
    /// Create a new hyperbolic region.
    pub fn new(center: HyperbolicPoint, radius: FixedPoint) -> Self {
        let dimension = center.dimension();

        // Create validation mask based on center point
        let validation_mask = {
            let one = FixedPoint::from_int(1);
            let mut mask = FixedVector::new(dimension);
            for i in 0..dimension {
                let x = center.coords()[i];
                mask[i] = x * (one + x.tanh());
            }
            mask
        };

        Self {
            center,
            radius,
            validation_mask,
        }
    }

    /// Check if a point is contained in this region.
    pub fn contains(&self, point: &HyperbolicPoint, poincare_disk: &PoincareDisk) -> bool {
        let distance = poincare_disk.distance(&self.center, point);
        distance <= self.radius
    }

    /// Get the center of the region.
    pub fn center(&self) -> &HyperbolicPoint {
        &self.center
    }

    /// Get the radius of the region.
    pub fn radius(&self) -> FixedPoint {
        self.radius
    }

    /// Get the validation mask.
    pub fn validation_mask(&self) -> &FixedVector {
        &self.validation_mask
    }

    /// Quick validation check against a point.
    /// Uses a generous threshold (positive similarity) as a fast heuristic.
    pub fn quick_validate(&self, point: &HyperbolicPoint) -> bool {
        let similarity = point.coords().dot(&self.validation_mask);
        similarity > constants::epsilon()
    }
}

// ---------------------------------------------------------------------------
// BucketEntry
// ---------------------------------------------------------------------------

/// An entry in the spatial index tracking a node's position in a bucket.
#[derive(Clone, Debug)]
pub struct BucketEntry {
    /// Node unique identifier
    pub unique_id: String,
    /// Position in the Poincaré disk
    pub point: HyperbolicPoint,
    /// Tree level
    pub level: u32,
    /// `‖point‖²`, cached at construction — turns every VP-tree score into
    /// one dot product plus arithmetic (no sqrt), same trick as
    /// `metric_tree::CachedNormPoint`.
    pub norm_sq: FixedPoint,
}

impl BucketEntry {
    /// Construct an entry, caching the squared norm of `point`.
    pub fn new(unique_id: String, point: HyperbolicPoint, level: u32) -> Self {
        let norm_sq = point.coords().length_squared();
        Self { unique_id, point, level, norm_sq }
    }
}

// ---------------------------------------------------------------------------
// VP-tree: O(log n) spatial queries under the hyperbolic metric
// ---------------------------------------------------------------------------

/// Compare two FixedPoint values for sorting.
fn cmp_fp(a: FixedPoint, b: FixedPoint) -> std::cmp::Ordering {
    if a < b { std::cmp::Ordering::Less }
    else if a > b { std::cmp::Ordering::Greater }
    else { std::cmp::Ordering::Equal }
}

/// Euclidean squared distance between two hyperbolic points.
/// Pure arithmetic (no transcendentals): O(d) multiply-adds, ~100ns.
fn euclidean_distance_sq(a: &HyperbolicPoint, b: &HyperbolicPoint) -> FixedPoint {
    // Deliberately storage-tier (fused-kernel adoption evaluated 2026-07-11 and
    // rejected by measurement): inputs are Poincaré-interior (|x| < 1,
    // d ≤ 4), so the accumulator is bounded by 4 and cannot wrap; the
    // fused compute-tier kernel costs ~3× at this ns-scale call site for
    // ULPs nothing downstream can observe.
    let mut sum = FixedPoint::from_int(0);
    let d = a.dimension().min(b.dimension());
    for i in 0..d {
        let diff = a.coords()[i] - b.coords()[i];
        sum = sum + diff * diff;
    }
    sum
}

/// Buffered insertion threshold: rebuild VP-tree after this many pending inserts.
const VP_BUFFER_THRESHOLD: usize = 32;

/// Rebuild when the buffer reaches `tree_len / VP_REBUILD_DIVISOR` (floor
/// `VP_BUFFER_THRESHOLD`). Smaller = fewer rebuilds, larger buffer, slower
/// queries. This is the insert/query dial.
const VP_REBUILD_DIVISOR: usize = 16;


/// Lazy deletion threshold: rebuild VP-tree after this many pending deletes.
const VP_DELETE_THRESHOLD: usize = 32;

/// Internal node of a Vantage Point tree.
#[derive(Clone, Debug)]
struct VPNode {
    /// The vantage point entry.
    entry: BucketEntry,
    /// Median distance from vantage point to entries in subtrees.
    median: FixedPoint,
    /// Left subtree: entries closer than median distance.
    left: Option<Box<VPNode>>,
    /// Right subtree: entries at or beyond median distance.
    right: Option<Box<VPNode>>,
}

/// Vantage Point tree for O(log n) spatial queries under the hyperbolic metric.
///
/// Uses a buffer + lazy-deletion strategy for efficient dynamic updates:
/// - Insertions accumulate in a small buffer; the tree rebuilds when the buffer fills.
/// - Deletions mark entries as dead; the tree rebuilds when too many are marked.
/// - Queries search both the tree and the buffer, preserving correctness.
///
/// Within each hash table bucket, this replaces the previous linear scan
/// (O(n/B) per bucket) with O(log(n/B)) queries.
#[derive(Clone, Debug)]
pub struct VPTree {
    /// Root of the VP-tree (None if tree portion is empty).
    root: Option<Box<VPNode>>,
    /// Recent insertions not yet incorporated into the tree.
    buffer: Vec<BucketEntry>,
    /// Unique IDs of lazily deleted entries still in the tree.
    deleted: HashSet<String>,
    /// Number of entries in the tree structure (including lazily deleted ones).
    tree_size: usize,
}

impl VPTree {
    /// Create an empty VP-tree.
    pub fn new() -> Self {
        Self {
            root: None,
            buffer: Vec::new(),
            deleted: HashSet::new(),
            tree_size: 0,
        }
    }

    /// Insert an entry. Duplicates (by unique_id) in the buffer are ignored.
    pub fn insert(&mut self, entry: BucketEntry) {
        if self.buffer.iter().any(|e| e.unique_id == entry.unique_id) {
            return;
        }
        // If previously lazily deleted, un-delete
        self.deleted.remove(&entry.unique_id);

        self.buffer.push(entry);
        // Rebuild on a FRACTION of tree size, not a fixed count. With a fixed
        // threshold the tree is rebuilt every 32 inserts no matter how large
        // it has grown, so amortized cost is O(m log m / 32) — linear in m,
        // which makes a bulk load quadratic. Scaling the threshold with m
        // makes it O(log m) amortized. The buffer is scanned linearly by
        // queries, so the divisor bounds that cost too.
        let tree_len = self.tree_size;
        let threshold = VP_BUFFER_THRESHOLD.max(tree_len / VP_REBUILD_DIVISOR);
        if self.buffer.len() >= threshold {
            self.rebuild();
        }
    }

    /// Remove an entry by unique_id.
    pub fn remove(&mut self, unique_id: &str) {
        // Try buffer first (cheaper than tree traversal)
        let before = self.buffer.len();
        self.buffer.retain(|e| e.unique_id != unique_id);
        if self.buffer.len() < before {
            return;
        }

        // Mark as lazily deleted in the tree
        self.deleted.insert(unique_id.to_string());
        if self.deleted.len() >= VP_DELETE_THRESHOLD {
            self.rebuild();
        }
    }

    /// Number of live entries (tree + buffer - deleted).
    pub fn live_count(&self) -> usize {
        let tree_live = self.tree_size.saturating_sub(self.deleted.len());
        tree_live + self.buffer.len()
    }

    /// Whether the tree has any live entries.
    pub fn is_empty(&self) -> bool {
        self.live_count() == 0
    }

    /// Find all live entries within hyperbolic radius of center.
    pub fn find_in_radius(&self, center: &HyperbolicPoint, radius: FixedPoint) -> Vec<(String, FixedPoint)> {
        let mut results = Vec::new();

        // One conversion for the whole query: tree and buffer both filter in
        // SQUARED-ratio proxy space (monotone, so `s <= s(radius)` is exactly
        // `d <= radius`); scores are sqrt-free via the cached norms, and only
        // reported hits pay the exact kernel — with values bit-identical to
        // the pre-proxy code.
        let radius_sq = {
            let r = distance_to_ratio(radius);
            r * r
        };
        let center_norm_sq = center.coords().length_squared();
        if let Some(ref root) = self.root {
            Self::search_radius(root, center, center_norm_sq, radius_sq, &self.deleted, &mut results);
        }

        for entry in &self.buffer {
            let s = hyperbolic_ratio_sq(center, center_norm_sq, &entry.point, entry.norm_sq);
            if s <= radius_sq {
                results.push((entry.unique_id.clone(), center.hyperbolic_distance(&entry.point)));
            }
        }

        results
    }

    /// Find the k nearest live entries to a point.
    /// Returns results sorted by ascending distance.
    pub fn find_nearest(&self, point: &HyperbolicPoint, k: usize) -> Vec<(String, FixedPoint)> {
        if k == 0 { return Vec::new(); }

        // Candidates and tau hold SQUARED Mobius ratios (the sqrt-free,
        // atanh-free proxy — one dot product per score thanks to the cached
        // norms). Monotone in distance, so ranking is identical; only the k
        // winners pay the exact kernel, below.
        let query_norm_sq = point.coords().length_squared();
        let mut candidates: Vec<(FixedPoint, &BucketEntry)> = Vec::with_capacity(k + 1);
        // Squared ratios are capped strictly below 1, so 1 clears every
        // reachable score (the proxy-space analogue of the old 200-distance
        // sentinel); no prune can fire against it.
        let mut tau = FixedPoint::from_int(1);

        // Search the VP-tree (O(log n) with pruning)
        if let Some(ref root) = self.root {
            Self::search_knn(root, point, query_norm_sq, k, &self.deleted, &mut candidates, &mut tau);
        }

        // Linear scan of the buffer, in the same proxy space as the tree
        // search so the shared candidate list stays in one unit.
        for entry in &self.buffer {
            let s = hyperbolic_ratio_sq(point, query_norm_sq, &entry.point, entry.norm_sq);
            if candidates.len() < k || s < tau {
                candidates.push((s, entry));
                candidates.sort_by(|a, b| cmp_fp(a.0, b.0).then_with(|| a.1.unique_id.cmp(&b.1.unique_id)));
                if candidates.len() > k {
                    candidates.truncate(k);
                }
                if candidates.len() == k {
                    tau = candidates.last().unwrap().0;
                }
            }
        }

        // Winners get exact distances via the full kernel — bit-identical
        // values to the pre-proxy code, paid k times instead of per entry.
        candidates
            .into_iter()
            .map(|(_, e)| (e.unique_id.clone(), point.hyperbolic_distance(&e.point)))
            .collect()
    }

    // ---- Internal VP-tree machinery ----

    /// Rebuild the VP-tree from all live entries.
    fn rebuild(&mut self) {
        let mut entries = Vec::with_capacity(self.tree_size + self.buffer.len());

        // Collect live entries from the existing tree
        if let Some(root) = self.root.take() {
            Self::collect_live(*root, &self.deleted, &mut entries);
        }

        // Drain the buffer
        entries.append(&mut self.buffer);

        self.deleted.clear();
        self.tree_size = entries.len();
        self.root = Self::build_tree(entries);
    }

    /// Recursively collect live entries from a VP-tree, consuming nodes.
    fn collect_live(node: VPNode, deleted: &HashSet<String>, out: &mut Vec<BucketEntry>) {
        if !deleted.contains(&node.entry.unique_id) {
            out.push(node.entry);
        }
        if let Some(left) = node.left {
            Self::collect_live(*left, deleted, out);
        }
        if let Some(right) = node.right {
            Self::collect_live(*right, deleted, out);
        }
    }

    /// The live entry whose hyperbolic distance from `center` is greatest,
    /// as `(unique_id, distance)`, or `None` if there are no live entries.
    ///
    /// Linear in the number of live entries (tree + buffer). Callers use this
    /// to recompute a bucket's effective pruning radius exactly after the
    /// farthest node is removed, so the bound shrinks back under churn rather
    /// than staying permanently inflated by a since-deleted outlier.
    pub fn farthest_from(&self, center: &HyperbolicPoint) -> Option<(String, FixedPoint)> {
        // Rank by the Mobius ratio: monotone in distance, so the argmax is
        // identical, at ~8.3 us against ~20.5 us per entry. This walks EVERY
        // live entry, so the saving is proportional to bucket size. The
        // winner is converted back before returning — callers use the value
        // as a real pruning radius, so it must be a true distance.
        let center_norm_sq = center.coords().length_squared();
        let mut best: Option<(FixedPoint, String, HyperbolicPoint)> = None;
        let mut consider = |entry: &BucketEntry| {
            let s = hyperbolic_ratio_sq(center, center_norm_sq, &entry.point, entry.norm_sq);
            if best.as_ref().is_none_or(|(m, _, _)| s > *m) {
                best = Some((s, entry.unique_id.clone(), entry.point.clone()));
            }
        };
        for entry in &self.buffer {
            consider(entry);
        }
        if let Some(ref root) = self.root {
            Self::visit_live(root, &self.deleted, &mut consider);
        }
        // The winner's exact distance, bit-identical to the pre-proxy code —
        // callers use it as a real pruning radius.
        best.map(|(_, id, pt)| (id, center.hyperbolic_distance(&pt)))
    }

    /// Visit each live (non-lazily-deleted) entry in the tree, borrowing.
    fn visit_live<F: FnMut(&BucketEntry)>(node: &VPNode, deleted: &HashSet<String>, f: &mut F) {
        if !deleted.contains(&node.entry.unique_id) {
            f(&node.entry);
        }
        if let Some(ref left) = node.left {
            Self::visit_live(left, deleted, f);
        }
        if let Some(ref right) = node.right {
            Self::visit_live(right, deleted, f);
        }
    }

    /// Build a balanced VP-tree from a set of entries.
    ///
    /// Algorithm (Yianilos 1993):
    /// 1. Pick a vantage point (first entry for determinism)
    /// 2. Compute distances from VP to all other entries
    /// 3. Find the median distance
    /// 4. Partition: entries closer than median go left, rest go right
    /// 5. Recurse on each partition
    fn build_tree(mut entries: Vec<BucketEntry>) -> Option<Box<VPNode>> {
        if entries.is_empty() {
            return None;
        }

        if entries.len() == 1 {
            return Some(Box::new(VPNode {
                entry: entries.remove(0),
                median: FixedPoint::from_int(0),
                left: None,
                right: None,
            }));
        }

        // Pick vantage point (first entry, deterministic)
        let vp = entries.swap_remove(0);

        // Rank by the Möbius RATIO, not the distance. atanh is strictly
        // monotone on [0,1), so ratio ordering == distance ordering and the
        // tree built from it is IDENTICAL — but the ratio skips the
        // transcendental (~20.5 us -> ~8.3 us per pair). Build is O(m log m)
        // of these, so it dominates insert cost; the median is converted back
        // to a true distance once per node, which is what search compares
        // against, so pruning is untouched.
        let mut with_dists: Vec<(BucketEntry, FixedPoint)> = entries
            .into_iter()
            .map(|e| {
                let s = hyperbolic_ratio_sq(&vp.point, vp.norm_sq, &e.point, e.norm_sq);
                (e, s)
            })
            .collect();

        // Sort by squared ratio to find the median (same order as by
        // distance — squaring is monotone on non-negatives).
        with_dists.sort_by(|a, b| cmp_fp(a.1, b.1));

        // Stored as a SQUARED ratio: build and both searches now share one
        // proxy space end to end, and the build never pays a sqrt at all.
        let median = with_dists[with_dists.len() / 2].1;

        // Partition: strictly less than median -> left, rest -> right
        let (left_vec, right_vec): (Vec<_>, Vec<_>) = with_dists
            .into_iter()
            .partition(|(_, d)| *d < median);

        let left = Self::build_tree(left_vec.into_iter().map(|(e, _)| e).collect());
        let right = Self::build_tree(right_vec.into_iter().map(|(e, _)| e).collect());

        Some(Box::new(VPNode {
            entry: vp,
            median,
            left,
            right,
        }))
    }

    /// Recursive range search on the VP-tree, in SQUARED-ratio proxy space.
    ///
    /// `radius_sq` is `distance_to_ratio(radius)²`, converted once by the
    /// caller. Scores are sqrt-free (cached norms); pruning delegates to
    /// `sq_ratio_separation_exceeds`, the conservatively-bounded tanh
    /// identity shared with the metric tree. Hits pay the exact kernel once
    /// each, so reported distances are bit-identical to the pre-proxy code.
    fn search_radius(
        node: &VPNode,
        center: &HyperbolicPoint,
        center_norm_sq: FixedPoint,
        radius_sq: FixedPoint,
        deleted: &HashSet<String>,
        results: &mut Vec<(String, FixedPoint)>,
    ) {
        let s = hyperbolic_ratio_sq(center, center_norm_sq, &node.entry.point, node.entry.norm_sq);

        if s <= radius_sq && !deleted.contains(&node.entry.unique_id) {
            results.push((
                node.entry.unique_id.clone(),
                center.hyperbolic_distance(&node.entry.point),
            ));
        }

        // Descend left unless provably d - median > radius.
        if let Some(ref left) = node.left {
            let prune = s > node.median && sq_ratio_separation_exceeds(s, node.median, radius_sq);
            if !prune {
                Self::search_radius(left, center, center_norm_sq, radius_sq, deleted, results);
            }
        }

        // Descend right unless provably median - d > radius.
        if let Some(ref right) = node.right {
            let prune = node.median > s && sq_ratio_separation_exceeds(node.median, s, radius_sq);
            if !prune {
                Self::search_radius(right, center, center_norm_sq, radius_sq, deleted, results);
            }
        }
    }

    /// Recursive KNN search on the VP-tree, in SQUARED-ratio proxy space.
    ///
    /// `candidates` and the shrinking `tau` hold squared Mobius ratios —
    /// sqrt-free and atanh-free per score (one dot product, thanks to the
    /// cached norms). Monotone in distance, so ranking is identical; the
    /// exact kernel is paid only by the k winners, in `find_nearest`.
    /// Pruning delegates to `sq_ratio_separation_exceeds` — the same
    /// conservatively-bounded tanh-identity predicate the metric tree uses.
    /// Searches the closer subtree first for better early pruning.
    #[allow(clippy::too_many_arguments)]
    fn search_knn<'a>(
        node: &'a VPNode,
        center: &HyperbolicPoint,
        center_norm_sq: FixedPoint,
        k: usize,
        deleted: &HashSet<String>,
        candidates: &mut Vec<(FixedPoint, &'a BucketEntry)>,
        tau: &mut FixedPoint,
    ) {
        let s = hyperbolic_ratio_sq(center, center_norm_sq, &node.entry.point, node.entry.norm_sq);

        // Consider the vantage point
        if !deleted.contains(&node.entry.unique_id) {
            if candidates.len() < k || s < *tau {
                candidates.push((s, &node.entry));
                candidates.sort_by(|a, b| cmp_fp(a.0, b.0).then_with(|| a.1.unique_id.cmp(&b.1.unique_id)));
                if candidates.len() > k {
                    candidates.truncate(k);
                }
                if candidates.len() == k {
                    *tau = candidates.last().unwrap().0;
                }
            }
        }

        // Search the closer subtree first for tighter pruning
        let search_left_first = s < node.median;

        let prune_left = |s: FixedPoint, tau: FixedPoint| {
            s > node.median && sq_ratio_separation_exceeds(s, node.median, tau)
        };
        let prune_right = |s: FixedPoint, tau: FixedPoint| {
            node.median > s && sq_ratio_separation_exceeds(node.median, s, tau)
        };

        if search_left_first {
            if let Some(ref left) = node.left {
                if !prune_left(s, *tau) {
                    Self::search_knn(left, center, center_norm_sq, k, deleted, candidates, tau);
                }
            }
            if let Some(ref right) = node.right {
                if !prune_right(s, *tau) {
                    Self::search_knn(right, center, center_norm_sq, k, deleted, candidates, tau);
                }
            }
        } else {
            if let Some(ref right) = node.right {
                if !prune_right(s, *tau) {
                    Self::search_knn(right, center, center_norm_sq, k, deleted, candidates, tau);
                }
            }
            if let Some(ref left) = node.left {
                if !prune_left(s, *tau) {
                    Self::search_knn(left, center, center_norm_sq, k, deleted, candidates, tau);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HyperbolicHashBucket
// ---------------------------------------------------------------------------

/// Hash bucket for the hyperbolic hash table.
///
/// Each bucket's VP-tree is protected by a Mutex for per-bucket
/// concurrent spatial index access.
#[derive(Debug)]
pub struct HyperbolicHashBucket {
    /// The hyperbolic region for this bucket
    region: HyperbolicRegion,
    /// Position signature for validation
    position_signature: Vec<i32>,
    /// Additional validation metrics
    _metrics: Vec<FixedPoint>,
    /// VP-tree spatial index for nodes in this bucket (per-bucket lock)
    vp_tree: Mutex<VPTree>,
    /// Effective pruning-radius bookkeeping. Deep nodes are assigned to the
    /// nearest bucket even when they fall outside every bucket's nominal
    /// region, so range/KNN pruning must widen past the nominal radius to
    /// reach them (see `EffRadius`). Held under its own mutex.
    eff: Mutex<EffRadius>,
}

/// Effective-radius bookkeeping for a bucket.
///
/// The pruning bound is the nominal region radius widened to reach the
/// farthest live member. Unlike a monotone high-water mark, it shrinks back
/// when that farthest member is removed: a bucket that briefly held a deep
/// outlier does not keep scanning a stale-wide radius forever under churn.
/// The shrink is exact — on removal of the bound-defining node the bound is
/// recomputed from the remaining live members via `VPTree::farthest_from`.
#[derive(Clone, Debug)]
struct EffRadius {
    /// The bucket's nominal region radius; the bound never drops below this.
    nominal: FixedPoint,
    /// Current effective radius: `max(nominal, farthest live-member distance)`.
    current: FixedPoint,
    /// `unique_id` of the member whose center-distance defines `current`, or
    /// `None` when `current == nominal` (no out-of-region member). Tracking
    /// the defining node lets removals recompute the bound only when the node
    /// that set it is the one being removed — O(1) for every other removal.
    max_uid: Option<String>,
}

impl EffRadius {
    fn new(nominal: FixedPoint) -> Self {
        Self { nominal, current: nominal, max_uid: None }
    }
}

impl Clone for HyperbolicHashBucket {
    fn clone(&self) -> Self {
        // Acquire and release each bucket lock in turn — never hold both at
        // once — so this can never form a lock cycle with `forget_node`
        // (which holds `eff` while taking `vp_tree`).
        let vp_tree = self.vp_tree.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let eff = self.eff.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Self {
            region: self.region.clone(),
            position_signature: self.position_signature.clone(),
            _metrics: self._metrics.clone(),
            vp_tree: Mutex::new(vp_tree),
            eff: Mutex::new(eff),
        }
    }
}

impl HyperbolicHashBucket {
    /// Create a new hyperbolic hash bucket.
    pub fn new(region: HyperbolicRegion, position_signature: Vec<i32>) -> Self {
        let mut metrics = Vec::new();

        let center = region.center();
        metrics.push(center.euclidean_norm());

        let sum_squares = center.coords().iter().enumerate().fold(
            FixedPoint::from_int(0),
            |acc, (_i, &x)| acc + x * x
        );
        metrics.push(sum_squares);

        let nominal_radius = region.radius();
        Self {
            region,
            position_signature,
            _metrics: metrics,
            vp_tree: Mutex::new(VPTree::new()),
            eff: Mutex::new(EffRadius::new(nominal_radius)),
        }
    }

    /// The pruning radius for range/KNN queries: the nominal region radius,
    /// widened to cover the farthest node currently registered in this bucket.
    pub fn effective_radius(&self) -> FixedPoint {
        self.eff.lock().unwrap_or_else(|e| e.into_inner()).current
    }

    /// Record the center-distance of a newly registered node, widening the
    /// effective radius (and remembering the node) if it lies farther than the
    /// current bound.
    fn note_node_distance(&self, unique_id: &str, center_dist: FixedPoint) {
        let mut e = self.eff.lock().unwrap_or_else(|e| e.into_inner());
        if center_dist > e.current {
            e.current = center_dist;
            e.max_uid = Some(unique_id.to_string());
        }
    }

    /// Drop a removed node's contribution to the effective radius.
    ///
    /// If the removed node is the one currently defining the bound, recompute
    /// the exact bound from the remaining live members, so a deleted outlier
    /// no longer inflates query pruning. The node must already be gone from
    /// the VP-tree when this is called. O(bucket size) only on removal of the
    /// bound-defining node; O(1) for every other removal.
    ///
    /// Locks `eff` then `vp_tree`; this is the only site that holds two bucket
    /// locks at once, and it always takes them in this order (see `Clone`).
    fn forget_node(&self, unique_id: &str) {
        let mut e = self.eff.lock().unwrap_or_else(|e| e.into_inner());
        if e.max_uid.as_deref() != Some(unique_id) {
            return;
        }
        let tree = self.vp_tree.lock().unwrap_or_else(|e| e.into_inner());
        match tree.farthest_from(self.region.center()) {
            Some((uid, dist)) if dist > e.nominal => {
                e.current = dist;
                e.max_uid = Some(uid);
            }
            _ => {
                e.current = e.nominal;
                e.max_uid = None;
            }
        }
    }

    /// Check if a point belongs to this bucket.
    pub fn contains(&self, point: &HyperbolicPoint, poincare_disk: &PoincareDisk) -> bool {
        self.region.contains(point, poincare_disk)
    }

    /// Get the region for this bucket.
    pub fn region(&self) -> &HyperbolicRegion {
        &self.region
    }

    /// Get the position signature.
    pub fn position_signature(&self) -> &[i32] {
        &self.position_signature
    }

    /// Perform a quick validation check.
    pub fn quick_validate(&self, point: &HyperbolicPoint) -> bool {
        self.region.quick_validate(point)
    }
}

// ---------------------------------------------------------------------------
// HyperbolicHashTable
// ---------------------------------------------------------------------------

/// Hyperbolic Hash Table for O(1) lookups in hyperbolic space.
///
/// Buckets partition the Poincaré disk into ~61 fixed regions. Each bucket
/// contains a VP-tree that provides O(log n) spatial queries within the bucket.
/// Combined with the O(1) bucket selection, total query time is O(log(n/B))
/// where B is the bucket count.
#[derive(Clone)]
pub struct HyperbolicHashTable {
    /// Poincaré disk model
    poincare_disk: PoincareDisk,
    /// Hash buckets organized by hash value
    buckets: HashMap<String, HyperbolicHashBucket>,
    /// Map from position signature to hash
    signature_map: HashMap<Vec<i32>, String>,
    /// Reverse map: unique_id -> bucket_hash (for O(1) unregistration)
    node_to_bucket: DashMap<String, String>,
}

impl HyperbolicHashTable {
    /// Create a new hyperbolic hash table with the specified dimension.
    pub fn new(dimension: usize) -> Self {
        let poincare_disk = PoincareDisk::new(dimension);

        let mut table = Self {
            poincare_disk,
            buckets: HashMap::new(),
            signature_map: HashMap::new(),
            node_to_bucket: DashMap::new(),
        };

        table.initialize_buckets();
        table
    }

    /// Initialize buckets for locality-sensitive hashing.
    fn initialize_buckets(&mut self) {
        let dimension = self.poincare_disk.dimension();

        // Distances from the origin (all as exact rationals)
        let distances = [
            FixedPoint::from_int(0),                                          // Origin
            constants::half(),                                                 // 0.5
            FixedPoint::from_int(1),                                          // 1.0
            FixedPoint::from_int(3) / FixedPoint::from_int(2),               // 1.5
            FixedPoint::from_int(2),                                          // 2.0
        ];

        let directions_per_distance = [
            1,               // Origin (just 1 point)
            dimension * 2,   // Close
            dimension * 3,   // Medium
            dimension * 4,   // Far
            dimension * 5,   // Very far
        ];

        for (dist_idx, &distance) in distances.iter().enumerate() {
            let num_directions = directions_per_distance[dist_idx];

            // Special case for the origin
            if dist_idx == 0 {
                let origin = self.poincare_disk.origin();
                let region = HyperbolicRegion::new(origin.clone(), constants::region_radius());

                let position_signature = vec![0; dimension];

                let bucket = HyperbolicHashBucket::new(region, position_signature.clone());
                let signature = self.compute_geometric_signature(&origin);
                let hash = self.compute_stable_hash(&signature);

                self.buckets.insert(hash.clone(), bucket);
                self.signature_map.insert(position_signature, hash);

                continue;
            }

            for dir_idx in 0..num_directions {
                let direction = self.generate_direction_vector(dir_idx, num_directions);

                let center = self.poincare_disk.point_at_distance_from_origin(
                    &direction, distance
                );

                // Radius: 1/5 + 1/10 * distance (pure FixedPoint)
                let one_fifth = FixedPoint::from_int(1) / FixedPoint::from_int(5);
                let one_tenth = FixedPoint::from_int(1) / FixedPoint::from_int(10);
                let radius = one_fifth + one_tenth * distance;
                let region = HyperbolicRegion::new(center.clone(), radius);

                let position_signature = self.generate_position_signature(&center);

                let bucket = HyperbolicHashBucket::new(region, position_signature.clone());
                let signature = self.compute_geometric_signature(&center);
                let hash = self.compute_stable_hash(&signature);

                self.buckets.insert(hash.clone(), bucket);
                self.signature_map.insert(position_signature, hash);
            }
        }
    }

    /// Generate a direction vector for bucket initialization.
    fn generate_direction_vector(&self, index: usize, total: usize) -> FixedVector {
        let dimension = self.poincare_disk.dimension();
        let mut direction = FixedVector::new(dimension);

        // For 2D, use angles evenly distributed around a circle
        if dimension == 2 {
            let angle = constants::two_pi()
                * FixedPoint::from_int(index as i32)
                / FixedPoint::from_int(total as i32);
            let (sin_a, cos_a) = angle.sincos();
            direction[0] = cos_a;
            direction[1] = sin_a;
            return direction;
        }

        // For higher dimensions, use golden spiral method
        let phi = constants::golden_angle();

        // Use (index + 1) to avoid zero vector when index == 0
        let idx = FixedPoint::from_int((index + 1) as i32);
        for i in 0..dimension {
            let phase = idx * phi * FixedPoint::from_int((i + 1) as i32);
            direction[i] = phase.sin();
        }

        let norm_sq = direction.dot(&direction);
        if norm_sq > constants::epsilon() {
            direction.normalize();
        } else {
            // Fallback: unit vector along first axis
            direction[0] = FixedPoint::from_int(1);
        }

        direction
    }

    /// Generate a position signature for a point.
    fn generate_position_signature(&self, point: &HyperbolicPoint) -> Vec<i32> {
        let dimension = self.poincare_disk.dimension();
        let mut signature = Vec::with_capacity(dimension);

        for i in 0..dimension {
            signature.push(constants::quantize_position(point.coords()[i]));
        }

        signature
    }

    /// Compute a geometric signature for a point.
    /// Uses x * (1 + tanh(x)) to produce a sign-sensitive signature
    /// (plain x*tanh(x) is even and loses sign information).
    fn compute_geometric_signature(&self, point: &HyperbolicPoint) -> Vec<i32> {
        let dimension = self.poincare_disk.dimension();
        let mut signature = Vec::with_capacity(dimension);
        let one = FixedPoint::from_int(1);

        for i in 0..dimension {
            let x = point.coords()[i];
            let transformed = x * (one + x.tanh());
            signature.push(constants::quantize_1000(transformed));
        }

        signature
    }

    /// Compute a stable hash value from a geometric signature.
    fn compute_stable_hash(&self, signature: &[i32]) -> String {
        let mut hasher = Sha3_512::new();

        for &value in signature {
            hasher.update(value.to_le_bytes());
        }

        let hash = hasher.finalize();
        hex::encode(&hash[..16])
    }

    /// Find the bucket containing a point.
    ///
    /// Uses a three-pass strategy:
    /// 1. Exact signature match (O(1) HashMap lookup)
    /// 2. Euclidean-distance prefilter: sort buckets by cheap Euclidean²
    ///    distance to their center, then check hyperbolic containment
    ///    starting from the nearest. Typically finds the match in 1-3 checks
    ///    (~20-60µs) instead of scanning all ~61 buckets (~1.2ms).
    /// 3. quick_validate fallback for edge cases.
    pub fn find_bucket(&self, point: &HyperbolicPoint) -> Option<String> {
        // Pass 1: exact signature match (O(1))
        let position_signature = self.generate_position_signature(point);
        if let Some(hash) = self.signature_map.get(&position_signature) {
            return Some(hash.clone());
        }

        // Pass 2: Euclidean prefilter — check nearest bucket centers first.
        // Euclidean distance² is O(d) pure arithmetic (~100ns per bucket),
        // sorting 61 entries is ~1µs. Then we check hyperbolic containment
        // on the nearest candidates, typically matching on the 1st or 2nd.
        let mut candidates: Vec<(&String, FixedPoint)> = self.buckets.iter()
            .map(|(hash, bucket)| {
                (hash, euclidean_distance_sq(point, bucket.region().center()))
            })
            .collect();
        // Tie-break equal distances by hash so assignment is deterministic
        // regardless of HashMap iteration order.
        candidates.sort_unstable_by(|a, b| cmp_fp(a.1, b.1).then_with(|| a.0.cmp(b.0)));

        for (hash, _) in &candidates {
            if let Some(bucket) = self.buckets.get(*hash) {
                if bucket.contains(point, &self.poincare_disk) {
                    return Some((*hash).clone());
                }
            }
        }

        // Pass 3: quick_validate fallback (for points far from any bucket
        // center). The pre-sorted candidate list keeps this pass deterministic
        // too — HashMap iteration order must never pick the bucket.
        for (hash, _) in &candidates {
            if let Some(bucket) = self.buckets.get(*hash) {
                if bucket.quick_validate(point) {
                    return Some((*hash).clone());
                }
            }
        }

        None
    }

    /// Create a geometric signature for a point.
    ///
    /// Uses the point's actual coordinates for the position signature (not the
    /// bucket center), ensuring unique signatures for distinct points even when
    /// they fall in the same geometric bucket. The hash field identifies the
    /// bucket for O(1) locality lookup.
    pub fn create_signature(&self, point: &HyperbolicPoint, level: u32) -> Option<GeometricSignature> {
        // Position signature from the actual point (unique per point)
        let position_signature = self.generate_position_signature(point);

        // Bucket hash for O(1) locality lookup
        let hash = if let Some(bucket_hash) = self.find_bucket(point) {
            bucket_hash
        } else {
            // Fallback: hash from geometric signature
            let geo_sig = self.compute_geometric_signature(point);
            self.compute_stable_hash(&geo_sig)
        };

        Some(GeometricSignature::new(hash, level, position_signature))
    }

    /// Check if a hyperbolic point is valid.
    pub fn validate_point(&self, point: &HyperbolicPoint) -> bool {
        let norm = point.euclidean_norm();
        if norm >= FixedPoint::from_int(1) {
            return false;
        }

        self.find_bucket(point).is_some()
    }

    /// Get the Poincaré disk.
    pub fn poincare_disk(&self) -> &PoincareDisk {
        &self.poincare_disk
    }

    /// Get the number of buckets.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Register a node in the spatial index.
    /// Returns the bucket hash the node was placed in.
    pub fn register_node(&self, point: &HyperbolicPoint, unique_id: &str, level: u32) -> Option<String> {
        self.register_node_with_hint(point, unique_id, level, None)
    }

    /// Register a node in the spatial index with an optional bucket hash hint.
    ///
    /// When `bucket_hint` is provided (e.g. from a prior `create_signature` call),
    /// skips the expensive `find_bucket` lookup entirely. Falls back to `find_bucket`
    /// if the hint is invalid.
    pub fn register_node_with_hint(&self, point: &HyperbolicPoint, unique_id: &str, level: u32, bucket_hint: Option<&str>) -> Option<String> {
        // Prevent duplicate registration
        if self.node_to_bucket.contains_key(unique_id) {
            return self.node_to_bucket.get(unique_id).map(|r| r.value().clone());
        }

        // Try the hint first (avoids second find_bucket call on the insert path)
        let bucket_hash = match bucket_hint {
            Some(hint) if self.buckets.contains_key(hint) => hint.to_string(),
            _ => self.find_bucket(point)?,
        };

        if let Some(bucket) = self.buckets.get(&bucket_hash) {
            // Deep nodes land in buckets whose nominal region doesn't contain
            // them; widen the bucket's pruning radius so range/KNN queries
            // never skip the bucket that actually holds them.
            let center_dist = self.poincare_disk.distance(point, bucket.region.center());
            bucket.note_node_distance(unique_id, center_dist);
            bucket.vp_tree.lock().unwrap_or_else(|e| e.into_inner()).insert(BucketEntry::new(unique_id.to_string(), point.clone(), level));
        }
        self.node_to_bucket.insert(unique_id.to_string(), bucket_hash.clone());
        Some(bucket_hash)
    }

    /// Remove a node from the spatial index.
    /// O(1) via node_to_bucket reverse map — only touches the correct bucket.
    pub fn unregister_node(&self, unique_id: &str) {
        if let Some((_, bucket_hash)) = self.node_to_bucket.remove(unique_id) {
            if let Some(bucket) = self.buckets.get(&bucket_hash) {
                // Remove from the spatial index first (releasing the vp_tree
                // lock), then let the bucket recompute its effective radius
                // from the remaining live members if this node defined it.
                bucket.vp_tree.lock().unwrap_or_else(|e| e.into_inner()).remove(unique_id);
                bucket.forget_node(unique_id);
            }
        }
    }

    /// Find all nodes within a hyperbolic radius of a center point.
    ///
    /// 1. Quick-reject entire buckets whose centers are beyond radius + bucket_radius.
    /// 2. Within each candidate bucket, use the VP-tree's O(log n) range query.
    pub fn find_nodes_in_radius(&self, center: &HyperbolicPoint, radius: FixedPoint) -> Vec<(String, FixedPoint)> {
        let mut results = Vec::new();

        for bucket in self.buckets.values() {
            // Quick reject: if every node the bucket can hold is too far, skip.
            // Uses the effective radius (widened by out-of-region nodes), not
            // the nominal region radius.
            let bucket_center_dist = self.poincare_disk.distance(
                center, bucket.region.center()
            );
            if bucket_center_dist > radius + bucket.effective_radius() {
                continue;
            }

            // VP-tree range query within this bucket
            let bucket_results = bucket.vp_tree.lock().unwrap_or_else(|e| e.into_inner()).find_in_radius(center, radius);
            results.extend(bucket_results);
        }

        results
    }

    /// Find the k nearest nodes to a point.
    ///
    /// Sorts buckets by distance to query point, queries each bucket's VP-tree
    /// for its k-nearest, and merges results with proper early termination:
    /// stops when the next bucket's minimum possible distance exceeds the
    /// k-th candidate's distance.
    pub fn find_nearest_nodes(&self, point: &HyperbolicPoint, k: usize) -> Vec<(String, FixedPoint)> {
        if k == 0 { return Vec::new(); }

        // Sort buckets by the minimum possible distance of any member node:
        // center distance minus the bucket's EFFECTIVE radius (widened by
        // out-of-region nodes), floored at zero. With heterogeneous radii,
        // center distance alone is not monotone in this bound, and the early
        // `break` below is only sound when buckets are ordered by it.
        // Tie-break by hash so tied buckets scan in a deterministic order.
        let zero = FixedPoint::from_int(0);
        let mut bucket_dists: Vec<(&String, FixedPoint)> = self.buckets.iter()
            .map(|(hash, bucket)| {
                let d = self.poincare_disk.distance(point, bucket.region.center());
                let r = bucket.effective_radius();
                let min_possible = if d > r { d - r } else { zero };
                (hash, min_possible)
            })
            .collect();
        bucket_dists.sort_by(|a, b| cmp_fp(a.1, b.1).then_with(|| a.0.cmp(b.0)));

        let mut candidates: Vec<(String, FixedPoint)> = Vec::new();

        for (hash, min_possible) in &bucket_dists {
            // Early termination: once we have k candidates, no later bucket
            // (sorted by min_possible) can contain anything closer.
            if candidates.len() >= k {
                let kth_dist = candidates.last().unwrap().1;
                if *min_possible > kth_dist {
                    break;
                }
            }

            if let Some(bucket) = self.buckets.get(*hash) {
                // VP-tree KNN within this bucket
                let bucket_results = bucket.vp_tree.lock().unwrap_or_else(|e| e.into_inner()).find_nearest(point, k);

                // Merge with global candidates
                for result in bucket_results {
                    candidates.push(result);
                }

                // Sort and keep top k
                candidates.sort_by(|a, b| cmp_fp(a.1, b.1));
                candidates.truncate(k);
            }
        }

        candidates
    }

    /// Verify the integrity of the hash table.
    pub fn verify_integrity(&self) -> bool {
        if self.buckets.is_empty() {
            return false;
        }

        for (sig, hash) in &self.signature_map {
            if !self.buckets.contains_key(hash) {
                return false;
            }

            let bucket = &self.buckets[hash];
            if bucket.position_signature() != sig.as_slice() {
                return false;
            }
        }

        true
    }
}

impl Debug for HyperbolicHashTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "HyperbolicHashTable(dim={}, buckets={}, nodes={})",
               self.poincare_disk.dimension(), self.buckets.len(),
               self.node_to_bucket.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_table_creation() {
        let table = HyperbolicHashTable::new(2);
        assert_eq!(table.poincare_disk().dimension(), 2);
        assert!(table.bucket_count() > 0);
    }

    #[test]
    fn test_geometric_signature() {
        let table = HyperbolicHashTable::new(2);
        let point = table.poincare_disk().point_from_f32_slice(&[0.5, 0.0]);

        let signature = table.create_signature(&point, 0).unwrap();
        assert_eq!(signature.level(), 0);
        assert!(!signature.hash().is_empty());
        assert!(!signature.position_signature().is_empty());
    }

    #[test]
    fn test_bucket_finding() {
        let table = HyperbolicHashTable::new(2);
        let origin = table.poincare_disk().origin();

        let bucket_hash = table.find_bucket(&origin);
        assert!(bucket_hash.is_some());
    }

    #[test]
    fn test_point_validation() {
        let table = HyperbolicHashTable::new(2);

        let valid_point = table.poincare_disk().point_from_f32_slice(&[0.5, 0.0]);
        assert!(table.validate_point(&valid_point));

        let projected_point = table.poincare_disk().point_from_f32_slice(&[1.5, 0.0]);
        assert!(table.validate_point(&projected_point));
    }

    #[test]
    fn test_hyperbolic_region() {
        let disk = PoincareDisk::new(2);
        let center = disk.point_from_f32_slice(&[0.5, 0.0]);
        let radius = constants::half();

        let region = HyperbolicRegion::new(center.clone(), radius);

        assert!(region.contains(&center, &disk));
        assert!(!region.contains(&disk.origin(), &disk));

        let far_point = disk.point_from_f32_slice(&[0.8, 0.0]);
        assert!(!region.contains(&far_point, &disk));
    }

    #[test]
    fn test_hash_bucket() {
        let disk = PoincareDisk::new(2);
        let center = disk.point_from_f32_slice(&[0.5, 0.0]);
        let radius = constants::half();

        let region = HyperbolicRegion::new(center.clone(), radius);
        let position_signature = vec![500, 0];

        let bucket = HyperbolicHashBucket::new(region, position_signature);

        assert!(bucket.contains(&center, &disk));
        assert!(bucket.quick_validate(&center));
    }

    #[test]
    fn test_integrity_verification() {
        let table = HyperbolicHashTable::new(2);
        assert!(table.verify_integrity());
    }

    // ---- VP-tree tests ----

    #[test]
    fn test_vp_tree_empty() {
        let vp = VPTree::new();
        assert!(vp.is_empty());
        assert_eq!(vp.live_count(), 0);

        let origin = HyperbolicPoint::origin(2);
        let results = vp.find_in_radius(&origin, FixedPoint::from_int(10));
        assert!(results.is_empty());

        let nearest = vp.find_nearest(&origin, 5);
        assert!(nearest.is_empty());
    }

    #[test]
    fn test_vp_tree_insert_and_find() {
        let disk = PoincareDisk::new(2);
        let mut vp = VPTree::new();

        // Insert several points at different positions
        let points: Vec<(&str, [f32; 2])> = vec![
            ("a", [0.1, 0.0]),
            ("b", [0.2, 0.0]),
            ("c", [0.3, 0.0]),
            ("d", [0.0, 0.1]),
            ("e", [0.0, 0.2]),
        ];

        for (id, coords) in &points {
            vp.insert(BucketEntry::new(id.to_string(), disk.point_from_f32_slice(coords), 0));
        }

        assert_eq!(vp.live_count(), 5);

        // Find nearest to origin — should return "a" and "d" first (closest)
        let origin = disk.origin();
        let nearest = vp.find_nearest(&origin, 2);
        assert_eq!(nearest.len(), 2);
        // Distances should be in ascending order
        assert!(nearest[0].1 <= nearest[1].1);

        // Range query with large radius should find all
        let all = vp.find_in_radius(&origin, FixedPoint::from_int(10));
        assert_eq!(all.len(), 5);

        // Range query with tiny radius should find none (or very few)
        let tiny = vp.find_in_radius(&origin, FixedPoint::from_int(1) / FixedPoint::from_int(10000));
        assert!(tiny.len() <= 1);
    }

    #[test]
    fn test_vp_tree_remove() {
        let disk = PoincareDisk::new(2);
        let mut vp = VPTree::new();

        vp.insert(BucketEntry::new("x".to_string(), disk.point_from_f32_slice(&[0.1, 0.0]), 0));
        vp.insert(BucketEntry::new("y".to_string(), disk.point_from_f32_slice(&[0.2, 0.0]), 0));

        assert_eq!(vp.live_count(), 2);

        vp.remove("x");
        assert_eq!(vp.live_count(), 1);

        // "x" should not appear in results
        let origin = disk.origin();
        let results = vp.find_in_radius(&origin, FixedPoint::from_int(10));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "y");
    }

    #[test]
    fn test_vp_tree_rebuild_on_buffer_threshold() {
        let disk = PoincareDisk::new(2);
        let mut vp = VPTree::new();

        // Insert more than VP_BUFFER_THRESHOLD entries to trigger a rebuild
        for i in 0..(VP_BUFFER_THRESHOLD + 5) {
            let angle = constants::two_pi()
                * FixedPoint::from_int(i as i32)
                / FixedPoint::from_int((VP_BUFFER_THRESHOLD + 5) as i32);
            let r = FixedPoint::from_int(3) / FixedPoint::from_int(10);
            let mut coords = FixedVector::new(2);
            let (sin_a, cos_a) = angle.sincos();
            coords[0] = r * cos_a;
            coords[1] = r * sin_a;

            vp.insert(BucketEntry::new(format!("node_{}", i), HyperbolicPoint::new(coords), 0));
        }

        // After rebuild, tree should be structured (root is Some)
        assert!(vp.root.is_some());
        assert_eq!(vp.live_count(), VP_BUFFER_THRESHOLD + 5);

        // Queries should still work correctly
        let origin = disk.origin();
        let all = vp.find_in_radius(&origin, FixedPoint::from_int(10));
        assert_eq!(all.len(), VP_BUFFER_THRESHOLD + 5);
    }

    #[test]
    fn test_vp_tree_knn_ordering() {
        let disk = PoincareDisk::new(2);
        let mut vp = VPTree::new();

        // Insert points at known increasing distances from origin
        let distances = [0.05f32, 0.1, 0.2, 0.3, 0.5, 0.7];
        for (i, &d) in distances.iter().enumerate() {
            vp.insert(BucketEntry::new(format!("p{}", i), disk.point_from_f32_slice(&[d, 0.0]), 0));
        }

        let origin = disk.origin();
        let nearest = vp.find_nearest(&origin, 3);
        assert_eq!(nearest.len(), 3);

        // Verify ascending distance order
        for i in 1..nearest.len() {
            assert!(nearest[i].1 >= nearest[i - 1].1,
                "Results not sorted: {:?} >= {:?}", nearest[i].1, nearest[i - 1].1);
        }

        // The closest 3 should be p0, p1, p2 (distances 0.05, 0.1, 0.2)
        let ids: Vec<&str> = nearest.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"p0"));
        assert!(ids.contains(&"p1"));
        assert!(ids.contains(&"p2"));
    }

    #[test]
    fn test_register_unregister_with_vp_tree() {
        let table = HyperbolicHashTable::new(2);
        let disk_clone = table.poincare_disk().clone();

        let p1 = disk_clone.point_from_f32_slice(&[0.1, 0.0]);
        let p2 = disk_clone.point_from_f32_slice(&[0.2, 0.0]);
        let p3 = disk_clone.point_from_f32_slice(&[0.3, 0.0]);

        table.register_node(&p1, "node1", 0);
        table.register_node(&p2, "node2", 1);
        table.register_node(&p3, "node3", 1);

        // Should find all three with large radius
        let origin = disk_clone.origin();
        let results = table.find_nodes_in_radius(&origin, FixedPoint::from_int(10));
        assert!(results.len() >= 3, "Expected at least 3, got {}", results.len());

        // Unregister node2
        table.unregister_node("node2");

        // Should no longer find node2
        let results = table.find_nodes_in_radius(&origin, FixedPoint::from_int(10));
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(!ids.contains(&"node2"), "node2 should be unregistered");
        assert!(ids.contains(&"node1"));
        assert!(ids.contains(&"node3"));
    }

    #[test]
    fn test_find_nearest_with_early_termination() {
        let table = HyperbolicHashTable::new(2);
        let disk_clone = table.poincare_disk().clone();

        // Insert nodes at various distances
        let positions: Vec<(&str, [f32; 2])> = vec![
            ("close1", [0.05, 0.0]),
            ("close2", [0.0, 0.05]),
            ("mid1", [0.3, 0.0]),
            ("mid2", [0.0, 0.3]),
            ("far1", [0.7, 0.0]),
            ("far2", [0.0, 0.7]),
        ];

        for (id, coords) in &positions {
            let point = disk_clone.point_from_f32_slice(coords);
            table.register_node(&point, id, 0);
        }

        let origin = disk_clone.origin();
        let nearest = table.find_nearest_nodes(&origin, 2);
        assert_eq!(nearest.len(), 2);

        // The two closest should be close1 and close2
        let ids: Vec<&str> = nearest.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"close1"));
        assert!(ids.contains(&"close2"));

        // Verify ascending order
        assert!(nearest[0].1 <= nearest[1].1);
    }

    #[test]
    fn test_duplicate_registration_prevented() {
        let table = HyperbolicHashTable::new(2);
        let point = table.poincare_disk().point_from_f32_slice(&[0.1, 0.0]);

        let h1 = table.register_node(&point, "dup_node", 0);
        let h2 = table.register_node(&point, "dup_node", 0);

        // Both should return the same bucket hash
        assert_eq!(h1, h2);

        // Should only appear once in results
        let origin = table.poincare_disk().origin();
        let results = table.find_nodes_in_radius(&origin, FixedPoint::from_int(10));
        let count = results.iter().filter(|(id, _)| id == "dup_node").count();
        assert_eq!(count, 1, "Duplicate registration should be prevented");
    }

    #[test]
    fn effective_radius_returns_to_nominal_when_lone_outlier_removed() {
        // A deep node lands in the nearest bucket even though it sits well
        // outside that bucket's nominal region, widening the bucket's pruning
        // radius. When it is removed, the bound must shrink back to nominal —
        // a stale-wide radius would make every later query over-scan forever.
        let table = HyperbolicHashTable::new(2);
        let disk = table.poincare_disk().clone();

        let deep = disk.point_from_f32_slice(&[0.95, 0.0]);
        let bucket_hash = table.register_node(&deep, "deep", 5).unwrap();

        let nominal = table.buckets.get(&bucket_hash).unwrap().region().radius();
        let inflated = table.buckets.get(&bucket_hash).unwrap().effective_radius();
        assert!(
            inflated > nominal,
            "deep node should widen the bucket past nominal (inflated={:?}, nominal={:?})",
            inflated, nominal
        );

        table.unregister_node("deep");

        let after = table.buckets.get(&bucket_hash).unwrap().effective_radius();
        assert_eq!(
            after, nominal,
            "with the only out-of-region member gone, the bound must return to nominal"
        );
    }

    #[test]
    fn effective_radius_falls_to_second_farthest_not_nominal() {
        // Two out-of-region nodes in the same bucket: removing the farther one
        // must shrink the bound to the remaining one's distance — not all the
        // way to nominal (that would under-prune and drop it from queries),
        // and not stay at the removed node's distance (that would over-scan).
        let table = HyperbolicHashTable::new(2);
        let disk = table.poincare_disk().clone();

        // Same radial direction so both map to the same outermost bucket.
        let near_deep = disk.point_from_f32_slice(&[0.85, 0.0]);
        let far_deep = disk.point_from_f32_slice(&[0.97, 0.0]);

        let h_near = table.register_node(&near_deep, "near_deep", 4).unwrap();
        let h_far = table.register_node(&far_deep, "far_deep", 6).unwrap();

        // The scenario only bites when both share a bucket; skip otherwise
        // rather than assert on placement details this test doesn't own.
        if h_near != h_far {
            return;
        }

        let bucket = || table.buckets.get(&h_near).unwrap();
        let nominal = bucket().region().radius();
        let with_both = bucket().effective_radius();

        // Distance from the bucket center to the node that should define the
        // bound after the farther node is removed.
        let center = bucket().region().center().clone();
        let near_dist = center.hyperbolic_distance(&near_deep);

        table.unregister_node("far_deep");
        let after = bucket().effective_radius();

        assert!(after < with_both, "removing the farther node must shrink the bound");
        assert!(after > nominal, "the remaining out-of-region node must keep the bound above nominal");
        assert_eq!(after, near_dist, "the bound must equal the remaining node's center distance");
    }
}

