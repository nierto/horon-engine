//! Lazy per-slice semantic index cache with epoch invalidation.
//!
//! Design: `docs/SEMANTIC_INDEX.md`. A VP-tree only prunes correctly for the
//! exact `dim_range` (metric) it was built over, so each queried slice gets
//! its own tree, built lazily on first query and cached. A single semantic
//! epoch counter — bumped by every mutation that could change semantic query
//! results — invalidates all cached slices at once: staleness is one integer
//! comparison, and a stale tree is discarded and rebuilt, never mutated.
//!
//! Race safety: writers mutate first, then bump; builders read the epoch
//! *before* snapshotting. A write landing mid-build bumps the counter after
//! the builder's pre-read, so the cached tree is tagged with the old epoch
//! and discarded on the next query. Concurrent racing builders both build
//! the same deterministic tree; last insert wins — harmless.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use g_math::fixed_point::FixedPoint;

use crate::constants::SEMANTIC_INDEX_MAX_SLICES;
use crate::metric_tree::{EuclideanMetric, MetricVpTree};

/// A VP-tree over one dimension slice, tagged with the epoch it was built at.
pub struct SliceIndex {
    /// Semantic epoch at build time; stale when != the cache's current epoch.
    epoch: u64,
    /// The tree over `(unique_id, decoded slice coords)`.
    pub tree: MetricVpTree<Vec<FixedPoint>>,
}

/// Per-slice index cache keyed by `(dim_range.start, dim_range.end)`.
pub struct SemanticIndexCache {
    /// Bumped (after the mutation) by every semantic-relevant write.
    epoch: AtomicU64,
    /// Cached slice trees. BTreeMap for deterministic eviction order.
    slices: RwLock<BTreeMap<(usize, usize), Arc<SliceIndex>>>,
}

impl SemanticIndexCache {
    /// Create an empty cache at epoch 0.
    pub fn new() -> Self {
        Self {
            epoch: AtomicU64::new(0),
            slices: RwLock::new(BTreeMap::new()),
        }
    }

    /// Record a mutation that may change semantic query results.
    /// Callers must apply the mutation *before* bumping.
    pub fn bump(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
    }

    /// Current epoch (test/diagnostic visibility).
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    /// Get the cached tree for `dim_range`, or build one via `snapshot`
    /// (a coherent read of all nodes' decoded slice coordinates) and cache it.
    ///
    /// `snapshot` runs without any cache lock held, so concurrent queries
    /// on other slices are never blocked by a build.
    pub fn get_or_build<F>(&self, dim_range: &Range<usize>, snapshot: F) -> Arc<SliceIndex>
    where
        F: FnOnce() -> Vec<(String, Vec<FixedPoint>)>,
    {
        let key = (dim_range.start, dim_range.end);

        // Fast path: fresh cached tree.
        let current = self.epoch.load(Ordering::SeqCst);
        {
            let slices = self.slices.read().unwrap_or_else(|e| e.into_inner());
            if let Some(idx) = slices.get(&key) {
                if idx.epoch == current {
                    return Arc::clone(idx);
                }
            }
        }

        // Build outside the lock. The epoch is read BEFORE the snapshot:
        // any write completing after this read tags the result stale.
        let build_epoch = self.epoch.load(Ordering::SeqCst);
        let entries = snapshot();
        let index = Arc::new(SliceIndex {
            epoch: build_epoch,
            tree: MetricVpTree::build(entries, &EuclideanMetric),
        });

        let mut slices = self.slices.write().unwrap_or_else(|e| e.into_inner());
        // Drop every stale slice while we hold the write lock anyway.
        let now = self.epoch.load(Ordering::SeqCst);
        slices.retain(|_, idx| idx.epoch == now);
        // If a racing builder already cached a fresh tree for this slice,
        // keep it (identical content — the build is deterministic).
        let entry = slices.entry(key).or_insert_with(|| Arc::clone(&index));
        let result = Arc::clone(entry);

        // Deterministic eviction: drop the lowest key that isn't the one
        // just used (no wall-clock LRU — determinism over recency).
        while slices.len() > SEMANTIC_INDEX_MAX_SLICES {
            let evict = slices
                .keys()
                .find(|&&k| k != key)
                .copied()
                .expect("cache over capacity implies a second key exists");
            slices.remove(&evict);
        }

        result
    }

    /// Number of currently cached slices (test visibility).
    pub fn cached_slice_count(&self) -> usize {
        self.slices.read().unwrap_or_else(|e| e.into_inner()).len()
    }
}

impl Default for SemanticIndexCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(v: i32) -> FixedPoint {
        FixedPoint::from_int(v)
    }

    fn snapshot_of(n: usize) -> Vec<(String, Vec<FixedPoint>)> {
        (0..n).map(|i| (format!("n{}", i), vec![fp(i as i32)])).collect()
    }

    #[test]
    fn cache_hit_and_epoch_invalidation() {
        let cache = SemanticIndexCache::new();
        let range = 16..17;

        let a = cache.get_or_build(&range, || snapshot_of(3));
        let b = cache.get_or_build(&range, || panic!("must be served from cache"));
        assert!(Arc::ptr_eq(&a, &b));

        cache.bump();
        let c = cache.get_or_build(&range, || snapshot_of(4));
        assert!(!Arc::ptr_eq(&a, &c));
        assert_eq!(c.tree.len(), 4);
    }

    #[test]
    fn distinct_slices_get_distinct_trees() {
        let cache = SemanticIndexCache::new();
        let a = cache.get_or_build(&(16..18), || snapshot_of(2));
        let b = cache.get_or_build(&(16..20), || snapshot_of(5));
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(cache.cached_slice_count(), 2);
    }

    #[test]
    fn eviction_is_bounded_and_deterministic() {
        let cache = SemanticIndexCache::new();
        for i in 0..(SEMANTIC_INDEX_MAX_SLICES + 4) {
            cache.get_or_build(&(i..(i + 1)), || snapshot_of(1));
        }
        assert!(cache.cached_slice_count() <= SEMANTIC_INDEX_MAX_SLICES);
        // The most recently used slice must survive eviction.
        let last = SEMANTIC_INDEX_MAX_SLICES + 3;
        let again = cache.get_or_build(&(last..(last + 1)), || panic!("evicted the hot slice"));
        assert_eq!(again.tree.len(), 1);
    }
}
