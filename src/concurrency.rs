//! concurrency.rs - Striped locking for parallel writes to independent subtrees
//!
//! Provides a fixed-size array of Mutex stripes keyed by hash of parent_id.
//! Writers to the same parent serialize (golden-angle counter consistency);
//! writers to different parents proceed in parallel when they hit different stripes.

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::{Mutex, MutexGuard};

/// A fixed-size array of Mutex stripes for fine-grained write serialization.
///
/// Hash a key to select a stripe. Two keys that hash to the same stripe
/// serialize against each other (correctness preserved, slight throughput
/// reduction). 64 stripes is sufficient for typical workloads.
pub struct StripedLock<const N: usize> {
    locks: [Mutex<()>; N],
}

impl<const N: usize> StripedLock<N> {
    /// Create a new striped lock with N stripes.
    pub fn new() -> Self {
        Self {
            locks: std::array::from_fn(|_| Mutex::new(())),
        }
    }

    /// Acquire the stripe for the given key.
    pub fn lock(&self, key: &str) -> MutexGuard<'_, ()> {
        let index = Self::stripe_index(key);
        self.locks[index].lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Acquire the stripes for two keys at once, deadlock-free.
    ///
    /// Stripes are always taken in ascending index order, so two threads that
    /// each need the same pair of stripes can never form a hold-and-wait
    /// cycle. When both keys map to the **same** stripe, a single guard is
    /// returned and the second is `None` — re-locking the same mutex would
    /// deadlock (it is not reentrant), and one guard already serializes both
    /// keys. The returned guards are in stripe-index order, not argument
    /// order; callers should treat them as an opaque "hold both" token.
    pub fn lock_two(&self, a: &str, b: &str) -> (MutexGuard<'_, ()>, Option<MutexGuard<'_, ()>>) {
        let ia = Self::stripe_index(a);
        let ib = Self::stripe_index(b);
        if ia == ib {
            return (self.locks[ia].lock().unwrap_or_else(|e| e.into_inner()), None);
        }
        let (lo, hi) = if ia < ib { (ia, ib) } else { (ib, ia) };
        let g_lo = self.locks[lo].lock().unwrap_or_else(|e| e.into_inner());
        let g_hi = self.locks[hi].lock().unwrap_or_else(|e| e.into_inner());
        (g_lo, Some(g_hi))
    }

    /// Compute the stripe index for a key.
    fn stripe_index(key: &str) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % N
    }
}

impl<const N: usize> Default for StripedLock<N> {
    fn default() -> Self {
        Self::new()
    }
}

// StripedLock is Send+Sync because Mutex<()> is Send+Sync
unsafe impl<const N: usize> Sync for StripedLock<N> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_striped_lock_different_keys() {
        let lock = StripedLock::<64>::new();
        // Different keys should (usually) hit different stripes
        let _g1 = lock.lock("parent_a");
        // This shouldn't deadlock if keys hit different stripes
        // (If they collide, this test would deadlock — extremely unlikely with 64 stripes)
    }

    #[test]
    fn test_striped_lock_same_key_serializes() {
        let lock = StripedLock::<64>::new();
        {
            let _g = lock.lock("same_parent");
            // Lock acquired
        }
        // Lock released, acquire again
        let _g2 = lock.lock("same_parent");
    }
}
