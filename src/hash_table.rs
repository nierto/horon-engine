//! hash_table.rs — geometric node identity
//!
//! [`GeometricSignature`] is a node's identity: its depth in the tree plus its
//! position, quantised at 2^-20. `unique_id()` digests those two and nothing
//! else, so identity is a pure function of where a node sits — independent of
//! any index.
//!
//! That independence is the point. Until 0.6.0 this module held a table of
//! geometric buckets, each with its own VP-tree, and `unique_id` digested the
//! bucket hash as well — which made every node's name a function of which
//! bucket the index happened to choose, so the index could not be replaced
//! without renaming every node. The buckets are gone (see
//! `docs/ARCHITECTURE.md`); spatial queries are answered by [`crate::cell_index`].
//!
//! The module keeps its name for one release so the public path
//! `horon_engine::hash_table::GeometricSignature` does not move twice.

use std::fmt::{self, Debug, Formatter};
use super::hyperbolic_geometry::HyperbolicPoint;
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

    /// Get a unique node identifier, derived from level and position alone.
    ///
    /// **Deliberately independent of the spatial index.** This digest used to
    /// include `hash()` — the geometric bucket — which made every node's
    /// identity a function of which bucket the index happened to choose, so
    /// the index could not be changed without renaming every node. The bucket
    /// hash is itself a function of the same position (it is the signature of
    /// the containing bucket's centre), so it contributed no entropy: two
    /// nodes sharing a level and a `position_signature` already shared a
    /// bucket. Dropping it loses no discrimination and buys a swappable index.
    ///
    /// Uniqueness rests on `position_signature`, quantised at 2^-20 — the
    /// resolution the hardening audit established (`quantize_1000` collided
    /// for depth-2 cousins at a few hundred nodes; 2^-20 pushes the birthday
    /// bound past millions).
    ///
    /// For stub signatures (data-only nodes), returns the hash directly.
    pub fn unique_id(&self) -> String {
        if self.position_signature.is_empty() {
            // Stub signature — hash IS the unique_id
            return self.hash.clone();
        }
        use sha3::{Sha3_256, Digest as _};
        let mut hasher = Sha3_256::new();
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


impl GeometricSignature {
    /// Create the signature of an embedded node at `point`, `level` deep.
    ///
    /// The position signature is the point's first `dimension` coordinates
    /// quantised at 2^-20; `hash` is set to the resulting `unique_id`, so for
    /// an embedded signature the two agree. No index is consulted: this is a
    /// pure function of position and depth.
    pub fn embedded(point: &HyperbolicPoint, dimension: usize, level: u32) -> Self {
        let position_signature: Vec<i32> = (0..dimension)
            .map(|i| constants::quantize_position(point.coords()[i]))
            .collect();
        let mut signature = Self {
            hash: String::new(),
            level,
            position_signature,
        };
        signature.hash = signature.unique_id();
        signature
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperbolic_geometry::PoincareDisk;

    #[test]
    fn embedded_signature_carries_level_and_position() {
        let disk = PoincareDisk::new(2);
        let point = disk.point_from_f32_slice(&[0.5, 0.0]);

        let signature = GeometricSignature::embedded(&point, 2, 3);
        assert_eq!(signature.level(), 3);
        assert_eq!(signature.position_signature().len(), 2);
        assert!(!signature.is_stub());
        // For an embedded signature the hash is the id.
        assert_eq!(signature.hash(), signature.unique_id());
    }

    #[test]
    fn distinct_positions_get_distinct_ids() {
        let disk = PoincareDisk::new(2);
        let a = GeometricSignature::embedded(&disk.point_from_f32_slice(&[0.5, 0.0]), 2, 1);
        let b = GeometricSignature::embedded(&disk.point_from_f32_slice(&[0.0, 0.5]), 2, 1);
        assert_ne!(a.unique_id(), b.unique_id());
    }

    #[test]
    fn same_position_at_different_levels_gets_distinct_ids() {
        let disk = PoincareDisk::new(2);
        let point = disk.point_from_f32_slice(&[0.25, 0.25]);
        let a = GeometricSignature::embedded(&point, 2, 1);
        let b = GeometricSignature::embedded(&point, 2, 2);
        assert_ne!(a.unique_id(), b.unique_id());
    }

    #[test]
    fn stub_signature_is_its_own_id() {
        let stub = GeometricSignature::stub("data-only-node");
        assert!(stub.is_stub());
        assert_eq!(stub.unique_id(), "data-only-node");
    }
}
