//! storage.rs - Hierarchical Storage Implementation
//!
//! High-performance storage backend using Hyperbolic Tree Tensors (HTT).
//! Provides hierarchical data storage with efficient path-based access
//! patterns and O(1) lookup operations.
//!
//! ## Features
//!
//! - Path-based hierarchical storage structure
//! - Automatic directory creation
//! - Content-type support
//! - Metadata attachment
//! - Efficient subtree listing
//! - Spatial proximity queries via hyperbolic geometry

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use g_math::fixed_point::FixedPoint;
use log::trace;
use super::tree_tensor::{HyperbolicTreeTensor, HTTConfig, SharedHTT, IntegrationError, IntegrationResult};
use super::config::HTTStorageConfig;

/// HTT Storage Backend implementation.
///
/// Uses Hyperbolic Tree Tensors for efficient hierarchical data storage
/// with O(1) operations and spatial queries.
pub struct HTTStorage {
    /// The shared HTT instance
    htt: SharedHTT,
    /// Configuration
    config: HTTStorageConfig,
}

impl HTTStorage {
    /// Create a new HTT storage instance.
    pub fn new(config: HTTStorageConfig) -> Self {
        let mut htt_config = HTTConfig::new(
            config.dimension,
            config.max_memory_nodes,
            config.cache_size,
        );
        if config.grid_resolution > 0 {
            htt_config = htt_config.with_grid_resolution(config.grid_resolution);
        }
        if config.tau > FixedPoint::from_int(0) {
            htt_config = htt_config.with_tau(config.tau);
        }

        let htt = Arc::new(HyperbolicTreeTensor::new(htt_config));

        // Initialize with a root node. Inserting "/" into a fresh tree is
        // infallible under a correct build; a failure here means a broken
        // invariant (e.g. wrong fixed-point profile), not a recoverable
        // runtime condition. Fail loudly rather than hand back a store with
        // no root — every path operation assumes the root exists.
        htt.insert("/", vec![], Some("application/x-directory".to_string()))
            .expect("failed to initialize HTT root node ('/')");

        Self { htt, config }
    }

    /// Get the shared HTT instance.
    pub fn shared_htt(&self) -> &SharedHTT {
        &self.htt
    }

    /// Store data without geometric embedding (data + semantic only).
    ///
    /// Much faster than `store()` — skips Sarkar embedding. Use for bulk
    /// loading when spatial queries are not needed (semantic queries still work).
    pub fn store_data_only(&self, key: &str, value: &[u8], content_type: Option<String>) -> IntegrationResult<()> {
        let normalized_key = Self::normalize_key(key);

        let missing_ancestors = Self::find_missing_ancestors(&self.htt, &normalized_key);
        for ancestor in &missing_ancestors {
            if !self.htt.exists(ancestor) {
                match self.htt.insert_data_only(ancestor, vec![], Some("application/x-directory".to_string())) {
                    Ok(()) => {},
                    Err(IntegrationError::AlreadyExists(_)) => {},
                    Err(e) => return Err(e),
                }
            }
        }

        if self.htt.exists(&normalized_key) {
            self.htt.update_value(&normalized_key, value.to_vec())?;
        } else {
            match self.htt.insert_data_only(&normalized_key, value.to_vec(), content_type) {
                Ok(()) => {},
                Err(IntegrationError::AlreadyExists(_)) => {
                    self.htt.update_value(&normalized_key, value.to_vec())?;
                },
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Store data by key.
    ///
    /// Batch-creates missing ancestor directories under a single write lock
    /// before inserting the target node. This avoids the previous recursive
    /// approach which acquired/released the lock once per ancestor.
    pub fn store(&self, key: &str, value: &[u8], content_type: Option<String>) -> IntegrationResult<()> {
        trace!("HTTStorage::store - key: {}, size: {} bytes", key, value.len());

        let normalized_key = Self::normalize_key(key);

        // Collect missing ancestors (cheap DashMap existence checks)
        let missing_ancestors = Self::find_missing_ancestors(&self.htt, &normalized_key);

        // Create missing ancestors root-to-leaf (each parent exists before its child).
        // Ignore AlreadyExists — another thread may have created the same ancestor
        // concurrently, which is correct behavior (TOCTOU between exists() and insert()).
        for ancestor in &missing_ancestors {
            if !self.htt.exists(ancestor) {
                match self.htt.insert(
                    ancestor,
                    vec![],
                    Some("application/x-directory".to_string()),
                ) {
                    Ok(()) => {},
                    Err(IntegrationError::AlreadyExists(_)) => {},
                    Err(e) => return Err(e),
                }
            }
        }

        // Insert or update the target node.
        // Same TOCTOU guard: if another thread inserted between our exists() check
        // and our insert(), fall through to update.
        if self.htt.exists(&normalized_key) {
            self.htt.update_value(&normalized_key, value.to_vec())?;
        } else {
            match self.htt.insert(&normalized_key, value.to_vec(), content_type) {
                Ok(()) => {},
                Err(IntegrationError::AlreadyExists(_)) => {
                    self.htt.update_value(&normalized_key, value.to_vec())?;
                },
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Store data with an explicit child_index for deterministic Sarkar reconstruction.
    ///
    /// Same as `store()` but passes child_index through so the node gets the same
    /// geometric position regardless of insertion order. Used during snapshot replay.
    pub fn store_positioned(&self, key: &str, value: &[u8], content_type: Option<String>, child_index: u32) -> IntegrationResult<()> {
        let normalized_key = Self::normalize_key(key);

        let missing_ancestors = Self::find_missing_ancestors(&self.htt, &normalized_key);
        for ancestor in &missing_ancestors {
            if !self.htt.exists(ancestor) {
                match self.htt.insert(
                    ancestor,
                    vec![],
                    Some("application/x-directory".to_string()),
                ) {
                    Ok(()) => {},
                    Err(IntegrationError::AlreadyExists(_)) => {},
                    Err(e) => return Err(e),
                }
            }
        }

        if self.htt.exists(&normalized_key) {
            self.htt.update_value(&normalized_key, value.to_vec())?;
        } else {
            match self.htt.insert_positioned(&normalized_key, value.to_vec(), content_type, child_index) {
                Ok(()) => {},
                Err(IntegrationError::AlreadyExists(_)) => {
                    self.htt.update_value(&normalized_key, value.to_vec())?;
                },
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Retrieve data by key.
    pub fn retrieve(&self, key: &str) -> IntegrationResult<Vec<u8>> {
        trace!("HTTStorage::retrieve - key: {}", key);

        let normalized_key = Self::normalize_key(key);
        let node = self.htt.get(&normalized_key)?;
        Ok(node.value().to_vec())
    }

    /// Delete data by key.
    pub fn delete(&self, key: &str) -> IntegrationResult<()> {
        trace!("HTTStorage::delete - key: {}", key);

        let normalized_key = Self::normalize_key(key);

        if normalized_key == "/" {
            return Err(IntegrationError::ValidationFailed(
                "Cannot delete root node".to_string(),
            ));
        }

        self.htt.delete(&normalized_key)
    }

    /// List keys with a prefix.
    pub fn list(&self, prefix: &str) -> IntegrationResult<Vec<String>> {
        trace!("HTTStorage::list - prefix: {}", prefix);

        let normalized_prefix = Self::normalize_key(prefix);
        self.htt.list_subtree(&normalized_prefix)
    }

    /// Check if a key exists.
    pub fn exists(&self, key: &str) -> bool {
        let normalized_key = Self::normalize_key(key);
        self.htt.exists(&normalized_key)
    }

    /// Get metadata for a key.
    pub fn get_metadata(&self, key: &str) -> IntegrationResult<HashMap<String, String>> {
        trace!("HTTStorage::get_metadata - key: {}", key);

        let normalized_key = Self::normalize_key(key);
        let node = self.htt.get(&normalized_key)?;
        let meta = node.metadata();

        let mut result = meta.metadata.clone();
        result.insert("key".to_string(), meta.key.clone());
        result.insert("size".to_string(), node.value().len().to_string());
        result.insert("created_at".to_string(), meta.created_at.to_string());
        result.insert("updated_at".to_string(), meta.updated_at.to_string());

        if let Some(ref ct) = meta.content_type {
            result.insert("content_type".to_string(), ct.clone());
        }

        Ok(result)
    }

    /// Set metadata for a key.
    pub fn set_metadata(&self, key: &str, meta_key: &str, meta_value: &str) -> IntegrationResult<()> {
        trace!("HTTStorage::set_metadata - key: {}, meta_key: {}", key, meta_key);

        let normalized_key = Self::normalize_key(key);
        self.htt.set_node_metadata(&normalized_key, meta_key, meta_value)
    }

    /// Set semantic coordinates for a key (raw Q64.64 bytes, 16 bytes per dimension).
    pub fn set_semantic(&self, key: &str, coords: Vec<u8>) -> IntegrationResult<()> {
        trace!("HTTStorage::set_semantic - key: {}, bytes: {}", key, coords.len());

        let normalized_key = Self::normalize_key(key);
        self.htt.set_semantic(&normalized_key, coords)
    }

    /// Get semantic coordinates for a key (raw Q64.64 bytes).
    pub fn get_semantic(&self, key: &str) -> IntegrationResult<Vec<u8>> {
        trace!("HTTStorage::get_semantic - key: {}", key);

        let normalized_key = Self::normalize_key(key);
        self.htt.get_semantic(&normalized_key)
    }

    /// The hyperbolic (Poincaré) position of a stored key.
    pub fn position(&self, key: &str) -> IntegrationResult<crate::hyperbolic_geometry::HyperbolicPoint> {
        let normalized_key = Self::normalize_key(key);
        self.htt.position(&normalized_key)
    }

    /// Upgrade a data-only key to a full geometric embedding (embed-on-demand — see
    /// [`crate::tree_tensor::HyperbolicTreeTensor::embed_existing`]).
    /// Returns whether this call performed the upgrade.
    pub fn embed_existing(&self, key: &str) -> IntegrationResult<bool> {
        let normalized_key = Self::normalize_key(key);
        self.htt.embed_existing(&normalized_key)
    }

    /// Monotone counter of semantic-relevant mutations (see
    /// [`crate::tensor_network::HyperbolicTensorNetwork::semantic_epoch`]).
    pub fn semantic_epoch(&self) -> u64 {
        self.htt.tensor_network().semantic_epoch()
    }

    /// Find the k nearest stored keys to the given key's position in hyperbolic space.
    /// Returns paths sorted by ascending hyperbolic distance.
    pub fn find_nearest(&self, path: &str, k: usize) -> IntegrationResult<Vec<String>> {
        let normalized = Self::normalize_key(path);
        let results = self.htt.find_nearest(&normalized, k)?;
        Ok(results.into_iter().map(|(p, _dist)| p).collect())
    }

    /// Find all keys within hyperbolic radius of the given key.
    pub fn find_in_radius(&self, path: &str, radius: FixedPoint) -> IntegrationResult<Vec<String>> {
        let normalized = Self::normalize_key(path);
        let results = self.htt.find_in_radius(&normalized, radius)?;
        Ok(results.into_iter().map(|(p, _dist)| p).collect())
    }

    // -----------------------------------------------------------------------
    // Semantic dimensional distance queries
    // -----------------------------------------------------------------------

    /// Find the k nearest nodes by Euclidean distance across a dimensional slice.
    ///
    /// `query_coords`: raw Q64.64 bytes for the query point.
    /// `k`: number of results.
    /// `dim_range`: which dimensions to compare.
    ///
    /// Returns paths sorted by distance ascending.
    pub fn nearest_semantic(
        &self,
        query_coords: &[u8],
        k: usize,
        dim_range: &Range<usize>,
    ) -> IntegrationResult<Vec<(String, FixedPoint)>> {
        self.htt.nearest_semantic(query_coords, k, dim_range)
    }

    /// Find the k nearest nodes to an existing node by semantic dimensional distance.
    /// The queried node is excluded from results.
    pub fn neighbors_semantic(
        &self,
        path: &str,
        k: usize,
        dim_range: &Range<usize>,
    ) -> IntegrationResult<Vec<(String, FixedPoint)>> {
        let normalized = Self::normalize_key(path);
        self.htt.neighbors_semantic(&normalized, k, dim_range)
    }

    /// Find the nearest stored node to an arbitrary point in the Poincaré disk.
    ///
    /// Uses the Nielsen power diagram for O(1) point location.
    /// Coordinates are in f32 (user-facing boundary); converted internally to FixedPoint.
    /// Returns (path, hyperbolic_distance_as_FixedPoint).
    pub fn nearest_neighbor_point(&self, coords: &[FixedPoint]) -> IntegrationResult<(String, FixedPoint)> {
        self.validate_query_coords(coords)?;
        let query = super::hyperbolic_geometry::HyperbolicPoint::from_slice(coords);
        self.htt.nearest_neighbor_point(&query)
    }

    /// Find the k nearest stored nodes to an arbitrary point in the Poincaré disk.
    ///
    /// Returns `(path, hyperbolic_distance)` sorted by ascending distance.
    pub fn nearest_neighbor_point_k(&self, coords: &[FixedPoint], k: usize) -> IntegrationResult<Vec<(String, FixedPoint)>> {
        self.validate_query_coords(coords)?;
        let query = super::hyperbolic_geometry::HyperbolicPoint::from_slice(coords);
        self.htt.nearest_neighbor_point_k(&query, k)
    }

    /// Reject query coordinates whose length doesn't match the configured
    /// embedding dimension — the geometry kernels assert on mismatched
    /// dimensions, and a panic must not be reachable from user input.
    fn validate_query_coords(&self, coords: &[FixedPoint]) -> IntegrationResult<()> {
        if coords.len() != self.config.dimension {
            return Err(IntegrationError::ValidationFailed(format!(
                "query has {} coordinates but the store dimension is {}",
                coords.len(),
                self.config.dimension
            )));
        }
        Ok(())
    }

    /// Collect all missing ancestor paths between `path` and the nearest existing
    /// ancestor, returned in root-to-leaf order for sequential creation.
    fn find_missing_ancestors(htt: &HyperbolicTreeTensor, path: &str) -> Vec<String> {
        let mut missing = Vec::new();
        let mut current = path.to_string();

        loop {
            let parent = match current.rfind('/') {
                Some(index) if index > 0 => current[0..index].to_string(),
                Some(0) if current != "/" => "/".to_string(),
                _ => break,
            };

            if parent == current {
                break;
            }

            if htt.exists(&parent) {
                break;
            }

            missing.push(parent.clone());
            current = parent;
        }

        missing.reverse(); // root-to-leaf order
        missing
    }

    /// Total number of nodes in the tree, including the root node.
    pub fn node_count(&self) -> usize {
        self.htt.node_count()
    }

    /// Get storage statistics.
    pub fn stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();

        for (key, value) in self.htt.stats() {
            stats.insert(format!("htt.{}", key), value);
        }
        stats.insert("node_count".to_string(), self.htt.node_count().to_string());

        stats.insert("dimension".to_string(), self.config.dimension.to_string());
        stats.insert(
            "max_memory_nodes".to_string(),
            self.config.max_memory_nodes.to_string(),
        );
        stats.insert("cache_size".to_string(), self.config.cache_size.to_string());

        stats
    }

    /// Normalize a key to have a leading '/'.
    fn normalize_key(key: &str) -> String {
        if !key.starts_with('/') {
            format!("/{}", key)
        } else {
            key.to_string()
        }
    }
}

#[cfg(test)]
mod tests {

/// Exact fixed-point coordinates from decimal literals.
fn fp(vals: &[f64]) -> Vec<g_math::fixed_point::FixedPoint> {
    vals.iter().map(|&v| g_math::fixed_point::FixedPoint::from_f64(v)).collect()
}

    use super::*;

    #[test]
    fn test_storage_creation() {
        let config = HTTStorageConfig::default();
        let storage = HTTStorage::new(config);
        assert!(storage.exists("/"));
    }

    #[test]
    fn test_storage_operations() {
        let config = HTTStorageConfig::default();
        let storage = HTTStorage::new(config);

        // Store data
        storage.store("/test", b"test data", None).unwrap();
        assert!(storage.exists("/test"));

        // Retrieve data
        let data = storage.retrieve("/test").unwrap();
        assert_eq!(data, b"test data");

        // Update data
        storage.store("/test", b"updated data", None).unwrap();
        let updated = storage.retrieve("/test").unwrap();
        assert_eq!(updated, b"updated data");

        // Create nested path
        storage.store("/parent/child", b"child data", None).unwrap();
        assert!(storage.exists("/parent"));

        // List with prefix
        let keys = storage.list("/").unwrap();
        assert!(keys.contains(&"/test".to_string()));
        assert!(keys.contains(&"/parent".to_string()));
        assert!(keys.contains(&"/parent/child".to_string()));

        // Delete
        storage.delete("/test").unwrap();
        assert!(!storage.exists("/test"));

        // Metadata
        storage
            .set_metadata("/parent", "description", "A parent directory")
            .unwrap();
        let metadata = storage.get_metadata("/parent").unwrap();
        assert_eq!(
            metadata.get("description"),
            Some(&"A parent directory".to_string())
        );
    }

    #[test]
    fn test_find_nearest_api() {
        let config = HTTStorageConfig::default();
        let storage = HTTStorage::new(config);

        storage.store("/a", b"a", None).unwrap();
        storage.store("/b", b"b", None).unwrap();
        storage.store("/c", b"c", None).unwrap();

        let nearest = storage.find_nearest("/a", 2).unwrap();
        assert!(!nearest.is_empty());
        assert!(nearest.len() <= 2);
        // Should not include /a itself
        assert!(!nearest.contains(&"/a".to_string()));
    }

    #[test]
    fn test_find_in_radius_api() {
        use g_math::fixed_point::FixedPoint;

        let config = HTTStorageConfig::default();
        let storage = HTTStorage::new(config);

        storage.store("/x", b"x", None).unwrap();
        storage.store("/y", b"y", None).unwrap();

        let large_radius = FixedPoint::from_int(10);
        let results = storage.find_in_radius("/x", large_radius).unwrap();
        // Should find at least /y and / (root)
        assert!(!results.is_empty());
    }

    #[test]
    fn test_storage_stats() {
        let config = HTTStorageConfig::default();
        let storage = HTTStorage::new(config);

        storage.store("/test1", b"data1", None).unwrap();
        storage.store("/test2", b"data2", None).unwrap();

        let stats = storage.stats();
        assert!(stats.contains_key("node_count"));
        assert!(stats.contains_key("dimension"));
    }

    #[test]
    fn test_nearest_neighbor_point_api() {
        let config = HTTStorageConfig::default();
        let storage = HTTStorage::new(config);

        storage.store("/a", b"a", None).unwrap();
        storage.store("/b", b"b", None).unwrap();
        storage.store("/c", b"c", None).unwrap();

        // Query at origin should return the root (which is at the origin)
        let (path, dist) = storage.nearest_neighbor_point(&fp(&[0.0, 0.0, 0.0, 0.0])).unwrap();
        assert_eq!(path, "/", "Query at origin should find root");
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(10);
        assert!(dist < tolerance, "Distance to root at origin should be small");
    }
}
