//! tree_tensor.rs - Hyperbolic Tree Tensor Core Implementation
//!
//! Efficient hierarchical data representation: path lookups are hash-map
//! access, spatial queries go through the cell index (see per-method docs on
//! `Store` for costs).

use std::collections::HashMap;
use std::fmt::{self, Debug, Formatter};
use std::ops::Range;
use std::sync::Arc;
use dashmap::DashMap;
use super::concurrency::StripedLock;
use super::hash_table::GeometricSignature;
use super::tensor_network::{HyperbolicTensorNetwork, CompressedNode, NodeMetadata};

/// Error type for tree tensor integration operations.
#[derive(Debug)]
pub enum IntegrationError {
    /// A node already exists at the target path.
    AlreadyExists(String),
    /// No node exists at the target path.
    NotFound(String),
    /// The operation could not be completed.
    OperationFailed(String),
    /// Input failed validation (dimensions, structure, or constraints).
    ValidationFailed(String),
    /// Stored bytes could not be deserialized.
    DeserializationError(String),
    /// A lock could not be acquired.
    LockError(String),
    /// The supplied configuration is invalid.
    ConfigurationError(String),
}

impl IntegrationError {
    /// Create a configuration error.
    pub fn configuration_error(msg: String) -> Self {
        Self::ConfigurationError(msg)
    }
}

impl std::fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyExists(s) => write!(f, "already exists: {s}"),
            Self::NotFound(s) => write!(f, "not found: {s}"),
            Self::OperationFailed(s) => write!(f, "operation failed: {s}"),
            Self::ValidationFailed(s) => write!(f, "validation failed: {s}"),
            Self::DeserializationError(s) => write!(f, "deserialization error: {s}"),
            Self::LockError(s) => write!(f, "lock error: {s}"),
            Self::ConfigurationError(s) => write!(f, "configuration error: {s}"),
        }
    }
}

impl std::error::Error for IntegrationError {}

/// Result type alias for tree-tensor integration operations.
pub type IntegrationResult<T> = Result<T, IntegrationError>;

/// Path operations for hierarchical data.
pub trait PathOperations {
    /// Split a path into components.
    fn split_path(&self, path: &str) -> Vec<String>;

    /// Join path components into a full path.
    fn join_path(&self, components: &[String]) -> String;

    /// Get parent path from a path.
    fn parent_path(&self, path: &str) -> Option<String>;

    /// Get the last component of a path.
    fn last_component(&self, path: &str) -> Option<String>;
}

/// Default path operations implementation using '/' separator.
pub struct DefaultPathOps;

impl PathOperations for DefaultPathOps {
    fn split_path(&self, path: &str) -> Vec<String> {
        path.split('/')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }

    fn join_path(&self, components: &[String]) -> String {
        let mut path = String::new();
        for component in components {
            path.push('/');
            path.push_str(component);
        }
        if path.is_empty() {
            path.push('/');
        }
        path
    }

    fn parent_path(&self, path: &str) -> Option<String> {
        if path == "/" {
            return None;
        }

        let components = self.split_path(path);
        if components.is_empty() {
            return Some("/".to_string());
        }

        let parent_components = &components[0..components.len() - 1];
        Some(self.join_path(parent_components))
    }

    fn last_component(&self, path: &str) -> Option<String> {
        let components = self.split_path(path);
        components.last().cloned()
    }
}

/// Configuration for the Hyperbolic Tree Tensor.
#[derive(Clone, Debug)]
pub struct HTTConfig {
    /// Dimension of the hyperbolic space
    dimension: usize,
    /// Maximum in-memory nodes before flushing to storage
    max_memory_nodes: usize,
    /// Cache size for frequently accessed nodes
    cache_size: usize,
    /// Sarkar embedding scale factor τ: parent-child hyperbolic distance
    tau: g_math::fixed_point::FixedPoint,
}

impl HTTConfig {
    /// Create a new HTT configuration (uses default τ = 1.0).
    pub fn new(dimension: usize, max_memory_nodes: usize, cache_size: usize) -> Self {
        Self {
            dimension,
            max_memory_nodes,
            cache_size,
            tau: crate::constants::default_tau(),
        }
    }

    /// Create a default configuration.
    pub fn default_config() -> Self {
        Self {
            dimension: 4,
            max_memory_nodes: 1000,
            cache_size: 100,
            tau: crate::constants::default_tau(),
        }
    }

    /// Set the Sarkar embedding scale factor τ.
    pub fn with_tau(mut self, tau: g_math::fixed_point::FixedPoint) -> Self {
        self.tau = tau;
        self
    }

    /// Get the dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the maximum memory nodes.
    pub fn max_memory_nodes(&self) -> usize {
        self.max_memory_nodes
    }

    /// Get the cache size.
    pub fn cache_size(&self) -> usize {
        self.cache_size
    }

    /// Get the Sarkar embedding scale factor τ.
    pub fn tau(&self) -> g_math::fixed_point::FixedPoint {
        self.tau
    }
}

impl Default for HTTConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Hyperbolic Tree Tensor - core data structure.
///
/// Path access is hash-map cost; spatial queries go through the cell index.
/// Nodes are embedded in the Poincare disk, and their positions are derived
/// from tree shape rather than stored.
///
/// Path maps use DashMap for lock-free concurrent reads.
pub struct HyperbolicTreeTensor {
    /// Tensor network for spatial embedding
    tensor_network: HyperbolicTensorNetwork,
    /// Path operations
    path_ops: Box<dyn PathOperations + Send + Sync>,
    /// Signature map for O(1) path lookups: path -> signature
    path_map: DashMap<String, GeometricSignature>,
    /// Reverse map: unique_id -> path (for spatial query results)
    id_to_path: DashMap<String, String>,
    /// Striped parent locks: serialize writes to the same parent,
    /// allow parallel writes to different parents (64 stripes)
    parent_locks: StripedLock<64>,
    /// Configuration
    config: HTTConfig,
}

impl HyperbolicTreeTensor {
    /// Create a new Hyperbolic Tree Tensor.
    pub fn new(config: HTTConfig) -> Self {
        let tensor_network = HyperbolicTensorNetwork::new(config.dimension(), config.tau());

        Self {
            tensor_network,
            path_ops: Box::new(DefaultPathOps),
            path_map: DashMap::new(),
            id_to_path: DashMap::new(),
            parent_locks: StripedLock::new(),
            config,
        }
    }

    /// Resolve a node's parent signature, enforcing the tree invariant.
    ///
    /// Returns `Ok(None)` for a root insert (parent path is `None`, i.e. the
    /// path is `/`), `Ok(Some(sig))` when the parent exists, and
    /// `Err(NotFound)` when a non-root node's parent is absent. The last case
    /// is the important one: `add_node` places a `None`-parent node at the
    /// disk origin as a root, so silently passing `None` for a missing parent
    /// would stack unrelated subtrees on top of each other at the centre.
    fn resolve_parent(
        &self,
        path: &str,
        parent_path: &Option<String>,
    ) -> IntegrationResult<Option<GeometricSignature>> {
        match parent_path {
            Some(p) if p.as_str() != path => match self.path_map.get(p.as_str()) {
                Some(r) => Ok(Some(r.value().clone())),
                None => Err(IntegrationError::NotFound(format!(
                    "cannot insert '{}': parent '{}' does not exist",
                    path, p
                ))),
            },
            _ => Ok(None), // root, or a degenerate self-parent
        }
    }

    /// Insert a node without geometric embedding (data + semantic only).
    ///
    /// Much faster than `insert()` — skips Sarkar embedding, VP-tree, power
    /// diagram, and point location grid. Use for bulk loading when spatial
    /// queries are not needed (semantic queries still work).
    pub fn insert_data_only(&self, path: &str, value: Vec<u8>, content_type: Option<String>) -> IntegrationResult<()> {
        // Fast path: lock-free duplicate check.
        if self.path_map.contains_key(path) {
            return Err(IntegrationError::AlreadyExists(
                format!("Node at path {} already exists", path),
            ));
        }

        // Enforce the tree invariant: a non-root node's parent must exist.
        // (HTTStorage creates ancestors first; a direct caller inserting an
        // orphan would otherwise leave a dangling path.)
        let parent_path = self.path_ops.parent_path(path);
        if let Some(p) = &parent_path {
            if p.as_str() != path && !self.path_map.contains_key(p.as_str()) {
                return Err(IntegrationError::NotFound(format!(
                    "cannot insert '{}': parent '{}' does not exist",
                    path, p
                )));
            }
        }

        // Serialize sibling creation under the same parent (data-only nodes
        // don't touch the geometric child list, but the stripe lock closes
        // the TOCTOU between the duplicate check and the insert). Root inserts
        // ("/") have no parent → no lock needed.
        let _stripe_guard = parent_path.as_deref().map(|p| self.parent_locks.lock(p));

        // Re-check under the lock.
        if self.path_map.contains_key(path) {
            return Err(IntegrationError::AlreadyExists(
                format!("Node at path {} already exists", path),
            ));
        }

        let metadata = NodeMetadata::new(path.to_string(), content_type);

        let unique_id = self.tensor_network.add_node_data_only(metadata, value, 0);

        // Create a stub signature for path_map (no geometric meaning)
        let stub_sig = GeometricSignature::stub(&unique_id);
        self.id_to_path.insert(unique_id, path.to_string());
        self.path_map.insert(path.to_string(), stub_sig);

        Ok(())
    }

    /// Insert a node at the specified path.
    ///
    /// Acquires a striped parent lock to serialize writes to the same parent
    /// while allowing parallel writes to different parents.
    pub fn insert(&self, path: &str, value: Vec<u8>, content_type: Option<String>) -> IntegrationResult<()> {
        // Fast path: lock-free DashMap read
        if self.path_map.contains_key(path) {
            return Err(IntegrationError::AlreadyExists(
                format!("Node at path {} already exists", path),
            ));
        }

        let parent_path = self.path_ops.parent_path(path);
        let level = self.path_depth(path);
        let metadata = NodeMetadata::new(path.to_string(), content_type);

        // Look up the parent signature (lock-free DashMap read). A non-root
        // node whose parent is absent is rejected — silently treating it as a
        // second root would place it at the origin, on top of the real root.
        let parent_signature = self.resolve_parent(path, &parent_path)?;

        // Acquire stripe lock on parent to serialize sibling creation.
        // This ensures child_counts consistency and prevents duplicate path creation
        // (TOCTOU between contains_key above and path_map.insert below).
        // Different parents hit different stripes → parallel writes to independent subtrees.
        let parent_uid = parent_signature.as_ref().map(|s| s.unique_id());
        let _stripe_guard = parent_uid.as_deref().map(|uid| self.parent_locks.lock(uid));

        // Re-check under the stripe lock (double-checked locking)
        if self.path_map.contains_key(path) {
            return Err(IntegrationError::AlreadyExists(
                format!("Node at path {} already exists", path),
            ));
        }

        let signature = self
            .tensor_network
            .add_node(metadata, value, parent_signature.as_ref(), level)
            .ok_or_else(|| {
                IntegrationError::OperationFailed(
                    "Failed to add node to tensor network".to_string(),
                )
            })?;

        let unique_id = signature.unique_id();
        self.id_to_path.insert(unique_id, path.to_string());
        self.path_map.insert(path.to_string(), signature);

        // _stripe_guard dropped here, releasing the parent stripe lock
        Ok(())
    }

    /// Insert a node with an explicit child_index for deterministic Sarkar reconstruction.
    ///
    /// Same as `insert()` but passes the child_index through to the tensor network
    /// so the node gets the same geometric position regardless of insertion order.
    pub fn insert_positioned(&self, path: &str, value: Vec<u8>, content_type: Option<String>, child_index: u32) -> IntegrationResult<()> {
        if self.path_map.contains_key(path) {
            return Err(IntegrationError::AlreadyExists(
                format!("Node at path {} already exists", path),
            ));
        }

        let parent_path = self.path_ops.parent_path(path);
        let level = self.path_depth(path);
        let metadata = NodeMetadata::new(path.to_string(), content_type);

        let parent_signature = self.resolve_parent(path, &parent_path)?;

        let parent_uid = parent_signature.as_ref().map(|s| s.unique_id());
        let _stripe_guard = parent_uid.as_deref().map(|uid| self.parent_locks.lock(uid));

        if self.path_map.contains_key(path) {
            return Err(IntegrationError::AlreadyExists(
                format!("Node at path {} already exists", path),
            ));
        }

        let signature = self
            .tensor_network
            .add_node_positioned(metadata, value, parent_signature.as_ref(), level, child_index)
            .ok_or_else(|| {
                IntegrationError::OperationFailed(
                    "Failed to add node to tensor network".to_string(),
                )
            })?;

        let unique_id = signature.unique_id();
        self.id_to_path.insert(unique_id, path.to_string());
        self.path_map.insert(path.to_string(), signature);

        Ok(())
    }

    /// Upgrade a data-only node to a full geometric embedding, in place.
    ///
    /// Data-only nodes (`insert_data_only`) live under a stub signature with
    /// no Poincaré position — semantic queries see them, spatial queries do
    /// not. This computes their Sarkar placement on demand: missing ancestors
    /// are embedded first (placement needs an embedded parent; recursion is
    /// bounded by the ~44/τ depth budget), then the node is re-registered
    /// under its position-derived signature with its identity preserved —
    /// key, value, user metadata, timestamps, and semantic coordinates all
    /// carry over.
    ///
    /// Returns `Ok(true)` when this call performed the upgrade, `Ok(false)`
    /// when the node was already embedded (idempotent), `NotFound` for
    /// missing paths.
    ///
    /// Concurrency: takes the same parent stripe lock as `insert`, so embeds
    /// serialize with sibling inserts and deletes; racing embeds of the same
    /// node resolve to one upgrade (double-checked under the lock). Readers
    /// never observe the node missing: the embedded replacement (with
    /// coordinates already copied) goes live in `path_map` before the
    /// data-only entry is retired — a concurrent semantic query may
    /// transiently see the key twice (same key, same coordinates) inside
    /// that window.
    ///
    /// Geometric positions are derived state and are NOT persisted: the
    /// position depends on the sibling order at embed time, so it is
    /// deterministic for a fixed operation sequence but not stable across
    /// sessions. Callers re-embed after reopening a lazily-loaded store.
    pub fn embed_existing(&self, path: &str) -> IntegrationResult<bool> {
        // Lock-free fast path.
        let sig = match self.path_map.get(path) {
            Some(r) => r.value().clone(),
            None => {
                return Err(IntegrationError::NotFound(format!(
                    "Node at path {} not found",
                    path
                )))
            }
        };
        if !sig.is_stub() {
            return Ok(false);
        }

        // Ancestors first. The recursive call acquires and releases its own
        // parent stripe before this frame takes any lock — no nested guards,
        // no ordering hazard.
        let parent_path = self.path_ops.parent_path(path);
        if let Some(p) = &parent_path {
            if p.as_str() != path {
                self.embed_existing(p)?;
            }
        }

        let parent_signature = self.resolve_parent(path, &parent_path)?;
        if let Some(ps) = &parent_signature {
            if ps.is_stub() {
                // A concurrent delete+reinsert put a fresh data-only parent
                // back between our recursion and this lookup. Surface it
                // rather than embedding against a positionless parent.
                return Err(IntegrationError::OperationFailed(format!(
                    "cannot embed '{}': parent lost its embedding concurrently",
                    path
                )));
            }
        }
        let parent_uid = parent_signature.as_ref().map(|s| s.unique_id());
        let _stripe_guard = parent_uid.as_deref().map(|uid| self.parent_locks.lock(uid));

        // Re-check under the lock: a racing embed may have won, or a racing
        // delete may have removed the node.
        let old_sig = match self.path_map.get(path) {
            Some(r) => r.value().clone(),
            None => {
                return Err(IntegrationError::NotFound(format!(
                    "Node at path {} not found",
                    path
                )))
            }
        };
        if !old_sig.is_stub() {
            return Ok(false);
        }
        let old_uid = old_sig.unique_id();

        // Preserve the node's identity.
        let node = self
            .tensor_network
            .get_node_by_signature(&old_sig)
            .ok_or_else(|| {
                IntegrationError::OperationFailed(format!(
                    "data-only node '{}' missing from the node map",
                    path
                ))
            })?;
        let metadata = node.metadata().clone();
        let value = node.value().to_vec();
        let coords = node.semantic_coords().to_vec();

        let level = self.path_depth(path);
        let signature = self
            .tensor_network
            .add_node(metadata, value, parent_signature.as_ref(), level)
            .ok_or_else(|| {
                IntegrationError::OperationFailed(format!(
                    "failed to embed node '{}'",
                    path
                ))
            })?;
        let new_uid = signature.unique_id();

        // Coordinates before the visibility swap: a concurrent semantic
        // query never misses the node.
        if !coords.is_empty() {
            self.tensor_network.set_node_semantic(&new_uid, coords);
        }

        // Swap visibility to the embedded node, then retire the stub entry.
        self.id_to_path.insert(new_uid, path.to_string());
        self.path_map.insert(path.to_string(), signature);
        self.id_to_path.remove(&old_uid);
        self.tensor_network.remove_detached_node(&old_uid);

        Ok(true)
    }

    /// Get a node by path (returns cloned value).
    pub fn get(&self, path: &str) -> IntegrationResult<CompressedNode> {
        let signature = self
            .path_map
            .get(path)
            .ok_or_else(|| IntegrationError::NotFound(format!("Node at path {} not found", path)))?;

        self.tensor_network
            .get_node_by_signature(signature.value())
            .ok_or_else(|| {
                IntegrationError::NotFound(format!(
                    "Node with signature {} not found",
                    signature.value().hash()
                ))
            })
    }

    /// Update the value of a node at the specified path.
    pub fn update_value(&self, path: &str, value: Vec<u8>) -> IntegrationResult<()> {
        // The path_map guard is held across the network read on purpose.
        // `embed_existing` swaps a stub for an embedded node by inserting the
        // new signature into `path_map` and THEN retiring the old node. A
        // reader that resolves the uid and drops the guard first can be
        // descheduled in that window and then look up a uid the network has
        // already released — a spurious NotFound for a key that never went
        // away. Holding the shard guard makes the swap's `path_map.insert`
        // wait. This is what `get()` has always done; these four did not.
        // No inversion: the writer releases every node lock inside `add_node`
        // before it touches `path_map`, so the orders never interleave.
        let guard = self
            .path_map
            .get(path)
            .ok_or_else(|| IntegrationError::NotFound(format!("Node at path {} not found", path)))?;
        let uid = guard.value().unique_id();
        let ok = self.tensor_network.update_node_value(&uid, value);
        drop(guard);

        if ok {
            Ok(())
        } else {
            Err(IntegrationError::NotFound(format!("Node with uid {} not found in network", uid)))
        }
    }

    /// Set a metadata key-value pair on a node.
    pub fn set_node_metadata(&self, path: &str, key: &str, value: &str) -> IntegrationResult<()> {
        // The path_map guard is held across the network read on purpose.
        // `embed_existing` swaps a stub for an embedded node by inserting the
        // new signature into `path_map` and THEN retiring the old node. A
        // reader that resolves the uid and drops the guard first can be
        // descheduled in that window and then look up a uid the network has
        // already released — a spurious NotFound for a key that never went
        // away. Holding the shard guard makes the swap's `path_map.insert`
        // wait. This is what `get()` has always done; these four did not.
        // No inversion: the writer releases every node lock inside `add_node`
        // before it touches `path_map`, so the orders never interleave.
        let guard = self
            .path_map
            .get(path)
            .ok_or_else(|| IntegrationError::NotFound(format!("Node at path {} not found", path)))?;
        let uid = guard.value().unique_id();
        let ok = self.tensor_network.set_node_metadata_entry(&uid, key, value);
        drop(guard);

        if ok {
            Ok(())
        } else {
            Err(IntegrationError::NotFound(format!("Node with uid {} not found in network", uid)))
        }
    }

    /// Set semantic coordinates on a node (raw Q64.64 bytes, 16 bytes per dimension).
    pub fn set_semantic(&self, path: &str, coords: Vec<u8>) -> IntegrationResult<()> {
        // The path_map guard is held across the network read on purpose.
        // `embed_existing` swaps a stub for an embedded node by inserting the
        // new signature into `path_map` and THEN retiring the old node. A
        // reader that resolves the uid and drops the guard first can be
        // descheduled in that window and then look up a uid the network has
        // already released — a spurious NotFound for a key that never went
        // away. Holding the shard guard makes the swap's `path_map.insert`
        // wait. This is what `get()` has always done; these four did not.
        // No inversion: the writer releases every node lock inside `add_node`
        // before it touches `path_map`, so the orders never interleave.
        let guard = self
            .path_map
            .get(path)
            .ok_or_else(|| IntegrationError::NotFound(format!("Node at path {} not found", path)))?;
        let uid = guard.value().unique_id();
        let ok = self.tensor_network.set_node_semantic(&uid, coords);
        drop(guard);

        if ok {
            Ok(())
        } else {
            Err(IntegrationError::NotFound(format!("Node with uid {} not found in network", uid)))
        }
    }

    /// Get semantic coordinates for a node (raw Q64.64 bytes).
    pub fn get_semantic(&self, path: &str) -> IntegrationResult<Vec<u8>> {
        // The path_map guard is held across the network read on purpose.
        // `embed_existing` swaps a stub for an embedded node by inserting the
        // new signature into `path_map` and THEN retiring the old node. A
        // reader that resolves the uid and drops the guard first can be
        // descheduled in that window and then look up a uid the network has
        // already released — a spurious NotFound for a key that never went
        // away. Holding the shard guard makes the swap's `path_map.insert`
        // wait. This is what `get()` has always done; these four did not.
        // No inversion: the writer releases every node lock inside `add_node`
        // before it touches `path_map`, so the orders never interleave.
        let guard = self
            .path_map
            .get(path)
            .ok_or_else(|| IntegrationError::NotFound(format!("Node at path {} not found", path)))?;
        let uid = guard.value().unique_id();
        let found = self.tensor_network.get_node_semantic(&uid);
        drop(guard);

        found
            .ok_or_else(|| IntegrationError::NotFound(format!("Node with uid {} not found in network", uid)))
    }

    /// Delete a node at the specified path.
    ///
    /// A node with live children cannot be deleted (that would orphan them).
    /// The check and the removal run under two stripe locks — the node's own
    /// (which blocks a concurrent insert of a child *under* it, closing the
    /// has-no-children TOCTOU) and its parent's (which serializes the parent's
    /// child-list mutation against sibling inserts/deletes). The two stripes
    /// are taken in canonical order, so this never deadlocks against a
    /// concurrent delete.
    pub fn delete(&self, path: &str) -> IntegrationResult<()> {
        // Resolve the node and its parent before locking.
        let node_uid = match self.path_map.get(path) {
            Some(r) => r.value().unique_id(),
            None => {
                return Err(IntegrationError::NotFound(format!(
                    "Node at path {} not found",
                    path
                )))
            }
        };
        let parent_uid = self
            .path_ops
            .parent_path(path)
            .and_then(|pp| self.path_map.get(&pp).map(|s| s.unique_id()));

        // Hold the node's stripe (and the parent's, if any) for the whole
        // check-then-remove. `_guards` keeps both alive to the end of scope.
        let _guards = match &parent_uid {
            Some(puid) => {
                let (g1, g2) = self.parent_locks.lock_two(&node_uid, puid);
                (Some(g1), g2)
            }
            None => (Some(self.parent_locks.lock(&node_uid)), None),
        };

        // Re-check under the lock: the node may have been removed, or gained a
        // child, since the pre-lock read.
        if !self.path_map.contains_key(path) {
            return Err(IntegrationError::NotFound(format!(
                "Node at path {} not found",
                path
            )));
        }
        let children = self.list_children(path)?;
        if !children.is_empty() {
            return Err(IntegrationError::ValidationFailed(format!(
                "Cannot delete node at {} because it has {} children",
                path,
                children.len()
            )));
        }

        // Remove from path map, reverse map, and spatial index. The parent's
        // unique id is passed so the parent's child list drops this node even
        // when the node has no power cell (data-only nodes).
        if let Some((_, sig)) = self.path_map.remove(path) {
            let unique_id = sig.unique_id();
            self.id_to_path.remove(&unique_id);
            self.tensor_network
                .unregister_node_with_parent(&unique_id, parent_uid.as_deref());
        }

        Ok(())
    }

    /// List children of a node at the specified path (cloned).
    pub fn list_children(&self, path: &str) -> IntegrationResult<Vec<CompressedNode>> {
        let signature = self
            .path_map
            .get(path)
            .ok_or_else(|| IntegrationError::NotFound(format!("Node at path {} not found", path)))?;

        let children = self.tensor_network.children_of(signature.value());

        // Filter out deleted nodes (removed from path_map but still in tensor network)
        let result = children
            .into_iter()
            .filter(|child| self.path_map.contains_key(&child.metadata().key))
            .collect();

        Ok(result)
    }

    /// List all node paths under a specified path.
    pub fn list_subtree(&self, path: &str) -> IntegrationResult<Vec<String>> {
        let prefix = if path.ends_with('/') {
            path.to_string()
        } else {
            format!("{}/", path)
        };

        let mut result = Vec::new();
        for entry in self.path_map.iter() {
            let node_path = entry.key();
            if node_path.as_str() != path && (path == "/" || node_path.starts_with(&prefix)) {
                result.push(node_path.clone());
            }
        }

        Ok(result)
    }

    /// Get the path depth (number of components).
    fn path_depth(&self, path: &str) -> u32 {
        if path == "/" {
            return 0;
        }
        self.path_ops.split_path(path).len() as u32
    }

    /// Check if a path exists.
    pub fn exists(&self, path: &str) -> bool {
        self.path_map.contains_key(path)
    }

    /// Get the number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        self.path_map.len()
    }

    /// Get tree statistics.
    pub fn stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("node_count".to_string(), self.node_count().to_string());
        stats.insert(
            "dimension".to_string(),
            self.config.dimension().to_string(),
        );
        stats
    }

    /// Validate the tree structure.
    ///
    /// Checks all structural invariants: parent-child consistency,
    /// path_map ↔ id_to_path bidirectionality, and tensor network integrity.
    pub fn validate(&self) -> bool {
        // Empty tree is valid
        if self.path_map.is_empty() {
            return true;
        }

        // Check that tensor network is valid
        if !self.tensor_network.validate_network() {
            return false;
        }

        // All non-root paths have valid parents
        for entry in self.path_map.iter() {
            let path = entry.key();
            if path == "/" {
                continue;
            }
            if let Some(parent_path) = self.path_ops.parent_path(path) {
                if !self.path_map.contains_key(&parent_path) {
                    return false;
                }
            }
        }

        // path_map ↔ id_to_path bidirectional consistency
        for entry in self.path_map.iter() {
            let path = entry.key();
            let sig = entry.value();
            let uid = sig.unique_id();
            match self.id_to_path.get(&uid) {
                Some(reverse_path) if reverse_path.value() == path => {},
                _ => return false,
            }
        }
        for entry in self.id_to_path.iter() {
            let uid = entry.key();
            let path = entry.value();
            match self.path_map.get(path.as_str()) {
                Some(sig) if sig.value().unique_id() == *uid => {},
                _ => return false,
            }
        }

        // path_map and id_to_path must have the same size
        if self.path_map.len() != self.id_to_path.len() {
            return false;
        }

        true
    }

    /// Get the underlying tensor network (for spatial queries).
    pub fn tensor_network(&self) -> &HyperbolicTensorNetwork {
        &self.tensor_network
    }

    /// Resolve a unique_id to a path.
    pub fn path_for_id(&self, unique_id: &str) -> Option<String> {
        self.id_to_path.get(unique_id).map(|r| r.value().clone())
    }

    /// The hyperbolic (Poincaré) position of a stored node.
    ///
    /// Errors with `NotFound` for unknown paths and `OperationFailed` for
    /// data-only nodes, which have no geometric embedding.
    pub fn position(&self, path: &str) -> IntegrationResult<super::hyperbolic_geometry::HyperbolicPoint> {
        let sig = self
            .path_map
            .get(path)
            .ok_or_else(|| IntegrationError::NotFound(format!("Node at path {} not found", path)))?;
        let unique_id = sig.value().unique_id();
        drop(sig);
        self.tensor_network.get_point(&unique_id).ok_or_else(|| {
            IntegrationError::OperationFailed(format!(
                "node {} has no geometric embedding (data-only)",
                path
            ))
        })
    }

    /// Get the path operations.
    pub fn path_ops(&self) -> &dyn PathOperations {
        &*self.path_ops
    }

    /// Find the nearest stored node to an arbitrary Poincaré disk point.
    ///
    /// Answered exactly by the cell index. Returns (path, hyperbolic_distance).
    /// Errors when the index holds nothing.
    pub fn nearest_neighbor_point(&self, query: &super::hyperbolic_geometry::HyperbolicPoint) -> IntegrationResult<(String, g_math::fixed_point::FixedPoint)> {
        let (uid, dist) = self.tensor_network
            .nearest_neighbor_point(query)
            .ok_or_else(|| IntegrationError::OperationFailed(
                "No nodes in tree for nearest neighbor query".to_string()
            ))?;

        let path = self.id_to_path.get(&uid)
            .ok_or_else(|| IntegrationError::NotFound(
                format!("No path for unique_id {}", uid)
            ))?;

        Ok((path.value().clone(), dist))
    }

    /// Find the k nearest stored nodes to an arbitrary Poincaré disk point.
    ///
    /// Returns `(path, hyperbolic_distance)` sorted by ascending distance.
    ///
    /// `k == 0` is a request for nothing and is answered with nothing — the
    /// same as [`Self::find_nearest`]. Only an index that holds no nodes is an
    /// error. Conflating the two used to make `nearest_k(q, 0)` report "no
    /// nodes in tree" against a fully populated store, which is simply untrue.
    pub fn nearest_neighbor_point_k(&self, query: &super::hyperbolic_geometry::HyperbolicPoint, k: usize) -> IntegrationResult<Vec<(String, g_math::fixed_point::FixedPoint)>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let results = self.tensor_network.nearest_neighbor_point_k(query, k);
        if results.is_empty() {
            return Err(IntegrationError::OperationFailed(
                "No nodes in tree for nearest neighbor query".to_string()
            ));
        }

        let found = results.len();
        let mut paths = Vec::with_capacity(found);
        for (uid, dist) in results {
            if let Some(p) = self.id_to_path.get(&uid) {
                paths.push((p.value().clone(), dist));
            }
        }
        Self::note_unmapped(found, paths.len(), "nearest_neighbor_point_k");
        Ok(paths)
    }

    /// Report results the index found but that had no path.
    ///
    /// The spatial index is keyed by `unique_id`; `id_to_path` is what turns
    /// one back into a key. A delete removes the path first and the index
    /// entry after, so a concurrent query can briefly see an id with no path.
    /// That is benign and self-correcting.
    ///
    /// It is logged rather than swallowed because the alternative is a silent
    /// short result: the caller asked for `k` and got fewer, with nothing
    /// anywhere saying why. A *persistent* count here is not a race, it is
    /// `id_to_path` drifting from the index, which no other check would catch.
    fn note_unmapped(found: usize, kept: usize, op: &str) {
        if kept < found {
            log::debug!(
                "{}: {} of {} index hits had no path and were dropped, so the result \
                 is short by that many. Transient during a concurrent delete; \
                 persistent means id_to_path has drifted from the spatial index.",
                op,
                found - kept,
                found,
            );
        }
    }

    /// Find the k nearest stored nodes to the given path's position in hyperbolic space.
    /// Returns paths sorted by ascending hyperbolic distance.
    pub fn find_nearest(&self, path: &str, k: usize) -> IntegrationResult<Vec<(String, g_math::fixed_point::FixedPoint)>> {
        let sig = self
            .path_map
            .get(path)
            .ok_or_else(|| IntegrationError::NotFound(format!("Node at path {} not found", path)))?;

        let unique_id = sig.value().unique_id();
        // Drop the DashMap guard before calling tensor_network
        drop(sig);

        let point = self
            .tensor_network
            .get_point(&unique_id)
            .ok_or_else(|| IntegrationError::OperationFailed("Point not found in spatial index".to_string()))?;

        let results = self
            .tensor_network
            .nearest_neighbor_point_k(&point, k + 1); // +1 to exclude self

        // Self-exclusion is intended and is not a dropped result, so it is
        // discounted before counting what genuinely had no path.
        let mut considered = 0usize;
        let mut paths = Vec::new();
        for (uid, dist) in results {
            if uid == unique_id {
                continue; // Skip self
            }
            considered += 1;
            if let Some(p) = self.id_to_path.get(&uid) {
                paths.push((p.value().clone(), dist));
            }
        }
        Self::note_unmapped(considered, paths.len(), "find_nearest");
        paths.truncate(k);
        Ok(paths)
    }

    // -----------------------------------------------------------------------
    // Semantic dimensional distance queries
    // -----------------------------------------------------------------------

    /// Find the k nearest nodes by Euclidean distance across a dimensional slice
    /// of the semantic coordinate space.
    ///
    /// `query_coords`: raw Q64.64 bytes representing the query point.
    /// `k`: number of results.
    /// `dim_range`: which dimensions to compare (e.g., `16..33` for category axes).
    ///
    /// Returns `(path, distance)` sorted by distance ascending.
    pub fn nearest_semantic(
        &self,
        query_coords: &[u8],
        k: usize,
        dim_range: &Range<usize>,
    ) -> IntegrationResult<Vec<(String, g_math::fixed_point::FixedPoint)>> {
        // The network returns node keys (== the paths registered at insert),
        // already sorted ascending by (distance, key).
        Ok(self.tensor_network.nearest_semantic(query_coords, k, dim_range))
    }

    /// Find the k nearest nodes to an existing node by semantic dimensional distance.
    ///
    /// Convenience wrapper: reads the node's semantic coordinates, then calls
    /// `nearest_semantic`. The queried node is excluded from results.
    pub fn neighbors_semantic(
        &self,
        path: &str,
        k: usize,
        dim_range: &Range<usize>,
    ) -> IntegrationResult<Vec<(String, g_math::fixed_point::FixedPoint)>> {
        let coords = self.get_semantic(path)?;

        // Request k+1 to account for self, then filter.
        // The network returns node keys directly; resolve the queried
        // node's canonical key through path_map so a non-normalized input
        // still matches its own entry.
        let results = self.tensor_network.nearest_semantic(&coords, k + 1, dim_range);

        let self_key = self
            .path_map
            .get(path)
            .and_then(|r| self.id_to_path.get(&r.value().unique_id()).map(|p| p.value().clone()));

        let mut paths = Vec::with_capacity(k);
        for (key, dist) in results {
            if Some(&key) == self_key.as_ref() {
                continue; // Skip self
            }
            paths.push((key, dist));
            if paths.len() >= k {
                break;
            }
        }
        Ok(paths)
    }

    /// Find all stored nodes within hyperbolic radius of the given path.
    /// Returns paths and their distances.
    pub fn find_in_radius(&self, path: &str, radius: g_math::fixed_point::FixedPoint) -> IntegrationResult<Vec<(String, g_math::fixed_point::FixedPoint)>> {
        let sig = self
            .path_map
            .get(path)
            .ok_or_else(|| IntegrationError::NotFound(format!("Node at path {} not found", path)))?;

        let unique_id = sig.value().unique_id();
        drop(sig);

        let point = self
            .tensor_network
            .get_point(&unique_id)
            .ok_or_else(|| IntegrationError::OperationFailed("Point not found in spatial index".to_string()))?;

        let results = self
            .tensor_network
            .nodes_in_radius(&point, radius);

        let mut paths = Vec::new();
        for (uid, dist) in results {
            if uid == unique_id {
                continue; // Skip self
            }
            if let Some(p) = self.id_to_path.get(&uid) {
                paths.push((p.value().clone(), dist));
            }
        }
        Ok(paths)
    }
}

impl Debug for HyperbolicTreeTensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "HyperbolicTreeTensor(nodes={})", self.node_count())
    }
}

/// Thread-safe wrapper for the HyperbolicTreeTensor.
///
/// No outer RwLock needed: all interior state uses DashMap, Mutex, or RwLock
/// for fine-grained concurrency. Methods take `&self`.
pub type SharedHTT = Arc<HyperbolicTreeTensor>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_operations() {
        let path_ops = DefaultPathOps;

        let components = path_ops.split_path("/a/b/c");
        assert_eq!(
            components,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );

        let path = path_ops.join_path(&["a".to_string(), "b".to_string(), "c".to_string()]);
        assert_eq!(path, "/a/b/c");

        assert_eq!(path_ops.parent_path("/a/b/c"), Some("/a/b".to_string()));
        assert_eq!(path_ops.parent_path("/a"), Some("/".to_string()));
        assert_eq!(path_ops.parent_path("/"), None);

        assert_eq!(path_ops.last_component("/a/b/c"), Some("c".to_string()));
        assert_eq!(path_ops.last_component("/a"), Some("a".to_string()));
        assert_eq!(path_ops.last_component("/"), None);
    }

    #[test]
    fn test_tree_tensor_creation() {
        let config = HTTConfig::default();
        let tree = HyperbolicTreeTensor::new(config);
        assert_eq!(tree.node_count(), 0);
    }

    #[test]
    fn test_tree_tensor_operations() {
        let config = HTTConfig::default();
        let tree = HyperbolicTreeTensor::new(config);

        // Insert root node
        tree.insert("/", vec![], None).unwrap();
        assert_eq!(tree.node_count(), 1);
        assert!(tree.exists("/"));

        // Get root node
        let root = tree.get("/").unwrap();
        assert_eq!(root.metadata().key, "/");

        // Insert child nodes
        tree.insert("/child1", b"child1 data".to_vec(), None).unwrap();
        tree.insert("/child2", b"child2 data".to_vec(), None).unwrap();
        assert_eq!(tree.node_count(), 3);

        // Insert grandchild
        tree.insert("/child1/grandchild", b"grandchild data".to_vec(), None).unwrap();
        assert_eq!(tree.node_count(), 4);

        // List children
        let children = tree.list_children("/").unwrap();
        assert_eq!(children.len(), 2);

        let child_keys: Vec<&str> = children.iter().map(|c| c.metadata().key.as_str()).collect();
        assert!(child_keys.contains(&"/child1"));
        assert!(child_keys.contains(&"/child2"));

        // Update a node
        tree.update_value("/child1", b"updated data".to_vec()).unwrap();
        let updated = tree.get("/child1").unwrap();
        assert_eq!(updated.value(), b"updated data");

        // List subtree
        let subtree = tree.list_subtree("/").unwrap();
        assert_eq!(subtree.len(), 3); // child1, child2, child1/grandchild

        // Try to delete node with children (should fail)
        let result = tree.delete("/child1");
        assert!(result.is_err());

        // Delete leaf node
        tree.delete("/child1/grandchild").unwrap();
        assert_eq!(tree.node_count(), 3);

        // Now delete former parent
        tree.delete("/child1").unwrap();
        assert_eq!(tree.node_count(), 2);
    }

    #[test]
    fn test_tree_validation() {
        let config = HTTConfig::default();
        let tree = HyperbolicTreeTensor::new(config);

        // Empty tree should be valid
        assert!(tree.validate());

        // Add some nodes
        tree.insert("/", vec![], None).unwrap();
        tree.insert("/child", b"child data".to_vec(), None).unwrap();

        // Tree with nodes should still be valid
        assert!(tree.validate());
    }

    #[test]
    fn test_id_to_path_mapping() {
        let config = HTTConfig::default();
        let tree = HyperbolicTreeTensor::new(config);

        tree.insert("/", vec![], None).unwrap();
        tree.insert("/test", b"test".to_vec(), None).unwrap();

        // Verify id_to_path works
        let uid = tree.path_map.get("/test").unwrap().value().unique_id();
        let resolved_path = tree.path_for_id(&uid);
        assert_eq!(resolved_path, Some("/test".to_string()));
    }

    #[test]
    fn test_find_nearest() {
        let config = HTTConfig::default();
        let tree = HyperbolicTreeTensor::new(config);

        tree.insert("/", vec![], None).unwrap();
        tree.insert("/a", b"a".to_vec(), None).unwrap();
        tree.insert("/b", b"b".to_vec(), None).unwrap();
        tree.insert("/c", b"c".to_vec(), None).unwrap();
        tree.insert("/a/child", b"ac".to_vec(), None).unwrap();

        // Find nearest to /a — should return other nodes sorted by distance
        let nearest = tree.find_nearest("/a", 3).unwrap();
        assert!(!nearest.is_empty());
        assert!(nearest.len() <= 3);

        // Results should not include /a itself
        let paths: Vec<&str> = nearest.iter().map(|(p, _)| p.as_str()).collect();
        assert!(!paths.contains(&"/a"));

        // Distances should be in ascending order
        for i in 1..nearest.len() {
            assert!(nearest[i].1 >= nearest[i - 1].1);
        }
    }

    #[test]
    fn test_find_in_radius() {
        use g_math::fixed_point::FixedPoint;

        let config = HTTConfig::default();
        let tree = HyperbolicTreeTensor::new(config);

        tree.insert("/", vec![], None).unwrap();
        tree.insert("/a", b"a".to_vec(), None).unwrap();
        tree.insert("/b", b"b".to_vec(), None).unwrap();

        // With a very large radius, should find other nodes
        let large_radius = FixedPoint::from_int(10);
        let results = tree.find_in_radius("/", large_radius).unwrap();
        assert!(results.len() >= 2, "Expected at least 2 nodes within large radius, got {}", results.len());

        // With a tiny radius, should find few or no nodes
        let tiny_radius = FixedPoint::from_int(1) / FixedPoint::from_int(10000);
        let results = tree.find_in_radius("/", tiny_radius).unwrap();
        // The root is at origin, children are close but not at zero distance
        // So with a very tiny radius, we might find 0 or a few
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_delete_unregisters_spatial() {
        let config = HTTConfig::default();
        let tree = HyperbolicTreeTensor::new(config);

        tree.insert("/", vec![], None).unwrap();
        tree.insert("/leaf", b"leaf".to_vec(), None).unwrap();

        // Get the unique_id before deletion
        let unique_id = tree.path_map.get("/leaf").unwrap().value().unique_id();

        // Verify point exists in tensor network
        assert!(tree.tensor_network().get_point(&unique_id).is_some());

        // Delete the leaf
        tree.delete("/leaf").unwrap();

        // Point should be removed from spatial index
        assert!(tree.tensor_network().get_point(&unique_id).is_none());
        assert!(tree.path_for_id(&unique_id).is_none());
    }
}
