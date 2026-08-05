//! tensor_network.rs - Hyperbolic Tensor Network for Hierarchical Data
//!
//! This module implements the core data network over hyperbolic space:
//!
//! - Nodes embedded in the Poincaré disk with parent-child geometric relationships
//! - O(1) spatial lookups via the hash table's bucket structure
//! - Exact storage of node data (metadata + raw value bytes)
//! - Spatial index for range and nearest-neighbor queries

use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::fmt::{self, Debug, Formatter};
use std::ops::Range;
use std::sync::{Mutex, RwLock};
use dashmap::DashMap;
use g_math::fixed_point::{FixedPoint, FixedVector};
use super::hyperbolic_geometry::{HyperbolicPoint, ratio_to_distance};
use super::hash_table::{GeometricSignature, HyperbolicHashTable};
use super::klein::{self, KleinPoint, PowerCell, PointLocationGrid};
use crate::constants;
use crate::metric_tree::EuclideanMetric;
use crate::semantic_index::SemanticIndexCache;

/// Exact metadata stored per node. Never lossy-compressed.
#[derive(Clone, Debug)]
pub struct NodeMetadata {
    /// Node key (path)
    pub key: String,
    /// Content type
    pub content_type: Option<String>,
    /// User-defined metadata
    pub metadata: HashMap<String, String>,
    /// Creation timestamp (seconds since epoch)
    pub created_at: u64,
    /// Last update timestamp
    pub updated_at: u64,
}

impl NodeMetadata {
    /// Create new metadata with current timestamp.
    pub fn new(key: String, content_type: Option<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            key,
            content_type,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Touch the updated_at timestamp.
    pub fn touch(&mut self) {
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}

/// Siblings per rainbow band: children beyond this cascade to the next
/// concentric ring. Derived from signature quantization capacity (see
/// `create_child_point`); 256 keeps a wide margin below the ~700-sibling
/// single-ring collision threshold.
pub const RAINBOW_BAND_CAPACITY: u32 = 256;
/// Each band steps outward by τ / RAINBOW_BAND_STEP_DIV.
const RAINBOW_BAND_STEP_DIV: i32 = 64;
/// Warn when fan-out reaches this many bands (spacing degradation).
const RAINBOW_BAND_WARN: u32 = 32;

/// A node in the hyperbolic tensor network.
///
/// Stores exact metadata and raw value bytes. Despite the historical name,
/// no compression is performed — the name is retained for API stability and
/// will be revisited before a 1.0 release.
#[derive(Clone, Debug)]
pub struct CompressedNode {
    /// Exact metadata (key, content_type, user metadata, timestamps)
    node_metadata: NodeMetadata,
    /// Raw value bytes (exact, no compression)
    value: Vec<u8>,
    /// Child node references by their geometric signatures
    children: Vec<GeometricSignature>,
    /// Semantic coordinates — raw Q64.64 bytes (16 bytes per dimension).
    /// Empty if no semantic dimensions are set.
    semantic_coords: Vec<u8>,
}

impl CompressedNode {
    /// Create a new node.
    pub fn new(metadata: NodeMetadata, value: Vec<u8>) -> Self {
        Self {
            node_metadata: metadata,
            value,
            children: Vec::new(),
            semantic_coords: Vec::new(),
        }
    }

    /// Get the node metadata.
    pub fn metadata(&self) -> &NodeMetadata {
        &self.node_metadata
    }

    /// Get a mutable reference to the metadata.
    pub fn metadata_mut(&mut self) -> &mut NodeMetadata {
        &mut self.node_metadata
    }

    /// Get the raw value bytes.
    pub fn value(&self) -> &[u8] {
        &self.value
    }

    /// Update the value bytes.
    pub fn update_value(&mut self, value: Vec<u8>) {
        self.value = value;
        self.node_metadata.touch();
    }

    /// Add a child node reference.
    pub fn add_child(&mut self, signature: GeometricSignature) {
        self.children.push(signature);
    }

    /// Remove a child reference by unique id (used when the child is deleted).
    pub fn remove_child(&mut self, unique_id: &str) {
        self.children.retain(|sig| sig.unique_id() != unique_id);
    }

    /// Get the child node references.
    pub fn children(&self) -> &[GeometricSignature] {
        &self.children
    }

    /// Check if this node has any children.
    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    /// Get the number of children.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Get the semantic coordinates (raw Q64.64 bytes, 16 bytes per dimension).
    pub fn semantic_coords(&self) -> &[u8] {
        &self.semantic_coords
    }

    /// Set the semantic coordinates (raw Q64.64 bytes).
    pub fn set_semantic_coords(&mut self, coords: Vec<u8>) {
        self.semantic_coords = coords;
    }
}

/// Hyperbolic Tensor Network for tree data representation.
///
/// Embeds hierarchical data into the Poincaré disk model using Sarkar's
/// cone-based construction. Each node occupies a point in hyperbolic space,
/// with children placed at hyperbolic distance τ from their parent using
/// Möbius reflections to preserve the tree structure as a Delaunay graph.
///
/// Internal maps use DashMap for lock-free concurrent reads, preparing
/// for multi-threaded access in later phases.
pub struct HyperbolicTensorNetwork {
    /// Hyperbolic hash table for O(1) spatial lookups
    hash_table: HyperbolicHashTable,
    /// Nodes mapped by their unique_id
    nodes: DashMap<String, CompressedNode>,
    /// Points in the Poincaré disk for each node (keyed by unique_id)
    point_map: DashMap<String, HyperbolicPoint>,
    /// Root node signature (Mutex: set once during root insert)
    root_signature: Mutex<Option<GeometricSignature>>,
    /// Sarkar embedding scale factor: parent-child hyperbolic distance
    tau: FixedPoint,
    /// Per-parent child count for Sarkar cone angular placement
    child_counts: DashMap<String, u32>,
    /// Klein model points: unique_id → KleinPoint (Nielsen power diagram)
    klein_points: DashMap<String, KleinPoint>,
    /// Power cells: unique_id → PowerCell (Voronoi regions in Klein model)
    power_cells: DashMap<String, PowerCell>,
    /// Point location grid for O(1) nearest-neighbor queries (RwLock: brief write hold on insert/delete)
    point_location: RwLock<PointLocationGrid>,
    /// Lazy per-slice VP-tree cache for semantic KNN, invalidated by a
    /// semantic-epoch counter (see `docs/SEMANTIC_INDEX.md`)
    semantic_index: SemanticIndexCache,
}

impl HyperbolicTensorNetwork {
    /// Default grid resolution for the point location grid.
    const DEFAULT_GRID_RESOLUTION: usize = 64;

    /// Create a new hyperbolic tensor network with the given Sarkar scale factor τ.
    pub fn new(dimension: usize, tau: FixedPoint) -> Self {
        Self::with_grid_resolution(dimension, tau, Self::DEFAULT_GRID_RESOLUTION)
    }

    /// Create a new network with a specific grid resolution.
    pub fn with_grid_resolution(dimension: usize, tau: FixedPoint, grid_resolution: usize) -> Self {
        // Semantic coordinates and persisted geometry assume a 16-byte Q64.64
        // FixedPoint. GMATH_PROFILE is a build-time env var, so a rebuild under
        // a different profile would silently reinterpret those bytes — fail
        // loudly instead.
        assert_eq!(
            FixedPoint::raw_byte_len(), 16,
            "horon-engine requires the 16-byte Q64.64 g_math profile (GMATH_PROFILE=embedded); \
             rebuild with the correct profile"
        );

        let hash_table = HyperbolicHashTable::new(dimension);

        Self {
            hash_table,
            nodes: DashMap::new(),
            point_map: DashMap::new(),
            root_signature: Mutex::new(None),
            tau,
            child_counts: DashMap::new(),
            klein_points: DashMap::new(),
            power_cells: DashMap::new(),
            point_location: RwLock::new(PointLocationGrid::with_dimension(grid_resolution, dimension)),
            semantic_index: SemanticIndexCache::new(),
        }
    }

    /// Add a node to the DashMap without computing geometric embedding.
    ///
    /// Creates a key-derived unique_id for path_map/id_to_path lookups.
    /// Semantic queries (nearest_semantic, neighbors_semantic, get/set)
    /// work normally. Spatial queries (nearest, neighbors) will not
    /// find this node until it is embedded.
    pub fn add_node_data_only(&self, metadata: NodeMetadata, value: Vec<u8>, _level: u32) -> String {
        use sha3::{Sha3_256, Digest as _};
        let mut hasher = Sha3_256::new();
        hasher.update(b"data_only:");
        hasher.update(metadata.key.as_bytes());
        let unique_id = hex::encode(&hasher.finalize()[..16]);

        let node = CompressedNode::new(metadata, value);
        self.nodes.insert(unique_id.clone(), node);
        // Fresh nodes carry no semantic coords, but bump anyway: free
        // insurance against future insert-with-coords paths.
        self.semantic_index.bump();
        unique_id
    }

    /// Add a node to the tensor network.
    ///
    /// Computes a position in the Poincaré disk using Sarkar's cone construction:
    /// children are placed at hyperbolic distance τ from their parent, at
    /// golden-angle-spaced angles in the parent's reflected frame.
    pub fn add_node(&self,
                    metadata: NodeMetadata,
                    value: Vec<u8>,
                    parent_signature: Option<&GeometricSignature>,
                    level: u32) -> Option<GeometricSignature> {
        self.add_node_inner(metadata, value, parent_signature, level, None)
    }

    /// Add a node with an explicit child_index for deterministic Sarkar reconstruction.
    ///
    /// Used during snapshot replay: the stored child_index ensures the node gets
    /// the same geometric position regardless of replay order.
    pub fn add_node_positioned(&self,
                    metadata: NodeMetadata,
                    value: Vec<u8>,
                    parent_signature: Option<&GeometricSignature>,
                    level: u32,
                    child_index: u32) -> Option<GeometricSignature> {
        self.add_node_inner(metadata, value, parent_signature, level, Some(child_index))
    }

    fn add_node_inner(&self,
                    metadata: NodeMetadata,
                    value: Vec<u8>,
                    parent_signature: Option<&GeometricSignature>,
                    level: u32,
                    child_index_hint: Option<u32>) -> Option<GeometricSignature> {
        // Q64.64 precision supports depth ≈ 44/τ before sibling separation
        // degrades near the disk boundary. Warn as inserts approach the
        // budget rather than silently losing angular precision.
        if self.tau > FixedPoint::from_int(0) {
            let depth_budget = (FixedPoint::from_int(44) / self.tau).to_int() as u32;
            if level.saturating_mul(10) >= depth_budget.saturating_mul(9) {
                log::warn!(
                    "insert '{}' at depth {} approaches the Q64.64 precision budget (~{} levels at tau={}); sibling positions may lose separation",
                    metadata.key, level, depth_budget, self.tau.to_f64()
                );
            }
        }
        // Resolve the child index and geometric point WITHOUT yet advancing
        // the parent's sibling counter — the counter is committed only after
        // the insert is known to succeed (below), so a refused collision or
        // any other early return can never leave a gap in the index sequence
        // (which would make placement depend on transient failed inserts).
        //
        // The requested slot may already be occupied by a DIFFERENT key, and
        // that is not always a precision failure. Not every insert path feeds
        // the sibling counter: data-only nodes (`add_node_data_only`) and
        // ancestors auto-created during replay take positions without
        // reserving an index, so a counter-assigned index and an index
        // recorded in `_child_index` can name the same point. Whichever node
        // is replayed second then lands on an occupied slot.
        //
        // Refusing outright made the file unopenable — the failure surfaced as
        // `Failed to add node to tensor network` on reopen, after a
        // `put_data_only` + `compact()`, with no way to recover the data. So
        // probe forward for the next free slot instead. A file whose slots do
        // not collide is unaffected: the first probe succeeds, placement is
        // unchanged, and on-disk geometry stays bit-identical.
        const MAX_PROBE: u32 = 1024;
        let mut probe = child_index_hint;
        let mut resolved = None;

        for _ in 0..MAX_PROBE {
            let (point, child_index) = match parent_signature {
                Some(parent_sig) => self.compute_child_placement(parent_sig, probe),
                None => (self.hash_table.poincare_disk().origin(), 0),
            };

            let signature = self.hash_table.create_signature(&point, level)?;
            let unique_id = signature.unique_id();

            // Re-inserting the same key is an update, not a collision.
            let taken_by_other = self
                .nodes
                .get(&unique_id)
                .map(|existing| existing.metadata().key != metadata.key)
                .unwrap_or(false);

            if !taken_by_other {
                resolved = Some((point, child_index, signature, unique_id));
                break;
            }

            // The root has exactly one slot (the origin); there is nothing to
            // probe, and a genuine clash there is a real error.
            if parent_signature.is_none() {
                break;
            }
            probe = Some(child_index.saturating_add(1));
        }

        let Some((point, child_index, signature, unique_id)) = resolved else {
            log::error!(
                "could not place '{}': no free sibling slot within {} probes — \
                 precision budget exceeded (depth/fan-out); insert refused",
                metadata.key, MAX_PROBE
            );
            return None;
        };

        // Commit the sibling-counter advance now that the insert will succeed.
        // Callers hold the parent stripe lock, so peek-then-commit is atomic
        // with respect to other inserts under the same parent.
        if let Some(parent_sig) = parent_signature {
            // Commit against the slot actually taken, not the one requested —
            // they differ when the probe above had to step past an occupied
            // position. `commit_child_index` takes the max, so for a
            // first-probe hit this is identical to the previous behaviour.
            self.commit_child_index(&parent_sig.unique_id(), Some(child_index), child_index);
        }

        let node = CompressedNode::new(metadata, value);
        self.nodes.insert(unique_id.clone(), node);
        // Fresh nodes carry no semantic coords, but bump anyway: free
        // insurance against future insert-with-coords paths.
        self.semantic_index.bump();

        // Store child_index in metadata for deterministic snapshot reconstruction
        if let Some(mut node_ref) = self.nodes.get_mut(&unique_id) {
            node_ref.metadata_mut().metadata.insert(
                "_child_index".to_string(),
                child_index.to_string(),
            );
        }
        self.point_map.insert(unique_id.clone(), point.clone());

        // Register in spatial index (VP-tree), reusing the bucket hash from
        // create_signature to skip a second find_bucket call
        self.hash_table.register_node_with_hint(&point, &unique_id, level, Some(signature.hash()));

        // --- Nielsen power diagram maintenance ---
        let klein_pt = klein::poincare_to_klein(&point);
        self.klein_points.insert(unique_id.clone(), klein_pt.clone());

        if let Some(parent_sig) = parent_signature {
            let parent_id = parent_sig.unique_id();

            // Compute bisectors between new leaf and parent
            if let Some(parent_klein) = self.klein_points.get(&parent_id).map(|r| r.value().clone()) {
                // Half-plane for leaf's cell: toward parent
                let hp_leaf = klein::compute_bisector(&klein_pt, &parent_klein, &parent_id);
                // Half-plane for parent's cell: toward leaf (shrinks parent)
                let hp_parent = klein::compute_bisector(&parent_klein, &klein_pt, &unique_id);

                // Create leaf's power cell (1 neighbor: parent)
                self.power_cells.insert(unique_id.clone(), PowerCell {
                    node_id: unique_id.clone(),
                    site: klein_pt.clone(),
                    half_planes: vec![hp_leaf],
                });

                // Add half-plane to parent's cell (shrinks it)
                if let Some(mut parent_cell) = self.power_cells.get_mut(&parent_id) {
                    parent_cell.half_planes.push(hp_parent);
                }

                // Update point location grid: carve new leaf's region from parent
                self.point_location.write().unwrap_or_else(|e| e.into_inner()).update_insert(&parent_id, &unique_id, &klein_pt, &parent_klein);
            }
        } else {
            // Root node: cell covers entire disk (no half-planes)
            self.power_cells.insert(unique_id.clone(), PowerCell {
                node_id: unique_id.clone(),
                site: klein_pt.clone(),
                half_planes: Vec::new(),
            });

            // Build initial grid with just the root
            let sites: Vec<(String, KleinPoint)> = vec![
                (unique_id.clone(), klein_pt.clone()),
            ];
            self.point_location.write().unwrap_or_else(|e| e.into_inner()).build(&sites);
        }
        // --- End power diagram maintenance ---

        if parent_signature.is_none() {
            let mut root = self.root_signature.lock().unwrap_or_else(|e| e.into_inner());
            if root.is_none() {
                *root = Some(signature.clone());
            }
        }

        if let Some(parent_sig) = parent_signature {
            if let Some(mut parent_node) = self.nodes.get_mut(&parent_sig.unique_id()) {
                parent_node.add_child(signature.clone());
            }
        }

        Some(signature)
    }

    /// Place a child node using Sarkar's cone construction.
    ///
    /// 1. Look up parent's position in the Poincaré disk
    /// 2. Compute child angle: child_count × golden_angle (irrational spacing)
    /// 3. Create child at origin frame: (r·cos θ, r·sin θ, 0, …) where r = tanh(τ/2)
    /// 4. Möbius-reflect from origin to parent's position
    ///
    /// This produces embeddings where the tree IS its own Delaunay triangulation
    /// (Sarkar 2011), with (1+ε) distance distortion for any tree.
    /// Resolve the child index and Poincaré-disk point for a new child of
    /// `parent_signature`, **without** mutating the parent's sibling counter.
    ///
    /// The returned index is what the child *would* receive; the caller
    /// commits the counter advance via [`Self::commit_child_index`] once the
    /// insert is certain to succeed. Splitting peek from commit keeps the
    /// sibling-index sequence gap-free across refused/failed inserts, which
    /// is what makes placement independent of transient failures.
    fn compute_child_placement(&self, parent_signature: &GeometricSignature, child_index_hint: Option<u32>) -> (HyperbolicPoint, u32) {
        let parent_id = parent_signature.unique_id();
        let dimension = self.hash_table.poincare_disk().dimension();

        // Get parent position (root is at origin)
        let parent_point = self.point_map.get(&parent_id)
            .map(|r| r.value().clone())
            .unwrap_or_else(|| HyperbolicPoint::origin(dimension));

        // Peek the child index: explicit hint (snapshot replay) or the current
        // auto-increment counter. No mutation here.
        let child_index = child_index_hint
            .unwrap_or_else(|| self.child_counts.get(&parent_id).map(|r| *r.value()).unwrap_or(0));

        // Rainbow bands: siblings fill concentric rings instead of exhausting
        // one circle. Band 0 sits at the classic Sarkar distance τ —
        // bit-identical to the historical placement, so existing trees keep
        // their exact geometry. Each full band cascades outward by τ/64:
        // angular quantization capacity is renewed per ring, so fan-out is
        // collision-free by construction rather than guarded by warnings.
        // The angle sequence runs continuously across bands (a discretized
        // Vogel/phyllotaxis spiral).
        //
        // Capacity math: signatures quantize positions to 1e-3 cells; at
        // ring radius tanh(τ/2) the golden-angle minimum chord falls below a
        // cell around ~700 siblings (earlier for deep parents, which Möbius
        // reflection compresses). 256 leaves a wide margin; the τ/64 radial
        // step keeps adjacent bands ~6 cells apart.
        let band = child_index / RAINBOW_BAND_CAPACITY;
        if band >= RAINBOW_BAND_WARN {
            log::warn!(
                "parent of child '{}' reached rainbow band {} ({}+ siblings): placement \
                 remains collision-free but subtree spacing is degrading — consider restructuring",
                child_index, band, child_index
            );
        }
        let effective_tau = self.tau
            + self.tau * FixedPoint::from_int(band as i32)
                / FixedPoint::from_int(RAINBOW_BAND_STEP_DIV);
        let half_tau = effective_tau / FixedPoint::from_int(2);
        let r = half_tau.tanh();

        // Child angle: golden angle spacing ensures no clustering regardless of child count
        let angle = FixedPoint::from_int(child_index as i32) * constants::golden_angle();

        // Build child position in the origin frame
        let mut child_at_origin = FixedVector::new(dimension);
        if dimension >= 2 {
            let (sin_a, cos_a) = angle.sincos();
            child_at_origin[0] = r * cos_a;
            child_at_origin[1] = r * sin_a;
            // Higher dimensions stay at zero — children lie in a 2D geodesic submanifold
        } else {
            // 1D: alternate left/right
            child_at_origin[0] = if child_index % 2 == 0 { r } else { -r };
        }
        let child_point = HyperbolicPoint::new(child_at_origin);

        // Möbius-reflect from origin to parent's position
        (child_point.reflect_from_origin(&parent_point), child_index)
    }

    /// Advance the parent's sibling counter after a successful insert.
    ///
    /// For an auto-increment insert this bumps the counter past `child_index`;
    /// for a hinted (snapshot-replay) insert it tracks the running maximum so
    /// later auto-increment inserts never collide with a replayed index.
    fn commit_child_index(&self, parent_id: &str, child_index_hint: Option<u32>, child_index: u32) {
        let next = match child_index_hint {
            Some(hint) => {
                let current = self.child_counts.get(parent_id).map(|r| *r.value()).unwrap_or(0);
                current.max(hint + 1)
            }
            None => child_index + 1,
        };
        self.child_counts.insert(parent_id.to_string(), next);
    }

    /// Get a node by its signature (returns cloned value).
    pub fn get_node_by_signature(&self, signature: &GeometricSignature) -> Option<CompressedNode> {
        self.nodes.get(&signature.unique_id()).map(|r| r.value().clone())
    }

    /// Update the value of a node by its unique_id.
    pub fn update_node_value(&self, unique_id: &str, value: Vec<u8>) -> bool {
        if let Some(mut node) = self.nodes.get_mut(unique_id) {
            node.update_value(value);
            true
        } else {
            false
        }
    }

    /// Set a metadata key-value pair on a node by its unique_id.
    pub fn set_node_metadata_entry(&self, unique_id: &str, key: &str, val: &str) -> bool {
        if let Some(mut node) = self.nodes.get_mut(unique_id) {
            node.metadata_mut().metadata.insert(key.to_string(), val.to_string());
            true
        } else {
            false
        }
    }

    /// Set semantic coordinates on a node by its unique_id.
    pub fn set_node_semantic(&self, unique_id: &str, coords: Vec<u8>) -> bool {
        if let Some(mut node) = self.nodes.get_mut(unique_id) {
            node.set_semantic_coords(coords);
            drop(node); // release the shard before the epoch bump
            // Mutation first, then bump: a builder that pre-read the old
            // epoch tags its tree stale (see semantic_index.rs).
            self.semantic_index.bump();
            true
        } else {
            false
        }
    }

    /// Get semantic coordinates for a node by its unique_id.
    pub fn get_node_semantic(&self, unique_id: &str) -> Option<Vec<u8>> {
        self.nodes.get(unique_id).map(|node| node.semantic_coords().to_vec())
    }

    /// Get the root node (cloned).
    pub fn root_node(&self) -> Option<CompressedNode> {
        let root = self.root_signature.lock().unwrap_or_else(|e| e.into_inner());
        root.as_ref().and_then(|sig| {
            self.get_node_by_signature(sig)
        })
    }

    /// Get the root node signature (cloned).
    pub fn root_signature(&self) -> Option<GeometricSignature> {
        self.root_signature.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Get the children of a node by its signature (cloned).
    pub fn children_of(&self, signature: &GeometricSignature) -> Vec<CompressedNode> {
        let node = match self.nodes.get(&signature.unique_id()) {
            Some(r) => r.value().clone(),
            None => return Vec::new(),
        };

        let mut children = Vec::new();
        for child_sig in node.children() {
            if let Some(child) = self.nodes.get(&child_sig.unique_id()) {
                children.push(child.value().clone());
            }
        }

        children
    }

    /// Get the hyperbolic point for a node (cloned).
    pub fn get_point(&self, unique_id: &str) -> Option<HyperbolicPoint> {
        self.point_map.get(unique_id).map(|r| r.value().clone())
    }

    /// Monotone counter of semantic-relevant mutations (coordinate writes,
    /// inserts, deletes). External caches — like the semantic disk's
    /// derived-position index — use it exactly as the internal per-slice
    /// cache does: tag on build, rebuild when it has advanced.
    pub fn semantic_epoch(&self) -> u64 {
        self.semantic_index.epoch()
    }

    /// Get the hyperbolic hash table.
    pub fn hash_table(&self) -> &HyperbolicHashTable {
        &self.hash_table
    }

    /// Get the number of nodes in the network.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Remove a node-map entry that has NO geometric registration — the
    /// data-only entry retired by `embed_existing` after its embedded
    /// replacement went live under a new signature-derived id. Not for
    /// embedded nodes: those need [`Self::unregister_node_with_parent`].
    pub fn remove_detached_node(&self, unique_id: &str) {
        self.nodes.remove(unique_id);
        // The candidate set changed shape (old id gone): invalidate the semantic index.
        self.semantic_index.bump();
    }

    /// Unregister a node from the spatial index (for deletion).
    ///
    /// Prefer [`Self::unregister_node_with_parent`] when the parent is known:
    /// data-only nodes have no power cell, so the parent cannot always be
    /// derived here, and the parent's child list must drop the deleted node.
    pub fn unregister_node(&self, unique_id: &str) {
        self.unregister_node_with_parent(unique_id, None)
    }

    /// Unregister a node, removing it from the node map and from its parent's
    /// child list. `parent_uid` is used when the parent cannot be derived from
    /// the power diagram (e.g. data-only nodes).
    pub fn unregister_node_with_parent(&self, unique_id: &str, parent_uid: Option<&str>) {
        // --- Klein/power diagram cleanup ---
        // Find the parent by checking which cell has this node as a neighbor
        let parent_id: Option<String> = self.power_cells.get(unique_id)
            .and_then(|cell| cell.half_planes.first().map(|hp| hp.neighbor_id.clone()));

        if let Some(ref pid) = parent_id {
            // Remove the half-plane from parent's cell that points to this node
            if let Some(mut parent_cell) = self.power_cells.get_mut(pid) {
                parent_cell.half_planes.retain(|hp| hp.neighbor_id != unique_id);
            }

            // Reassign grid tiles from deleted node to parent
            self.point_location.write().unwrap_or_else(|e| e.into_inner()).update_delete(unique_id, pid);
        }

        self.klein_points.remove(unique_id);
        self.power_cells.remove(unique_id);
        self.child_counts.remove(unique_id);
        // --- End Klein cleanup ---

        self.hash_table.unregister_node(unique_id);
        self.point_map.remove(unique_id);

        // Remove the node itself — a ghost entry would keep serving stale
        // semantic coordinates to nearest_semantic and permanently fail
        // validate()'s nodes↔point_map invariant.
        self.nodes.remove(unique_id);
        // Deletion changes the semantic candidate set: invalidate the semantic index.
        self.semantic_index.bump();

        // Drop the deleted node from its parent's child list.
        if let Some(pid) = parent_uid.map(str::to_string).or(parent_id) {
            if let Some(mut parent_node) = self.nodes.get_mut(&pid) {
                parent_node.remove_child(unique_id);
            }
        }

        // If the root itself was deleted, clear the root signature.
        let mut root = self.root_signature.lock().unwrap_or_else(|e| e.into_inner());
        if root.as_ref().map(|s| s.unique_id()).as_deref() == Some(unique_id) {
            *root = None;
        }
    }

    /// Find all descendants of a node using the spatial index.
    ///
    /// Uses the parent's stored point + a τ-based radius to find all nodes
    /// within the Sarkar cone. Radius = 3·τ covers ~3 levels of descendants.
    pub fn find_descendants_spatial(&self, signature: &GeometricSignature) -> Vec<(String, FixedPoint)> {
        let unique_id = signature.unique_id();
        let point = match self.point_map.get(&unique_id) {
            Some(r) => r.value().clone(),
            None => return Vec::new(),
        };

        // Subtree radius: 3·τ — covers descendants within 3 levels of the Sarkar cone
        let subtree_radius = FixedPoint::from_int(3) * self.tau;

        self.hash_table.find_nodes_in_radius(&point, subtree_radius)
            .into_iter()
            .filter(|(uid, _)| *uid != unique_id)
            .collect()
    }

    /// Find the nearest node to an arbitrary point using the power diagram grid.
    ///
    /// Algorithm:
    /// 1. Convert Poincaré → Klein coordinates: O(d)
    /// 2. Grid lookup for candidate: O(1)
    /// 3. Power-distance pre-filter: rank all tree neighbors by pd (~24ns each)
    /// 4. Hyperbolic distance verification: only top-K candidates (~59µs each)
    /// 5. Return true nearest
    ///
    /// Uses hyperbolic_ratio (~200ns) for all comparisons instead of
    /// hyperbolic_distance (~62µs), computing the full distance only once
    /// for the final winner. For high-degree nodes, power distance pre-filter
    /// ranks neighbors at ~24ns each, then ratio-verifies top-K.
    ///
    /// Falls back to VP-tree KNN if grid misses.
    pub fn nearest_neighbor_point(&self, query_poincare: &HyperbolicPoint) -> Option<(String, FixedPoint)> {
        if self.klein_points.is_empty() {
            return None;
        }

        // Max neighbors to verify with hyperbolic_ratio after power-distance pre-filter
        const VERIFY_K: usize = 5;

        let query_klein = klein::poincare_to_klein(query_poincare);

        // O(1) grid lookup (read lock — brief, shared with other readers)
        let grid_candidate = self.point_location.read().unwrap_or_else(|e| e.into_inner()).query(&query_klein.coords).map(|s| s.to_string());
        if let Some(candidate_id) = grid_candidate.as_deref() {
            // Clone the power cell data we need to avoid holding DashMap guards
            let cell_data = self.power_cells.get(candidate_id).map(|r| r.value().clone());

            if let Some(cell) = cell_data {
                let n_neighbors = cell.half_planes.len();

                if n_neighbors <= VERIFY_K {
                    // Few neighbors: verify all with cheap hyperbolic_ratio
                    let mut best_id = candidate_id.to_string();
                    let mut best_ratio = self.point_map.get(candidate_id)
                        .map(|p| query_poincare.hyperbolic_ratio(p.value()))
                        .unwrap_or(FixedPoint::from_int(1));

                    for hp in &cell.half_planes {
                        if let Some(neighbor_point) = self.point_map.get(&hp.neighbor_id) {
                            let r = query_poincare.hyperbolic_ratio(neighbor_point.value());
                            if r < best_ratio {
                                best_ratio = r;
                                best_id = hp.neighbor_id.clone();
                            }
                        }
                    }

                    // The winning ratio already determines the exact distance
                    // (d = 2·atanh(ratio)); no second lookup or sentinel needed,
                    // and this stays correct even if the winner's point is
                    // concurrently removed after selection.
                    return Some((best_id, ratio_to_distance(best_ratio)));
                }

                // Many neighbors: power-distance pre-filter (O(d_max × 24ns))
                let candidate_klein = self.klein_points.get(candidate_id).map(|r| r.value().clone());
                let mut pd_ranked: Vec<(String, FixedPoint)> = Vec::with_capacity(n_neighbors + 1);

                if let Some(ref ck) = candidate_klein {
                    pd_ranked.push((candidate_id.to_string(), klein::power_distance(&query_klein.coords, ck)));
                }

                for hp in &cell.half_planes {
                    if let Some(nk) = self.klein_points.get(&hp.neighbor_id) {
                        pd_ranked.push((hp.neighbor_id.clone(), klein::power_distance(&query_klein.coords, nk.value())));
                    }
                }

                pd_ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                // Verify top-K with cheap hyperbolic_ratio
                let mut best_id = String::new();
                let mut best_ratio = FixedPoint::from_int(1);

                for (id, _pd) in pd_ranked.iter().take(VERIFY_K) {
                    if let Some(point) = self.point_map.get(id) {
                        let r = query_poincare.hyperbolic_ratio(point.value());
                        if r < best_ratio {
                            best_ratio = r;
                            best_id = id.clone();
                        }
                    }
                }

                if !best_id.is_empty() {
                    // Exact distance from the winning ratio (see above).
                    return Some((best_id, ratio_to_distance(best_ratio)));
                }
            } else {
                if let Some(candidate_point) = self.point_map.get(candidate_id) {
                    let dist = query_poincare.hyperbolic_distance(candidate_point.value());
                    return Some((candidate_id.to_string(), dist));
                }
            }
        }

        // Fallback: use VP-tree KNN (O(log n) per bucket)
        let results = self.hash_table.find_nearest_nodes(query_poincare, 1);
        if let Some((id, dist)) = results.into_iter().next() {
            return Some((id, dist));
        }

        None
    }

    /// Find the k nearest stored nodes to an arbitrary Poincaré disk point.
    ///
    /// Combines the Nielsen grid candidate + its neighbors with the VP-tree
    /// fallback to produce k results sorted by ascending hyperbolic distance.
    pub fn nearest_neighbor_point_k(&self, query_poincare: &HyperbolicPoint, k: usize) -> Vec<(String, FixedPoint)> {
        if self.klein_points.is_empty() || k == 0 {
            return Vec::new();
        }

        let query_klein = klein::poincare_to_klein(query_poincare);

        // Collect candidates from grid cell + neighbors (ratio-scored)
        let mut candidates: Vec<(String, FixedPoint)> = Vec::new();

        let grid_candidate = self.point_location.read().unwrap_or_else(|e| e.into_inner())
            .query(&query_klein.coords).map(|s| s.to_string());

        if let Some(candidate_id) = grid_candidate.as_deref() {
            // Add grid candidate itself
            if let Some(point) = self.point_map.get(candidate_id) {
                let dist = query_poincare.hyperbolic_distance(point.value());
                candidates.push((candidate_id.to_string(), dist));
            }

            // Add all its Voronoi neighbors
            if let Some(cell) = self.power_cells.get(candidate_id).map(|r| r.value().clone()) {
                for hp in &cell.half_planes {
                    if let Some(point) = self.point_map.get(&hp.neighbor_id) {
                        let dist = query_poincare.hyperbolic_distance(point.value());
                        candidates.push((hp.neighbor_id.clone(), dist));
                    }
                }
            }
        }

        // Merge with VP-tree results to cover gaps the grid may miss
        let vp_results = self.hash_table.find_nearest_nodes(query_poincare, k + candidates.len());
        for (id, dist) in vp_results {
            if !candidates.iter().any(|(cid, _)| cid == &id) {
                candidates.push((id, dist));
            }
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    /// Get the Klein point for a node by unique_id (cloned).
    pub fn get_klein_point(&self, unique_id: &str) -> Option<KleinPoint> {
        self.klein_points.get(unique_id).map(|r| r.value().clone())
    }

    /// Get the power cell for a node by unique_id (cloned).
    pub fn get_power_cell(&self, unique_id: &str) -> Option<PowerCell> {
        self.power_cells.get(unique_id).map(|r| r.value().clone())
    }

    /// Get the number of assigned tiles in the point location grid.
    pub fn grid_assigned_tile_count(&self) -> usize {
        self.point_location.read().unwrap_or_else(|e| e.into_inner()).assigned_tile_count()
    }

    // -----------------------------------------------------------------------
    // Semantic dimensional distance queries
    // -----------------------------------------------------------------------

    /// Compute Euclidean distance between two semantic coordinate vectors
    /// across a dimensional slice (specified dimension range).
    ///
    /// Each dimension is 16 bytes (i128 LE, Q64.64 fixed-point).
    /// Dimensions outside the vectors are treated as zero.
    ///
    /// Uses gMath's fused kernel: differences, squares, and the accumulator
    /// all live at the compute tier, so the sum cannot wrap the way a
    /// storage-tier Q64.64 accumulator would for large coordinates or many
    /// dimensions.
    pub fn semantic_distance(
        coords_a: &[u8],
        coords_b: &[u8],
        dim_range: &Range<usize>,
    ) -> FixedPoint {
        let a = Self::decode_semantic_slice(coords_a, dim_range);
        let b = Self::decode_semantic_slice(coords_b, dim_range);
        g_math::fixed_point::imperative::fused::euclidean_distance(&a, &b)
    }

    /// Decode a dimension slice of a raw Q64.64 coordinate vector.
    ///
    /// Dimensions beyond the end of `coords` decode as zero — short vectors
    /// are zero-extended, matching [`Self::semantic_distance`] semantics.
    pub fn decode_semantic_slice(coords: &[u8], dim_range: &Range<usize>) -> Vec<FixedPoint> {
        dim_range
            .clone()
            .map(|dim| {
                let start = dim * 16;
                let end = start + 16;
                if coords.len() >= end {
                    FixedPoint::from_raw(i128::from_le_bytes(
                        coords[start..end].try_into().unwrap(),
                    ))
                } else {
                    FixedPoint::from_int(0)
                }
            })
            .collect()
    }

    /// Find the k nearest nodes by Euclidean distance in semantic dimension space.
    ///
    /// `query_coords`: raw Q64.64 byte vector representing the query point.
    /// `k`: number of nearest neighbors to return.
    /// `dim_range`: which semantic dimensions to compare (the "dimensional slice").
    ///
    /// Returns `Vec<(key, distance)>` sorted ascending by `(distance, key)` —
    /// ties break deterministically by the user-visible node key, both for
    /// ordering and for which ties survive the k-boundary.
    ///
    /// Routing (`docs/SEMANTIC_INDEX.md`): stores below
    /// [`constants::SEMANTIC_INDEX_MIN_NODES`] use the brute-force scan;
    /// larger stores query a lazily built per-`dim_range` VP-tree, rebuilt
    /// when the semantic epoch has advanced (any coord write, insert, or
    /// delete). Warm-index queries are O(log n) expected on low-dimensional
    /// slices; the first query for a slice after a mutation pays the
    /// O(n log n) build. Results are identical to the scan path.
    pub fn nearest_semantic(
        &self,
        query_coords: &[u8],
        k: usize,
        dim_range: &Range<usize>,
    ) -> Vec<(String, FixedPoint)> {
        if k == 0 {
            return Vec::new();
        }

        if self.nodes.len() < constants::SEMANTIC_INDEX_MIN_NODES {
            return self.nearest_semantic_scan(query_coords, k, dim_range);
        }

        let query = Self::decode_semantic_slice(query_coords, dim_range);
        let index = self.semantic_index.get_or_build(dim_range, || {
            self.nodes
                .iter()
                .filter(|entry| !entry.value().semantic_coords().is_empty())
                .map(|entry| {
                    (
                        entry.value().metadata().key.clone(),
                        Self::decode_semantic_slice(entry.value().semantic_coords(), dim_range),
                    )
                })
                .collect()
        });
        index.tree.knn(&query, k, &EuclideanMetric)
    }

    /// Reference brute-force path for [`Self::nearest_semantic`]:
    /// O(n × d) scan over every node with semantic coordinates.
    ///
    /// Same ordering contract as the indexed path — ascending
    /// `(distance, key)`. Public so tests and benchmarks can compare
    /// the two paths directly; prefer `nearest_semantic`, which picks.
    pub fn nearest_semantic_scan(
        &self,
        query_coords: &[u8],
        k: usize,
        dim_range: &Range<usize>,
    ) -> Vec<(String, FixedPoint)> {
        if k == 0 {
            return Vec::new();
        }

        // Max-heap of size k on (distance, key): the peek is the current
        // worst candidate under the same total order the index uses, so
        // ties at the k-boundary break by key on both paths.
        let mut heap: BinaryHeap<(FixedPoint, String)> = BinaryHeap::new();

        for entry in self.nodes.iter() {
            let coords = entry.value().semantic_coords();

            // Skip nodes with no semantic coordinates
            if coords.is_empty() {
                continue;
            }

            let dist = Self::semantic_distance(query_coords, coords, dim_range);
            let key = entry.value().metadata().key.as_str();

            if heap.len() < k {
                heap.push((dist, key.to_string()));
            } else if let Some(worst) = heap.peek() {
                if (dist, key) < (worst.0, worst.1.as_str()) {
                    heap.pop();
                    heap.push((dist, key.to_string()));
                }
            }
        }

        // Extract and sort ascending by (distance, uid)
        let mut results: Vec<(String, FixedPoint)> = heap
            .into_iter()
            .map(|(dist, uid)| (uid, dist))
            .collect();
        results.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        results
    }

    /// Check if the network has a valid structure.
    ///
    /// Verifies structural invariants across all internal data structures:
    /// nodes, point_map, klein_points, power_cells, and the hash table.
    pub fn validate_network(&self) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

        let root_sig = self.root_signature.lock().unwrap_or_else(|e| e.into_inner());
        if root_sig.is_none() {
            return false;
        }

        let root_id = root_sig.as_ref().unwrap().unique_id();
        drop(root_sig);
        if !self.nodes.contains_key(&root_id) {
            return false;
        }

        // Child signatures reference existing nodes
        for entry in self.nodes.iter() {
            let node = entry.value();
            for child_sig in node.children() {
                if !self.nodes.contains_key(&child_sig.unique_id()) {
                    return false;
                }
            }
        }

        // Every node has a point_map entry
        for entry in self.nodes.iter() {
            if !self.point_map.contains_key(entry.key()) {
                return false;
            }
        }

        // point_map keys are a subset of nodes
        for entry in self.point_map.iter() {
            if !self.nodes.contains_key(entry.key()) {
                return false;
            }
        }

        // klein_points ↔ nodes sync
        for entry in self.klein_points.iter() {
            if !self.nodes.contains_key(entry.key()) {
                return false;
            }
        }

        // power_cells ↔ nodes sync
        for entry in self.power_cells.iter() {
            if !self.nodes.contains_key(entry.key()) {
                return false;
            }
        }

        // power_cell half-plane neighbors reference existing nodes
        for entry in self.power_cells.iter() {
            for hp in &entry.value().half_planes {
                if !self.nodes.contains_key(&hp.neighbor_id) {
                    return false;
                }
            }
        }

        // child_counts keys are a subset of nodes (no orphan entries)
        for entry in self.child_counts.iter() {
            if !self.nodes.contains_key(entry.key()) {
                return false;
            }
        }

        true
    }
}

impl Debug for HyperbolicTensorNetwork {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "HyperbolicTensorNetwork(nodes={}, dimension={})",
               self.nodes.len(),
               self.hash_table.poincare_disk().dimension())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressed_node() {
        let metadata = NodeMetadata::new("test".to_string(), None);
        let value = b"Node data".to_vec();

        let node = CompressedNode::new(metadata, value.clone());

        assert_eq!(node.metadata().key, "test");
        assert_eq!(node.value(), &value[..]);
        assert!(!node.has_children());
        assert_eq!(node.child_count(), 0);
    }

    #[test]
    fn test_tensor_network_creation() {
        let network = HyperbolicTensorNetwork::new(2, constants::default_tau());

        assert_eq!(network.node_count(), 0);
        assert!(network.root_node().is_none());
    }

    #[test]
    fn test_adding_nodes() {
        let network = HyperbolicTensorNetwork::new(2, constants::default_tau());

        let root_meta = NodeMetadata::new("/".to_string(), None);
        let root_sig = network.add_node(
            root_meta,
            b"Root node data".to_vec(),
            None,
            0
        ).unwrap();

        assert_eq!(network.node_count(), 1);
        assert!(network.root_node().is_some());

        let child_meta = NodeMetadata::new("/child".to_string(), None);
        let child_sig = network.add_node(
            child_meta,
            b"Child node data".to_vec(),
            Some(&root_sig),
            1
        ).unwrap();

        assert_eq!(network.node_count(), 2);

        let root_node = network.get_node_by_signature(&root_sig).unwrap();
        assert_eq!(root_node.child_count(), 1);
        assert_eq!(root_node.children()[0].unique_id(), child_sig.unique_id());
    }

    #[test]
    fn test_network_validation() {
        let network = HyperbolicTensorNetwork::new(2, constants::default_tau());

        assert!(!network.validate_network());

        let root_sig = network.add_node(
            NodeMetadata::new("/".to_string(), None),
            b"Root data".to_vec(),
            None,
            0
        ).unwrap();

        assert!(network.validate_network());

        network.add_node(
            NodeMetadata::new("/child1".to_string(), None),
            b"Child 1 data".to_vec(),
            Some(&root_sig),
            1
        ).unwrap();

        network.add_node(
            NodeMetadata::new("/child2".to_string(), None),
            b"Child 2 data".to_vec(),
            Some(&root_sig),
            1
        ).unwrap();

        assert!(network.validate_network());
    }

    #[test]
    fn test_point_map() {
        let network = HyperbolicTensorNetwork::new(2, constants::default_tau());

        let root_sig = network.add_node(
            NodeMetadata::new("/".to_string(), None),
            b"root".to_vec(),
            None,
            0
        ).unwrap();

        // Root should be at origin
        let root_point = network.get_point(&root_sig.unique_id()).unwrap();
        assert!(root_point.euclidean_norm() < constants::epsilon());

        let child_sig = network.add_node(
            NodeMetadata::new("/child".to_string(), None),
            b"child".to_vec(),
            Some(&root_sig),
            1
        ).unwrap();

        // Child should be away from origin
        let child_point = network.get_point(&child_sig.unique_id()).unwrap();
        assert!(child_point.euclidean_norm() > constants::epsilon());
    }

    #[test]
    fn test_spatial_descendants() {
        let network = HyperbolicTensorNetwork::new(2, constants::default_tau());

        let root_sig = network.add_node(
            NodeMetadata::new("/".to_string(), None),
            b"root".to_vec(),
            None,
            0
        ).unwrap();

        let child_sig = network.add_node(
            NodeMetadata::new("/child".to_string(), None),
            b"child".to_vec(),
            Some(&root_sig),
            1
        ).unwrap();

        let _grandchild_sig = network.add_node(
            NodeMetadata::new("/child/grandchild".to_string(), None),
            b"grandchild".to_vec(),
            Some(&child_sig),
            2
        ).unwrap();

        // Root's spatial descendants should include child and grandchild
        let descendants = network.find_descendants_spatial(&root_sig);
        assert!(descendants.len() >= 2,
            "Expected at least 2 descendants, got {}", descendants.len());
    }

    #[test]
    fn test_sarkar_child_distance() {
        // Children should be at exactly hyperbolic distance τ from parent
        let tau = constants::default_tau();
        let network = HyperbolicTensorNetwork::new(2, tau);

        let root_sig = network.add_node(
            NodeMetadata::new("/".to_string(), None),
            b"root".to_vec(),
            None,
            0
        ).unwrap();

        let child_sig = network.add_node(
            NodeMetadata::new("/child".to_string(), None),
            b"child".to_vec(),
            Some(&root_sig),
            1
        ).unwrap();

        let root_point = network.get_point(&root_sig.unique_id()).unwrap();
        let child_point = network.get_point(&child_sig.unique_id()).unwrap();

        let dist = root_point.hyperbolic_distance(&child_point);
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);
        assert!((dist - tau).abs() < tolerance,
            "Child should be at distance τ={} from parent, got {}", tau, dist);
    }

    #[test]
    fn test_sarkar_sibling_separation() {
        // Multiple children of the same parent should be at distinct positions
        let network = HyperbolicTensorNetwork::new(2, constants::default_tau());

        let root_sig = network.add_node(
            NodeMetadata::new("/".to_string(), None),
            b"root".to_vec(),
            None,
            0
        ).unwrap();

        let mut child_sigs = Vec::new();
        for i in 0..5 {
            let sig = network.add_node(
                NodeMetadata::new(format!("/child{}", i), None),
                format!("child{}", i).into_bytes(),
                Some(&root_sig),
                1
            ).unwrap();
            child_sigs.push(sig);
        }

        // All children should be at the same distance from root
        let tau = constants::default_tau();
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);
        let root_point = network.get_point(&root_sig.unique_id()).unwrap();

        for sig in &child_sigs {
            let child_point = network.get_point(&sig.unique_id()).unwrap();
            let dist = root_point.hyperbolic_distance(&child_point);
            assert!((dist - tau).abs() < tolerance,
                "All children should be at distance τ from parent");
        }

        // All siblings should be pairwise distinct (non-zero distance)
        for i in 0..child_sigs.len() {
            for j in (i+1)..child_sigs.len() {
                let pi = network.get_point(&child_sigs[i].unique_id()).unwrap();
                let pj = network.get_point(&child_sigs[j].unique_id()).unwrap();
                let dist = pi.hyperbolic_distance(&pj);
                assert!(dist > constants::epsilon(),
                    "Siblings {} and {} should be at distinct positions", i, j);
            }
        }
    }

    #[test]
    fn test_klein_points_created() {
        // Verify Klein points are created alongside Poincaré points
        let network = HyperbolicTensorNetwork::new(2, constants::default_tau());

        let root_sig = network.add_node(
            NodeMetadata::new("/".to_string(), None),
            b"root".to_vec(),
            None,
            0
        ).unwrap();

        let root_klein = network.get_klein_point(&root_sig.unique_id());
        assert!(root_klein.is_some(), "Root should have a Klein point");

        let rk = root_klein.unwrap();
        // Root at origin → Klein origin, weight = 1
        assert!(rk.coords[0].abs() < constants::epsilon());
        assert!(rk.weight > constants::half());

        let child_sig = network.add_node(
            NodeMetadata::new("/child".to_string(), None),
            b"child".to_vec(),
            Some(&root_sig),
            1
        ).unwrap();

        let child_klein = network.get_klein_point(&child_sig.unique_id());
        assert!(child_klein.is_some(), "Child should have a Klein point");
        assert!(child_klein.unwrap().coords.length() > constants::epsilon(),
            "Child Klein point should be away from origin");
    }

    #[test]
    fn test_power_cells_created() {
        let network = HyperbolicTensorNetwork::new(2, constants::default_tau());

        let root_sig = network.add_node(
            NodeMetadata::new("/".to_string(), None),
            b"root".to_vec(),
            None,
            0
        ).unwrap();

        // Root cell should have no half-planes (covers entire disk)
        let root_cell = network.get_power_cell(&root_sig.unique_id()).unwrap();
        assert!(root_cell.half_planes.is_empty(), "Root cell should have no constraints initially");

        let child_sig = network.add_node(
            NodeMetadata::new("/child".to_string(), None),
            b"child".to_vec(),
            Some(&root_sig),
            1
        ).unwrap();

        // After adding child: root should have 1 half-plane, child should have 1
        let root_cell = network.get_power_cell(&root_sig.unique_id()).unwrap();
        assert_eq!(root_cell.half_planes.len(), 1, "Root should have 1 half-plane after adding child");

        let child_cell = network.get_power_cell(&child_sig.unique_id()).unwrap();
        assert_eq!(child_cell.half_planes.len(), 1, "Child (leaf) should have 1 half-plane");
    }

    #[test]
    fn test_nearest_neighbor_point_finds_self() {
        // Query at a node's own position should return that node
        let network = HyperbolicTensorNetwork::new(2, constants::default_tau());

        let root_sig = network.add_node(
            NodeMetadata::new("/".to_string(), None),
            b"root".to_vec(),
            None,
            0
        ).unwrap();

        let child_sig = network.add_node(
            NodeMetadata::new("/child".to_string(), None),
            b"child".to_vec(),
            Some(&root_sig),
            1
        ).unwrap();

        // Query at child's position should find child
        let child_point = network.get_point(&child_sig.unique_id()).unwrap();
        let (nn_id, nn_dist) = network.nearest_neighbor_point(&child_point).unwrap();

        assert_eq!(nn_id, child_sig.unique_id(),
            "Nearest neighbor at child's position should be child itself");
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);
        assert!(nn_dist < tolerance,
            "Distance to self should be ~0, got {}", nn_dist);
    }

    #[test]
    fn test_grid_reflects_insertions() {
        let network = HyperbolicTensorNetwork::new(2, constants::default_tau());

        let root_sig = network.add_node(
            NodeMetadata::new("/".to_string(), None),
            b"root".to_vec(),
            None,
            0
        ).unwrap();

        // Grid should have assigned tiles after root
        assert!(network.grid_assigned_tile_count() > 0, "Grid should have tiles after root insert");

        let _child_sig = network.add_node(
            NodeMetadata::new("/child".to_string(), None),
            b"child".to_vec(),
            Some(&root_sig),
            1
        ).unwrap();

        // After child insert, grid should still have assigned tiles
        assert!(network.grid_assigned_tile_count() > 0, "Grid should still have tiles after child insert");
    }

    #[test]
    fn test_delete_cleans_up_klein_state() {
        let network = HyperbolicTensorNetwork::new(2, constants::default_tau());

        let root_sig = network.add_node(
            NodeMetadata::new("/".to_string(), None),
            b"root".to_vec(),
            None,
            0
        ).unwrap();

        let child_sig = network.add_node(
            NodeMetadata::new("/child".to_string(), None),
            b"child".to_vec(),
            Some(&root_sig),
            1
        ).unwrap();

        let child_id = child_sig.unique_id();

        // Verify Klein state exists
        assert!(network.get_klein_point(&child_id).is_some());
        assert!(network.get_power_cell(&child_id).is_some());

        // Delete child
        network.unregister_node(&child_id);

        // Klein state should be cleaned up
        assert!(network.get_klein_point(&child_id).is_none());
        assert!(network.get_power_cell(&child_id).is_none());

        // Parent's cell should no longer have the child's half-plane
        let root_cell = network.get_power_cell(&root_sig.unique_id()).unwrap();
        assert!(root_cell.half_planes.is_empty(),
            "Root cell should have no half-planes after child deletion");
    }

    #[test]
    fn test_semantic_distance_identical() {
        // Identical coordinates → distance = 0
        let coords = {
            let mut v = vec![0u8; 3 * 16]; // 3 dims
            let val = FixedPoint::from_f64(0.5).raw().to_le_bytes();
            v[0..16].copy_from_slice(&val);
            v[16..32].copy_from_slice(&val);
            v[32..48].copy_from_slice(&val);
            v
        };
        let dist = HyperbolicTensorNetwork::semantic_distance(&coords, &coords, &(0..3));
        assert!(dist < constants::epsilon(), "Distance to self should be ~0, got {}", dist);
    }

    #[test]
    fn test_semantic_distance_known_value() {
        // dim0: (1.0, 0.0), dim1: (0.0, 0.0) → distance = 1.0
        let mut a = vec![0u8; 2 * 16];
        let one = FixedPoint::from_f64(1.0).raw().to_le_bytes();
        a[0..16].copy_from_slice(&one);
        // dim1 stays zero

        let b = vec![0u8; 2 * 16]; // all zero

        let dist = HyperbolicTensorNetwork::semantic_distance(&a, &b, &(0..2));
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);
        assert!((dist - FixedPoint::from_int(1)).abs() < tolerance,
            "Distance should be 1.0, got {}", dist);
    }

    #[test]
    fn test_semantic_distance_dimensional_slice() {
        // Only compare dim 1, ignore dim 0
        let mut a = vec![0u8; 2 * 16];
        let one = FixedPoint::from_f64(1.0).raw().to_le_bytes();
        a[0..16].copy_from_slice(&one); // dim 0 = 1.0

        let b = vec![0u8; 2 * 16]; // all zero

        // Slice dim 1 only → both are 0.0 at dim 1 → distance = 0
        let dist = HyperbolicTensorNetwork::semantic_distance(&a, &b, &(1..2));
        assert!(dist < constants::epsilon(),
            "Slicing only dim 1 should give distance ~0, got {}", dist);

        // Slice dim 0 only → (1.0 vs 0.0) → distance = 1.0
        let dist_full = HyperbolicTensorNetwork::semantic_distance(&a, &b, &(0..1));
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);
        assert!((dist_full - FixedPoint::from_int(1)).abs() < tolerance,
            "Slicing dim 0 should give distance 1.0, got {}", dist_full);
    }

    #[test]
    fn test_nearest_semantic_basic() {
        let network = HyperbolicTensorNetwork::new(2, constants::default_tau());

        // Add 3 nodes with semantic coords in 2 dims
        let root_sig = network.add_node(
            NodeMetadata::new("/".to_string(), None),
            b"root".to_vec(), None, 0,
        ).unwrap();

        let a_sig = network.add_node(
            NodeMetadata::new("/a".to_string(), None),
            b"a".to_vec(), Some(&root_sig), 1,
        ).unwrap();

        let b_sig = network.add_node(
            NodeMetadata::new("/b".to_string(), None),
            b"b".to_vec(), Some(&root_sig), 1,
        ).unwrap();

        let c_sig = network.add_node(
            NodeMetadata::new("/c".to_string(), None),
            b"c".to_vec(), Some(&root_sig), 1,
        ).unwrap();

        // Set semantic coords: /a at (0.8, 0.1), /b at (0.7, 0.2), /c at (0.1, 0.9)
        let make_coords = |d0: f64, d1: f64| -> Vec<u8> {
            let mut v = vec![0u8; 2 * 16];
            v[0..16].copy_from_slice(&FixedPoint::from_f64(d0).raw().to_le_bytes());
            v[16..32].copy_from_slice(&FixedPoint::from_f64(d1).raw().to_le_bytes());
            v
        };

        network.set_node_semantic(&a_sig.unique_id(), make_coords(0.8, 0.1));
        network.set_node_semantic(&b_sig.unique_id(), make_coords(0.7, 0.2));
        network.set_node_semantic(&c_sig.unique_id(), make_coords(0.1, 0.9));

        // Query near /a's position → /b should be closest, /c farthest
        let query = make_coords(0.8, 0.1);
        let results = network.nearest_semantic(&query, 3, &(0..2));

        assert!(!results.is_empty());

        // First result should be /a (distance ~0)
        let first_dist = results[0].1;
        assert!(first_dist < FixedPoint::from_f64(0.01),
            "Nearest to (0.8,0.1) should be /a at ~0 distance, got {}", first_dist);

        // /c should be much farther than /b
        if results.len() >= 3 {
            assert!(results[2].1 > results[1].1,
                "Third result should be farther than second");
        }
    }
}
