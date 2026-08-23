//! store.rs - Simple, ergonomic wrapper around HTTStorage
//!
//! Provides a dead-simple API that hides all internal types (FixedPoint,
//! IntegrationError, geometric signatures, etc.) behind standard Rust types.
//!
//! # Quick Start
//!
//! ```
//! use horon_engine::Store;
//!
//! let store = Store::new();
//! store.put("/greeting", b"Hello, world!").unwrap();
//! let data = store.get("/greeting").unwrap();
//! assert_eq!(data, b"Hello, world!");
//! ```

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

use g_math::fixed_point::FixedPoint;

use super::config::HTTStorageConfig;
use super::constants::{OUTLIER_KNN, OUTLIER_MIN_POPULATION};
use super::metric_tree::{EuclideanMetric, MetricVpTree};
use super::storage::HTTStorage;
use super::tensor_network::HyperbolicTensorNetwork;
use super::tree_tensor::IntegrationError;

// ---------------------------------------------------------------------------
// StoreError
// ---------------------------------------------------------------------------

/// Simplified error type for Store operations.
#[derive(Debug)]
pub enum StoreError {
    /// The requested key was not found.
    NotFound(String),
    /// A key already exists (when an exclusive insert was expected).
    AlreadyExists(String),
    /// The operation was invalid (bad key, configuration error, etc.).
    InvalidOperation(String),
    /// An internal error occurred (lock poisoned, deserialization, etc.).
    Internal(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::NotFound(msg) => write!(f, "not found: {}", msg),
            StoreError::AlreadyExists(msg) => write!(f, "already exists: {}", msg),
            StoreError::InvalidOperation(msg) => write!(f, "invalid operation: {}", msg),
            StoreError::Internal(msg) => write!(f, "internal error: {}", msg),
        }
    }
}

impl std::error::Error for StoreError {}

/// A semantic outlier found by [`Store::find_outliers`]: a node whose
/// average distance to its nearest peers is anomalously large relative to
/// the population under the queried prefix.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticOutlier {
    /// The outlying node's key.
    pub key: String,
    /// Its average distance to its k nearest peers in the population.
    pub avg_knn_distance: FixedPoint,
    /// How many standard deviations that average sits above the population
    /// mean (always > the requested threshold).
    pub z_score: FixedPoint,
    /// The closest peer — "even its best match is this far away".
    pub nearest_peer: String,
    /// Distance to that closest peer.
    pub nearest_distance: FixedPoint,
}

impl From<IntegrationError> for StoreError {
    fn from(e: IntegrationError) -> Self {
        match e {
            IntegrationError::NotFound(msg) => StoreError::NotFound(msg),
            IntegrationError::AlreadyExists(msg) => StoreError::AlreadyExists(msg),
            IntegrationError::ValidationFailed(msg) | IntegrationError::ConfigurationError(msg) => {
                StoreError::InvalidOperation(msg)
            }
            IntegrationError::OperationFailed(msg)
            | IntegrationError::DeserializationError(msg)
            | IntegrationError::LockError(msg) => StoreError::Internal(msg),
        }
    }
}

// ---------------------------------------------------------------------------
// StoreConfig
// ---------------------------------------------------------------------------

/// Minimal configuration for a Store.
///
/// Power users who need control over dimension, grid resolution, or other
/// internals should use [`HTTStorage`] directly.
pub struct StoreConfig {
    capacity: usize,
    tau: FixedPoint,
}

impl StoreConfig {
    /// Create a new config with default capacity (10,000 nodes).
    pub fn new() -> Self {
        Self { capacity: 10_000, tau: FixedPoint::from_int(0) }
    }

    /// Set the expected number of in-memory nodes.
    ///
    /// **Advisory only**: this sizes internal caches; it does not enforce a
    /// limit. Inserts beyond `capacity` succeed and the store grows unbounded.
    pub fn capacity(mut self, n: usize) -> Self {
        self.capacity = n;
        self
    }

    /// Set the Sarkar embedding scale factor τ.
    ///
    /// Controls the hyperbolic distance between parent and child nodes.
    /// Default is 1.0. Smaller values allow deeper trees within the same
    /// Q64.64 precision budget; larger values give better angular separation
    /// between siblings.
    pub fn tau(mut self, t: FixedPoint) -> Self {
        self.tau = t;
        self
    }

    fn to_htt_config(&self) -> HTTStorageConfig {
        HTTStorageConfig {
            dimension: 4,
            max_memory_nodes: self.capacity,
            cache_size: std::cmp::max(self.capacity / 10, 10),
            storage_path: None,
            flush_interval: 60,
            optimize_on_shutdown: true,
            tau: self.tau,
        }
    }
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// QueryAdapter
// ---------------------------------------------------------------------------

/// Result from a query adapter.
#[derive(Debug, Clone)]
pub enum QueryResult {
    /// A single entry with key, data, and metadata.
    Entry {
        /// The entry's path key.
        key: String,
        /// The entry's raw data payload.
        data: Vec<u8>,
        /// The entry's key-value metadata.
        meta: HashMap<String, String>,
    },
    /// A count of matching entries.
    Count(usize),
    /// A list of matching keys.
    Keys(Vec<String>),
}

/// Trait for pluggable query adapters.
///
/// Adapters encapsulate query logic (e.g. search by metadata, semantic
/// similarity, path patterns) and call `Store` methods internally.
/// This is object-safe: `dyn QueryAdapter` works.
pub trait QueryAdapter: Send + Sync {
    /// Execute a query against the store.
    fn execute(&self, store: &Store, query: &str) -> Result<Vec<QueryResult>, StoreError>;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// A simple, ergonomic hierarchical data store backed by Hyperbolic Tree Tensors.
///
/// All keys are path-like strings (e.g. `"/users/alice"`). Parent directories
/// are created automatically. The root node `"/"` always exists.
///
/// # Examples
///
/// ```
/// use horon_engine::Store;
/// use g_math::fixed_point::FixedPoint;
///
/// let store = Store::new();
///
/// // Store and retrieve data
/// store.put("/config/db", b"postgres://localhost").unwrap();
/// assert_eq!(store.get("/config/db").unwrap(), b"postgres://localhost");
///
/// // Coordinates and distances are Q64.64 fixed point, never floats: the
/// // same query returns bit-identical results on any platform. Converting
/// // from a decimal literal is explicit, so the lossy step is visible at
/// // the call site rather than hidden inside the API.
/// let origin: Vec<FixedPoint> = (0..4).map(|_| FixedPoint::from_int(0)).collect();
/// let (path, distance) = store.nearest(&origin).unwrap();
/// ```
pub struct Store {
    inner: HTTStorage,
}

impl Store {
    /// Create a new Store with default settings (capacity: 10,000).
    pub fn new() -> Self {
        Self::with_config(StoreConfig::new())
    }

    /// Create a new Store with custom configuration.
    pub fn with_config(config: StoreConfig) -> Self {
        Self {
            inner: HTTStorage::new(config.to_htt_config()),
        }
    }

    /// Store data at a key (upsert — inserts or updates).
    ///
    /// Parent directories are created automatically.
    pub fn put(&self, key: &str, data: &[u8]) -> Result<(), StoreError> {
        self.inner.store(key, data, None)?;
        Ok(())
    }

    /// Store data without geometric embedding (data + semantic only).
    ///
    /// Much faster than `put()` for bulk loading — skips Sarkar embedding,
    /// VP-tree, and power diagram construction. Semantic queries
    /// (`nearest_semantic`, `neighbors_semantic`, `get_semantic`) work
    /// normally. Spatial queries (`nearest`, `neighbors`) will not find
    /// nodes loaded this way.
    pub fn put_data_only(&self, key: &str, data: &[u8]) -> Result<(), StoreError> {
        self.inner.store_data_only(key, data, None)?;
        Ok(())
    }

    /// Store data with an explicit child_index for deterministic Sarkar reconstruction.
    ///
    /// Used during snapshot replay: the stored child_index ensures the node gets
    /// the same geometric position regardless of replay order.
    pub fn put_positioned(&self, key: &str, data: &[u8], child_index: u32) -> Result<(), StoreError> {
        self.inner.store_positioned(key, data, None, child_index)?;
        Ok(())
    }

    /// Retrieve data by key.
    pub fn get(&self, key: &str) -> Result<Vec<u8>, StoreError> {
        Ok(self.inner.retrieve(key)?)
    }

    /// Remove a key and its data.
    pub fn remove(&self, key: &str) -> Result<(), StoreError> {
        self.inner.delete(key)?;
        Ok(())
    }

    /// Check if a key exists.
    pub fn exists(&self, key: &str) -> bool {
        self.inner.exists(key)
    }

    /// List immediate children of a path.
    ///
    /// Returns only direct children, not the full subtree.
    pub fn children(&self, path: &str) -> Result<Vec<String>, StoreError> {
        let htt = self.inner.shared_htt();
        let nodes = htt.list_children(path)?;
        Ok(nodes.into_iter().map(|n| n.metadata().key.clone()).collect())
    }

    /// List all keys under a prefix (full subtree).
    pub fn list(&self, prefix: &str) -> Result<Vec<String>, StoreError> {
        Ok(self.inner.list(prefix)?)
    }

    /// Set a metadata field on a key.
    pub fn set_meta(&self, key: &str, name: &str, value: &str) -> Result<(), StoreError> {
        self.inner.set_metadata(key, name, value)?;
        Ok(())
    }

    /// Get all metadata for a key.
    pub fn get_meta(&self, key: &str) -> Result<HashMap<String, String>, StoreError> {
        Ok(self.inner.get_metadata(key)?)
    }

    /// Set semantic coordinates on a key (raw Q64.64 bytes, 16 bytes per dimension).
    ///
    /// Each coordinate is 16 bytes (i128 LE). For example, 2 semantic dimensions
    /// requires 32 bytes. Use `FixedPoint::from_f64(value).raw().to_le_bytes()`
    /// to encode each coordinate.
    ///
    /// Returns `InvalidOperation` if `coords` is not a multiple of 16 bytes —
    /// a misaligned vector would otherwise have its trailing partial dimension
    /// silently ignored by distance computations.
    pub fn set_semantic(&self, key: &str, coords: Vec<u8>) -> Result<(), StoreError> {
        if coords.len() % 16 != 0 {
            return Err(StoreError::InvalidOperation(format!(
                "semantic coordinates must be a multiple of 16 bytes (one Q64.64 value per dimension); got {} bytes",
                coords.len()
            )));
        }
        self.inner.set_semantic(key, coords)?;
        Ok(())
    }

    /// Get semantic coordinates for a key (raw Q64.64 bytes).
    ///
    /// Returns empty Vec if no semantic coordinates have been set.
    pub fn get_semantic(&self, key: &str) -> Result<Vec<u8>, StoreError> {
        Ok(self.inner.get_semantic(key)?)
    }

    /// The deepest node this store can place, given its `tau`.
    ///
    /// A node sits at hyperbolic radius `depth × tau`, and Q64.64 stops
    /// representing coordinate differences faithfully past a radius of about
    /// 21 — beyond that every node saturates to the same distance and ranking
    /// becomes arbitrary. Placement past the limit is **refused**, so this is
    /// the number to design against rather than discover by failing.
    ///
    /// Raising `tau` for wider fan-out lowers this proportionally: the default
    /// `tau = 1.0` gives 21, `tau = 2.0` gives 10.
    ///
    /// Full metric fidelity degrades before the hard limit — the step error
    /// along a geodesic is 3.4e-8 at radius 16, 2.0e-4 at 20. See
    /// `docs/ARCHITECTURE.md`.
    pub fn max_depth(&self) -> u32 {
        let tau = self.inner.shared_htt().tensor_network().tau();
        (crate::constants::max_safe_radius() / tau).to_int().max(0) as u32
    }

    /// Find the nearest stored node to an arbitrary point in hyperbolic space.
    ///
    /// Coordinates are in the Poincare disk model (each component in `(-1, 1)`).
    /// The answer is the true nearest node: every surviving candidate is
    /// ranked by exact hyperbolic distance.
    ///
    /// **Cost**: the query's own cell, then rings outward until a proven lower
    /// bound rules out every cell not yet visited. Exact — nothing is capped,
    /// sampled or windowed. How many cells that takes depends on how the tree
    /// is shaped, so this is not O(1); measured figures are in BENCHMARKS.md.
    ///
    /// Returns `(key, hyperbolic_distance)`.
    pub fn nearest(&self, coords: &[FixedPoint]) -> Result<(String, FixedPoint), StoreError> {
        Ok(self.inner.nearest_neighbor_point(coords)?)
    }

    /// Find the k nearest stored nodes to an arbitrary point in hyperbolic space.
    ///
    /// Like `nearest()` but returns multiple candidates, enabling the caller
    /// to post-filter and still get results.
    ///
    /// **Cost**: as `nearest()` — the query's own cell, then rings outward until a proven lower
    /// bound rules out every cell not yet visited. Exact — nothing is capped,
    /// sampled or windowed. How many cells that takes depends on how the tree
    /// is shaped, so this is not O(1); measured figures are in BENCHMARKS.md.
    ///
    /// Returns `(key, hyperbolic_distance)` sorted by ascending distance.
    pub fn nearest_k(&self, coords: &[FixedPoint], k: usize) -> Result<Vec<(String, FixedPoint)>, StoreError> {
        Ok(self.inner.nearest_neighbor_point_k(coords, k)?)
    }

    /// Find the k nearest neighbors of an existing node.
    ///
    /// Returns keys sorted by ascending hyperbolic distance.
    /// The queried key itself is excluded from results.
    ///
    /// **Cost**: as `nearest_k()`, from the queried node's own position.
    pub fn neighbors(&self, path: &str, k: usize) -> Result<Vec<String>, StoreError> {
        Ok(self.inner.find_nearest(path, k)?)
    }

    // -----------------------------------------------------------------------
    // Semantic dimensional distance queries
    // -----------------------------------------------------------------------

    /// Reject a dimension slice that cannot carry information.
    ///
    /// `decode_semantic_slice` zero-extends: a dimension past the end of a
    /// stored vector reads as zero. That is deliberate and lets a short vector
    /// compare against a long one. Two cases abuse it:
    ///
    /// - an **empty** range compares nothing, so every pair scores 0;
    /// - a range starting past the end of the *query's own* coordinates means
    ///   the query contributes zeros across the whole slice.
    ///
    /// Either way every candidate ties at distance zero, the deterministic
    /// key tie-break picks k of them, and the caller receives a confident,
    /// reproducible, information-free answer. That is the one thing this
    /// engine refuses to do, so it is an error instead.
    ///
    /// A range that merely *extends past* the data is still fine — that is
    /// zero-extension working as designed.
    fn check_slice(dim_range: &Range<usize>, query_dims: usize, what: &str) -> Result<(), StoreError> {
        if dim_range.start >= dim_range.end {
            return Err(StoreError::InvalidOperation(format!(
                "dimension range {}..{} is empty, so every node would tie at distance \
                 zero and the ranking would be arbitrary",
                dim_range.start, dim_range.end
            )));
        }
        if dim_range.start >= query_dims {
            return Err(StoreError::InvalidOperation(format!(
                "dimension range {}..{} starts past the {} of the {}, which has {} \
                 dimension(s); every value compared would be a zero-extension and the \
                 ranking would be arbitrary",
                dim_range.start, dim_range.end, "end", what, query_dims
            )));
        }
        Ok(())
    }

    /// Find the k nearest nodes by Euclidean distance across a dimensional slice.
    ///
    /// A dimensional slice selects which semantic dimensions to compare.
    /// For example, `16..33` compares only category preference axes,
    /// ignoring operational dimensions. Different slices answer different
    /// questions from the same data.
    ///
    /// `query_coords`: raw Q64.64 bytes (16 bytes per dimension).
    /// `k`: number of nearest neighbors to return.
    /// `dim_range`: which dimensions to include in the distance calculation.
    ///
    /// Returns `(key, distance)` sorted ascending by `(distance, key)` —
    /// ties break deterministically.
    /// Returns `InvalidOperation` if `query_coords` is not a multiple of 16 bytes.
    ///
    /// **Complexity** (`docs/SEMANTIC_INDEX.md`): stores below
    /// `SEMANTIC_INDEX_MIN_NODES` use a brute-force O(n × d) scan. Larger
    /// stores use a lazily built per-`dim_range` VP-tree: O(log n) expected
    /// per warm query on low-dimensional slices; the first query for a slice
    /// after any semantic write pays an O(n log n) rebuild. Results are
    /// identical on both paths.
    pub fn nearest_semantic(
        &self,
        query_coords: &[u8],
        k: usize,
        dim_range: Range<usize>,
    ) -> Result<Vec<(String, FixedPoint)>, StoreError> {
        if query_coords.len() % 16 != 0 {
            return Err(StoreError::InvalidOperation(format!(
                "semantic query coordinates must be a multiple of 16 bytes; got {} bytes",
                query_coords.len()
            )));
        }
        Self::check_slice(&dim_range, query_coords.len() / 16, "query vector")?;
        let results = self.inner.nearest_semantic(query_coords, k, &dim_range)?;
        Ok(results)
    }

    /// Find the k nearest nodes to an existing node by semantic dimensional distance.
    ///
    /// Reads the node's semantic coordinates and finds the closest other nodes
    /// in the specified dimensional slice. The queried node is excluded.
    ///
    /// **Complexity**: same routing as [`Store::nearest_semantic`] (indexed
    /// above the node floor, brute-force below).
    ///
    /// Returns keys sorted by ascending semantic distance.
    pub fn neighbors_semantic(
        &self,
        path: &str,
        k: usize,
        dim_range: Range<usize>,
    ) -> Result<Vec<(String, FixedPoint)>, StoreError> {
        let anchor_dims = self.get_semantic(path).map(|c| c.len() / 16).unwrap_or(0);
        Self::check_slice(&dim_range, anchor_dims, "anchor node")?;
        let results = self.inner.neighbors_semantic(path, k, &dim_range)?;
        Ok(results)
    }

    /// Find the k stored nodes most similar to an existing node across a
    /// dimensional slice — "what's like this one?".
    ///
    /// This is [`Store::neighbors_semantic`] under a task-shaped name: it
    /// reads the node's semantic coordinates and returns the k nearest other
    /// nodes by Euclidean distance over `dim_range`, sorted ascending by
    /// `(distance, key)`. Same routing and cost as `nearest_semantic`.
    pub fn find_similar(
        &self,
        key: &str,
        k: usize,
        dim_range: Range<usize>,
    ) -> Result<Vec<(String, FixedPoint)>, StoreError> {
        self.neighbors_semantic(key, k, dim_range)
    }

    /// Find semantic outliers among the nodes under a key prefix:
    /// nodes whose average distance to their nearest peers is anomalously
    /// large relative to the population.
    ///
    /// For each node under `prefix` that has semantic coordinates, computes
    /// the average distance to its `OUTLIER_KNN` (10, capped at
    /// population−1) nearest peers **within the same population** over
    /// `dim_range`, then flags nodes whose average exceeds the population
    /// mean by more than `z_threshold` standard deviations. Returns outliers
    /// sorted by descending z-score (ties by key). This is the
    /// "room 403 rates unlike its floor-mates" / "course far from every
    /// peer" query as one call.
    ///
    /// Statistics are computed strictly within the prefix population — nodes
    /// outside `prefix` (or without coordinates) neither appear nor skew the
    /// baseline. Populations below `OUTLIER_MIN_POPULATION` (5) return no
    /// outliers: z-scores over a handful of nodes are noise, not findings.
    ///
    /// **Complexity**: builds a dedicated VP-tree over the population
    /// (O(m log m) distance evaluations) plus one k-NN query per node —
    /// ~seconds at 10k nodes, versus the O(m²) pairwise scan this replaces.
    /// Deterministic: identical stores produce identical results.
    ///
    /// Returns `InvalidOperation` if `z_threshold` is not a finite positive
    /// number.
    pub fn find_outliers(
        &self,
        prefix: &str,
        z_threshold: FixedPoint,
        dim_range: Range<usize>,
    ) -> Result<Vec<SemanticOutlier>, StoreError> {
        if z_threshold <= FixedPoint::from_int(0) {
            return Err(StoreError::InvalidOperation(format!(
                "z_threshold must be a positive number; got {}",
                z_threshold.to_f64()
            )));
        }
        // No single query vector here, so only the empty-range case is
        // checkable without scanning the population.
        if dim_range.start >= dim_range.end {
            return Err(StoreError::InvalidOperation(format!(
                "dimension range {}..{} is empty, so every node would tie at distance \
                 zero and no node could be an outlier",
                dim_range.start, dim_range.end
            )));
        }

        // Population: prefix members with semantic coordinates, key-sorted
        // (deterministic accumulation order for the statistics below).
        let mut keys = self.list(prefix)?;
        keys.sort();
        let entries: Vec<(String, Vec<FixedPoint>)> = keys
            .into_iter()
            .filter_map(|key| {
                let coords = self.inner.get_semantic(&key).ok()?;
                if coords.is_empty() {
                    return None;
                }
                Some((
                    key,
                    HyperbolicTensorNetwork::decode_semantic_slice(&coords, &dim_range),
                ))
            })
            .collect();

        if entries.len() < OUTLIER_MIN_POPULATION {
            return Ok(Vec::new());
        }

        // Population-local index: outlier statistics must not be skewed by
        // nodes outside the prefix, so the store-wide slice cache is not
        // reusable here.
        let tree = MetricVpTree::build(entries.clone(), &EuclideanMetric);
        let k = OUTLIER_KNN.min(entries.len() - 1);

        // Per-node average k-NN distance (querying k+1 to skip self).
        let zero = FixedPoint::from_int(0);
        let mut scored: Vec<(String, FixedPoint, String, FixedPoint)> = entries
            .iter()
            .map(|(key, point)| {
                let peers: Vec<(String, FixedPoint)> = tree
                    .knn(point, k + 1, &EuclideanMetric)
                    .into_iter()
                    .filter(|(id, _)| id != key)
                    .take(k)
                    .collect();
                let sum = peers.iter().fold(zero, |acc, (_, d)| acc + *d);
                let avg = sum / FixedPoint::from_int(k as i32);
                let (nearest_peer, nearest_distance) = peers[0].clone();
                (key.clone(), avg, nearest_peer, nearest_distance)
            })
            .collect();

        // Population statistics in fixed point: these feed the z-score that
        // decides which nodes are reported, so float arithmetic here would
        // put a nondeterministic step inside a query result.
        let n = FixedPoint::from_int(scored.len() as i32);
        let mean = scored.iter().fold(zero, |acc, (_, avg, _, _)| acc + *avg) / n;
        let variance = scored
            .iter()
            .fold(zero, |acc, (_, avg, _, _)| {
                let d = *avg - mean;
                acc + d * d
            })
            / n;
        let stdev = variance.sqrt();
        if stdev <= FixedPoint::from_raw(1) {
            return Ok(Vec::new()); // uniform population — no outliers
        }

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        Ok(scored
            .into_iter()
            .filter_map(|(key, avg, nearest_peer, nearest_distance)| {
                let z_score = (avg - mean) / stdev;
                (z_score > z_threshold).then_some(SemanticOutlier {
                    key,
                    avg_knn_distance: avg,
                    z_score,
                    nearest_peer,
                    nearest_distance,
                })
            })
            .collect())
    }

    /// Upgrade a data-only key (inserted via [`Store::put_data_only`]) to a
    /// full geometric embedding, in place.
    ///
    /// Missing ancestors are embedded first; the node's key, value,
    /// metadata, and semantic coordinates are preserved. After this call the
    /// key participates in spatial queries (`nearest`, `neighbors`,
    /// `find_within`) and has a [`Store::position`].
    ///
    /// Returns whether this call performed the upgrade (`false` = the key
    /// was already embedded; idempotent). Positions are derived state, not
    /// persisted: deterministic for a fixed operation sequence, but a
    /// lazily-loaded store must re-embed after reopening.
    pub fn embed_existing(&self, key: &str) -> Result<bool, StoreError> {
        Ok(self.inner.embed_existing(key)?)
    }

    /// Embed the prefix node (when it exists) and every data-only key under
    /// it (convenience). Parents embed before children (sorted order +
    /// ancestor recursion). Returns how many keys this call upgraded.
    pub fn embed_all(&self, prefix: &str) -> Result<usize, StoreError> {
        let mut upgraded = 0;
        if self.exists(prefix) && self.embed_existing(prefix)? {
            upgraded += 1;
        }
        let mut keys = self.list(prefix)?;
        keys.sort();
        for key in keys {
            if self.embed_existing(&key)? {
                upgraded += 1;
            }
        }
        Ok(upgraded)
    }

    /// The hyperbolic (Poincaré) position of a stored key.
    ///
    /// Errors for unknown keys and for data-only nodes (no embedding).
    pub fn position(&self, key: &str) -> Result<Vec<FixedPoint>, StoreError> {
        let point = self.inner.position(key)?;
        Ok(point.coords().iter().copied().collect())
    }

    /// The exact fixed-point position — crate-internal (semantic disk
    /// derives barycenters from it without an f64 round-trip).
    pub(crate) fn position_fixed(
        &self,
        key: &str,
    ) -> Result<crate::hyperbolic_geometry::HyperbolicPoint, StoreError> {
        Ok(self.inner.position(key)?)
    }

    /// Monotone counter of semantic-relevant mutations (coordinate writes,
    /// inserts, deletes). External caches over semantic state — e.g. the semantic disk
    /// [`crate::semantic_disk::SemanticDisk`] — tag their builds with it and
    /// rebuild when it has advanced, exactly like the internal index cache.
    pub fn semantic_epoch(&self) -> u64 {
        self.inner.semantic_epoch()
    }

    /// Compute the Euclidean distance between two raw semantic coordinate vectors
    /// across a dimensional slice.
    ///
    /// Utility method for computing distances without querying the store.
    pub fn semantic_distance(
        coords_a: &[u8],
        coords_b: &[u8],
        dim_range: Range<usize>,
    ) -> FixedPoint {
        HyperbolicTensorNetwork::semantic_distance(coords_a, coords_b, &dim_range)
    }

    /// Find all nodes within a hyperbolic distance of an existing node.
    ///
    /// **Cost**: ring expansion bounded by `radius` rather than by a running
    /// k-th distance, so it grows with the radius and the result size. A
    /// radius too large to express as a `cosh` prunes nothing and sweeps the
    /// whole index — slow, but still exact.
    pub fn find_within(&self, path: &str, radius: FixedPoint) -> Result<Vec<String>, StoreError> {
        Ok(self.inner.find_in_radius(path, radius)?)
    }

    /// Execute a query using a pluggable adapter.
    ///
    /// The adapter receives a reference to this store and the query string,
    /// and returns results by calling `get`, `list`, `neighbors`, etc.
    pub fn query(&self, adapter: &dyn QueryAdapter, query: &str) -> Result<Vec<QueryResult>, StoreError> {
        adapter.execute(self, query)
    }

    /// Number of stored entries (excludes the root node).
    pub fn len(&self) -> usize {
        self.inner.node_count().saturating_sub(1) // exclude root
    }

    /// Returns `true` if the store contains no user data.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Access the underlying `HTTStorage` for advanced operations.
    pub fn inner(&self) -> &HTTStorage {
        &self.inner
    }

    /// Mutably access the underlying `HTTStorage` for advanced operations.
    #[deprecated(note = "All HTTStorage methods now take &self; use inner() instead")]
    pub fn inner_mut(&mut self) -> &mut HTTStorage {
        &mut self.inner
    }
}

impl Default for Store {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {

/// Exact fixed-point coordinates from decimal literals.
fn fp(vals: &[f64]) -> Vec<g_math::fixed_point::FixedPoint> {
    vals.iter().map(|&v| g_math::fixed_point::FixedPoint::from_f64(v)).collect()
}

    use super::*;

    #[test]
    fn test_new_store_is_empty() {
        let store = Store::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_put_get_roundtrip() {
        let store = Store::new();
        store.put("/hello", b"world").unwrap();
        assert_eq!(store.get("/hello").unwrap(), b"world");
    }

    #[test]
    fn test_upsert() {
        let store = Store::new();
        store.put("/key", b"v1").unwrap();
        store.put("/key", b"v2").unwrap();
        assert_eq!(store.get("/key").unwrap(), b"v2");
    }

    #[test]
    fn test_remove() {
        let store = Store::new();
        store.put("/tmp", b"data").unwrap();
        assert!(store.exists("/tmp"));
        store.remove("/tmp").unwrap();
        assert!(!store.exists("/tmp"));
    }

    #[test]
    fn test_exists() {
        let store = Store::new();
        assert!(!store.exists("/nope"));
        store.put("/yes", b"").unwrap();
        assert!(store.exists("/yes"));
    }

    #[test]
    fn test_children() {
        let store = Store::new();
        store.put("/a/b", b"1").unwrap();
        store.put("/a/c", b"2").unwrap();
        store.put("/a/c/d", b"3").unwrap();

        let kids = store.children("/a").unwrap();
        assert!(kids.contains(&"/a/b".to_string()));
        assert!(kids.contains(&"/a/c".to_string()));
        // /a/c/d is a grandchild, not a direct child
        assert!(!kids.contains(&"/a/c/d".to_string()));
    }

    #[test]
    fn test_list() {
        let store = Store::new();
        store.put("/x/y", b"1").unwrap();
        store.put("/x/z", b"2").unwrap();

        let all = store.list("/x").unwrap();
        assert!(all.contains(&"/x/y".to_string()));
        assert!(all.contains(&"/x/z".to_string()));
    }

    #[test]
    fn test_metadata() {
        let store = Store::new();
        store.put("/doc", b"content").unwrap();
        store.set_meta("/doc", "author", "alice").unwrap();

        let meta = store.get_meta("/doc").unwrap();
        assert_eq!(meta.get("author"), Some(&"alice".to_string()));
    }

    #[test]
    fn test_nearest() {
        let store = Store::new();
        store.put("/a", b"a").unwrap();
        store.put("/b", b"b").unwrap();

        let (path, dist) = store.nearest(&fp(&[0.0, 0.0, 0.0, 0.0])).unwrap();
        // Origin query should find root "/"
        assert_eq!(path, "/");
        assert!(dist.to_f64() < 0.1);
    }

    #[test]
    fn test_neighbors() {
        let store = Store::new();
        store.put("/a", b"a").unwrap();
        store.put("/b", b"b").unwrap();
        store.put("/c", b"c").unwrap();

        let nbrs = store.neighbors("/a", 2).unwrap();
        assert!(!nbrs.is_empty());
        assert!(nbrs.len() <= 2);
        assert!(!nbrs.contains(&"/a".to_string()));
    }

    #[test]
    fn test_find_within() {
        let store = Store::new();
        store.put("/x", b"x").unwrap();
        store.put("/y", b"y").unwrap();

        let results = store.find_within("/x", g_math::fixed_point::FixedPoint::from_f64(10.0)).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_error_not_found() {
        let store = Store::new();
        let err = store.get("/missing").unwrap_err();
        assert!(matches!(err, StoreError::NotFound(_)));
    }

    #[test]
    fn test_len_tracking() {
        let store = Store::new();
        assert_eq!(store.len(), 0);

        store.put("/one", b"1").unwrap();
        assert_eq!(store.len(), 1);

        store.put("/two", b"2").unwrap();
        assert_eq!(store.len(), 2);

        store.remove("/one").unwrap();
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_inner_escape_hatch() {
        let store = Store::new();
        store.put("/test", b"data").unwrap();

        // Read access via inner()
        assert!(store.inner().exists("/test"));

        // Write access via inner() — all methods are now &self
        store.inner().store("/via_inner", b"inner", None).unwrap();
        assert!(store.exists("/via_inner"));
    }

    #[test]
    fn test_with_config() {
        let config = StoreConfig::new().capacity(500);
        let store = Store::with_config(config);
        assert!(store.is_empty());
    }

    #[test]
    fn test_tau_config() {
        let store = Store::with_config(StoreConfig::new().capacity(1000).tau(FixedPoint::from_f64(0.8)));
        store.put("/a", b"a").unwrap();
        store.put("/b", b"b").unwrap();
        store.put("/a/child", b"c").unwrap();
        assert_eq!(store.len(), 3);

        // NN should still work
        let (path, dist) = store.nearest(&fp(&[0.0, 0.0, 0.0, 0.0])).unwrap();
        assert_eq!(path, "/");
        assert!(dist.to_f64() < 0.1);
    }

    #[test]
    fn test_tau_deep_tree() {
        // A smaller tau buys depth, because a node sits at radius depth × tau
        // and the usable radius is fixed by the arithmetic, not by tau.
        //
        // This test previously built 40 levels at tau=0.8 — radius 32, well
        // past the point where the distance kernel saturates — and asserted
        // only that the key existed. It passed while every distance among
        // those nodes was meaningless. Placement past the limit is now
        // refused, so the honest assertions are: the limit scales with tau,
        // depth up to it works, and beyond it fails loudly.
        let store = Store::with_config(StoreConfig::new().tau(FixedPoint::from_f64(0.8)));
        let limit = store.max_depth();
        assert_eq!(limit, 26, "21 / 0.8 = 26 levels");
        assert!(
            limit > Store::new().max_depth(),
            "a smaller tau must allow more depth than the default"
        );

        let mut path = String::new();
        for i in 0..limit {
            path = format!("{}/n{}", path, i);
            store.put(&path, b"x").unwrap_or_else(|e| {
                panic!("level {i} is inside the limit of {limit} but was refused: {e:?}")
            });
        }
        assert!(store.exists(&path));

        // Past the limit: refused, not silently placed in the saturated band.
        let mut over = path.clone();
        let mut refused = false;
        for i in limit..(limit + 6) {
            over = format!("{}/n{}", over, i);
            if store.put(&over, b"x").is_err() {
                refused = true;
                break;
            }
        }
        assert!(refused, "placement past the depth limit must fail, not saturate");

        let (nn, _) = store.nearest(&fp(&[0.0, 0.0, 0.0, 0.0])).unwrap();
        assert!(store.exists(&nn));
    }

    #[test]
    fn test_query_adapter() {
        // Simple adapter that lists children of a path
        struct ChildrenAdapter;
        impl QueryAdapter for ChildrenAdapter {
            fn execute(&self, store: &Store, query: &str) -> Result<Vec<QueryResult>, StoreError> {
                let children = store.children(query)?;
                Ok(vec![QueryResult::Keys(children)])
            }
        }

        let store = Store::new();
        store.put("/a/b", b"1").unwrap();
        store.put("/a/c", b"2").unwrap();

        let results = store.query(&ChildrenAdapter, "/a").unwrap();
        assert_eq!(results.len(), 1);
        match &results[0] {
            QueryResult::Keys(keys) => {
                assert!(keys.contains(&"/a/b".to_string()));
                assert!(keys.contains(&"/a/c".to_string()));
            }
            _ => panic!("Expected Keys result"),
        }
    }

    #[test]
    fn test_query_adapter_object_safe() {
        // Verify QueryAdapter is object-safe (dyn QueryAdapter works)
        struct CountAdapter;
        impl QueryAdapter for CountAdapter {
            fn execute(&self, store: &Store, query: &str) -> Result<Vec<QueryResult>, StoreError> {
                let keys = store.list(query)?;
                Ok(vec![QueryResult::Count(keys.len())])
            }
        }

        let adapter: Box<dyn QueryAdapter> = Box::new(CountAdapter);
        let store = Store::new();
        store.put("/x", b"x").unwrap();
        store.put("/y", b"y").unwrap();

        let results = store.query(&*adapter, "/").unwrap();
        match &results[0] {
            QueryResult::Count(n) => assert_eq!(*n, 2),
            _ => panic!("Expected Count result"),
        }
    }

    #[test]
    fn test_nearest_semantic() {
        use g_math::fixed_point::FixedPoint;

        let store = Store::new();
        store.put("/courses/trauma/emdr", b"EMDR").unwrap();
        store.put("/courses/trauma/ptss", b"PTSS").unwrap();
        store.put("/courses/cgt/basis", b"CGT").unwrap();

        // Encode helper: 2 dims (dim 0 = trauma, dim 1 = cgt)
        let coords = |d0: f64, d1: f64| -> Vec<u8> {
            let mut v = vec![0u8; 2 * 16];
            v[0..16].copy_from_slice(&FixedPoint::from_f64(d0).raw().to_le_bytes());
            v[16..32].copy_from_slice(&FixedPoint::from_f64(d1).raw().to_le_bytes());
            v
        };

        store.set_semantic("/courses/trauma/emdr", coords(0.9, 0.1)).unwrap();
        store.set_semantic("/courses/trauma/ptss", coords(0.8, 0.2)).unwrap();
        store.set_semantic("/courses/cgt/basis", coords(0.1, 0.9)).unwrap();

        // Query: student with strong trauma preference
        let query = coords(0.85, 0.15);
        let results = store.nearest_semantic(&query, 3, 0..2).unwrap();

        assert_eq!(results.len(), 3);
        // EMDR and PTSS should be closer than CGT
        let paths: Vec<&str> = results.iter().map(|(p, _)| p.as_str()).collect();
        assert!(paths[0].contains("trauma"), "Nearest should be a trauma course, got {}", paths[0]);
        assert!(paths[2].contains("cgt"), "Farthest should be CGT, got {}", paths[2]);
    }

    #[test]
    fn test_neighbors_semantic() {
        use g_math::fixed_point::FixedPoint;

        let store = Store::new();
        store.put("/a", b"a").unwrap();
        store.put("/b", b"b").unwrap();
        store.put("/c", b"c").unwrap();

        let coords = |v: f64| -> Vec<u8> {
            let mut buf = vec![0u8; 16];
            buf[0..16].copy_from_slice(&FixedPoint::from_f64(v).raw().to_le_bytes());
            buf
        };

        store.set_semantic("/a", coords(0.1)).unwrap();
        store.set_semantic("/b", coords(0.2)).unwrap();
        store.set_semantic("/c", coords(0.9)).unwrap();

        // Neighbors of /a: /b should be closest, /c farthest
        let results = store.neighbors_semantic("/a", 2, 0..1).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "/b", "Nearest semantic neighbor of /a should be /b");
        assert_eq!(results[1].0, "/c", "Second neighbor of /a should be /c");

        // Self (/a) should not appear in results
        let paths: Vec<&str> = results.iter().map(|(p, _)| p.as_str()).collect();
        assert!(!paths.contains(&"/a"), "Self should be excluded from neighbors_semantic");
    }

    #[test]
    fn test_semantic_distance_utility() {
        use g_math::fixed_point::FixedPoint;

        let coords = |d0: f64, d1: f64| -> Vec<u8> {
            let mut v = vec![0u8; 2 * 16];
            v[0..16].copy_from_slice(&FixedPoint::from_f64(d0).raw().to_le_bytes());
            v[16..32].copy_from_slice(&FixedPoint::from_f64(d1).raw().to_le_bytes());
            v
        };

        let a = coords(0.0, 0.0);
        let b = coords(0.3, 0.4);

        // Euclidean distance should be 0.5 (3-4-5 triangle)
        let dist = Store::semantic_distance(&a, &b, 0..2);
        assert!((dist.to_f64() - 0.5).abs() < 0.01,
            "Distance (0,0)→(0.3,0.4) should be 0.5, got {}", dist.to_f64());
    }
}
