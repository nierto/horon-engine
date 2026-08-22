//! # horon-engine
//!
//! horon-engine is a Rust implementation of the Hyperbolic Tree Tensor data structure:
//! hierarchical data embedded in the Poincaré disk so that tree structure becomes
//! spatial proximity. Path lookups are hash-map access; spatial queries are
//! answered exactly by a computed cell index.
//!
//! ## Core Principles
//!
//! 1. **Hyperbolic Geometry**: Maps hierarchical data into hyperbolic space using the Poincaré disk model
//! 2. **Spatial Indexing**: a computed cell index — radial bands × angular sectors —
//!    walked outward until a proven lower bound rules out every unvisited cell
//! 3. **Exact by default**: no candidate cap, no window, no count-based stopping
//!    rule. Degradation is toward slow, never toward wrong.
//!
//! ## Key Features
//!
//! - **Honest costs**: `get`/`exists` are hash-map access. Spatial queries are not
//!   O(1) — they cost as many cells as the bound needs to rule out, which depends on
//!   tree shape; see BENCHMARKS.md for measured figures. Semantic queries use lazy
//!   per-slice VP-trees above a node floor and a linear scan below — see
//!   `Store::nearest_semantic` and `docs/SEMANTIC_INDEX.md`.
//! - **Spatial Queries**: Find nearest neighbors and range queries in hyperbolic space
//! - **Architecture**: `docs/ARCHITECTURE.md` — the three coordinate systems and
//!   which query serves each. Read it first; the distinction is easy to get backwards.
//! - **Mathematically Grounded**: Sarkar embedding (see PROOF.md). PROOF.md's Delaunay
//!   guarantee is conditional on `tau >= -log(tan(pi/(2*d_max)))` and the default
//!   `tau = 1.0` satisfies it only up to `d_max ~= 4.5` (see `StoreConfig::tau`) — but
//!   since 0.6.0 no query path depends on it. A violated bound costs spacing quality,
//!   not correctness. See `docs/GEOMETRY_TRACK.md`.
//! - **Deterministic Results**: Q64.64 fixed-point arithmetic for bit-identical results across platforms
//! - **Extensible**: Modular architecture with pluggable components and extension system
//!
//! ## Basic Usage
//!
//! ```
//! use horon_engine::{HTTStorage, HTTStorageConfig};
//!
//! // Create storage configuration
//! let config = HTTStorageConfig::default();
//!
//! // Create HTT storage
//! let storage = HTTStorage::new(config);
//!
//! // Store and retrieve data
//! storage.store("/example/path", b"Hello, HTT!", None).unwrap();
//! let data = storage.retrieve("/example/path").unwrap();
//! assert_eq!(data, b"Hello, HTT!");
//! ```

#![warn(missing_docs)]

// Core modules
pub mod error;
pub mod registry;
pub mod metrics;
pub mod config;
pub mod hash_table;
pub mod hyperbolic_geometry;
pub mod metric_tree;
pub mod cell_index;
// A `spatial_index` module (a MetricVpTree wrapper) was held in reserve while
// the cell index was being proven. It is nothing's dependency and never
// shipped, so it is not published; it sits in the untracked `archive/` after a
// full per-item inspection (248 lines, 5 tests, zero references).
pub mod semantic_disk;
pub mod semantic_index;
pub mod tensor_network;
pub mod tree_tensor;
pub mod storage;
pub mod extension;
pub mod utils;
pub mod constants;
pub mod concurrency;
pub mod klein;
pub mod store;
pub mod init;

// Re-export key types
pub use error::{HTTError, HTTResult};
pub use registry::{ComponentRegistry, HTTComponentRegistry, RegistryError};
pub use metrics::{MetricsProvider, SimpleMetrics};
pub use config::HTTStorageConfig;
pub use tree_tensor::HTTConfig;
pub use hash_table::GeometricSignature;
pub use hyperbolic_geometry::{PoincareDisk, HyperbolicPoint, distance_to_ratio};
pub use tensor_network::{HyperbolicTensorNetwork, CompressedNode, NodeMetadata};
pub use tree_tensor::HyperbolicTreeTensor;
pub use storage::HTTStorage;
pub use extension::{HTTExtension, HTTStorageProvider, ExtensionRegistry};
pub use klein::{KleinPoint, poincare_to_klein, klein_to_poincare, power_distance};
pub use store::{Store, StoreConfig, StoreError, QueryAdapter, QueryResult, SemanticOutlier};
pub use semantic_disk::SemanticDisk;


/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Crate authors (from Cargo.toml)
pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
/// Crate description (from Cargo.toml)
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");
