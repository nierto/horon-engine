//! config.rs - Configuration System for horon-engine storage
//!
//! Configuration for the Hyperbolic Tree Tensor (HTT) storage system.
//!
//! Options for:
//! - Hyperbolic space dimensionality
//! - Memory management and caching
//! - Persistence configuration

use std::fmt::{self, Debug, Formatter};
use g_math::fixed_point::FixedPoint;

/// Configuration for HTT storage.
#[derive(Clone)]
pub struct HTTStorageConfig {
    /// Dimension of the hyperbolic space
    pub dimension: usize,
    /// Maximum in-memory nodes before flushing to storage
    pub max_memory_nodes: usize,
    /// Cache size for frequently accessed nodes
    pub cache_size: usize,
    /// Persistent storage path (if used)
    pub storage_path: Option<String>,
    /// Flush interval in seconds (if persistence is enabled)
    pub flush_interval: u64,
    /// Whether to optimize on shutdown
    pub optimize_on_shutdown: bool,
    /// Sarkar embedding scale factor τ (zero = use default of 1.0)
    pub tau: FixedPoint,
}

impl HTTStorageConfig {
    /// Create a new HTT storage configuration.
    pub fn new(
        dimension: usize,
        max_memory_nodes: usize,
        cache_size: usize,
        storage_path: Option<String>,
        flush_interval: u64,
        optimize_on_shutdown: bool,
    ) -> Self {
        Self {
            dimension,
            max_memory_nodes,
            cache_size,
            storage_path,
            flush_interval,
            optimize_on_shutdown,
            tau: FixedPoint::from_int(0),
        }
    }

    /// Check if persistence is enabled.
    pub fn is_persistence_enabled(&self) -> bool {
        self.storage_path.is_some()
    }
}

impl Default for HTTStorageConfig {
    fn default() -> Self {
        Self {
            dimension: 4,
            max_memory_nodes: 1000,
            cache_size: 100,
            storage_path: None,
            flush_interval: 60,
            optimize_on_shutdown: true,
            tau: FixedPoint::from_int(0),
        }
    }
}

impl Debug for HTTStorageConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("HTTStorageConfig")
            .field("dimension", &self.dimension)
            .field("max_memory_nodes", &self.max_memory_nodes)
            .field("cache_size", &self.cache_size)
            .field("storage_path", &self.storage_path)
            .field("flush_interval", &self.flush_interval)
            .field("optimize_on_shutdown", &self.optimize_on_shutdown)
            .field("tau", &self.tau)
            .finish()
    }
}

/// Builder for HTT storage configuration.
pub struct HTTStorageConfigBuilder {
    config: HTTStorageConfig,
}

impl HTTStorageConfigBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self {
            config: HTTStorageConfig::default(),
        }
    }

    /// Set the dimension.
    pub fn dimension(mut self, dimension: usize) -> Self {
        self.config.dimension = dimension;
        self
    }

    /// Set the maximum in-memory nodes.
    pub fn max_memory_nodes(mut self, max_memory_nodes: usize) -> Self {
        self.config.max_memory_nodes = max_memory_nodes;
        self
    }

    /// Set the cache size.
    pub fn cache_size(mut self, cache_size: usize) -> Self {
        self.config.cache_size = cache_size;
        self
    }

    /// Set the storage path.
    pub fn storage_path(mut self, storage_path: Option<String>) -> Self {
        self.config.storage_path = storage_path;
        self
    }

    /// Set the flush interval.
    pub fn flush_interval(mut self, flush_interval: u64) -> Self {
        self.config.flush_interval = flush_interval;
        self
    }

    /// Set optimize on shutdown.
    pub fn optimize_on_shutdown(mut self, optimize_on_shutdown: bool) -> Self {
        self.config.optimize_on_shutdown = optimize_on_shutdown;
        self
    }

    /// Set the Sarkar embedding scale factor τ (zero = use default of 1.0).
    pub fn tau(mut self, tau: FixedPoint) -> Self {
        self.config.tau = tau;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> HTTStorageConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = HTTStorageConfig::default();

        assert_eq!(config.dimension, 4);
        assert_eq!(config.max_memory_nodes, 1000);
        assert_eq!(config.cache_size, 100);
        assert_eq!(config.storage_path, None);
        assert_eq!(config.flush_interval, 60);
        assert_eq!(config.optimize_on_shutdown, true);
    }

    #[test]
    fn test_config_builder() {
        let config = HTTStorageConfigBuilder::new()
            .dimension(8)
            .max_memory_nodes(2000)
            .cache_size(200)
            .storage_path(Some("/tmp/htt".to_string()))
            .flush_interval(120)
            .optimize_on_shutdown(false)
            .build();

        assert_eq!(config.dimension, 8);
        assert_eq!(config.max_memory_nodes, 2000);
        assert_eq!(config.cache_size, 200);
        assert_eq!(config.storage_path, Some("/tmp/htt".to_string()));
        assert_eq!(config.flush_interval, 120);
        assert_eq!(config.optimize_on_shutdown, false);
    }

    #[test]
    fn test_persistence_enabled() {
        let config1 = HTTStorageConfigBuilder::new()
            .storage_path(Some("/tmp/htt".to_string()))
            .build();
        assert!(config1.is_persistence_enabled());

        let config2 = HTTStorageConfigBuilder::new()
            .storage_path(None)
            .build();
        assert!(!config2.is_persistence_enabled());
    }
}
