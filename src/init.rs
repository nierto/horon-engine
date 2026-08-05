//! init.rs - Initialization Module for horon-engine storage
//!
//! Provides functions for initializing and managing
//! hyperbolic tree tensor (HTT) storage instances.

use log::info;

use super::storage::HTTStorage;
use super::config::HTTStorageConfig;
use super::tree_tensor::{SharedHTT, HyperbolicTreeTensor, IntegrationResult};

/// Initialize HTT storage with a configuration.
pub fn initialize_htt_storage(config: HTTStorageConfig) -> IntegrationResult<HTTStorage> {
    info!("Initializing HTT storage components");

    let storage = HTTStorage::new(config);

    info!("HTT storage components initialized");
    Ok(storage)
}

/// Execute a function with access to a shared HTT instance.
///
/// No locking needed: all HyperbolicTreeTensor methods take `&self`
/// with fine-grained interior mutability (DashMap, Mutex, RwLock).
pub fn with_htt<R, F>(
    htt: &SharedHTT,
    f: F,
) -> IntegrationResult<R>
where
    F: FnOnce(&HyperbolicTreeTensor) -> IntegrationResult<R>,
{
    f(htt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_htt_initialization() {
        let config = HTTStorageConfig::default();
        let result = initialize_htt_storage(config);
        assert!(result.is_ok());

        let storage = result.unwrap();
        assert!(storage.exists("/"));
    }

    #[test]
    fn test_htt_with_custom_config() {
        let config = HTTStorageConfig::new(
            8,
            2000,
            200,
            None,
            120,
            false,
        );
        let result = initialize_htt_storage(config);
        assert!(result.is_ok());
    }
}
