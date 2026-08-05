//! extension.rs - Pluggable Extension System for horon-engine
//!
//! Provides a comprehensive extension framework for the HTT system,
//! enabling integration with various backends, databases, and specialized
//! processing capabilities.
//!
//! ## Extension Categories
//!
//! - **Storage Adapters**: Connect HTT to databases (MySQL, PostgreSQL, MongoDB, SQLite)
//! - **Persistence Providers**: Enable durable storage on disk or distributed systems
//! - **Semantic Analyzers**: Extract meaning and structure from text or code
//! - **Geometric Enhancers**: Discover and utilize hidden geometric patterns in data

use std::any::Any;
use std::collections::HashMap;
use std::fmt::{self, Debug, Formatter};
use std::sync::{Arc, RwLock};
use super::tensor_network::CompressedNode;
use super::tree_tensor::IntegrationError;

/// Result type for extension operations.
pub type ExtensionResult<T> = Result<T, ExtensionError>;

/// Error type for extension operations.
#[derive(Debug)]
pub enum ExtensionError {
    /// Storage-related errors
    Storage(String),
    /// Processing-related errors
    Processing(String),
    /// Configuration errors
    Configuration(String),
    /// Serialization errors
    Serialization(String),
    /// Backend-specific errors
    Backend(String),
    /// Unsupported operation
    Unsupported(String),
    /// Integration errors from the main system
    Integration(IntegrationError),
}

impl From<IntegrationError> for ExtensionError {
    fn from(err: IntegrationError) -> Self {
        ExtensionError::Integration(err)
    }
}

impl From<serde_json::Error> for ExtensionError {
    fn from(err: serde_json::Error) -> Self {
        ExtensionError::Serialization(err.to_string())
    }
}

impl fmt::Display for ExtensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtensionError::Storage(msg) => write!(f, "Storage error: {}", msg),
            ExtensionError::Processing(msg) => write!(f, "Processing error: {}", msg),
            ExtensionError::Configuration(msg) => write!(f, "Configuration error: {}", msg),
            ExtensionError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            ExtensionError::Backend(msg) => write!(f, "Backend error: {}", msg),
            ExtensionError::Unsupported(msg) => write!(f, "Unsupported operation: {}", msg),
            ExtensionError::Integration(err) => write!(f, "Integration error: {}", err),
        }
    }
}

impl std::error::Error for ExtensionError {}

/// Extension capability flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensionCapabilities {
    /// Can store data
    pub can_store: bool,
    /// Can retrieve data
    pub can_retrieve: bool,
    /// Can process nodes
    pub can_process: bool,
    /// Can enhance queries
    pub can_enhance_queries: bool,
    /// Can optimize structure
    pub can_optimize: bool,
    /// Can analyze semantic content
    pub can_analyze_semantics: bool,
}

impl ExtensionCapabilities {
    /// Create new capabilities with all flags disabled.
    pub fn none() -> Self {
        Self {
            can_store: false,
            can_retrieve: false,
            can_process: false,
            can_enhance_queries: false,
            can_optimize: false,
            can_analyze_semantics: false,
        }
    }

    /// Create capabilities for a storage provider.
    pub fn storage() -> Self {
        Self {
            can_store: true,
            can_retrieve: true,
            can_process: false,
            can_enhance_queries: false,
            can_optimize: false,
            can_analyze_semantics: false,
        }
    }

    /// Create capabilities for a processor extension.
    pub fn processor() -> Self {
        Self {
            can_store: false,
            can_retrieve: false,
            can_process: true,
            can_enhance_queries: true,
            can_optimize: false,
            can_analyze_semantics: false,
        }
    }

    /// Create capabilities for a semantic analyzer.
    pub fn semantic_analyzer() -> Self {
        Self {
            can_store: false,
            can_retrieve: false,
            can_process: true,
            can_enhance_queries: true,
            can_optimize: false,
            can_analyze_semantics: true,
        }
    }

    /// Create capabilities for an optimizer.
    pub fn optimizer() -> Self {
        Self {
            can_store: false,
            can_retrieve: false,
            can_process: false,
            can_enhance_queries: false,
            can_optimize: true,
            can_analyze_semantics: false,
        }
    }
}

/// Base extension interface for the HTT module.
pub trait HTTExtensionBase: Send + Sync {
    /// Get the extension name.
    fn name(&self) -> &str;

    /// Get the extension version.
    fn version(&self) -> &str;

    /// Get the extension description.
    fn description(&self) -> &str;

    /// Get the extension capabilities.
    fn capabilities(&self) -> ExtensionCapabilities;

    /// Get extension metadata.
    fn metadata(&self) -> HashMap<String, String>;

    /// Initialize the extension.
    fn initialize(&mut self) -> ExtensionResult<()>;

    /// Shut down the extension.
    fn shutdown(&mut self) -> ExtensionResult<()>;

    /// Check if the extension is compatible with the given data type.
    fn is_compatible_with(&self, data_type: &str) -> bool;

    /// Convert the extension to Any for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Convert the extension to mutable Any for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Node processing extension interface.
///
/// Allows extensions to process CompressedNode instances,
/// enhancing them with additional capabilities or extracting information.
pub trait HTTExtension: HTTExtensionBase {
    /// Process a node, potentially transforming it.
    fn process_node(&self, node: &CompressedNode) -> ExtensionResult<CompressedNode>;

    /// Process multiple nodes in batch.
    fn process_nodes(&self, nodes: &[CompressedNode]) -> ExtensionResult<Vec<CompressedNode>> {
        let mut results = Vec::with_capacity(nodes.len());
        for node in nodes {
            results.push(self.process_node(node)?);
        }
        Ok(results)
    }

    /// Enhance a query before it's executed.
    fn enhance_query(
        &self,
        path: &str,
        query_params: &HashMap<String, String>,
    ) -> ExtensionResult<(String, HashMap<String, String>)>;

    /// Extract information from a node.
    fn extract_info(&self, node: &CompressedNode) -> ExtensionResult<HashMap<String, String>>;

    /// Check if this extension can process the given node.
    fn can_process(&self, node: &CompressedNode) -> bool;
}

/// Storage provider interface for HTT.
///
/// Allows HTT to be integrated with various storage backends.
pub trait HTTStorageProvider: HTTExtensionBase {
    /// Store data at the given path.
    fn store(&mut self, path: &str, data: &[u8]) -> ExtensionResult<()>;

    /// Retrieve data from the given path.
    fn retrieve(&self, path: &str) -> ExtensionResult<Vec<u8>>;

    /// Delete data at the given path.
    fn delete(&mut self, path: &str) -> ExtensionResult<()>;

    /// Check if data exists at the given path.
    fn exists(&self, path: &str) -> ExtensionResult<bool>;

    /// List paths matching the given prefix.
    fn list(&self, prefix: &str) -> ExtensionResult<Vec<String>>;

    /// Get metadata for the given path.
    fn get_metadata(&self, path: &str) -> ExtensionResult<HashMap<String, String>>;

    /// Set metadata for the given path.
    fn set_metadata(&mut self, path: &str, key: &str, value: &str) -> ExtensionResult<()>;

    /// Flush any pending changes to durable storage.
    fn flush(&mut self) -> ExtensionResult<()>;

    /// Begin a transaction.
    fn begin_transaction(&mut self) -> ExtensionResult<()>;

    /// Commit a transaction.
    fn commit_transaction(&mut self) -> ExtensionResult<()>;

    /// Rollback a transaction.
    fn rollback_transaction(&mut self) -> ExtensionResult<()>;

    /// Check if the provider supports transactions.
    fn supports_transactions(&self) -> bool;

    /// Check if the provider is available.
    fn is_available(&self) -> bool;

    /// Get provider statistics.
    fn stats(&self) -> ExtensionResult<HashMap<String, String>>;
}

/// Extension manager for coordinating multiple extensions.
pub struct HTTExtensionManager {
    /// Registered extensions
    extensions: HashMap<String, Box<dyn HTTExtensionBase>>,
    /// Extension configurations
    configs: HashMap<String, HashMap<String, String>>,
    /// Extension dependency ordering
    dependency_order: Vec<String>,
    /// Extension capabilities cache
    capabilities_cache: HashMap<String, ExtensionCapabilities>,
}

impl HTTExtensionManager {
    /// Create a new extension manager.
    pub fn new() -> Self {
        Self {
            extensions: HashMap::new(),
            configs: HashMap::new(),
            dependency_order: Vec::new(),
            capabilities_cache: HashMap::new(),
        }
    }

    /// Register an extension.
    pub fn register_extension<E: HTTExtensionBase + 'static>(
        &mut self,
        extension: E,
        config: HashMap<String, String>,
    ) -> ExtensionResult<()> {
        let name = extension.name().to_string();

        self.capabilities_cache
            .insert(name.clone(), extension.capabilities());
        self.extensions.insert(name.clone(), Box::new(extension));
        self.configs.insert(name.clone(), config);

        if !self.dependency_order.contains(&name) {
            self.dependency_order.push(name);
        }

        Ok(())
    }

    /// Get an extension by name.
    pub fn get_extension(&self, name: &str) -> Option<&dyn HTTExtensionBase> {
        self.extensions.get(name).map(|ext| ext.as_ref())
    }

    /// Get a mutable extension by name.
    pub fn get_extension_mut(
        &mut self,
        name: &str,
    ) -> Option<&mut (dyn HTTExtensionBase + 'static)> {
        self.extensions.get_mut(name).map(|ext| &mut **ext)
    }

    /// Get an extension as a specific type.
    pub fn get_extension_as<T: 'static>(&self, name: &str) -> Option<&T> {
        self.get_extension(name)
            .and_then(|ext| ext.as_any().downcast_ref::<T>())
    }

    /// Get a mutable extension as a specific type.
    pub fn get_extension_mut_as<T: 'static>(&mut self, name: &str) -> Option<&mut T> {
        self.get_extension_mut(name)
            .and_then(|ext| ext.as_any_mut().downcast_mut::<T>())
    }

    /// Get all registered extension names.
    pub fn extension_names(&self) -> Vec<String> {
        self.extensions.keys().cloned().collect()
    }

    /// Get all extensions with specific capabilities.
    pub fn extensions_with_capability(
        &self,
        capability: fn(&ExtensionCapabilities) -> bool,
    ) -> Vec<&dyn HTTExtensionBase> {
        self.extensions
            .values()
            .filter(|ext| capability(&ext.capabilities()))
            .map(|ext| ext.as_ref())
            .collect()
    }

    /// Initialize all extensions in dependency order.
    pub fn initialize_all(&mut self) -> ExtensionResult<()> {
        for name in &self.dependency_order {
            if let Some(ext) = self.extensions.get_mut(name) {
                ext.initialize()?;
            }
        }
        Ok(())
    }

    /// Shut down all extensions in reverse dependency order.
    pub fn shutdown_all(&mut self) -> ExtensionResult<()> {
        for name in self.dependency_order.iter().rev() {
            if let Some(ext) = self.extensions.get_mut(name) {
                ext.shutdown()?;
            }
        }
        Ok(())
    }

    /// Get configuration for an extension.
    pub fn get_config(&self, name: &str) -> Option<&HashMap<String, String>> {
        self.configs.get(name)
    }

    /// Set configuration for an extension.
    pub fn set_config(
        &mut self,
        name: &str,
        config: HashMap<String, String>,
    ) -> ExtensionResult<()> {
        if self.extensions.contains_key(name) {
            self.configs.insert(name.to_string(), config);
            Ok(())
        } else {
            Err(ExtensionError::Configuration(format!(
                "Extension {} not found",
                name
            )))
        }
    }

    /// Check if an extension is registered.
    pub fn has_extension(&self, name: &str) -> bool {
        self.extensions.contains_key(name)
    }

    /// Remove an extension.
    pub fn remove_extension(&mut self, name: &str) -> ExtensionResult<()> {
        if !self.extensions.contains_key(name) {
            return Err(ExtensionError::Configuration(format!(
                "Extension {} not found",
                name
            )));
        }

        if let Some(ext) = self.extensions.get_mut(name) {
            ext.shutdown()?;
        }

        self.extensions.remove(name);
        self.configs.remove(name);
        self.capabilities_cache.remove(name);
        self.dependency_order.retain(|n| n != name);

        Ok(())
    }
}

impl Debug for HTTExtensionManager {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("HTTExtensionManager")
            .field("extension_count", &self.extensions.len())
            .field("extension_names", &self.extension_names())
            .finish()
    }
}

/// Extension registry for system-wide extension management.
pub struct ExtensionRegistry {
    /// The singleton manager instance
    manager: Arc<RwLock<HTTExtensionManager>>,
}

impl ExtensionRegistry {
    /// Get the global extension registry instance.
    pub fn instance() -> Self {
        use std::sync::OnceLock;
        static MANAGER: OnceLock<Arc<RwLock<HTTExtensionManager>>> = OnceLock::new();

        let manager = MANAGER.get_or_init(|| {
            Arc::new(RwLock::new(HTTExtensionManager::new()))
        });

        Self {
            manager: manager.clone(),
        }
    }

    /// Get a read-only reference to the extension manager.
    pub fn manager(
        &self,
    ) -> ExtensionResult<std::sync::RwLockReadGuard<'_, HTTExtensionManager>> {
        self.manager.read().map_err(|e| {
            ExtensionError::Configuration(format!("Failed to lock extension manager: {}", e))
        })
    }

    /// Get a mutable reference to the extension manager.
    pub fn manager_mut(
        &self,
    ) -> ExtensionResult<std::sync::RwLockWriteGuard<'_, HTTExtensionManager>> {
        self.manager.write().map_err(|e| {
            ExtensionError::Configuration(format!("Failed to lock extension manager: {}", e))
        })
    }
}

/// Simple file system storage provider implementation.
pub struct FileSystemStorageProvider {
    /// Root directory for storage
    root_dir: String,
    /// Extension name
    name: String,
    /// Extension version
    version: String,
    /// Extension description
    description: String,
    /// Extension metadata
    ext_metadata: HashMap<String, String>,
    /// In-memory metadata cache
    metadata_cache: HashMap<String, HashMap<String, String>>,
}

impl FileSystemStorageProvider {
    /// Create a new file system storage provider.
    pub fn new(root_dir: String) -> Self {
        Self {
            root_dir,
            name: "FileSystemStorageProvider".to_string(),
            version: "1.0.0".to_string(),
            description: "File system storage provider for HTT".to_string(),
            ext_metadata: HashMap::new(),
            metadata_cache: HashMap::new(),
        }
    }

    fn path_to_fs_path(&self, path: &str) -> String {
        let normalized_path = if path.starts_with('/') {
            path.trim_start_matches('/')
        } else {
            path
        };
        format!("{}/{}", self.root_dir, normalized_path)
    }

    fn ensure_parent_dirs(&self, path: &str) -> ExtensionResult<()> {
        let fs_path = self.path_to_fs_path(path);
        if let Some(parent) = std::path::Path::new(&fs_path).parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ExtensionError::Storage(format!("Failed to create directories: {}", e)))?;
        }
        Ok(())
    }

    /// Read and parse a path's metadata straight from disk, without touching
    /// the cache. Returns an empty map when no metadata file exists. Pure
    /// (`&self`) so callers holding only a shared reference can read without
    /// cloning the whole provider.
    fn read_metadata_from_disk(&self, path: &str) -> ExtensionResult<HashMap<String, String>> {
        let metadata_path = format!("{}.metadata", self.path_to_fs_path(path));

        if std::path::Path::new(&metadata_path).exists() {
            let metadata_str = std::fs::read_to_string(&metadata_path)
                .map_err(|e| ExtensionError::Storage(format!("Failed to read metadata: {}", e)))?;

            serde_json::from_str(&metadata_str).map_err(|e| {
                ExtensionError::Serialization(format!("Failed to parse metadata: {}", e))
            })
        } else {
            Ok(HashMap::new())
        }
    }

    fn load_metadata(&mut self, path: &str) -> ExtensionResult<HashMap<String, String>> {
        let metadata = self.read_metadata_from_disk(path)?;
        if !metadata.is_empty() {
            self.metadata_cache
                .insert(path.to_string(), metadata.clone());
        }
        Ok(metadata)
    }

    fn save_metadata(
        &self,
        path: &str,
        metadata: &HashMap<String, String>,
    ) -> ExtensionResult<()> {
        let metadata_path = format!("{}.metadata", self.path_to_fs_path(path));
        let metadata_str = serde_json::to_string(metadata).map_err(|e| {
            ExtensionError::Serialization(format!("Failed to serialize metadata: {}", e))
        })?;
        self.ensure_parent_dirs(path)?;
        std::fs::write(&metadata_path, metadata_str)
            .map_err(|e| ExtensionError::Storage(format!("Failed to write metadata: {}", e)))?;
        Ok(())
    }
}

impl HTTExtensionBase for FileSystemStorageProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn capabilities(&self) -> ExtensionCapabilities {
        ExtensionCapabilities::storage()
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = self.ext_metadata.clone();
        meta.insert("root_dir".to_string(), self.root_dir.clone());
        meta
    }

    fn initialize(&mut self) -> ExtensionResult<()> {
        std::fs::create_dir_all(&self.root_dir)
            .map_err(|e| ExtensionError::Storage(format!("Failed to create root directory: {}", e)))?;
        Ok(())
    }

    fn shutdown(&mut self) -> ExtensionResult<()> {
        Ok(())
    }

    fn is_compatible_with(&self, data_type: &str) -> bool {
        data_type == "bytes" || data_type == "CompressedNode"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl HTTStorageProvider for FileSystemStorageProvider {
    fn store(&mut self, path: &str, data: &[u8]) -> ExtensionResult<()> {
        let fs_path = self.path_to_fs_path(path);
        self.ensure_parent_dirs(path)?;
        std::fs::write(&fs_path, data)
            .map_err(|e| ExtensionError::Storage(format!("Failed to write file: {}", e)))?;
        Ok(())
    }

    fn retrieve(&self, path: &str) -> ExtensionResult<Vec<u8>> {
        let fs_path = self.path_to_fs_path(path);
        std::fs::read(&fs_path)
            .map_err(|e| ExtensionError::Storage(format!("Failed to read file: {}", e)))
    }

    fn delete(&mut self, path: &str) -> ExtensionResult<()> {
        let fs_path = self.path_to_fs_path(path);
        let metadata_path = format!("{}.metadata", fs_path);

        self.metadata_cache.remove(path);

        if std::path::Path::new(&metadata_path).exists() {
            std::fs::remove_file(&metadata_path)
                .map_err(|e| ExtensionError::Storage(format!("Failed to delete metadata file: {}", e)))?;
        }

        if std::path::Path::new(&fs_path).exists() {
            std::fs::remove_file(&fs_path)
                .map_err(|e| ExtensionError::Storage(format!("Failed to delete file: {}", e)))?;
        }

        Ok(())
    }

    fn exists(&self, path: &str) -> ExtensionResult<bool> {
        let fs_path = self.path_to_fs_path(path);
        Ok(std::path::Path::new(&fs_path).exists())
    }

    fn list(&self, prefix: &str) -> ExtensionResult<Vec<String>> {
        let prefix_path = self.path_to_fs_path(prefix);
        let prefix_dir = if std::path::Path::new(&prefix_path).is_dir() {
            prefix_path.clone()
        } else {
            std::path::Path::new(&prefix_path)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| self.root_dir.clone())
        };

        fn walk_dir(
            dir: &std::path::Path,
            prefix: &str,
            root_dir: &str,
        ) -> ExtensionResult<Vec<String>> {
            let mut results = Vec::new();
            if dir.exists() && dir.is_dir() {
                for entry in std::fs::read_dir(dir)
                    .map_err(|e| ExtensionError::Storage(format!("Failed to read directory: {}", e)))?
                {
                    let entry = entry.map_err(|e| {
                        ExtensionError::Storage(format!("Failed to read entry: {}", e))
                    })?;
                    let path = entry.path();

                    if path.is_file() && !path.to_string_lossy().ends_with(".metadata") {
                        let logical_path = path
                            .to_string_lossy()
                            .trim_start_matches(root_dir)
                            .replace('\\', "/")
                            .to_string();
                        if logical_path.starts_with(prefix) {
                            results.push(logical_path);
                        }
                    } else if path.is_dir() {
                        let sub = walk_dir(&path, prefix, root_dir)?;
                        results.extend(sub);
                    }
                }
            }
            Ok(results)
        }

        walk_dir(
            std::path::Path::new(&prefix_dir),
            prefix,
            &self.root_dir,
        )
    }

    fn get_metadata(&self, path: &str) -> ExtensionResult<HashMap<String, String>> {
        if let Some(metadata) = self.metadata_cache.get(path) {
            return Ok(metadata.clone());
        }
        let mut this = self.clone();
        this.load_metadata(path)
    }

    fn set_metadata(&mut self, path: &str, key: &str, value: &str) -> ExtensionResult<()> {
        let mut metadata = self.get_metadata(path)?;
        metadata.insert(key.to_string(), value.to_string());
        self.metadata_cache
            .insert(path.to_string(), metadata.clone());
        self.save_metadata(path, &metadata)
    }

    fn flush(&mut self) -> ExtensionResult<()> {
        for (path, metadata) in &self.metadata_cache {
            self.save_metadata(path, metadata)?;
        }
        Ok(())
    }

    fn begin_transaction(&mut self) -> ExtensionResult<()> {
        Err(ExtensionError::Unsupported(
            "Transactions not supported".to_string(),
        ))
    }

    fn commit_transaction(&mut self) -> ExtensionResult<()> {
        Err(ExtensionError::Unsupported(
            "Transactions not supported".to_string(),
        ))
    }

    fn rollback_transaction(&mut self) -> ExtensionResult<()> {
        Err(ExtensionError::Unsupported(
            "Transactions not supported".to_string(),
        ))
    }

    fn supports_transactions(&self) -> bool {
        false
    }

    fn is_available(&self) -> bool {
        std::path::Path::new(&self.root_dir).exists()
    }

    fn stats(&self) -> ExtensionResult<HashMap<String, String>> {
        let mut stats = HashMap::new();

        fn walk_dir_stats(dir: &std::path::Path) -> ExtensionResult<(usize, u64)> {
            let mut count = 0;
            let mut size = 0;
            if dir.exists() && dir.is_dir() {
                for entry in std::fs::read_dir(dir)
                    .map_err(|e| ExtensionError::Storage(format!("Failed to read directory: {}", e)))?
                {
                    let entry = entry.map_err(|e| {
                        ExtensionError::Storage(format!("Failed to read entry: {}", e))
                    })?;
                    let path = entry.path();
                    if path.is_file() && !path.to_string_lossy().ends_with(".metadata") {
                        count += 1;
                        size += entry
                            .metadata()
                            .map_err(|e| {
                                ExtensionError::Storage(format!("Failed to get metadata: {}", e))
                            })?
                            .len();
                    } else if path.is_dir() {
                        let (sc, ss) = walk_dir_stats(&path)?;
                        count += sc;
                        size += ss;
                    }
                }
            }
            Ok((count, size))
        }

        let (file_count, total_size) =
            walk_dir_stats(std::path::Path::new(&self.root_dir))?;

        stats.insert("file_count".to_string(), file_count.to_string());
        stats.insert("total_size_bytes".to_string(), total_size.to_string());
        stats.insert("root_dir".to_string(), self.root_dir.clone());

        Ok(stats)
    }
}

impl Clone for FileSystemStorageProvider {
    fn clone(&self) -> Self {
        Self {
            root_dir: self.root_dir.clone(),
            name: self.name.clone(),
            version: self.version.clone(),
            description: self.description.clone(),
            ext_metadata: self.ext_metadata.clone(),
            metadata_cache: self.metadata_cache.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_extension_manager() {
        let mut manager = HTTExtensionManager::new();

        struct TestExtension;

        impl HTTExtensionBase for TestExtension {
            fn name(&self) -> &str {
                "test"
            }
            fn version(&self) -> &str {
                "1.0.0"
            }
            fn description(&self) -> &str {
                "Test extension"
            }
            fn capabilities(&self) -> ExtensionCapabilities {
                ExtensionCapabilities::none()
            }
            fn metadata(&self) -> HashMap<String, String> {
                HashMap::new()
            }
            fn initialize(&mut self) -> ExtensionResult<()> {
                Ok(())
            }
            fn shutdown(&mut self) -> ExtensionResult<()> {
                Ok(())
            }
            fn is_compatible_with(&self, _data_type: &str) -> bool {
                true
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
        }

        let result = manager.register_extension(TestExtension, HashMap::new());
        assert!(result.is_ok());

        assert!(manager.has_extension("test"));

        let extension = manager.get_extension("test");
        assert!(extension.is_some());
        assert_eq!(extension.unwrap().name(), "test");

        let result = manager.remove_extension("test");
        assert!(result.is_ok());
        assert!(!manager.has_extension("test"));
    }

    #[test]
    fn test_filesystem_provider() {
        let temp_dir = tempdir().unwrap();
        let root_dir = temp_dir.path().to_string_lossy().to_string();

        let mut provider = FileSystemStorageProvider::new(root_dir.clone());

        let result = provider.initialize();
        assert!(result.is_ok());

        let data = b"test data";
        let result = provider.store("/test.txt", data);
        assert!(result.is_ok());

        let exists = provider.exists("/test.txt").unwrap();
        assert!(exists);

        let retrieved = provider.retrieve("/test.txt").unwrap();
        assert_eq!(retrieved, data);

        let result = provider.set_metadata("/test.txt", "content_type", "text/plain");
        assert!(result.is_ok());

        let metadata = provider.get_metadata("/test.txt").unwrap();
        assert_eq!(
            metadata.get("content_type"),
            Some(&"text/plain".to_string())
        );

        let files = provider.list("/").unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], "/test.txt");

        let result = provider.delete("/test.txt");
        assert!(result.is_ok());

        let exists = provider.exists("/test.txt").unwrap();
        assert!(!exists);
    }
}
