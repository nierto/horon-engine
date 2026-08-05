//! registry.rs - ComponentRegistry for module independence
//!
//! This module implements a flexible component registry system that enables
//! HTT to function as a standalone library while also supporting integration
//! with GSD. It provides a uniform interface for component registration,
//! retrieval, and management across different runtime environments.
//!
//! The registry maintains type safety through runtime type checking and
//! manages component lifecycles efficiently through reference counting
//! and thread-safe access patterns.

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// Error type for registry operations
#[derive(Debug, Clone)]
pub struct RegistryError {
    message: String,
}

impl RegistryError {
    /// Create a registry error with the given message.
    pub fn new(message: &str) -> Self {
        RegistryError {
            message: message.to_string(),
        }
    }
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Registry error: {}", self.message)
    }
}

impl std::error::Error for RegistryError {}

/// ComponentRegistry trait for registering and retrieving components
///
/// This trait defines the interface for a component registry used by HTT
/// to store and retrieve components. It's designed to be implemented by
/// both the standalone HTT library and adapters for external systems like GSD.
pub trait ComponentRegistry: Send + Sync {
    /// Register a component with the given name
    fn register<T: 'static + Send + Sync>(&mut self, name: &str, component: Arc<RwLock<T>>) -> Result<(), RegistryError>;
    
    /// Get a component by name
    fn get<T: 'static + Send + Sync>(&self, name: &str) -> Option<Arc<RwLock<T>>>;
    
    /// Get a component by name for mutation.
    ///
    /// Returns the same `Arc<RwLock<T>>` as [`get`](Self::get); mutation goes
    /// through the returned `RwLock`, so no separate exclusive handle is
    /// needed. Retained as a distinct method for call-site intent.
    fn get_mut<T: 'static + Send + Sync>(&self, name: &str) -> Option<Arc<RwLock<T>>>;
    
    /// Check if a component exists
    fn contains(&self, name: &str) -> bool;
}

/// Standalone implementation of ComponentRegistry for use in HTT without GSD
pub struct HTTComponentRegistry {
    components: HashMap<String, Arc<RwLock<Box<dyn Any + Send + Sync>>>>,
}

impl Debug for HTTComponentRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HTTComponentRegistry")
            .field("components_count", &self.components.len())
            .finish()
    }
}

impl HTTComponentRegistry {
    /// Create a new empty component registry
    pub fn new() -> Self {
        HTTComponentRegistry {
            components: HashMap::new(),
        }
    }
    
    /// Get the number of registered components
    pub fn len(&self) -> usize {
        self.components.len()
    }
    
    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }
}

impl ComponentRegistry for HTTComponentRegistry {
    fn register<T: 'static + Send + Sync>(&mut self, name: &str, component: Arc<RwLock<T>>) -> Result<(), RegistryError> {
        // Store the Arc<RwLock<T>> as a boxed Any inside an Arc<RwLock<_>>
        let boxed: Box<dyn Any + Send + Sync> = Box::new(component);
        self.components.insert(name.to_string(), Arc::new(RwLock::new(boxed)));
        Ok(())
    }

    fn get<T: 'static + Send + Sync>(&self, name: &str) -> Option<Arc<RwLock<T>>> {
        self.components.get(name).and_then(|boxed_any| {
            let guard = boxed_any.read().ok()?;
            guard.downcast_ref::<Arc<RwLock<T>>>().cloned()
        })
    }

    fn get_mut<T: 'static + Send + Sync>(&self, name: &str) -> Option<Arc<RwLock<T>>> {
        // Mutation is via the returned RwLock; the handle is identical to get.
        self.get(name)
    }
    
    fn contains(&self, name: &str) -> bool {
        self.components.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock component for testing
    #[derive(Debug)]
    struct TestComponent {
        value: i32,
    }
    
    #[test]
    fn test_registry_operations() {
        // Create registry
        let mut registry = HTTComponentRegistry::new();
        
        // Register a component
        let component = Arc::new(RwLock::new(TestComponent { value: 42 }));
        registry.register("test", component).unwrap();
        
        // Check contains
        assert!(registry.contains("test"));
        assert!(!registry.contains("nonexistent"));
        
        // Get the component
        let retrieved = registry.get::<TestComponent>("test").unwrap();
        assert_eq!(retrieved.read().unwrap().value, 42);
        
        // Modify the component
        {
            let mut comp = retrieved.write().unwrap();
            comp.value = 100;
        }
        
        // Get it again to verify the change
        let retrieved_again = registry.get::<TestComponent>("test").unwrap();
        assert_eq!(retrieved_again.read().unwrap().value, 100);
    }
    
    #[test]
    fn test_multiple_components() {
        // Create registry
        let mut registry = HTTComponentRegistry::new();
        
        // Register multiple components
        registry.register("comp1", Arc::new(RwLock::new(TestComponent { value: 1 }))).unwrap();
        registry.register("comp2", Arc::new(RwLock::new(TestComponent { value: 2 }))).unwrap();
        registry.register("comp3", Arc::new(RwLock::new(TestComponent { value: 3 }))).unwrap();
        
        // Check registry size
        assert_eq!(registry.len(), 3);
        
        // Get all components
        let comp1 = registry.get::<TestComponent>("comp1").unwrap();
        let comp2 = registry.get::<TestComponent>("comp2").unwrap();
        let comp3 = registry.get::<TestComponent>("comp3").unwrap();
        
        // Verify values
        assert_eq!(comp1.read().unwrap().value, 1);
        assert_eq!(comp2.read().unwrap().value, 2);
        assert_eq!(comp3.read().unwrap().value, 3);
    }
}