//! Error handling for the horon-engine library
//!
//! This module defines a comprehensive error handling system for the Hyperbolic Tree Tensor
//! library. It establishes HTT-specific error types that eliminate dependencies on
//! GSD's IntegrationError, enabling HTT to function as a fully standalone library.
//! The module provides bidirectional conversions between HTT errors and GSD integration
//! errors when the "gsd" feature is enabled.

use std::fmt;
use std::error::Error;
use std::io;

/// Main error type for HTT operations
#[derive(Debug)]
pub enum HTTError {
    /// Configuration error
    Config(String),
    
    /// Storage operation error
    Storage(String),
    
    /// Tensor network operation error
    Tensor(String),
    
    /// Hyperbolic geometry error
    Geometry(String),
    
    /// Dimension mismatch
    DimensionMismatch {
        /// The dimension the store was configured with.
        expected: usize,
        /// The dimension the caller supplied.
        actual: usize,
    },
    
    /// Component registry error
    Registry(String),
    
    /// I/O error (for file operations)
    IO(io::Error),
    
    /// Serialization error
    Serialization(String),
    
    /// Extension error
    Extension(String),
    
    /// Initialization error
    Initialization(String),
}

impl fmt::Display for HTTError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HTTError::Config(msg) => write!(f, "Configuration error: {}", msg),
            HTTError::Storage(msg) => write!(f, "Storage error: {}", msg),
            HTTError::Tensor(msg) => write!(f, "Tensor operation error: {}", msg),
            HTTError::Geometry(msg) => write!(f, "Hyperbolic geometry error: {}", msg),
            HTTError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            },
            HTTError::Registry(msg) => write!(f, "Registry error: {}", msg),
            HTTError::IO(err) => write!(f, "I/O error: {}", err),
            HTTError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            HTTError::Extension(msg) => write!(f, "Extension error: {}", msg),
            HTTError::Initialization(msg) => write!(f, "Initialization error: {}", msg),
        }
    }
}

impl Error for HTTError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            HTTError::IO(err) => Some(err),
            _ => None,
        }
    }
}

// Implement From trait for common error types

impl From<io::Error> for HTTError {
    fn from(err: io::Error) -> Self {
        HTTError::IO(err)
    }
}

impl From<serde_json::Error> for HTTError {
    fn from(err: serde_json::Error) -> Self {
        HTTError::Serialization(err.to_string())
    }
}

impl From<super::registry::RegistryError> for HTTError {
    fn from(err: super::registry::RegistryError) -> Self {
        HTTError::Registry(err.to_string())
    }
}

/// Result type alias for HTT operations.
pub type HTTResult<T> = Result<T, HTTError>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let errors = vec![
            HTTError::Config("Invalid configuration".to_string()),
            HTTError::Storage("Storage failure".to_string()),
            HTTError::Tensor("Tensor operation failed".to_string()),
            HTTError::Geometry("Invalid hyperbolic coordinates".to_string()),
            HTTError::DimensionMismatch { expected: 3, actual: 2 },
            HTTError::Registry("Component not found".to_string()),
            HTTError::IO(io::Error::new(io::ErrorKind::NotFound, "File not found")),
            HTTError::Serialization("Invalid JSON".to_string()),
            HTTError::Extension("Extension failed to load".to_string()),
            HTTError::Initialization("Failed to initialize HTT".to_string()),
        ];
        
        for error in errors {
            // Just check that display doesn't panic
            let _display = format!("{}", error);
            assert!(!_display.is_empty());
        }
    }
    
    #[test]
    fn test_io_error_conversion() {
        let io_error = io::Error::new(io::ErrorKind::PermissionDenied, "Permission denied");
        let htt_error: HTTError = io_error.into();
        
        match htt_error {
            HTTError::IO(_) => (), // Expected
            _ => panic!("Expected IO error variant"),
        }
    }
}