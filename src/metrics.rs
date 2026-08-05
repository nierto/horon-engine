//! Performance metrics collection for horon-engine
//!
//! This module implements a comprehensive metrics collection system for HTT.
//! It provides performance monitoring capabilities for operations within the 
//! library, while maintaining compatibility with GSD's MetricsCollector through
//! an optional adapter. The design allows HTT to function as either a standalone
//! library or as an integrated component within a larger system.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Trait for metrics providers
///
/// This trait defines the interface for metrics collection
/// in HTT. It can be implemented by both the standalone HTT
/// library and by adapters for external metrics systems.
pub trait MetricsProvider: Send + Sync {
    /// Record the duration of an operation
    fn record_operation(&self, operation: &str, duration: Duration);
    
    /// Increment a counter
    fn increment_counter(&self, counter: &str, value: u64);
    
    /// Record a gauge value
    fn record_gauge(&self, gauge: &str, value: f64);
    
    /// Create a timer that will automatically record an operation
    /// when it goes out of scope
    fn timer<'a>(&'a self, operation: &'a str) -> OperationTimer<'a>;
    
    /// Get metrics summary as a string
    fn summary(&self) -> String;
}

/// Timer for automatically recording operation duration
pub struct OperationTimer<'a> {
    provider: &'a dyn MetricsProvider,
    operation: &'a str,
    start: Instant,
}

impl<'a> OperationTimer<'a> {
    fn new(provider: &'a dyn MetricsProvider, operation: &'a str) -> Self {
        OperationTimer {
            provider,
            operation,
            start: Instant::now(),
        }
    }
}

impl<'a> Drop for OperationTimer<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.provider.record_operation(self.operation, duration);
    }
}

/// Simple in-memory metrics implementation for standalone use
#[derive(Debug)]
pub struct SimpleMetrics {
    counters: RwLock<HashMap<String, u64>>,
    gauges: RwLock<HashMap<String, f64>>,
    timers: RwLock<HashMap<String, Vec<Duration>>>,
}

impl SimpleMetrics {
    /// Create a new empty metrics collector
    pub fn new() -> Self {
        SimpleMetrics {
            counters: RwLock::new(HashMap::new()),
            gauges: RwLock::new(HashMap::new()),
            timers: RwLock::new(HashMap::new()),
        }
    }
    
    /// Get a specific counter value
    pub fn get_counter(&self, counter: &str) -> Option<u64> {
        self.counters.read().unwrap_or_else(|e| e.into_inner()).get(counter).cloned()
    }
    
    /// Get a specific gauge value
    pub fn get_gauge(&self, gauge: &str) -> Option<f64> {
        self.gauges.read().unwrap_or_else(|e| e.into_inner()).get(gauge).cloned()
    }
    
    /// Get average operation duration
    pub fn get_average_duration(&self, operation: &str) -> Option<Duration> {
        let timers = self.timers.read().unwrap_or_else(|e| e.into_inner());
        let durations = timers.get(operation)?;
        
        if durations.is_empty() {
            return None;
        }
        
        let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
        let avg_nanos = total_nanos / durations.len() as u128;
        
        Some(Duration::from_nanos(avg_nanos as u64))
    }
}

impl MetricsProvider for SimpleMetrics {
    fn record_operation(&self, operation: &str, duration: Duration) {
        let mut timers = self.timers.write().unwrap_or_else(|e| e.into_inner());
        timers.entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
    }
    
    fn increment_counter(&self, counter: &str, value: u64) {
        let mut counters = self.counters.write().unwrap_or_else(|e| e.into_inner());
        *counters.entry(counter.to_string()).or_insert(0) += value;
    }
    
    fn record_gauge(&self, gauge: &str, value: f64) {
        let mut gauges = self.gauges.write().unwrap_or_else(|e| e.into_inner());
        gauges.insert(gauge.to_string(), value);
    }
    
    fn timer<'a>(&'a self, operation: &'a str) -> OperationTimer<'a> {
        OperationTimer::new(self, operation)
    }
    
    fn summary(&self) -> String {
        let mut result = String::new();
        
        // Add counters
        result.push_str("Counters:\n");
        for (name, value) in self.counters.read().unwrap_or_else(|e| e.into_inner()).iter() {
            result.push_str(&format!("  {}: {}\n", name, value));
        }
        
        // Add gauges
        result.push_str("\nGauges:\n");
        for (name, value) in self.gauges.read().unwrap_or_else(|e| e.into_inner()).iter() {
            result.push_str(&format!("  {}: {:.6}\n", name, value));
        }
        
        // Add operation timers
        result.push_str("\nOperations:\n");
        for (name, durations) in self.timers.read().unwrap_or_else(|e| e.into_inner()).iter() {
            if durations.is_empty() {
                continue;
            }
            
            let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
            let avg_nanos = total_nanos / durations.len() as u128;
            let avg_duration = Duration::from_nanos(avg_nanos as u64);
            let min_duration = durations.iter().min().unwrap();
            let max_duration = durations.iter().max().unwrap();
            
            result.push_str(&format!(
                "  {}: count={}, avg={:?}, min={:?}, max={:?}\n",
                name, durations.len(), avg_duration, min_duration, max_duration
            ));
        }
        
        result
    }
}

impl Default for SimpleMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    
    #[test]
    fn test_counter_operations() {
        let metrics = SimpleMetrics::new();
        
        // Increment counters
        metrics.increment_counter("test_counter", 1);
        metrics.increment_counter("test_counter", 2);
        metrics.increment_counter("another_counter", 5);
        
        // Check values
        assert_eq!(metrics.get_counter("test_counter"), Some(3));
        assert_eq!(metrics.get_counter("another_counter"), Some(5));
        assert_eq!(metrics.get_counter("nonexistent_counter"), None);
    }
    
    #[test]
    fn test_gauge_operations() {
        let metrics = SimpleMetrics::new();
        
        // Record gauges
        metrics.record_gauge("test_gauge", 3.14);
        metrics.record_gauge("another_gauge", 2.71);
        
        // Check values
        assert!((metrics.get_gauge("test_gauge").unwrap() - 3.14).abs() < 0.0001);
        assert!((metrics.get_gauge("another_gauge").unwrap() - 2.71).abs() < 0.0001);
        assert_eq!(metrics.get_gauge("nonexistent_gauge"), None);
    }
    
    #[test]
    fn test_timer_operations() {
        let metrics = SimpleMetrics::new();
        
        // Record operations manually
        metrics.record_operation("op1", Duration::from_millis(100));
        metrics.record_operation("op1", Duration::from_millis(200));
        
        // Use timer
        {
            let _timer = metrics.timer("op2");
            sleep(Duration::from_millis(10)); // Sleep to ensure measurable duration
        }
        
        // Check values
        let avg_op1 = metrics.get_average_duration("op1").unwrap();
        assert_eq!(avg_op1, Duration::from_millis(150));
        
        let avg_op2 = metrics.get_average_duration("op2").unwrap();
        assert!(avg_op2.as_millis() >= 10); // At least 10ms
    }
    
    #[test]
    fn test_summary() {
        let metrics = SimpleMetrics::new();
        
        // Add some data
        metrics.increment_counter("requests", 42);
        metrics.record_gauge("memory_usage", 123.456);
        metrics.record_operation("fetch", Duration::from_millis(50));
        
        // Get summary
        let summary = metrics.summary();
        
        // Verify summary contains expected data
        assert!(summary.contains("requests: 42"));
        assert!(summary.contains("memory_usage:"));
        assert!(summary.contains("fetch: count=1"));
    }
}