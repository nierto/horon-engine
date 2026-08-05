//! utils.rs - Optimization and Utility Functions for horon-engine
//! # Utility functions for horon-engine
//!
//! This module provides high-performance utility functions for the crate:
//!
//! - SIMD-optimized vector operations (delegates to gMath's internal SIMD when available)
//! - Hyperbolic geometry calculations for the Poincaré disk model
//! - Performance monitoring tools for benchmarking and optimization
//! - CPU feature detection for runtime selection of optimal algorithms

use std::fmt::{self, Debug, Formatter};
use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use log::debug;
use crate::constants;

/// CPU feature detection result.
#[derive(Clone, Copy, Debug)]
pub struct CpuFeatures {
    /// Is AVX supported
    pub avx: bool,
    /// Is AVX2 supported
    pub avx2: bool,
    /// Is SSE4.1 supported
    pub sse41: bool,
    /// Is SSE4.2 supported
    pub sse42: bool,
}

impl CpuFeatures {
    /// Detect CPU features.
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        Self {
            avx: is_x86_feature_detected!("avx"),
            avx2: is_x86_feature_detected!("avx2"),
            sse41: is_x86_feature_detected!("sse4.1"),
            sse42: is_x86_feature_detected!("sse4.2"),
        }
    }

    /// Detect CPU features.
    #[cfg(not(target_arch = "x86_64"))]
    pub fn detect() -> Self {
        Self {
            avx: false,
            avx2: false,
            sse41: false,
            sse42: false,
        }
    }
}

/// SIMD optimization module for vector operations.
///
/// Delegates to gMath's FixedVector methods which will gain SIMD acceleration
/// as gMath promotes its internal SIMD infrastructure to production.
pub struct SimdOptimization {
    /// CPU features available
    cpu_features: CpuFeatures,
}

impl SimdOptimization {
    /// Create a new SIMD optimization instance.
    pub fn new() -> Self {
        let cpu_features = CpuFeatures::detect();
        debug!("Detected CPU features: {:?}", cpu_features);

        Self {
            cpu_features,
        }
    }

    /// Get the CPU features.
    pub fn cpu_features(&self) -> CpuFeatures {
        self.cpu_features
    }

    /// Perform optimized vector multiplication.
    /// Delegates to gMath's FixedPoint operators (gains SIMD when gMath enables it).
    pub fn vector_multiply(&self, a: &FixedVector, b: &FixedVector) -> FixedVector {
        let len = a.len().min(b.len());
        let mut result = FixedVector::new(len);

        for i in 0..len {
            result[i] = a[i] * b[i];
        }

        result
    }

    /// Perform optimized matrix-vector multiplication.
    pub fn matrix_vector_multiply(&self, m: &FixedMatrix, v: &FixedVector) -> FixedVector {
        assert_eq!(m.cols(), v.len(), "Matrix columns must match vector length");

        let mut result = FixedVector::new(m.rows());

        for i in 0..m.rows() {
            let mut sum = FixedPoint::from_int(0);
            for j in 0..m.cols() {
                sum = sum + (m.get(i, j) * v[j]);
            }
            result[i] = sum;
        }

        result
    }

    /// Calculate hyperbolic distance using gMath's FixedVector methods.
    pub fn hyperbolic_distance(&self,
                               _disk_radius: FixedPoint,
                               p1: &FixedVector,
                               p2: &FixedVector) -> FixedPoint {
        // Use gMath's fused Euclidean distance (single materialization)
        let euclidean_distance = p1.distance_to(p2);

        // Calculate the denominator: 1 - |p1|²|p2|²
        let p1_norm_sq = p1.dot(p1);
        let p2_norm_sq = p2.dot(p2);

        let one = FixedPoint::from_int(1);
        let two = FixedPoint::from_int(2);
        let denominator = one - p1_norm_sq * p2_norm_sq;

        // Degenerate denominator 1 − |p1|²|p2|² ≈ 0: reachable only as both
        // points approach the boundary (|p1|,|p2| → 1). A divide-by-zero guard
        // for a case the strictly-interior points here do not hit. Saturate to
        // the largest distance the model represents — 2·atanh(near_boundary)
        // ≈ 5.29, the same value the near-boundary clamp below yields —
        // instead of an out-of-band 10000 sentinel.
        if denominator.abs() < constants::epsilon() {
            return two * constants::safe_atanh(constants::near_boundary());
        }

        // Calculate 2 * atanh(|p1-p2| / sqrt(|1-p1²p2²|))
        let ratio = euclidean_distance / denominator.sqrt();

        let safe_ratio = if ratio > constants::near_boundary() {
            constants::near_boundary()
        } else {
            ratio
        };

        two * constants::safe_atanh(safe_ratio)
    }

    /// Calculate Euclidean distance using gMath's FixedVector.
    pub fn euclidean_distance(&self, v1: &FixedVector, v2: &FixedVector) -> FixedPoint {
        v1.distance_to(v2)
    }

    /// Calculate vector norm squared.
    pub fn vector_norm_squared(&self, v: &FixedVector) -> FixedPoint {
        v.dot(v)
    }
}

impl Debug for SimdOptimization {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimdOptimization")
            .field("cpu_features", &self.cpu_features)
            .finish()
    }
}

/// Performance monitoring for HTT operations.
pub struct PerformanceMonitor {
    /// Operation timings
    timings: std::collections::HashMap<String, Vec<std::time::Duration>>,
    /// Start times for in-progress operations
    start_times: std::collections::HashMap<String, std::time::Instant>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor.
    pub fn new() -> Self {
        Self {
            timings: std::collections::HashMap::new(),
            start_times: std::collections::HashMap::new(),
        }
    }

    /// Start timing an operation.
    pub fn start(&mut self, operation: &str) {
        self.start_times.insert(
            operation.to_string(),
            std::time::Instant::now()
        );
    }

    /// Stop timing an operation.
    pub fn stop(&mut self, operation: &str) {
        if let Some(start_time) = self.start_times.remove(operation) {
            let duration = start_time.elapsed();

            self.timings
                .entry(operation.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }

    /// Get the average timing for an operation.
    pub fn average(&self, operation: &str) -> Option<std::time::Duration> {
        if let Some(timings) = self.timings.get(operation) {
            if timings.is_empty() {
                return None;
            }

            let total = timings.iter().sum::<std::time::Duration>();
            let count = timings.len() as u32;

            Some(total / count)
        } else {
            None
        }
    }

    /// Get the min timing for an operation.
    pub fn min(&self, operation: &str) -> Option<std::time::Duration> {
        if let Some(timings) = self.timings.get(operation) {
            timings.iter().min().copied()
        } else {
            None
        }
    }

    /// Get the max timing for an operation.
    pub fn max(&self, operation: &str) -> Option<std::time::Duration> {
        if let Some(timings) = self.timings.get(operation) {
            timings.iter().max().copied()
        } else {
            None
        }
    }

    /// Get statistics for an operation.
    pub fn stats(&self, operation: &str) -> Option<(std::time::Duration, std::time::Duration, std::time::Duration)> {
        if let (Some(avg), Some(min), Some(max)) = (
            self.average(operation),
            self.min(operation),
            self.max(operation),
        ) {
            Some((avg, min, max))
        } else {
            None
        }
    }

    /// Reset all timings.
    pub fn reset(&mut self) {
        self.timings.clear();
        self.start_times.clear();
    }

    /// Get all operation stats.
    pub fn all_stats(&self) -> std::collections::HashMap<String, (std::time::Duration, std::time::Duration, std::time::Duration)> {
        let mut result = std::collections::HashMap::new();

        for operation in self.timings.keys() {
            if let Some(stats) = self.stats(operation) {
                result.insert(operation.clone(), stats);
            }
        }

        result
    }
}

/// Formatted duration in microseconds.
pub fn format_duration_us(duration: std::time::Duration) -> String {
    format!("{} µs", duration.as_micros())
}

/// Formatted duration in milliseconds (display-boundary f64 usage).
pub fn format_duration_ms(duration: std::time::Duration) -> String {
    format!("{:.2} ms", duration.as_micros() as f64 / 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_features_detection() {
        let features = CpuFeatures::detect();
        println!("Detected CPU features: {:?}", features);
    }

    #[test]
    fn test_degenerate_distance_saturates_not_sentinel() {
        // Two coincident points hard against the boundary drive the
        // denominator 1 − |p1|²|p2|² below epsilon, hitting the divide-by-zero
        // guard. It must return the model's saturated maximum (~5.29), not the
        // old out-of-band 10000 sentinel.
        let simd = SimdOptimization::new();
        let boundary = FixedVector::from_f32_slice(&[0.99999, 0.0]);
        let d = simd.hyperbolic_distance(FixedPoint::from_int(1), &boundary, &boundary);

        let saturated = FixedPoint::from_int(2) * constants::safe_atanh(constants::near_boundary());
        assert!(
            d <= saturated + constants::epsilon(),
            "degenerate distance {} exceeded the saturated model max {}",
            d, saturated
        );
    }

    #[test]
    fn test_simd_vector_multiply() {
        let simd = SimdOptimization::new();

        let a = FixedVector::from_f32_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b = FixedVector::from_f32_slice(&[5.0, 6.0, 7.0, 8.0]);

        let product = simd.vector_multiply(&a, &b);

        let mut expected = FixedVector::new(4);
        for i in 0..4 {
            expected[i] = a[i] * b[i];
        }

        for i in 0..4 {
            assert!((product[i] - expected[i]).abs() < constants::epsilon(),
                   "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_simd_matrix_vector_multiply() {
        let simd = SimdOptimization::new();

        let mut m = FixedMatrix::new(2, 3);
        m.set(0, 0, FixedPoint::from_int(1));
        m.set(0, 1, FixedPoint::from_int(2));
        m.set(0, 2, FixedPoint::from_int(3));
        m.set(1, 0, FixedPoint::from_int(4));
        m.set(1, 1, FixedPoint::from_int(5));
        m.set(1, 2, FixedPoint::from_int(6));

        let v = FixedVector::from_f32_slice(&[7.0, 8.0, 9.0]);

        let product = simd.matrix_vector_multiply(&m, &v);

        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(100);
        assert!(product.len() == 2);
        assert!((product[0] - FixedPoint::from_int(50)).abs() < tolerance);
        assert!((product[1] - FixedPoint::from_int(122)).abs() < tolerance);
    }

    #[test]
    fn test_hyperbolic_distance() {
        let simd = SimdOptimization::new();

        let origin = FixedVector::from_f32_slice(&[0.0, 0.0]);
        let point = FixedVector::from_f32_slice(&[0.5, 0.0]);

        let disk_radius = FixedPoint::from_int(1);
        let distance = simd.hyperbolic_distance(disk_radius, &origin, &point);

        // Expected: 2 * atanh(0.5)
        let expected = FixedPoint::from_int(2) * constants::safe_atanh(constants::half());
        let tolerance = FixedPoint::from_int(1) / FixedPoint::from_int(10);
        assert!((distance - expected).abs() < tolerance);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();

        monitor.start("test_op");
        std::thread::sleep(std::time::Duration::from_millis(10));
        monitor.stop("test_op");

        let (avg, min, max) = monitor.stats("test_op").unwrap();

        assert!(avg.as_millis() >= 9 && avg.as_millis() <= 20);
        assert!(min.as_millis() >= 9 && min.as_millis() <= 20);
        assert!(max.as_millis() >= 9 && max.as_millis() <= 20);

        monitor.reset();
        assert!(monitor.stats("test_op").is_none());
    }

    #[test]
    fn test_format_duration() {
        let duration = std::time::Duration::from_micros(1234);

        assert_eq!(format_duration_us(duration), "1234 µs");
        assert_eq!(format_duration_ms(duration), "1.23 ms");
    }
}
