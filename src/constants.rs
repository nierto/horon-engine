//! Shared fixed-point constants for the engine.
//!
//! All constants are constructed as exact rational values using
//! FixedPoint integer division. No floats are used.

use g_math::fixed_point::FixedPoint;

/// Below this many nodes, `nearest_semantic` stays on the brute-force scan:
/// a VP-tree build costs O(n log n) distance evaluations, which small stores
/// never amortize (see `docs/SEMANTIC_INDEX.md`).
pub const SEMANTIC_INDEX_MIN_NODES: usize = 256;

/// Maximum number of dimension slices the semantic index caches at once.
/// Eviction is deterministic (lowest key first), not wall-clock LRU.
pub const SEMANTIC_INDEX_MAX_SLICES: usize = 16;

/// Minimum population for `find_outliers`: z-scores over fewer nodes
/// are statistically meaningless, so smaller populations return no outliers.
pub const OUTLIER_MIN_POPULATION: usize = 5;

/// Neighborhood size for `find_outliers`: each node's outlier score is
/// its average distance to this many nearest peers (capped at population−1).
pub const OUTLIER_KNN: usize = 10;

/// 0.5 = 1/2
#[inline]
pub fn half() -> FixedPoint {
    FixedPoint::from_int(1) / FixedPoint::from_int(2)
}

/// Largest norm/ratio the model represents: `1 − 10⁻¹²`.
///
/// This bounds the greatest expressible hyperbolic distance at
/// `2·atanh(1 − 10⁻¹²) ≈ 28.3`, which with the default τ = 1 is roughly 27
/// levels of nesting. The limit is a deliberate margin, not an arithmetic
/// one: `atanh`/`tanh` round-trip at 0 ULP out to `1 − 10⁻¹⁸`, and `1 − r²`
/// stays representable until about `1 − 10⁻¹⁹`, so this sits six orders of
/// magnitude clear of the floor.
///
/// It was 0.99 until 2026-07-30, which capped distance at 5.29 — about seven
/// levels — and, worse, made `ensure_in_disk` collapse anything deeper back
/// onto this value, so norms cycled instead of growing. See
/// `docs/HYPERBOLIC_INDEX.md`.
#[inline]
pub fn near_boundary() -> FixedPoint {
    FixedPoint::from_str("0.999999999999")
}

/// 0.99999 = 99999/100000 — tanh saturation bound
#[inline]
pub fn near_one() -> FixedPoint {
    FixedPoint::from_int(99999) / FixedPoint::from_int(100000)
}

/// 0.0001 = 1/10000 — standard epsilon for near-zero checks
#[inline]
pub fn epsilon() -> FixedPoint {
    FixedPoint::from_int(1) / FixedPoint::from_int(10000)
}

/// 0.00001 = 1/100000 — small epsilon for origin detection
#[inline]
pub fn small_epsilon() -> FixedPoint {
    FixedPoint::from_int(1) / FixedPoint::from_int(100000)
}

/// How close to the boundary a point may sit before it is projected back in:
/// `ensure_in_disk` rescales when `1 − ‖p‖² < 10⁻¹²`.
///
/// Matched to [`near_boundary`] so a rescaled point lands *inside* the
/// margin and does not immediately re-trigger. Was 1/1000, which fired at
/// ‖p‖ ≈ 0.9995 — reachable by depth 8 — and rescaled all the way back to
/// 0.99, producing an observable 4-cycle in the norms
/// (0.99 → 0.9963 → 0.9986 → 0.9995 → 0.99) rather than a depth limit.
#[inline]
pub fn boundary_margin() -> FixedPoint {
    FixedPoint::from_str("0.000000000001")
}

/// Below this, the Möbius denominator `|1 − p̄q|²` is treated as degenerate.
///
/// Purely a division-safety bound, not a geometric one. `dist_sq ≤ 4` for any
/// two points in the disk, so the quotient stays inside Q64.64 as long as the
/// denominator exceeds `4 / (2⁶³−1) ≈ 8` ULP; 16 ULP doubles that margin.
///
/// This was `epsilon()² = 10⁻⁸` until 2026-07-30 — ten orders of magnitude
/// above the real floor. Since `|1 − p̄q|² = (1 − ‖p‖²)² + ‖p − q‖²` for
/// points at equal radius, ordinary sibling geometry at depth 11 evaluates to
/// ~3.5e-9 and tripped the guard, making the kernel return the saturation
/// value for *every* pair from that depth on. Every node became equidistant,
/// so nearest-neighbour ranking became arbitrary — the real cause of what
/// looked like a spatial-index limit.
#[inline]
pub fn min_safe_denominator() -> FixedPoint {
    FixedPoint::from_raw(16)
}

/// 0.3 = 3/10 — region radius for hash table origin bucket
#[inline]
pub fn region_radius() -> FixedPoint {
    FixedPoint::from_int(3) / FixedPoint::from_int(10)
}

/// Pi, parsed from string for maximum precision
#[inline]
pub fn pi() -> FixedPoint {
    // 20 digits of pi — enough for any gMath profile
    FixedPoint::from_str("3.14159265358979323846")
}

/// Two * pi
#[inline]
pub fn two_pi() -> FixedPoint {
    FixedPoint::from_int(2) * pi()
}

/// Golden angle = pi * (3 - sqrt(5))
#[inline]
pub fn golden_angle() -> FixedPoint {
    pi() * (FixedPoint::from_int(3) - FixedPoint::from_int(5).sqrt())
}

/// Safe atanh: clamps input to (-0.99, 0.99) then calls gMath's .atanh()
#[inline]
pub fn safe_atanh(x: FixedPoint) -> FixedPoint {
    let max = near_boundary();
    let clamped = if x > max {
        max
    } else if x < -max {
        -max
    } else {
        x
    };
    clamped.atanh()
}

/// Default Sarkar embedding scale factor τ = 1.0.
/// Controls parent-child hyperbolic distance. Q64.64 supports depth ~44/τ.
#[inline]
pub fn default_tau() -> FixedPoint {
    FixedPoint::from_int(1)
}

/// Quantization helper: converts a FixedPoint to an i32 by multiplying
/// by 1000 and rounding to the nearest integer.
/// Replaces the pattern `(x.to_f32() * 1000.0).round() as i32`
#[inline]
pub fn quantize_1000(x: FixedPoint) -> i32 {
    let scaled = x * FixedPoint::from_int(1000);
    // Round to nearest integer: add 0.5 (or subtract 0.5 if negative) then truncate
    let rounded = if scaled.is_negative() {
        scaled - half()
    } else {
        scaled + half()
    };
    rounded.to_int()
}

/// High-resolution quantization for node-identity position signatures:
/// 2^20 steps per unit (coordinates live inside the unit disk, so the
/// result fits an i32 with room to spare).
///
/// Node identity must NOT use `quantize_1000`: at 1/1000 resolution two
/// depth-2 cousins in *different branches* can quantize to identical
/// signatures at a few hundred nodes (silent data
/// crossover). 2^-20 resolution pushes the same birthday bound past
/// millions of nodes, and rainbow fan-out banding guarantees the true
/// positions are distinct.
#[inline]
pub fn quantize_position(x: FixedPoint) -> i32 {
    let scaled = x * FixedPoint::from_int(1 << 20);
    let rounded = if scaled.is_negative() {
        scaled - half()
    } else {
        scaled + half()
    };
    rounded.to_int()
}
