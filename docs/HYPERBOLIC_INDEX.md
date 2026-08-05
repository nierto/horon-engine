# Hyperbolic Index: radius-uniform sparse cells

Status: **design, not implemented** — and no longer urgent. The depth limit
this document was written to explain turned out to be two over-conservative
constants, both fixed on 2026-07-30 (see *What actually limited depth*
below). Usable nesting went from 7 to 21 with no index changes at all. What
remains here is a **scaling** design for large trees, not a correctness fix.

No file-format implications either way — geometric signatures are in-memory
only and never reach a `.htt` (verified: neither `format.rs`, `snapshot.rs`
nor `wal.rs` serialize them). Confirmed empirically: after changing the
geometry constants, every Horon determinism golden and all 15 conformance
corpus cases passed unchanged.

## What actually limited depth

The original diagnosis in this document blamed the bucket partition. That was
wrong, and the way it was wrong is worth recording: the partition *is*
structurally unsuited to deep trees (see below), which made it a plausible
culprit, but it was not what produced the observed cliff.

Extending the shells from hyperbolic radius 2.4 out to 12 — a 5× increase in
coverage, 24 shells instead of 5 — changed the measured depth **not at all**.
That negative result forced a comparison of the distance kernel against a
reference implementation of the same formula:

| depth | kernel d(a,b) | reference d(a,b) |
|---|---|---|
| 10 | 1.89439788 | 1.89439788 |
| 11 | **28.32416825** | 1.89439788 |
| 15 | **28.32416825** | 1.89439788 |

From depth 11 the kernel returned the *same saturation constant for every
pair*. Nothing was mis-ranked and nothing was missing from the index — every
node was reported equidistant, so nearest-neighbour ordering was arbitrary.

Two constants caused it:

**1. The degenerate-denominator guard, `epsilon()² = 1e-8`.** For two points
at equal radius the Möbius denominator is exactly
`(1 − ‖p‖²)² + ‖p − q‖²`, which shrinks quadratically with depth. Ordinary
sibling geometry evaluates to 2.6e-8 at depth 10 and **3.5e-9 at depth 11** —
below the guard, which then returned the saturation value. The guard existed
to prevent division overflow, a condition that needs only ~8 ULP of headroom
(`dist_sq ≤ 4`, quotient must stay under 2⁶³). It was set ten orders of
magnitude too high. Now `min_safe_denominator() = 16` ULP.

**2. The boundary clamp, `near_boundary() = 0.99`.** This capped the greatest
expressible distance at `2·atanh(0.99) ≈ 5.29`, about seven levels — and
`boundary_margin() = 1e-3` made `ensure_in_disk` rescale anything deeper back
*onto* 0.99, so norms cycled (0.99 → 0.9963 → 0.9986 → 0.9995 → 0.99) rather
than approaching the boundary. Both are now `1e-12`, six orders clear of the
measured arithmetic floor (`atanh`/`tanh` round-trip at 0 ULP to `1 − 1e-18`;
`1 − r²` stays representable to about `1 − 1e-19`).

Measured effect, cumulative — the two changes are **interdependent**, and
neither alone is worth much:

| configuration | deepest usable depth |
|---|---|
| as shipped before 2026-07-30 | 7 |
| guard alone | 7 |
| clamp alone | 10 |
| **guard + clamp (current)** | **21** |

The ceiling is now the guard again, at its new threshold, and it is a real
arithmetic bound rather than an arbitrary one: pushing to 4 ULP buys depth 22
at the cost of the overflow margin. 21 is where the headroom runs out.

## Why the current partition is still wrong for large trees

Three structural problems remain. None of them limits *depth* — that was the
misdiagnosis — but each limits *scale*, and together they mean bucket quality
degrades as a tree grows outward.

**1. The shells stop at hyperbolic radius 2.4.** `initialize_buckets` places
centers at `{0, 0.5, 1.0, 1.5, 2.0}` with radius `1/5 + 1/10·d`. A node at
depth *n* sits at radius ≈ *n·τ*, so with τ=1 the outermost shell is exceeded
by depth 2. Everything past it lands in whichever bucket
`effective_radius()` has been widened enough to swallow, and that widening
destroys the pruning bound: a bucket grown to cover half the tree can never
be excluded. Queries stay *correct* — the fallback degenerates to a scan —
but the O(1) bucket-selection claim does not survive it.

**2. Angular capacity grows linearly where the space grows exponentially.**
Directions per shell are `dimension × {2,3,4,5}` — linear in shell index —
while the circumference of a hyperbolic circle of radius *r* grows as
`2π·sinh(r)`. Covering radius 12 with radius-1.4 balls needs roughly 365,000
directions; radius 20 needs ~10⁸. A preallocated partition cannot cover the
space at any depth worth having, which is the core argument for lazy cells.

**3. The signature quantizes Euclidean coordinates.**
`compute_geometric_signature` computes `x·(1 + tanh x)` per axis and rounds on
a 1/1000 lattice. The lattice is uniform in Euclidean coordinates while tree
nodes crowd exponentially toward the boundary (at depth 11 siblings are
5.4e-5 apart, 20× below one quantization step), and the transform *saturates*
— as `x → 1`, `x·(1 + tanh x) → 1.7616` — so the signature's output range
compresses exactly where node density is highest.

## Design

Replace the fixed partition with **sparse cells that are uniform in
hyperbolic measure and materialize only where nodes exist.**

### Cell coordinates

A point `p` is addressed by `(shell, direction)`:

```
r_h    = 2·atanh(‖p‖)                  hyperbolic radius from the origin
shell  = floor(r_h / δ)                δ = shell width, default τ/2
u      = p / ‖p‖                       unit direction (dimension components)
sector = quantize(u, ANGULAR_BITS)     per-component fixed-point quantization
key    = (shell, sector)
```

`r_h` is the natural radial coordinate: a step of δ is the same hyperbolic
distance at every radius, near the origin and at ‖p‖ = 0.99999 alike. This is
the single substantive change — the current code quantizes ‖p‖-space, which
compresses; `r_h`-space does not.

Angular resolution is uniform on the unit sphere, which is legitimate
*because the sphere does not crowd* — but sibling subtrees do converge in
direction as depth grows (angular separation shrinks roughly as `e^{-r_h}`),
so `ANGULAR_BITS` must be generous. At depth 15 the separation is ~1e-6, so
20 bits per component is the floor; 24 gives headroom. The combinatorial cell
count is astronomically large and **irrelevant**: cells are created lazily, so
memory tracks node count, never cell count.

### Lazy materialization

Cells are entries in a `HashMap<(u32, Vec<i32>), Cell>` created on first
insert. A cell holds its member nodes plus the two quantities the search
needs: a representative center (the first-inserted point, kept stable) and the
true maximum hyperbolic distance from that center to any member. That max is
the pruning radius, and unlike today's `effective_radius` it is *tight* —
cells are small by construction, so it never widens into uselessness.

Cells split when they exceed a capacity threshold, by increasing angular
resolution locally (one more bit) rather than by rebalancing globally.

### Query

Two levels, both hyperbolic:

1. **Radial band.** For a query at radius `r_q` seeking distance `≤ R`, only
   shells intersecting `[r_q − R, r_q + R]` can contribute — `O(R/δ)` shells,
   independent of node count. This is the pruning the current design lacks
   entirely.
2. **Angular window within a shell.** At radius `r`, an angular separation
   `Δθ` corresponds to arc length `≈ sinh(r)·Δθ`, so a hyperbolic radius `R`
   admits `Δθ ≤ R / sinh(r)` — a window that *narrows* exponentially with
   depth. Deep queries therefore touch very few cells, which is the whole
   point: the exponential growth of the space becomes a pruning asset instead
   of a scaling problem.

Candidate cells are then ranked by `d(q, center) − max_member_distance` and
scanned with the existing branch-and-bound early termination, which is
already correct and can be reused unchanged.

For k-NN without a radius bound, start with `R = δ` and double until k
candidates are found — `O(log)` rounds, each cheap.

### Determinism

Everything above is fixed-point:

- `atanh` is exact to **0 ULP** through `1 − r = 1e-18` (measured), so `r_h`
  is safe far past any depth the tree will reach.
- `shell` is an integer floor of a fixed-point quotient — exact.
- Angular quantization is a fixed-point multiply and round, as
  `quantize_position` already does.
- No floating point anywhere; cell assignment is a pure function of the
  point, so two machines index identically.

The one ordering hazard: cell iteration must never depend on `HashMap` order.
Sort candidates by `(pruning_bound, shell, sector)` — the same
deterministic tie-break discipline `find_bucket` uses today.

### Depth is no longer this design's problem

The constants described above are already fixed, so a new index inherits
depth 21 rather than having to earn it. What this design must not do is
*lose* it: the acceptance test below is the existing `tests/depth.rs`
continuing to pass unchanged.

## Test plan

1. **Depth sweep** — `tests/depth.rs` already pins siblings as mutually
   discoverable to depth 20, norms as monotonically increasing, and the
   kernel as non-saturating. A new index must keep all three passing. This is
   the regression floor, not the acceptance bar.
2. **Exactness** — the k-NN returned by the index must equal brute-force k-NN
   over all nodes under the same metric. Equality, not approximation: the
   index may only prune, never approximate. Already covered by
   `nearest_neighbours_stay_exact_at_depth`, and — worth stating plainly —
   **the current index passes it**: 100% agreement with brute force at depths
   4, 8, 12, 16, 18, 20, 21 and 22, measured on trees with fan-out so each
   neighbour slot has real competitors. A replacement must preserve that, and
   the bar is equality on every query, not a hit rate.
3. **Determinism** — same insert script, two processes, identical cell
   assignment for every node; and the existing golden-CRC suite must be
   unaffected (it should be, since signatures never reach disk).
4. **Distribution** — cell occupancy histogram across a deep tree; no cell
   should hold a constant fraction of the tree, which is the failure mode
   today.
5. **Cost** — query time versus depth must stay flat. If it grows with depth,
   the angular window is not narrowing as designed.

## Open questions

- **Shell width δ.** τ/2 makes each level straddle two shells, which bounds
  the radial band at 2–3 shells for a sibling query. Worth measuring against
  δ = τ and δ = τ/4.
- **Angular quantization for dimension > 4.** Per-component quantization of a
  unit vector wastes resolution (most of the cube is off-sphere). A proper
  spherical code would be tighter, but adds a table dependency; per-component
  is the honest first implementation.
- **Cell splitting policy.** Local angular refinement is simple but produces
  mixed resolutions within a shell, so the query must consider both. The
  alternative — fixed high resolution, never split — may simply be better
  given that cells are sparse.
- **Migration.** In-memory only, so a version bump is unnecessary; but any
  benchmark quoting bucket counts or `find_bucket` timings becomes stale.
- **The root is returned but cannot be ranked.** `list("/")` includes `/` and
  `neighbors()` returns it as a neighbour, but `position("/")` errors — the
  format spec says the root is not a node, and two of three APIs disagree.
  Any exactness test has to exclude it from both sides, which is a smell. A
  new index should decide the question rather than inherit it: either the root
  is a positioned node at the origin, or it is not returned by queries.
- **Sibling separation steps at depth 16.** Measured separation holds at
  1.89439788 through depth 15, then changes to 1.45569705 and stays there
  through depth 21. A single discrete step, stable either side of it — a
  change in placement geometry, not precision decay (precision loss would
  drift gradually and worsen). Both values discriminate comfortably, so this
  is a quality question rather than a correctness one, but the mechanism is
  unexplained and worth finding: it likely indicates a branch in the Sarkar
  child-placement angle selection that only engages past a certain radius.

## Non-goals

**Switching to the hyperboloid model.** The measurement shows the Poincaré
model returns exact, constant distances at every depth tested — the model is
not what fails. The hyperboloid would need its own spatial index facing the
identical question this document answers, would trade the current
near-boundary behaviour for catastrophic cancellation in `arcosh` for *nearby*
points (the common case in a tree), and its coordinates grow as `cosh(r)`,
reaching the Q64.64 integer range near radius 43 — a comparable ceiling, not
an unbounded one. It remains a defensible choice for a future major version
on other grounds; it is not a fix for this.
