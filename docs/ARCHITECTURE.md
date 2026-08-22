# Architecture

Read this before the code. It exists because the same questions kept being
re-derived from source — what the coordinate systems are, which one an index
serves, what "mode" means — and each rediscovery got at least one thing wrong.

Every claim here names the code that makes it true. Where the engine does not
do what an older document said, that is called out rather than smoothed over.

---

## 1. A node has up to three positions

This is the distinction everything else depends on, and the one most often
gotten backwards.

| | **structural** | **semantic** | **concept** |
|---|---|---|---|
| What | where the node sits in the tree's disk | the caller's meaning vector | derived place in a *second* disk |
| Type | `HyperbolicPoint`, `dimension` wide | `CompressedNode.semantic_coords: Vec<u8>` | `HyperbolicPoint`, derived per query |
| Set by | `put()` → `compute_child_placement` | `set_semantic()` | nothing — computed from the semantic dims |
| Stored in `.htt`? | **no**, replayed from topology | **yes**, Q64.64 per dim | no, derived |
| Served by | `cell_index` | semantic index (`metric_tree`) | `SemanticDisk` |
| API | `nearest`, `neighbors`, `find_within` | `nearest_semantic`, `neighbors_semantic` | `concept_of`, `SemanticDisk::nearest` |
| Header field | `dimension` | `semantic_dims` | — |

**They are never concatenated.** There is exactly one `point_map.insert` in the
engine and its point comes straight from `compute_child_placement`; nothing
builds a structural `HyperbolicPoint` out of semantic bytes. On load, `horon`
calls `store.put(...)` and `store.set_semantic(...)` as separate operations.

A query answers against exactly one of the three. Asking `nearest` about
semantic similarity, or `nearest_semantic` about tree structure, is a category
error rather than a tuning problem.

> `HTT_FORMAT.md` §7 previously described the reader as concatenating
> `[structural | semantic]` and indexing the result. No release has done that;
> the description was corrected in 2026-08.

### Worked example: the same file, both ways

A document at `/reports/2026/q3/audit.pdf` in a store with `semantic_dims = 24`.

**Structural** — *where it sits in the tree.* You never choose this. It is a
function of the path and the node's sibling index: depth 4, in the branch under
`/reports/2026/q3`, at hyperbolic distance τ from its parent. Two nodes close in
structural distance are close **in the hierarchy**.

```rust
store.put("/reports/2026/q3/audit.pdf", bytes)?;   // placement happens here
store.neighbors("/reports/2026/q3/audit.pdf", 5)?; // -> siblings, parent
```

**Semantic** — *what it is about.* You choose every one of these, and every one
is stored and queryable:

```rust
// dims 16.. are yours; 0-15 are reserved for access bands
store.set_semantic("/reports/2026/q3/audit.pdf", coords(&[
    0.95,  // 16: finance
    0.40,  // 17: legal
    0.02,  // 18: engineering
    0.80,  // 19: audience = executive
    0.70,  // 20: urgency
    // ... up to 238 more axes, all meaningful
]))?;
store.nearest_semantic(&query, 10, 16..21)?;  // -> similar documents, anywhere
```

Structural distance says *this sits beside the Q3 revenue sheet*. Semantic
distance says *this is about the same things as last year's tax filing*, which
lives in a completely different branch. Neither is a worse version of the other.

**Where it gets interesting.** Configure a concept taxonomy and the semantic
vector produces a *third* position — in its own disk — and disagreement between
the two becomes a signal:

```rust
let disk = SemanticDisk::build(&[("/finance", 16), ("/legal", 17), ...])?;
disk.concept_of(&store, "/misc/scan_0042.pdf")?;   // -> Some("/finance")
```

Filed under `/misc`, measures as finance. That mismatch is the
miscategorization primitive, and it only exists *because* the two coordinate
systems are independent. Collapsing them into one vector would destroy it.

### Semantic dimensions

Every semantic dimension is meaningful and every one is stored. Dims **0–15**
are reserved — six lo/hi access bands (read, write, exec, domain,
classification, identity) plus four spare — and **16+** are yours.

`MAX_HILBERT_DIMS = 8` in `horon` chooses how many of the user dims drive the
space-filling curve for *disk ordering* in meaning-addressed files. It does not
limit what is stored, indexed, or queryable.

### Structural placement is planar — and why that is not a limit on meaning

This is the point most likely to look wrong, so it is worth stating carefully.

**Semantic width is not capped at 2.** It is capped at 255, every dimension is
stored, and every dimension is queryable. The "place data on coordinates you
choose, and the coordinates become meaning" model is exactly right — that is
what `semantic_dims` is for.

**Structural width is a different question**, because structural coordinates do
not carry user meaning. They carry *tree position*, which is fully determined by
`(path, sibling index)`. A tree has no more structure to express than that, so a
third structural axis would not let it express more: Sarkar embeds any tree in
the hyperbolic **plane** with (1+ε) distortion, and adding axes does not reduce
that distortion.

The one real argument for a third structural axis is packing, not
expressiveness: children compete for angular room, and on a sphere they separate
like `1/√k` rather than `1/k`, which lowers the τ a given fan-out needs — worth
up to 1.75 in τ at degree 16. See `GEOMETRY_TRACK.md`.

So: `dimension` is not your semantic budget. If you set it to 23 expecting 23
axes of meaning, you have configured 23 *structural* components — 21 of which
are always zero — and left `semantic_dims` untouched. That is a naming problem
in the API, and the fix is to read this section, not to narrow the field.

Regardless of what `dimension` says, structural coordinates only ever use the
first two components. The root is at the origin; `compute_child_placement`
writes only `[0]` and `[1]`; `mobius_add` is a per-component linear combination,
so a dimension that is zero in both inputs stays zero; `ensure_in_disk` scales
uniformly. By induction every structural point has dims ≥2 exactly zero, at any
depth. Measured across 1 281 nodes in four tree shapes: zero off-plane
components.

Sarkar embeds trees in the hyperbolic **plane**; that is a property of the
construction, not an implementation shortcut. See `GEOMETRY_TRACK.md` for what
it would take to change, and why it might be worth it.

---

## 2. The three things called "modes"

They are independent switches at two different layers and compose freely.

**Per node — embedded or data-only** (engine). `put()` embeds: the node gets
structural coordinates and enters the spatial index. `put_data_only()` skips
that entirely — data and semantic coordinates only. Semantic queries still work;
spatial queries will never see it. This is the fast bulk-load path.

**Per file — ordinary or meaning-addressed** (format, flag bit 6 → v3). When
set, a bounds section follows the header and snapshot entries are written in
global **Hilbert order** over the first ≤8 user dims, so a node's byte offset is
a function of its semantic coordinates and similar content lands in nearby
pages. Requires an uncompressed snapshot. When clear, entries are in ordinary
order and disk placement is unrelated to content.

**Per file — semantic tail encoding** (format, flag bit 7 → v4). Quantised
tails: user dims as TQ1.9, 2 bytes/dim instead of 16.

Plus GACL enforcement (bit 5), which is orthogonal to all three.

---

## 3. Layers

```
Store                    path-keyed CRUD, the public surface
  HTTStorage             validation, key normalisation
    HyperbolicTreeTensor tree shape, parent/child, path -> node
      HyperbolicTensorNetwork
        point_map        AUTHORITY: unique_id -> structural position
        nodes            data, metadata, semantic_coords
        cell_index       derived: structural spatial index — every spatial read
        semantic_index   derived: epoch-cached per-slice VP-trees
```

`hash_table` is no longer a layer. Since 0.6.0 the module holds only
`GeometricSignature` — a node's identity — and nothing that answers a query.

`point_map` is the single source of truth for where a node is. Everything else
spatial is *derived* from it — which is exactly why the integrity check in §5
matters.

**Placement.** Children sit at hyperbolic distance τ from their parent
(`compute_child_placement`), at golden-angle-spaced angles, via a Möbius
translation from the origin frame. Past 256 siblings, **rainbow bands** move
further children onto concentric rings (`effective_tau = τ(1 + band/64)`) so
angular resolution renews rather than crowding one circle.

**Indexing.** `cell_index` computes a node's cell from its coordinates and
answers every spatial query exactly. No transcendental appears on any per-node
or per-cell path. Detailed in §4 — read it before changing anything there.

**Semantic.** Slice queries use a `MetricVpTree` cached against the store's
semantic epoch, rebuilt lazily on invalidation. `SemanticDisk` Sarkar-embeds a
concept taxonomy into its own disk and derives each node's position there as a
weighted Klein barycenter of the anchors.

---

## 4. The cell index: how a spatial answer is actually produced

Every spatial query — `nearest`, `nearest_k`, `neighbors`, `find_within` — is
answered here. It replaced two indexes that were each structurally unable to
answer correctly, so the design is worth understanding rather than trusting.

### The shape

The disk is cut into **radial bands** × **angular sectors**. A node's cell is a
pure function of its coordinates, so insertion is a computation, not a search.

**Bands** come from a threshold table on the *squared* Euclidean norm. The
hyperbolic radius of a point is `2·artanh(‖p‖)`, and `artanh` costs 16.1 µs —
unaffordable per node. So the table is defined *as* the thresholds: band `b`
ends where `‖p‖ = tanh(b·W/2)`, and the squared value is precomputed at build
time. Finding a band is then a comparison against a small sorted table, with
neither `artanh` nor `sqrt` on the query path.

**Sectors** come from a **diamond pseudo-angle** — a monotone function of the
true angle costing one division, against `atan2`'s 33.2 µs. It is not the
angle; it only has to *order* like one, which is all a sector assignment needs.
Sector count grows per band (`≈ 2π·sinh(r)/arc`), because hyperbolic
circumference grows exponentially and a fixed count would make outer cells
enormous. It is capped at `2²⁸`, because `sinh` passes `i32::MAX` around band 39.

### Why it is exact

Cell membership is a heuristic. **Correctness comes from the bound, not the
cells.** A query reads its own cell, then walks rings outward, and each
unvisited cell is tested against a lower bound on the distance to *anything it
could possibly hold*. Expansion stops only when the bound rules out every
remaining cell — never on a candidate cap, a window, or a result count.

The radial bound follows from `d ≥ |r_q − r_p|` through the origin:

```
cosh(r_q − r_edge) = cosh r_q · cosh r_edge − sinh r_q · sinh r_edge
```

The full cell bound is the hyperbolic law of cosines minimised over the band's
radius range, using the *minimum angular gap* to the sector. Both are expressed
in `cosh d` rather than `d`, so no `atanh` is needed to compare them.

This is why a badly sized cell costs scan time and nothing else — and why a τ
below `PROOF.md`'s bound is now a spacing-quality issue rather than a
correctness one (§7).

### Where the bodies are buried

Eight defects have been found in this bound. Every one was a *numerical* or
*ordering* mistake, not an algebraic one, and they cluster into four lessons
worth carrying to any similar code:

| Lesson | What went wrong |
|---|---|
| A bound needs the **maximum** slope, never the average | The pseudo-angle→angle conversion used an average slope; the exact maximum is `dp/dθ = 1/(cos+sin)² = 1`. 107 821 violations. |
| Circular quantities must be wrapped **per operand** | The angular gap was computed linearly then wrapped once. A query at 0.1 against sector [3,4) scored 1.1 instead of 0.1. 3 592 violations. |
| **Never square a small quantity** | `sinh r_q` as `4‖q‖²/(1−‖q‖²)²` leaves under 3 significant digits at radius 20. The first-power form `2‖q‖/(1−‖q‖²)` is exact where the squared one was wrong by 6e5. |
| **Never square a large one either** | `cosh d` passes 3e9 around radius 22, so `v > limit²` wraps Q64.64. `Bound::Squared` divides instead: `cosh_d ≥ v / cosh_d`. |

Two more were structural: sector aliasing that only appears at `k > 1` (0/300
wrong at k=1, then failures at k=5 — **never test k=1 alone**), and an early
`break` over sectors that assumed monotonicity the bounds do not have, fixed
with a monotone *envelope* break plus a per-sector `continue`.

The eighth was not in the algebra at all: `within_radius` fed a caller-supplied
radius into `cosh`, which **panics** rather than saturating in Q64.64. It took
the query down for any radius past ~44, and "give me everything" is a normal
way to call it. The ceiling is now optional; its absence means *prune nothing*.

`cell_index::every_cell_bound_is_a_true_lower_bound` checks >10 000 pairs with a
quarter of queries in the deep regime, and caught six of the eight.
`tests/adversarial_inputs.rs` covers the caller-supplied extremes that caught
the eighth. **Both are needed; neither substitutes for the other.**

---

## 5. Integrity: referential vs functional

Two different questions, and only one of them was being asked.

**Referential** — do these maps point at things that exist?
`validate_network`: `nodes ↔ point_map` both ways, child signatures resolve to
real nodes, `child_counts ⊆ nodes`.

**Functional** — can the index answer a query about what it holds?
`verify_index_locates_all_nodes`: query at each node's own stored position and
require distance zero back, because distance 0 is the global minimum of a metric
and the expected answer is known without any oracle.

A structure can pass every referential check and still find nothing. That is not
hypothetical: the bucket layer was referentially perfect while `nearest`
returned the wrong node for 25 of 42 nodes in a deep tree, and nothing asked it
to locate a node it had itself indexed. The functional check now runs inside
`validate_network`.

---

## 6. What is legacy

**Removed in 0.6.0** — the exact-Voronoi machinery, replaced by `CellIndex`:

- **`PointLocationGrid`** — one owner per tile, but cells fall below tile size
  within a few levels; in the exact diagram 34 of 43 nodes own no tile, none
  deeper than 2. Not repairable at any affordable resolution. Its
  `grid_resolution` config knob is gone with it.
- **`PowerCell` / `HalfPlane` / `compute_bisector` / `point_in_cell`** —
  `compute_bisector` ran twice per insert and its `normal`/`offset` were never
  read in production; every real read used `neighbor_id`, which is parent +
  children and already in the tree. The parent it derived for deletion was a
  fallback the production path never took — `tree_tensor::delete` resolves the
  parent from `path_map`, which is authoritative.
- **`klein_points`** — a per-node Klein cache that existed only to feed the
  bisectors. Callers wanting Klein coordinates can convert on demand:
  `poincare_to_klein(&network.get_point(id)?)`.

- **The bucket layer** — `HyperbolicHashTable`, `HyperbolicRegion`,
  `HyperbolicHashBucket`, `BucketEntry` and the per-bucket `VPTree`. Measured
  reason: **0 of 57 buckets could contain any node.** 56 centres sit off the
  data plane (median 71.7% of their norm in dims 2–3) while placement is
  planar, so every node was assigned by a half-space sign test rather than by
  containment. It also charged one **exact** 37.5 µs hyperbolic distance per
  insert to keep a bucket's pruning radius current.

Signature creation moved to `GeometricSignature::embedded(point, dimension,
level)`, which consults no index at all. `HyperbolicTensorNetwork` now owns its
own `PoincareDisk` — it only ever wanted the dimension and the origin.

**What stayed, and why.** `KleinPoint`, `poincare_to_klein`,
`klein_to_poincare` and `power_distance`: `SemanticDisk` is built on the Klein
model, and `power_distance` is a documented primitive with a tested caveat —
it equals the hyperbolic diagram only when all sites share one Klein norm,
which is exactly the condition whose absence caused the 0.5.2 defect. It has no
production caller; unreferenced is not the same as dead.

**The name.** `hash_table` no longer holds a hash table. It keeps the name for
0.6.0 so `horon_engine::hash_table::GeometricSignature` does not move twice —
a rename is a separate decision, alongside `tensor_network`, which has never
contained a tensor.

---

## 7. Limits, and where they come from

**Depth is capped, and the cap is enforced.** A node sits at hyperbolic radius
`depth × τ`, and coordinate differences shrink like `e^(−r)`. In Q64.64,
`‖p−q‖²` underflows once `‖p−q‖ < 2⁻³²`, after which the distance kernel returns
its saturation value for every pair — every node equidistant, ranking arbitrary.

Placement beyond `constants::max_safe_radius()` (**21**) is **refused**. Past
that point queries do not get slower, they get wrong, and a wrong answer that
looks like an answer is the failure this engine has already paid for once.

```rust
store.max_depth()   // 21 at the default tau = 1.0
                    // 26 at tau = 0.8, 10 at tau = 2.0
```

Fidelity degrades gradually *before* the hard limit. Measured error in a unit
step along a geodesic:

| radius | step error |
|--------|------------|
| 16     | 3.4e-8     |
| 18     | 8.9e-6     |
| 20     | 2.0e-4     |
| 22     | 4.5e-3     |
| 24     | saturated  |

So depth ≈17 at τ=1 is where full fidelity ends, and 21 is where meaning ends.
Between them, ranking still works and absolute distances drift. Choose τ for
your fan-out and read `max_depth()` before designing a deep hierarchy — a
smaller τ buys depth, a larger one spends it.

**τ and fan-out.** `PROOF.md`'s Delaunay theorem requires
`τ ≥ −log(tan(π/(2·d_max)))`. The default τ=1.0 satisfies that only to
`d_max ≈ 4.46`, while rainbow fan-out targets 256 children (needing τ ≥ 5.094).
Nothing enforces it. Since no query path depends on the Delaunay identity, a τ
below the bound costs spacing quality — more nodes per cell, longer scans — not
correctness. `GEOMETRY_TRACK.md` covers the forward plan.

**Not implemented.** There is no tensor compression anywhere: `CompressedNode`
stores raw bytes and says so in its own docstring. The `tensor` in the crate's
lineage is historical.

---

## 8. Reading order

1. This file.
2. `PROOF.md` — the placement theorem and, in *Implementation status*, where the
   code stands against it.
3. `GEOMETRY_TRACK.md` — τ, cone placement, dimension: findings without a
   release attached.
4. `SEMANTIC_DISK.md`, `SEMANTIC_INDEX.md` — the meaning-side layers.
5. `../horon/docs/HTT_FORMAT.md` — the on-disk format.
