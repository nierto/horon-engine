# Geometry track: the τ hypothesis, cone placement, and dimension

Status: **findings recorded, no code change planned for 0.6.0.**

This is the home for the placement-geometry work that is deliberately *not*
part of the cell-index release. It exists because the material kept falling out
of the index spec, which is about indexing, not about where nodes go.

Everything here is measured. Where a result is a prototype rather than a
property of the shipped engine, it says so.

---

## 1. The two coordinate systems are separate

This is the distinction the rest of the document depends on, and it is easy to
get backwards.

| | structural | semantic |
|---|---|---|
| Type | `HyperbolicPoint`, `dimension` wide | `CompressedNode.semantic_coords: Vec<u8>` |
| Source | Sarkar placement from tree topology | caller-supplied, via `set_semantic` |
| Stored in `.htt`? | no — replayed from topology | yes, Q64.64 per dim |
| Indexed by | the spatial index (`cell_index`) | semantic index, `SemanticDisk`, Hilbert |
| Header field | `dimension` | `semantic_dims` |

**They never share storage.** There is exactly one `point_map.insert` in the
engine, and its point comes straight from `compute_child_placement`; nothing
anywhere constructs a `HyperbolicPoint` from semantic bytes for the structural
index. On load, `horon` calls `store.put(...)` (which places structurally) and
`store.set_semantic(...)` (which does not) as separate operations.

**Every semantic dimension is meaningful**, and all of them are carried: dims
0–15 reserved (GACL bands), 16+ user-defined. The `MAX_HILBERT_DIMS = 8` cap in
`horon` applies only to *disk ordering* for meaning-addressed files — it
chooses which dims drive the space-filling curve, not which dims are stored.

`SemanticDisk` is a third thing again: it *derives* a `HyperbolicPoint` in a
**second, separate disk** as the weighted barycenter of concept anchors. That
position is not the structural one and is never indexed as such.

> **Doc/code discrepancy.** `HTT_FORMAT.md` §7 says the reader
> "Concatenates: full_point = [structural | semantic]" and builds the spatial
> index on the result. The code does not do this. The claim should be corrected
> or the behaviour implemented; today the spatial index sees structural
> coordinates only.

### What `dimension` actually costs

Structural placement is **provably planar**: the root is at the origin,
`compute_child_placement` writes only coordinates `[0]` and `[1]`, `mobius_add`
is a per-component linear combination so a dimension that is zero in both
inputs stays zero, and `ensure_in_disk` scales uniformly. By induction every
structural point has dims ≥2 exactly zero, at any depth, for any tree. Measured
across 1 281 nodes in four tree shapes: zero off-plane components.

So a `HyperbolicPoint` allocated `dimension` wide uses two of them. At the
`dimension = 23` some deployments use, that is 368 B per point of which 336 B
are structurally zero. **This is unused width in the structural point — not
discarded semantic data.** Semantic capacity is `semantic_dims`, a different
field, and it is unaffected.

Two consequences worth separating:

- **Memory**: 368 MB vs 32 MB at 1M nodes. Fixable by storing structural points
  at width 2 internally while keeping `dimension` in the header and API — a
  breaking change, since it alters what a public parameter means.
- **The bucket layer**: `1 + 14 × dimension` buckets, so 57 at the default 4 but
  323 at 23, and each query sorted all of them with the exact kernel. Removed in
  0.6.0 — the cell index has no per-dimension term.

---

## 2. The τ hypothesis, and a forward solution that breaks nothing

`PROOF.md`'s theorem is conditional:

```
tau >= -log(tan(pi / (2 * d_max')))
```

The engine ships a fixed `tau = 1.0`, which satisfies that only up to
`d_max ≈ 4.46`, while rainbow fan-out is built for 256 children per node —
needing `tau ≥ 5.094`. `CONTRACT.scn.md:89` declares the REQUIRES; nothing
checks it.

| d_max | τ required | usable depth at that τ |
|-------|-----------|------------------------|
| 4     | 0.881     | 19.9 |
| 6     | 1.317     | 13.3 |
| 16    | 2.318     | 7.6  |
| 256   | 5.094     | 3.4  |

(Depth from the measured Q64.64 budget: metric fidelity to hyperbolic radius
≈17.5, saturating near 22.)

### Why the obvious fixes don't work

**Raising τ globally** re-places every node, breaking "zero perturbation to
existing embeddings", the band-0 bit-identity promise, and the geometry of
every stored `.htt`.

**Per-node τ that grows with degree** (the rainbow mechanism, applied earlier)
does not satisfy the theorem. The hypothesis is on `d_max'` *after* insertion,
so a child placed when the parent had 2 children sits too close once the parent
has 20 — and it cannot be moved.

**Choosing τ at calibration** requires knowing the final maximum degree when the
tree is created, which is not knowable.

### The solution: retire the dependency, then be honest about the parameter

**The architectural half is already happening.** Once the cell index lands, no
query path depends on `Tree = Delaunay`. Correctness comes from a metric index
and a proven lower bound, not from the placement being Delaunay. A τ below the
theorem's threshold then stops being a *correctness* issue and becomes a
*spacing quality* issue: crowded siblings mean more nodes per cell and longer
scans — measurable, and never wrong.

That is the answer to "a solution that doesn't break our arch or promises": we
stop needing the promise that was being broken. Nothing is re-placed, no format
changes, and the guarantee that is dropped is one the engine was not delivering
anyway.

**The honesty half** is cheap and non-breaking:

1. **Validate at insert.** `max_degree(tau) = pi / (2 * arctan(e^-tau))` is one
   comparison against a value precomputed from the configured τ. When a parent's
   degree passes it, warn (or error under a strict flag). The engine then either
   operates inside its proven regime or says that it has left it — instead of
   silently exceeding a REQUIRES it declares.
2. **Document the table** so `StoreConfig::tau()` can be set deliberately for a
   known tree shape.

**If the walk is ever wanted** — an index-free `nearest` that navigates tree
edges, which is the paper's actual thesis — then real cone placement becomes
necessary, and that is a versioned placement change with a format decision
attached. That is a 0.7.0-and-beyond conversation, and §3 is the evidence for it.

---

## 3. Dimension and the cone: a refutation, corrected

An early sweep concluded that a 3-D (ball) embedding gave no reduction in the τ
required for an exact greedy walk. **That conclusion was wrong**, and the error
was in the construction, not the geometry.

Sarkar does **not** confine children to a narrow cone. Children are spread
uniformly with the **parent occupying one angular slot**, so every child sits at
least `2π/deg` from the parent direction and from every sibling; cone
containment is an emergent consequence of that plus adequate τ, not an input.
The first sweep tested two constructions that were both wrong — one confined
children to a half-plane (narrower than Sarkar), the other ignored the parent
direction entirely, so a child could be placed on top of it.

Re-run with the real construction, and with the 3-D analogue placing `deg`
points as a spherical code with the parent taking slot 0:

The **theorem bound** column counts `children + 1` points, because the parent
occupies one angular slot. `PROOF.md`'s table counts `d_max` points and so
reads slightly lower (2.318 vs 2.379 at 16 children) — same formula, one extra
point.

| children | theorem bound | 2-D τ | 3-D τ | saved |
|----------|---------------|-------|-------|-------|
| 2  | 0.549 | 1.25 | 1.25 | 0.00 |
| 3  | 0.881 | 2.00 | 1.75 | 0.25 |
| 4  | 1.124 | 2.25 | 2.00 | 0.25 |
| 6  | 1.477 | 3.00 | 2.25 | 0.75 |
| 8  | 1.735 | 3.50 | 2.75 | 0.75 |
| 12 | 2.108 | 4.25 | 3.00 | 1.25 |
| 16 | 2.379 | 5.00 | 3.25 | 1.75 |
| 24 | 2.766 | 5.00 | 3.75 | ≥1.25 |

τ is the smallest value at which greedy walk was exact for every node's own
position and for 4 000 random queries per configuration.

**The saving grows with degree**, which is what the packing argument predicts:
`k` points on a circle separate like `1/k`, on a sphere like `1/√k`. At 16
children that is usable depth 3.5 → 5.4 within the same precision budget.

### Caveats, stated plainly

- Measured τ still exceeds the theorem's own bound in **both** dimensions
  (children 6: bound 1.477, 2-D needs 3.00). So the construction is still not
  identical to Sarkar's — most likely because the cone is not narrowed
  recursively with depth. The *relative* 2-D/3-D comparison is sound; the
  absolute values are upper bounds on what a correct construction would need.
- The nn-violation metric used alongside this is unsound at high degree: in a
  star, siblings are each other's nearest neighbours but are not tree-adjacent,
  so violations are expected even in a perfect embedding.
- All of §3 is an f64 prototype, not the shipped engine.

**What this changes:** if placement is ever revisited, going 3-D is worth real
consideration rather than dismissal. It buys depth at exactly the fan-out where
the current τ is most badly violated. It has no bearing on the cell index, which
is metric-space generic and does not care how many dimensions placement uses.

---

## 4. How many dimensions? (and why not five)

A recurring intuition is that dimension 5 is special, because the **Euclidean
unit ball's volume peaks there** (`V₅ = 5.264`, falling after; the unit
sphere's *surface area* peaks at n=7). The fact is real. It does not apply
here, for two independent reasons.

**It is a units artifact.** `V_n` has units of lengthⁿ, so comparing `V₅` to
`V₄` compares a 5-dimensional measure with a 4-dimensional one. Fix a different
radius and the peak moves:

| radius | volume peaks at |
|---|---|
| 0.9 | n = 4 |
| 1.0 | n = 5 |
| 1.1 | n = 7 |
| 1.5 | n = 13 |
| 2.0 | n = 24 |

"Five" is a property of choosing r = 1, not of space.

**It is the wrong quantity.** Sarkar placement does not care how much volume a
ball holds. It cares about the **minimum angular separation** between the
directions leaving a node, because that is what the τ bound is a function of.
Separation on `S^(n-1)` improves *monotonically* with n and never peaks —
k points separate like `1/k` on a circle, `1/√k` on a sphere, `1/k^(1/(n-1))`
in general.

### What the separation is actually worth

From the spherical-cap packing bound, with `k = degree + 1` points (the parent
takes a slot) and `τ ≥ −log(tan(θ_min/4))`:

| degree | n=2 | n=3 | n=4 | n=5 | n=6 | n=8 |
|---|---|---|---|---|---|---|
| 4   | 1.124 | 0.693 | 0.539 | 0.455 | 0.400 | 0.333 |
| 16  | 2.379 | 1.386 | 1.052 | 0.877 | 0.767 | 0.631 |
| 64  | 3.723 | 2.079 | 1.541 | 1.268 | 1.098 | 0.895 |
| 256 | 5.097 | 2.773 | 2.016 | 1.637 | 1.407 | 1.134 |

(Sanity check: the n=2 column reproduces `PROOF.md`'s bound exactly.)

Usable depth is `floor(21 / τ)`, and a point costs `n × 16` bytes:

| degree | n=2 | n=3 | n=4 | n=5 | n=6 | n=8 |
|---|---|---|---|---|---|---|
| 16  | 8 | 15 | 19 | 23 | 27 | 33 |
| 256 | 4 | 7 | 10 | 12 | 14 | 18 |
| **bytes/point** | 32 | 48 | 64 | 80 | 96 | 128 |

**The returns are front-loaded.** Of the whole τ gain from n=2 to n=8, the
2→3 step captures ~57%, 3→4 another ~19%, 4→5 only ~10%. Three is the
value-dense step; five buys a tenth of the benefit for 2.5× the memory.

### The blocker is the index, and it has been measured

`cell_index` assigns bands from `planar_norm_sq` and sectors from
`pseudo_angle(coords[0], coords[1])` — **its geometry is planar**. Off-plane
*nodes* do not make it wrong: projection onto a totally geodesic plane through
the origin is 1-Lipschitz, so `d(q,p) ≥ d(proj q, proj p)` and the bound stays
a true lower bound. They make it **weak** — two nodes sharing a projection are
indistinguishable to it, so pruning decays toward a full scan.

Measured, same index, same node counts, self-nearest-neighbour queries at k=5,
planar placement versus directions spread on a sphere:

| nodes | planar | spherical | penalty |
|---|---|---|---|
| 500  | 208 µs | 520 µs | 2.5× |
| 2 000 | 308 µs | 1 053 µs | 3.4× |
| 5 000 | 467 µs | 1 925 µs | 4.1× |

(Harsh synthetic fixture — radii to 0.95 — so read the *ratio*, not the
absolutes.) The penalty **grows with n**, which is the signature of pruning
decaying rather than a constant overhead. Occupied cells *fall* (379 → 233 at
5 000 nodes): projecting a sphere onto a plane piles nodes into fewer, fatter
cells.

So 3-D placement dropped into today's index would hand back most of what 0.6.0
won. The honest comparison — 3-D placement against a 3-D-*aware* index, with
cells from a spherical code instead of one pseudo-angle — has not been built or
measured, and that measurement is the gate.

### Consequence nobody should discover late

Structural positions are **derived, not stored** (`HTT_FORMAT.md` §7: the reader
reconstructs them via Sarkar from tree topology). So changing the placement
construction changes every position in every existing `.htt` *on next load*.
File bytes are untouched — horon's determinism goldens hash file contents and
would still pass — but **spatial query results on existing data would change**.
That is a behavioural break without a format break, and it needs a version gate
in the header rather than a silent swap.

### Recommendation

Do not narrow, and do not add an angle yet. In order:

1. Build a spherical-code cell scheme and measure it against the planar one on
   planar data. If it cannot match today's numbers, the question is closed.
2. Only then measure 3-D placement against it, and compare the τ gained
   (≈ halved, so ≈ 2× depth) against the query cost.
3. Decide the width last. It is a separate decision from the construction, and
   `dimension` is a public config parameter *and* a header byte — narrowing it
   changes what a published parameter means.
