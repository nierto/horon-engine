# Semantic Disk: taxonomy-embedded meaning space

Status: **shipped**.
Companions: `SEMANTIC_INDEX.md` (supplies the metric-pluggable tree and
the epoch invalidation model), `../horon/docs/TEMPORAL_EPOCHS.md` (
supplies the trajectory record this design turns into symbolic sequences).

## Problem

The engine has two query worlds with asymmetric machinery. The **structural**
world (storage hierarchy → Poincaré disk via Sarkar) gets the full stack:
Sarkar placement and an exact cell index over the disk. The
**semantic** world (flat coordinate axes, dims 16+) got Euclidean space and,
with the semantic index, an O(log n) VP-tree. But semantic axes usually carry a hidden
hierarchy too — a course catalog's 17 category dims are not unrelated numbers,
they are a *concept taxonomy* (trauma and EMDR are kin; trauma and
werk_coaching are not). Flat Euclidean distance is blind to that kinship.

The semantic disk embeds the domain's concept taxonomy into a second Poincaré disk and
gives every data node a position in it, so "near in meaning" becomes
hyperbolic proximity in a space whose geometry *matches the domain's actual
shape* — same engine, two disks.

## Ratified design decisions

Three forks were resolved in design review (2026-07-11):

1. **Taxonomy source: derived from structure within the data.** The concept
   tree arrives with the data — the category tree already present in the
   paths (`/catalog/course/{category}/…`), a directory tree, or any tree the
   caller names. No new declaration machinery; anchors are few (dozens, not
   thousands). The caller supplies the concept paths plus the mapping from
   each concept to the affinity dimension that weights it — this mapping is
   **calibration**, fixed for a file's life like the dimensional schema.

2. **Movement is first-class; no residual-offset model.** A node's position
   in the concept disk travels freely as its affinity dims change. The
   trajectory itself is the information carrier: one moving point encodes
   what would otherwise be thousands of static observations. Whatever
   determines concept position must therefore be captured by the epoch
   record — a second, separately-maintained position history is forbidden.

3. **Coexistence via derivation ("E+D").** Evaluated alternatives:
   *replacement mode* (killed: operational dims like occupancy have no home
   in a taxonomy, and arbitrary-slice queries must survive); *stored
   parallel layer* (killed: new WAL op + format version + a divergence bug
   class — stored position vs. dims disagreeing with no arbiter);
   *reserved-band storage* (viable, kept as a documented later extension
   for hypothetical hand-placement domains — NOT built now); **derived
   positions (E)**: concept position is a pure deterministic function of
   the affinity dims — nothing new is stored anywhere; **anchor
   power-diagram (D)**: the taxonomy itself is the index.

## Architecture

```text
SemanticDisk (standalone object; Store is untouched)
 ├─ taxonomy: private Store holding ONLY the concept tree
 │    → Sarkar embedding, power diagram, O(1) grid — all reused verbatim
 ├─ mapping: Vec<(concept_path, affinity_dim)>       // the calibration
 └─ cache: epoch-tagged MetricVpTree<HyperbolicPoint>  // exact NN layer
```

### Derived position (E)

For a node with affinity weights `w_i` (read from its mapped dims, negatives
clamped to zero):

```text
position = WeightedKleinBarycenter( anchor_i, w_i )
         = klein_to_poincare( Σ w_i·γ_i·k_i / Σ w_i·γ_i ),  γ_i = 1/√(1−|k_i|²)
```

the weighted Einstein midpoint of the anchor sites in Klein coordinates —
the standard hyperbolic barycenter, computed in Q64.64 fixed point
(deterministic; verified against the gyro-midpoint for the two-point
equal-weight case). All-zero weights → the node has no concept position and
is excluded from concept queries (same convention as empty semantic coords).

Consequences, by construction:
- **Single source of truth.** Concept position ≡ a view of the dims; it can
  never disagree with them. The divergence bug class cannot exist.
- **Zero storage / zero format impact.** Horon is untouched. Existing
  `.htt` files gain a concept disk retroactively — existing category
  affinities are already the weight vector.
- **Temporal epochs for free, replayable.** The epoch record already captures every
  affinity change; since the mapping is deterministic, replaying history
  replays the concept-space trajectory bit-identically. Recalibration
  (Phase 3) *moves nodes through meaning-space* and the movement is
  automatically on the record.

### Anchor power-diagram (D)

The taxonomy is a real tree, so it is embedded by inserting its concept
paths into a private `Store` — Sarkar placement and the Klein conversions come
along unchanged. Classification is the exact Nielsen reduction over the anchor
sites, `argmin_i (1 - <x, k_i>) * gamma_i`, evaluated directly; there is no
precomputed cell structure. Because the *sites* are the anchors (dozens), not
the data nodes (unbounded):

- **`concept_of(key)` — constant-in-data-size classification.** Locate the
  node's derived position among the anchor cells — linear in the anchor
  count, i.e. dozens; independent of data-node count. "Which concept does
  this node belong to right now" is one point location. Compared against
  the node's storage path, this is miscategorization detection as a
  primitive.

  The cell test is the Nielsen affine reduction of the hyperbolic Voronoi
  diagram — `argmin_i (1 − ⟨x, k_i⟩)·γ_i` in Klein coordinates, reusing the
  γ the barycenter already caches. It must be that reduction and **not** the
  Euclidean power distance `‖x − k_i‖² − (1 − ‖k_i‖²)`: the two agree only
  when every anchor shares one Klein norm. A flat single-depth spec
  satisfies that by construction; a **nested** one never does, because
  Sarkar places deeper concepts further out. Scored the Euclidean way, a
  shallow anchor swallows a deeper anchor's own site — single-anchor
  identity breaks and whole subtrees end up owning no territory at all.
- **Blend behavior, stated honestly:** a *strongly dominant* affinity
  classifies to its dominant concept. A *balanced* blend of two distant
  concepts derives a position in the middle of the disk — and the middle
  belongs to whichever anchor's cell occupies it, which can be a third,
  angularly-intermediate concept. That is correct hyperbolic geometry, not
  an error: a node that is "half trauma, half systemisch" genuinely isn't
  in either cell. Callers who need blend-awareness should read the weight
  vector, not just the cell.
- **Symbolic trajectories.** Push a temporal trajectory through the anchor
  diagram and a moving point becomes a sequence of discrete meaning-states —
  `trauma → trauma → systemisch` across epochs — plus the continuous path
  between them (`classify_trajectory`).

### Exact NN layer

`nearest` (k nearest data nodes in the concept disk) uses the semantic index's
`MetricVpTree` with the **hyperbolic** metric (the `Metric` trait earns its
"pluggable" adjective): build over all derived positions, cache tagged with
the store's semantic epoch, rebuild lazily on invalidation — the index's model
verbatim. O(log n) expected, results identical to a brute-force hyperbolic
scan (property-tested).

**Honest scope note:** v1 ships O(1) *classification* (the anchor grid) and
O(log n) *NN* (the metric tree). Bucketing data nodes per anchor cell to
give NN the O(1) grid entry as well is a benchmark-driven follow-up — it
changes constants, not results.

## API (v1)

```rust
let disk = SemanticDisk::build(&store_dimension_agnostic_spec)?;
//   spec: &[(&str, usize)] — (concept path, affinity dim), e.g.
//   [("/trauma", 16), ("/cgt", 17), … ] — nested paths allowed and
//   embedded with their real tree shape.

disk.position_of(&store, key)?           // derived Poincaré coords
disk.concept_of(&store, key)?            // O(1): which concept, right now
disk.nearest(&store, key, k)?            // k nearest in meaning-space
disk.nearest_to_weights(&store, &w, k)?  // query by explicit weights
disk.classify_trajectory(start_dim, &samples)
//   samples: the (epoch, coords) pairs HttHistory::trajectory returns →
//   Vec<(epoch, concept_path)> — the symbolic trajectory.
```

`SemanticDisk` is a standalone object the application owns (built per
calibration, cheap: anchors are few). `Store` gains no new state, no new
locks — minimal-diff, and the disk's lifetime maps naturally onto the
punctuated calibrate→seal→query model.

## Determinism

Anchor insertion order is the sorted concept-path order; the barycenter is
fixed-point arithmetic; classification and NN inherit the index's
`(distance, key)` total order. Identical stores + identical spec →
byte-identical answers. The (spec, mapping) pair is part of a file's
calibration: changing it mid-epoch-series is a schema-epoch boundary, the
same discipline as the dimensional schema itself.

## Out of scope (documented, not built)

- Reserved-band stored positions (hand-placement escape hatch — option C).
- Per-anchor-cell NN partitioning (the O(1)-entry NN optimization).
- Learned/induced taxonomies (clustering); the taxonomy comes from data
  structure per the ratified decision.
- Any `.htt` format change — none is needed, which is the point.

## Verification

- Classification: single-anchor identity — a pure one-anchor weight vector
  derives exactly that anchor's site and must classify to it — asserted for
  a **nested** spec and for a spec whose anchors are each other's
  ancestors, not only for a flat one.
- Barycenter: single-anchor identity; two-equal-weights ≈
  `hyperbolic_midpoint` (verified against the gyro-midpoint oracle); weight-scale invariance;
  all results strictly inside the disk.
- NN: property-test equality against brute-force hyperbolic distance over
  derived positions (same bar as the index: equality, not approximation).
- Re-homing: change a node's dominant affinity → `concept_of` moves.
- Miscategorization: node filed under `/overig` with trauma-shaped weights
  classifies to `/trauma`.
- Symbolic trajectory: scripted drift across a cell boundary produces the
  expected concept sequence, end-to-end from an `HttHistory` readout.
