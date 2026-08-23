# Changelog

## Unreleased

Three honesty fixes. No behavioural change to a correct call; each replaces a
silent or vacuous answer with a stated one.

### Added

- **`constants::max_degree_for_tau`** and **`HyperbolicTensorNetwork::max_degree`**
  — the inverse of `PROOF.md`'s hypothesis, `d_max = pi / (2 * arctan(e^-tau))`.
  Reproduces the published table exactly: tau = 1.0 gives 4 (documented as
  `d_max ~= 4.46`), and 256 children need tau >= 5.094.

### Changed

- **A parent exceeding `max_degree(tau)` now warns**, once, on the crossing.
  `CONTRACT.scn.md` has always declared `tau >= -log(tan(pi / (2 * d_max)))`
  and nothing checked it. Since 0.6.0 a violation costs spacing quality —
  crowded siblings, more nodes per cell, longer scans — and never correctness,
  so this is a warning and not an error. Computed once per network; the insert
  path pays one integer comparison.

- **Dropped results are logged instead of vanishing.** `nearest_k` and
  `neighbors` translate index hits back to paths through `id_to_path` and
  silently skipped anything missing, so a caller could ask for `k` and get
  fewer with nothing saying why. A delete removes the path before the index
  entry, so a concurrent query seeing an unmapped id is benign and
  self-correcting — but a *persistent* count means `id_to_path` has drifted
  from the index, which no other check would catch. Self-exclusion in
  `neighbors` is discounted and not counted as a drop.

- **Semantic slices that cannot carry information are rejected.**
  `decode_semantic_slice` zero-extends by design, which lets a short vector
  compare against a long one. Two cases abused it: an **empty** range, and one
  starting past the end of the *query's own* coordinates. Either way every
  candidate tied at distance zero and the deterministic key tie-break picked
  `k` of them — a confident, reproducible, information-free answer.
  `nearest_semantic`, `neighbors_semantic` and `find_similar` now return
  `InvalidOperation`; `find_outliers` rejects the empty range (it has no single
  query vector to measure against). A range that merely *extends past* the
  data is still valid — that is zero-extension working as intended.

## 0.6.1

Documentation only. No code change, no API change; `0.6.0` and `0.6.1` are
behaviourally identical.

0.6.0's README was updated in part and not in whole, and crates.io renders the
README from the published tarball rather than from the repository — so the
landing page for the release advertised the previous version.

- **The install snippet said `horon-engine = "0.5"`.** That is the line people
  copy.
- **The performance table was the pre-0.6.0 set**, still crediting a grid probe
  and a power distance, and carrying a note that its rows predated 0.5.0.
  Replaced with the re-measured structural rows and a comparison column;
  semantic rows are unaffected by 0.6.0 and now say so.
- `nearest` was still described as "O(log n): the grid proposes candidates".
- `HTTStorage` was still offered as the way to set a grid resolution that no
  longer exists.
- README and crate-level docs linked into `docs/`, which is not packaged, so
  those links resolved to nothing on crates.io and docs.rs. They are absolute
  now.

## 0.6.0

Index replacement. `nearest`, `nearest_k`, `neighbors` and `find_within` are
answered by a computed cell index; the geometric bucket table, the Nielsen
power cells and the point-location grid are removed. **Breaking** — see
*Removed* — but no `.htt` format change and no stored coordinate moves: the
index is derived from the tree on load and never serialized.

The point of the release is that correctness stops depending on placement
quality. The cell index looks up the query's cell, then expands rings until a
proven lower bound rules out every cell it has not visited. Nothing is capped,
sampled or windowed; the search widens instead. `docs/ARCHITECTURE.md` is the
orientation doc.

### Added

- **`cell_index`** — radial bands from squared-norm thresholds, angular sectors
  from a diamond pseudo-angle. One division per query for the angle, at most
  one square root; no `atanh`, no `atan2`, no transcendental on any per-node or
  per-cell path.
- **`GeometricSignature::embedded(point, dimension, level)`** — builds a
  signature without consulting any index. Replaces
  `HyperbolicHashTable::create_signature`.
- **`Store::max_depth()`** — `floor(max_safe_radius / tau)`; 21 at the default
  τ = 1.0.
- **`HyperbolicTensorNetwork::verify_index_locates_all_nodes`**, wired into
  `validate_network`. Every existing integrity check was *referential* (do
  these maps point at things that exist?). This one is **functional**: it asks
  the index to find each node at that node's own stored position, where the
  answer — distance zero — is known without an oracle. The bucket layer passed
  every referential check while `nearest` returned the wrong node for 25 of 42
  nodes.

### Changed

- **Depth past `max_safe_radius` is now refused, not warned about.** Beyond
  hyperbolic radius 21 the Q64.64 distance kernel saturates: every node reads
  as the same distance from every other and ranking becomes arbitrary. An
  answer from that band is not slow, it is meaningless, so placement there is
  an error.
- **Measured, same machine and toolchain as the published 0.5.x figures**
  (i7-7700, rustc 1.94.0, `GMATH_PROFILE=embedded`):

  | Operation | 0.5.x | 0.6.0 |
  |---|---|---|
  | `nearest` | ~250–290 µs/query | **51–99 µs/query** (n = 10…200) |
  | `nearest` at the origin | ~82 µs | **7.6 µs** |
  | `neighbors` (k = 1…5) | ~2.1–4.3 ms/query | **38–223 µs/query** |
  | `put` (fresh flat tree) | ~3.5–7 ms/node | **46.8–95.6 µs/node** |
  | `remove` | ~0.2–3 ms/op | **7.9–30.7 µs/op** |

  Marginal `put` into a populated tree is ~1.18 µs against a published ~0.95 µs;
  the old code was not re-run, so that gap is not attributable. Every other row
  improved. The write-path gain is mostly the removal of one **exact** 37.5 µs
  hyperbolic distance per insert, which the bucket layer paid to keep a
  bucket's pruning radius current, plus two Klein bisectors and a grid-tile
  reassignment.

  Read the query rows with care: the 0.5.x `nearest` figure is the cost of an
  answer that was frequently wrong. 0.5.2 fixed the correctness and cost ~2 ms
  doing it; 0.6.0 is what removed that cost.

### Fixed

- **`find_within` panicked on a large radius.** `within_radius` turned the
  caller's radius into a `cosh` ceiling for the ring bound, and `cosh` is
  infallible in Q64.64 — it panics rather than saturating. Any radius past
  ~44 took the query down, and "give me everything" is a normal way to call
  this: `horon`'s own API-surface test passes 1000. A radius too large to
  express in cosh space is a radius that prunes nothing, so the ceiling is now
  optional and its absence means *scan everything*. Degrades to slow, never to
  wrong. Regression test: `a_radius_too_large_for_cosh_returns_everything`.

- **`nearest_k(q, 0)` claimed the tree was empty.** It read an empty result
  set as an empty index — two different causes behind one error — so asking a
  200-node store for zero neighbours got back
  `No nodes in tree for nearest neighbor query`. `neighbors(path, 0)` always
  returned `Ok([])`, so the sibling APIs disagreed. `k == 0` is now answered
  with nothing, and the error is reserved for the state it describes.

- Stale doc claim on `HyperbolicTreeTensor::nearest_neighbor_point` ("Uses the
  Nielsen power diagram grid for O(1) lookup") — neither the grid nor the O(1)
  claim survived this release.

### Removed

Breaking. Nothing here has a caller in `horon`, which imports only `Store`,
`StoreConfig`, `StoreError`, `SemanticDisk` and `SemanticOutlier`.

- `HyperbolicHashTable` and the whole bucket layer — `HyperbolicRegion`,
  `HyperbolicHashBucket`, `BucketEntry`, and the per-bucket `VPTree`. Measured
  reason: **0 of 57 buckets could contain any node.** 56 centres sit off the
  data plane (median 71.7% of their norm in dimensions 2–3) while structural
  placement is provably planar, so every node was placed by a half-space sign
  test rather than by containment.
- `PointLocationGrid`, and the `grid_resolution` knob on `HTTStorageConfig` /
  `HTTConfig` that configured it. One owner per tile, but Sarkar placement
  drives cells below tile size within a few levels — in the exact diagram 34 of
  43 nodes own no tile, none deeper than 2. Not repairable at any affordable
  resolution.
- `PowerCell`, `HalfPlane`, `compute_bisector`, `point_in_cell`. Two bisectors
  per insert whose `normal` and `offset` were never read in production; every
  real read used `neighbor_id`, which is parent + children and already in the
  tree.
- `HyperbolicTensorNetwork::{hash_table, get_klein_point, get_power_cell,
  grid_assigned_tile_count, with_grid_resolution}`, and the per-node
  `klein_points` cache that fed the bisectors. For Klein coordinates, convert
  on demand: `poincare_to_klein(&network.get_point(id)?)`.
- `pub use hash_table::HyperbolicHashTable` from the crate root, replaced by
  `pub use hash_table::GeometricSignature`.

`KleinPoint`, `poincare_to_klein`, `klein_to_poincare` and `power_distance`
stay: `SemanticDisk` is built on the Klein model.

Also not shipped: `src/spatial_index.rs`, a `MetricVpTree` wrapper held in
reserve while the cell index was being proven. It is nothing's dependency and
was never released; the file remains in the tree but is no longer a module.

### Notes

- The `hash_table` module keeps its name for this release although it no longer
  holds a hash table, so the public path
  `horon_engine::hash_table::GeometricSignature` does not move twice.
- `tests/power_diagram.rs` is now `tests/spatial_queries.rs` (all 34 tests
  kept — 31 never tested the power diagram), and `benches/power_diagram.rs` is
  `benches/spatial_queries.rs`. The bench name in CI moved with it.

## 0.5.2

Correctness release. No API changes, no `.htt` format changes, no stored
coordinates move. Two spatial queries were returning confidently wrong
answers; both are fixed and guarded by regression tests.

### Fixed

- **`Store::nearest` returned the wrong node.** The O(1) point-location grid
  holds one owner per tile, and Sarkar placement drives a node's power cell
  below tile size within a few levels — in a 43-node tree, no node deeper than
  2 owns a tile at all. The grid's answer was returned unverified, and the
  VP-tree was consulted only when the grid *missed*, never when it was
  *wrong*. The grid now proposes candidates and hyperbolic distance decides.

  Querying at a node's own exact stored position: **25 of 42 nodes wrong → 0**
  in a deep unbalanced tree, **4 of 30 → 0** in a flat one. Against brute force
  over 5,000 arbitrary query points: **2,023 wrong → 0**.

  Cost: `nearest` goes from 61.6 µs to 2.0 ms per query. That is not a
  regression against a working baseline — it is what the correct answer has
  always cost through `nearest_k`, which was already exact. The 2.0 ms is
  dominated by `find_nearest_nodes` taking a full `hyperbolic_distance` to
  every bucket centre purely to order buckets; that scan is the target of the
  index replacement in 0.6.0.

- **`SemanticDisk::concept_of` misclassified nested taxonomies.** Anchors were
  scored by the Euclidean power distance `‖x − k‖² − (1 − ‖k‖²)`, which agrees
  with the hyperbolic Voronoi diagram only when every anchor shares one Klein
  norm. Flat, single-depth specs satisfy that by construction; nested ones
  never do, because Sarkar places deeper concepts further out. Shallow anchors
  swallowed deeper anchors' own sites, and whole subtrees owned no cell.

  Now uses the exact Nielsen reduction `argmin_i (1 − ⟨x, k_i⟩)·γ_i`, reusing
  the γ already cached for the barycenter — one dot product per anchor, no
  sqrt, no division. Verified exact against true hyperbolic argmin on 200,000
  random queries; the old formula disagreed on 9.6% of the disk for a
  ten-anchor nested spec. Also ~4% *faster*, since it drops a per-call clone.

### Documentation

Several published claims did not match the implementation and have been
corrected rather than left standing.

- `nearest` is documented as O(log n), not O(1), with the reason.
- `PROOF.md` gains an **Implementation status** section: the Delaunay theorem
  is conditional on `tau >= -log(tan(pi / (2·d_max)))`, the shipped default
  `tau = 1.0` satisfies that only to `d_max ≈ 4.46`, and the cone construction
  the proof assumes is not implemented (21 of 43 nodes have a nearest
  neighbour that is not a tree neighbour). Query results are unaffected —
  every spatial query is decided by hyperbolic distance, never by the Delaunay
  identity.
- `README.md` gains a **Current limitations** section covering the `tau` /
  fan-out relationship, the Q64.64 depth bound (usable hyperbolic radius ≈ 17.5,
  saturating near 22), and the grid's role as candidate proposer.
- `klein.rs` documents that `power_distance` is the *Euclidean* power distance
  and coincides with the hyperbolic diagram only at equal Klein norms.
- Corrected stale timings in source comments: `hyperbolic_ratio` is 23.9 µs
  (not ~200 ns) and `hyperbolic_distance` is 37.5 µs (not ~62 µs);
  `power_distance` at 21.8 ns was accurate.

### Tests

- `tests/nearest_exactness.rs` — a node must be its own nearest neighbour, and
  arbitrary queries must match brute force. Both verified to fail on 0.5.1.
- `tests/semantic_disk.rs` — single-anchor identity for a nested taxonomy and
  for a spec whose anchors are each other's ancestors. Both verified to fail
  on 0.5.1.

243 tests pass.

### Not fixed in this release

The point-location grid and the fixed bucket partition are structurally unable
to do their jobs and are replaced, not repaired, in 0.6.0. They no longer
affect correctness; they cost query time.
