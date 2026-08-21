# Changelog

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
