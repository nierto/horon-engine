# Semantic Index: lazy per-slice VP-trees with epoch invalidation

Status: **shipped**.
Companion: `../horon/docs/TEMPORAL_EPOCHS.md` — the two designs share the
epoch model but are independently buildable.

## Problem

`TensorNetwork::nearest_semantic(query, k, dim_range)` is a brute-force scan:
O(n × d) per query (measured ~195 ms/query at 10k nodes, linear in n). Every
semantic query — `Store::nearest_semantic`, `neighbors_semantic`, and
everything Horon layers on top — funnels through this one function.

A VP-tree fixes this, but with a hard constraint: **VP-tree pruning is only
valid when the build metric equals the query metric.** Semantic distance is
Euclidean *over a caller-chosen `dim_range`*, so a tree built over dims
`16..33` cannot answer a query over `16..18` — the triangle-inequality bounds
baked in at build time are wrong for the narrower slice, and the tree would
silently prune away true neighbors. One tree per queried slice, no exceptions.

## Rejected alternatives (and why)

- **Registered fixed slices** (declare indexable ranges up front): predictable,
  but requires per-domain configuration. HTT is domain-neutral — a course
  catalog slices differently from a hotel-occupancy tree; no fixed registry fits all.
- **One tree over the full user vector**: only accelerates full-vector queries;
  real workloads (e.g. a category slice 16..33) are sub-slices, so the
  index would rarely be used. Blending unrelated axes into one distance is
  also usually semantically wrong.
- **Lazy per-slice cache with write-triggered rebuilds**: the general design,
  but under interleaved writes it degenerates (every write invalidates; next
  query pays a full O(n log n) rebuild — potentially slower than brute force).

The accepted design is the third option **made safe by the workload model**:
calibrated HTT usage is *punctuated* — bursts of writes (calibration/build
phases), then a frozen manifold queried many times (see
`TEMPORAL_EPOCHS.md`). Between calibrations nothing
moves, so lazily built indexes stay valid; invalidation collapses to a single
counter comparison.

## Design

### 1. Generic static metric tree (`src/metric_tree.rs`)

A static (build-once, no incremental insert/delete) VP-tree, generic over
point type and metric — the "pluggable DistanceMetric" half of the design:

```rust
pub trait Metric<P> {
    fn distance(&self, a: &P, b: &P) -> FixedPoint;
}

pub struct MetricVpTree<P> { /* nodes: (uid: String, point: P) */ }

impl<P> MetricVpTree<P> {
    pub fn build<M: Metric<P>>(entries: Vec<(String, P)>, metric: &M) -> Self;
    pub fn knn<M: Metric<P>>(&self, query: &P, k: usize, metric: &M)
        -> Vec<(String, FixedPoint)>;
}
```

Same proven algorithm as the per-bucket hyperbolic `VPTree` in
`hash_table.rs` (Yianilos 1993: first-entry vantage point for determinism,
median partition, tau-shrinking KNN with closer-subtree-first descent,
inclusive pruning bounds). The bucket VPTree is dynamic (buffer + lazy
delete) and stays as-is — the static core serves the epoch model, where
indexes are never mutated, only discarded. Migrating the bucket tree onto
the generic core is possible later but is not part of this change.

**Determinism upgrade:** all candidate ordering is lexicographic on
`(distance, unique_id)`. Ties at the k-boundary previously depended on heap
iteration order in the brute-force path; both paths now break ties by
`unique_id`, making `nearest_semantic` results fully deterministic and
byte-identical between the indexed and fallback paths. Inclusive subtree
bounds (`d - tau <= median`, `d + tau >= median`) keep pruning correct for
distance-equal candidates.

### 2. Per-slice cache + epoch counter (`src/semantic_index.rs`)

```text
TensorNetwork
 ├─ semantic_epoch: AtomicU64            // bumped on any relevant mutation
 └─ semantic_indexes: RwLock<BTreeMap<(usize, usize), Arc<SliceIndex>>>
      SliceIndex { epoch: u64, tree: MetricVpTree<Vec<FixedPoint>> }
```

Query routing inside `nearest_semantic` (all public APIs inherit it):

1. `k == 0` → empty. Live semantic node count `< SEMANTIC_INDEX_MIN_NODES`
   → brute-force scan (tree build isn't worth it; matches current behavior).
2. Cache hit for `(dim_range.start, dim_range.end)` with
   `index.epoch == semantic_epoch` → `tree.knn`.
3. Otherwise: read `semantic_epoch` **first**, snapshot all nodes with
   non-empty coords (decode `dim_range`, zero-extending short vectors —
   identical to `semantic_distance` semantics), build the tree, insert into
   the cache tagged with the pre-read epoch, answer from it.

**Epoch-bump sites** (mutation → then bump, one relaxed atomic increment):
`set_node_semantic`, `unregister_node_with_parent` (the only `nodes.remove`),
and both `nodes.insert` sites (`add_node` / `add_node_data_only`). Inserts
carry no coords today and cannot change results, but the bump is free
insurance against future insert-with-coords paths.

**Race safety:** writers mutate *then* bump; builders read the epoch *before*
snapshotting. A write that lands mid-build bumps the counter after the
builder's pre-read, so the cached index is tagged stale and discarded on the
next query. A query racing a concurrent write has the same semantics the
brute-force scan already has. Two racing builders may both build; results are
deterministic and identical, last insert wins — harmless.

**Eviction:** the cache holds at most `SEMANTIC_INDEX_MAX_SLICES` slices.
On overflow, the lowest `(start, end)` key not equal to the incoming one is
evicted (deterministic; no wall-clock LRU — determinism > recency). Stale
entries (old epoch) are dropped whenever encountered.

### 3. Honest cost model

| | cost |
|---|---|
| Query, warm index | O(log n) expected (low-dim); worst case O(n) — never wrong, inclusive bounds |
| First query for a slice after any semantic write | O(n log n) distance evals (build) + O(log n) |
| Memory per cached slice | n × (d × 16 B + uid) ≈ the slice's raw coordinate data |
| Every write | +1 relaxed atomic increment |
| n < `SEMANTIC_INDEX_MIN_NODES` | unchanged brute force — small trees pay nothing |

**Limits, stated plainly:** VP-tree pruning degrades as slice dimensionality
grows (past ~10–15 dims it visits most nodes; results stay correct, speedup
evaporates). This accelerates the low-dimensional, interpretable-axis world
that meaning-addressed HTT already lives in (`MAX_HILBERT_DIMS = 8` for the
same reason). It is not an ANN index for high-dim embeddings and does not
claim to be. Interleaved write/query workloads pay a rebuild per write-then-
query transition — the punctuated (calibrate → seal → query) model is the
intended usage; worst-case degradation is to brute-force-equivalent, never
to wrong answers.

## Out of scope

- Grid prefilter (benchmarks decide later if it's ever needed).
- Migrating the per-bucket hyperbolic VPTree onto `MetricVpTree`.
- Radius-based semantic queries (no current caller; `knn` only).
- Persisting indexes into `.htt` (rebuild-on-open is cheap relative to load).

## Verification

- Property test: indexed results == brute-force results (same nodes, same
  order) across randomized coords, slices, k values, and short/zero-extended
  vectors — the correctness bar is *equality*, not approximation.
- Invalidation test: query → `set_semantic` → query returns updated answers.
- Determinism test: identical inputs → identical output ordering under ties.
- Bench (`benches/semantic.rs`): brute vs indexed at 1k/10k nodes, warm and
  cold (first-query build cost reported honestly, not hidden).
