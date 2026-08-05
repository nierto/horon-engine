# CONTRACT.scn :: horon-engine agent primer

Condensed prime for an agent building against `horon-engine`. Insertion
theorem: [PROOF.md](PROOF.md). Measurements: [BENCHMARKS.md](BENCHMARKS.md).
Per-symbol truth: docblocks (`cargo doc --open`). Code wins over docs.

## ::PRIME

```
Store := in-memory hyperbolic tree-tensor store | Sarkar embedding in the Poincare disk | tree == its own spatial index
Implements: path-keyed CRUD + spatial k-NN + semantic slice queries via geometric hashing (~61 buckets) + per-bucket VP-trees + Nielsen power diagram
Maintains: O(1) Delaunay-preserving leaf insertion (PROOF.md); positions are a pure function of the insertion sequence, derived not stored
Coordinates: g_math Q64.64 fixed point (pinned =0.4.31); persisted by the horon crate; no I/O of its own
```

## ::ANCHOR

```
DOMAIN: hyperbolic-geometry, poincare-disk, klein-model, sarkar-embedding, mobius-transform, delaunay, power-diagram, VP-tree, k-NN, geometric-hashing, locality-sensitive, fixed-point-Q64.64, concept-taxonomy, dimension-slice
PATTERN: derived-state, lazy-index, epoch-invalidation, bucket-partition, striped-locking, lock-free-reads, deterministic-tie-break
RUST: &self-everywhere, Arc<Store>, DashMap, Send+Sync, builder-config
CONCEPT: determinism-by-construction, structural-similarity-as-distance, punctuated-calibration, exponential-capacity
```

## ::CONTEXT

```
ARCHITECTURAL_LAYER: in-memory engine (no persistence; the horon crate owns .htt durability)
STACK: g_math (arithmetic) -> horon-engine (this crate) -> horon (.htt persistence)
LIFECYCLE: Store::new() | Store::with_config(StoreConfig::new().capacity(n).tau(t)) -> put/query -> drop
BUILD: GMATH_PROFILE=embedded cargo build      # the determinism contract is defined on this profile
TEST:  GMATH_PROFILE=embedded cargo test --all-features
```

## ::INTERFACE

Compressed signatures. Fallible calls return `Result<T, StoreError>`.

```
Store::new() | Store::with_config(StoreConfig) -> Store
put(key, &[u8])                                  # full geometric embedding, parents auto-created (~ms fresh)
put_data_only(key, &[u8])                        # no embedding (~µs); no spatial queries until embedded
embed_existing(key) -> bool | embed_all(prefix) -> usize   # upgrade in place, idempotent
get(key) -> Vec<u8> | remove(key) | exists(key) -> bool
children(path) -> Vec<String> | list(prefix) -> Vec<String>
set_meta(key, k, v) | get_meta(key) -> HashMap<String, String>
set_semantic(key, Vec<u8>) | get_semantic(key) -> Vec<u8>
  # raw Q64.64 LE, 16 B/dim; all-zero = "not set" sentinel, rejected
nearest(&[FixedPoint]) -> (String, FixedPoint)   # O(1) power-diagram probe + exact verification
nearest_k(coords, k) -> Vec<(String, FixedPoint)>          # bucketed VP-tree path
neighbors(key, k) -> Vec<String> | find_within(key, radius) -> Vec<String>
position(key) -> Vec<FixedPoint>                 # errors for unknown or data-only keys
nearest_semantic(&[u8], k, Range) | neighbors_semantic(key, k, Range) | find_similar(key, k, Range)
find_outliers(prefix, z: FixedPoint, Range) -> Vec<SemanticOutlier>   # prefix-local z-scored k-NN
Store::semantic_distance(a, b, Range) -> FixedPoint         # associated fn, pure
semantic_epoch() -> u64                          # monotone; external caches rebuild when it advances
query(&dyn QueryAdapter, &str) -> Vec<QueryResult>
len() | is_empty() | inner() -> &HTTStorage      # advanced: dimension, grid resolution, tau

SemanticDisk::build(&[(&str, usize)]) -> Result<SemanticDisk, StoreError>
  concepts() | concept_of(store, key)            # the miscategorization primitive
  position_of(store, key) | nearest(store, key, k) | nearest_to_weights(store, &[f64], k)
  classify_trajectory(..)                        # epochs -> symbolic concept sequence
```

## ::PATTERNS

```
STANDALONE:   Store::new() -> put -> children/neighbors/nearest; no persistence, no setup
BULK_LOAD:    put_data_only(all) -> embed_all(prefix) once; skips per-insert geometry
SLICE_QUERY:  set_semantic(d dims) -> nearest_semantic(q, k, a..b); slice = the question asked
SIMILARITY:   find_similar(key, k, dims) | anomalies: find_outliers(prefix, z, dims)
CONCEPT_DISK: SemanticDisk::build(taxonomy) -> concept_of/nearest over derived positions;
              positions are a pure function of affinity dims; nothing stored, divergence impossible
CALIBRATION:  write bursts, then frozen manifold queried many times; lazy per-slice VP-tree
              indexes amortize (first query after a semantic write pays the rebuild)
CUSTOM_QUERY: impl QueryAdapter -> store.query(adapter, expr)
```

## ::INVARIANTS

```
GUARANTEE: deterministic results: identical operation sequence, identical output, any platform
REQUIRES:  GMATH_PROFILE=embedded; g_math pinned =0.4.31
ENSURES:   replayable state; ties always break by (distance, key)
NEVER:     floats in the compute path (owner ruling; API-boundary f64 display values only)

GUARANTEE: the tree is its own spatial index
REQUIRES:  tau >= -log(tan(pi / (2 * d_max)))   # Delaunay condition, PROOF.md
ENSURES:   O(1) Delaunay-preserving leaf insertion; power-diagram point location stays valid
NEVER:     persist positions; geometry is derived state; re-derive by replaying inserts

GUARANTEE: indexed == brute force, byte for byte
REQUIRES:  semantic-epoch invalidation on coordinate writes, inserts, deletes
ENSURES:   VP-tree routed results identical to the O(n*d) scan they replace
NEVER:     serve stale index results across a semantic write

CONSTRAINT: all-zero semantic vector = "not set", rejected on write
CONSTRAINT: data-only keys have no position and join spatial queries only after embed_existing
CONSTRAINT: reads are lock-free; writes stripe on the parent (64 stripes); no outer lock
```

## ::GRAPH

```
DEPENDS_ON:   g_math =0.4.31 (fixed point), dashmap, serde, sha3
PROVIDES_TO:  horon (.htt persistence facade re-exports Store, StoreError, SemanticOutlier, SemanticDisk)
DOCS:         PROOF.md (insertion theorem) | BENCHMARKS.md (measured costs)
              | docs/SEMANTIC_INDEX.md | docs/SEMANTIC_DISK.md | docs/HYPERBOLIC_INDEX.md
ORACLES:      tests/properties.rs (metric axioms) | tests/semantic_index.rs (indexed == scan)
              | tests/vocabulary.rs (naming) | tests/hardening_regressions.rs
```

---

Built by **Niels Erik Toren** · [support](https://github.com/nierto/horon#author--support) · Apache-2.0
