<p align="center">
  <a href="https://geodineum.com">
    <img src="https://geodineum.com/wp-content/uploads/2026/07/logo_geodineum_launch.png" alt="Geodineum" width="128">
  </a>
</p>

# horon-engine

**Store data as a tree. Query it as a space.**

[![Crates.io](https://img.shields.io/crates/v/horon-engine)](https://crates.io/crates/horon-engine)
[![Documentation](https://docs.rs/horon-engine/badge.svg)](https://docs.rs/horon-engine)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## The idea

Organize data in paths, like files in folders. The engine embeds that tree in
hyperbolic space, so structural similarity becomes spatial proximity. One
query primitive answers most questions: "what's nearby?"

## Why hyperbolic space fits trees

Hyperbolic space grows exponentially with radius. Trees grow exponentially
with depth. The match is exact: the engine places every node with Sarkar's
construction, items in the same branch cluster, items in distant branches sit
far apart. Each leaf insertion preserves the spatial structure with O(1)
work, so the tree is its own spatial index. Proof: [PROOF.md](PROOF.md).

## In production

A production course recommender runs on this engine: 183 courses and 741
students in one tree file. Recommendations are geometric, courses close to a
student's enrollment history rather than keyword matches. The same file
reveals miscategorized courses, demand gaps, and domains that share student
populations.

## Quick start

```rust
use horon_engine::Store;

let store = Store::new();

// Tree storage; parents are auto-created
store.put("/courses/trauma/emdr_basis", b"EMDR Basiscursus").unwrap();
store.put("/courses/trauma/emdr_kind", b"EMDR Kind & Jeugd").unwrap();
store.put("/courses/systemisch/eft", b"Emotion Focused Therapy").unwrap();

// Retrieval and hierarchy
let data = store.get("/courses/trauma/emdr_basis").unwrap();
let kids = store.children("/courses/trauma").unwrap();

// Spatial query: structurally nearby nodes.
// emdr_kind comes first (same branch), then eft (sibling branch).
let neighbors = store.neighbors("/courses/trauma/emdr_basis", 3).unwrap();

// Metadata
store.set_meta("/courses/trauma/emdr_basis", "capacity", "24").unwrap();
```

## Semantic dimensions

Any node can carry a coordinate vector encoding domain meaning: categories,
popularity, demand signals. Queries then run over any slice of those
dimensions, and different slices answer different questions from the same
data.

```rust
// Attach a 40-dimension coordinate vector (Q64.64, 16 bytes per dim)
let coords: Vec<u8> = encode_my_dimensions(...);
store.set_semantic("/courses/trauma/emdr_basis", coords.clone()).unwrap();

// The 5 nearest nodes by dimensions 16..33 (category axes),
// as Vec<(path, distance)> sorted by distance across exactly those dims
let similar = store.nearest_semantic(&coords, 5, 16..33).unwrap();

// Distance between two coordinate vectors, no store involved
let dist = Store::semantic_distance(&coords_a, &coords_b, 16..33);
```

This is how a catalog separates an item's *labeled* category from where its
population actually positions it. The gap between the two is a
miscategorization, visible in geometry and invisible in metadata.

## Install

```toml
[dependencies]
horon-engine = "0.4"
```

All arithmetic is [gMath](https://github.com/nierto/gMath) Q64.64 fixed
point. The determinism contract is defined on the embedded profile, so build
with:

```sh
GMATH_PROFILE=embedded cargo build
```

## API reference

All methods take `&self`. Share freely via `Arc<Store>` across threads.
Per-symbol truth lives in the docblocks: `cargo doc --open`.

| Method | What it does |
|--------|--------------|
| `put(key, data)` | Insert or update. Auto-creates parent nodes. |
| `put_data_only(key, data)` | Insert without a geometric embedding. The cheap bulk path. |
| `embed_existing(key)` / `embed_all(prefix)` | Upgrade data-only keys to full embeddings, in place. |
| `get(key)` | Retrieve data by path. |
| `remove(key)` | Delete a node. |
| `exists(key)` | Check existence. |
| `children(path)` | List direct children. |
| `list(prefix)` | List the full subtree. |
| `set_meta(key, k, v)` / `get_meta(key)` | Per-node key-value metadata. |
| `nearest(coords)` | Nearest node to a point: O(1) power-diagram grid probe plus candidate verification. |
| `nearest_k(coords, k)` | The k nearest nodes to a point. |
| `neighbors(key, k)` | The k nearest neighbors of a stored node. |
| `find_within(key, r)` | All nodes within hyperbolic radius r. |
| `position(key)` | A stored node's Poincare coordinates. |
| `set_semantic(key, coords)` / `get_semantic(key)` | Attach or read dimensional coordinates (raw Q64.64 bytes). |
| `nearest_semantic(coords, k, range)` | k nearest by Euclidean distance across a dimension slice. |
| `neighbors_semantic(key, k, range)` | Semantic neighbors of a stored node. |
| `find_similar(key, k, range)` | "What's like this one?": task-shaped name for `neighbors_semantic`. |
| `find_outliers(prefix, z, range)` | Nodes anomalously far from their peers under a prefix (average k-NN distance, z-score). |
| `semantic_distance(a, b, range)` | Euclidean distance between two coordinate vectors. |
| `SemanticDisk::build(spec)` | Embed a concept taxonomy, derived from the data's own category tree, into a second Poincare disk. |
| `disk.concept_of(store, key)` | Which concept a node belongs to *right now*, from its affinity dims. The miscategorization primitive. |
| `disk.nearest(store, key, k)` | k nearest nodes in taxonomy-aware meaning-space. |
| `disk.classify_trajectory(...)` | Turn a Horon `HoronHistory` trajectory into a symbolic concept sequence across epochs. |
| `query(adapter, query)` | Execute a pluggable query via the `QueryAdapter` trait. |
| `len()` / `is_empty()` | Node count. |

## HTTStorage, the layer below

`Store` wraps `HTTStorage`. Reach for it when you need direct control over
embedding dimension or grid resolution:

```rust
use horon_engine::{HTTStorage, HTTStorageConfig};

let storage = HTTStorage::new(HTTStorageConfig {
    dimension: 8,
    max_memory_nodes: 50_000,
    cache_size: 5_000,
    grid_resolution: 128,
    ..Default::default()
});
storage.store("/data", b"value", Some("text/plain".into())).unwrap();
```

## QueryAdapter

Pluggable query interface for building custom query languages on top of the
store:

```rust
use horon_engine::store::{QueryAdapter, QueryResult};

impl QueryAdapter for MyAdapter {
    fn execute(&self, store: &Store, query: &str) -> Result<Vec<QueryResult>, StoreError> {
        // Parse query, call store methods, return results
    }
}
```

## Concurrency

- Reads are lock-free: `get`, `exists`, `children`, `get_meta`, and all
  spatial queries. No reader ever blocks another reader.
- Writes stripe on the parent node: 64 lock stripes, so independent subtrees
  write in parallel.
- The spatial index is ~61 buckets, each with its own VP-tree and lock.
- There is no outer lock. Every method takes `&self`; wrap in `Arc<Store>`
  and share across threads, async tasks, or HTTP handlers.

## How it works

Traditional trees scale lookups with depth. Spatial indexes do not
understand hierarchy. Five mechanisms remove the choice:

1. **Sarkar embedding.** Every node gets a position in the Poincare disk;
   children sit at hyperbolic distance tau from their parent via Mobius
   reflection. The tree *is* its own spatial index.
2. **Geometric hashing.** ~61 fixed buckets partition the disk by distance
   from origin. Path lookups are O(1) HashMap access.
3. **Per-bucket VP-trees.** Range and KNN queries in O(log n) under the
   hyperbolic metric.
4. **Nielsen power diagram.** A Poincare-to-Klein projection gives O(1)
   nearest-neighbor via a uniform grid, verified with exact distances.
5. **Semantic dimensions.** Orthogonal to the spatial embedding: Euclidean
   distance over user-defined dimension slices, no tree change required.

## Determinism

All geometry is gMath Q64.64 fixed point. There are no floats in the compute
path. The same operation sequence produces bit-identical state on any
platform; that property is what lets a write-ahead log double as a
replication protocol. CI runs the suite on x86-64 and arm64.

## Performance

Measured 2026-07 on an i7-7700 (4c/8t) with `GMATH_PROFILE=embedded`. Full
tables, machine specs, and reproduction commands:
[BENCHMARKS.md](BENCHMARKS.md).

| Operation | Measured cost |
|-----------|---------------|
| `get` | ~0.9 µs |
| `exists` | ~0.1 µs |
| `put` (into an existing populated tree) | ~1 µs |
| `put` (fresh flat tree, n ≤ 100) | ~3-7 ms/node (grid-tile assignment worst case) |
| `nearest` (power diagram, full API) | ~250-290 µs |
| `neighbors` (VP-tree KNN, full API) | ~3-10 ms (exact 0-ULP distance per candidate) |
| `remove` | ~0.2-3 ms |
| `nearest_semantic` (lazy per-slice VP-tree, warm) | ~283 µs at 10k nodes, ~309 µs at 100k, d=8 |
| `nearest_semantic` (first query after a semantic write) | ~1.2 s at 10k nodes; index rebuild, amortizes after ~7 queries |

Geometric *primitives* run in nanoseconds (grid probe ~15 ns, power distance
~24 ns). Full-API queries pay for exact hyperbolic verification of every
candidate. Concurrent read throughput: ~2.6M reads/sec on 8 threads.

## Persistence

The engine is in-memory. [Horon](https://github.com/nierto/horon) persists
it: the `.htt` single-file format with WAL durability, zstd compression,
snapshot compaction, and geometric access control.

```rust
use horon::Horon;

let htt = Horon::open("data.htt")?;
htt.put("/config/db", b"postgres://localhost")?;
htt.compact()?;
```

## Architecture

```
Store                         <- public API, all &self, Arc-shareable
  └─ HTTStorage               <- path normalization, CRUD, striped parent locks
       └─ HyperbolicTreeTensor     <- path-to-signature maps (DashMap)
            └─ HyperbolicTensorNetwork  <- Sarkar embedding, spatial index
                 ├─ HyperbolicHashTable  <- ~61 buckets, per-bucket VP-trees
                 ├─ PoincareDisk         <- hyperbolic geometry, Mobius transforms
                 └─ PointLocationGrid    <- Nielsen power diagram point location
```

## Ecosystem

- **[gMath](https://github.com/nierto/gMath)**: Q64.64 fixed-point
  arithmetic with ZASC-Binary transcendentals.
- **[Horon](https://github.com/nierto/horon)**: `.htt` WAL persistence over
  this engine.

## Recent work

- **Semantic disk**: the concept taxonomy embedded hyperbolically, positions
  derived from affinity dimensions
  ([docs/SEMANTIC_DISK.md](docs/SEMANTIC_DISK.md)).
- **Semantic spatial index**: lazy per-slice VP-trees, ~690x faster at 10k
  nodes ([docs/SEMANTIC_INDEX.md](docs/SEMANTIC_INDEX.md)), plus
  `find_similar` and `find_outliers`.
- In [Horon](https://github.com/nierto/horon): temporal epochs (v0.6.0) and
  WAL-based replication (v0.5.0).

## License

Apache-2.0 (see [LICENSE](LICENSE)).
