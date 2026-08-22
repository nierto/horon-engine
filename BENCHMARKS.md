# gHyper Benchmarks — measured 2026-07-09

> This project was renamed (engine: horon-engine, storage layer: Horon)
> after these measurements were taken. Measurement records are kept
> verbatim under the names in use on the day they were made.

Every number in this file is from a fresh run on the machine below, after the
Hardening fixes (effective-radius pruning, delete cleanup, input validation,
high-resolution position signatures). These replace all previously published
numbers (`TEST_RESULTS.md` retired).

> Measured before the 2026-07-30 boundary-constant change
> (`docs/HYPERBOLIC_INDEX.md`), which raised usable nesting from 7 levels to
> 21. The structures benchmarked here are wide rather than deep, so the
> figures should still hold — but anything involving paths nested past ~7
> levels was previously computing against a saturating distance kernel and
> wants re-measuring before it is quoted.

## Machine & toolchain

- CPU: Intel Core i7-7700 (4 cores / 8 threads, 3.6–4.2 GHz)
- RAM: 62 GiB · Kernel: Linux 5.15.0-185-generic (x86-64)
- rustc 1.94.0 · criterion, `--release` (lto = fat, codegen-units = 1)
- `GMATH_PROFILE=embedded` (Q64.64, I256 compute tier)

## How to reproduce

```sh
GMATH_PROFILE=embedded cargo bench --bench spatial_queries
GMATH_PROFILE=embedded cargo bench --bench semantic       # semantic suite
GMATH_PROFILE=embedded cargo run --release --example concurrency_bench
```

## Geometric primitives (criterion medians)

| Benchmark | Result |
|---|---|
| Power distance, single 4D | 23.8 ns |
| Brute-force NN, 10 / 100 sites (reference) | 233 ns / 2.45 µs (linear in N) |
| Poincaré→Klein, 2D / 4D | 794 / 825 ns |
| Klein roundtrip 4D (2 conversions + sqrt) | 26.9 µs |
| Hyperbolic distance, single 2D (one-sqrt kernel) | 35.4 µs (was 76.5 µs — the old form paid four sqrts, two of them a norm→square round-trip) |
| Hyperbolic ratio, single 2D (no atanh) | 22.8 µs (was 61.4 µs) |

An exact hyperbolic distance is the unit of cost that matters: at ~23–35 µs it
dwarfs everything else on the query path, so a spatial query's cost is very
nearly *how many exact distances it was forced to evaluate*. The full-API
figures below are best read that way — `nearest` at 51 µs on a 10-node tree is
roughly two exact distances, and at 99 µs on 200 nodes roughly four.

## Full storage-API operations

Re-measured 2026-08-22 on the same machine and toolchain after the 0.6.0 index
replacement. The 0.5.x column is the previously published figure, kept so the
change is auditable; both columns are criterion medians.

| Operation | 0.5.x | 0.6.0 | Notes |
|---|---|---|---|
| `get` | ~0.94 µs/op | not re-measured | 1 000-key store, hot loop; untouched by the index change |
| `exists` | ~0.08 µs/op | not re-measured | same |
| `put` (marginal, steady state) | ~0.95 µs | ~1.18 µs | flat at base sizes 10/50/100. The one row that did not improve; the old code was not re-run, so the gap is not attributable |
| `put` (fresh flat tree, n ≤ 100) | ~3.5–7 ms/node | **46.8–95.6 µs/node** | 95.6 at n=10, 52.2 at 50, 46.8 at 100 — the per-node cost *falls* with tree size now |
| `remove` | ~0.2–3 ms/op | **7.9–30.7 µs/op** | delete-all over 10/50/100-node trees |
| `nearest` (full API) | ~250–290 µs/query | **51–99 µs/query** | 51 at n=10, 65 at 50, 76 at 100, 99 at 200 |
| `nearest` at the origin | ~82 µs | **7.6 µs** | 50-node tree; the query sits exactly on the root |
| `neighbors` (KNN, full API) | ~2.1–4.3 ms/query | **38–223 µs/query** | k=1: 38/65/95/138 µs · k=5: 125/148/177/223 µs, at n=10/50/100/200 |
| Concurrent read throughput | see `examples/concurrency_bench.rs` output on your machine | | |

Two cautions on reading the query rows. First, the 0.5.x `nearest` figure is
the cost of an answer that was **often wrong** — the point-location grid named
a tile owner and the result was returned unverified (25 of 42 self-queries
wrong in a deep tree). Comparing against it measures the replacement of a
broken thing, not a tuning win. Second, 0.5.2 shipped correctness before speed
and its `nearest` cost ~2 ms; the 0.6.0 figures are what removed that.

The insert and delete rows have a simpler story: every insert used to compute
two Klein bisectors, reassign grid tiles, and pay one **exact** 37.5 µs
hyperbolic distance to maintain a bucket's pruning radius. All three are gone.

Honest note: geometric *primitives* run in nanoseconds; full-API spatial
queries still run in tens to hundreds of µs because every surviving candidate
is verified with the exact (0-ULP) hyperbolic distance. That is the guarantee,
not an inefficiency — the index's job is to make the number of survivors
small, and the ring bound is what proves it may stop.

## Semantic queries — the O(n×d) wall, and the index killing it

Before the semantic index (2026-07-09), `nearest_semantic` was a brute-force scan. After it,
(2026-07-10), stores ≥256 nodes route through a lazy per-slice VP-tree
(`docs/SEMANTIC_INDEX.md`). Criterion medians, k=10, d=8, same machine:

| Store size | brute-force scan | indexed, warm | speedup |
|---|---|---|---|
| 10 000 | 195 ms | 283 µs | ~690× |
| 50 000 | 968 ms | 311 µs | ~3 100× |
| 100 000 | 1 979 ms | 309 µs | ~6 400× |

(d=2 slices: 275 µs / 294 µs / 206 µs at 10k/50k/100k.) Warm-query cost is
flat-to-logarithmic in n — the O(n×d) wall is gone. Side-by-side on identical
10k-node data (`scan_vs_indexed_10k` bench): scan 161.8 ms vs indexed 267 µs.

**The honest costs.** The index is lazy: the *first* query for a slice after
any semantic write pays a full O(n log n) rebuild — measured **1.17 s** at
10k nodes, d=8 (`index_rebuild_10k` bench), ~7× one scan. It amortizes after
~7 queries on the same slice; a workload that interleaves every write with a
query is better off below the 256-node routing floor or batching its writes
(the punctuated calibrate-then-query model both paths are designed for).
Results are verified identical to the scan, byte for byte, ties broken by
`(distance, key)` (tests/semantic_index.rs).

Single-pair `semantic_distance` (d=8): 17 µs (unchanged primitive; the old
scan was almost exactly n × pair-cost).

## Quantized semantic storage (gFile v0.7.0, format v4) — sizes, test-asserted

Not criterion numbers — exact byte-count equalities asserted by
`gFile/tests/quantized.rs` (`on_disk_size_shrinks`, `wal_record_sizes`) at
40 semantic dims (16 reserved + 24 user, a production catalog shape):

| record | full-width | quantized (no GACL) | quantized + GACL |
|---|---|---|---|
| semantic tail (snapshot entry / INSERT / SET_SEMANTIC) | 640 B | **48 B (13.3×)** | 304 B (2.1×) |
| SET_SEMANTIC WAL record, 2-byte key | 651 B | 59 B | 315 B |
| 100-node compacted file delta | — | **−59,200 B exactly** | — |

The same shrink applies to temporal trajectory records (the "640 B full-vector
writes" known limit of temporal epochs falls to 48 B in quantized files). Honesty
ledger: the design estimate was "8×" — wrong in both directions (13.3×
without GACL because the reserved region is elided; 2.1× with GACL because
access bands deliberately stay full-width). The planned "zero-multiply
distance kernels" were dropped: g_math 0.4.24 has no quantized *distance*
kernel (dot products only, behind an `inference` feature requiring rayon).
Precision: user dims carry ~4.3 significant digits (step 1/19683);
determinism is byte-exact (own golden CRC 0xA636A63E; pre-quantization golden
0x044E5F72 unchanged — default OFF is byte-identical).

## gFile end-to-end operations (throughput bench, measured 2026-07-12, gFile v0.8.0)

`GMATH_PROFILE=embedded cargo bench --bench throughput` in gFile, same machine.
First recorded gFile-side numbers — per-op figures derived from batch medians.

| Operation | Measured | Per-op |
|---|---|---|
| `get` / `exists` / `get_meta` (500-key file) | 477 µs / 61 µs / 731 µs per 500 | **0.95 / 0.12 / 1.46 µs** |
| `put` via WAL, default durability (fsync per append) | 1.21 s per 100 | **~12 ms — fsync-bound (disk physics)** |
| `put` marginal into existing tree (geometry + WAL + fsync) | — | ~18 ms |
| WAL entry serialization (insert / +16 sem dims / delete) | — | 237 / 207 / 89 ns |
| Snapshot serialize 500 entries (raw / zstd) | 9.7 / 58.9 µs | ~20 / 118 ns per entry |
| Snapshot parse 500 entries (raw / zstd) | 128 / 148 µs | ~256 / 296 ns per entry |
| Cold open, FULL geometry (Sarkar rebuild) | 349 ms per 100 nodes | **~3.5 ms/node — use lazy/partial for big files** |
| `compact()` (≤100 nodes) | 17–25 ms | zstd adds ~4 ms at 100 |
| `nearest` (origin / off-center, ≤100 nodes) | — | 47–67 / 78–100 µs |
| `neighbors` k=3 (≤100 nodes) | — | 2.3–2.6 ms |
| GACL checks (permits / can_access / band decode) | — | 1.4 / 3.7 / 16 ns — free at query scale |
| File size, 100 nodes (raw / zstd) | 4 122 / 411 B | 10× zstd on structured data |

Honest reading: reads are hashmap-class; default-durability writes are
fsync-bound (~12 ms is the disk, not the code — `Relaxed` + final `compact()`
or a batch size is the documented bulk pattern, WAL serialization itself is
~0.2 µs/entry); full-geometry cold open costs ~3.5 ms/node (the Sarkar
rebuild), which is exactly why `lazy_geometry`/`partial_reads` and embed-on-demand
embed-on-demand exist.

## Semantic disk, 10k nodes (`semantic_disk_10k` bench)

| Query | before proxy search | after proxy search | speedup |
|---|---|---|---|
| `disk.nearest`, warm, k=10 | 4.5 ms | **0.73 ms** | 6.2× (proxy search 1.21 ms → one-sqrt winner finalization 0.73 ms) |
| `concept_of` (classification) | 70.8 µs | **24.0 µs** | 2.9× |

Proxy search (squared Möbius ratio): candidate scoring costs a dot
product + one division (~4.3 µs) instead of the exact hyperbolic kernel
(~78 µs — dominated by four fixed-point sqrts at ~15 µs each, NOT the
atanh, which is ~16 µs; the "~200 ns ratio" code comment was wrong by
300×). Pruning runs at near-exact strength in
ratio space via the tanh subtraction identity,
`d_q − m > τ ⟺ (r_q−r_m) > r_τ(1−r_q·r_m)`, squared so that the single
irrational term `√(s_q·s_m)` is replaced by a **guaranteed one-sided
integer Newton upper bound** (power-of-two seed from the bit length + two
Newton steps, ~0.18% slack) — NO floats anywhere in the compute path, no
error pad needed, determinism verifiable by construction. The proxy
diagnostic (`examples/disk_nn_diag.rs`) shows 97 visited nodes of 10 000
(exact-sqrt pruning would visit ~60; the ~15% total-time premium is the
price of float purity). The k winners are re-verified with the
exact kernel so results stay identical to a brute-force exact scan, and
the one-sqrt squared-space form halved that kernel (78 → 35.4 µs), taking
the warm query to 0.73 ms. The fused kernels (`euclidean_distance_squared`,
`dot`, `mobius_denominator_sq`) are implemented in gMath 0.4.26 (local,
unpublished); adopting them once published moves the remaining hand-rolled
storage-tier accumulators (hash_table `euclidean_distance_sq`, klein
`power_distance`) onto wrap-proof compute-tier kernels. Two rejected designs, recorded honestly: polynomial
bounds on atanh (`d ≥ 2r` saturates at 2 → pruning collapses at medians
≥ 1.4 → measured 74 ms), and an ε-pad of 1e-4 (swamped dense clouds whose
r-differences are ~1e-6 → 9 856 of 10 000 nodes visited → 49 ms). A third
rejected design used IEEE f64 sqrt for the pruning comparisons (fast,
correctly rounded, arguably deterministic) — rejected on principle: the
no-floats claim must be verifiable by construction, not by trusting libm.
The semantic-disk barycenter derivation also lost its per-node sqrts (anchor γ
factors cached at build): ~75 µs → ~30 µs per node.

For byte-locality semantic queries at scale, see Horon's meaning-addressed
mode (`horon/BENCHMARKS.md`): windowed queries touch a small fraction of
entries instead of scanning all of them.
