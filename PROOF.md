# Delaunay-Preserving Leaf Insertion in Sarkar-Embedded Trees

## Theorem (Self-Indexing Dynamic Trees)

Let T be a weighted tree embedded in the hyperbolic plane H² via Sarkar's
construction with scale factor τ. Let T' = T ∪ {l, (p,l)} where l is a
new leaf with parent p. If

    τ ≥ −log(tan(π / (2·d_max')))

where d_max' is the maximum vertex degree in T', then:

1. The edge (p, l) is Delaunay in the embedding of T'.
2. Every edge of T that was Delaunay remains Delaunay in T'.
3. No non-tree edge becomes Delaunay.

Consequently, the tree IS its own Delaunay triangulation after insertion,
maintained with O(1) computational work (one Möbius reflection).

---

## Definitions

**Poincaré disk model.** Points in the open unit disk D² ⊂ ℝ² with
hyperbolic distance:

    d_H(p, q) = 2·atanh(|p − q| / √(1 − 2⟨p,q⟩ + |p|²|q|²))

**Sarkar embedding.** Recursive construction (Sarkar 2011, GD):
- Root at origin.
- Each child placed at hyperbolic distance τ from parent.
- Children distributed in angular cones of width 2π/d(v) at each vertex v.
- Subtrees reflected to origin via Möbius isometry, then recurse.

**Midpoint witness disk.** For edge (u, v) at distance τ: the hyperbolic
disk centered at the midpoint Q of the geodesic uv, with radius τ/2.
Sarkar proves this disk is empty of all other vertices, witnessing that
(u, v) is Delaunay.

**Cone containment (Sarkar Lemma 2).** Under the τ-degree condition, the
Voronoi cell of every descendant of v is contained within v's angular cone
at v's parent. This ensures Voronoi cells of different subtrees are disjoint,
so only tree edges can be Delaunay.

---

## Key Lemma: Midpoint Exclusion

**Lemma.** Let p, q be points in H² with d_H(p, q) = τ > 0. Let Q be
their hyperbolic midpoint, so d_H(p, Q) = d_H(q, Q) = τ/2. Let l be
any point with d_H(p, l) = τ that makes angle θ > 0 with the geodesic
pq at vertex p. Then:

    d_H(Q, l) > τ/2

That is, l lies strictly outside the midpoint witness disk of edge (p, q).

**Proof.** Apply the hyperbolic law of cosines in triangle QpL:

    cosh(d(Q, l)) = cosh(d(Q, p))·cosh(d(p, l)) − sinh(d(Q, p))·sinh(d(p, l))·cos θ

Substituting d(Q, p) = τ/2 and d(p, l) = τ:

    cosh(d(Q, l)) = cosh(τ/2)·cosh(τ) − sinh(τ/2)·sinh(τ)·cos θ

We require cosh(d(Q, l)) > cosh(τ/2), i.e.:

    cosh(τ/2)·cosh(τ) − sinh(τ/2)·sinh(τ)·cos θ > cosh(τ/2)

    cosh(τ/2)·(cosh(τ) − 1) > sinh(τ/2)·sinh(τ)·cos θ

Apply the identities cosh(τ) − 1 = 2·sinh²(τ/2) and sinh(τ) = 2·sinh(τ/2)·cosh(τ/2):

    cosh(τ/2)·2·sinh²(τ/2) > sinh(τ/2)·2·sinh(τ/2)·cosh(τ/2)·cos θ

    2·sinh²(τ/2)·cosh(τ/2) > 2·sinh²(τ/2)·cosh(τ/2)·cos θ

Divide both sides by 2·sinh²(τ/2)·cosh(τ/2) > 0 (since τ > 0):

    1 > cos θ

This holds for all θ ∈ (0, 2π). ∎

**Remark.** The lemma is unconditional: it requires no bound on τ, no
bound on θ (beyond θ ≠ 0), and no special structure. Any point at the
same hyperbolic distance from an edge endpoint, at any nonzero angle,
is excluded from the midpoint witness disk.

---

## Proof of the Theorem

### Setup

Let T be embedded via Sarkar's construction with scale τ satisfying the
τ-degree condition. Denote the embedding map φ. Insert leaf l as a child
of existing node p, placed by the Sarkar construction:

- In p's reflected frame (p at origin): l is at Euclidean radius
  r = tanh(τ/2) at angle θ_l (golden-angle offset from existing children).
- In the global frame: l = φ(p) ⊕ l_origin (Möbius reflection).

The embedding distances are:
- d_H(p, l) = τ (exact, by construction)
- d_H(l, u) ≥ 2τ/(1+ε) for all u ≠ p (Sarkar distortion lower bound)

where ε < 1 for any τ satisfying the τ-degree condition.

### Part 1: The new edge (p, l) is Delaunay.

We show the midpoint witness disk D of (p, l) — centered at Q with
radius τ/2 — contains no node of T.

**Case 1a: Node u is adjacent to p in T (existing child or parent).**

d_H(p, u) = τ. The angle ∠upQ at vertex p equals the angle ∠upl
(since Q lies on the geodesic from p toward l). This angle is nonzero
because l was placed at a distinct angular position from all existing
children and from p's parent direction.

By the Midpoint Exclusion Lemma (with the roles of l and q swapped —
here u is at distance τ from p at angle ∠upl > 0 from the geodesic pl):

    d_H(u, Q) > τ/2

So u ∉ D. ∎

**Case 1b: Node u is not adjacent to p in T.**

d_H(u, p) ≥ 2τ/(1+ε) > τ (since ε < 1). By the reverse triangle
inequality:

    d_H(u, Q) ≥ d_H(u, p) − d_H(p, Q) > τ − τ/2 = τ/2

So u ∉ D. ∎

### Part 2: Existing edges remain Delaunay.

Let (u, v) be an existing tree edge with midpoint witness disk D_{uv}
(center Q_{uv}, radius τ/2). We show l ∉ D_{uv}.

**Case 2a: One endpoint is p.** Say v = p, so (u, p) is an existing
edge. The angle ∠upl at vertex p is nonzero (l is at a distinct
angular position from u).

By the Midpoint Exclusion Lemma (l at distance τ from p, at angle
∠upl > 0 from the geodesic pu):

    d_H(l, Q_{up}) > τ/2

So l ∉ D_{up}. ∎

**Case 2b: Neither endpoint is p.**

d_H(l, u) ≥ 2τ/(1+ε) and d_H(l, v) ≥ 2τ/(1+ε) (by Sarkar distortion;
tree distance from l to u or v is ≥ 2). Since Q_{uv} is at distance
τ/2 from u:

    d_H(l, Q_{uv}) ≥ d_H(l, u) − d_H(u, Q_{uv}) ≥ 2τ/(1+ε) − τ/2

For ε < 1: 2τ/(1+ε) > τ, so 2τ/(1+ε) − τ/2 > τ/2.

So l ∉ D_{uv}. ∎

### Part 3: Non-tree edges remain non-Delaunay.

By the cone containment lemma (Sarkar Lemma 2), l's Voronoi cell is
contained within the angular cone assigned to l's subtree at p. This
cone is a subset of p's cone at p's parent.

For any non-tree edge (l, w) where w ≠ p: w's Voronoi cell is either
in a different subtree's cone (disjoint from l's cone by cone separation)
or is separated from l by p's Voronoi cell (for w = ancestor of p).
In either case, l and w are not Voronoi-adjacent, so (l, w) is not
Delaunay.

For any existing non-tree edge (u, w): these were not Delaunay before
the insertion (by Sarkar's theorem on T). Adding l only modifies the
Voronoi cells in the local neighborhood of l (within p's cone). Since
(u, w) was not Delaunay before and the Voronoi cells of u and w are
unchanged outside p's cone, (u, w) remains non-Delaunay. ∎

---

## The τ-Degree Condition

The condition τ ≥ −log(tan(π/(2·d_max))) ensures cone containment. For
golden-angle child spacing with n children, the effective cone half-angle
is α ≈ π/n, giving:

| Max degree | Min τ  | Max tree depth (Q64.64) |
|------------|--------|-------------------------|
| 4          | 0.88   | ~50                     |
| 8          | 1.62   | ~27                     |
| 16         | 2.32   | ~19                     |
| 32         | 3.01   | ~14                     |
| 64         | 3.71   | ~11                     |
| 128        | 4.40   | ~10                     |

Max tree depth ≈ 44/τ (Q64.64 precision limit: tanh^{depth}(τ/2) → 1).

For dynamic trees where the maximum degree is not known in advance,
set τ = log(2·d_budget/π) where d_budget is the maximum degree you
wish to support. The penalty is reduced maximum tree depth.

---

## Implications: The Index-Data Isomorphism

This theorem establishes that in a Sarkar-embedded tree:

**The tree IS the index.** The Delaunay triangulation equals the tree
at all times, through arbitrary leaf insertions. The Voronoi diagram
(dual of Delaunay) is maintained automatically — no separate spatial
index is structurally necessary.

**O(1) insertion.** Adding a leaf requires:
1. One Möbius reflection (compute child coordinates): O(d) where d is
   the space dimension (typically 2-4).
2. One spatial registration (VP-tree buffer insert): O(1).
3. Zero recomputation of existing embeddings: all existing coordinates
   are unchanged.

**Nearest-neighbor via tree structure.** Since the tree equals the
Delaunay graph, the nearest neighbor of any embedded point can be
found by navigating the tree structure. For a query point q:
- Find the Voronoi cell containing q (point location in the Voronoi
  diagram = power diagram in the Klein model, by Nielsen 2009).
- The cell's generator is the nearest tree node.

**What is novel.** Each component exists independently:
- Sarkar 2011: static tree = Delaunay triangulation.
- Cvetkovski-Crovella 2009: dynamic hyperbolic embeddings.
- Nielsen 2009: hyperbolic Voronoi = Euclidean power diagram.
- Guibas-Knuth-Sharir 1992: O(1) expected Delaunay updates.

The synthesis is new: **a dynamic tree data structure that provably
maintains the identity Tree = Delaunay Triangulation = Voronoi Index
under leaf insertion, with O(1) update cost and zero perturbation to
existing embeddings.** This unifies the data structure and its spatial
index into a single geometric object.

---

## References

- Sarkar, R. (2011). Low Distortion Delaunay Embedding of Trees in
  Hyperbolic Plane. GD 2011, LNCS 7034, pp. 355-366.
- Cvetkovski, A. & Crovella, M. (2009). Hyperbolic Embedding and
  Routing for Dynamic Graphs. INFOCOM 2009.
- Nielsen, F. & Nock, R. (2009). Hyperbolic Voronoi Diagrams Made Easy.
  ICCSA 2009.
- Bogdanov, M., Devillers, O. & Teillaud, M. (2014). Hyperbolic
  Delaunay Complexes and Voronoi Diagrams Made Practical. JoCG 5(1).
- Despre, V., Schlenker, J-M. & Teillaud, M. (2020). Flipping Geometric
  Triangulations on Hyperbolic Surfaces. SoCG 2020.
- Guibas, L., Knuth, D. & Sharir, M. (1992). Randomized Incremental
  Construction of Delaunay and Voronoi Diagrams. Algorithmica 7, 381-413.
- Kisfaludi-Bak, S. et al. (2024). Dynamic Approximate Nearest Neighbor
  Search in Hyperbolic Space. SoCG 2024.
- Sala, F. et al. (2018). Representation Tradeoffs for Hyperbolic
  Embeddings. ICML 2018.
