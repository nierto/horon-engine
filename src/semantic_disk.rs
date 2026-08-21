//! Semantic disk — taxonomy-embedded meaning space (E+D design).
//!
//! Design: `docs/SEMANTIC_DISK.md` (ratified 2026-07-11). The domain's
//! concept taxonomy — derived from structure within the data (a category
//! tree, a directory tree) — is Sarkar-embedded into its own Poincaré disk
//! by inserting the concept paths into a private [`Store`]. Every data
//! node's position in that disk is a **pure function** of its affinity
//! dimensions: the weighted Klein barycenter (Einstein midpoint) of the
//! concept anchors. Nothing is stored; the position can never disagree
//! with the dims; and because the mapping is deterministic, an epoch
//! record of the dims replays the node's path through meaning-space
//! bit-identically.
//!
//! Three query families fall out:
//! - [`SemanticDisk::concept_of`] — which concept does this node belong to
//!   *right now* (hyperbolic Voronoi cell location among the anchors;
//!   constant in data-node count). Compared with the node's storage path,
//!   this is miscategorization detection as a primitive.
//! - [`SemanticDisk::nearest`] — k nearest data nodes in meaning-space
//!   (hyperbolic distance between derived positions; the semantic index's metric tree with
//!   [`crate::metric_tree::HyperbolicMetric`], epoch-cached).
//! - [`SemanticDisk::classify_trajectory`] — a temporal trajectory readout
//!   pushed through the anchor cells: a moving point becomes a sequence of
//!   discrete meaning-states across epochs.
//!
//! The disk is a standalone object the application owns (anchors are few —
//! dozens, not thousands — so building one is cheap). `Store` gains no new
//! state. The (concept path ↔ affinity dim) mapping is **calibration**:
//! fixed for a dataset's life, like the dimensional schema itself.

use std::sync::{Arc, Mutex};

use g_math::fixed_point::FixedPoint;

use crate::hyperbolic_geometry::HyperbolicPoint;
use crate::klein::{self, KleinPoint};
use crate::metric_tree::{CachedNormPoint, HyperbolicMetric, MetricVpTree};
use crate::store::{Store, StoreError};
use crate::tensor_network::HyperbolicTensorNetwork;

/// A concept anchor: its taxonomy path, the affinity dimension that weights
/// it, and its embedded site in the concept disk.
struct Anchor {
    path: String,
    dim: usize,
    site: KleinPoint,
    /// Cached Einstein-midpoint factor γ = 1/√(1−‖site‖²): the barycenter
    /// needs it per (node × anchor), and it never changes — caching it
    /// removes every sqrt from position derivation (measured: ~75 µs/node
    /// → ~5 µs/node at 5 anchors).
    gamma: FixedPoint,
}

/// The taxonomy-embedded meaning space (see module docs).
pub struct SemanticDisk {
    /// Mapped anchors, sorted by concept path (deterministic order for the
    /// barycenter accumulation and all downstream results). The embedding
    /// store used to place them is dropped after build — the anchor sites
    /// are the complete geometry.
    anchors: Vec<Anchor>,
    /// Derived-position NN index, tagged with the data store's semantic
    /// epoch it was built at (the semantic index's invalidation model). Entries cache
    /// their squared norms so the proxy search never pays a sqrt.
    cache: Mutex<Option<(u64, Arc<MetricVpTree<CachedNormPoint>>)>>,
}

impl SemanticDisk {
    /// Build a semantic disk from a concept specification: `(concept path,
    /// affinity dim)` pairs — e.g. `[("/trauma", 16), ("/cgt", 17), …]`.
    /// Nested paths are allowed and embed with their real tree shape;
    /// missing ancestors are created automatically (they become unmapped,
    /// purely structural anchors).
    ///
    /// Errors on an empty spec, duplicate concept paths, or duplicate dims.
    pub fn build(spec: &[(&str, usize)]) -> Result<Self, StoreError> {
        if spec.is_empty() {
            return Err(StoreError::InvalidOperation(
                "semantic disk spec must name at least one concept".to_string(),
            ));
        }
        let mut pairs: Vec<(String, usize)> = spec
            .iter()
            .map(|(p, d)| (normalize_concept_path(p), *d))
            .collect();
        pairs.sort();
        for w in pairs.windows(2) {
            if w[0].0 == w[1].0 {
                return Err(StoreError::InvalidOperation(format!(
                    "duplicate concept path in spec: {}",
                    w[0].0
                )));
            }
        }
        {
            let mut dims: Vec<usize> = pairs.iter().map(|(_, d)| *d).collect();
            dims.sort_unstable();
            if dims.windows(2).any(|w| w[0] == w[1]) {
                return Err(StoreError::InvalidOperation(
                    "duplicate affinity dim in spec: each concept needs its own dimension"
                        .to_string(),
                ));
            }
        }

        // Embed the taxonomy: insert concept paths (ancestors first) into a
        // private store. Sorted order is the deterministic insertion order.
        let taxonomy = Store::new();
        for (path, _) in &pairs {
            for ancestor in ancestors_of(path) {
                if !taxonomy.exists(&ancestor) {
                    taxonomy.put(&ancestor, ancestor.as_bytes())?;
                }
            }
            if !taxonomy.exists(path) {
                taxonomy.put(path, path.as_bytes())?;
            }
        }

        // Resolve anchor sites (Klein coordinates of the embedded concepts)
        // and cache their Einstein factors.
        let one = FixedPoint::from_int(1);
        let mut anchors = Vec::with_capacity(pairs.len());
        for (path, dim) in pairs {
            let point = taxonomy.position_fixed(&path)?;
            let site = klein::poincare_to_klein(&point);
            let radicand = if site.weight > crate::constants::small_epsilon() {
                site.weight
            } else {
                crate::constants::small_epsilon()
            };
            let gamma = one / radicand.sqrt();
            anchors.push(Anchor { path, dim, site, gamma });
        }

        Ok(Self { anchors, cache: Mutex::new(None) })
    }

    /// The mapped concept paths, in canonical (sorted) order.
    pub fn concepts(&self) -> Vec<&str> {
        self.anchors.iter().map(|a| a.path.as_str()).collect()
    }

    /// A node's derived position in the concept disk, as f64 Poincaré
    /// coordinates. `Ok(None)` when the node has no positive affinity on
    /// any mapped dim (no concept position — same convention as empty
    /// semantic coords).
    pub fn position_of(&self, store: &Store, key: &str) -> Result<Option<Vec<f64>>, StoreError> {
        Ok(self
            .derive_from_coords(&store.get_semantic(key)?)
            .map(|p| p.coords().iter().map(|c| c.to_f64()).collect()))
    }

    /// Which concept a node belongs to **right now**: the anchor whose
    /// hyperbolic Voronoi cell contains the node's derived position.
    /// Constant in data-node count (linear only in the anchor count —
    /// dozens).
    /// `Ok(None)` when the node has no concept position.
    ///
    /// Compared against the node's storage path, this is the
    /// miscategorization primitive: filed under `/overig`, classifies to
    /// `/trauma`.
    pub fn concept_of(&self, store: &Store, key: &str) -> Result<Option<String>, StoreError> {
        Ok(self
            .derive_from_coords(&store.get_semantic(key)?)
            .and_then(|p| self.classify_point(&p)))
    }

    /// The k data nodes nearest to `key` in the concept disk (hyperbolic
    /// distance between derived positions), excluding `key` itself.
    /// Sorted ascending by `(distance, key)`.
    pub fn nearest(
        &self,
        store: &Store,
        key: &str,
        k: usize,
    ) -> Result<Vec<(String, f64)>, StoreError> {
        let Some(query) = self.derive_from_coords(&store.get_semantic(key)?) else {
            return Err(StoreError::InvalidOperation(format!(
                "{} has no concept position (no positive affinity on any mapped dim)",
                key
            )));
        };
        let index = self.index(store)?;
        Ok(index
            .knn(&CachedNormPoint::new(query), k + 1, &HyperbolicMetric)
            .into_iter()
            .filter(|(id, _)| id != key)
            .take(k)
            .map(|(id, d)| (id, d.to_f64()))
            .collect())
    }

    /// The k data nodes nearest to an explicit affinity-weight vector
    /// (one weight per mapped concept, in [`Self::concepts`] order).
    pub fn nearest_to_weights(
        &self,
        store: &Store,
        weights: &[f64],
        k: usize,
    ) -> Result<Vec<(String, f64)>, StoreError> {
        if weights.len() != self.anchors.len() {
            return Err(StoreError::InvalidOperation(format!(
                "expected {} weights (one per mapped concept), got {}",
                self.anchors.len(),
                weights.len()
            )));
        }
        let fixed: Vec<FixedPoint> = weights.iter().map(|w| FixedPoint::from_f64(*w)).collect();
        let Some(query) = self.derive_from_weights(&fixed) else {
            return Err(StoreError::InvalidOperation(
                "no positive weight supplied — the query has no concept position".to_string(),
            ));
        };
        let index = self.index(store)?;
        Ok(index
            .knn(&CachedNormPoint::new(query), k, &HyperbolicMetric)
            .into_iter()
            .map(|(id, d)| (id, d.to_f64()))
            .collect())
    }

    /// Push a temporal trajectory readout through the anchor cells: for each
    /// `(epoch, coords)` sample — the shape `HttHistory::trajectory`
    /// returns — classify the derived position, yielding the node's
    /// **symbolic trajectory** (`(epoch, concept)` pairs). Samples whose
    /// weights are all non-positive are omitted (no position at that
    /// epoch).
    ///
    /// `sample_dim_start` is the first dimension index the samples cover
    /// (the `dim_range.start` the trajectory was read with); mapped dims
    /// outside the sampled range weigh zero.
    pub fn classify_trajectory(
        &self,
        sample_dim_start: usize,
        samples: &[(u64, Vec<f64>)],
    ) -> Vec<(u64, String)> {
        samples
            .iter()
            .filter_map(|(epoch, values)| {
                let weights: Vec<FixedPoint> = self
                    .anchors
                    .iter()
                    .map(|a| {
                        a.dim
                            .checked_sub(sample_dim_start)
                            .and_then(|i| values.get(i))
                            .map_or(FixedPoint::from_int(0), |v| FixedPoint::from_f64(*v))
                    })
                    .collect();
                let point = self.derive_from_weights(&weights)?;
                self.classify_point(&point).map(|c| (*epoch, c))
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// Decode a node's mapped affinity dims from raw semantic bytes and
    /// derive its concept-disk position.
    fn derive_from_coords(&self, coords: &[u8]) -> Option<HyperbolicPoint> {
        if coords.is_empty() {
            return None;
        }
        let weights: Vec<FixedPoint> = self
            .anchors
            .iter()
            .map(|a| {
                HyperbolicTensorNetwork::decode_semantic_slice(coords, &(a.dim..a.dim + 1))[0]
            })
            .collect();
        self.derive_from_weights(&weights)
    }

    /// Weighted Klein barycenter of the anchors (negatives ignored),
    /// using the cached per-anchor γ factors — no sqrt per node. Same
    /// formula as [`klein::weighted_barycenter`] (property-tested there);
    /// `None` when no weight is positive.
    fn derive_from_weights(&self, weights: &[FixedPoint]) -> Option<HyperbolicPoint> {
        let zero = FixedPoint::from_int(0);
        let one = FixedPoint::from_int(1);
        let mut denom = zero;
        let mut numer: Option<g_math::fixed_point::FixedVector> = None;
        for (a, w) in self.anchors.iter().zip(weights) {
            if *w <= zero {
                continue;
            }
            let coeff = *w * a.gamma;
            let dim = a.site.dimension();
            let acc = numer.get_or_insert_with(|| g_math::fixed_point::FixedVector::new(dim));
            for i in 0..dim {
                acc[i] += a.site.coords[i] * coeff;
            }
            denom += coeff;
        }
        let numer = numer?;
        if denom <= zero {
            return None;
        }
        let inv = one / denom;
        let dim = numer.len();
        let mut coords = g_math::fixed_point::FixedVector::new(dim);
        for i in 0..dim {
            coords[i] = numer[i] * inv;
        }
        Some(klein::klein_to_poincare(&KleinPoint::new(coords)))
    }

    /// Hyperbolic Voronoi cell location among the anchor sites.
    ///
    /// The cell of anchor *i* is `{x : d_H(x, k_i) ≤ d_H(x, k_j) ∀ j}`. In
    /// Klein coordinates
    ///
    /// ```text
    /// cosh d_H(x, k_i) = (1 − ⟨x, k_i⟩) · γ_i / √(1 − ‖x‖²),   γ_i = 1/√(1 − ‖k_i‖²)
    /// ```
    ///
    /// and the `√(1 − ‖x‖²)` factor is common to every anchor, so the cell is
    /// decided by `argmin_i (1 − ⟨x, k_i⟩)·γ_i` — the Nielsen affine
    /// reduction of the hyperbolic Voronoi diagram, reusing the same cached
    /// γ the barycenter already needs. One dot product per anchor: no sqrt,
    /// no division, the same cost class as the Euclidean power distance it
    /// replaced, and exact for **any** anchor placement.
    ///
    /// Anchors sitting at unequal Klein norms — which is every nested
    /// taxonomy, since Sarkar places deeper concepts further out — are why
    /// this has to be the true reduction. Scoring them by
    /// `‖x − k_i‖² − (1 − ‖k_i‖²)` instead agrees with `d_H` only when all
    /// γ_i are equal (a flat, single-depth spec); otherwise a shallow anchor
    /// can swallow a deeper anchor's own site, breaking single-anchor
    /// identity.
    ///
    /// Ties keep the first anchor in canonical path order (deterministic).
    fn classify_point(&self, point: &HyperbolicPoint) -> Option<String> {
        let query = klein::poincare_to_klein(point);
        let one = FixedPoint::from_int(1);
        let mut best: Option<(usize, FixedPoint)> = None;
        for (i, a) in self.anchors.iter().enumerate() {
            let score = (one - query.coords.dot(&a.site.coords)) * a.gamma;
            let better = match &best {
                None => true,
                Some((_, incumbent)) => score < *incumbent,
            };
            if better {
                best = Some((i, score));
            }
        }
        best.map(|(i, _)| self.anchors[i].path.clone())
    }

    /// The derived-position NN index, rebuilt lazily when the data store's
    /// semantic epoch has advanced (the semantic index's invalidation model, one layer up).
    fn index(&self, store: &Store) -> Result<Arc<MetricVpTree<CachedNormPoint>>, StoreError> {
        let epoch = store.semantic_epoch();
        {
            let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((tagged, tree)) = cache.as_ref() {
                if *tagged == epoch {
                    return Ok(Arc::clone(tree));
                }
            }
        }

        // Build outside the lock (racing builders produce identical trees).
        let build_epoch = store.semantic_epoch();
        let mut keys = store.list("/")?;
        keys.sort();
        let entries: Vec<(String, CachedNormPoint)> = keys
            .into_iter()
            .filter_map(|key| {
                let coords = store.get_semantic(&key).ok()?;
                let point = self.derive_from_coords(&coords)?;
                Some((key, CachedNormPoint::new(point)))
            })
            .collect();
        let tree = Arc::new(MetricVpTree::build(entries, &HyperbolicMetric));

        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        *cache = Some((build_epoch, Arc::clone(&tree)));
        Ok(tree)
    }
}

/// Normalize a concept path: ensure a leading `/`, strip a trailing one.
fn normalize_concept_path(path: &str) -> String {
    let mut p = if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{}", path)
    };
    while p.len() > 1 && p.ends_with('/') {
        p.pop();
    }
    p
}

/// Proper ancestors of a normalized path, shallowest first
/// (`/a/b/c` → `/a`, `/a/b`).
fn ancestors_of(path: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut idx = 1;
    while let Some(next) = path[idx..].find('/') {
        out.push(path[..idx + next].to_string());
        idx += next + 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_helpers() {
        assert_eq!(normalize_concept_path("trauma"), "/trauma");
        assert_eq!(normalize_concept_path("/a/b/"), "/a/b");
        assert_eq!(ancestors_of("/a"), Vec::<String>::new());
        assert_eq!(ancestors_of("/a/b/c"), vec!["/a".to_string(), "/a/b".to_string()]);
    }

    #[test]
    fn build_rejects_bad_specs() {
        assert!(SemanticDisk::build(&[]).is_err());
        assert!(SemanticDisk::build(&[("/a", 16), ("/a", 17)]).is_err());
        assert!(SemanticDisk::build(&[("/a", 16), ("/b", 16)]).is_err());
    }

    #[test]
    fn anchors_are_sorted_and_embedded() {
        let disk = SemanticDisk::build(&[("/zeta", 18), ("/alpha", 16), ("/mid", 17)]).unwrap();
        assert_eq!(disk.concepts(), vec!["/alpha", "/mid", "/zeta"]);
        // Anchors are distinct embedded sites.
        for w in disk.anchors.windows(2) {
            assert!(w[0].site.coords[0] != w[1].site.coords[0]
                || w[0].site.coords[1] != w[1].site.coords[1]);
        }
    }
}
