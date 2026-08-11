# 🌳 Hierarchical Clustering Cheat Sheet

> **Import convention:** `from sklearn.cluster import AgglomerativeClustering` | `from scipy.cluster.hierarchy import dendrogram, linkage, fcluster`
>
> **Category:** Hierarchical (agglomerative) clustering — scikit-learn (core) + SciPy (core, for dendrograms)

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Agglomerative vs Divisive](#-agglomerative-vs-divisive)
- [How Agglomerative Clustering Works](#-how-agglomerative-clustering-works)
- [Linkage Methods](#-linkage-methods)
- [Distance Metrics](#-distance-metrics)
- [scikit-learn Quick Syntax](#-scikit-learn-quick-syntax)
- [SciPy Quick Syntax (Dendrograms)](#-scipy-quick-syntax-dendrograms)
- [Choosing the Number of Clusters](#-choosing-the-number-of-clusters)
- [Connectivity Constraints](#-connectivity-constraints)
- [Feature Agglomeration](#-feature-agglomeration)
- [Comparison with Partitioning Methods](#-comparison-with-partitioning-methods)
- [Use Cases](#-use-cases)
- [Cautions & Gotchas](#-cautions--gotchas)
- [Quick Reference Summary](#-quick-reference-summary)

---

## 🔎 Overview

**Hierarchical clustering** builds a tree of nested clusters (a **dendrogram**) rather than a single flat partition. You don't choose `k` up front — you build the full tree, then decide where to cut it. scikit-learn's `AgglomerativeClustering` fits the tree and returns flat labels; SciPy's `scipy.cluster.hierarchy` builds and visualizes the tree itself and offers more flexibility in cutting it after the fact.

> 💡 **Tip:** Use SciPy when you want to *see* the dendrogram and decide the cut interactively. Use scikit-learn's `AgglomerativeClustering` when you want flat cluster labels as part of a pipeline (e.g., feeding into `Pipeline`, `GridSearchCV`, or alongside other estimators).

---

## 🌲 Agglomerative vs Divisive

| Approach | Direction | Availability |
|---|---|---|
| **Agglomerative** (bottom-up) | Starts with every point as its own cluster, repeatedly merges the closest pair | `sklearn.cluster.AgglomerativeClustering`, `scipy.cluster.hierarchy.linkage` — both use this |
| **Divisive** (top-down) | Starts with all points in one cluster, repeatedly splits the least cohesive cluster | Not implemented in scikit-learn or SciPy; rarely used in practice due to cost |

> ⚠️ **Warning:** "Hierarchical clustering" in scikit-learn and SciPy always means *agglomerative*. There is no built-in divisive implementation — if you see "divisive" in a course or textbook, it's usually a conceptual comparison, not something you'll call directly.

---

## ⚙️ How Agglomerative Clustering Works

1. Start with every observation as its own singleton cluster
2. Compute pairwise distances between all clusters
3. Merge the two closest clusters into one
4. Recompute distances from the new cluster to all others, using a **linkage** rule
5. Repeat steps 2–4 until everything is merged into a single cluster (or a stopping condition is met)

The full merge sequence is the dendrogram. Cutting it at a given height (or asking for a specific number of clusters) yields a flat partition.

> 💡 **Tip:** Because every merge is recorded, you get the full hierarchy "for free" — useful when you're not sure what k should be, or when clusters-within-clusters is a meaningful structure for your domain (e.g., taxonomies, organizational hierarchies, phylogenies).

---

## 🔗 Linkage Methods

Linkage defines how the distance between two *clusters* (not just two points) is computed once a cluster has more than one member.

| Linkage | Definition | Tends to produce | Notes |
|---|---|---|---|
| `'ward'` | Minimizes the increase in total within-cluster variance after merging | Compact, similarly-sized clusters | **Euclidean distance only**; scikit-learn default |
| `'complete'` (max) | Distance between the farthest pair of points across two clusters | Compact, tight clusters; resistant to chaining | Sensitive to outliers |
| `'average'` | Mean distance between all cross-cluster point pairs | A balance between single and complete | Works with any metric |
| `'single'` (min) | Distance between the closest pair of points across two clusters | Elongated, "chained" clusters — good for non-convex shapes | Prone to the **chaining effect** — one noisy bridge point can merge otherwise-distinct clusters |

> ✅ **Best practice:** Start with `'ward'` for general-purpose, roughly spherical clusters on scaled continuous data. Switch to `'average'` or `'complete'` if you need a non-Euclidean metric. Try `'single'` only when you specifically expect elongated or chain-like cluster shapes.

---

## 📏 Distance Metrics

| Metric | Compatible linkage | Notes |
|---|---|---|
| `'euclidean'` | All (`ward`, `complete`, `average`, `single`) | Default; required for `ward` |
| `'manhattan'`, `'cosine'`, `'l1'`, `'l2'`, others | `complete`, `average`, `single` only | Set via `metric=` in scikit-learn ≥ 1.2 (was `affinity=` in older versions) |
| `'precomputed'` | `complete`, `average`, `single` | Pass a full distance matrix instead of raw features |

> ⚠️ **Warning:** `ward` linkage mathematically requires Euclidean distance — scikit-learn will raise an error if you combine `linkage='ward'` with a non-Euclidean `metric`.

---

## 🔵 scikit-learn Quick Syntax

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

# Flat clustering with a fixed number of clusters
agg = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward',       # 'ward', 'complete', 'average', 'single'
    metric='euclidean'     # only used when linkage != 'ward'
)
labels = agg.fit_predict(X_scaled)

agg.labels_               # cluster assignment per observation
agg.n_leaves_               # number of original observations
agg.n_connected_components_  # 1 if fully connected (no connectivity constraint)
```

```python
# Cut by distance threshold instead of a fixed k
agg_thresh = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=5.0,   # merges beyond this linkage distance stop
    linkage='ward',
    compute_distances=True     # needed to inspect/plot merge distances afterward
)
labels = agg_thresh.fit_predict(X_scaled)
print(f"Number of clusters found: {agg_thresh.n_clusters_}")
```

> 💡 **Tip:** `n_clusters` and `distance_threshold` are mutually exclusive — set exactly one, and set the other to `None`. Use `compute_distances=True` any time you want to plot a dendrogram from a fitted `AgglomerativeClustering` object.

---

## 🌿 SciPy Quick Syntax (Dendrograms)

SciPy's `linkage` builds the same kind of merge tree, returned as a `(n-1, 4)` linkage matrix, and pairs naturally with `dendrogram` for visualization.

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Build the linkage matrix
Z = linkage(X_scaled, method='ward')   # method: 'ward','complete','average','single'

# Each row of Z: [cluster_1_idx, cluster_2_idx, merge_distance, num_original_points_in_new_cluster]
print(Z.shape)   # (n_samples - 1, 4)

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='lastp', p=15)   # show only the last 15 merges for readability
plt.xlabel("Cluster size or sample index")
plt.ylabel("Merge distance")
plt.show()

# Cut the tree to get flat labels
labels_by_k = fcluster(Z, t=3, criterion='maxclust')       # exactly 3 clusters
labels_by_dist = fcluster(Z, t=5.0, criterion='distance')   # cut at height 5.0
```

> 💡 **Tip:** `fcluster(..., criterion='maxclust')` and `AgglomerativeClustering(n_clusters=k)` should agree on cluster assignment (up to label numbering) when using the same linkage and metric — they're doing the same underlying computation.

---

## 🔢 Choosing the Number of Clusters

Unlike KMeans/KMedoids, hierarchical clustering gives you the *whole tree* — the choice of k becomes "where do I cut it," which you can make visually or quantitatively.

```python
from sklearn.metrics import silhouette_score

# Visual: look for the tallest vertical gap the cut line can pass through
# without crossing a horizontal merge line
dendrogram(Z, truncate_mode='lastp', p=15)

# Quantitative: silhouette score across candidate k values
scores = {}
for k in range(2, 10):
    labels = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

best_k = max(scores, key=scores.get)
```

> 💡 **Tip:** The dendrogram's y-axis (merge distance) gives an additional signal `silhouette_score` alone doesn't: a big vertical jump between two merge heights suggests a natural cut point, even before checking silhouette scores.

---

## 🔗 Connectivity Constraints

You can restrict which points are allowed to merge directly, using a connectivity graph (e.g., k-nearest-neighbors) — useful for enforcing structure like spatial or temporal adjacency.

```python
from sklearn.neighbors import kneighbors_graph

connectivity = kneighbors_graph(X_scaled, n_neighbors=10, include_self=False)

agg_connected = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward',
    connectivity=connectivity
)
labels = agg_connected.fit_predict(X_scaled)
```

> ⚠️ **Warning:** A connectivity constraint can leave the data split into more connected components than `n_clusters` requested — check `agg_connected.n_connected_components_` if results look unexpected.

---

## 🧩 Feature Agglomeration

The same merge logic can cluster **features** instead of observations — a dimensionality-reduction technique that groups correlated columns together.

```python
from sklearn.cluster import FeatureAgglomeration

fa = FeatureAgglomeration(n_clusters=5, linkage='ward')
X_reduced = fa.fit_transform(X_scaled)   # (n_samples, 5) -- fewer columns, not fewer rows
```

> 💡 **Tip:** `FeatureAgglomeration` is a useful lighter-weight alternative to PCA when you want reduced dimensionality that stays interpretable as "groups of original features," rather than PCA's linear combinations.

---

## ⚖️ Comparison with Partitioning Methods

| Aspect | Hierarchical (Agglomerative) | KMeans | KMedoids |
|---|---|---|---|
| Need k in advance? | No — decide after seeing the tree | Yes | Yes |
| Output | Full nested tree + flat labels | Flat labels only | Flat labels only |
| Determinism | Deterministic given data + linkage | Depends on initialization | Depends on initialization |
| Cluster shape assumption | Flexible (depends on linkage) | Roughly spherical | Roughly spherical |
| Scalability | O(n²) memory, O(n³) naive time (O(n² log n) with optimizations) | O(n) per iteration | O(n²) or worse per iteration |
| Cluster center concept | None inherent — no centroid/medoid | Centroid (mean) | Medoid (real point) |
| Revisiting a merge | Never — merges are permanent | N/A (reassignment each iteration) | N/A (reassignment each iteration) |

> ⚠️ **Warning:** Agglomerative merges are **greedy and irrevocable** — once two clusters are merged, that decision is never undone, even if a later merge would have produced a better overall partition. KMeans/KMedoids, by contrast, can reassign a point away from a bad early decision on a later iteration.

---

## 🎯 Use Cases

| Scenario | Recommended | Why |
|---|---|---|
| Don't know k, want to explore structure visually | Hierarchical | Dendrogram shows the whole nested structure before committing to a cut |
| Natural hierarchy exists in the domain (taxonomy, org chart, phylogenetic tree) | Hierarchical | Output *is* a hierarchy, not just a flat partition |
| Small-to-moderate dataset (up to a few thousand rows) | Hierarchical | O(n²) memory becomes prohibitive on very large n |
| Need reproducible, deterministic clustering (no random initialization) | Hierarchical | No stochastic initialization step, unlike KMeans/KMedoids |
| Need to reduce correlated features, keeping interpretability | `FeatureAgglomeration` | Groups related columns instead of creating synthetic components like PCA |
| Very large dataset, flat partition is enough | KMeans (or MiniBatchKMeans) | O(n²) memory in hierarchical clustering won't scale |
| Elongated / chain-shaped clusters (e.g., along a spatial path) | Hierarchical with `single` linkage, or DBSCAN | `single` linkage naturally follows chains; DBSCAN also handles this well |

---

## ⚠️ Cautions & Gotchas

> ⚠️ **Warning:** Memory scales as **O(n²)** for the pairwise distance computations — a dataset of 50,000 rows already implies a 2.5-billion-entry distance matrix. Hierarchical clustering is not a large-data tool without approximation (e.g., `connectivity` constraints to sparsify, or subsampling first).

> ⚠️ **Warning:** `'single'` linkage is prone to the **chaining effect** — a thin bridge of intermediate points can cause two visually distinct clusters to merge into one long chain. Inspect the dendrogram, not just the flat labels, before trusting `single` linkage results.

> ⚠️ **Warning:** `ward` linkage requires Euclidean distance — don't pair it with `metric='cosine'`, `'manhattan'`, or `'precomputed'`; scikit-learn will raise a `ValueError`.

> ⚠️ **Warning:** Scale your features before clustering, same as with KMeans/KMedoids — Euclidean-family linkages (especially `ward`) are dominated by whichever feature has the largest numeric range if you skip `StandardScaler`.

> ⚠️ **Warning:** In scikit-learn ≥ 1.2, the parameter was renamed from `affinity=` to `metric=` on `AgglomerativeClustering`. If you're following an older tutorial or DataCamp course PDF, expect this naming discrepancy — the underlying behavior is the same.

> ⚠️ **Warning:** Merges are **permanent** — a poor early merge (e.g., forced by noisy points) can't be corrected later in the process, unlike the iterative reassignment in KMeans/KMedoids. If results look off, check the dendrogram for a merge that happened at a suspiciously low distance.

---

## 📋 Quick Reference Summary

| Question | Hierarchical (Agglomerative) |
|---|---|
| Need k up front? | No — decide by cutting the tree |
| Deterministic? | Yes — no random initialization |
| Output | Full dendrogram + flat labels |
| Default linkage (scikit-learn) | `ward` (Euclidean only) |
| Non-Euclidean metrics? | Yes, with `complete`/`average`/`single` linkage |
| Scalability | O(n²) memory — small/medium data only |
| Reduce features instead of rows? | Yes — `FeatureAgglomeration` |
| Core package | `sklearn.cluster` + `scipy.cluster.hierarchy` (both core, no extra install) |

---

*Validated against scikit-learn 1.8 and SciPy 1.17. `metric=` parameter naming applies to scikit-learn ≥ 1.2; use `affinity=` on older versions.*
