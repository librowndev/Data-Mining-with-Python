# 🎯 KMeans vs KMedoids Cheat Sheet

> **Import convention:** `from sklearn.cluster import KMeans` | `from sklearn_extra.cluster import KMedoids`
>
> **Category:** Partitioning clustering — scikit-learn (core) + scikit-learn-extra (companion package)

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Core Difference](#-core-difference)
- [How Each Algorithm Works](#-how-each-algorithm-works)
- [Installation & Compatibility](#-installation--compatibility)
- [KMeans Quick Syntax](#-kmeans-quick-syntax)
- [KMedoids Quick Syntax](#-kmedoids-quick-syntax)
- [Distance Metrics](#-distance-metrics)
- [Choosing k](#-choosing-k)
- [Initialization Methods](#-initialization-methods)
- [Use Cases](#-use-cases)
- [Cautions & Gotchas](#-cautions--gotchas)
- [Quick Reference Summary](#-quick-reference-summary)

---

## 🔎 Overview

Both **KMeans** and **KMedoids** are *partitioning* clustering methods — they split `n` observations into `k` non-overlapping clusters by minimizing dissimilarity between points and a cluster representative. The difference is what that representative is and what distances it can use.

---

## ⚖️ Core Difference

| Aspect | KMeans | KMedoids |
|---|---|---|
| Cluster center | **Centroid** — mean of all points in the cluster (synthetic, may not exist in data) | **Medoid** — an actual data point minimizing total dissimilarity to the rest of its cluster |
| Distance metric | Euclidean only | Any metric (Euclidean, Manhattan, cosine, precomputed) |
| Package | `sklearn.cluster.KMeans` (core) | `sklearn_extra.cluster.KMedoids` (separate package) |
| Outlier sensitivity | High — mean is pulled toward extreme values | Low — medoid is a real point, harder to drag |
| Time complexity | ~O(n) per iteration | O(k(n−k)²) per iteration for PAM; slower |
| Interpretability | Centroid is a synthetic "average" row | Medoid is a real, citable observation |

---

## ⚙️ How Each Algorithm Works

### KMeans (Lloyd's algorithm)
1. Initialize `k` centroids
2. Assign each point to its nearest centroid
3. Recompute each centroid as the **mean** of its assigned points
4. Repeat 2–3 until assignments stabilize (or `max_iter` reached)

> 💡 **Tip:** The mean is the exact minimizer of squared Euclidean distance — this is *why* KMeans is mathematically locked to Euclidean distance. Swapping in another metric breaks the guarantee that the update step improves the objective.

### KMedoids (PAM — Partitioning Around Medoids, or faster variants)
1. Initialize `k` medoids (real data points)
2. Assign each point to its nearest medoid
3. For each cluster, search for the point that minimizes total distance to all other members in that cluster; if a non-medoid point improves the total cost, swap it in
4. Repeat until no swap improves the cost (or `max_iter` reached)

> 💡 **Tip:** The swap search is what makes KMedoids metric-agnostic — it doesn't need a closed-form "mean," so any dissimilarity measure (or precomputed distance matrix) works.

---

## 📦 Installation & Compatibility

> ⚠️ **Warning:** `scikit-learn-extra` (last released 2021) ships compiled Cython extensions built against the **NumPy 1.x C-API** and imports `distutils`, which was removed from the standard library in **Python 3.12**. On a modern stack (current scikit-learn + NumPy 2.x + Python 3.12), it will fail to import — this is not a scikit-learn version conflict, it's a NumPy ABI break plus a removed stdlib module.

Validated failure on a current local environment (Python 3.12, NumPy 2.4, scikit-learn 1.8):

```
ImportError: numpy.core.multiarray failed to import
  (compiled against the NumPy 1.x C-API)
ModuleNotFoundError: No module named 'distutils'
```

Validated **working** combination:

```bash
pip install "numpy<2" "setuptools<70" scikit-learn scikit-learn-extra
```

- `numpy<2` avoids the C-API break in the compiled extensions
- `setuptools<70` restores a `distutils` shim (removed from setuptools in 70.0)
- Current scikit-learn (tested at 1.9.0) works fine against this combo — sklearn itself was **not** the blocker

> ✅ **Best practice:** Because this downgrades NumPy, don't install `scikit-learn-extra` into a shared `ml-general` environment. Use a disposable environment (a fresh Colab runtime, or an isolated venv/conda env) dedicated to KMedoids work.

| Approach | When to use |
|---|---|
| Google Colab, fresh runtime | Fastest path — no local environment pollution; matches the companion notebook |
| Isolated venv/conda env with pinned `numpy<2` + `setuptools<70` | Local, repeatable work outside Colab |
| Skip `scikit-learn-extra` entirely | Use `KMeans` only, or implement a small custom medoid-swap loop for one-off needs |

---

## 🔵 KMeans Quick Syntax

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

km = KMeans(
    n_clusters=3,
    init='k-means++',   # smart init (default)
    n_init=10,           # number of independent runs; keeps best by inertia
    max_iter=300,
    random_state=42
)
labels = km.fit_predict(X_scaled)

km.cluster_centers_   # (k, n_features) synthetic centroids
km.inertia_            # sum of squared distances to nearest centroid
km.n_iter_              # iterations run
```

---

## 🟠 KMedoids Quick Syntax

```python
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

kmed = KMedoids(
    n_clusters=3,
    metric='manhattan',   # or 'euclidean', 'cosine', 'precomputed', ...
    method='pam',          # 'pam' (exact swaps) or 'alternate' (faster, approximate)
    init='k-medoids++',
    max_iter=300,
    random_state=42
)
labels = kmed.fit_predict(X_scaled)

kmed.cluster_centers_    # (k, n_features) — actual data points
kmed.medoid_indices_      # row indices of X for each medoid
kmed.inertia_              # sum of distances to nearest medoid
```

> 💡 **Tip:** With `metric='precomputed'`, pass a full `(n, n)` distance matrix instead of raw features — useful for sequence alignment scores, graph shortest-path distances, or any custom dissimilarity that doesn't fit a feature-vector representation.

---

## 📏 Distance Metrics

| Metric | KMeans | KMedoids | Typical use |
|---|:---:|:---:|---|
| Euclidean | ✅ | ✅ | Continuous, scaled numeric features |
| Manhattan | ❌ | ✅ | Robust to outliers along each axis; grid-like data |
| Cosine | ❌ | ✅ | Text embeddings, direction matters more than magnitude |
| Precomputed | ❌ | ✅ | Custom dissimilarities, sequence/graph distances |
| Gower / mixed-type | ❌ | ✅ (via precomputed) | Mixed categorical + numeric data |

---

## 🔢 Choosing k

Neither algorithm infers `k` — both need it supplied in advance.

```python
from sklearn.metrics import silhouette_score

scores = {}
for k in range(2, 10):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

best_k = max(scores, key=scores.get)
```

- **Elbow method:** plot `inertia_` vs `k`, look for the bend
- **Silhouette score:** higher is better (range −1 to 1); works for both KMeans and KMedoids since it only needs labels + a distance matrix

---

## 🌱 Initialization Methods

| Method | KMeans | KMedoids | Notes |
|---|---|---|---|
| `'random'` | ✅ | ✅ | Simple, more prone to bad local optima |
| `'k-means++'` | ✅ (default) | — | Spreads initial centroids apart probabilistically |
| `'k-medoids++'` | — | ✅ (default) | Medoid analogue of k-means++ |
| `'heuristic'` | — | ✅ | Greedy initial medoid selection |
| `'build'` | — | ✅ | Classic PAM build step |

> ⚠️ **Warning:** Both algorithms can converge to different local optima depending on initialization. Always set `random_state` for reproducibility, and consider comparing a couple of `init` options if cluster quality looks unstable.

---

## 🎯 Use Cases

| Scenario | Recommended | Why |
|---|---|---|
| Large dataset (10,000+ rows), continuous scaled features | KMeans | O(n) per iteration scales; Euclidean is a natural fit |
| Need cluster "representatives" that are real, citable rows | KMedoids | Medoid is an actual observation, not a synthetic average |
| Data has known outliers that can't be cleaned first | KMedoids | Medoid resists being dragged by extreme values |
| Mixed categorical/numeric features | KMedoids (precomputed Gower distance) | KMeans has no valid distance for non-numeric data |
| Fast baseline clustering before trying more complex models | KMeans | Simple, fast, well understood |
| Clustering on a precomputed similarity/distance matrix (e.g., sequence alignment) | KMedoids | `metric='precomputed'` support |

---

## ⚠️ Cautions & Gotchas

> ⚠️ **Warning:** Both assume roughly convex, similarly-sized clusters. Elongated, non-convex, or highly unequal-density clusters are better served by DBSCAN, HDBSCAN, or Gaussian Mixture Models.

> ⚠️ **Warning:** Always scale features before either algorithm when using Euclidean-family metrics — unscaled features with larger numeric ranges will dominate the distance calculation.

> ⚠️ **Warning:** `scikit-learn-extra` is a lightly-maintained, separate package (not core scikit-learn). Pin its version and its NumPy/setuptools compatibility explicitly in any environment file — don't assume it will install cleanly against a current stack. See [Installation & Compatibility](#-installation--compatibility).

> ⚠️ **Warning:** KMedoids' PAM method is O(k(n−k)²) per iteration. On datasets beyond a few thousand rows, prefer `method='alternate'` (faster, slightly less exhaustive swap search) or subsample before clustering.

> ⚠️ **Warning:** `n_init` matters for both — a single run (`n_init=1`) is more prone to landing in a poor local optimum. scikit-learn's `KMeans` default (`n_init=10` in most recent versions, though this has changed across releases — verify for your installed version) runs multiple initializations and keeps the best.

---

## 📋 Quick Reference Summary

| Question | KMeans | KMedoids |
|---|---|---|
| Cluster center type | Centroid (mean) | Medoid (real point) |
| Distance metrics | Euclidean only | Any (incl. precomputed) |
| Outlier robustness | Low | High |
| Speed on large n | Fast | Slow (PAM) / moderate (`alternate`) |
| Core scikit-learn? | Yes | No — `scikit-learn-extra` |
| Best for | Large, clean, continuous, scaled data | Small–medium, mixed-type, outlier-prone, or needing real exemplars |

---

*Validated against scikit-learn 1.8–1.9 and scikit-learn-extra 0.3.0. Compatibility notes current as of August 2026 — re-verify if scikit-learn-extra receives a new release.*
