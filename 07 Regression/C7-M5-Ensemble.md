# 🌳 Ensemble Methods — Bagging, Boosting & Stacking

> **Module:** C7-M5 — Ensemble | **Source:** Course Slide Deck

---

## 📑 Table of Contents

- [🤔 We Always Want to Know: What Is Best?](#-we-always-want-to-know-what-is-best)
- [🎒 Bootstrap Aggregating (Bagging)](#-bootstrap-aggregating-bagging)
- [🌲 Random Forest](#-random-forest)
- [👥 Random Nearest Neighbor](#-random-nearest-neighbor)
- [🚀 Boosting](#-boosting)
- [⚠️ AdaBoost: Overfitting & Advantages](#️-adaboost-overfitting-advantages)
- [📈 Gradient Boost](#-gradient-boost)
- [🥊 Bagging or Boosting?](#-bagging-or-boosting)
- [🥞 Stacking](#-stacking)
- [✅ Best Practices](#-best-practices)
- [🧾 Quick Reference Summary](#-quick-reference-summary)

---

## 🤔 We Always Want to Know: What Is Best?

Choosing a single "best" model is hard — **we don't know future data**, so no single model can be guaranteed optimal in advance. Two broad strategies help address this:

- **Model tuning** — try all combinations of hyperparameters for a single model
- **Ensemble learning** — combine multiple models together

> 💡 **Tip:** Rather than relying on one expert, an ensemble "hires a team." The way that team is assembled and coordinated defines the type of ensemble method.

### Three Families of Ensemble Methods

| Method | Core Idea |
|---|---|
| **Bagging** | Train models independently on different random subsets of the data, then combine |
| **Boosting** | Train models sequentially, each one focusing on the previous model's mistakes |
| **Stacking** | Train a meta-model to combine the predictions of several different base models |

---

## 🎒 Bootstrap Aggregating (Bagging)

**Bootstrap Aggregating**, or **Bagging**, builds an ensemble through three steps.

### Step 1 — Generate Bootstrapped Datasets

- Each bootstrapped dataset is the **same size** as the original dataset
- Data points are drawn **randomly, with replacement** — so duplicates are possible
- Whatever original data points are **left out** of a given bootstrapped sample form that sample's **out-of-bag (OOB) dataset**

> 💡 **Tip:** The out-of-bag dataset is essentially "free" held-out data for that model instance — it can be used as a built-in testing/validation set without sacrificing any training data up front.

#### Worked Example

Given an original dataset:

$$x_{original} = \{x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9\}$$

One possible bootstrapped sample (drawn with replacement, same size as the original):

$$x_{b1} = \{x_1, x_1, x_1, x_4, x_4, x_6, x_8, x_8, x_9\}$$

The associated out-of-bag set is everything from the original that didn't get drawn:

$$x_{oob1} = x_{original} - x_{b1} = \{x_2, x_3, x_5, x_7\}$$

$$x_{oob1} \text{ is used as testing data for this bootstrapped model.}$$

### Step 2 — Train a Model Instance on Each Bootstrapped Set

- Because each training set is bootstrapped, it represents a **different perspective** on the original data
- The learning process can also be restricted to a **limited subset of features**, which forces the model to explore features it might otherwise overlook
- The result: each trained model instance reflects a **different perspective** of both the data and the feature space

### Step 3 — Aggregate the Model Instances

- The combination of bootstrapped datasets and limited features drives **variance and diversity** among the ensemble's individual models
- Each model produces its own individual assessment; a combination mechanism turns these into one **collective assessment**

| Task Type | Aggregation Mechanism |
|---|---|
| Classification | **Majority voting** (simple or weighted) |
| Regression | **Averaging** (simple or weighted) |

> 💡 **Tip:** Random Forest is the canonical example of bagging in practice — see below.

---

## 🌲 Random Forest

### Why Random Forest?

- Decision trees are easy to explain, understand, and induct
- However, a **fully trained decision tree is very likely to overfit**
- The decision-tree learning algorithm tends to **increase variance to reduce bias** — pushing toward overfitting rather than underfitting
- **Random Forest reduces variance** (improving generalization) by introducing randomization at multiple levels

### Sources of Randomization

| Randomization Type | Description |
|---|---|
| **Random datasets (Bagging)** | Each tree trains on a bootstrapped subset of the original dataset, drawn with replacement |
| **Random subspaces** | Each split considers only a random subset of the available features |
| **Random patches** | A tree is trained using both a random dataset *and* a random feature subspace at once |

### Scikit-learn Implementation

| Estimator | Use Case |
|---|---|
| `RandomForestRegressor` | Regression tasks |
| `RandomForestClassifier` | Classification tasks |

### Key Hyperparameters

| Hyperparameter | Purpose |
|---|---|
| `n_estimators` | Number of trees to build. Higher is generally better but slower. **Default: 100** |
| `bootstrap` / `max_samples` | Controls the size of the bootstrapped sample of rows |
| `max_features` | Size of the random feature subset per split — **Regression:** all features by default; **Classification:** `sqrt(#features)` by default |
| `max_depth` | Controls how deep (complete) each individual tree is allowed to grow |

> ⚠️ **Warning:** The default `max_features` behavior differs between regression and classification (all features vs. `sqrt(#features)`) — don't assume the same default applies to both task types.

---

## 👥 Random Nearest Neighbor

K-Nearest Neighbors (K-NN) can be turned into an ensemble the same way a decision tree can be turned into a Random Forest:

- Generate random (bootstrapped) datasets
- Train on random subsets of features
- Average the individual assessments together

This combination is known as **Random KNN**.

---

## 🚀 Boosting

### Motivating Example: Weak Rules for Spam Detection

Recall Naive Bayes as a classifier: given a list of conditions, it computes a posterior probability. For spam email detection, it's easy to find **rules of thumb (weak learners)** that are broadly reasonable but individually imperfect:

- If the title contains keyword set 1 → predict spam
- If the title contains keyword set 2 → predict spam
- If the content contains "sale" → predict spam
- ...and so on

It's hard to find a **single** rule that is highly accurate on its own — but boosting combines many such weak rules into one strong rule.

### How Boosting Works

Boosting **incrementally builds an ensemble**, where each new model instance emphasizes the training data that the previous instance handled poorly.

| Stage | What Happens |
|---|---|
| **Initialization** | All data points start with equal weight |
| **After each model instance** | Data points the model got wrong have their weight **increased (boosted)** |
| **Next iteration** | The next model instance pays more attention to these higher-weighted (harder) points |
| **After T rounds** | All weak learners are combined into a single, highly accurate rule |

> ⚠️ **Warning:** Because boosting repeatedly re-focuses on the training data's mistakes, it **tends to be more prone to overfitting** than bagging-based methods.

**Example:** AdaBoost (Adaptive Boosting)

---

## ⚠️ AdaBoost: Overfitting & Advantages

### AdaBoost Overfitting

- Because AdaBoost pays increasing attention to errors, it tends to **fit the training set as closely as possible** — similar to the overfitting tendency of decision trees
- Focusing heavily on errors leads to an increasingly **complex final assessment**
- This causes **overfitting**, and it's **hard to know when to stop training** (i.e., how many rounds $T$ to use)

The classic error-vs-rounds curve shows **training error** decreasing steadily as rounds ($T$) increase, while **test error** initially drops, reaches a minimum, and then **rises again** as the model overfits.

> 💡 **Tip:** This train/test divergence is exactly why the number of boosting rounds, $T$, is the key hyperparameter to tune carefully — too few rounds underfits, too many overfits.

*Source: [cs.princeton.edu/~schapire/talks/nips-tutorial.pdf](https://www.cs.princeton.edu/~schapire/talks/nips-tutorial.pdf)*

### AdaBoost Advantages

| Advantage | Description |
|---|---|
| **Fast, simple, and easy** | Straightforward to implement and run |
| **Minimal tuning** | Only $T$ (number of rounds) really needs to be tuned |
| **Flexible** | Works with any underlying learning algorithm as the weak learner |
| **Effective** | Produces strong accuracy from combinations of weak learners |
| **No prior knowledge required** | Doesn't need domain-specific assumptions to work well |
| **Versatile** | Can be used with textual, numeric, and discrete data |

*Source: [cs.princeton.edu/~schapire/talks/nips-tutorial.pdf](https://www.cs.princeton.edu/~schapire/talks/nips-tutorial.pdf)*

---

## 📈 Gradient Boost

- The main idea of boosting is to **add new models to the ensemble sequentially**
- **Gradient Boost** combines **gradient descent** with boosting into a single robust algorithm
- Because it can use **any differentiable loss function**, Gradient Boost is more robust to outliers and more flexible than AdaBoost
- Gradient Boost handles **both classification and regression**, whereas AdaBoost was originally designed mainly for **binary classification**
- The two methods identify "difficult" observations differently:

| Method | How It Targets Difficult Observations |
|---|---|
| **Gradient Boost** | Via **gradients** of the loss function |
| **AdaBoost** | Via **shifting sample weights** |

### Regularizing Gradient Boosting

A **Gradient Boosting regularization** plot (test set deviance vs. boosting iterations) compares several regularization settings:

- No shrinkage
- `learning_rate=0.2`
- `subsample=0.5`
- `learning_rate=0.2, subsample=0.5`
- `learning_rate=0.2, max_features=2`

> 💡 **Tip:** Combining a reduced `learning_rate` with `subsample` typically produces the lowest, most stable test-set deviance — reducing the learning rate alone or subsampling alone still leaves the curve prone to drifting upward (overfitting) as iterations increase.

> ⚠️ **Warning:** With **no shrinkage** (learning rate effectively unconstrained), test set deviance tends to bottom out early and then **increase again** with more boosting iterations — a clear overfitting signature, similar to AdaBoost's overfitting curve above.

---

## 🥊 Bagging or Boosting?

Choosing between bagging and boosting often comes down to how "strong" the base model already is.

| | **Bagging** | **Boosting** |
|---|---|---|
| **Best suited for** | Strong, complex models (e.g., fully developed decision trees) | Weak models (e.g., shallow decision trees) |
| **Typical performance** | Often significantly better than a single classifier | Generally better than bagging |
| **Robustness to noise** | Not considerably worse than a single model; more robust overall | May **overfit to misclassified data** |
| **Prediction quality** | Provides improved accuracy | Can be highly accurate, but at higher overfitting risk |

> 💡 **Tip:** As a rule of thumb, reach for **bagging** when your base learner is already fairly strong and you mainly want to reduce variance; reach for **boosting** when your base learner is intentionally weak and you want to reduce bias by combining many of them.

---

## 🥞 Stacking

**Stacking**, or **stacked generalization**, trains a learning algorithm to **combine the predictions of several other learning algorithms**.

### How Stacking Works

1. **Step 1:** Train several different base algorithms on the training data
2. **Step 2:** Train a **combiner algorithm**, using the *predictions* of the base algorithms as its input features

> 💡 **Tip:** In practice, a **logistic regression model** is often used as the combiner — it's simple, fast, and works well at learning how to weight the base models' outputs.

---

## ✅ Best Practices

| Practice | Do | Avoid |
|---|---|---|
| Choosing an ensemble strategy | Match the strategy to your base model's strength: bagging for strong/complex models, boosting for weak models | Defaulting to one method regardless of base model strength |
| Building bootstrapped datasets | Sample with replacement, same size as the original dataset | Sampling without replacement (that's not bootstrapping) |
| Using out-of-bag data | Use OOB samples as a built-in test set per bootstrapped model | Discarding OOB data or re-using training data for validation |
| Tuning Random Forest | Tune `n_estimators`, `max_features`, and `max_depth` deliberately | Leaving all hyperparameters at defaults without checking task type (regression vs. classification) |
| Tuning AdaBoost / Gradient Boost rounds | Monitor train vs. test error across rounds ($T$) to catch overfitting | Increasing $T$ indefinitely assuming more rounds always helps |
| Regularizing Gradient Boost | Combine a lower `learning_rate` with `subsample` for stability | Using no shrinkage, which tends to overfit as iterations increase |
| Building a stacked ensemble | Train the combiner on the base models' *predictions*, not raw features | Feeding raw input features into the combiner instead of base-model outputs |

---

## 🧾 Quick Reference Summary

| Task | Key Idea / Code |
|---|---|
| Bootstrapped dataset | Same size as original, drawn **with replacement** |
| Out-of-bag (OOB) dataset | $x_{oob} = x_{original} - x_{b}$ |
| Bagging aggregation (classification) | Majority voting (simple or weighted) |
| Bagging aggregation (regression) | Averaging (simple or weighted) |
| Random Forest regressor | `RandomForestRegressor` |
| Random Forest classifier | `RandomForestClassifier` |
| Key Random Forest hyperparameters | `n_estimators`, `bootstrap`/`max_samples`, `max_features`, `max_depth` |
| Default `max_features` (classification) | `sqrt(#features)` |
| Default `max_features` (regression) | All features |
| Boosting weight update | Misclassified points get **increased weight** each round |
| AdaBoost | Combines weak learners; tune $T$ (rounds) to avoid overfitting |
| Gradient Boost | Gradient descent + boosting; supports any differentiable loss |
| Gradient Boost regularization | Lower `learning_rate` + `subsample` for stability |
| Bagging fits best | Strong/complex base models |
| Boosting fits best | Weak/shallow base models |
| Stacking Step 1 | Train multiple base algorithms |
| Stacking Step 2 | Train combiner (often logistic regression) on base model predictions |
