# 📊 Classification Evaluation Metrics

> **Module:** C6-M6 — Evaluation | **Source:** Uploaded PDF slide deck (Coursera-style course)

---

## 📑 Table of Contents

- [🎯 Evaluation Criteria](#-evaluation-criteria)
- [🧮 Classification Evaluation: The Confusion Matrix](#-classification-evaluation-the-confusion-matrix)
- [✅ Accuracy](#-accuracy)
- [🔎 Recall / Sensitivity](#-recall--sensitivity)
- [🎯 Precision](#-precision)
- [⚖️ Sensitivity and Precision Tradeoff](#️-sensitivity-and-precision-tradeoff)
- [🧾 F-Measure (F1 Score)](#-f-measure-f1-score)
- [🚫 Specificity](#-specificity)
- [📈 ROC Curve](#-roc-curve)
- [📐 AUC](#-auc)
- [🔢 Multiclass Classification](#-multiclass-classification)
- [✅ Best Practices](#-best-practices)
- [🧾 Quick Reference Summary](#-quick-reference-summary)

---

## 🎯 Evaluation Criteria

Before diving into specific metrics, it's worth stepping back to see the full range of dimensions a classification (or prediction) model can be judged on:

- **Accuracy** — classification vs. prediction correctness
- **Speed** — time to construct / use the model
- **Robustness** — handling noise and missing values
- **Scalability** — performance on large amounts of data
- **Interpretability** — understanding and insight into *why* the model made a decision
- **Goodness of rules** — e.g., decision tree size, compactness of classification rules

> 💡 **Tip:** "Accuracy" is only one of six evaluation criteria. A model can score well on accuracy but poorly on interpretability or scalability — the right tradeoff depends on the use case.

---

## 🧮 Classification Evaluation: The Confusion Matrix

A **confusion matrix** shows the ways in which a classification model is confused when it makes predictions — i.e., where its predictions match or don't match the actual labels.

### Structure (Binary Classification)

| | Predicted C1 | Predicted C2 |
|---|---|---|
| **Actual C1** | true positives | false negatives |
| **Actual C2** | false positives | true negatives |

### Definitions

| Term | Meaning |
|---|---|
| **True positive (TP)** | Predicted a positive label for an actual positive case |
| **True negative (TN)** | Predicted a negative label for an actual negative case |
| **False positive (FP)** | Predicted the *wrong* (positive) label for an actual negative case |
| **False negative (FN)** | Predicted the *wrong* (negative) label for an actual positive case |

> 💡 **Tip:** Every metric in this cheat sheet — accuracy, recall, precision, specificity, F1, ROC/AUC — is just a different ratio built from these four counts (TP, TN, FP, FN). Once the confusion matrix is set, every other number falls out of it.

---

## ✅ Accuracy

**Accuracy** measures how accurate a prediction is overall.

$$\text{Accuracy} = \frac{\#\text{accurate}}{\#\text{total}} = \frac{TP + TN}{P + N}$$

- Measures model performance **in general**, not for any one class
- **Accuracy = 1** → perfect
- **Accuracy = 0** → the model's predictions are exactly flipped

### Worked Examples

| | Predicted C1 | Predicted C2 |
|---|---|---|
| Actual C1 | 0 | 100 |
| Actual C2 | 100 | 0 |

Every prediction is wrong → **Accuracy = 0**

| | Predicted C1 | Predicted C2 |
|---|---|---|
| Actual C1 | 100 | 0 |
| Actual C2 | 0 | 100 |

Every prediction is right → **Accuracy = 1**

> ⚠️ **Warning:** Accuracy alone can be misleading on imbalanced datasets — a model that always predicts the majority class can score high accuracy while being useless for the minority class. This is exactly why recall, precision, and specificity exist (see below).

---

## 🔎 Recall / Sensitivity

**Recall** (also called **Sensitivity**) measures how accurate the model is *for actual positive cases* — of all the truly positive cases, how many did the model correctly catch?

$$\text{Sensitivity} = \frac{\#\text{true positive}}{\#\text{positive}} = \frac{TP}{TP + FN}$$

- **Sensitivity = 1** → no false negatives; every actual positive case is predicted as positive
- **Sensitivity = 0** → no true positives; every actual positive case is predicted as negative
- To achieve **high sensitivity**, the model should predict as many cases as positive as possible

**Real-world examples where high sensitivity matters:** fire alarms, COVID tests, job interviews (you'd rather over-flag than miss a true case)

### Worked Examples

| | Predicted C1 | Predicted C2 |
|---|---|---|
| Actual C1 | 100 | 0 |
| Actual C2 | 100 | 0 |

All actual C1 cases caught → **Sensitivity = 1**

| | Predicted C1 | Predicted C2 |
|---|---|---|
| Actual C1 | 100 | 0 |
| Actual C2 | 0 | 100 |

All actual C1 cases still caught → **Sensitivity = 1**

> ⚠️ **Warning:** The source material uses "predicate" throughout where "predict" / "predicted" is clearly meant (e.g., *"we should predicate as many positive as possible"*). This looks like a repeated typo rather than an intentional word choice, and has been silently read as "predict" / "predicted" everywhere in this cheat sheet. The phrase "high sensitively" in the original (should read "high sensitivity") has been corrected the same way.

---

## 🎯 Precision

**Precision** measures how accurate the model is *for predicted positive cases* — of all the cases the model labeled positive, how many actually were positive?

$$\text{Precision} = \frac{\#\text{true positive}}{\#\text{predicted positive}} = \frac{TP}{TP + FP}$$

- **Precision = 1** → no false positives; every predicted-positive case is actually positive
- **Precision = 0** → no true positives; every predicted-positive case is actually negative
- To achieve **high precision**, the model should predict as *few false positives* as possible

**Real-world examples where high precision matters:** nuclear missile launch decisions, heart surgery, death sentences (you'd rather miss a case than act on a false alarm)

### Worked Examples

| | Predicted C1 | Predicted C2 |
|---|---|---|
| Actual C1 | 1 | 99 |
| Actual C2 | 0 | 100 |

Only 1 predicted positive, and it's correct → **Precision = 1**

| | Predicted C1 | Predicted C2 |
|---|---|---|
| Actual C1 | 100 | 0 |
| Actual C2 | 0 | 100 |

**Precision = 1**

---

## ⚖️ Sensitivity and Precision Tradeoff

Pushing a model to be **more sensitive** (catch more true positives) generally makes it **less precise** (pick up more false positives along the way) — and vice versa. That's the **sensitivity/precision tradeoff**.

This tradeoff can be visualized with a **Precision vs. Recall plot**: precision starts near 1.0 at low recall and steadily declines as recall increases toward 1.0, often with a sharp drop-off near the highest recall values.

> 💡 **Tip:** There's no single "correct" point on this curve — where you sit depends on whether your application (like the examples above) penalizes false negatives or false positives more heavily.

---

## 🧾 F-Measure (F1 Score)

The **F-Measure (F1 score)** combines sensitivity and precision into a single number using their harmonic mean:

$$F_1 = \frac{2 \times (\text{Sensitivity} \times \text{Precision})}{\text{Sensitivity} + \text{Precision}}$$

### Worked Examples

| Actual C1 → Pred. | Actual C2 → Pred. | Sensitivity | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| 100, 0 | 0, 100 | 1 | 1 | **1** | 1 |
| 0, 100 | 100, 0 | 0 | 0 | **0** | 0 |
| 1, 99 | 0, 100 | 0.01 | 1 | **0.0198** | 0.5050 |
| 100, 0 | 100, 0 | 1 | 0.5 | **0.6667** | 0.5 |
| 0, 100 | 1, 99 | 0 | 0 | **0** | 0.4950 |
| 80, 20 | 20, 80 | 0.8 | 0.8 | **0.8** | 0.8 |

*(Each row's "Actual C1 → Pred." and "Actual C2 → Pred." columns give the Predicted-C1, Predicted-C2 counts for that actual class row of the confusion matrix.)*

> 💡 **Tip:** F1 = 1 is perfect and F1 = 0 is bad, but F1 alone doesn't tell you *why* — the same F1 = 0 can come from a model with zero sensitivity, zero precision, or both. Always check sensitivity and precision individually alongside F1, not instead of it.

> ⚠️ **Warning:** F1 is not the same as accuracy, and the two can diverge sharply. In the row above with Sensitivity = 0.01 and Precision = 1, F1 collapses to 0.0198 even though accuracy is a deceptively "healthy-looking" 0.5050 — a reminder that accuracy alone can mask a model that is barely catching any true positives.

---

## 🚫 Specificity

**Specificity** measures how accurate the model is *for actual negative cases* — the mirror image of sensitivity.

$$\text{Specificity} = \frac{\#\text{true negative}}{\#\text{negative}} = \frac{TN}{TN + FP}$$

- **Specificity = 1** → no false positives; every actual negative case is predicted as negative
- **Specificity = 0** → no true negatives; every actual negative case is predicted as positive
- To achieve **high specificity**, the model should predict as many cases as negative as possible

### Worked Examples

| | Predicted C1 | Predicted C2 |
|---|---|---|
| Actual C1 | 100 | 0 |
| Actual C2 | 0 | 100 |

**Specificity = 1**

| | Predicted C1 | Predicted C2 |
|---|---|---|
| Actual C1 | 0 | 100 |
| Actual C2 | 0 | 100 |

**Specificity = 1** (no false positives — but note sensitivity here is 0, since no actual C1 case was ever predicted as C1)

> 💡 **Tip:** This second example is a good illustration of why no single metric tells the whole story — a "predict everything as negative" model can score perfect specificity while being completely useless at catching positives.

---

## 📈 ROC Curve

The **Receiver Operating Characteristics (ROC) Curve** plots how a model's true-positive and false-positive rates trade off as the classification **threshold** is moved.

### True Positive Rate (TPR) and False Positive Rate (FPR)

$$\text{TPR} = \frac{\#\text{true positive}}{\#\text{positive}} = \frac{TP}{TP + FN} \quad (\text{this is the same as Sensitivity})$$

$$\text{FPR} = \frac{\#\text{false positive}}{\#\text{negative}} = \frac{FP}{FP + TN}$$

- **TPR** — of all actual positive cases, how many were predicted as positive
- **FPR** — of all actual negative cases, how many were (incorrectly) predicted as positive

### TPR/FPR at the Extremes

| | Predicted C1 | Predicted C2 | TPR | FPR |
|---|---|---|---|---|
| Actual C1 | 100 | 0 | | |
| Actual C2 | 100 | 0 | **1** | **1** |

| | Predicted C1 | Predicted C2 | TPR | FPR |
|---|---|---|---|---|
| Actual C1 | 0 | 100 | | |
| Actual C2 | 0 | 100 | **0** | **0** |

### TPR/FPR at Intermediate Thresholds

| | Predicted C1 | Predicted C2 | TPR | FPR |
|---|---|---|---|---|
| Actual C1 | 80 | 20 | | |
| Actual C2 | 20 | 80 | **0.8** | **0.2** |

| | Predicted C1 | Predicted C2 | TPR | FPR |
|---|---|---|---|---|
| Actual C1 | 20 | 80 | | |
| Actual C2 | 80 | 20 | **0.2** | **0.8** |

### Building the Curve by Sweeping the Threshold

- The **higher** the threshold, the **lower** the false positive rate — and the lower the true positive rate too
- The **lower** the threshold, the **higher** the false positive rate — and the higher the true positive rate too
- Sweeping the threshold from low to high traces out a curve of FPR (x-axis) vs. TPR (y-axis) — that curve **is** the ROC curve

| Threshold Step | Actual C1 (Pred C1, Pred C2) | Actual C2 (Pred C1, Pred C2) | TPR | FPR |
|---|---|---|---|---|
| 1 | 0, 100 | 0, 100 | 0 | 0 |
| 2 | 30, 70 | 5, 95 | 0.3 | 0.05 |
| 3 | 50, 50 | 10, 90 | 0.5 | 0.1 |
| 4 | 80, 20 | 30, 70 | 0.8 | 0.3 |
| 5 | 100, 0 | 100, 0 | 1 | 1 |

> ⚠️ **Warning:** The source slide lists row 3's "Actual C1" counts as **50, 70** — which sums to 120 instead of the 100 that every other row in this table sums to (the total number of actual positive cases should stay fixed as only the threshold changes). Since the slide's own stated TPR for this row is 0.5 (i.e., 50/100), the corrected value used above is **50, 50**. This looks like a copy-paste slip from the neighboring column rather than an intentional value.

### Shape of the Curve

- A curve that bows sharply toward the top-left corner (high TPR reached at very low FPR) indicates a **high-performing model**
- The diagonal line from (0,0) to (1,1) represents the performance of **random guessing**
- The further the curve sits above the diagonal, the better the model separates the two classes

> 💡 **Tip:** ROC curves are threshold-independent — they show you the *full range* of TPR/FPR tradeoffs available, letting you pick the operating threshold that fits your application's tolerance for false positives vs. false negatives.

---

## 📐 AUC

Reading a full ROC curve isn't always straightforward, so it's common to reduce it to a single number: the **AUC (Area Under the ROC Curve)**.

| AUC Value | Meaning |
|---|---|
| `1` | Perfect classifier |
| `0` | Worst possible classifier (perfectly wrong) |
| `0.5` | No better than random guessing |

### Reading AUC from Curve Shape

| Curve Shape | Interpretation |
|---|---|
| Rises steeply toward the top-left, then flattens near the top | **Good separation** — model strongly distinguishes the classes |
| Rises toward the top-left but more gradually | **Reasonable separation** |
| Wanders close to (but a bit above) the diagonal | **Poor separation** |
| Follows the diagonal closely | **Random separation** — the model isn't distinguishing the classes at all |

> 💡 **Tip:** AUC and the shape of the curve carry complementary information — two models can have similar AUC while behaving very differently at the specific threshold you actually plan to deploy, so it's still worth inspecting the curve itself, not just the summary number.

---

## 🔢 Multiclass Classification

The same confusion-matrix idea **expands** naturally beyond two classes: for $k$ classes, the confusion matrix becomes a $k \times k$ grid, with the diagonal holding correct predictions and every off-diagonal cell representing a specific type of misclassification (row = actual class, column = predicted class).

**Example — 9-class confusion matrix (e.g., digit classification):**

```
[[5578,    0,   22,    7,    8,   45,   35,    5,  222,    1],
 [   0, 6410,   35,   26,    4,   44,    4,    8,  198,   13],
 [  28,   27, 5232,  100,   74,   27,   68,   37,  354,   11],
 [  23,   18,  115, 5254,    2,  209,   26,   38,  373,   73],
 [  11,   14,   45,   12, 5219,   11,   33,   26,  299,  172],
 [  26,   16,   31,  173,   54, 4484,   76,   14,  482,   65],
 [  31,   17,   45,    2,   42,   98, 5556,    3,  123,    1],
 [  20,   10,   53,   27,   50,   13,    3, 5696,  173,  220],
 [  17,   64,   47,   91,    3,  125,   24,   11, 5421,   48],
 [  24,   18,   29,   67,  116,   39,    1,  174,  329, 5152]]
```

- A **bright diagonal** in a heatmap rendering of this matrix indicates the model gets most predictions right for most classes
- Bright **off-diagonal** cells reveal systematic confusions — e.g., a class the model consistently mixes up with a specific other class, rather than errors spread randomly across all classes

> 💡 **Tip:** For multiclass problems, precision/recall/F1 are typically computed **per class** (treating each class as "positive" and all others as "negative" in turn), then optionally averaged (macro-average treats all classes equally; weighted/micro-average accounts for class size) to get a single summary number.

---

## ✅ Best Practices

| Practice | Do | Avoid |
|---|---|---|
| Choosing a metric | Pick the metric(s) that match what errors actually cost in your application (e.g., high sensitivity for medical screening, high precision for irreversible actions) | Reporting accuracy alone as if it's the whole story |
| Imbalanced datasets | Check sensitivity, precision, and specificity individually, not just accuracy | Trusting a high accuracy score on a skewed class distribution |
| Summarizing precision & recall | Use F1 (or another combined metric) alongside — not instead of — the individual sensitivity/precision values | Reporting F1 alone without knowing *which* component is driving it down |
| Comparing classifiers | Compare full ROC curves and AUC together | Judging a model from AUC alone without checking curve shape at your actual operating threshold |
| Reading a multiclass confusion matrix | Look at both diagonal (correct) and off-diagonal (confused pairs) cells | Collapsing straight to a single accuracy number and ignoring which classes get confused with which |
| Threshold selection | Choose the operating threshold based on the ROC curve and your application's FP/FN cost tradeoff | Defaulting to a 0.5 threshold without checking whether it fits the problem |

---

## 🧾 Quick Reference Summary

| Task | Formula |
|---|---|
| Accuracy | $(TP + TN) / (P + N)$ |
| Recall / Sensitivity / TPR | $TP / (TP + FN)$ |
| Precision | $TP / (TP + FP)$ |
| Specificity | $TN / (TN + FP)$ |
| False Positive Rate (FPR) | $FP / (FP + TN)$ |
| F1 Score | $2 \times (\text{Sensitivity} \times \text{Precision}) / (\text{Sensitivity} + \text{Precision})$ |
| ROC Curve axes | x = FPR, y = TPR, swept across classification thresholds |
| AUC — perfect / worst / random | `1` / `0` / `0.5` |
| Multiclass confusion matrix | $k \times k$ grid; diagonal = correct, off-diagonal = specific misclassifications |
