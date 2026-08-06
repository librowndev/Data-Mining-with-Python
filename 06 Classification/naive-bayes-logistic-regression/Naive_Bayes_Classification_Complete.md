# 🎲 Naïve Bayes Classification — Reference Notes

> **Topic:** Naïve Bayes Classification | **Source:** Data Mining Lecture Slides | **Worked example adapted from:** *Data Mining: Concepts and Techniques* (3rd ed.), Jiawei Han, Micheline Kamber, and Jian Pei, Morgan Kaufmann, 2011

---

## 📑 Table of Contents

- [🧭 What Is Naïve Bayes Classification?](#-what-is-naïve-bayes-classification)
- [📐 Bayes' Theorem Fundamentals](#-bayes-theorem-fundamentals)
- [🔥 Worked Example: Fire and Alarm](#-worked-example-fire-and-alarm)
- [🤝 Worked Example: Working Together and Cheating](#-worked-example-working-together-and-cheating)
- [🧮 The Naïve Bayes Classifier, Formally](#-the-naïve-bayes-classifier-formally)
- [💻 Full Worked Classification Example](#-full-worked-classification-example)
- [🚫 Avoiding the Zero-Probability Problem](#-avoiding-the-zero-probability-problem)
- [⚖️ Advantages and Disadvantages](#️-advantages-and-disadvantages)
- [✅ Best Practices](#-best-practices)
- [🧾 Quick Reference Summary](#-quick-reference-summary)

---

## 🧭 What Is Naïve Bayes Classification?

**Naïve Bayes** is a statistical classifier that predicts **class membership probabilities** — that is, the probability that a given data sample belongs to each of several possible classes.

- **Foundation:** built directly on **Bayes' Theorem**
- **Performance:** despite its simplifying assumption, it's comparable to decision tree classifiers and some neural network classifiers
- **Type:** it can be trained **incrementally**, updating its probability estimates as new data arrives

### The Classification Problem

| Symbol | Meaning | Example |
|---|---|---|
| $X$ | A data sample (evidence) whose class is unknown | age = 35, income = \$40,000 |
| $H$ | A hypothesis that $X$ belongs to class $C$ | H: buys a computer |
| $P(H\|X)$ | **Posterior probability** — what we want to determine | Probability the sample buys a computer, given its attributes |

> 💡 **Tip:** Classification with Naïve Bayes always boils down to computing $P(H\|X)$ for every candidate class and picking the class with the highest value.

---

## 📐 Bayes' Theorem Fundamentals

### The General Form

$$P(H|X) = \frac{P(X|H)\,P(H)}{P(X)}$$

| Term | Name | Status |
|---|---|---|
| $P(H)$ | **Prior probability** | Known |
| $P(X)$ | **Marginal probability** (marginalization) | Known |
| $P(X\|H)$ | **Likelihood** | Known |
| $P(H\|X)$ | **Posterior probability** | Unknown — this is what we solve for |

### Deriving Bayes' Theorem from Conditional Probability

Let $A$ and $B$ be two events:

- $P(A)$ — probability that $A$ occurs
- $P(B)$ — probability that $B$ occurs
- $P(A \,\&\, B)$ — probability that $A$ and $B$ occur together

**Conditional probability definitions:**

$$P(A|B) = \frac{P(A\,\&\,B)}{P(B)}, \quad P(B) > 0$$

$$P(B|A) = \frac{P(A\,\&\,B)}{P(A)}, \quad P(A) > 0$$

Since both expressions share the same numerator $P(A \,\&\, B)$:

$$P(A|B)\,P(B) = P(B|A)\,P(A)$$

Dividing both sides by $P(B)$ gives **Bayes' Theorem**:

$$P(A|B) = \frac{P(B|A)\,P(A)}{P(B)}$$

> 💡 **Tip:** Bayes' Theorem is nothing more than conditional probability rearranged — it lets you "flip" a conditional probability you *don't* know ($P(A\|B)$) into one built from probabilities you *do* know ($P(B\|A)$, $P(A)$, $P(B)$).

---

## 🔥 Worked Example: Fire and Alarm

Two events: **Alarm** and **Fire**.

| Quantity | Value |
|---|---|
| $P(\text{Alarm})$ | 1% |
| $P(\text{Fire})$ | 0.1% |
| $P(\text{Alarm}\|\text{Fire})$ | 99% |

**Question:** What is the probability of a fire, given that the alarm goes off?

### Applying Bayes' Theorem

$$P(\text{Fire}|\text{Alarm}) = \frac{P(\text{Alarm}|\text{Fire}) \times P(\text{Fire})}{P(\text{Alarm})} = \frac{99\% \times 0.1\%}{1\%} = 9.9\%$$

We can also compute the complementary case, using $P(\text{NoAlarm}|\text{Fire}) = 1 - 99\% = 1\%$ and $P(\text{NoAlarm}) = 1 - 1\% = 99\%$:

$$P(\text{Fire}|\text{NoAlarm}) = \frac{P(\text{NoAlarm}|\text{Fire}) \times P(\text{Fire})}{P(\text{NoAlarm})} = \frac{1\% \times 0.1\%}{99\%} \approx 0.001\%$$

> 💡 **Tip:** Even with a highly reliable alarm (99% true-positive rate), the posterior probability of an actual fire is still fairly low (9.9%) because fires are rare to begin with ($P(\text{Fire}) = 0.1\%$). This is the classic **base-rate effect** — a low prior probability pulls the posterior down even when the evidence is strong.

---

## 🤝 Worked Example: Working Together and Cheating

Two events: **WT** (Work Together) and **C** (Cheating).

| Quantity | Value |
|---|---|
| $P(\text{WT})$ | 90% |
| $P(C)$ | 1% |
| $P(\text{WT}\|C)$ | 99% |

**Question:** What is the probability that a student cheated, given that they worked with others?

### Applying Bayes' Theorem

$$P(C|\text{WT}) = \frac{P(\text{WT}|C) \times P(C)}{P(\text{WT})} = \frac{99\% \times 1\%}{90\%} = 1.1\%$$

Using $P(\overline{\text{WT}}|C) = 1 - 99\% = 1\%$ and $P(\overline{\text{WT}}) = 1 - 90\% = 10\%$:

$$P(C|\overline{\text{WT}}) = \frac{P(\overline{\text{WT}}|C) \times P(C)}{P(\overline{\text{WT}})} = \frac{1\% \times 1\%}{10\%} = 0.1\%$$

> ⚠️ **Warning:** The source material states that "a student who works with others will be **10 times** more likely to cheat than a student who works alone." Comparing the two results precisely: $1.1\% \div 0.1\% = 11\times$, not $10\times$. The "10 times" figure in the original slide is a rounded approximation — the exact ratio computed from the stated inputs is closer to **11×**.

---

## 🧮 The Naïve Bayes Classifier, Formally

### Setup

- $X = (x_1, x_2, \dots, x_n)$ — a data sample described by $n$ attributes
- $M$ classes: $C_1, C_2, \dots, C_m$
- **Goal:** classify $X$ into whichever class maximizes $P(C_i|X)$

### Applying Bayes' Theorem to Classification

$$P(C_i|X) = \frac{P(X|C_i)\,P(C_i)}{P(X)}$$

> 💡 **Tip:** Since $P(X)$ is the same constant for every class $C_i$ being compared, it can be dropped from the optimization — you only need to find the class that maximizes the **numerator**:
>
> $$P(X|C_i)\,P(C_i)$$

### The "Naïve" Assumption

Computing $P(X|C_i)$ directly is usually impossible in practice — a specific combination of attribute values (like the exact $X$ you're trying to classify) may never have been observed in the training data at all.

To work around this, Naïve Bayes makes the **naïve assumption of class-conditional independence**: it assumes there is **no dependence between attributes**, given the class. This lets the compound probability be broken down into a simple product:

$$P(X|C_i) = \prod_{k=1}^{n} P(x_k|C_i) = P(x_1|C_i) \times P(x_2|C_i) \times \dots \times P(x_n|C_i)$$

> ⚠️ **Warning:** This independence assumption is rarely true in the real world (see [Advantages and Disadvantages](#️-advantages-and-disadvantages) below) — it's an approximation made purely for computational tractability, not a claim that the attributes are actually unrelated.

---

## 💻 Full Worked Classification Example

### The Training Data

**2 classes** for the target attribute `buys_computer`:

- $C_1$: Yes
- $C_2$: No

| CID | age | income | student | credit_rating | buys_computer |
|---|---|---|---|---|---|
| 1 | ≤30 | high | no | fair | no |
| 2 | ≤30 | high | no | excellent | no |
| 3 | 31–40 | high | no | fair | **yes** |
| 4 | >40 | medium | no | fair | **yes** |
| 5 | >40 | low | yes | fair | **yes** |
| 6 | >40 | low | yes | excellent | no |
| 7 | 31–40 | low | yes | excellent | **yes** |
| 8 | ≤30 | medium | no | fair | no |
| 9 | ≤30 | low | yes | fair | **yes** |
| 10 | >40 | medium | yes | fair | **yes** |
| 11 | ≤30 | medium | yes | excellent | **yes** |
| 12 | 31–40 | medium | no | excellent | **yes** |
| 13 | 31–40 | high | yes | fair | **yes** |
| 14 | >40 | medium | no | excellent | no |

**The sample to classify:**

$$X = (\text{age} \le 30,\ \text{income} = \text{medium},\ \text{student} = \text{yes},\ \text{credit\_rating} = \text{fair})$$

Ideally we'd look up $P(X|\text{buys\_computer} = \text{"yes"})$ and $P(X|\text{buys\_computer} = \text{"no"})$ directly — but this exact combination of attributes doesn't appear anywhere in the training table. This is exactly the situation the **naïve independence assumption** is designed to solve.

---

### Step 1 — Compute the Prior Probabilities $P(C_i)$

```text
P(buys_computer = "yes") = 9/14 = 0.643
P(buys_computer = "no")  = 5/14 = 0.357
```

---

### Step 2 — Compute Each Class-Conditional Probability $P(x_k|C_i)$

**$P(\text{age} \le 30 \mid C_i)$**

```text
P(age = "<=30" | buys_computer = "yes") = 2/9 = 0.222
P(age = "<=30" | buys_computer = "no")  = 3/5 = 0.6
```

**$P(\text{income} = \text{medium} \mid C_i)$**

```text
P(income = "medium" | buys_computer = "yes") = 4/9 = 0.444
P(income = "medium" | buys_computer = "no")  = 2/5 = 0.4
```

**$P(\text{student} = \text{yes} \mid C_i)$**

```text
P(student = "yes" | buys_computer = "yes") = 6/9 = 0.667
P(student = "yes" | buys_computer = "no")  = 1/5 = 0.2
```

**$P(\text{credit\_rating} = \text{fair} \mid C_i)$**

```text
P(credit_rating = "fair" | buys_computer = "yes") = 6/9 = 0.667
P(credit_rating = "fair" | buys_computer = "no")  = 2/5 = 0.4
```

> 💡 **Tip:** Each of these is a simple frequency count *within one class only* — e.g., "of the 9 rows where `buys_computer = yes`, how many have `age <= 30`?" This is what makes Naïve Bayes cheap to train: every conditional probability is just a tally over the training table.

---

### Step 3 — Combine via the Naïve Assumption: $P(X|C_i)$

$$P(X|C_i) = P(\text{age}\!\le\!30|C_i) \times P(\text{income}\!=\!\text{med}|C_i) \times P(\text{student}\!=\!\text{yes}|C_i) \times P(\text{credit}\!=\!\text{fair}|C_i)$$

```text
P(X | buys_computer = "yes") = 0.222 × 0.444 × 0.667 × 0.667 = 0.044
P(X | buys_computer = "no")  = 0.600 × 0.400 × 0.200 × 0.400 = 0.019
```

---

### Step 4 — Multiply by the Prior: $P(X|C_i)\,P(C_i)$

```text
P(X | buys_computer = "yes") × P(buys_computer = "yes") = 0.044 × 0.643 = 0.028
P(X | buys_computer = "no")  × P(buys_computer = "no")  = 0.019 × 0.357 = 0.007
```

---

### Step 5 — Classify

Since $0.028 > 0.007$, the class that **maximizes** $P(X|C_i)\,P(C_i)$ is `buys_computer = "yes"`.

> ✅ **Result:** Naïve Bayes classifies this customer as **likely to buy a computer**.

> 💡 **Tip:** Notice we never actually divided by $P(X)$ — because it's the same constant for both classes, comparing the un-normalized products $P(X\|C_i)\,P(C_i)$ is sufficient to determine which class has the higher posterior probability.

---

## 🚫 Avoiding the Zero-Probability Problem

Because $P(X|C_i)$ is a **product** of individual attribute probabilities, a single zero anywhere in that product collapses the entire result to zero:

$$P(X|C_i) = \prod_{k=1}^{n} P(x_k|C_i) = P(x_1|C_i) \times P(x_2|C_i) \times \cdots \times P(x_n|C_i)$$

**Example:** suppose a training set of 1,000 records has an `income` attribute with counts:

```text
income: low(0), medium(990), high(10)
```

Since no record has `income = low`, $P(\text{income} = \text{low}\,|\,C_i) = 0$ — and any future sample with `income = low` would get a **zero probability for the entire class**, no matter how well the other attributes match.

> ⚠️ **Warning:** A single unseen attribute value can silently zero out an otherwise strong classification. This is a structural weakness of the product-of-probabilities approach, not a rare edge case — it happens whenever a category is simply missing from the training sample.

### The Fix: Laplacian Correction (Laplace Estimator)

**Add 1** to the count of every category. This keeps all resulting probabilities **non-zero** while staying close to the original proportions:

```text
Before correction: income: low(0),   medium(990), high(10)
After correction:  income: low(1),   medium(991), high(11)
```

> 💡 **Tip:** With enough training data, adding 1 to every bucket barely shifts the probabilities that were already well-represented (990 → 991 is negligible), while it rescues the categories that would otherwise be permanently zeroed out.

---

## ⚖️ Advantages and Disadvantages

| | Detail |
|---|---|
| ✅ **Advantage** | Easy and cheap to compute; delivers good results in most practical cases |
| ⚠️ **Disadvantage** | Relies on the **class-conditional independence** assumption |
| ⚠️ **Disadvantage** | Real-world attributes are frequently *not* independent — e.g., for hospital patients, `age`, `family history`, `fever`, `cough`, `lung cancer`, and `diabetes` are all interrelated in practice |

> 💡 **Tip:** The independence assumption is "naïve" precisely because it's usually false — yet the classifier often still performs well, because it only needs to rank classes correctly (not estimate exact probabilities) to make the right prediction.

---

## ✅ Best Practices

| Practice | Do | Avoid |
|---|---|---|
| Computing the posterior | Drop the constant $P(X)$ denominator and just compare $P(X\|C_i)\,P(C_i)$ across classes | Trying to compute the exact posterior probability when only the *ranking* between classes matters |
| Handling unseen attribute values | Apply **Laplacian correction** (add 1 to every category count) | Leaving zero-count categories as-is, which zeroes out the whole class probability |
| Estimating $P(X\|C_i)$ | Use the naïve class-conditional independence assumption to decompose it into a product of per-attribute probabilities | Trying to look up the exact compound condition $X$ directly in the training data (it usually doesn't exist) |
| Interpreting results | Remember Naïve Bayes gives *relative* posterior strength, useful for ranking/classification | Treating the un-normalized products as true probabilities (they don't sum to 1 until divided by $P(X)$) |
| Assessing the independence assumption | Acknowledge it as a simplifying approximation | Assuming attributes are truly independent just because the model works well in practice |

---

## 🧾 Quick Reference Summary

| Task | Formula / Code |
|---|---|
| Bayes' Theorem (general) | $P(H\|X) = \dfrac{P(X\|H)\,P(H)}{P(X)}$ |
| Bayes' Theorem (two events) | $P(A\|B) = \dfrac{P(B\|A)\,P(A)}{P(B)}$ |
| Conditional probability | $P(A\|B) = \dfrac{P(A\,\&\,B)}{P(B)}$ |
| Posterior probability for classification | $P(C_i\|X) = \dfrac{P(X\|C_i)\,P(C_i)}{P(X)}$ |
| Simplified classification rule (drop constant $P(X)$) | Maximize $P(X\|C_i)\,P(C_i)$ |
| Naïve class-conditional independence | $P(X\|C_i) = \prod_{k=1}^{n} P(x_k\|C_i)$ |
| Laplacian correction | Add 1 to every attribute-value count before computing probabilities |

---

