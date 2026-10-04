# 🧠 AXIOM

### ⚡ Adversarial Evidence Reconstruction Engine

> 🧪 **A tiny repository for a very inconvenient question.**


> 🔮 **Does the most frequently observed claim actually represent the most probable truth?**

Axiom is a self-contained machine-learning/reasoning experiment for reconstructing the most coherent explanation from **noisy, contradictory, duplicated, temporally inconsistent, and adversarial evidence**.

It deliberately separates:

\[
\text{frequency} \neq \text{independent evidence} \neq \text{truth}
\]

Rather than trusting the majority claim, Axiom estimates how much *independent* support each hypothesis actually receives, penalizes contradiction and temporal inconsistency, and performs counterfactual checks against expected observations.

The entire project is intentionally tiny:

```text
axiom/
├── README.md
├── LICENSE
└── axiom.ipynb
```

The notebook is both the implementation and the research playground.

---

## 🔍 What Axiom Does

Given observations such as:

```text
Source A: The reactor failed because of overheating.
Source B: The reactor failed because of overheating.
Source C: The reactor failed because of overheating.
Source D: The coolant system leaked.
Source E: Coolant pressure dropped before the failure.
Source F: Temperature remained normal.
```

a naïve majority vote favors overheating.

Axiom asks a harder question:

> 🧠 **How much of that apparent majority is genuinely independent evidence?**

If A, B, and C are downstream copies of the same report, their apparent 3-to-1 advantage is misleading.

Axiom therefore models:

- semantic claim similarity
- supporting and contradicting evidence
- source reliability
- evidence dependence
- temporal consistency
- hypothesis likelihood
- counterfactual expectations
- adversarial narrative amplification

---

## ⚙️ Core Pipeline

```text
             RAW OBSERVATIONS
                    │
                    ▼
          ┌───────────────────┐
          │ Semantic Encoding │
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │ Claim Clustering  │
          └─────────┬─────────┘
                    ▼
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   Reliability  Contradiction  Time
       │            │            │
       └────────────┼────────────┘
                    ▼
          ┌───────────────────┐
          │  Evidence Graph   │
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │ Independence      │
          │ Correction        │
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │ Bayesian-style    │
          │ Hypothesis Score  │
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │ Counterfactual    │
          │ Stress Test       │
          └─────────┬─────────┘
                    ▼
            RECONSTRUCTED
               REALITY
```

---

## 🧮 Mathematical Intuition

For hypothesis \(H\), Axiom computes an evidence-adjusted score:

\[
S(H) =
\log P(H)
+
\sum_i
w_i \cdot d_i \cdot r_i \cdot \ell_i(H)
-
\lambda C(H)
-
\mu T(H)
\]

where:

- \(P(H)\) is the prior
- \(w_i\) is the base evidence weight
- \(d_i\) is the independence discount
- \(r_i\) is source reliability
- \(\ell_i(H)\) is the signed likelihood contribution
- \(C(H)\) is contradiction pressure
- \(T(H)\) is temporal inconsistency
- \(\lambda,\mu\) control penalties

The independence term is deliberately simple and interpretable:

\[
d_i = \frac{1}{1 + \alpha \cdot \text{dependency}(i)}
\]

This is not presented as a universal truth estimator. It is a transparent experimental model whose behavior can be inspected, challenged, and benchmarked.

---

## 🧩 Why This Is Interesting

A majority-vote system assumes that repeated observations provide repeated information.

Real information ecosystems frequently violate that assumption:

```text
Original report
      │
      ├── News article
      │      ├── Blog
      │      ├── Social post
      │      └── Aggregator
      │
      └── Press summary
             ├── Newsletter
             └── Forum post
```

Ten downstream observations can therefore encode approximately one underlying observation.

Axiom attempts to distinguish:

```text
10 independent observations
```

from:

```text
1 observation repeated 10 times
```

That distinction is the project's central research question.

---

## 📓 Notebook Contents

`axiom.ipynb` contains:

1. **Problem definition**
2. **Synthetic evidence generation**
3. **Semantic claim representation**
4. **Claim clustering**
5. **Contradiction graph construction**
6. **Source-dependency modeling**
7. **Independence correction**
8. **Bayesian-style hypothesis scoring**
9. **Counterfactual evaluation**
10. **Adversarial majority experiments**
11. **Sensitivity analysis**
12. **Data tables**
13. **Pictographical visualizations**
14. **Failure-case analysis**

The notebook intentionally uses synthetic data so the experiments are reproducible and do not pretend that an arbitrary internet scrape constitutes ground truth.

---

## 🧪 Example Experiment

The notebook constructs a synthetic event with competing hypotheses:

```text
H1: overheating caused the failure
H2: coolant-system failure caused the failure
H3: electrical failure caused the failure
```

It then generates observations containing:

- direct observations
- copied reports
- contradictory observations
- temporal clues
- adversarially repeated claims

A majority system can be deliberately fooled.

Axiom discounts correlated evidence and evaluates whether each hypothesis explains the *whole* observation set.

---

## 🛡️ Adversarial Narrative Experiment

The notebook varies the proportion of repeated misinformation:

```text
False narrative amplification
0% ─────────────────────────────── 100%

        Majority Vote
        ████████████████████

        Axiom
        ████████░░░░░░░░░░░░
```

The exact results are generated by the notebook rather than hard-coded.

The objective is to measure the point at which:

\[
\text{narrative volume}
\]

overwhelms:

\[
\text{independent evidence}
\]

and whether independence-aware inference delays or prevents that failure.

---

## 📊 Visualizations

The notebook includes pictographical representations of the data, including:

- evidence-source network diagrams
- hypothesis confidence bars
- claim-dependency graphs
- support/contradiction matrices
- adversarial amplification curves
- reliability distributions
- temporal evidence timelines
- heatmaps of evidence influence

The visualizations are generated from the experiment itself.

---

## 🛠️ Installation

Python 3.10+ is recommended.

Core notebook dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn networkx jupyter
```

Launch:

```bash
jupyter notebook axiom.ipynb
```

The notebook also contains a lightweight TF-IDF semantic baseline, so the project does not require a large language model or GPU.

---

## 🚀 Optional Semantic Upgrade

The architecture can be extended with sentence embeddings:

```python
from sentence_transformers import SentenceTransformer
```

The notebook deliberately keeps the default implementation lightweight so the experiment can run on an ordinary machine.

---

## 🧭 Design Principles

### 1. 🔎 Interpretability over magic

Every major score should be inspectable.

### 2. 🔗 Independence over repetition

Repeated evidence is not automatically independent evidence.

### 3. ⚔️ Contradictions are information

A contradiction should reduce confidence rather than disappear during preprocessing.

### 4. ⏱️ Time matters

An explanation should fit the sequence of observations.

### 5. 💥 Attack your own conclusion

Counterfactual testing is used to search for evidence that would make a hypothesis less plausible.

### 6. 🧪 Synthetic experiments are honest

Synthetic data is useful for controlled experiments, but it is not proof that the model understands reality.

---

## 🚫 What Axiom Is Not

Axiom is **not**:

- an oracle
- a universal fact checker
- a truth machine
- a replacement for human investigation
- proof that Bayesian scoring solves epistemology

It is an experimental framework for studying how an inference system behaves when **evidence volume and evidence quality diverge**.

That distinction matters.

---

## 🔬 Research Questions

The project can be extended into a serious research paper around questions such as:

### 🔹 RQ1
How much does correlated evidence inflate confidence under majority voting?

### 🔹 RQ2
Can source-dependency correction preserve minority-but-independent evidence?

### 🔹 RQ3
How robust is independence-aware inference against adversarial narrative amplification?

### 🔹 RQ4
How should evidence reliability and evidence independence interact?

### 🔹 RQ5
Can temporal coherence improve hypothesis reconstruction under contradiction?

### 🔹 RQ6
Where does independence correction itself fail?

The last question is particularly important. A model that can only demonstrate why it is right is a toy. A model that can demonstrate **how it can be wrong** is much more interesting.

---

## 🌱 Possible Extensions

The three-file version is intentionally compact, but the concept can grow into:

- learned source-reliability estimation
- graph neural networks
- probabilistic graphical models
- causal discovery
- retrieval-augmented evidence gathering
- provenance tracking
- temporal knowledge graphs
- uncertainty calibration
- active evidence acquisition
- human-in-the-loop adjudication
- benchmark datasets for dominant-narrative bias
- adversarial misinformation generation

---

## 📜 License

MIT. See `LICENSE`.

---

## 💡 Project Thesis

> 🧠 **A claim does not become more true merely because more people repeat it.**

Axiom exists to turn that intuition into a measurable machine-learning experiment.
