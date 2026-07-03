# qrs-nisq-qksvm

**Quantifying Noise Resilience in Quantum Kernel Classifiers on NISQ Hardware**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE_Format-red.svg)](paper/QML_IEEE_Final.pdf)
[![Status](https://img.shields.io/badge/Status-Under_Review-yellow.svg)]()
[![Author](https://img.shields.io/badge/Author-Shriyanss_Behera-orange.svg)](mailto:25155633@kiit.ac.in)

> A proposed metric --- the **Quantum Robustness Score (QRS)** --- for systematic,
> normalised evaluation of quantum machine learning classifier degradation under
> gate-level noise on NISQ devices. Tested across 5 datasets, 3 noise channels,
> 4 circuit depths, and ~2,000 simulation runs.

---

## Paper

**"Noise-Resilient Quantum Machine Learning Models for Small-Scale Classification Tasks"**
**Shriyanss Behera** | 25155633@kiit.ac.in | IEEE Conference Format | 9 Pages

Read the full paper: [`paper/QML_IEEE_Final.pdf`](paper/QML_IEEE_Final.pdf)

---

## The Problem This Paper Solves

Quantum machine learning benchmarks almost universally report raw classification
accuracy at a single, arbitrarily chosen noise level. This is insufficient for
two reasons:

1. **Accuracy conflates dataset difficulty with noise sensitivity.** A model on
   an easy dataset always looks more robust than an identical model on a hard one,
   even if the reverse is true once noise is factored out.

2. **A single point tells you nothing about degradation trends.** Two models can
   sit at identical accuracy at p=0.05 and behave completely differently at p=0.10.

This paper fixes both problems with a single normalised metric.

---

## The Quantum Robustness Score (QRS)

```
QRS = (1/K) * sum_k [ Acc(p_k) / Acc(0) ]
```

| Symbol | Meaning |
|---|---|
| K | Number of noise levels tested (K=3: p in {0.01, 0.05, 0.10}) |
| Acc(p_k) | Mean test accuracy across 5 seeds at noise level p_k |
| Acc(0) | Mean noiseless test accuracy across 5 seeds |

**Key properties:**
- Bounded in [0, 1]
- Equals 1.0 for a model completely unaffected by noise
- **Invariant to dataset baseline difficulty** --- the key innovation
- Captures degradation trend across multiple levels, not a single snapshot

**Why normalisation matters:** On Breast Cancer, Baseline vs Shallow differ
by just **0.5%** in noiseless accuracy. QRS reveals a **26.4 percentage-point**
resilience gap that raw accuracy would make entirely invisible.

---

## Key Results

### QRS Summary (averaged across depolarising, bit-flip, phase-damping noise)

| Dataset | Baseline (d=2) | Shallow (d=1) | Noise-Aware (d=2) |
|---|:---:|:---:|:---:|
| Iris | 0.918 | 0.915 | 0.913 |
| Breast Cancer | 0.643 | **0.812** | 0.757 |
| Wine | 0.888 | **0.941** | 0.922 |
| Digits (0 vs 1) | 0.822 | 0.861 | **0.870** |
| Adult (Income) | 0.812 | **0.896** | 0.877 |

### Four Main Findings

**1. Shallow circuits are more noise-resilient.**
Depth-1 (~12 gates) outperforms depth-2 on 4/5 datasets across all noise types.
Classical ML says deeper = better. On NISQ hardware, the opposite holds.

**2. Bit-flip noise is the most destructive channel.**
At p=0.10, G=24 gates: (0.90)^24 = 0.080 --- 92% of kernel signal destroyed.
Depolarising at same rate: (0.95)^24 = 0.292 --- 3.7x more survivable.

**3. Noise-aware training is selective, not universal.**
Adversarial kernel training beats Shallow in only 1 of 15 dataset-noise
conditions (Wine, bit-flip). Partial cross-noise transfer explains the exception.

**4. QRS reveals what accuracy conceals.**
Breast Cancer: 0.5% accuracy gap becomes a 26.4 QRS-point gap.
Any study using noiseless accuracy alone would call these architectures
interchangeable. They are not.

### Depth Scaling Law

```
QRS(reps) ~= QRS(1) - beta * (reps - 1)
```

- beta ~= 0.048 per layer under depolarising noise
- beta ~= 0.071 per layer under bit-flip noise

QRS declines monotonically from reps=1 to reps=4 on every dataset and noise
channel tested. Circuit designers get a quantitative resilience cost per depth layer.

---

## Method

### Model: Quantum Kernel SVM (QKSVM)

```
x in R^d  -->  ZZFeatureMap  -->  |phi(x)>
Kernel: K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2
Gram matrix K  -->  Classical SVM (C=1.0)  -->  Prediction
```

No trainable quantum parameters. Immune to the barren plateau problem.

### ZZFeatureMap Encoding

- n=4 qubits, d=4 features
- Each block: Hadamard layer -> R_Z(2x_i) rotations -> R_ZZ(2x_i*x_j) entanglers
- Gate count: G ~= 12 * reps
- reps=1: ~12 gates (Shallow) | reps=2: ~24 gates (Baseline, Noise-Aware)

### Three Noise Channels

| Channel | Formula | Physical Meaning |
|---|---|---|
| Depolarising | delta = (1-p)^G | General hardware imperfection |
| Bit-flip | delta = (1-2p)^G | Pauli-X error |
| Phase damping | delta = exp(-pG) | Environmental decoherence |

Noise applied analytically: K_noisy = delta * K_ideal + (1-delta) * (1/2^n)

Plus finite-shot noise: sigma^2_ij = K_ij*(1-K_ij)/256

### Three Circuit Configurations

| Name | reps | Gates | Training Kernel |
|---|:---:|:---:|---|
| Baseline | 2 | ~24 | Noiseless |
| Shallow | 1 | ~12 | Noiseless |
| Noise-Aware | 2 | ~24 | Corrupted (depolarising p=0.03) |

### Five Datasets

| Dataset | N | Features -> PCA | Task |
|---|:---:|:---:|---|
| Iris | 100 | 4->4 | Setosa vs. Versicolor |
| Breast Cancer Wisconsin | 569 | 30->4 | Malignant vs. Benign |
| Wine | 130 | 13->4 | Class 0 vs. Class 1 |
| Digits | 360 | 64->4 | Digit 0 vs. Digit 1 |
| Adult (Census Income) | 1000 | ~100->4 | Income <=50K vs. >50K |

All: MinMax scaling to [0, pi], stratified 75/25 split.

### Experimental Scale

| Experiment | Description | Runs |
|---|---|:---:|
| A - Core | 3 configs x 5 datasets x 3 noise x 3 levels x 5 seeds | 750 |
| B - Depth Scaling | reps{1,2,3,4} x 5 datasets x 3 noise x 3 levels x 5 seeds | 900 |
| C - Classical Baseline | 2 models x 5 datasets x 2 noise x 3 levels x 5 seeds | 300 |
| **Total** | | **~2,000** |

### Metric Validation

QRS validated against Worst-Case Drop (WCD) and AUC-Noise from identical data:
- Agrees on top-ranked configuration in **14 of 15** conditions
- In the one disagreement: QRS shows **4x larger discrimination gap** than WCD
- Frobenius-norm analysis: r = +0.38-0.41 between kernel preservation and QRS

---

## Repository Structure

```
qrs-nisq-qksvm/
|
+-- paper/
|   +-- QML_IEEE_Final.pdf             # Compiled paper (9 pages)
|   +-- QML_IEEE_Final.tex             # LaTeX source
|   +-- references.bib                 # 78-entry bibliography
|
+-- figures/
|   +-- fig1a_iris.pdf                 # Accuracy vs noise - Iris
|   +-- fig1b_breast_cancer.pdf        # Accuracy vs noise - Breast Cancer
|   +-- fig1c_wine.pdf                 # Accuracy vs noise - Wine
|   +-- fig1d_digits.pdf               # Accuracy vs noise - Digits
|   +-- fig1e_adult.pdf                # Accuracy vs noise - Adult
|   +-- fig2_qrs_comparison.pdf        # QRS bar chart, all datasets
|   +-- fig3_noiseless_accuracy.pdf    # Noiseless accuracy baseline
|   +-- fig4_qrs_heatmap.pdf           # QRS heatmap (star = best config)
|   +-- fig5_depth_scaling.pdf         # QRS vs circuit depth reps 1-4
|   +-- fig6_classical_comparison.pdf  # Quantum QRS vs Classical CRS
|   +-- fig7_frobenius_vs_qrs.pdf      # Kernel stability scatter
|   +-- fig8_metric_validation.pdf     # QRS vs WCD vs AUC-Noise
|
+-- code/                              # Simulation scripts (coming soon)
+-- LICENSE                            # Apache 2.0
+-- .gitignore
+-- README.md
```

---

## Compiling the Paper Locally

Requires TeX Live with IEEEtran, quantikz, physics, tikz, booktabs.

```bash
git clone https://github.com/ShriyanssB/qrs-nisq-qksvm.git
cd qrs-nisq-qksvm/paper
cp ../figures/*.pdf .

pdflatex QML_IEEE_Final.tex
bibtex QML_IEEE_Final
pdflatex QML_IEEE_Final.tex
pdflatex QML_IEEE_Final.tex
```

> Do NOT add: hyperref, microtype, pgfplots, tikzit, or dblfloatfix.
> Each causes fatal conflicts with IEEEtran. The .tex must remain pure ASCII.

---

## Future Work

- Multi-noise adversarial training (simultaneous exposure to multiple channels)
- Validation on real NISQ hardware (IBM Quantum, IonQ, Quantinuum)
- Extending QRS to variational quantum classifiers and quantum neural networks
- QRS as a hardware quality proxy for device selection

---

## Citation

```bibtex
@article{behera2026qrs,
  author  = {Behera, Shriyanss},
  title   = {Noise-Resilient Quantum Machine Learning Models
             for Small-Scale Classification Tasks},
  journal = {IEEE Conference Proceedings},
  year    = {2026},
  note    = {Under review}
}
```

---

## Licence

Apache License 2.0 --- see [LICENSE](LICENSE) for full terms.
Free to use, modify, and distribute with attribution.
Patent use explicitly permitted.

---

**Shriyanss Behera** | shriyanss.behera@gmail.com
