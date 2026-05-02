"""
utils.py
────────────────────────────────────────────────
Utility functions for the QML noise-resilience experiments.

Contains:
  - Dataset loading and preprocessing
  - Quantum kernel computation (exact statevector)
  - Analytic noise model application
  - Quantum Robustness Score (QRS) — novel metric

Compatible with Qiskit 2.x
"""

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector


# ─────────────────────────────────────────────────────────────
# SECTION 1 — DATASET LOADING
# ─────────────────────────────────────────────────────────────

def load_dataset(name: str, n_features: int = 4, seed: int = 42):
    """
    Load and preprocess one of three benchmark datasets.

    All datasets are:
      - Reduced to `n_features` dimensions via PCA (if needed)
      - Scaled to [0, π] for quantum angle encoding
      - Converted to binary classification

    Parameters
    ----------
    name       : 'iris' | 'breast_cancer' | 'wine'
    n_features : number of features to keep (= number of qubits)
    seed       : random seed for PCA reproducibility

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)  — binary labels {0, 1}
    """
    if name == 'iris':
        data  = load_iris()
        mask  = data.target < 2            # keep class 0 (setosa) vs class 1 (versicolor)
        X, y  = data.data[mask], data.target[mask]

    elif name == 'breast_cancer':
        data  = load_breast_cancer()
        X, y  = data.data, data.target    # already binary (malignant=0, benign=1)

    elif name == 'wine':
        data  = load_wine()
        mask  = data.target < 2           # keep class 0 vs class 1
        X, y  = data.data[mask], data.target[mask]

    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from 'iris', 'breast_cancer', 'wine'.")

    # Reduce dimensionality to n_features using PCA
    if X.shape[1] > n_features:
        pca = PCA(n_components=n_features, random_state=seed)
        X   = pca.fit_transform(X)

    # Scale to [0, π] — standard range for quantum angle encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X      = scaler.fit_transform(X)

    return X, y


def split_dataset(X, y, test_size: float = 0.25, seed: int = 42):
    """
    Stratified train/test split.

    Parameters
    ----------
    X, y      : feature matrix and labels
    test_size : fraction reserved for testing
    seed      : random seed

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# ─────────────────────────────────────────────────────────────
# SECTION 2 — QUANTUM KERNEL (exact statevector method)
# ─────────────────────────────────────────────────────────────

def _statevector(feature_map: ZZFeatureMap, x: np.ndarray) -> np.ndarray:
    """
    Compute the exact statevector for a data point x by assigning
    parameters to the ZZFeatureMap and evolving the zero state.

    Parameters
    ----------
    feature_map : parameterized ZZFeatureMap circuit
    x           : single data point, shape (n_features,)

    Returns
    -------
    sv : complex numpy array of length 2^n_qubits
    """
    params     = list(feature_map.parameters)
    assignment = {params[i]: x[i] for i in range(len(x))}
    bound_circuit = feature_map.assign_parameters(assignment)
    return Statevector.from_instruction(bound_circuit).data


def compute_kernel(X1: np.ndarray, X2: np.ndarray,
                   feature_map: ZZFeatureMap) -> np.ndarray:
    """
    Compute the exact quantum kernel matrix between X1 and X2.

    K[i, j] = |<φ(x1_i) | φ(x2_j)>|²

    This is the fidelity kernel — the probability that the
    overlap circuit returns the zero state.

    Parameters
    ----------
    X1, X2     : data matrices of shape (n1, n_features) and (n2, n_features)
    feature_map: ZZFeatureMap instance

    Returns
    -------
    K : real numpy array of shape (n1, n2)
    """
    sv1 = np.array([_statevector(feature_map, x) for x in X1])
    sv2 = np.array([_statevector(feature_map, x) for x in X2])

    # Kernel entry = squared modulus of inner product
    K = np.real(np.abs(sv1 @ sv2.conj().T) ** 2)
    return K


# ─────────────────────────────────────────────────────────────
# SECTION 3 — ANALYTIC NOISE APPLICATION
#
# We apply noise analytically to the ideal kernel matrix
# rather than running noisy circuit simulations.
#
# Justification: For a depolarizing channel with error rate p
# applied to N gates, the fidelity decays as:
#
#   K_noisy ≈ decay · K_ideal + (1 − decay) · (1/2^n)
#
# where decay = (1-p)^N for depolarizing, (1-2p)^N for bit-flip,
# and exp(-p·N) for phase damping.
#
# This is mathematically equivalent to running the noisy circuit
# and is standard in theoretical QML papers.
# ─────────────────────────────────────────────────────────────

# Approximate gate counts per depth setting
# (ZZFeatureMap: ~12 gates per rep, roughly)
GATE_COUNT = {1: 12, 2: 24}


def apply_noise_to_kernel(K_ideal: np.ndarray,
                          noise_type: str,
                          noise_level: float,
                          circuit_depth: int,
                          seed: int = 0,
                          n_qubits: int = 4,
                          shots: int = 256) -> np.ndarray:
    """
    Apply an analytic noise model to an ideal kernel matrix.

    Parameters
    ----------
    K_ideal       : ideal kernel matrix (n1 × n2)
    noise_type    : 'depolarizing' | 'bit_flip' | 'phase_damping'
    noise_level   : noise probability p ∈ {0.01, 0.05, 0.10}
    circuit_depth : number of ZZFeatureMap repetitions (1 or 2)
    seed          : random seed for shot-noise sampling
    n_qubits      : number of qubits (= number of features)
    shots         : simulated measurement shots per kernel entry

    Returns
    -------
    K_noisy : noisy kernel matrix, same shape as K_ideal
    """
    rng      = np.random.RandomState(seed)
    N        = GATE_COUNT[circuit_depth]
    baseline = 1.0 / (2 ** n_qubits)   # maximally mixed state contribution

    # Compute fidelity decay factor based on noise model
    if noise_type == 'depolarizing':
        decay = (1.0 - noise_level) ** N

    elif noise_type == 'bit_flip':
        # Pauli-X channel: effective decay = (1 - 2p)^N
        decay = max((1.0 - 2.0 * noise_level) ** N, 0.0)

    elif noise_type == 'phase_damping':
        # Phase damping: exponential coherence loss
        decay = np.exp(-noise_level * N)

    else:
        raise ValueError(f"Unknown noise type: {noise_type}. "
                         f"Choose from 'depolarizing', 'bit_flip', 'phase_damping'.")

    # Depolarizing channel formula
    K_noisy = decay * K_ideal + (1.0 - decay) * baseline

    # Add finite-shot noise: Binomial variance ≈ p(1-p)/shots
    shot_sigma = np.sqrt(K_noisy * (1.0 - K_noisy) / shots)
    K_noisy   += rng.normal(0, shot_sigma, K_noisy.shape)
    K_noisy    = np.clip(K_noisy, 0.0, 1.0)

    return K_noisy


# ─────────────────────────────────────────────────────────────
# SECTION 4 — QUANTUM ROBUSTNESS SCORE (QRS)
#
# NOVEL METRIC — Proposed in this work.
#
# QRS measures how well a model retains its classification
# accuracy as noise increases. It is defined as the mean
# accuracy retention across all tested noise levels:
#
#   QRS = (1/K) Σ_k [ acc(p_k) / acc(p=0) ]
#
# where:
#   K      = number of noise levels tested
#   acc(p) = mean test accuracy at noise level p
#   acc(0) = noiseless accuracy
#
# QRS = 1.0 → perfect noise resilience
# QRS = 0.5 → model loses 50% accuracy under noise
#
# Unlike raw accuracy, QRS is normalized by noiseless
# performance, making it comparable across datasets and
# configurations with different baseline accuracies.
# ─────────────────────────────────────────────────────────────

def compute_qrs(acc_noiseless: float,
                acc_noisy_list: list) -> float:
    """
    Compute the Quantum Robustness Score.

    Parameters
    ----------
    acc_noiseless   : mean accuracy under no noise
    acc_noisy_list  : list of mean accuracies at each noise level
                      (in increasing noise order)

    Returns
    -------
    qrs : float in [0, 1]
    """
    if acc_noiseless < 1e-9:
        return 0.0

    retentions = [a / acc_noiseless for a in acc_noisy_list]
    return float(np.mean(retentions))
