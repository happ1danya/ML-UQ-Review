"""Uncertainty-quantification methods for classification with MC dropout.

All functions accept ``obs`` with shape ``(T, N, C)`` where ``T`` is the
number of stochastic observations, ``N`` is the number of samples, and ``C``
is the number of classes. Each function returns one uncertainty value per
sample.

The module keeps the original ML-UQ-Review method names and adds the methods
used in the EPRDS paper. Paper-style aliases are provided for M_E, E_M, and
Max_E so the implementation can be mapped directly to the manuscript.
"""

from __future__ import annotations

import numpy as np

_EPS = np.finfo(np.float64).eps


def _validate_observations(obs: np.ndarray) -> np.ndarray:
    """Return finite, normalized probabilities with shape ``(T, N, C)``."""
    arr = np.asarray(obs, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"obs must have shape (T, N, C), received {arr.shape}")
    if min(arr.shape) < 1 or arr.shape[2] < 2:
        raise ValueError("obs must contain at least one pass, one sample, and two classes")
    if not np.all(np.isfinite(arr)):
        raise ValueError("obs contains NaN or infinite values")
    if np.any(arr < 0):
        raise ValueError("class probabilities cannot be negative")

    totals = arr.sum(axis=2, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("class probabilities must have a positive sum")
    arr = arr / totals
    return np.clip(arr, _EPS, 1.0)


def _minmax(values: np.ndarray) -> np.ndarray:
    """Min-max normalize an array without producing NaN for constant input."""
    values = np.asarray(values, dtype=np.float64)
    minimum = values.min()
    span = values.max() - minimum
    if span <= _EPS:
        return np.zeros_like(values)
    return (values - minimum) / span


def _categorical_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Normalized categorical entropy along the final axis."""
    classes = probabilities.shape[-1]
    return -(probabilities * np.log(probabilities) / np.log(classes)).sum(axis=-1)


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    """Binary entropy of predicted class versus all remaining classes."""
    p = np.clip(np.asarray(probability, dtype=np.float64), _EPS, 1.0 - _EPS)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def _stable_predicted_class(obs: np.ndarray) -> np.ndarray:
    """Predicted class selected from the mean of the stochastic observations."""
    return obs.mean(axis=0).argmax(axis=1)


def _predicted_class_probabilities(obs: np.ndarray) -> np.ndarray:
    """Return p(y_hat|x) in every observation, with shape ``(T, N)``."""
    predicted = _stable_predicted_class(obs)
    return obs[:, np.arange(obs.shape[1]), predicted]


def misclassification_probability(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return single-pass MP and MC-mean MP."""
    arr = _validate_observations(obs)

    baseline = arr[0]
    baseline_class = baseline.argmax(axis=1)
    mp = 1.0 - baseline[np.arange(len(baseline_class)), baseline_class]

    mean_probability = arr.mean(axis=0)
    mean_class = mean_probability.argmax(axis=1)
    mp_mean = 1.0 - mean_probability[np.arange(len(mean_class)), mean_class]
    return mp, mp_mean


def entropy(obs: np.ndarray) -> np.ndarray:
    """Entropy from the first deterministic/baseline observation."""
    return _categorical_entropy(_validate_observations(obs)[0])


def entropy_mean(obs: np.ndarray) -> np.ndarray:
    """E_M: entropy of the mean class-probability vector."""
    arr = _validate_observations(obs)
    return _categorical_entropy(arr.mean(axis=0))


def mean_entropy(obs: np.ndarray) -> np.ndarray:
    """M_E: mean entropy across stochastic observations."""
    arr = _validate_observations(obs)
    return _categorical_entropy(arr).mean(axis=0)


def max_entropy(obs: np.ndarray) -> np.ndarray:
    """Max_E: maximum entropy among stochastic observations."""
    arr = _validate_observations(obs)
    return _categorical_entropy(arr).max(axis=0)


def std_predicted_prob(obs: np.ndarray, normalize: bool = True) -> np.ndarray:
    """DPP/DPkP: deviation of the stable predicted-class probability.

    The important class is selected once from the mean predictive distribution,
    and the standard deviation of that same class probability is measured over
    all stochastic observations.
    """
    arr = _validate_observations(obs)
    deviation = _predicted_class_probabilities(arr).std(axis=0)
    return _minmax(deviation) if normalize else deviation


def predicted_vs_rest_entropy(obs: np.ndarray) -> np.ndarray:
    """Mean predicted-versus-rest entropy across stochastic observations."""
    arr = _validate_observations(obs)
    per_observation = _binary_entropy(_predicted_class_probabilities(arr))
    return per_observation.mean(axis=0)


def predicted_vs_rest_entropy_deviation(
    obs: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Deviation of predicted-versus-rest entropy across observations."""
    arr = _validate_observations(obs)
    per_observation = _binary_entropy(_predicted_class_probabilities(arr))
    deviation = per_observation.std(axis=0)
    return _minmax(deviation) if normalize else deviation


def eprds(obs: np.ndarray, normalize_deviation: bool = True) -> np.ndarray:
    """Entropy of Predicted vs Rest with Deviation Scaling (EPRDS).

    For each sample, the stable predicted class is obtained from the mean MC
    predictive distribution. Each observation is reduced to a binary
    distribution: predicted class versus the sum of all remaining classes.
    EPRDS multiplies the mean binary entropy by the across-observation entropy
    deviation. By default, the deviation term is min-max normalized over the
    evaluated dataset, matching the paper's use of a [0, 1] uncertainty scale.
    """
    arr = _validate_observations(obs)
    per_observation = _binary_entropy(_predicted_class_probabilities(arr))
    mean_epr = per_observation.mean(axis=0)
    deviation = per_observation.std(axis=0)
    if normalize_deviation:
        deviation = _minmax(deviation)
    return mean_epr * deviation


# Paper-style aliases -------------------------------------------------------
# These aliases make the code and manuscript notation directly comparable.
M_E = mean_entropy
E_M = entropy_mean
Max_E = max_entropy
DPkP = std_predicted_prob
DPP = std_predicted_prob
EPR = predicted_vs_rest_entropy
EPRD = predicted_vs_rest_entropy_deviation
EPRDS = eprds
