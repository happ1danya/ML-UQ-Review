import json

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def cal_bin(uc, truth_flags, slices=10, return_rate=True):
    """Compute per-bin accuracy or count; empty accuracy bins are NaN."""
    uc = np.asarray(uc)
    edges = np.linspace(0, 1, slices + 1)
    results = []
    for index in range(slices):
        if index == slices - 1:
            mask = (uc >= edges[index]) & (uc <= edges[index + 1])
        else:
            mask = (uc >= edges[index]) & (uc < edges[index + 1])
        selected = [truth_flags[i] for i in np.where(mask)[0]]
        if return_rate:
            results.append(selected.count("T") / len(selected) if selected else np.nan)
        else:
            results.append(len(selected))
    return np.asarray(results, dtype=float)


def cal_bin_counts(uc, truth_flags, slices=10):
    return cal_bin(uc, truth_flags, slices, return_rate=False)


def grouper(uc, truth, preds, slices=10):
    """Group ground-truth and predicted labels by uncertainty bin."""
    uc = np.asarray(uc)
    edges = np.linspace(0, 1, slices + 1)
    y_true_bins, y_pred_bins = [], []
    for index in range(slices):
        if index == slices - 1:
            mask = (uc >= edges[index]) & (uc <= edges[index + 1])
        else:
            mask = (uc >= edges[index]) & (uc < edges[index + 1])
        y_true_bins.append(truth[mask])
        y_pred_bins.append(preds[mask])
    return y_true_bins, y_pred_bins


def cal_bin_others(y_true_bins, preds_bins):
    """Precision, recall, and F1 (macro/micro/weighted) for every bin."""
    output = np.full((9, len(y_true_bins)), np.nan)
    for index, (truth, pred) in enumerate(zip(y_true_bins, preds_bins)):
        if len(truth) == 0:
            continue
        kwargs = {"zero_division": 0}
        output[0, index] = precision_score(truth, pred, average="macro", **kwargs)
        output[1, index] = precision_score(truth, pred, average="micro", **kwargs)
        output[2, index] = precision_score(truth, pred, average="weighted", **kwargs)
        output[3, index] = recall_score(truth, pred, average="macro", **kwargs)
        output[4, index] = recall_score(truth, pred, average="micro", **kwargs)
        output[5, index] = recall_score(truth, pred, average="weighted", **kwargs)
        output[6, index] = f1_score(truth, pred, average="macro", **kwargs)
        output[7, index] = f1_score(truth, pred, average="micro", **kwargs)
        output[8, index] = f1_score(truth, pred, average="weighted", **kwargs)
    return output


def build_bins_from_arrays(
    y_onehot,
    *,
    uncertainty,
    predictions,
    truth_flags,
    slices=10,
):
    """Build accuracy, classification-metric, and count curves.

    Parameters are dictionaries keyed by method name. This generic interface
    allows new uncertainty methods to be added without editing this function.
    """
    method_names = list(uncertainty)
    if set(method_names) != set(predictions) or set(method_names) != set(truth_flags):
        raise ValueError("uncertainty, predictions, and truth_flags must share keys")

    u_lists = {
        name: cal_bin(uncertainty[name], truth_flags[name], slices)
        for name in method_names
    }
    c_lists = {
        name: cal_bin_counts(uncertainty[name], truth_flags[name], slices)
        for name in method_names
    }

    ground_truth = np.asarray(y_onehot).argmax(axis=1)
    m_lists = {}
    for name in method_names:
        truth_bins, pred_bins = grouper(
            uncertainty[name], ground_truth, predictions[name], slices
        )
        m_lists[name] = cal_bin_others(truth_bins, pred_bins)

    return u_lists, m_lists, c_lists


def summarise_metric_diffs(u_lists, m_lists, baseline="MP", threshold=0.9):
    """Summarize per-bin metric differences against a baseline method."""
    if baseline not in u_lists:
        raise KeyError(f"baseline {baseline!r} is not present")

    metric_names = [
        "Accuracy",
        "Macro Precision",
        "Micro Precision",
        "Weighted Precision",
        "Macro Recall",
        "Micro Recall",
        "Weighted Recall",
        "Macro F1",
        "Micro F1",
        "Weighted F1",
    ]

    baseline_accuracy = np.asarray(u_lists[baseline])
    baseline_metrics = np.asarray(m_lists[baseline])
    summaries = {}

    for method, accuracy in u_lists.items():
        if method == baseline:
            continue
        accuracy = np.asarray(accuracy)
        metrics = np.asarray(m_lists[method])
        differences = np.vstack(
            [accuracy - baseline_accuracy, metrics - baseline_metrics]
        )

        summary = {}
        for name, values in zip(metric_names, differences):
            finite_values = values[np.isfinite(values)]
            high_accuracy = values[(accuracy > threshold) & np.isfinite(values)]
            summary[name] = {
                "avg90": float(np.mean(high_accuracy)) if high_accuracy.size else None,
                "avg_all": float(np.mean(finite_values)) if finite_values.size else None,
                "max_all": float(np.max(finite_values)) if finite_values.size else None,
            }
        summaries[method] = summary
    return summaries


def save_results_json(
    u_lists,
    m_lists,
    c_lists,
    diff_summary,
    out_path,
    *,
    uncertainty=None,
):
    """Save binned curves, raw uncertainty arrays, and comparisons to JSON."""

    def convert(mapping):
        return {key: np.asarray(value).tolist() for key, value in mapping.items()}

    data = {
        "u_lists": convert(u_lists),
        "m_lists": convert(m_lists),
        "c_lists": convert(c_lists),
        "diff_summary": diff_summary,
    }
    if uncertainty is not None:
        data["uncertainty"] = convert(uncertainty)

    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)
