"""Helpers for working with decision-model probabilities."""

from typing import Any, Dict

import numpy as np
import pandas as pd


def suggest_threshold(
    df: pd.DataFrame, probability_column: str, label_column: str, target_recall: float = 0.95
) -> Dict[str, Any]:
    """Find the highest probability cutoff that still reaches the target recall on labeled data.

    Items with probability >= threshold count as included. A cutoff fitted on one backend and dataset does not carry
    over to another, so fit it on a labeled sample of your own data.

    Parameters:
        df: DataFrame with one row per item.
        probability_column: Column with the include probabilities (e.g., "round-A_Jev_include_probability").
        label_column: Column with the true labels (1/True = include, 0/False = exclude).
        target_recall: The recall to reach, in (0, 1].

    Returns:
        A dict with `threshold`, `recall`, `precision`, `n_included` (items at or above the threshold), and `wss`
        (work saved over sampling: the fraction of items screened out, minus the recall lost, (TN+FN)/N - (1-recall)).
    """
    if not 0 < target_recall <= 1:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")
    data = df[[probability_column, label_column]].dropna()
    probabilities = data[probability_column].astype(float).to_numpy()
    labels = data[label_column].astype(float).to_numpy()
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{label_column} must contain only 0/1 or boolean labels")
    labels = labels.astype(bool)
    n, n_positive = len(labels), int(labels.sum())
    if n_positive == 0:
        raise ValueError(f"{label_column} has no positive (included) items")

    # Sort by probability, highest first; each distinct probability is a candidate threshold.
    order = np.argsort(-probabilities, kind="stable")
    probabilities, labels = probabilities[order], labels[order]
    true_positives = np.cumsum(labels)
    n_included = np.arange(1, n + 1)
    last_of_group = np.append(probabilities[1:] != probabilities[:-1], True)
    thresholds = probabilities[last_of_group]
    true_positives, n_included = true_positives[last_of_group], n_included[last_of_group]
    recalls = true_positives / n_positive

    i = int(np.argmax(recalls >= target_recall - 1e-12))
    recall = float(recalls[i])
    return {
        "threshold": float(thresholds[i]),
        "recall": recall,
        "precision": float(true_positives[i] / n_included[i]),
        "n_included": int(n_included[i]),
        "wss": float((n - n_included[i]) / n - (1 - recall)),
    }
