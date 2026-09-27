"""Unit tests for suggest_threshold."""

import pandas as pd
import pytest

from lattereview.utils import suggest_threshold

# 10 items, 4 included. Sorted by probability: 0.95+ 0.90- 0.80+ 0.70+ 0.60- 0.40- 0.30+ 0.20- 0.10- 0.05-
TOY = pd.DataFrame(
    {
        "p": [0.95, 0.90, 0.80, 0.70, 0.60, 0.40, 0.30, 0.20, 0.10, 0.05],
        "label": [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    }
)


def test_hand_computed_thresholds():
    # Recall 0.75 is first reached at p >= 0.70: 4 included, TP=3, precision 0.75, WSS = 6/10 - 0.25 = 0.35.
    assert suggest_threshold(TOY, "p", "label", target_recall=0.75) == pytest.approx(
        {"threshold": 0.70, "recall": 0.75, "precision": 0.75, "n_included": 4, "wss": 0.35}
    )
    # Recall 1.0 needs p >= 0.30: 7 included, precision 4/7, WSS = 3/10 - 0 = 0.3.
    assert suggest_threshold(TOY, "p", "label", target_recall=0.95) == pytest.approx(
        {"threshold": 0.30, "recall": 1.0, "precision": 4 / 7, "n_included": 7, "wss": 0.3}
    )


def test_ties_are_one_threshold_and_bool_labels_work():
    df = pd.DataFrame({"p": [0.9, 0.5, 0.5, 0.5, 0.1], "label": [True, False, True, False, False]})
    result = suggest_threshold(df, "p", "label", target_recall=1.0)
    assert result["threshold"] == 0.5 and result["n_included"] == 4 and result["recall"] == 1.0


def test_missing_values_are_dropped():
    df = pd.concat([TOY, pd.DataFrame({"p": [None], "label": [1]})], ignore_index=True)
    assert suggest_threshold(df, "p", "label", 0.75)["n_included"] == 4


@pytest.mark.parametrize(
    "df, target, message",
    [
        (TOY, 0, "target_recall"),
        (TOY, 1.5, "target_recall"),
        (TOY.assign(label=0), 0.95, "no positive"),
        (TOY.assign(label=2), 0.95, "only 0/1"),
    ],
)
def test_invalid_inputs(df, target, message):
    with pytest.raises(ValueError, match=message):
        suggest_threshold(df, "p", "label", target)
