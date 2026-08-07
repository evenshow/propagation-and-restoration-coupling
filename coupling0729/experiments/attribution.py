"""The phase x mechanism attribution table and its hypothesis tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .factorial import HEADLINE_METRICS

CHANNELS = ("delta_prop", "delta_rec", "delta_int", "total")

CHANNEL_LABELS = {
    "delta_prop": "propagation coupling",
    "delta_rec": "restoration coupling",
    "delta_int": "interaction",
    "total": "total (11 - 00)",
}

METRIC_LABELS = {
    "loss": "1 - R  (overall resilience loss)",
    "Lambda": "Lambda  (depth of drop)",
    "T_rec": "T_rec  (recovery delay)",
    "T_rec_bounded": "T_rec  (censored -> bound; conservative)",
    "lock_in": "lock-in  (never recovers in horizon)",
}


NOT_ADDITIVE_MARK = "  [n differs — not additive]"
TOLERANCE = 1e-9


def _channel_columns(effects: pd.DataFrame, metric: str) -> dict[str, str]:
    return {
        channel: f"{metric}__{channel}"
        for channel in CHANNELS
        if f"{metric}__{channel}" in effects.columns
    }


def _complete_case_mask(effects: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Replications in which every channel of this metric is defined."""
    return effects[list(columns)].notna().all(axis=1)


def attribution_table(
    effects: pd.DataFrame,
    metrics: Sequence[str] = HEADLINE_METRICS,
    statistic: str = "mean",
    complete_case: bool = False,
) -> pd.DataFrame:
    """Rows are metrics, columns are mechanisms.

    A row sums to its total **only if every channel is averaged over the same
    replications**, which in turn requires the metric to be defined for every
    replication in every arm. ``loss``, ``Lambda`` and ``lock_in`` satisfy that by
    construction. ``T_rec`` does not: it does not exist when a run never
    recovers, so its channels are averaged over different subsets and the
    additive identity — which holds exactly *within* a replication — is broken by
    the averaging.

    Rows in that state are marked in the index rather than silently emitted, so a
    table cannot be read as additive when it is not. Pass ``complete_case=True``
    to restrict every channel of a metric to the replications where all of them
    are defined, which restores additivity at the cost of conditioning on the
    outcome (the surviving subsample is the one where even the most strongly
    coupled arm recovered, i.e. the milder scenarios).
    """
    reducer = {"mean": np.nanmean, "median": np.nanmedian}[statistic]
    rows: dict[str, dict[str, float]] = {}

    for metric in metrics:
        columns = _channel_columns(effects, metric)
        frame = effects
        if complete_case and columns:
            frame = effects[_complete_case_mask(effects, list(columns.values()))]

        row = {}
        counts = {}
        for channel in CHANNELS:
            column = columns.get(channel)
            if column is None:
                row[channel], counts[channel] = np.nan, 0
                continue
            values = frame[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            counts[channel] = int(finite.size)
            row[channel] = float(reducer(values)) if finite.size else np.nan

        label = METRIC_LABELS.get(metric, metric)
        present = [c for c in CHANNELS if counts[c] > 0]
        if len(set(counts[c] for c in present)) > 1:
            label += NOT_ADDITIVE_MARK
        rows[label] = row

    table = pd.DataFrame(rows).T[list(CHANNELS)]
    table.index.name = "metric"
    table.columns = [CHANNEL_LABELS[c] for c in CHANNELS]
    return table


def attribution_audit(
    effects: pd.DataFrame,
    metrics: Sequence[str] = HEADLINE_METRICS,
) -> pd.DataFrame:
    """Per-metric evidence for whether its attribution row may be read as additive.

    Reports the sample size behind each channel, whether they agree, and the
    residual ``sum of parts - total``. A non-zero residual is never a numerical
    artefact: it means the channels were averaged over different replications.
    """
    rows = []
    for metric in metrics:
        columns = _channel_columns(effects, metric)
        if not columns:
            continue
        counts = {
            channel: int(effects[column].notna().sum())
            for channel, column in columns.items()
        }
        means = {
            channel: float(np.nanmean(effects[column])) if counts[channel] else np.nan
            for channel, column in columns.items()
        }
        parts = sum(means.get(c, 0.0) for c in ("delta_prop", "delta_rec", "delta_int"))
        total = means.get("total", np.nan)
        residual = parts - total

        complete = int(_complete_case_mask(effects, list(columns.values())).sum())
        rows.append(
            {
                "metric": metric,
                "n_delta_prop": counts.get("delta_prop", 0),
                "n_delta_rec": counts.get("delta_rec", 0),
                "n_delta_int": counts.get("delta_int", 0),
                "n_total": counts.get("total", 0),
                "n_complete_case": complete,
                "samples_consistent": len(set(counts.values())) == 1,
                "sum_of_parts": parts,
                "total": total,
                "residual": residual,
                "additive": bool(abs(residual) < TOLERANCE),
            }
        )
    return pd.DataFrame(rows)


def conditional_recovery_table(
    effects: pd.DataFrame,
    metric: str = "T_rec",
) -> pd.DataFrame:
    """``T_rec`` presented as what it is: a quantity conditional on recovering.

    Each channel is reported with its own sample size and **no total**, because
    summing across channels drawn from different subsamples is exactly the error
    this replaces. The unconditional recovery-side effect belongs in the
    ``lock_in`` row, which is defined for every replication and does decompose
    additively.
    """
    rows = []
    for channel in ("delta_prop", "delta_rec", "delta_int"):
        column = f"{metric}__{channel}"
        if column not in effects.columns:
            continue
        values = effects[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "channel": CHANNEL_LABELS[channel],
                "n": int(finite.size),
                "n_censored": int(values.size - finite.size),
                "mean": float(np.mean(finite)) if finite.size else np.nan,
                "median": float(np.median(finite)) if finite.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def diagonal_dominance(
    effects: pd.DataFrame,
    depth_metric: str = "Lambda",
    delay_metric: str = "T_rec",
) -> pd.DataFrame:
    """Test H1: propagation owns the depth row, restoration owns the delay row.

    For each row the dominance ratio is ``|expected channel| / |off-diagonal
    channel|``. Values comfortably above 1 support phase separation; values near
    or below 1 mean the two coupling phases are not cleanly distinguishable on
    that metric, which is a reportable negative result rather than a failure.
    """
    rows = []
    for metric, expected, other in (
        (depth_metric, "delta_prop", "delta_rec"),
        (delay_metric, "delta_rec", "delta_prop"),
    ):
        on = np.abs(np.nanmean(effects[f"{metric}__{expected}"]))
        off = np.abs(np.nanmean(effects[f"{metric}__{other}"]))
        rows.append(
            {
                "metric": metric,
                "expected_channel": expected,
                "off_diagonal_channel": other,
                "expected_effect": float(on),
                "off_diagonal_effect": float(off),
                "dominance_ratio": float(on / off) if off > 1e-12 else np.inf,
            }
        )
    return pd.DataFrame(rows)


def superadditivity(
    effects: pd.DataFrame,
    metrics: Sequence[str] = HEADLINE_METRICS,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Test H4: is the interaction term positive (couplings amplify each other)?"""
    from .factorial import _bootstrap_ci

    rows = []
    for metric in metrics:
        column = f"{metric}__delta_int"
        if column not in effects:
            continue
        values = effects[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        low, high = _bootstrap_ci(finite, confidence)
        rows.append(
            {
                "metric": metric,
                "mean_interaction": float(np.mean(finite)) if finite.size else np.nan,
                "ci_low": low,
                "ci_high": high,
                "share_positive": float(np.mean(finite > 0)) if finite.size else np.nan,
                "superadditive": bool(low > 0) if np.isfinite(low) else False,
            }
        )
    return pd.DataFrame(rows)


def format_table(table: pd.DataFrame, digits: int = 4) -> str:
    return table.round(digits).to_string()
