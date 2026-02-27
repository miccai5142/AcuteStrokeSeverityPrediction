from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from aispycore.training.schemes.base import FoldResult



_STRUCTURAL_METRIC_COLS: frozenset[str] = frozenset({
    "fold", "repeat", "n_train", "n_val", "n_test",
})


def _performance_cols(metrics_df: pd.DataFrame) -> list[str]:
    """
    Return numeric columns that represent model performance metrics.
    Excludes structural bookkeeping columns.
    """
    numeric = metrics_df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric if c not in _STRUCTURAL_METRIC_COLS]


def _prediction_cols(df: pd.DataFrame) -> list[str]:
    """Return all y_true / y_pred* columns present in df (in original order)."""
    return [c for c in df.columns if c.startswith(("y_true", "y_pred"))]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate_fold_results(
    fold_results: list[FoldResult],
    output_dir: Path,
    *,
    id_col: str = "id",
    repeat: Optional[int] = None,
) -> pd.DataFrame:
    """
    Aggregate per-fold results into summary statistics and combined outputs.

    Parameters
    ----------
    fold_results : list[FoldResult]
        One FoldResult per fold.  HoldoutScheme produces a single-element list;
        KFoldScheme produces one element per fold.
    output_dir : Path
        Root output directory.  A ``summary/`` subdirectory is created here.
    id_col : str, optional
        Column in the test DataFrame that identifies each sample (e.g. "id",
        "subject_id").  Carried through to all_predictions.csv when present.
        Silently omitted if the column is not in predictions_df.
    repeat : int or None, optional
        Repeat index from a multi-repeat parallelised run (``--repeat_number``
        CLI argument).  Written as the leading ``repeat`` column in
        all_predictions.csv and per_fold_metrics.csv.  Omitted when None.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per performance metric with columns:
        [metric, mean, std, median, min, max, per_fold_values].
        Also written to summary/summary_metrics.csv.
    """
    output_dir = Path(output_dir)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    n_folds = len(fold_results)
    scheme_label = "holdout" if n_folds == 1 else "kfold"

    # ------------------------------------------------------------------
    # 1. Combined predictions across all folds → all_predictions.csv
    # ------------------------------------------------------------------
    all_predictions_df = _build_predictions_df(fold_results, id_col=id_col, repeat=repeat)
    all_predictions_df.to_csv(summary_dir / "all_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Per-fold metrics table → per_fold_metrics.csv
    # ------------------------------------------------------------------
    per_fold_df = _build_per_fold_df(fold_results, repeat=repeat)
    per_fold_df.to_csv(summary_dir / "per_fold_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Summary statistics → summary_metrics.csv
    # ------------------------------------------------------------------
    summary_df = _build_summary_df(per_fold_df)
    summary_df.to_csv(summary_dir / "summary_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Console output
    # ------------------------------------------------------------------
    _print_summary(summary_df, scheme_label=scheme_label, n_folds=n_folds)

    return summary_df


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _build_predictions_df(
    fold_results: list[FoldResult],
    id_col: str,
    repeat: Optional[int],
) -> pd.DataFrame:
    """
    Concatenate test-set predictions from all folds.

    Output column order:
        [repeat,]  fold,  [<id_col>,]  <y_true / y_pred* columns>

    Non-prediction columns from the original test_df (e.g. demographic
    variables) are intentionally excluded to keep the file compact and
    task-agnostic.  They are still available in each fold's predictions.csv.
    """
    frames: list[pd.DataFrame] = []

    for fr in fold_results:
        src = fr.predictions_df

        # --- Prediction columns (task-specific, detected dynamically) ---
        pred_cols = _prediction_cols(src)

        # --- Identity column (present only when id_col exists in src) ---
        id_cols: list[str] = [id_col] if (id_col and id_col in src.columns) else []

        # --- Assemble the output slice ---
        out = src[id_cols + pred_cols].copy()
        out.insert(0, "fold", fr.fold_index)
        if repeat is not None:
            out.insert(0, "repeat", repeat)

        frames.append(out)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Enforce column order: [repeat,] fold, [id,] y_*
    ordered: list[str] = []
    if repeat is not None:
        ordered.append("repeat")
    ordered.append("fold")
    if id_col and id_col in combined.columns:
        ordered.append(id_col)
    ordered += [c for c in combined.columns if c not in ordered]

    return combined[ordered]


def _build_per_fold_df(
    fold_results: list[FoldResult],
    repeat: Optional[int],
) -> pd.DataFrame:
    """
    One row per fold, containing all metrics (performance + structural).

    Column order:
        [repeat,]  fold,  <performance metrics>,  n_train,  n_val,  n_test
    """
    rows: list[dict] = []
    for fr in fold_results:
        row = dict(fr.metrics)           # already contains fold, n_train, n_val, n_test
        if repeat is not None:
            row["repeat"] = repeat
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ordered column groups
    front: list[str] = []
    if repeat is not None and "repeat" in df.columns:
        front.append("repeat")
    if "fold" in df.columns:
        front.append("fold")

    size_cols   = [c for c in ("n_train", "n_val", "n_test") if c in df.columns]
    perf_cols   = _performance_cols(df)
    other_cols  = [c for c in df.columns
                   if c not in front and c not in perf_cols and c not in size_cols]

    col_order = front + perf_cols + size_cols + other_cols
    return df[[c for c in col_order if c in df.columns]]


def _build_summary_df(per_fold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics across folds for every performance metric.

    Returns a tidy DataFrame:
        metric | mean | std | median | min | max | per_fold_values

    ``per_fold_values`` is a Python list of the raw per-fold scalars,
    useful for downstream plotting or significance testing without having
    to re-read per_fold_metrics.csv.

    std uses ddof=0 (population std over N folds, not a sample estimate).
    """
    rows: list[dict] = []
    for col in _performance_cols(per_fold_df):
        values = per_fold_df[col].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        rows.append({
            "metric":           col,
            "mean":             float(np.mean(values)),
            "std":              float(np.std(values, ddof=0)),
            "median":           float(np.median(values)),
            "min":              float(np.min(values)),
            "max":              float(np.max(values)),
            "per_fold_values":  values.tolist(),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_summary(
    summary_df: pd.DataFrame,
    scheme_label: str,
    n_folds: int,
) -> None:
    width = 62
    print("\n" + "=" * width)
    if scheme_label == "holdout":
        print("  Evaluation summary  (holdout — single split)")
    else:
        print(f"  Cross-validation summary  ({n_folds} folds)")
    print("=" * width)

    for _, row in summary_df.iterrows():
        name = row["metric"]
        mean = row["mean"]
        std  = row["std"]
        if n_folds > 1:
            print(f"  {name:<30}  {mean:.4f} ± {std:.4f}")
        else:
            # Single split: std is 0 by definition — just show the value
            print(f"  {name:<30}  {mean:.4f}")

    print("=" * width + "\n")