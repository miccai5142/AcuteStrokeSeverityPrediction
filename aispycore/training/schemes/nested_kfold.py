from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from aispycore.config.schema import HyperparameterSet, PipelineConfig
from aispycore.data.dataset import check_no_overlap, make_dataset_from_df
from aispycore.data.stratification import StratificationManager
from aispycore.evaluation.evaluators import get_evaluator
from aispycore.models.builder import build_model
from aispycore.training.schemes.base import BaseTrainingScheme, FoldResult
from aispycore.training.trainer import get_trainer
from aispycore.training.tuning import get_tuner
from aispycore.utils.logging import write_log


# ---------------------------------------------------------------------------
# HP aggregation helper (module-level so it's easy to unit-test)
# ---------------------------------------------------------------------------

def _aggregate_hyperparameters(hp_sets: List[HyperparameterSet]) -> HyperparameterSet:
    """
    Combine HP values across inner folds into a single HyperparameterSet.

    Strategy:
        Numeric (int / float) → arithmetic mean across all inner folds.
        Non-numeric (str / bool) → majority vote (most frequent value).

    All keys present in any inner fold's HP set are included.  If a key is
    absent from some folds (should not happen with a consistent tuner but
    handled defensively), only the folds that have the key contribute.

    Args:
        hp_sets: Non-empty list of HyperparameterSet objects, one per inner fold.

    Returns:
        A single HyperparameterSet whose values are the aggregated HPs.

    Raises:
        ValueError: If hp_sets is empty.
    """
    if not hp_sets:
        raise ValueError("_aggregate_hyperparameters: received an empty list of HP sets.")
    if len(hp_sets) == 1:
        return hp_sets[0]

    all_keys: set[str] = set()
    for hps in hp_sets:
        all_keys.update(hps.values.keys())

    aggregated: dict = {}
    for key in sorted(all_keys):
        values = [hps.values[key] for hps in hp_sets if key in hps.values]
        if not values:
            continue
        if all(isinstance(v, (int, float)) for v in values):
            aggregated[key] = float(np.mean(values))
        else:
            aggregated[key] = Counter(values).most_common(1)[0][0]

    return HyperparameterSet(values=aggregated)


# ---------------------------------------------------------------------------
# Nested K-fold scheme
# ---------------------------------------------------------------------------

class NestedKFoldScheme(BaseTrainingScheme):
    """
    Nested K-fold cross-validation.

    See module docstring for full architecture description.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> list[FoldResult]:
        config    = self.config
        label_col = config.data.target_col
        id_col    = getattr(config.data, "id_col", "id")

        n_outer_folds  = config.cross_validation.n_folds
        n_inner_folds  = getattr(config.cross_validation, "n_inner_folds",  5)
        final_val_split = getattr(config.cross_validation, "final_val_split", 0.30)

        write_log(
            self.log_file,
            f"NestedKFoldScheme: outer={n_outer_folds} folds, "
            f"inner={n_inner_folds} folds, "
            f"final_val_split={final_val_split:.0%}, "
            f"tuning={'enabled' if config.tuning.enabled else 'disabled'}",
        )

        outer_stratifier = StratificationManager.from_config(
            config.cross_validation, target_col=label_col, random_state=self.seed
        )
        full_stratify = outer_stratifier._build_stratify_data(df)
        outer_kf      = outer_stratifier._make_kfold(n_outer_folds)

        fold_results: list[FoldResult] = []

        for outer_fold_idx, (outer_train_pos, outer_test_pos) in enumerate(
            outer_kf.split(np.arange(len(df)), full_stratify)
        ):
            outer_fold_dir = self._make_outer_fold_dir(outer_fold_idx)
            outer_train_df = df.iloc[outer_train_pos]   # positional — keeps original index
            outer_test_df  = df.iloc[outer_test_pos]

            print(f"\n{'='*60}")
            print(f"  Outer Fold {outer_fold_idx + 1} / {n_outer_folds}")
            print(f"  outer_train={len(outer_train_pos)}  outer_test={len(outer_test_pos)}")
            print(f"{'='*60}")

            write_log(self.log_file, f"\n[Outer fold {outer_fold_idx}] "
                      f"outer_train={len(outer_train_pos)}, outer_test={len(outer_test_pos)}")

            # Leakage guard: outer train vs outer test
            check_no_overlap(
                outer_train_df, outer_test_df,
                id_col=id_col, fold_num=outer_fold_idx, log_file=self.log_file,
            )

            # ------------------------------------------------------------------
            # STAGE 1 — Inner loop: hyperparameter search
            # ------------------------------------------------------------------
            best_config = config
            if config.tuning.enabled:
                write_log(
                    self.log_file,
                    f"[Outer fold {outer_fold_idx}] "
                    f"Starting inner tuning ({n_inner_folds} inner folds).",
                )
                inner_hp_sets = self._run_inner_tuning(
                    outer_train_df  = outer_train_df,
                    outer_fold_idx  = outer_fold_idx,
                    outer_fold_dir  = outer_fold_dir,
                    n_inner_folds   = n_inner_folds,
                    config          = config,
                    label_col       = label_col,
                    id_col          = id_col,
                )
                agg_hps = _aggregate_hyperparameters(inner_hp_sets)
                best_config = config.resolve(agg_hps)
                write_log(
                    self.log_file,
                    f"[Outer fold {outer_fold_idx}] "
                    f"Aggregated HPs (mean of {len(inner_hp_sets)} inner folds): "
                    f"{agg_hps.values}",
                )
            else:
                write_log(
                    self.log_file,
                    f"[Outer fold {outer_fold_idx}] "
                    "Tuning disabled — using base config for final training.",
                )

            # ------------------------------------------------------------------
            # STAGE 2 — Final training split: stratified 70/30 on outer_train_df.
            #
            # _carve_val_from_train() expects:
            #   train_outer_idx — indices into stratify_data
            #   stratify_data   — built from the same DataFrame those indices address
            # ------------------------------------------------------------------
            final_stratifier = StratificationManager(
                stratify_cols     = outer_stratifier.stratify_cols,
                n_folds           = 2,          # not used by _carve_val_from_train
                val_size          = final_val_split,
                priority_weights  = config.cross_validation.priority_weights,
                binning_threshold = config.cross_validation.binning_threshold,
                random_state      = self.seed,
            )
            outer_train_stratify = final_stratifier._build_stratify_data(outer_train_df)
            all_outer_train_pos  = np.arange(len(outer_train_df))

            final_train_local, final_val_local = final_stratifier._carve_val_from_train(
                all_outer_train_pos, outer_train_stratify, final_val_split
            )
            final_train_df = outer_train_df.iloc[final_train_local]
            final_val_df   = outer_train_df.iloc[final_val_local]

            # Leakage guards for final training components
            check_no_overlap(
                final_train_df, final_val_df,
                id_col=id_col, fold_num=outer_fold_idx, log_file=self.log_file,
            )
            check_no_overlap(
                final_train_df, outer_test_df,
                id_col=id_col, fold_num=outer_fold_idx, log_file=self.log_file,
            )
            check_no_overlap(
                final_val_df, outer_test_df,
                id_col=id_col, fold_num=outer_fold_idx, log_file=self.log_file,
            )

            write_log(
                self.log_file,
                f"[Outer fold {outer_fold_idx}] Final split — "
                f"final_train={len(final_train_df)}, "
                f"final_val={len(final_val_df)}, "
                f"test={len(outer_test_df)}",
            )
            write_log(self.log_file,
                f"  final_train class dist: {final_train_df[label_col].value_counts().to_dict()}")
            write_log(self.log_file,
                f"  final_val class dist:   {final_val_df[label_col].value_counts().to_dict()}")
            write_log(self.log_file,
                f"  test class dist:        {outer_test_df[label_col].value_counts().to_dict()}")

            # Steps for Keras (train dataset repeats indefinitely)
            final_train_steps = math.ceil(len(final_train_df) / config.data.batch_size)
            final_val_steps   = math.ceil(len(final_val_df)   / config.data.batch_size)
            test_steps        = math.ceil(len(outer_test_df)  / config.data.batch_size)

            write_log(self.log_file,
                f"  Steps — train={final_train_steps}, val={final_val_steps}, test={test_steps}")

            # ------------------------------------------------------------------
            # Build datasets
            # ------------------------------------------------------------------
            final_dir = outer_fold_dir / "final_training"
            final_dir.mkdir(parents=True, exist_ok=True)

            final_train_ds = make_dataset_from_df(
                sub_df     = final_train_df,
                dir        = self.output_dir,
                name       = f"outer_{outer_fold_idx}_final_train",
                target_col = config.data.target_col,
                batch_size = config.data.batch_size,
                shuffle    = True,
                repeat     = True,
                seed       = config.training.seed,
            )
            final_val_ds = make_dataset_from_df(
                sub_df     = final_val_df,
                dir        = self.output_dir,
                name       = f"outer_{outer_fold_idx}_final_val",
                target_col = config.data.target_col,
                batch_size = config.data.batch_size,
                shuffle    = False,
                repeat     = False,
                seed       = config.training.seed,
            )
            outer_test_ds = make_dataset_from_df(
                sub_df     = outer_test_df,
                dir        = self.output_dir,
                name       = f"outer_{outer_fold_idx}_test",
                target_col = config.data.target_col,
                batch_size = config.data.batch_size,
                shuffle    = False,
                repeat     = False,
                seed       = config.training.seed,
            )

            # ------------------------------------------------------------------
            # STAGE 3 — Train final model with best HPs
            # ------------------------------------------------------------------
            model       = build_model(best_config)
            trainer_cls = get_trainer(best_config.training.trainer_type)
            trainer     = trainer_cls(best_config, final_dir)

            training_result = trainer.train(
                model          = model,
                train_dataset  = final_train_ds,
                train_steps    = final_train_steps,
                val_dataset    = final_val_ds,
                val_steps      = final_val_steps,
            )
            write_log(self.log_file,
                f"[Outer fold {outer_fold_idx}] Final training complete. "
                "Loading best weights for evaluation.")

            # Reload best weights saved by ModelCheckpoint callback
            model.load_weights(str(training_result.best_weights_path))

            # ------------------------------------------------------------------
            # STAGE 4 — Evaluate on the outer held-out test set
            # ------------------------------------------------------------------
            evaluator = get_evaluator(self.task)
            metrics, predictions_df = evaluator.evaluate(
                model, outer_test_ds, outer_test_df, test_steps
            )

            write_log(self.log_file,
                f"[Outer fold {outer_fold_idx}] Evaluation complete: {metrics}")

            # Annotate with fold structure metadata
            metrics["fold"]         = outer_fold_idx
            metrics["n_outer_train"] = len(outer_train_pos)
            metrics["n_final_train"] = len(final_train_df)
            metrics["n_final_val"]   = len(final_val_df)
            metrics["n_test"]        = len(outer_test_pos)

            # Save per-fold outputs
            training_result.history_df.to_csv(outer_fold_dir / "history.csv",     index=False)
            predictions_df.to_csv(            outer_fold_dir / "predictions.csv", index=False)
            pd.DataFrame([metrics]).to_csv(   outer_fold_dir / "metrics.csv",     index=False)

            fold_results.append(FoldResult(
                fold_index      = outer_fold_idx,
                training_result = training_result,
                metrics         = metrics,
                predictions_df  = predictions_df,
                metadata        = {
                    "n_inner_folds":   n_inner_folds,
                    "final_val_split": final_val_split,
                    "aggregated_hps":  agg_hps.values if config.tuning.enabled else {},
                },
            ))

        return fold_results

    # ------------------------------------------------------------------
    # Inner tuning loop
    # ------------------------------------------------------------------

    def _run_inner_tuning(
        self,
        outer_train_df : pd.DataFrame,
        outer_fold_idx : int,
        outer_fold_dir : Path,
        n_inner_folds  : int,
        config         : PipelineConfig,
        label_col      : str,
        id_col         : str,
    ) -> list[HyperparameterSet]:
        """
        Run n_inner_folds independent tuner searches on outer_train_df.

        Each inner fold is a separate train/val split of outer_train_df.
        The tuner searches for the best HPs on inner_train_ds validated
        against inner_val_ds.  The outer test set never enters this loop.

        Args:
            outer_train_df : The outer fold's training set (outer test excluded).
            outer_fold_idx : Index of the current outer fold (for logging/paths).
            outer_fold_dir : Output directory for this outer fold.
            n_inner_folds  : Number of inner folds to run.
            config         : Pipeline config (tuning settings etc.).
            label_col      : Target column name.
            id_col         : Sample ID column for overlap checks.

        Returns:
            List of HyperparameterSet, one per inner fold (length = n_inner_folds).
        """
        # Build the inner-fold stratifier on outer_train_df.
        # We use the same stratify_cols as the outer loop to preserve any
        # multilabel stratification setup, defaulting to label_col if empty.
        stratify_cols = config.cross_validation.stratify_cols or [label_col]
        inner_stratifier = StratificationManager(
            stratify_cols     = stratify_cols,
            n_folds           = n_inner_folds,
            val_size          = 0.0,            # unused in _make_kfold path
            priority_weights  = config.cross_validation.priority_weights,
            binning_threshold = config.cross_validation.binning_threshold,
            random_state      = self.seed,
        )
        inner_stratify = inner_stratifier._build_stratify_data(outer_train_df)
        inner_kf       = inner_stratifier._make_kfold(n_inner_folds)

        hp_sets: list[HyperparameterSet] = []

        for inner_fold_idx, (inner_train_local, inner_val_local) in enumerate(
            inner_kf.split(np.arange(len(outer_train_df)), inner_stratify)
        ):
            inner_train_df = outer_train_df.iloc[inner_train_local]
            inner_val_df   = outer_train_df.iloc[inner_val_local]

            print(f"\n  {'─'*54}")
            print(f"    Outer {outer_fold_idx + 1} / Inner fold {inner_fold_idx + 1}/{n_inner_folds}"
                  f"  —  inner_train={len(inner_train_df)}, inner_val={len(inner_val_df)}")
            print(f"  {'─'*54}")

            write_log(
                self.log_file,
                f"[Outer {outer_fold_idx} / Inner {inner_fold_idx}] "
                f"inner_train={len(inner_train_df)}, inner_val={len(inner_val_df)}",
            )
            write_log(self.log_file,
                f"  inner_train class dist: {inner_train_df[label_col].value_counts().to_dict()}")
            write_log(self.log_file,
                f"  inner_val class dist:   {inner_val_df[label_col].value_counts().to_dict()}")

            # Leakage guard inside inner split
            check_no_overlap(
                inner_train_df, inner_val_df,
                id_col=id_col, fold_num=outer_fold_idx, log_file=self.log_file,
            )

            inner_train_steps = math.ceil(len(inner_train_df) / config.data.batch_size)
            inner_val_steps   = math.ceil(len(inner_val_df)   / config.data.batch_size)

            # Each inner fold gets its own tuning directory so searches
            # don't overwrite each other's keras-tuner project files.
            inner_tuning_dir = (
                outer_fold_dir / "inner_tuning" / f"inner_fold_{inner_fold_idx:02d}"
            )
            inner_tuning_dir.mkdir(parents=True, exist_ok=True)

            inner_train_ds = make_dataset_from_df(
                sub_df     = inner_train_df,
                dir        = self.output_dir,
                name       = f"outer_{outer_fold_idx}_inner_{inner_fold_idx}_train",
                target_col = config.data.target_col,
                batch_size = config.data.batch_size,
                shuffle    = True,
                repeat     = True,
                seed       = config.training.seed,
            )
            inner_val_ds = make_dataset_from_df(
                sub_df     = inner_val_df,
                dir        = self.output_dir,
                name       = f"outer_{outer_fold_idx}_inner_{inner_fold_idx}_val",
                target_col = config.data.target_col,
                batch_size = config.data.batch_size,
                shuffle    = False,
                repeat     = False,
                seed       = config.training.seed,
            )

            tuner    = get_tuner(config, inner_tuning_dir, self.log_file)
            best_hps = tuner.tune(
                train_dataset = inner_train_ds,
                train_steps   = inner_train_steps,
                val_dataset   = inner_val_ds,
                val_steps     = inner_val_steps,
            )

            write_log(
                self.log_file,
                f"[Outer {outer_fold_idx} / Inner {inner_fold_idx}] "
                f"Best HPs: {best_hps.values}",
            )
            hp_sets.append(best_hps)

        return hp_sets

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _make_outer_fold_dir(self, outer_fold_idx: int) -> Path:
        """Create and return the output directory for one outer fold."""
        d = self.output_dir / f"outer_fold_{outer_fold_idx:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d