from __future__ import annotations

import gc
import logging

import numpy as np
import tensorflow as tf


try:
    from keras_tuner.tuners import BayesianOptimization
    from keras_tuner import Objective
except ImportError:
    # Fallback for environments that still have the legacy `kerastuner` package.
    from kerastuner.tuners import BayesianOptimization  # type: ignore[no-redef]
    from kerastuner import Objective  # type: ignore[no-redef]

from aispycore.training.callbacks import ValidationMetricsCallback, _SUPPORTED_CUSTOM_METRICS

log = logging.getLogger(__name__)


class CustomBayesianOptimization(BayesianOptimization):
    """
    BayesianOptimization tuner with flexible multi-epoch trial scoring.

    Args:
        last_n_epochs:           Number of most-recent epochs used for scoring
                                 and mean objective calculation.
        improvement_weight:      Weight given to the improvement-rate component
                                 (0–1). Set to 0.0 to rank purely by mean
                                 objective over last N epochs.
        exclude_increasing_loss: When True, trials are tiered by validity.
                                 Invalid trials (> max_direction_changes wrong
                                 moves) are used only as fallback.
        max_direction_changes:   Number of wrong-direction epoch moves to
                                 tolerate before marking a trial invalid.
                                 0 = any wrong move is invalid (strict default).
                                 2 = up to 2 wrong moves tolerated, etc.
        objective:               Metric name to optimise (default "val_loss").
        objective_direction:     "min" or "max".
        class_weight:            Per-class loss weights forwarded to model.fit().
    """

    def __init__(
        self,
        *args,
        last_n_epochs: int = 3,
        improvement_weight: float = 0.5,
        exclude_increasing_loss: bool = True,
        max_direction_changes: int = 0,
        objective: str = "val_loss",
        objective_direction: str = "min",
        class_weight: dict | None = None,
        **kwargs,
    ):
        # Pass objective + direction to the parent oracle so it ranks trials
        # correctly regardless of whether the metric is loss-like or score-like.
        super().__init__(*args, objective=Objective(objective, direction=objective_direction), **kwargs)
        self.last_n_epochs = last_n_epochs
        self.improvement_weight = improvement_weight
        self.exclude_increasing_loss = exclude_increasing_loss
        self.max_direction_changes = max_direction_changes
        self.objective = objective
        self.objective_direction = objective_direction
        self.class_weight = class_weight  # applied to every trial's model.fit()
        # Per-trial state — populated by run_trial()
        self.trial_histories: dict[str, dict] = {}
        self.excluded_trials: set[str] = set()
        self.trial_increase_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Keras-tuner hook — override run_trial for new API compatibility
    # ------------------------------------------------------------------

    def run_trial(self, trial, *args, **kwargs):
        """
        Build the model, run model.fit(), record history, and return the
        objective metric value as a float.
        """
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)

        # ------------------------------------------------------------------
        # Inject ValidationMetricsCallback for custom (non-Keras-native) metrics
        # ------------------------------------------------------------------
        if self.objective in _SUPPORTED_CUSTOM_METRICS:
            val_dataset = kwargs.get("validation_data")
            val_steps   = kwargs.get("validation_steps")
            if val_dataset is not None and val_steps is not None:
                metric_cb = ValidationMetricsCallback(
                    val_dataset=val_dataset,
                    val_steps=val_steps,
                    metrics=[self.objective],
                )
                # Prepend so it runs before EarlyStopping in the callbacks list.
                existing = list(kwargs.get("callbacks") or [])
                kwargs = {**kwargs, "callbacks": [metric_cb] + existing}


        if self.class_weight is not None and 'class_weight' not in kwargs:
            kwargs = {**kwargs, 'class_weight': self.class_weight}
        history = model.fit(*args, **kwargs)

        # ------------------------------------------------------------------
        # Populate per-trial history and validity state
        # ------------------------------------------------------------------
        trial_id  = trial.trial_id
        objective = self.objective
        obj_values: list[float] = [
            float(v) for v in history.history.get(objective, [])
        ]

        self.trial_histories[trial_id] = {objective: obj_values}

        # Count epochs where the metric moved in the wrong direction and
        # mark the trial invalid if exclude_increasing_loss is True.
        #   "min" metrics: wrong direction = value went UP   (got worse)
        #   "max" metrics: wrong direction = value went DOWN (got worse)
        for i in range(1, len(obj_values)):
            moved_wrong = (
                obj_values[i] > obj_values[i - 1]
                if self.objective_direction == "min"
                else obj_values[i] < obj_values[i - 1]
            )
            if moved_wrong:
                self.trial_increase_counts[trial_id] = (
                    self.trial_increase_counts.get(trial_id, 0) + 1
                )
                if self.exclude_increasing_loss:
                    self.excluded_trials.add(trial_id)

        # ------------------------------------------------------------------
        # Return float — required by new keras_tuner engine.
        # The oracle uses this alongside the Objective direction to rank
        # trials, so we return the best (not worst) epoch value.
        # ------------------------------------------------------------------
        if not obj_values:
            return float("inf") if self.objective_direction == "min" else float("-inf")
        return (
            min(obj_values) if self.objective_direction == "min"
            else max(obj_values)
        )

    def on_trial_end(self, trial):
        """Clear GPU memory and run GC after each trial."""
        super().on_trial_end(trial)
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_trial_score(self, trial):
        """
        Returns a score for this trial where lower is always better.
        Delegates to _score_for_ranking so the oracle and get_best_hyperparameters
        use the same ranking logic. Validity tiering is handled separately in
        get_best_hyperparameters — no penalty is applied here.
        """
        trial_id = trial.trial_id
        if (
            trial_id not in self.trial_histories
            or not self.trial_histories[trial_id].get(self.objective)
        ):
            return super().get_trial_score(trial)
        return self._score_for_ranking(trial_id)

    def get_best_hyperparameters(self, num_trials: int = 1):
        """
        Return best HPs using two-tier selection:

        Tier 1 — Valid trials: increase_count ≤ max_direction_changes.
                  Ranked by _score_for_ranking() (lower is always better).
        Tier 2 — Invalid trials: used only when no valid trials exist.
                  Ranked by (increase_count ASC, score ASC) so the least-bad
                  trial is preferred.

        When exclude_increasing_loss=False all trials are valid (no tiering).
        """
        if num_trials is None:
            num_trials = 1

        try:
            all_trials = self.oracle.get_best_trials(num_trials=1000)
        except Exception:
            all_trials = []
            for trial_id in self.trial_histories:
                try:
                    all_trials.append(self.oracle.get_trial(trial_id))
                except Exception:
                    continue

        if not all_trials:
            raise ValueError("No tuning trials found.")

        valid   = [t for t in all_trials if     self._is_trial_valid(t.trial_id)]
        invalid = [t for t in all_trials if not self._is_trial_valid(t.trial_id)]

        # Rank each tier by our scoring function (direction-aware, mode-aware).
        valid.sort(key=lambda t: self._score_for_ranking(t.trial_id))

        if valid:
            best = valid[:num_trials]
        else:
            log.warning(
                "No valid tuning trials found (all exceeded max_direction_changes=%d). "
                "Falling back to least-bad invalid trial.",
                self.max_direction_changes,
            )
            invalid.sort(
                key=lambda t: (
                    self._get_increase_count(t.trial_id),
                    self._score_for_ranking(t.trial_id),
                )
            )
            best = invalid[:num_trials]

        return [self.oracle.get_trial(t.trial_id).hyperparameters for t in best]

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def get_best_trials_info(self, num_trials: int = 5) -> list[dict]:
        """Return a list of per-trial summary dicts for logging."""
        if num_trials is None:
            num_trials = 5
        try:
            all_trials = self.oracle.get_best_trials(num_trials=1000)
        except Exception:
            all_trials = []
            for trial_id in self.trial_histories:
                try:
                    all_trials.append(self.oracle.get_trial(trial_id))
                except Exception:
                    continue
            all_trials.sort(
                key=lambda t: t.score if t.score is not None else float("inf")
            )

        info_list = []
        for trial in all_trials[:num_trials]:
            metrics = self.get_trial_metrics(trial.trial_id)
            if metrics:
                info_list.append({
                    "trial_id":        trial.trial_id,
                    "score":           trial.score,
                    "is_valid":        self._is_trial_valid(trial.trial_id),
                    "increase_count":  self._get_increase_count(trial.trial_id),
                    "mean_val_loss":   metrics[f"mean_{self.objective}_last_n"],
                    "improvement_pct": metrics["improvement_pct"],
                    f"all_{self.objective}": metrics[f"all_{self.objective}"],
                })
        return info_list

    def get_trial_metrics(self, trial_id: str) -> dict | None:
        """Return the scoring metrics for a single trial (for logging / debugging)."""
        objective = self.objective
        if (
            trial_id not in self.trial_histories
            or not self.trial_histories[trial_id].get(objective)
        ):
            return None

        val_losses = self.trial_histories[trial_id][objective]
        n_epochs = min(len(val_losses), self.last_n_epochs)
        mean_val_loss = float(np.mean(val_losses[-n_epochs:]))
        improvement_score = self._calculate_improvement_score(val_losses, n_epochs)
        has_increases = any(
            val_losses[i] > val_losses[i - 1] for i in range(1, len(val_losses))
        )
        return {
            f"mean_{objective}_last_n": mean_val_loss,
            "improvement_pct":          -improvement_score,  # actual % change
            "n_epochs_used":            n_epochs,
            f"all_{objective}":         val_losses,
            "has_increasing_loss":      has_increases,
            "is_excluded":              trial_id in self.excluded_trials,
            "increase_count":           self._get_increase_count(trial_id),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_trial_valid(self, trial_id: str) -> bool:
        """
        A trial is valid when exclude_increasing_loss is False (no filtering),
        or when its wrong-direction move count is within the allowed threshold.
        """
        if not self.exclude_increasing_loss:
            return True
        return self._get_increase_count(trial_id) <= self.max_direction_changes

    def _get_increase_count(self, trial_id: str) -> int:
        return self.trial_increase_counts.get(trial_id, 0)

    def _score_for_ranking(self, trial_id: str) -> float:
        """
        Returns a score where LOWER is always better, regardless of whether
        the objective is a "min" or "max" metric. Used consistently by
        get_trial_score, get_best_hyperparameters, and the invalid-tier fallback.

        Two modes controlled by improvement_weight:

        improvement_weight == 0.0  →  Pure mean mode
            score = mean_obj * direction_sign
            For "min" metrics: lower mean → lower score → better.
            For "max" metrics: higher mean → more negative score → better.

        improvement_weight  > 0.0  →  Composite mode
            score = (1 - w) * mean_obj * direction_sign
                  + w       * (-improvement_score)
            improvement_score is positive when the metric is moving in the
            right direction, so negating it rewards improving trials.
        """
        if (
            trial_id not in self.trial_histories
            or not self.trial_histories[trial_id].get(self.objective)
        ):
            return float("inf")

        values   = self.trial_histories[trial_id][self.objective]
        n        = min(len(values), self.last_n_epochs)
        mean_obj = float(np.mean(values[-n:]))

        # Flip sign for "max" metrics so that a higher mean produces a lower
        # (better) score — consistent with "lower is better" convention.
        direction_sign = 1.0 if self.objective_direction == "min" else -1.0

        if self.improvement_weight == 0.0:
            return direction_sign * mean_obj

        improvement = self._calculate_improvement_score(values, n)
        return (
            (1 - self.improvement_weight) * direction_sign * mean_obj
            + self.improvement_weight     * (-improvement)
        )

    def _calculate_improvement_score(self, values: list[float], n_epochs: int) -> float:
        """
        Return a positive score when the metric improved over the last n_epochs.

        For "min" metrics (val_loss): improvement = value went DOWN.
            score = -(% change)  →  positive when values decrease.
        For "max" metrics (val_balanced_accuracy, val_f1): improvement = value went UP.
            score = +(% change)  →  positive when values increase.

        A higher score always means a more improving trial, regardless of direction.
        """
        window = values[-n_epochs:] if len(values) >= n_epochs else values
        if len(window) < 2 or window[0] == 0:
            return 0.0
        pct_change = (window[-1] - window[0]) / abs(window[0]) * 100
        return -pct_change if self.objective_direction == "min" else pct_change