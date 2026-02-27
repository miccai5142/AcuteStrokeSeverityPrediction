from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import tensorflow as tf

from aispycore.config.schema import HyperparameterSet, PipelineConfig
from aispycore.training.bayestuner import CustomBayesianOptimization
from aispycore.training.callbacks import EpochLogger

log = logging.getLogger(__name__)


def get_pretuned_hps(config, fold_idx: int):
    """
    Return a HyperparameterSet from config.tuning.pretuned_hyperparameters,
    or None if no pre-tuned values are configured.

    Args:
        config:   PipelineConfig whose tuning.pretuned_hyperparameters is checked.
        fold_idx: 0-based fold index (outer fold for nested CV, fold for kfold,
                  always 0 for holdout).

    Returns:
        HyperparameterSet if pre-tuned values exist for this fold, else None.

    Raises:
        ValueError: If a list is provided but has fewer entries than fold_idx + 1.
    """
    raw = config.tuning.pretuned_hyperparameters
    if raw is None:
        return None

    if isinstance(raw, dict):
        # Single set — applies to every fold unchanged.
        return HyperparameterSet(values=raw)

    # Per-fold list
    if len(raw) == 1:
        return HyperparameterSet(values=raw[0])

    if fold_idx >= len(raw):
        raise ValueError(
            f"tuning.pretuned_hyperparameters has {len(raw)} entries but "
            f"fold_idx={fold_idx} was requested. "
            "Provide one entry per fold, or a single dict to reuse across all folds."
        )
    return HyperparameterSet(values=raw[fold_idx])


class BaseTuner(ABC):

    def __init__(self, config: PipelineConfig, output_dir: Path, log_file) -> None:
        self.config = config
        self.output_dir = output_dir / "tuning"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_file

    @abstractmethod
    def tune(
        self,
        train_dataset: tf.data.Dataset,
        train_steps: int,
        val_dataset: tf.data.Dataset,
        val_steps: int
    ) -> HyperparameterSet:
        """Run tuning and return the best hyperparameters."""
        ...


class KerasTuner(BaseTuner):
    """
    Tuner backed by CustomBayesianOptimization.
    """

    def tune(
        self,
        train_dataset: tf.data.Dataset,
        train_steps: int,
        val_dataset: tf.data.Dataset,
        val_steps: int
    ) -> HyperparameterSet:
        try:
            import kerastuner as kt
        except ImportError:
            raise ImportError(
                "keras-tuner is required for tuning. "
                "Install with: pip install keras-tuner"
            )

        from aispycore.models.builder import build_tuning_model

        tc = self.config.tuning
        config = self.config  # closure capture

        # ------------------------------------------------------------------
        # GPU memory growth
        # ------------------------------------------------------------------
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                log.info("GPU memory growth enabled for %d device(s).", len(gpus))
            except RuntimeError as e:
                log.warning("Could not enable GPU memory growth: %s", e)

        tf.keras.backend.clear_session()
        gc.collect()

        # ------------------------------------------------------------------
        # Build tuner
        # ------------------------------------------------------------------
        def hypermodel(hp):
            return build_tuning_model(config, hp)

        tuner = CustomBayesianOptimization(
            hypermodel=hypermodel,
            objective=tc.objective,
            objective_direction=tc.objective_direction,
            max_trials=tc.max_trials,
            directory=str(self.output_dir),
            project_name="tuning",
            overwrite=True,
            last_n_epochs=tc.last_n_epochs,
            improvement_weight=tc.improvement_weight,
            exclude_increasing_loss=tc.exclude_increasing_loss,
            max_direction_changes=tc.max_direction_changes,
            class_weight=getattr(self.config.training, 'class_weight', None),
        )

        # ------------------------------------------------------------------
        # Callbacks — EarlyStopping always present; LR scheduler optional
        # ------------------------------------------------------------------
        tuning_log_metrics = list(config.training.metrics or [])
        if tc.objective not in tuning_log_metrics:
            tuning_log_metrics.append(tc.objective)
        epoch_logger = EpochLogger(self.log_file, tuning_log_metrics)
        from tensorflow.keras.callbacks import EarlyStopping
        callbacks = [
            EarlyStopping(
                monitor=tc.objective,
                mode=tc.objective_direction,
                min_delta=tc.trial_early_stopping_min_delta,
                patience=tc.trial_early_stopping_patience,
            ),
            epoch_logger,
        ]

        if getattr(self.config.training, "lr_scheduler", None) == "Stepwise":
            try:
                from aispycore.callbacks.keras_callbacks import stepwise_lr
                from tensorflow.keras.callbacks import LearningRateScheduler
                callbacks.append(LearningRateScheduler(stepwise_lr))
                log.info("Using Stepwise LR scheduler during tuning.")
            except ImportError:
                log.warning(
                    "lr_scheduler='Stepwise' set but aispycore is not installed. "
                    "Skipping stepwise LR during tuning."
                )

        # ------------------------------------------------------------------
        # Search
        # ------------------------------------------------------------------
        log.info(
            "Starting tuning search: max_trials=%d, tuning_epochs=%d, "
            "last_n_epochs=%d, improvement_weight=%.2f, "
            "exclude_increasing_loss=%s",
            tc.max_trials, tc.tuning_epochs, tc.last_n_epochs,
            tc.improvement_weight, tc.exclude_increasing_loss,
        )

        tuner.search(
            train_dataset,
            validation_data=val_dataset,
            epochs=tc.tuning_epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            verbose=2,
            callbacks=callbacks,
        )

        # ------------------------------------------------------------------
        # Log trial summary — mirrors tune_model() reporting block
        # ------------------------------------------------------------------
        total_trials = len(tuner.trial_histories)
        valid_count = sum(
            1 for tid in tuner.trial_histories
            if tuner._is_trial_valid(tid)
        )
        excluded_count = len(tuner.excluded_trials)

        log.info(
            "Tuning complete. Valid trials: %d/%d, "
            "Excluded (with val_loss increases): %d/%d",
            valid_count, total_trials, excluded_count, total_trials,
        )

        trials_info = tuner.get_best_trials_info(
            num_trials=min(5, total_trials)
        )
        log.info("Top trials summary:")
        for i, info in enumerate(trials_info, 1):
            status = (
                "VALID" if info["is_valid"]
                else f"INVALID (increases: {info['increase_count']})"
            )
            last_n = info[f"all_{tc.objective}"][-tc.last_n_epochs:]
            log.info(
                "  Trial %d [%s]: mean_val_loss=%.6f  improvement=%.2f%%  "
                "val_losses=%s",
                i, status, info["mean_val_loss"], info["improvement_pct"],
                [f"{v:.6f}" for v in info[f"all_{tc.objective}"]],
            )

        # ------------------------------------------------------------------
        # Extract best HPs and log selection rationale
        # ------------------------------------------------------------------
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        try:
            best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
            metrics = tuner.get_trial_metrics(best_trial.trial_id)
            if metrics:
                reason = (
                    "valid trial" if not metrics["is_excluded"]
                    else f"best invalid trial (fewest increases: {metrics['increase_count']})"
                )
                log.info(
                    "Selected best trial (%s): "
                    "mean_val_loss=%.6f  val_loss_pct_change=%.2f%%  "
                    "increases=%d  all_%s=%s",
                    reason,
                    metrics["mean_val_loss_last_n"],
                    metrics["improvement_pct"],
                    metrics["increase_count"],
                    tc.objective,
                    [f"{v:.6f}" for v in metrics[f"{tc.objective}"]],
                )
        except Exception:
            pass  # trial detail logging is best-effort

        log.info(
            "Best hyperparameters — dropout=%.4f  weight_decay=%.6f  "
            "learning_rate=%.6f",
            best_hps.get("dropout"),
            best_hps.get("weight_decay"),
            best_hps.get("learning_rate"),
        )

        # ------------------------------------------------------------------
        # Cleanup
        # ------------------------------------------------------------------
        tuner = None  # noqa: F841
        tf.keras.backend.clear_session()
        gc.collect()

        return HyperparameterSet(values={
            "model.dropout":          best_hps.get("dropout"),
            "model.weight_decay":     best_hps.get("weight_decay"),
            "training.learning_rate": best_hps.get("learning_rate"),
        })


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TUNER_REGISTRY: dict[str, type[BaseTuner]] = {
    "keras_tuner": KerasTuner,
}


def get_tuner(config: PipelineConfig, output_dir: Path, log_file: str) -> BaseTuner:
    backend = config.tuning.backend
    if backend not in TUNER_REGISTRY:
        available = ", ".join(TUNER_REGISTRY.keys())
        raise KeyError(
            f"Unknown tuning backend '{backend}'. Available: {available}"
        )
    return TUNER_REGISTRY[backend](config, output_dir, log_file)