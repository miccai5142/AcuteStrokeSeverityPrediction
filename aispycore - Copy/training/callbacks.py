from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import time
import psutil
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

import logging

from aispycore.utils.logging import write_log

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric names supported by ValidationMetricsCallback
# ---------------------------------------------------------------------------

_SUPPORTED_CUSTOM_METRICS = frozenset({
    "val_balanced_accuracy",
    "val_f1",
    "val_recall",
    "val_specificity",
})


def build_callbacks(config, output_dir, checkpoint_path,
                    val_dataset=None, val_steps=None):
    """
    Build the list of Keras callbacks for a training run.

    Args:
        config:           PipelineConfig.
        output_dir:       Fold output directory (Path).
        checkpoint_path:  Where ModelCheckpoint saves weights (Path).
        val_dataset:      tf.data.Dataset for validation (repeat=False).
                          Required when config.training.custom_val_metrics
                          is non-empty; ignored otherwise.
        val_steps:        Number of validation batches per epoch.
                          Required alongside val_dataset.

    Returns:
        List of Keras Callback instances.
    """
    callbacks = []

    monitor = config.training.early_stopping_metric or "val_loss"

    # ------------------------------------------------------------------
    # ValidationMetricsCallback -- must come BEFORE EarlyStopping and
    # ModelCheckpoint so its metrics are in logs when they read them.
    # ------------------------------------------------------------------
    custom_metrics = list(getattr(config.training, "custom_val_metrics", []))
    if custom_metrics:
        unknown = [m for m in custom_metrics if m not in _SUPPORTED_CUSTOM_METRICS]
        if unknown:
            raise ValueError(
                f"Unknown custom_val_metrics: {unknown}. "
                f"Supported: {sorted(_SUPPORTED_CUSTOM_METRICS)}"
            )
        if val_dataset is None or val_steps is None:
            raise ValueError(
                "custom_val_metrics requires val_dataset and val_steps to be "
                "passed to build_callbacks()."
            )
        callbacks.append(
            ValidationMetricsCallback(
                val_dataset=val_dataset,
                val_steps=val_steps,
                metrics=custom_metrics,
            )
        )

    # ------------------------------------------------------------------
    # ModelCheckpoint
    # ------------------------------------------------------------------
    ckpt_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=True,
        monitor=monitor,
        mode="auto",
        verbose=1,
    )
    callbacks.append(ckpt_cb)

    # ------------------------------------------------------------------
    # Epoch logger
    # ------------------------------------------------------------------
    if config.training.log_epoch_metrics:
        log_file = output_dir / "epoch_metrics.log"
        metrics = list(config.training.metrics or [])
        metrics += list(getattr(config.training, "custom_val_metrics", []))
        callbacks.append(EpochLogger(log_file, metrics))

    # ------------------------------------------------------------------
    # EarlyStopping
    # ------------------------------------------------------------------
    if config.training.use_early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=config.training.early_stopping_patience or 5,
            restore_best_weights=True,
            mode=config.training.early_stopping_mode,
        ))

    # ------------------------------------------------------------------
    # Stepwise LR scheduler
    # ------------------------------------------------------------------
    if config.training.use_stepwise_lr:
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(stepwise_lr))

    return callbacks


# ---------------------------------------------------------------------------
# ValidationMetricsCallback
# ---------------------------------------------------------------------------

class ValidationMetricsCallback(Callback):
    """
    Compute sklearn validation metrics at the end of each epoch and inject
    them into the Keras logs dict so that EarlyStopping and ModelCheckpoint
    can monitor them by name.

    Args:
        val_dataset: Validation tf.data.Dataset (repeat=False, not shuffled).
        val_steps:   Number of batches to consume from val_dataset.
        metrics:     List of metric names to compute, e.g.
                     ["val_balanced_accuracy", "val_f1"].
                     Must be a subset of _SUPPORTED_CUSTOM_METRICS.
    """

    def __init__(self, val_dataset, val_steps, metrics):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_steps   = val_steps
        self.metrics     = metrics

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        y_true_list, y_pred_list = [], []

        # Single pass -- collect (true, pred) in one sweep to avoid
        # iterating the dataset twice.
        for x_batch, y_batch in self.val_dataset.take(self.val_steps):
            preds = self.model(x_batch, training=False)
            y_true_list.append(y_batch.numpy())
            y_pred_list.append(preds.numpy())

        if not y_true_list:
            return  # no data -- skip silently

        # Cast y_true to int — dataset pipelines emit float32 labels and
        # sklearn's f1_score (unlike balanced_accuracy_score) raises
        # ValueError on non-integer inputs or non-binary label sets.
        y_true = np.concatenate(y_true_list).flatten().astype(int)
        y_prob  = np.concatenate(y_pred_list).flatten()
        y_pred  = (y_prob >= 0.5).astype(int)

        for metric_name in self.metrics:
            value = self._compute(metric_name, y_true, y_pred)
            if value is not None:
                logs[metric_name] = value

    @staticmethod
    def _compute(metric_name, y_true, y_pred):
        """
        Compute a single metric. Returns float on success, 0.0 on failure.
        """
        try:
            if metric_name == "val_balanced_accuracy":
                return float(balanced_accuracy_score(y_true, y_pred))

            elif metric_name == "val_f1":
                return float(f1_score(y_true, y_pred, average="binary", zero_division=0))

            elif metric_name == "val_recall":
                return float(recall_score(y_true, y_pred, average="binary", zero_division=0))

            elif metric_name == "val_specificity":
                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):
                    tn, fp = cm[0, 0], cm[0, 1]
                    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                return 0.0  # not binary — return safe fallback

        except Exception as e:
            _log.warning(
                "ValidationMetricsCallback: failed to compute %s: %s: %s — "
                "returning 0.0 so the metric remains monitorable.",
                metric_name, type(e).__name__, e,
            )
            return 0.0


# ---------------------------------------------------------------------------
# EpochLogger
# ---------------------------------------------------------------------------

class EpochLogger(Callback):
    def __init__(self, log_file, metrics):
        super().__init__()
        self.log_file = log_file
        self.metrics = metrics

    def _get_ram_gb(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)

    def _get_gpu_vram_gb(self):
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return None
            mem_info = tf.config.experimental.get_memory_info("GPU:0")
            return mem_info["current"] / (1024 ** 3)
        except Exception:
            return None

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        write_log(self.log_file, f"Epoch {epoch + 1} started")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        duration = time.time() - self.start_time
        parts = []

        if "loss" in logs and logs["loss"] is not None:
            parts.append(f"loss={logs['loss']:.4f}")

        for metric in self.metrics:
            if metric in logs and logs[metric] is not None:
                parts.append(f"{metric}={logs[metric]:.4f}")

        ram_gb = self._get_ram_gb()
        parts.append(f"RAM={ram_gb:.2f}GB")

        gpu_vram_gb = self._get_gpu_vram_gb()
        if gpu_vram_gb is not None:
            parts.append(f"VRAM={gpu_vram_gb:.2f}GB")

        write_log(
            self.log_file,
            f"Epoch {epoch + 1} ended | {', '.join(parts)}, duration={duration:.2f}s",
        )


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def stepwise_lr(epoch, lr):
    if epoch in [20, 40, 60]:
        print(f"Epoch {epoch} reached. Decreasing LR from {lr:.7f} to {lr/3:.7f}")
        return lr / 3
    return lr