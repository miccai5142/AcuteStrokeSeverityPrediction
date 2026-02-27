from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix
)

from aispycore.tasks import Task
from aispycore.utils.logging import write_log

class BaseEvaluator(ABC):
    """Abstract evaluator. One instance per task type."""

    @abstractmethod
    def evaluate(
        self,
        model: tf.keras.Model,
        test_dataset: tf.data.Dataset,
        test_df: pd.DataFrame,
        test_steps: int
    ) -> tuple[dict, pd.DataFrame]:
        """
        Evaluate model on the test set.

        Args:
            model:        Trained Keras model with best weights loaded.
            test_dataset: Batched tf.data.Dataset (images, labels).
            test_df:      The raw DataFrame slice for the test set.
                          Used to attach patient IDs and true labels to
                          the predictions DataFrame.

        Returns:
            metrics:        dict of metric name -> scalar value.
            predictions_df: DataFrame with columns for patient ID,
                            true label, and prediction(s).
        """
        ...

    def _collect_predictions(
        self, model: tf.keras.Model, dataset: tf.data.Dataset, test_df:pd.DataFrame, test_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference over the full dataset and collect (y_true, y_pred)."""

        # Predictions
        preds = model.predict(dataset, steps=test_steps, verbose=2)
        preds = preds.flatten()[: len(test_df)]

        true_labels = []
        for _, y in dataset.take(test_steps):
            true_labels.append(y.numpy())
        true_labels = np.concatenate(true_labels).flatten()[: len(test_df)]


        return true_labels, preds


class BinaryClassificationEvaluator(BaseEvaluator):

    def evaluate(self, model, test_dataset, test_df, test_steps):
        y_true, y_pred_prob = self._collect_predictions(model, test_dataset, test_df, test_steps)
        # y_pred_prob = y_pred_prob.squeeze()
        y_pred_label = (y_pred_prob >= 0.5).astype(int)
        
        # Confusion matrix: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = {
            "auc": roc_auc_score(y_true, y_pred_prob),            
            "accuracy": accuracy_score(y_true, y_pred_label),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred_label),
            "f1": f1_score(y_true, y_pred_label, zero_division=0),
            "recall": recall_score(y_true, y_pred_label, zero_division=0),
            "specificity": specificity,
            "n_test": len(y_true),
        }

        predictions_df = test_df.copy()
        predictions_df["y_true"] = y_true
        predictions_df["y_pred_prob"] = y_pred_prob
        predictions_df["y_pred_label"] = y_pred_label

        return metrics, predictions_df


class RegressionEvaluator(BaseEvaluator):

    def evaluate(self, model, test_dataset, test_df, test_steps):
        y_true, y_pred = self._collect_predictions(model, test_dataset, test_df, test_steps)
        y_pred = y_pred.squeeze()

        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "n_test": len(y_true),
        }

        predictions_df = test_df.copy()
        predictions_df["y_true"] = y_true
        predictions_df["y_pred"] = y_pred

        return metrics, predictions_df


class MultilabelClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for multilabel classification.
    Computes per-label and macro-averaged metrics.
    """

    def evaluate(self, model, test_dataset, test_df, test_steps):
        y_true, y_pred_prob = self._collect_predictions(model, test_dataset, test_df, test_steps)
        y_pred_label = (y_pred_prob >= 0.5).astype(int)
        n_labels = y_true.shape[1]

        per_label_auc = {}
        per_label_f1 = {}
        for i in range(n_labels):
            per_label_auc[f"auc_label_{i}"] = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
            per_label_f1[f"f1_label_{i}"] = f1_score(y_true[:, i], y_pred_label[:, i], zero_division=0)

        metrics = {
            "macro_auc": roc_auc_score(y_true, y_pred_prob, average="macro"),
            "macro_f1": f1_score(y_true, y_pred_label, average="macro", zero_division=0),
            "n_test": len(y_true),
            **per_label_auc,
            **per_label_f1,
        }

        predictions_df = test_df.copy()
        for i in range(n_labels):
            predictions_df[f"y_true_{i}"] = y_true[:, i]
            predictions_df[f"y_pred_prob_{i}"] = y_pred_prob[:, i]
            predictions_df[f"y_pred_label_{i}"] = y_pred_label[:, i]

        return metrics, predictions_df


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EVALUATOR_REGISTRY: dict[Task, BaseEvaluator] = {
    Task.BINARY_CLASSIFICATION: BinaryClassificationEvaluator(),
    Task.REGRESSION: RegressionEvaluator(),
    Task.MULTILABEL_CLASSIFICATION: MultilabelClassificationEvaluator(),
}


def get_evaluator(task: Task) -> BaseEvaluator:
    if task not in EVALUATOR_REGISTRY:
        available = [t.name for t in EVALUATOR_REGISTRY.keys()]
        raise KeyError(
            f"No evaluator registered for task '{task.name}'. "
            f"Available tasks: {', '.join(available)}"
        )
    return EVALUATOR_REGISTRY[task]
