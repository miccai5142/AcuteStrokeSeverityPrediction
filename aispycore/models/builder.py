from __future__ import annotations

import tensorflow as tf
from keras import backend as K
from tensorflow.keras import Model, regularizers, initializers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

"""
------------------------------------------------------------------
Attribution

This model builder compiles models using components and pretrained model weights 
imported from the Pyment package:

Repository:
https://github.com/estenhl/pyment-public

Associated publication:
Leonardsen, E.H. et al. (2024).
Constructing personalized characterizations of structural brain
aberrations in patients with dementia using explainable artificial
intelligence. npj Digital Medicine, 7(1), 110.
https://doi.org/10.1038/s41746-024-01123-7

Please cite the above repository and publication if using this code.
------------------------------------------------------------------
"""
from pyment.models import BinarySFCN, RegressionSFCN

from aispycore.config.schema import PipelineConfig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(config: PipelineConfig) -> tf.keras.Model:
    """
    Build and compile a model for training.

    Args:
        config: Fully resolved PipelineConfig (post-tuning if applicable).

    Returns:
        Compiled tf.keras.Model ready for model.fit().
    """
    K.clear_session()

    mc = config.model
    tc = config.training

    model = _build_backbone(
        backbone=mc.backbone,
        task_type=mc.task_type,
        weights=mc.backbone_weights,
        dropout=mc.dropout,
        weight_decay=mc.weight_decay,
        prediction_range=mc.prediction_range,
    )

    _compile(model, tc.optimizer, tc.learning_rate, tc.loss_fn, tc.metrics)
    return model


def build_tuning_model(config: PipelineConfig, hp) -> tf.keras.Model:
    """
    Build and compile a model with keras-tuner hp sampling.

    Args:
        config: Base PipelineConfig (pre-tuning defaults used as hp defaults).
        hp:     keras-tuner HyperParameters object.

    Returns:
        Compiled tf.keras.Model.
    """
    K.clear_session()

    mc = config.model
    tc = config.training

    # Hyperparameter definitions — identical ranges to original build_tuning_model
    hp_dropout = hp.Float(
        "dropout", 0, 0.6, step=0.05, default=mc.dropout
    )
    hp_weight_decay = hp.Float(
        "weight_decay", 0, 1e-3, step=1e-5, default=mc.weight_decay
    )
    hp_learning_rate = hp.Float(
        "learning_rate", 1e-5, 3e-2, sampling="log", default=tc.learning_rate
    )

    model = _build_backbone(
        backbone=mc.backbone,
        task_type=mc.task_type,
        weights=mc.backbone_weights,
        dropout=hp_dropout,
        weight_decay=hp_weight_decay,
        prediction_range=mc.prediction_range,
    )

    _compile(model, tc.optimizer, hp_learning_rate, tc.loss_fn, tc.metrics)
    return model


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_backbone(
    backbone: str,
    task_type: str,
    weights: str | None,
    dropout: float,
    weight_decay: float,
    prediction_range: list[float] | None,
) -> tf.keras.Model:
    """
    Construct a pyment backbone and apply head-swapping where needed.

    Head-swapping rules:
      BinarySFCN  + binary_classification → use as-is
      RegressionSFCN + binary_classification → swap: layers[-4] → Dense(1, sigmoid)
      RegressionSFCN + regression           → use as-is (prediction_range forwarded)
      BinarySFCN  + regression              → swap: layers[-2] → Dense(1, relu)
    """
    regularizer = regularizers.l2(weight_decay)

    # ---- Binary classification ----------------------------------------
    if task_type == "binary_classification":
        base_model = _instantiate_pyment(
            backbone, weights, regularizer, dropout, prediction_range=None
        )
        if backbone == "RegressionSFCN":
            # Replace the regression head with a sigmoid output
            x = base_model.layers[-4].output
            x = Dense(
                1, activation="sigmoid",
                name="RegressionSFCN_predictions",
                kernel_initializer=initializers.GlorotUniform(),
            )(x)
            return Model(inputs=base_model.input, outputs=x)
        else:
            # BinarySFCN used as-is for binary classification
            return base_model

    # ---- Regression ---------------------------------------------------
    elif task_type == "regression":
        if backbone == "RegressionSFCN":
            # prediction_range forwarded to pyment constructor for bounded output
            pr = tuple(prediction_range) if prediction_range is not None else None
            return _instantiate_pyment(
                backbone, weights, regularizer, dropout, prediction_range=pr
            )
        else:
            # BinarySFCN backbone: replace sigmoid head with relu regression head
            base_model = _instantiate_pyment(
                backbone, weights, regularizer, dropout, prediction_range=None
            )
            x = base_model.layers[-2].output
            x = Dense(
                1, activation="relu",
                name="BinarySFCN_predictions",
                kernel_initializer=initializers.GlorotUniform(),
            )(x)
            return Model(inputs=base_model.input, outputs=x)

    else:
        raise ValueError(
            f"Unknown task_type '{task_type}'. "
            "Expected 'binary_classification' or 'regression'."
        )


def _instantiate_pyment(
    backbone: str,
    weights: str | None,
    regularizer,
    dropout: float,
    prediction_range: tuple | None,
) -> tf.keras.Model:
    """Instantiate a pyment model class by backbone name."""
    cls_map = {
        "BinarySFCN": BinarySFCN,
        "RegressionSFCN": RegressionSFCN,
    }
    if backbone not in cls_map:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Expected one of: {list(cls_map.keys())}"
        )
    cls = cls_map[backbone]

    kwargs: dict = dict(weights=weights, regularizer=regularizer, dropout=dropout)
    if prediction_range is not None:
        kwargs["prediction_range"] = prediction_range

    return cls(**kwargs)


def _compile(
    model: tf.keras.Model,
    optimizer_name: str,
    learning_rate: float,
    loss_fn: str,
    metrics: list[str],
) -> None:
    """Compile model in-place. Matches model.compile() call in original code."""
    optimizer_map = {
        "sgd": SGD(learning_rate=learning_rate),
        "adam": Adam(learning_rate=learning_rate),
    }
    if optimizer_name not in optimizer_map:
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. "
            f"Expected one of: {list(optimizer_map.keys())}"
        )
    model.compile(
        optimizer=optimizer_map[optimizer_name],
        loss=loss_fn,
        metrics=metrics,
    )