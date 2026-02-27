from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    target_col: str = "mRS_Label"
    image_col: str = "path"
    id_col: str = "id"
    batch_size: int = 10
    num_parallel_calls: int = 4
    cache: bool = False

    class Config:
        frozen = True


class AugmentationConfig(BaseModel):
    enabled: bool = False
    # Each entry is a dict: {name: str, params: dict}
    # e.g. [{name: "random_flip", params: {axes: [0, 1]}}]
    transforms: list[dict[str, Any]] = Field(default_factory=list)

    class Config:
        frozen = True


class ModelConfig(BaseModel):
    # Which pyment backbone to instantiate: "BinarySFCN" | "RegressionSFCN"
    backbone: str = "BinarySFCN"

    # Task the model is solving. Drives head-swapping logic in the builder
    # and evaluator selection in the training scheme.
    # "binary_classification" | "regression"
    task_type: str = "binary_classification"

    dropout: float = 0.5
    weight_decay: float = 1e-5

    # Path to pre-trained pyment weights passed directly to the pyment constructor.
    # None = random initialisation.
    backbone_weights: Optional[str] = None

    # Only relevant for RegressionSFCN + regression task.
    # Clips output to [min, max], e.g. [0, 6] for mRS score.
    prediction_range: Optional[list[float]] = None

    class Config:
        frozen = True


class TrainingConfig(BaseModel):
    epochs: int = 160
    learning_rate: float = 1e-4

    # Seed stored here so it travels with the resolved config and is logged
    # in the run manifest. Set from CLI in train.py via
    #   cli_overrides["training.seed"] = seed
    # and read by kfold.py when constructing each fold's dataset.
    seed: Optional[int] = None

    # ----- Optimizer -------------------------------------------------------
    # "sgd" | "adam"
    optimizer: str = "sgd"

    # ----- Loss / metrics --------------------------------------------------
    # Passed verbatim to model.compile(loss=...).
    loss_fn: str = "binary_crossentropy"

    # List of Keras metric name strings passed to model.compile(metrics=...).
    metrics: list[str] = Field(default_factory=lambda: ["accuracy"])
    custom_val_metrics: list[str] = Field(default_factory=list)
    # ----- Early stopping --------------------------------------------------
    # Master switch. When False, no EarlyStopping callback is added by
    # build_callbacks(). GradualUnfreezingTrainer always adds its own
    # per-segment EarlyStopping regardless of this flag.
    use_early_stopping: bool = True

    # Keras metric name to monitor (e.g. "val_loss", "val_accuracy").
    # Used by both build_callbacks() and KerasTuner.
    early_stopping_metric: str = "val_loss"

    # "min" (lower is better) or "max" (higher is better).
    early_stopping_mode: str = "min"

    # Minimum change to count as an improvement.
    early_stopping_min_delta: float = 0.0

    # Number of epochs with no improvement before stopping.
    early_stopping_patience: int = 10

    # ----- Learning-rate reduction -----------------------------------------
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7

    # ----- Stepwise LR scheduler -------------------------------------------
    # When True, build_callbacks() adds a LearningRateScheduler that divides
    # the LR by 3 at epochs 20, 40, and 60. Separate from reduce_lr_on_plateau.
    use_stepwise_lr: bool = False

    # ----- Epoch-level logging ---------------------------------------------
    # When True, build_callbacks() adds EpochLogger to write per-epoch
    # metrics (loss, configured metrics, RAM, VRAM) to a log file.
    log_epoch_metrics: bool = True

    # ----- Custom trainers -------------------------------------------------
    trainer_type: str = "standard"
    trainer_kwargs: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True


class CrossValidationConfig(BaseModel):
    n_folds: int = 4
    n_repeats: int = 1
    # Columns used for stratification. Multiple columns = multilabel stratification.
    # If empty, StratificationManager.from_config() falls back to target_col.
    stratify_cols: list[str] = Field(default_factory=list)
    # Priority weights per stratification column for multilabel case.
    # Dict of {col_name: int} — a column repeated N times has N× the influence
    # on the stratification balance.
    priority_weights: Optional[dict[str, int]] = None
    # Fraction of each fold's outer train set held out as validation
    # (for Keras early stopping). Does NOT affect the rotating test set.
    val_size: float = 0.2
    # NestedKFoldScheme settings — ignored by KFoldScheme and HoldoutScheme.
    # Number of inner folds used for hyperparameter search.
    n_inner_folds: int = 5
    # Fraction of the outer training set held out as validation during
    # final model training (after inner-loop HP selection).
    final_val_split: float = 0.30
    # Columns with more unique values than this threshold are quantile-binned
    # before stratification.
    binning_threshold: int = 10

    class Config:
        frozen = True


class TuningConfig(BaseModel):
    enabled: bool = False
    # "keras_tuner" is the only supported backend currently.
    backend: str = "keras_tuner"
    max_trials: int = 20

    # Keras metric name that the tuner optimises (e.g. "val_loss").
    # Must match the metric produced by the compiled model during search.
    # Stored here so CustomBayesianOptimization can access it without
    # needing a reference to PipelineConfig (it stores this as self.objective).
    objective: str = "val_loss"

    # Number of epochs to run per tuning trial (distinct from training epochs).
    tuning_epochs: int = 10

    # --- CustomBayesianOptimization scoring parameters ---
    # Number of most-recent epochs used to compute mean val_loss and
    # improvement score when ranking trials.
    last_n_epochs: int = 3

    # Weight given to the improvement-rate metric (0–1).
    # (1 - improvement_weight) is given to mean objective value.
    # 0.0 = rank purely by mean objective.
    # 1.0 = rank purely by improvement rate.
    improvement_weight: float = 0.5

    # If True, trials where the objective ever increased between consecutive
    # epochs are penalised. They are still ranked and used as fallback
    # if no valid trials exist.
    exclude_increasing_loss: bool = True

    # Direction for the tuning objective.
    # "min" for loss-like metrics (val_loss, val_mae) — lower is better.
    # "max" for score-like metrics (val_balanced_accuracy, val_f1) — higher is better.
    # This controls both the keras-tuner oracle ranking and the per-epoch
    # direction check used to detect "bad" trials (trials where the metric
    # moves in the wrong direction are penalised).
    objective_direction: str = "min"

    # --- Per-trial early stopping -------------------------------------------
    # These are independent of the training.early_stopping_* fields so you
    # can use aggressive values during tuning (cut bad trials fast) without
    # affecting full training runs.
    #
    # trial_early_stopping_patience: epochs without improvement before a
    #   trial is stopped. Rule of thumb: ~20% of tuning_epochs.
    #   e.g. tuning_epochs=40 → trial_early_stopping_patience=8
    #
    # trial_early_stopping_min_delta: minimum change to count as improvement.
    #   A small positive value (e.g. 0.005) avoids keeping trials alive that
    #   are only making negligible progress.
    trial_early_stopping_patience: int = 10
    trial_early_stopping_min_delta: float = 0.0

    # Maximum number of wrong-direction epoch moves a trial may have and
    # still be considered valid. Requires exclude_increasing_loss=True.
    #   0  →  strict: any wrong move marks the trial invalid (default)
    #   2  →  tolerant: up to 2 wrong moves are acceptable
    # Invalid trials are used as fallback only when no valid trials exist,
    # ranked by (increase_count ASC, score ASC).
    max_direction_changes: int = 0


    # ---------------------------------------------------------------------------
    # Pre-tuned hyperparameters (optional)
    # ---------------------------------------------------------------------------
    # When provided, the tuner is skipped entirely and these values are used
    # directly as if they were tuner output.
    #
    # Accepts two forms:
    #
    #   Single set (used for every fold / all outer folds in nested CV):
    #     pretuned_hyperparameters:
    #       model.dropout: 0.35
    #       model.weight_decay: 0.00001
    #       training.learning_rate: 0.0005
    #
    #   Per-fold list (one entry per fold, in fold order):
    #     pretuned_hyperparameters:
    #       - model.dropout: 0.30
    #         training.learning_rate: 0.001
    #       - model.dropout: 0.40
    #         training.learning_rate: 0.0005
    #
    # If a list is provided with fewer entries than there are folds, a
    # ValueError is raised at runtime so misconfigured runs fail fast.
    # If tuning.enabled is False this field is ignored.
    pretuned_hyperparameters: Optional[Union[List[Dict], Dict]] = None


    class Config:
        frozen = True


class OutputConfig(BaseModel):
    save_best_only: bool = True
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"

    class Config:
        frozen = True


# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------

class InferenceConfig(BaseModel):
    """
    Settings used by predict.py (inference + evaluation + embedding extraction).
    Not used during training — safe to omit from training configs.
    """
    embedding_layer_name: Optional[str] = "top_pool"
    batch_size: Optional[int] = None

    class Config:
        frozen = True


# ---------------------------------------------------------------------------
# Hyperparameter set returned by the tuner
# ---------------------------------------------------------------------------

class HyperparameterSet(BaseModel):
    """
    Flat mapping of hyperparameter name -> value returned by the tuner.
    Only contains values that override the base config.
    Keys use dot notation: "training.learning_rate", "model.dropout", etc.
    """
    values: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    cross_validation: CrossValidationConfig = Field(default_factory=CrossValidationConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    class Config:
        frozen = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load and validate a config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def resolve(self, hyperparameters: HyperparameterSet) -> PipelineConfig:
        """
        Merge tuner results into this config, returning a NEW resolved config.
        The original config is never mutated.

        Hyperparameter keys use dot notation, e.g. "training.learning_rate".
        """
        raw = self.model_dump()
        for key, value in hyperparameters.values.items():
            parts = key.split(".")
            node = raw
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value
        return PipelineConfig(**raw)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()