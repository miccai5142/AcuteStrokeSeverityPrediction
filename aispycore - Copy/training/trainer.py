from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import tensorflow as tf

from aispycore.config.schema import PipelineConfig
from aispycore.training.callbacks import build_callbacks


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """
    Everything produced by a training run.

    Use this dataclass, not loose tuples. It's explicit and extensible
    without breaking callers when new fields are added.
    """
    history_df: pd.DataFrame
    best_weights_path: Path
    final_epoch: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------

class BaseTrainer(ABC):
    """
    Abstract trainer. Subclasses implement _run_training_loop().
    """

    def __init__(self, config: PipelineConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        train_steps: int,
        val_dataset: tf.data.Dataset,
        val_steps: int
    ) -> TrainingResult:
        """
        Public interface. Builds callbacks, delegates to _run_training_loop,
        and packages results.
        """

        best_weights_path = self.output_dir / "training_checkpoint.weights.h5"
        callbacks = build_callbacks(self.config, self.output_dir, best_weights_path,
                            val_dataset=val_dataset, val_steps=val_steps)
        history = self._run_training_loop(model, train_dataset, train_steps, val_dataset, val_steps, callbacks)

        history_df = pd.DataFrame(history.history)
        
        final_epoch = len(history_df)

        return TrainingResult(
            history_df=history_df,
            best_weights_path=best_weights_path,
            final_epoch=final_epoch,
        )

    @abstractmethod
    def _run_training_loop(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        train_steps: int,
        val_dataset: tf.data.Dataset,
        val_steps: int,
        callbacks: list,
    ) -> tf.keras.callbacks.History:
        ...


# ---------------------------------------------------------------------------
# Standard trainer (model.fit)
# ---------------------------------------------------------------------------

class StandardTrainer(BaseTrainer):
    """Trainer that delegates to model.fit(). Suitable for most use cases."""

    def _run_training_loop(self, model, train_dataset, train_steps, val_dataset, val_steps, callbacks):
        class_weight = getattr(self.config.training, "class_weight", None)
        return model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.training.epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2,
        )


# ---------------------------------------------------------------------------
# Gradual (blockwise) unfreezing trainer
# ---------------------------------------------------------------------------

class GradualUnfreezingTrainer(BaseTrainer):
    """
    Trainer that progressively unfreezes layer blocks during training.

    Config (training.trainer_kwargs):
        blocks: list of lists, each inner list is a set of layer indices that
                form one block. Blocks are unfrozen from last to first (deepest
                layers last), matching the original implementation.
        min_delta: minimum val_loss improvement to count as progress (default 0.001).

    Example YAML:
        trainer_type: "gradual_unfreezing"
        trainer_kwargs:
          blocks:
            - [0, 1, 2]          # block 0 — earliest layers, unfrozen last
            - [3, 4, 5, 6]       # block 1
            - [7, 8, 9]          # block 2 — output-side layers, unfrozen first
          min_delta: 0.001
    """

    def _run_training_loop(self, model, train_dataset, train_steps, val_dataset, val_steps, callbacks):
        from collections import defaultdict
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.optimizers import SGD, Adam

        tc = self.config.training
        kwargs = tc.trainer_kwargs

        blocks   = kwargs.get("blocks", None)
        min_delta = kwargs.get("min_delta", 0.001)

        # loss_fn and metrics come from config — NEVER from model.loss / model.metrics
        loss_fn  = tc.loss_fn
        metrics  = list(tc.metrics)

        optimizer_name = tc.optimizer.lower()
        def optimizer_fn():
            if optimizer_name == "sgd":
                return SGD(learning_rate=tc.learning_rate)
            elif optimizer_name == "adam":
                from tensorflow.keras.optimizers import Adam
                return Adam(learning_rate=tc.learning_rate)
            else:
                raise ValueError(f"Unknown optimizer '{tc.optimizer}' in GradualUnfreezingTrainer.")

        # If no blocks provided fall back to simple per-layer unfreezing
        if blocks is None:
            blocks = [[i] for i in range(len(model.layers))]

        for layer in model.layers:
            layer.trainable = False

        current_block = len(blocks) - 1  # start at output-side block
        self._unfreeze_block(model, blocks, current_block)
        current_block -= 1

        all_histories = defaultdict(list)

        while True:
            # Fresh optimizer + recompile before each segment
            model.compile(
                optimizer=optimizer_fn(),
                loss=loss_fn,
                metrics=metrics,
            )

            # Fresh EarlyStopping per segment so patience resets after each unfreeze
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=tc.early_stopping_patience,
                restore_best_weights=True,
                min_delta=min_delta,
                verbose=1,
            )

            # Merge scheme-level callbacks (ModelCheckpoint etc.) with per-segment ES
            segment_callbacks = list(callbacks) + [early_stopping]

            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=tc.epochs,
                steps_per_epoch=train_steps,
                validation_steps=val_steps,
                callbacks=segment_callbacks,
                class_weight=getattr(tc, "class_weight", None),
                verbose=1,
            )

            for k, v in history.history.items():
                all_histories[k].extend(v)

            if current_block < 0:
                print("GradualUnfreezingTrainer: all blocks unfrozen. Training complete.")
                self._print_layer_trainability(model)
                break

            self._unfreeze_block(model, blocks, current_block)
            self._print_layer_trainability(model)
            current_block -= 1

        # Return a mock History-like object so BaseTrainer.train() can call .history
        return _DictHistory(dict(all_histories))

    @staticmethod
    def _unfreeze_block(model: tf.keras.Model, blocks: list[list[int]], block_idx: int) -> None:
        """Unfreeze all layers whose indices are listed in blocks[block_idx]."""
        if block_idx < 0 or block_idx >= len(blocks):
            return
        print(f"GradualUnfreezingTrainer: unfreezing block {block_idx} "
              f"(layer indices {blocks[block_idx]})")
        for layer_idx in blocks[block_idx]:
            model.layers[layer_idx].trainable = True

    @staticmethod
    def _print_layer_trainability(model: tf.keras.Model) -> None:
        for layer in model.layers:
            print(f"  {layer.name}: trainable={layer.trainable}")


class _DictHistory:
    """
    Minimal History-like wrapper so BaseTrainer.train() can call .history
    on the dict returned by GradualUnfreezingTrainer without changes to the
    base class interface.
    """
    def __init__(self, history: dict):
        self.history = history


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRAINER_REGISTRY: dict[str, type[BaseTrainer]] = {
    "standard": StandardTrainer,
    "gradual_unfreezing": GradualUnfreezingTrainer,
}


def get_trainer(trainer_type: str) -> type[BaseTrainer]:
    if trainer_type not in TRAINER_REGISTRY:
        available = ", ".join(TRAINER_REGISTRY.keys())
        raise KeyError(f"Unknown trainer '{trainer_type}'. Available: {available}")
    return TRAINER_REGISTRY[trainer_type]