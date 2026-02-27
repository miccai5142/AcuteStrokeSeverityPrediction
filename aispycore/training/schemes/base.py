from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from aispycore.config.schema import PipelineConfig
from aispycore.tasks import Task
from aispycore.training.trainer import TrainingResult


# ---------------------------------------------------------------------------
# task_type string → Task enum
# ---------------------------------------------------------------------------

_TASK_TYPE_MAP: dict[str, Task] = {
    "binary_classification": Task.BINARY_CLASSIFICATION,
    "regression":            Task.REGRESSION,
    "multilabel_classification": Task.MULTILABEL_CLASSIFICATION,
}


def task_from_config(config: PipelineConfig) -> Task:
    """
    Derive the Task enum from config.model.task_type.
    Raises a clear ValueError for unknown task types.
    """
    task_type = config.model.task_type
    if task_type not in _TASK_TYPE_MAP:
        available = ", ".join(_TASK_TYPE_MAP.keys())
        raise ValueError(
            f"Unknown task_type '{task_type}' in config.model. "
            f"Expected one of: {available}"
        )
    return _TASK_TYPE_MAP[task_type]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Results for a single fold or split."""
    fold_index: int
    training_result: TrainingResult
    metrics: dict
    predictions_df: pd.DataFrame
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base scheme
# ---------------------------------------------------------------------------

class BaseTrainingScheme(ABC):

    def __init__(
        self,
        config: PipelineConfig,
        output_dir: Path,
        log_file: str,
        seed: Optional[int] = None
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.seed = seed
        self.log_file = log_file
        # Derived from config — not passed in separately
        self.task = task_from_config(config)

    @abstractmethod
    def run(self, df: pd.DataFrame) -> list[FoldResult]:
        """
        Run the full training scheme on the given dataframe.

        Args:
            df: Full dataset dataframe (loaded from the CSV file).

        Returns:
            List of FoldResult, one per fold or split.
        """
        ...

    def _make_fold_output_dir(self, fold_idx: int) -> Path:
        fold_dir = self.output_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        return fold_dir