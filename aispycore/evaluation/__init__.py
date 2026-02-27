"""
aispycore.evaluation

Public API for model evaluation and result aggregation.
"""

from aispycore.evaluation.evaluators import (
    BaseEvaluator,
    BinaryClassificationEvaluator,
    RegressionEvaluator,
    MultilabelClassificationEvaluator,
    get_evaluator,
    EVALUATOR_REGISTRY,
)
from aispycore.evaluation.results import aggregate_fold_results

__all__ = [
    # Evaluators
    "BaseEvaluator",
    "BinaryClassificationEvaluator",
    "RegressionEvaluator",
    "MultilabelClassificationEvaluator",
    "get_evaluator",
    "EVALUATOR_REGISTRY",
    # Aggregation
    "aggregate_fold_results",
]