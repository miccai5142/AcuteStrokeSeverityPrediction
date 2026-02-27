from aispycore.training.schemes.base import BaseTrainingScheme, FoldResult, task_from_config
from aispycore.training.schemes.kfold import KFoldScheme
from aispycore.training.schemes.holdout import HoldoutScheme
from aispycore.training.schemes.nested_kfold import NestedKFoldScheme


def get_scheme(name: str) -> type[BaseTrainingScheme]:
    if name == "kfold":
        return KFoldScheme
    elif name == "holdout":
        return HoldoutScheme
    elif name == "nested_kfold":
        return NestedKFoldScheme
    else:
        raise ValueError(
            f"Unknown training scheme '{name}'. "
            "Expected 'kfold', 'holdout', or 'nested_kfold'."
        )


__all__ = [
    "BaseTrainingScheme", "FoldResult", "task_from_config",
    "KFoldScheme", "HoldoutScheme", "NestedKFoldScheme",
    "get_scheme",
]