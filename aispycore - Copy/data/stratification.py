from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

from aispycore.config.schema import CrossValidationConfig


class StratificationManager:
    """
    Unified interface over sklearn and iterstrat stratifiers.

    Callers always receive integer index arrays and never need to know
    whether single-label or multilabel splitting is active underneath.
    """

    def __init__(
        self,
        stratify_cols: list[str],
        n_folds: int = 5,
        val_size: float = 0.15,
        priority_weights: dict[str, int] | None = None,
        binning_threshold: int = 10,
        random_state: int | None = None,
    ) -> None:
        if not stratify_cols:
            raise ValueError("At least one stratification column must be specified.")

        self.stratify_cols = stratify_cols
        self.n_folds = n_folds
        self.val_size = val_size
        self.priority_weights = priority_weights or {}
        self.binning_threshold = binning_threshold
        self.random_state = random_state
        self.is_multilabel = len(stratify_cols) > 1

    @classmethod
    def from_config(
        cls, config: CrossValidationConfig, target_col: str, random_state: int | None = None
    ) -> StratificationManager:
        # check if stratify_cols is empty and if so, default to target_col for stratification
        stratify_cols = config.stratify_cols
        if not stratify_cols:
            stratify_cols = [target_col]
        return cls(
            stratify_cols=stratify_cols,
            n_folds=config.n_folds,
            val_size=config.val_size,
            priority_weights=config.priority_weights,
            binning_threshold=config.binning_threshold,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def split(
        self, df: pd.DataFrame
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Yield (train_inner_idx, val_idx, test_idx) for each fold.

        The outer k-fold produces the rotating test set — every sample
        appears as a test sample exactly once across all folds.
        The val set is carved from the outer train set per fold via
        a stratified hold-out (used for early stopping in DL training).

        Args:
            df: Full dataset DataFrame.

        Yields:
            train_inner_idx: Indices for model training (augmentation applied).
            val_idx:         Indices for validation / early stopping.
            test_idx:        Indices for final held-out evaluation this fold.
        """
        stratify_data = self._build_stratify_data(df)
        outer_kf = self._make_kfold(self.n_folds)

        for train_outer_idx, test_idx in outer_kf.split(
            np.arange(len(df)), stratify_data
        ):
            # Split outer train -> inner train + val for DL early stopping
            train_inner_idx, val_idx = self._carve_val_from_train(
                train_outer_idx, stratify_data, self.val_size
            )
            yield train_inner_idx, val_idx, test_idx

    def single_split(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return a single (train_idx, val_idx, test_idx) split for the holdout scheme.

        Args:
            df:        Full dataset DataFrame.
            test_size: Fraction of the full dataset to hold out as test.

        Returns:
            train_idx, val_idx, test_idx as integer index arrays.
        """
        stratify_data = self._build_stratify_data(df)
        all_idx = np.arange(len(df))

        # 1. Hold out test set from full dataset
        trainval_idx, test_idx = self._stratified_shuffle_split(
            all_idx, stratify_data, size=test_size
        )

        # 2. Carve val from trainval
        train_idx, val_idx = self._carve_val_from_train(
            trainval_idx, stratify_data, self.val_size
        )

        return train_idx, val_idx, test_idx

    # ------------------------------------------------------------------
    # Stratification matrix construction
    # ------------------------------------------------------------------

    def _build_stratify_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build the array passed to the stratifier's split() call.

        Single-label: 1-D array of label values (or quantile-binned values).
        Multilabel:   2-D one-hot matrix where priority_weights repeats a
                    column's block to give it proportionally more influence
                    on stratification balance.

        Continuous columns (unique values > binning_threshold) are quantile-
        binned into 5 bins before one-hot encoding.

        Modified behavior:
        - Values == 0 get a dedicated bin (0).
        - Non-zero values are quantile-binned into 5 bins and shifted to 1..N.
        """
        df_work = df.copy()
        binned_col_map: dict[str, str] = {}

        for col in self.stratify_cols:
            if df[col].nunique() > self.binning_threshold:
                bin_col = f"__bin_{col}"

                series = df[col]
                zero_mask = series == 0

                df_work[bin_col] = np.nan

                # Dedicated zero bin
                df_work.loc[zero_mask, bin_col] = 0

                # Quantile-bin non-zero values
                if (~zero_mask).any():
                    non_zero_bins = pd.qcut(
                        series[~zero_mask],
                        q=5,
                        labels=False,
                        duplicates="drop",
                    )

                    # Shift by +1 so bins become 1..N
                    df_work.loc[~zero_mask, bin_col] = non_zero_bins + 1

                df_work[bin_col] = df_work[bin_col].astype(int)
                binned_col_map[col] = bin_col

        if not self.is_multilabel:
            col = self.stratify_cols[0]
            use_col = binned_col_map.get(col, col)
            return df_work[use_col].to_numpy()

        # Multilabel: build weighted one-hot matrix
        encoded_blocks: list[pd.DataFrame] = []
        for col in self.stratify_cols:
            use_col = binned_col_map.get(col, col)
            one_hot = pd.get_dummies(df_work[use_col], prefix=col)
            repeat = self.priority_weights.get(col, 1)
            for _ in range(repeat):
                encoded_blocks.append(one_hot)

        return pd.concat(encoded_blocks, axis=1).to_numpy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_kfold(self, n_splits: int):
        if self.is_multilabel:
            return MultilabelStratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )

    def _carve_val_from_train(
        self,
        train_outer_idx: np.ndarray,
        stratify_data: np.ndarray,
        val_size: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split outer train indices into (inner_train, val).

        Operates on relative positions within train_outer_idx, then maps
        back to absolute dataset indices before returning.

        Args:
            train_outer_idx: Absolute indices of the outer train set.
            stratify_data:   Full-dataset stratification array — indexed by
                             train_outer_idx to extract the outer-train subset.
            val_size:        Fraction of the outer train set to use as val.

        Returns:
            (train_inner_idx, val_idx) as absolute index arrays.
        """
        if self.is_multilabel:
            outer_train_labels = stratify_data[train_outer_idx]
        else:
            outer_train_labels = stratify_data[train_outer_idx]

        relative_idx = np.arange(len(train_outer_idx))
        rel_train, rel_val = self._stratified_shuffle_split(
            relative_idx, outer_train_labels, size=val_size
        )

        return train_outer_idx[rel_train], train_outer_idx[rel_val]

    def _stratified_shuffle_split(
        self,
        idx: np.ndarray,
        labels: np.ndarray,
        size: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Single stratified shuffle split of idx into (majority, held_out).
        Returns absolute index arrays (values from idx, not positions within it).
        """
        if self.is_multilabel:
            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=size, random_state=self.random_state
            )
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=size, random_state=self.random_state
            )
        majority_rel, held_out_rel = next(splitter.split(idx, labels))
        return idx[majority_rel], idx[held_out_rel]