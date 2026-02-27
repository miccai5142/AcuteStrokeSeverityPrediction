from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

log = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extract per-sample embeddings from an intermediate layer of a Keras model.

    Args:
        model:             Trained tf.keras.Model with weights loaded.
        layer_name_substr: Substring used to locate the embedding layer.
                           The first layer whose name contains this string
                           (case-sensitive) is used. Raises ValueError if
                           no matching layer is found.
    """

    def __init__(self, model: tf.keras.Model, layer_name_substr: str) -> None:
        self.model = model
        self.layer_name_substr = layer_name_substr
        self.embedding_layer_name = self._find_layer(model, layer_name_substr)
        self._intermediate_model = self._build_intermediate_model()
        log.info(
            "EmbeddingExtractor: using layer '%s' (matched substring '%s')",
            self.embedding_layer_name, layer_name_substr,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        dataset: tf.data.Dataset,
        df: pd.DataFrame,
        steps: int,
        config,
    ) -> pd.DataFrame:
        """
        Run inference and return a DataFrame of embeddings with metadata.

        Output columns:
            id, path, <target_col>, y_true, embedding_0 … embedding_N

        Args:
            dataset: Batched tf.data.Dataset (images, labels). repeat=False.
            df:      DataFrame slice corresponding to the dataset rows.
                     Must contain config.data.id_col, config.data.image_col,
                     and config.data.target_col.
            steps:   Number of batches to consume from dataset.
            config:  PipelineConfig for column names.

        Returns:
            DataFrame with id, path, label, and embedding columns.
        """
        id_col     = config.data.id_col
        image_col  = config.data.image_col
        target_col = config.data.target_col

        log.info(
            "Extracting embeddings from layer '%s' (%d samples, %d steps)...",
            self.embedding_layer_name, len(df), steps,
        )

        # ── Collect embeddings and true labels in one pass ────────────────
        emb_list:    list[np.ndarray] = []
        y_true_list: list[np.ndarray] = []

        for x_batch, y_batch in dataset.take(steps):
            batch_emb = self._intermediate_model(x_batch, training=False)
            emb_list.append(batch_emb.numpy())
            y_true_list.append(y_batch.numpy())

        if not emb_list:
            raise RuntimeError("No batches were produced by the dataset.")

        embeddings = np.concatenate(emb_list, axis=0)
        y_true     = np.concatenate(y_true_list, axis=0).flatten()

        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(len(embeddings), -1)

        # Trim to actual dataset length (last batch may be padded)
        n = len(df)
        embeddings = embeddings[:n]
        y_true     = y_true[:n]

        log.info(
            "Embeddings extracted: shape=%s  (layer='%s')",
            embeddings.shape, self.embedding_layer_name,
        )

        # ── Build output DataFrame ─────────────────────────────────────────
        emb_cols = [f"embedding_{i}" for i in range(embeddings.shape[1])]
        emb_df   = pd.DataFrame(embeddings, columns=emb_cols)

        # Prepend metadata columns
        if id_col in df.columns:
            emb_df.insert(0, id_col, df[id_col].values[:n])
        if image_col in df.columns:
            emb_df.insert(1 if id_col in df.columns else 0,
                          image_col, df[image_col].values[:n])

        emb_df[target_col] = df[target_col].values[:n]
        emb_df["y_true"]   = y_true.astype(int)
        emb_df["source_layer"] = self.embedding_layer_name

        return emb_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_layer(model: tf.keras.Model, substr: str) -> str:
        """
        Return the name of the first layer whose name contains `substr`.

        Raises ValueError with a helpful message listing all layer names
        if no match is found.
        """
        for layer in model.layers:
            if substr in layer.name:
                return layer.name

        all_names = [l.name for l in model.layers]
        raise ValueError(
            f"No layer with name containing '{substr}' found in model.\n"
            f"Available layers:\n" + "\n".join(f"  {n}" for n in all_names)
        )

    def _build_intermediate_model(self) -> tf.keras.Model:
        """Build a model that outputs the embedding layer's activations."""
        return tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(self.embedding_layer_name).output,
        )
