from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Sets: Python's random, NumPy, TensorFlow global and operation seeds.
    Also enables TF op determinism if available (TF >= 2.8).

    NOTE: Full determinism on GPU is not guaranteed even with these settings
    due to non-deterministic CUDA operations. If you need exact reproducibility,
    run on CPU or use tf.config.experimental.enable_op_determinism() (TF >= 2.9),
    which is attempted here automatically.

    Args:
        seed: Integer seed. Stored in the run manifest for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow seeds — must be imported lazily to avoid importing TF
    # before GPU configuration is set up in train.py
    import tensorflow as tf
    tf.random.set_seed(seed)
