from __future__ import annotations

from typing import Callable

import tensorflow as tf

from aispycore.config.schema import AugmentationConfig


# ---------------------------------------------------------------------------
# Transform implementations
# ---------------------------------------------------------------------------

def random_flip(axes: list[int] = (0, 1)) -> Callable:
    def _apply(volume: tf.Tensor) -> tf.Tensor:
        for axis in axes:
            volume = tf.cond(
                tf.random.uniform(()) > 0.5,
                lambda v=volume, a=axis: tf.reverse(v, axis=[a]),
                lambda v=volume: v,
            )
        return volume
    return _apply


def random_rotation(max_angle: float = 10.0) -> Callable:
    """
    Placeholder: apply a random rotation within [-max_angle, max_angle] degrees.
    Replace with a proper 3D rotation implementation (e.g. via tfa or custom op).
    """
    def _apply(volume: tf.Tensor) -> tf.Tensor:
        # TODO: implement 3D rotation
        return volume
    return _apply


def gaussian_noise(std: float = 0.01) -> Callable:
    def _apply(volume: tf.Tensor) -> tf.Tensor:
        noise = tf.random.normal(shape=tf.shape(volume), stddev=std)
        return volume + noise
    return _apply


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRANSFORM_REGISTRY: dict[str, Callable[..., Callable]] = {
    "random_flip": random_flip,
    "random_rotation": random_rotation,
    "gaussian_noise": gaussian_noise,
}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """
    A callable that applies a sequence of transforms to a single volume tensor.
    Passed directly into make_dataset_from_df.
    """

    def __init__(self, transforms: list[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, volume: tf.Tensor) -> tf.Tensor:
        for transform in self.transforms:
            volume = transform(volume)
        return volume

    @classmethod
    def from_config(cls, config: AugmentationConfig) -> AugmentationPipeline | None:
        """
        Build a pipeline from config. Returns None if augmentation is disabled,
        so callers can pass the result directly without extra checks.
        """
        if not config.enabled or not config.transforms:
            return None

        transforms = []
        for entry in config.transforms:
            name = entry["name"]
            params = entry.get("params", {})
            if name not in TRANSFORM_REGISTRY:
                available = ", ".join(TRANSFORM_REGISTRY.keys())
                raise ValueError(
                    f"Unknown augmentation '{name}'. Available: {available}"
                )
            transforms.append(TRANSFORM_REGISTRY[name](**params))

        return cls(transforms)
