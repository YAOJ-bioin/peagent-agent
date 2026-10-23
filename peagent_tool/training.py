"""Experimental training utilities.

The v1 package is inference-first. These helpers expose only small reusable
TensorFlow operations used by PEAgent training scripts; full preprocessing and
training pipelines remain outside the stable public API.
"""

from __future__ import annotations


def configure_tensorflow_memory_growth() -> None:
    """Enable TensorFlow GPU memory growth when TensorFlow is installed."""

    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def final_dense_layer(model):
    """Return the final Dense layer from a Keras model."""

    import tensorflow as tf

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            return layer
    raise ValueError("No Dense layers found in model.")


def set_trainable_final_only(model) -> None:
    """Freeze all layers except the final Dense layer."""

    final_layer = final_dense_layer(model)
    for layer in model.layers:
        layer.trainable = layer is final_layer


def set_trainable_all(model) -> None:
    """Mark all model layers trainable."""

    for layer in model.layers:
        layer.trainable = True


__all__ = [
    "configure_tensorflow_memory_growth",
    "final_dense_layer",
    "set_trainable_all",
    "set_trainable_final_only",
]
