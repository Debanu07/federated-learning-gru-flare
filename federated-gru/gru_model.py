
"""Shared GRU model for federated multivariate sensor forecasting."""

import numpy as np
import tensorflow as tf
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print(f"Could not enable GPU memory growth: {exc}", flush=True)

# Forecasting configuration
LOOKBACK = 12
HORIZON = 6
N_FEATURES = 5
GRU_UNITS = 32


def build_model():
    """Create and compile the GRU forecasting model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LOOKBACK, N_FEATURES)),
            tf.keras.layers.GRU(GRU_UNITS),
            tf.keras.layers.Dense(HORIZON * N_FEATURES),
            tf.keras.layers.Reshape((HORIZON, N_FEATURES)),
        ],
        name="duo_sensor_gru_forecaster",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    return model


def pack_weights(weights):
    """Flatten Keras model weights into one vector for NVIDIA FLARE."""
    return np.concatenate(
        [np.asarray(w, dtype=np.float32).ravel() for w in weights]
    ).astype(np.float32)


def unpack_weights(flat_weights, model):
    """Restore flattened federated weights to Keras weight shapes."""
    flat_weights = np.asarray(flat_weights, dtype=np.float32).ravel()

    restored = []
    offset = 0

    for current_weight in model.get_weights():
        size = current_weight.size
        end = offset + size

        if end > flat_weights.size:
            raise ValueError("GRU parameter vector is too short.")

        restored.append(
            flat_weights[offset:end].reshape(current_weight.shape)
        )
        offset = end

    if offset != flat_weights.size:
        raise ValueError(
            f"GRU parameter mismatch: expected {offset}, "
            f"received {flat_weights.size}."
        )

    return restored
