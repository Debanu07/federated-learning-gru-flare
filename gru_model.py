
"""Shared GRU forecasting model and weight serialization."""

import numpy as np
import tensorflow as tf

LOOKBACK = 12
HORIZON = 6
N_FEATURES = 5
GRU_UNITS = 32


def build_model():
    """Build a multivariate sensor forecasting GRU."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(LOOKBACK, N_FEATURES)
            ),
            tf.keras.layers.GRU(GRU_UNITS),
            tf.keras.layers.Dense(HORIZON * N_FEATURES),
            tf.keras.layers.Reshape(
                (HORIZON, N_FEATURES)
            ),
        ],
        name="duo_sensor_gru_forecaster",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def pack_weights(weights):
    """Flatten model weights into one NumPy vector for FLARE."""

    return np.concatenate(
        [
            np.asarray(weight, dtype=np.float32).ravel()
            for weight in weights
        ]
    ).astype(np.float32)


def unpack_weights(flat_weights, model):
    """Restore a flat vector to the model's weight-array shapes."""

    flat_weights = np.asarray(
        flat_weights, dtype=np.float32
    ).ravel()

    restored = []
    offset = 0

    for weight in model.get_weights():
        size = weight.size
        end = offset + size

        if end > flat_weights.size:
            raise ValueError(
                "Received GRU parameter vector is too short."
            )

        restored.append(
            flat_weights[offset:end].reshape(weight.shape)
        )
        offset = end

    if offset != flat_weights.size:
        raise ValueError(
            f"GRU parameter mismatch: consumed {offset}, "
            f"received {flat_weights.size}."
        )

    return restored
