"""NVIDIA FLARE client for local GRU sensor forecasting."""

import os
from pathlib import Path
from nvflare.client import FLModel
import numpy as np
import pandas as pd
import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants

from gru_model import (
    LOOKBACK,
    HORIZON,
    N_FEATURES,
    build_model,
    pack_weights,
    unpack_weights,
)

TIMESTAMP_COL = "timestamp"
TEMP_COLS = ["temperature_one", "temperature_two"]
INTERVAL = "5min"
TRAIN_FRACTION = 0.80
LOCAL_EPOCHS = 100
BATCH_SIZE = 64
SCALES = np.array([100.0, 100.0, 1.0, 1.0, 1.0], dtype=np.float32)

DATA_DIR = Path(
    os.environ.get("FL_DATA_DIR", str(Path(__file__).resolve().parent / "data"))
)

SITE_FILES = {
    "site-1": "Duo_001_24-6-25_to_31-7-25 - Duo_001_24-6-25_to_31-7-25.csv",
    "site-2": "Duo_008_24-6-25_to_29-7-25 - Duo_008_24-6-25_to_29-7-25.csv",
    "site-3": "Duo_005_24-6-25_to_05-8-25 - Duo_005_24-6-25_to_05-8-25.csv",
}


def identify_site(site_name):
    """Map the FLARE client identity to its local CSV."""
    normalized = site_name.lower()

    for site_key in SITE_FILES:
        if site_key in normalized:
            return site_key

    raise ValueError(
        f"Cannot map FLARE site {site_name!r} to a dataset. "
        f"Expected one of: {list(SITE_FILES)}"
    )


def make_windows(values, start, end, training):
    """Create (12, 5) input sequences and (6, 5) forecast targets."""
    X, Y = [], []

    if training:
        origins = range(LOOKBACK, end - HORIZON + 1)
    else:
        # Start each test forecast at the test split boundary, then
        # advance by HORIZON to avoid counting target timestamps twice.
        origins = range(start, end - HORIZON + 1, HORIZON)

    for origin in origins:
        if origin - LOOKBACK < 0:
            continue

        if training and origin + HORIZON > end:
            continue

        X.append(values[origin - LOOKBACK:origin])
        Y.append(values[origin:origin + HORIZON])

    return (
        np.asarray(X, dtype=np.float32).reshape(-1, LOOKBACK, N_FEATURES),
        np.asarray(Y, dtype=np.float32).reshape(-1, HORIZON, N_FEATURES),
    )


def load_data(site_key):
    """Load and preprocess one site's sensor data locally."""
    explicit_file = os.environ.get("FL_DATA_FILE")
    csv_path = Path(explicit_file) if explicit_file else DATA_DIR / SITE_FILES[site_key]

    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    print(f"[{site_key}] Loading {csv_path}")
    df = pd.read_csv(csv_path)

    required = [TIMESTAMP_COL] + TEMP_COLS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    vibration_cols = [
        column for column in df.columns if "vibration" in column.lower()
    ]
    if len(vibration_cols) != 3:
        raise ValueError(
            f"Expected 3 vibration columns, found: {vibration_cols}"
        )

    columns = TEMP_COLS + vibration_cols

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL)
    df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
    df = df.set_index(TIMESTAMP_COL)

    df = df[columns].resample(INTERVAL).mean()
    df[columns] = df[columns].interpolate(
        method="time", limit=2, limit_area="inside"
    )
    df = df.dropna(subset=columns)

    values = df.to_numpy(dtype=np.float32)

    if len(values) < LOOKBACK + HORIZON + 10:
        raise ValueError(
            f"[{site_key}] Not enough valid readings after preprocessing: "
            f"{len(values)}"
        )

    split = int(len(values) * TRAIN_FRACTION)
    if split < LOOKBACK + HORIZON:
        raise ValueError(f"[{site_key}] Training partition is too short.")

    # Use the same fixed feature scales at all sites.
    scaled = values / SCALES

    X_train, Y_train = make_windows(
        scaled, LOOKBACK, split, training=True
    )
    X_test, Y_test = make_windows(
        scaled, split, len(scaled), training=False
    )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"[{site_key}] Empty windows: train={len(X_train)}, "
            f"test={len(X_test)}"
        )

    print(f"[{site_key}] Features: {columns}")
    print(f"[{site_key}] Resampled rows: {len(values):,}")
    print(f"[{site_key}] Training sequences: {len(X_train):,}")
    print(f"[{site_key}] Testing sequences: {len(X_test):,}")

    return X_train, Y_train, X_test, Y_test, columns


def evaluate(model, X_test, Y_test):
    """Evaluate forecasts in the original sensor units."""
    predictions = model.predict(X_test, verbose=0)

    predictions = predictions * SCALES.reshape(1, 1, -1)
    actual = Y_test * SCALES.reshape(1, 1, -1)

    errors = predictions - actual
    mse = float(np.mean(np.square(errors)))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))

    return {"test_mse": mse, "test_mae": mae, "test_rmse": rmse}

def main():
    flare.init()

    site_name = flare.get_site_name()
    site_key = identify_site(site_name)

    # Load local data once per client.
    X_train, y_train, X_test, y_test, columns = load_data(site_key)

    # Reuse the model architecture across federated rounds.
    model = build_model()

    while True:
        # Receive the latest global model for this round.
        input_model = flare.receive()

        # FLARE can signal that there are no more tasks.
        if input_model is None:
            print(f"[{site_name}] No more tasks. Exiting.", flush=True)
            break

        print(
            f"[{site_name}] Received federated model; "
            "starting local training.",
            flush=True,
        )

        received_params = input_model.params[NPConstants.NUMPY_KEY]
        model.set_weights(unpack_weights(received_params, model))

        history = model.fit(
            X_train,
            y_train,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            verbose=2,
        )

        metrics = evaluate(model, X_test, y_test)
        metrics["train_loss"] = float(history.history["loss"][-1])

        updated_params = pack_weights(model.get_weights())

        output_model = FLModel(
            params={NPConstants.NUMPY_KEY: updated_params},
            metrics=metrics,
            meta={"NUM_STEPS_CURRENT_ROUND": len(X_train)},
        )

        flare.send(output_model)

        print(
            f"[{site_name}] Sent updated model; "
            "ready for the next round.",
            flush=True,
        )


if __name__ == "__main__":
    main()
