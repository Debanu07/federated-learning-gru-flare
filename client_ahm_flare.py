
"""NVIDIA FLARE federated multivariate time-series forecasting client."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants


# ---------------- Configuration ----------------
TIMESTAMP_COL = "timestamp"
TEMP_COLS = ["temperature_one", "temperature_two"]

INTERVAL = "5min"
LOOKBACK = 12       # Previous 12 readings = 60 minutes
HORIZON = 6         # Predict next 6 readings = 30 minutes

TRAIN_FRACTION = 0.80
LOCAL_EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 1024

# Fixed scales must be identical across all sites.
# Adjust vibration scales after inspecting the actual data ranges.
SCALES = np.array([100.0, 100.0, 1.0, 1.0, 1.0],
                  dtype=np.float32)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR = Path(os.environ.get("FL_DATA_DIR", str(DEFAULT_DATA_DIR)))

SITE_FILES = {
    "site-1": "Duo_001_24-6-25_to_31-7-25 - Duo_001_24-6-25_to_31-7-25.csv",
    "site-2": "Duo_008_24-6-25_to_29-7-25 - Duo_008_24-6-25_to_29-7-25.csv",
    "site-3": "Duo_005_24-6-25_to_05-8-25 - Duo_005_24-6-25_to_05-8-25.csv",
}

N_FEATURES = 5
N_INPUTS = LOOKBACK * N_FEATURES
N_OUTPUTS = HORIZON * N_FEATURES
PARAM_COUNT = N_INPUTS * N_OUTPUTS + N_OUTPUTS


def identify_site(site_name):
    for key in SITE_FILES:
        if key in site_name.lower():
            return key
    raise ValueError(f"Unrecognized FLARE site: {site_name}")


def load_data(site_key):
    explicit_file = os.environ.get("FL_DATA_FILE")
    csv_path = (
        Path(explicit_file)
        if explicit_file
        else DATA_DIR / SITE_FILES[site_key]
    )

    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    print(f"[{site_key}] Loading {csv_path}")

    df = pd.read_csv(csv_path)

    required = [TIMESTAMP_COL] + TEMP_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Detect vibration columns rather than guessing their names.
    vibration_cols = [
        c for c in df.columns
        if "vibration" in c.lower()
    ]

    if len(vibration_cols) != 3:
        raise ValueError(
            "Expected exactly 3 vibration columns, but found "
            f"{vibration_cols}. Update vibration_cols detection "
            "to match the actual CSV headers."
        )

    columns = TEMP_COLS + vibration_cols
    print(f"[{site_key}] Forecast columns: {columns}")

    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL], errors="coerce"
    )

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL)
    df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
    df = df.set_index(TIMESTAMP_COL)

    # Convert readings to equal 5-minute intervals.
    df = df[columns].resample(INTERVAL).mean()

    # Fill only short gaps. Long gaps remain missing and are dropped.
    df[columns] = df[columns].interpolate(
        method="time", limit=2, limit_area="inside"
    )
    df = df.dropna(subset=columns)

    if len(df) < LOOKBACK + HORIZON + 10:
        raise ValueError("Not enough valid data after resampling.")

    # The chronological split happens BEFORE creating windows.
    values = df[columns].to_numpy(dtype=np.float32)
    split = int(len(values) * TRAIN_FRACTION)

    if split <= LOOKBACK:
        raise ValueError("Training partition is too short.")

    # Fixed scaling across sites preserves shared parameter meaning.
    scaled = values / SCALES

    # Training windows: inputs and targets must stay in the first 80%.
    X_train, Y_train = make_windows(
        scaled, 0, split, training=True
    )

    # Test windows: targets start in the final 20%.
    # The first test forecast can use the last LOOKBACK train readings.
    # Later forecasts use newly observed readings in rolling-origin mode.
    X_test, Y_test = make_windows(
        scaled, split - LOOKBACK, len(scaled), training=False
    )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("No training or testing windows were created.")

    print(f"[{site_key}] Resampled rows: {len(values):,}")
    print(f"[{site_key}] Training windows: {len(X_train):,}")
    print(f"[{site_key}] Testing windows: {len(X_test):,}")
    print(f"[{site_key}] Vibration columns: {vibration_cols}")

    return X_train, Y_train, X_test, Y_test, columns


def make_windows(values, start, end, training):
    """
    X: previous LOOKBACK readings.
    Y: next HORIZON readings.
    For testing, forecast origins advance by HORIZON to avoid
    counting the same target timestamp repeatedly.
    """
    X, Y = [], []

    first_origin = start + LOOKBACK
    last_origin = end - HORIZON

    if training:
        origins = range(first_origin, last_origin + 1)
    else:
        # The first target timestamp is the start of the test partition.
        origins = range(first_origin, last_origin + 1, HORIZON)

    for origin in origins:
        target_start = origin
        target_end = origin + HORIZON

        if training and target_end > end:
            continue

        if not training and target_start < int(
            # Split boundary is supplied through start + LOOKBACK.
            start + LOOKBACK
        ):
            continue

        X.append(values[origin - LOOKBACK:origin].reshape(-1))
        Y.append(values[target_start:target_end].reshape(-1))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32),
    )


def unpack_params(params):
    params = np.asarray(params, dtype=np.float32).reshape(-1)

    if params.size != PARAM_COUNT:
        raise ValueError(
            f"Expected {PARAM_COUNT} model parameters, got {params.size}. "
            "Update job.py to initialize the global model with this size."
        )

    split = N_INPUTS * N_OUTPUTS
    W = params[:split].reshape(N_INPUTS, N_OUTPUTS)
    b = params[split:]
    return W, b


def pack_params(W, b):
    return np.concatenate([W.reshape(-1), b]).astype(np.float32)


def train_local(global_params, X, Y):
    W, b = unpack_params(global_params)

    rng = np.random.default_rng(42)

    for epoch in range(LOCAL_EPOCHS):
        order = rng.permutation(len(X))

        for offset in range(0, len(order), BATCH_SIZE):
            ids = order[offset:offset + BATCH_SIZE]
            xb = X[ids]
            yb = Y[ids]

            # Multi-output linear regression.
            predictions = xb @ W + b
            errors = predictions - yb

            grad_W = (2.0 / len(ids)) * (xb.T @ errors)
            grad_b = 2.0 * errors.mean(axis=0)

            W -= LEARNING_RATE * grad_W
            b -= LEARNING_RATE * grad_b

        mse = float(np.mean((X @ W + b - Y) ** 2))
        print(
            f"  Epoch {epoch + 1}/{LOCAL_EPOCHS}"
            f" | normalized train MSE: {mse:.7f}"
        )

    return pack_params(W, b)


def evaluate(params, X, Y):
    W, b = unpack_params(params)

    # Convert predictions and targets back to original sensor units.
    pred_scaled = X @ W + b
    predictions = pred_scaled.reshape(-1, HORIZON, N_FEATURES)
    actual = Y.reshape(-1, HORIZON, N_FEATURES)

    predictions = predictions * SCALES
    actual = actual * SCALES

    results = {}

    for i, name in enumerate(
        ["temperature_one", "temperature_two",
         "vibration_1", "vibration_2", "vibration_3"]
    ):
        y_true = actual[:, :, i].reshape(-1)
        y_pred = predictions[:, :, i].reshape(-1)
        errors = y_pred - y_true

        mse = float(np.mean(errors ** 2))
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(mse))

        denominator = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = (
            float(1.0 - np.sum(errors ** 2) / denominator)
            if denominator > 0 else float("nan")
        )

        results[f"{name}_mse"] = mse
        results[f"{name}_rmse"] = rmse
        results[f"{name}_mae"] = mae
        results[f"{name}_r2"] = r2

    return results


def main():
    flare.init()
    site_name = flare.get_site_name()
    site_key = identify_site(site_name)

    print(f"\n[{site_name}] Starting federated forecasting client")

    X_train, Y_train, X_test, Y_test, columns = load_data(site_key)

    while flare.is_running():
        input_model = flare.receive()

        if input_model is None:
            break

        if not isinstance(input_model.params, dict):
            raise RuntimeError("Expected a NumPy model parameter dictionary.")

        if NPConstants.NUMPY_KEY not in input_model.params:
            raise RuntimeError("Received model is missing NumPy parameters.")

        global_params = input_model.params[NPConstants.NUMPY_KEY]

        updated_params = train_local(
            global_params, X_train, Y_train
        )

        # These are local validation-like diagnostics during training.
        # The final held-out test must be run after training is complete.
        metrics = evaluate(updated_params, X_train, Y_train)

        print(f"[{site_name}] Local training metrics: {metrics}")

        output_model = flare.FLModel(
            params={NPConstants.NUMPY_KEY: updated_params},
            metrics=metrics,
            meta={
                "NUM_STEPS_CURRENT_ROUND": int(len(X_train))
            },
        )

        flare.send(output_model)
        print(
            f"[{site_name}] Submitted model for round "
            f"{input_model.current_round}"
        )

    print(f"[{site_name}] Federated training finished.")


if __name__ == "__main__":
    main()
