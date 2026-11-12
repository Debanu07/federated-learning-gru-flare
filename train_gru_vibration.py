
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==================================================
# CONFIGURATION
# ==================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("gru_vibration_results")

TIMESTAMP_COL = "timestamp"
VIBRATION_COLS = ["vibration_x", "vibration_y", "vibration_z"]

INTERVAL = "5min"
SEQUENCE_LENGTH = 12
HORIZON = 6
TRAIN_FRACTION = 0.80

EPOCHS = 200
BATCH_SIZE = 32
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))


# ==================================================
# LOAD AND PREPROCESS DATA
# ==================================================
def load_series(csv_path, sensor):
    df = pd.read_csv(csv_path)

    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL], errors="coerce"
    )
    df[sensor] = pd.to_numeric(df[sensor], errors="coerce")

    df = df.dropna(subset=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL)
    df = df.drop_duplicates(subset=[TIMESTAMP_COL])

    series = (
        df.set_index(TIMESTAMP_COL)[sensor]
        .resample(INTERVAL)
        .mean()
        .interpolate(
            method="time",
            limit=2,
            limit_area="inside"
        )
        .dropna()
        .to_numpy(dtype=np.float32)
    )

    return series


# ==================================================
# CREATE SEQUENCES
# ==================================================
def make_windows(series):
    X, y = [], []

    for i in range(
        len(series) - SEQUENCE_LENGTH - HORIZON + 1
    ):
        X.append(series[i:i + SEQUENCE_LENGTH])

        y.append(
            series[
                i + SEQUENCE_LENGTH:
                i + SEQUENCE_LENGTH + HORIZON
            ]
        )

    X = np.asarray(X, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(y, dtype=np.float32)

    return X, y


# ==================================================
# BUILD GRU MODEL
# ==================================================
def build_model():
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 1)),

        GRU(64, return_sequences=True),
        Dropout(0.2),

        GRU(32),
        Dropout(0.1),

        Dense(16, activation="relu"),
        Dense(HORIZON, activation="linear"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    return model


# ==================================================
# TRAIN ONE SITE / ONE SENSOR
# ==================================================
def train_one(csv_path, sensor, site_name):
    print(f"\n{'=' * 60}")
    print(f"Site: {site_name} | Sensor: {sensor}")
    print(f"{'=' * 60}")

    series = load_series(csv_path, sensor)

    if len(series) < 100:
        print("Insufficient data. Skipping.")
        return None

    # Chronological split before creating windows.
    split = int(len(series) * TRAIN_FRACTION)

    train_raw = series[:split]
    test_raw = series[split:]

    # Normalize using training data only.
    mean = float(train_raw.mean())
    std = float(train_raw.std())

    if std < 1e-8:
        std = 1.0

    train_scaled = (train_raw - mean) / std
    test_scaled = (test_raw - mean) / std

    # Training windows cannot cross the split.
    X_train, y_train = make_windows(train_scaled)

    # Use the last training observations as test context.
    # Test targets start at the chronological split boundary.
    test_context = np.concatenate([
        train_scaled[-SEQUENCE_LENGTH:],
        test_scaled,
    ])

    X_test, y_test = make_windows(test_context)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Insufficient windows. Skipping.")
        return None

    print("Training windows:", len(X_train))
    print("Testing windows:", len(X_test))
    print("Input shape:", X_train.shape)
    print("Output shape:", y_train.shape)

    model = build_model()
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        shuffle=True,
        callbacks=[early_stop],
        verbose=1,
    )

    # Predict the six future readings.
    pred_scaled = model.predict(X_test, verbose=0)

    # Convert back to original sensor units.
    y_true = y_test * std + mean
    y_pred = pred_scaled * std + mean

    rmse = float(np.sqrt(
        mean_squared_error(y_true.ravel(), y_pred.ravel())
    ))
    mae = float(
        mean_absolute_error(y_true.ravel(), y_pred.ravel())
    )
    r2 = float(
        r2_score(y_true.ravel(), y_pred.ravel())
    )

    print(f"\nResults for {site_name} - {sensor}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R2:   {r2:.6f}")

    # Save trained model.
    model_path = OUTPUT_DIR / f"{site_name}_{sensor}.keras"
    model.save(model_path)

    # Save predictions and corresponding forecast steps.
    pd.DataFrame({
        "window_id": np.repeat(
            np.arange(len(y_true)), HORIZON
        ),
        "forecast_step": np.tile(
            np.arange(1, HORIZON + 1), len(y_true)
        ),
        "actual": y_true.ravel(),
        "predicted": y_pred.ravel(),
    }).to_csv(
        OUTPUT_DIR / f"{site_name}_{sensor}_predictions.csv",
        index=False,
    )

    return {
        "site": site_name,
        "sensor": sensor,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_windows": len(X_train),
        "test_windows": len(X_test),
        "epochs_trained": len(history.history["loss"]),
        "model_path": str(model_path),
    }


# ==================================================
# MAIN
# ==================================================
def main():
    files = sorted(DATA_DIR.glob("Duo_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No Duo_*.csv files found in {DATA_DIR.resolve()}"
        )

    available_gpus = tf.config.list_physical_devices("GPU")

    if available_gpus:
        print("\nGPU acceleration is available.")
    else:
        print("\nWARNING: No GPU detected. Training may use CPU.")

    results = []

    for csv_path in files:
        # Works with filenames such as Duo_001_...
        site_name = csv_path.name.split("_")[0] + "_" + csv_path.name.split("_")[1]

        for sensor in VIBRATION_COLS:
            try:
                result = train_one(csv_path, sensor, site_name)

                if result is not None:
                    results.append(result)

            except (KeyError, ValueError) as exc:
                print(f"Skipping {site_name} {sensor}: {exc}")

    if results:
        metrics = pd.DataFrame(results)
        metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

        print("\n" + "=" * 60)
        print("FINAL GRU BENCHMARK RESULTS")
        print("=" * 60)
        print(metrics.to_string(index=False))

        print(f"\nResults saved in: {OUTPUT_DIR.resolve()}")
    else:
        print("No models were trained successfully.")


if __name__ == "__main__":
    main()
