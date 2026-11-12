import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from gru_model import build_model, unpack_weights
from client_gru_flare import load_data, SCALES, SITE_FILES

# =========================================================
# PATHS / CONFIG
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
STANDALONE_DIR = ROOT_DIR / "gru_vibration_results"

os.environ["FL_DATA_DIR"] = str(DATA_DIR)

FEDERATED_MODEL_PATH = Path(
    "/tmp/nvflare/simulation/duo-sensor-federated-gru/"
    "server/simulate_job/models/server.npy"
)

SITES = {
    "site-1": "Duo_001",
    "site-2": "Duo_008",
    "site-3": "Duo_005",
}

VIBRATION_SENSORS = ["vibration_x", "vibration_y", "vibration_z"]
LOOKBACK = 12
HORIZON = 6
TRAIN_FRACTION = 0.80


# =========================================================
# STANDALONE PREPROCESSING (MATCH train_gru_vibration.py)
# =========================================================

def load_standalone_series(csv_path, sensor):
    df = pd.read_csv(csv_path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )
    df[sensor] = pd.to_numeric(df[sensor], errors="coerce")

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    # Matches standalone code: keep first duplicate
    df = df.drop_duplicates(subset=["timestamp"])

    series = (
        df.set_index("timestamp")[sensor]
        .resample("5min")
        .mean()
        .interpolate(
            method="time",
            limit=2,
            limit_area="inside",
        )
        .dropna()
        .to_numpy(dtype=np.float32)
    )

    return series


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(actual, predicted):
    actual = actual.reshape(-1)
    predicted = predicted.reshape(-1)

    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


# =========================================================
# MAIN COMPARISON
# =========================================================

def main():
    if not FEDERATED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Federated model not found: {FEDERATED_MODEL_PATH}"
        )

    print("Loading federated global model...")
    global_weights = np.load(FEDERATED_MODEL_PATH)

    federated_model = build_model()
    federated_model.set_weights(
        unpack_weights(global_weights, federated_model)
    )

    results = []

    for site_key, site_name in SITES.items():
        print(f"\n{'=' * 65}")
        print(f"Evaluating {site_name}")
        print("=" * 65)

        # FL test windows: chronological, non-overlapping, 6-step blocks
        X_train, y_train, X_test, y_test, columns = load_data(site_key)

        print("FL test windows:", len(X_test))
        print("Feature order:", columns)

        # Federated predictions in original units
        fed_pred = federated_model.predict(X_test, verbose=0)
        fed_pred = fed_pred * SCALES.reshape(1, 1, -1)

        fed_actual = y_test * SCALES.reshape(1, 1, -1)

        # Locate vibration channels in FL feature order
        for sensor in VIBRATION_SENSORS:
            feature_index = columns.index(sensor)

            actual = fed_actual[:, :, feature_index]

            # ---------------------------------------------
            # Federated model metrics
            # ---------------------------------------------
            federated_prediction = fed_pred[:, :, feature_index]

            fed_metrics = calculate_metrics(
                actual, federated_prediction
            )

            # ---------------------------------------------
            # Standalone model
            # ---------------------------------------------
            standalone_model_path = (
                STANDALONE_DIR / f"{site_name}_{sensor}.keras"
            )

            if not standalone_model_path.exists():
                raise FileNotFoundError(
                    f"Standalone model not found: {standalone_model_path}"
                )

            standalone_model = tf.keras.models.load_model(
                standalone_model_path
            )

            csv_path = DATA_DIR / SITE_FILES[site_key]

            # Reproduce standalone sensor-specific normalization
            raw_series = load_standalone_series(csv_path, sensor)

            split = int(len(raw_series) * TRAIN_FRACTION)
            train_raw = raw_series[:split]

            mean = float(train_raw.mean())
            std = float(train_raw.std())

            if std < 1e-8:
                std = 1.0

            # FL test windows are used to ensure same forecast origins.
            # Vibration channels have scale factor 1 in FL preprocessing.
            standalone_input_raw = X_test[:, :, feature_index]

            standalone_input_scaled = (
                standalone_input_raw - mean
            ) / std

            standalone_input_scaled = (
                standalone_input_scaled[..., np.newaxis]
            ).astype(np.float32)

            standalone_prediction_scaled = (
                standalone_model.predict(
                    standalone_input_scaled, verbose=0
                )
            )

            standalone_prediction = (
                standalone_prediction_scaled * std + mean
            )

            standalone_metrics = calculate_metrics(
                actual, standalone_prediction
            )

            # ---------------------------------------------
            # Comparison
            # Positive improvement = FL error lower than standalone
            # ---------------------------------------------
            rmse_improvement = (
                (standalone_metrics["rmse"] - fed_metrics["rmse"])
                / standalone_metrics["rmse"]
                * 100
            )

            mae_improvement = (
                (standalone_metrics["mae"] - fed_metrics["mae"])
                / standalone_metrics["mae"]
                * 100
            )

            result = {
                "site": site_name,
                "sensor": sensor,
                "test_windows": len(X_test),
                "test_points": actual.size,

                "standalone_rmse": standalone_metrics["rmse"],
                "standalone_mae": standalone_metrics["mae"],
                "standalone_r2": standalone_metrics["r2"],

                "federated_rmse": fed_metrics["rmse"],
                "federated_mae": fed_metrics["mae"],
                "federated_r2": fed_metrics["r2"],

                "rmse_improvement_percent": rmse_improvement,
                "mae_improvement_percent": mae_improvement,
            }

            results.append(result)

            print(f"\nSensor: {sensor}")
            print(
                f"Standalone  RMSE: {standalone_metrics['rmse']:.6f} | "
                f"MAE: {standalone_metrics['mae']:.6f} | "
                f"R²: {standalone_metrics['r2']:.6f}"
            )
            print(
                f"Federated   RMSE: {fed_metrics['rmse']:.6f} | "
                f"MAE: {fed_metrics['mae']:.6f} | "
                f"R²: {fed_metrics['r2']:.6f}"
            )
            print(
                f"RMSE improvement: {rmse_improvement:+.2f}%"
            )
            print(
                f"MAE improvement:  {mae_improvement:+.2f}%"
            )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results_df = pd.DataFrame(results)

    output_path = PROJECT_DIR / "gru_model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    print(
        results_df[
            [
                "site",
                "sensor",
                "standalone_rmse",
                "federated_rmse",
                "rmse_improvement_percent",
                "standalone_mae",
                "federated_mae",
                "mae_improvement_percent",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved comparison to: {output_path}")


if __name__ == "__main__":
    main()
