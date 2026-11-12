
"""Evaluate the saved NVIDIA FLARE global model on held-out test data."""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import client_ahm_flare as client


MODEL_PATH = Path(
    "/tmp/nvflare/simulation/"
    "duo-sensor-temperature-vibration-forecast/"
    "server/simulate_job/models/server.npy"
)

OUTPUT_DIR = Path("/tmp/flare_evaluation_results")
PLOTS_DIR = Path("/tmp/flare_evaluation_plots")


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)

    errors = predicted - actual
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))

    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = (
        float(1.0 - np.sum(errors ** 2) / denominator)
        if denominator > 0
        else float("nan")
    )

    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mae,
        "R2": r2,
    }


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Global model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    params = np.load(MODEL_PATH)
    W, b = client.unpack_params(params)

    if not np.isfinite(params).all():
        raise ValueError("Model contains NaN or infinite parameters.")

    print("=" * 68)
    print("HELD-OUT TEST EVALUATION — NVIDIA FLARE GLOBAL MODEL")
    print("=" * 68)
    print(f"Model parameters: {params.size}")
    print(f"Lookback: {client.LOOKBACK} readings")
    print(f"Forecast horizon: {client.HORIZON} readings")
    print(f"Training fraction: {client.TRAIN_FRACTION:.0%}")

    metric_rows = []
    prediction_rows = []

    for site_key in client.SITE_FILES:
        print(f"\n{'-' * 68}\nEvaluating {site_key}\n{'-' * 68}")

        _, _, X_test, Y_test, columns = client.load_data(site_key)

        n_windows = len(X_test)
        horizon = client.HORIZON
        n_features = client.N_FEATURES

        predicted = (X_test @ W + b).reshape(
            n_windows, horizon, n_features
        ) * client.SCALES.reshape(1, 1, -1)

        actual = Y_test.reshape(
            n_windows, horizon, n_features
        ) * client.SCALES.reshape(1, 1, -1)

        print(f"Test windows: {n_windows:,}")
        print(f"Target readings: {n_windows * horizon:,}")

        for feature_idx, sensor_name in enumerate(columns):
            y_true = actual[:, :, feature_idx].reshape(-1)
            y_pred = predicted[:, :, feature_idx].reshape(-1)

            scores = calculate_metrics(y_true, y_pred)

            metric_rows.append({
                "site": site_key,
                "sensor": sensor_name,
                "MSE": scores["MSE"],
                "RMSE": scores["RMSE"],
                "MAE": scores["MAE"],
                "R2": scores["R2"],
                "test_windows": n_windows,
                "test_readings": len(y_true),
            })

            for window_idx in range(n_windows):
                for horizon_idx in range(horizon):
                    prediction_rows.append({
                        "site": site_key,
                        "sensor": sensor_name,
                        "test_window": window_idx + 1,
                        "horizon_step": horizon_idx + 1,
                        "test_target_index": (
                            window_idx * horizon + horizon_idx
                        ),
                        "actual": float(
                            actual[window_idx, horizon_idx, feature_idx]
                        ),
                        "predicted": float(
                            predicted[window_idx, horizon_idx, feature_idx]
                        ),
                    })

            plot_path = PLOTS_DIR / f"{site_key}_{sensor_name}.png"
            x_axis = np.arange(len(y_true))

            plt.figure(figsize=(13, 5))
            plt.plot(x_axis, y_true, label="Actual", linewidth=1.5)
            plt.plot(x_axis, y_pred, label="FL global model", linewidth=1.2)
            plt.title(f"{site_key}: {sensor_name} — Held-out test")
            plt.xlabel("Test target position (5-minute resampled row order)")
            plt.ylabel(sensor_name)
            plt.legend()
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            try:
                plt.savefig(plot_path, dpi=150)
                print(f"Saved plot: {plot_path}")
            finally:
                 plt.close()

            print(
                f"{sensor_name}: "
                f"MSE={scores['MSE']:.6f} | "
                f"RMSE={scores['RMSE']:.6f} | "
                f"MAE={scores['MAE']:.6f} | "
                f"R2={scores['R2']:.6f}"
            )

    metrics_path = OUTPUT_DIR / "metrics.csv"
    predictions_path = OUTPUT_DIR / "predictions.csv"

    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    print(f"\n{'=' * 68}")
    print("EVALUATION COMPLETE")
    print(f"Metrics:     {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Plots:       {PLOTS_DIR}")
    print(f"Metric rows: {len(metric_rows)}")
    print(f"Prediction rows: {len(prediction_rows)}")
    print("=" * 68)


if __name__ == "__main__":
    main()

