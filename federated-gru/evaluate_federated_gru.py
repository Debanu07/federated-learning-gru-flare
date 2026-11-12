import os
from pathlib import Path

import numpy as np

from gru_model import build_model, unpack_weights
from client_gru_flare import load_data, evaluate

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

MODEL_PATH = (
    "/tmp/nvflare/simulation/duo-sensor-federated-gru/"
    "server/simulate_job/models/server.npy"
)

os.environ["FL_DATA_DIR"] = str(DATA_DIR)

# Sensor site mapping
SITES = {
    "site-1": "Duo_001",
    "site-2": "Duo_008",
    "site-3": "Duo_005",
}


def main():
    print("Loading federated global model...")

    params = np.load(MODEL_PATH)
    print("Global model parameter shape:", params.shape)

    model = build_model()
    model.set_weights(unpack_weights(params, model))

    print("\nEvaluating global model on each site's test data")
    print("=" * 60)

    results = []

    for site_key, site_name in SITES.items():
        print(f"\nEvaluating {site_key} ({site_name})...")

        X_train, y_train, X_test, y_test, columns = load_data(site_key)

        metrics = evaluate(model, X_test, y_test)

        print("Test samples:", len(X_test))
        print("Metrics:", metrics)

        results.append({
            "site": site_name,
            "test_samples": len(X_test),
            **metrics,
        })

    print("\nFINAL RESULTS")
    print("=" * 60)

    for result in results:
        print(result)


if __name__ == "__main__":
    main()