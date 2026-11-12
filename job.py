
"""NVIDIA FLARE job for federated multivariate forecasting."""

import numpy as np

from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
from nvflare.recipe import SimEnv
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

os.environ["FL_DATA_DIR"] = str(DATA_DIR)


# Forecasting model configuration
LOOKBACK = 12
HORIZON = 6
N_FEATURES = 5
NUM_ROUNDS = 30
NUM_CLIENTS = 3

# Each input window contains 12 readings x 5 signals.
n_inputs = LOOKBACK * N_FEATURES

# Each output contains 6 future readings x 5 signals.
n_outputs = HORIZON * N_FEATURES

# Linear model parameters: weight matrix plus bias vector.
n_parameters = n_inputs * n_outputs + n_outputs

initial_params = np.zeros(n_parameters, dtype=np.float32)

print("Initial global model parameters:", initial_params.shape)
assert initial_params.shape == (1830,)


def main():
    recipe = NumpyFedAvgRecipe(
        name="duo-sensor-temperature-vibration-forecast",
        model=initial_params.tolist(),
        min_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        train_script="client_ahm_flare.py",
    )

    env = SimEnv(num_clients=NUM_CLIENTS)

    print("Starting NVIDIA FLARE federated simulation...")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Federated rounds: {NUM_ROUNDS}")
    print(f"Historical readings: {LOOKBACK}")
    print(f"Future readings: {HORIZON}")
    print(f"Forecast signals: {N_FEATURES}")

    run = recipe.execute(env)

    print("Federated simulation finished.")
    print("Simulation output: /tmp/nvflare")


if __name__ == "__main__":
    main()
