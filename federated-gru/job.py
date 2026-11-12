"""NVIDIA FLARE job for federated GRU sensor forecasting."""

import os
from pathlib import Path

from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
from nvflare.recipe import SimEnv

from gru_model import build_model, pack_weights

# Project configuration
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

NUM_CLIENTS = 3
NUM_ROUNDS = 30

# Ensure clients can locate their local datasets.
os.environ["FL_DATA_DIR"] = str(DATA_DIR)


def main():
    print("Initializing federated GRU job...", flush=True)
    print(f"Project directory: {PROJECT_DIR}", flush=True)
    print(f"Data directory: {DATA_DIR}", flush=True)

    # Verify that the datasets exist before launching the simulation.
    required_files = [
        "Duo_001_24-6-25_to_31-7-25 - Duo_001_24-6-25_to_31-7-25.csv",
        "Duo_008_24-6-25_to_29-7-25 - Duo_008_24-6-25_to_29-7-25.csv",
        "Duo_005_24-6-25_to_05-8-25 - Duo_005_24-6-25_to_05-8-25.csv",
    ]

    for filename in required_files:
        file_path = DATA_DIR / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        print(f"Found dataset: {filename}", flush=True)

    # Initialize the global GRU model and flatten its parameters.
    print("Building initial GRU model...", flush=True)
    model = build_model()
    initial_params = pack_weights(model.get_weights())

    print(f"Initial parameter vector: {initial_params.shape}", flush=True)
    if initial_params.shape != (4734,):
        raise ValueError(
            f"Expected 4734 GRU parameters, got {initial_params.size}"
        )

    # Configure federated averaging.
    print("Creating NVIDIA FLARE recipe...", flush=True)
    recipe = NumpyFedAvgRecipe(
        name="duo-sensor-federated-gru",
        model=initial_params.tolist(),
        min_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        train_script="client_gru_flare.py",
    )

    # Simulate three participating clients.
    env = SimEnv(num_clients=NUM_CLIENTS)

    print("Starting NVIDIA FLARE simulation.", flush=True)
    print(f"Clients: {NUM_CLIENTS}", flush=True)
    print(f"Federated rounds: {NUM_ROUNDS}", flush=True)
    print(f"Input shape: (None, 12, 5)", flush=True)
    print(f"Forecast shape: {model.output_shape}", flush=True)

    recipe.execute(env)

    print("Federated GRU simulation finished.", flush=True)


if __name__ == "__main__":
    main()
