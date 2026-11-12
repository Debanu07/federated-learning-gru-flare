
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import nvflare.client as flare

from model import SimpleNet


# ============================================================
# 1. CONFIGURATION — MATCH THE FLOWER EXPERIMENT
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 20
BATCH = 32
LR = 0.01
CLIENTS = 2

# Dataset is stored outside the virtual environment
DATA_DIR = r"D:\FL_Data"


# ============================================================
# 2. LOAD AND PARTITION MNIST DATA
# ============================================================

def load_data(client_id):
    if client_id not in range(CLIENTS):
        raise ValueError(f"Invalid client_id: {client_id}")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    testset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    # Same sequential partitioning as the Flower experiment
    partition = len(trainset) // CLIENTS

    start_index = client_id * partition
    end_index = start_index + partition

    indices = list(range(start_index, end_index))

    train_loader = DataLoader(
        Subset(trainset, indices),
        batch_size=BATCH,
        shuffle=True
    )

    test_loader = DataLoader(
        testset,
        batch_size=BATCH,
        shuffle=False
    )

    return train_loader, test_loader


# ============================================================
# 3. LOCAL TRAINING
# ============================================================

def train(model, loader, optimizer, criterion):
    model.train()

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()


# ============================================================
# 4. MODEL EVALUATION
# ============================================================

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            # Weighted loss across all test examples
            total_loss += loss.item() * batch_size

            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total += batch_size

    if total == 0:
        raise RuntimeError("The evaluation dataset is empty.")

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# 5. NVIDIA FLARE CLIENT
# ============================================================

def main():
    # Initialize NVIDIA FLARE Client API
    flare.init()

    site_name = flare.get_site_name()
    normalized_site = site_name.lower()

    print(f"\n[{site_name}] Connected to NVIDIA FLARE")
    print(f"[{site_name}] Device: {DEVICE}")

    # Map actual FLARE site names to dataset partitions
    if "site-1" in normalized_site:
        client_id = 0
    elif "site-2" in normalized_site:
        client_id = 1
    else:
        raise ValueError(
            f"Unrecognized FLARE site name: {site_name}. "
            "Update the site-to-client mapping in this file."
        )

    print(f"[{site_name}] Starting as Client {client_id}")

    # Load this client's local dataset
    train_loader, test_loader = load_data(client_id)

    print(
        f"[{site_name}] Training samples: "
        f"{len(train_loader.dataset)}"
    )
    print(
        f"[{site_name}] Test samples: "
        f"{len(test_loader.dataset)}"
    )

    # Initialize model and optimizer
    model = SimpleNet().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR
    )

    print(f"[{site_name}] Waiting for global model...")

    # ========================================================
    # 6. FEDERATED TRAINING LOOP
    # ========================================================

    while flare.is_running():

        # Receive the global model for this round
        input_model = flare.receive()

        if input_model is None:
            print(f"[{site_name}] No model received.")
            break

        round_num = input_model.current_round

        print(
            f"\n[{site_name}] "
            f"Federated round {round_num} started"
        )

        # Require server-provided global parameters
        if input_model.params is None:
            raise RuntimeError(
                "No global model parameters received. "
                "Check the FLARE server/job configuration."
            )

        # FLARE parameters are commonly NumPy arrays.
        # Convert them to tensors before loading into PyTorch.
        global_params = {
            name: torch.as_tensor(value)
            for name, value in input_model.params.items()
        }

        model.load_state_dict(global_params, strict=True)

        # ----------------------------------------------------
        # Local training: 20 epochs
        # ----------------------------------------------------

        for epoch in range(EPOCHS):
            train(
                model,
                train_loader,
                optimizer,
                criterion
            )

            loss, accuracy = evaluate(
                model,
                test_loader,
                criterion
            )

            print(
                f"  Epoch {epoch + 1}/{EPOCHS}"
                f" | Accuracy: {accuracy:.4f}"
                f" | Loss: {loss:.4f}"
            )

        # ----------------------------------------------------
        # Final evaluation for this federated round
        # ----------------------------------------------------

        loss, accuracy = evaluate(
            model,
            test_loader,
            criterion
        )

        print(
            f"\n[{site_name}] Round {round_num} results"
            f" | Accuracy: {accuracy:.4f}"
            f" | Loss: {loss:.4f}"
        )

        # ----------------------------------------------------
        # Convert trained parameters to CPU NumPy arrays
        # ----------------------------------------------------

        trained_params = {
            name: tensor.detach().cpu().numpy().copy()
            for name, tensor in model.state_dict().items()
        }

        # Number of local mini-batch optimizer steps
        steps = EPOCHS * len(train_loader)

        # Prepare updated model for the FLARE server
        output_model = flare.FLModel(
            params=trained_params,
            metrics={
                "accuracy": accuracy,
                "loss": loss
            },
            meta={
                "NUM_STEPS_CURRENT_ROUND": steps
            }
        )

        # Send locally trained model back to the server
        flare.send(output_model)

        print(
            f"[{site_name}] Round {round_num} "
            f"model submitted successfully"
        )

    print(f"\n[{site_name}] Training complete.")


if __name__ == "__main__":
    main()
