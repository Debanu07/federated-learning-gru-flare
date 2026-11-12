import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_diff_flwr import Net


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

client_id = int(input("Enter client id (1 = MNIST, 2 = FashionMNIST): "))

transform = transforms.ToTensor()

# ----------------------------
# Dataset Loading
# ----------------------------

if client_id == 1:

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    print("Client 1 using MNIST")

else:

    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    print("Client 2 using FashionMNIST")


trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = Net().to(DEVICE)


# ----------------------------
# Training Function
# ----------------------------

def train(model, loader, epochs=20):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(epochs):

        running_loss = 0

        for images, labels in loader:

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/20 Loss: {running_loss/len(loader):.4f}")


# ----------------------------
# Accuracy Evaluation
# ----------------------------

def test(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


# ----------------------------
# Flower Client
# ----------------------------

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):

        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):

        params_dict = zip(model.state_dict().keys(), parameters)

        state_dict = {k: torch.tensor(v) for k, v in params_dict}

        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        train(model, trainloader, epochs=20)

        accuracy = test(model, testloader)

        print(f"\nTest Accuracy: {accuracy:.2f}%\n")

        return self.get_parameters(config={}), len(train_dataset), {"accuracy": accuracy}

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        accuracy = test(model, testloader)

        return 0.0, len(test_dataset), {"accuracy": accuracy}


fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=FlowerClient()
)