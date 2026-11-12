import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import sys
from model import SimpleNet 
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH  = 32

def load_data(client_id):
    transform = transforms.Compose([transforms.ToTensor()])
    data_dir= Path(r"D:\FL_Data")
    trainset  = datasets.MNIST(root=str(data_dir), train=True,  download=True, transform=transform)
    testset   = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
    partition = len(trainset) // 2
    indices   = list(range(client_id * partition, (client_id + 1) * partition))
    return (
        DataLoader(Subset(trainset, indices), batch_size=BATCH, shuffle=True),
        DataLoader(testset, batch_size=BATCH)
    )

def train(model, loader, optimizer, criterion):
    model.train()
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        criterion(model(images), labels).backward()
        optimizer.step()

def evaluate(model, loader, criterion):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out     = model(images)
            loss   += criterion(out, labels).item()
            correct += (out.argmax(1) == labels).sum().item()
    return loss / len(loader), correct / len(loader.dataset)

class MNISTClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.model     = SimpleNet().to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.train_loader, self.test_loader = load_data(client_id)
        print(f"[Client {client_id}] Ready with {len(self.train_loader.dataset)} samples")

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        keys  = list(self.model.state_dict().keys())
        state = dict(zip(keys, [torch.tensor(p) for p in params]))
        self.model.load_state_dict(state, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        for _ in range(EPOCHS):
            train(self.model, self.train_loader, self.optimizer, self.criterion)
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate(self.model, self.test_loader, self.criterion)
        print(f"  → Accuracy: {acc:.4f}  Loss: {loss:.4f}")
        return loss, len(self.test_loader.dataset), {"accuracy": acc}

if __name__ == "__main__":
    client_id = int(sys.argv[1])
    fl.client.start_numpy_client(
        server_address="127.0.0.1:9090",
        client=MNISTClient(client_id)
    )