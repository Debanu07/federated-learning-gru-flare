import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import nvflare.client as flare
 
# ── Device ────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# ── Absolute data path (works regardless of NVFlare CWD) ──
DATA_DIR = os.path.expanduser("~/fl_data")
 
# ── Hyperparameters ───────────────────────────────────────
EPOCHS = 20
BATCH  = 32
LR     = 0.001
 
 
# ── Model — same Net as your manual project ───────────────
class Net(nn.Module):
 
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
 
# ── Data loading — site-1 gets MNIST, site-2 gets FashionMNIST ──
def load_data(site_name):
    transform = transforms.ToTensor()
 
    if "site-1" in site_name:
        train_dataset = datasets.MNIST(
            root=DATA_DIR, train=True,  download=True, transform=transform)
        test_dataset  = datasets.MNIST(
            root=DATA_DIR, train=False, download=True, transform=transform)
        print(f"[{site_name}] Using MNIST dataset")
 
    else:
        train_dataset = datasets.FashionMNIST(
            root=DATA_DIR, train=True,  download=True, transform=transform)
        test_dataset  = datasets.FashionMNIST(
            root=DATA_DIR, train=False, download=True, transform=transform)
        print(f"[{site_name}] Using FashionMNIST dataset")
 
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH)
 
    return train_loader, test_loader
 
 
# ── Train function ────────────────────────────────────────
def train(model, loader):
    model.to(DEVICE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
 
    for epoch in range(EPOCHS):
        running_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{EPOCHS}  Loss: {running_loss / len(loader):.4f}")
 
 
# ── Test function ─────────────────────────────────────────
def test(model, loader):
    model.to(DEVICE)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred     = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    accuracy = 100 * correct / total
    print(f"  Test Accuracy: {accuracy:.2f}%")
    return accuracy
 
 
# ── FLARE client logic ────────────────────────────────────
def main():
    flare.init()
 
    site_name = flare.get_site_name()
    print(f"\n[{site_name}] Starting")
 
    # load dataset based on site identity
    train_loader, test_loader = load_data(site_name)
 
    # build model
    model = Net().to(DEVICE)
 
    # FL training loop
    while flare.is_running():
 
        # receive global model from server
        input_model = flare.receive()
        round_num   = input_model.current_round
        print(f"\n[{site_name}] Round {round_num + 1} started")
 
        # load global weights if server has them (skip on round 0)
        if input_model.params:
            model.load_state_dict(input_model.params)
 
        # train locally for 20 epochs
        train(model, train_loader)
 
        # evaluate
        acc = test(model, test_loader)
        print(f"[{site_name}] Round {round_num + 1} → Accuracy: {acc:.2f}%")
 
        # send updated weights back to server
        output_model = flare.FLModel(
            params=model.state_dict(),
            metrics={"accuracy": acc},
            meta={"NUM_STEPS_CURRENT_ROUND": len(train_loader)}
        )
        flare.send(output_model)
 
    print(f"\n[{site_name}] Training complete")
 
 
if __name__ == "__main__":
    main()