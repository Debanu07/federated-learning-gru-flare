import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Exact same hyperparameters as FL ──────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS  = 60
BATCH   = 32     # same as FL
LR      = 0.01   # same as FL

# ── Exact same model as model.py ──────────────────────────
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)   
        self.fc2 = nn.Linear(128, 64)   
        self.fc3 = nn.Linear(64, 10)     

    def forward(self, x):
        x = x.view(-1, 784)             
        x = F.relu(self.fc1(x))          
        x = F.relu(self.fc2(x))          
        return self.fc3(x)              

# ── Load full MNIST ────────────────────────────────────────
def load_data():
    transform    = transforms.Compose([transforms.ToTensor()])
    trainset     = datasets.MNIST("./data", train=True,  download=False, transform=transform)
    testset      = datasets.MNIST("./data", train=False, download=False, transform=transform)
    train_loader = DataLoader(trainset, batch_size=BATCH, shuffle=True)  # same batch
    test_loader  = DataLoader(testset,  batch_size=BATCH)                # same batch
    return train_loader, test_loader

# ── Exact same train logic as FL client ───────────────────
def train(model, loader, optimizer, criterion):
    model.train()
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)   # same loss fn
        loss.backward()
        optimizer.step()                          # same optimizer

# ── Exact same evaluate logic as FL client ────────────────
def evaluate(model, loader, criterion):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs  = model(images)
            loss    += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    accuracy = correct / len(loader.dataset)
    return loss / len(loader), accuracy

if __name__ == "__main__":
    print("=" * 55)
    print("        CENTRALIZED TRAINING — Normal NN")
    print("=" * 55)
    print(f"  Model    : SimpleNet (784→128→64→10)")
    print(f"  Epochs   : {EPOCHS}  (3 rounds × 2 epochs, same as FL)")
    print(f"  Batch    : {BATCH}")
    print(f"  LR       : {LR}")
    print(f"  Optimizer: SGD")
    print(f"  Loss     : CrossEntropyLoss")
    print(f"  Data     : 60,000 train / 10,000 test")
    print(f"  Device   : {DEVICE}")
    print("=" * 55)

    train_loader, test_loader = load_data()

    # exact same setup as FL client
    model     = SimpleNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # simulate rounds for fair comparison display
    for round_num in range(1, 4):           # 3 rounds
        print(f"\n  --- Round {round_num} ---")
        for epoch in range(1, 3):           # 2 epochs per round
            train(model, train_loader, optimizer, criterion)
            loss, acc = evaluate(model, test_loader, criterion)
            print(f"  Epoch {epoch} → Accuracy: {acc:.4f}  Loss: {loss:.4f}")

        # show round summary same as FL output
        print(f"  → Round {round_num} Accuracy: {acc:.4f}  Loss: {loss:.4f}")

    print("\n" + "=" * 55)
    print(f"  FINAL ACCURACY : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  FINAL LOSS     : {loss:.4f}")
    print("=" * 55)