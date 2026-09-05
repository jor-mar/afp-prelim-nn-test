import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train(model, train_loader, device, optimizer, criterion):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    average_loss = total_loss / len(train_loader)

    return average_loss, accuracy


def evaluate(model, test_loader, device, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    average_loss = total_loss / len(test_loader)

    return average_loss, accuracy


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )

    model = MNISTMLP().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 10

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(
            model,
            train_loader,
            device,
            optimizer,
            criterion
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            device,
            criterion
        )

        print(
            f"Epoch {epoch:2d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.2f}%"
        )

    """
    torch.save(
        model.state_dict(),
        "mnist_mlp.pth"
    )
    """
    torch.save(
        model.state_dict(),
        SCRIPT_DIR / "mnist_mlp.pth"
    )

    print("\nModel saved to mnist_mlp.pth")


if __name__ == "__main__":
    main()