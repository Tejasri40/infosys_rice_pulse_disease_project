import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Paths
DATA_DIR = "data/ricedataset"      # or pulse dataset folder
MODEL_SAVE_PATH = "models/rice_model.pth"

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 16
LR = 0.001

def train_model():
    print("\n========================================")
    print(" LOADING DATASET ")
    print("========================================")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    num_classes = len(dataset.classes)
    print(f"\nModel will train for {num_classes} classes:", dataset.classes)

    # Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0
    best_epoch = 0

    print("\n========================================")
    print(" STARTING TRAINING ")
    print("========================================")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == labels)

        train_acc = 100 * correct / len(train_data)

        # Validation
        model.eval()
        val_correct = 0
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                output = model(images)
                loss = criterion(output, labels)

                val_loss += loss.item()
                _, preds = torch.max(output, 1)
                val_correct += torch.sum(preds == labels)

        val_acc = 100 * val_correct / len(val_data)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✓ Best model saved! (Epoch {best_epoch}, Acc: {best_acc:.2f}%)")

    print("\n========================================")
    print(" TRAINING COMPLETE ")
    print("========================================")
    print(f"BEST ACCURACY: {best_acc:.2f}% at Epoch {best_epoch}")
    print(f"Model saved at: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
