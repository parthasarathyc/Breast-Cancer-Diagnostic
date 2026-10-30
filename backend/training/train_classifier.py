import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, accuracy_score

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def train_busi_classifier(epochs=15, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    data_dir = "backend/data/busi_processed"
    train_dir = os.path.join(data_dir, "train", "images")
    val_dir = os.path.join(data_dir, "val", "images")
    test_dir = os.path.join(data_dir, "test", "images")

    train_tf, val_tf = get_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_tf)

    print(f"[*] Dataset classes: {train_dataset.classes}")
    print(f"[*] Splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Backbone: EfficientNet-B0 pretrained
    print("[*] Instantiating EfficientNet-B0 backbone...")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=3)
    model.to(device)

    # Class-weighted loss to handle dataset imbalance
    # Benign: ~437, Malignant: ~210, Normal: ~133
    weights = torch.tensor([1.0, 2.0, 3.2], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    os.makedirs("backend/models", exist_ok=True)
    best_weights_path = "backend/models/best_classifier.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total if total > 0 else 0
        train_loss = running_loss / total if total > 0 else 0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            print(f"   -> [BEST] Saved new checkpoint with Val Acc: {best_val_acc*100:.2f}% to {best_weights_path}")

    # Evaluate on Holdout Test Set
    print("\n[*] Evaluating on Test Set...")
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_targets, all_preds)
    print(f"[OK] Final Test Set Accuracy: {test_acc*100:.2f}%")
    print("\nClassification Report:\n", classification_report(all_targets, all_preds, target_names=train_dataset.classes))

if __name__ == "__main__":
    train_busi_classifier(epochs=10, batch_size=16, lr=1e-4)
