import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split


def create_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataset, train_loader, test_loader


def create_model(num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1")

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def train(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {running_loss / len(train_loader):.3f} | "
            f"Train Accuracy: {accuracy:.1f}%"
        )


def evaluate(model, test_loader, class_names):
    model.eval()

    num_classes = len(class_names)

    correct = 0
    total = 0

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i].item()

                class_correct[label] += (
                    predicted[i].item() == label
                )

                class_total[label] += 1

    print(f"\nOverall Test Accuracy: {100 * correct / total:.1f}%")

    print("\nPer Class Accuracy:")

    for i, class_name in enumerate(class_names):
        acc = 100 * class_correct[i] / class_total[i]
        print(f"{class_name}: {acc:.1f}%")


def main():
    data_dir = "path/to/spectrogram_dataset"
    model_path = "models/chicken_model.pth"

    dataset, train_loader, test_loader = create_dataloaders(data_dir)

    print(f"Total images: {len(dataset)}")
    print(f"Classes found: {dataset.classes}")

    model = create_model(len(dataset.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    train(
        model,
        train_loader,
        criterion,
        optimizer,
        epochs=10
    )

    evaluate(
        model,
        test_loader,
        dataset.classes
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
