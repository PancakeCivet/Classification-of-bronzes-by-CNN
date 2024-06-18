import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import accuracy_score
from data_dill import get_data_loaders


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = './data/'
    batch_size = 64
    train_loader, valid_loader = get_data_loaders(data_dir, batch_size=batch_size, num_workers=2, pin_memory=True)

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    num_ftrs = model.fc.in_features
    num_classes = len(train_loader.dataset.classes)
    model.fc = nn.Linear(num_ftrs, num_classes)

    checkpoint_path = 'final_fine_grained_resnet_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")
            print("Attempting to load partial state_dict and reinitialize the final layer.")
            state_dict = checkpoint.copy()
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            model.load_state_dict(state_dict, strict=False)
            model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        print(f"No previous weights found at {checkpoint_path}. Using pre-trained weights.")

    model = model.to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 100
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if device.type == 'cuda':
                print(f"Training on GPU: {images.device} | Labels on GPU: {labels.device}")

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # 验证模型
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                if device.type == 'cuda':
                    print(f"Validation on GPU: {images.device} | Labels on GPU: {labels.device}")

                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        accuracy = accuracy_score(val_labels, val_preds)
        print(f'Validation Accuracy: {accuracy:.4f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_fine_grained_resnet_model.pth')
            print("New best model saved!")

    torch.save(model.state_dict(), 'final_fine_grained_resnet_model.pth')
    print("Final model saved to 'final_fine_grained_resnet_model.pth'")


# 使用标准的Python入口
if __name__ == "__main__":
    main()