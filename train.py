import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import accuracy_score

from data_dill import get_data_loaders

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据
data_dir = './data/'
batch_size = 32
train_loader, valid_loader = get_data_loaders(data_dir, batch_size=batch_size)

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

num_ftrs = model.fc.in_features
num_classes = len(train_loader.dataset.classes)
model.fc = nn.Linear(num_ftrs, num_classes)

checkpoint_path = 'bronze_resnet_model.pth'
if os.path.exists(checkpoint_path):
    print(f"Loading weights from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    print(f"No previous weights found at {checkpoint_path}")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    model.eval()
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(val_labels, val_preds)
    print(f'Validation Accuracy: {accuracy:.4f}')

torch.save(model.state_dict(), 'bronze_resnet_model.pth')
print("Model saved to 'bronze_resnet_model.pth'")
