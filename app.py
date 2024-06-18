import  json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def classify_image(model, img_path):
    model.eval()
    img = Image.open(img_path)
    img = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        return predicted.item()

test_image_path = "./data/valid/0/17020830.png"

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

num_classes = 17
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('final_fine_grained_resnet_model.pth'), strict=False)

model.to(device)

predicted_class = classify_image(model, test_image_path)


with open("bronze_vessel.json", "r") as f:
    label_map = json.load(f)

predicted_label = "Unknown"
for item in label_map["bronze_vessel"]:
    if item["id"] == predicted_class:
        predicted_label = item["name"]
        break

print(f"预测结果是: {predicted_label}")
