import torch
from torchvision import models, transforms
from PIL import Image
import os


# LOAD MODEL
model_path = "models/rice_model.pth"

print("\n Loading trained model...\n")

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 3)   # 3 classes
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()


# IMAGE TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

#class labels
class_names = ["blast", "blight", "tungro"]


# TEST MULTIPLE IMAGES
test_folder = "data/ricedataset"  # change to Bean_Dataset also

print(" Testing multiple images...\n")

for root, dirs, files in os.walk(test_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            img_path = os.path.join(root, file)

            try:
                img = Image.open(img_path).convert("RGB")
                img_t = transform(img).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(img_t)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, predicted = torch.max(probs, 1)

                print(f"Image: {file}")
                print(f"Prediction: {class_names[predicted.item()]} ({conf.item()*100:.2f}%)")
                print("-" * 60)

            except Exception as e:
                print(f"Error reading image {file}: {str(e)}")
