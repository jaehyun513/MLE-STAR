import os
import random
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import timm

# Configuration
DATA_DIR = "./input/"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224


# Prepare data
def extract_data():
    if not os.path.exists(TRAIN_DIR):
        with zipfile.ZipFile(os.path.join(DATA_DIR, "train.zip"), "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

    if not os.path.exists(TEST_DIR):
        with zipfile.ZipFile(os.path.join(DATA_DIR, "test.zip"), "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)


extract_data()


# Dataset class
class DogsCatsDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.image_names = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.train:
            label = 1 if "dog" in image_name else 0
        else:
            label = 0  # Placeholder for test data

        if self.transform:
            image = self.transform(image)

        return image, label, image_name


# Transformations
train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create datasets
full_dataset = DogsCatsDataset(TRAIN_DIR, transform=train_transform, train=True)

# Split into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]
)

test_dataset = DogsCatsDataset(TEST_DIR, transform=test_transform, train=False)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Model
class DogCatClassifier(nn.Module):
    def __init__(self):
        super(DogCatClassifier, self).__init__()
        self.efficientnet = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=2
        )

    def forward(self, x):
        return self.efficientnet(x)


model = DogCatClassifier().to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probabilities = torch.softmax(outputs, dim=1)
                predictions = probabilities[:, 1].cpu().numpy()
                labels = labels.cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(labels)

        val_logloss = log_loss(all_labels, all_predictions)
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Val LogLoss: {val_logloss}"
        )

        if val_logloss < best_val_loss:
            best_val_loss = val_logloss
            torch.save(model.state_dict(), "best_model.pth")

    return best_val_loss


best_val_loss = train_model(
    model, train_loader, val_loader, criterion, optimizer, EPOCHS, DEVICE
)
print(f"Best Validation LogLoss: {best_val_loss}")


# Prediction and submission
def predict(model, test_loader, device):
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    predictions = []
    image_ids = []
    with torch.no_grad():
        for images, _, image_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_probs = probabilities[:, 1].cpu().numpy()
            predictions.extend(predicted_probs)
            image_ids.extend([int(name.split(".")[0]) for name in image_names])

    submission = pd.DataFrame({"id": image_ids, "label": predictions})
    submission = submission.sort_values("id")
    return submission


submission = predict(model, test_loader, DEVICE)

# Save submission
os.makedirs("./submission", exist_ok=True)
submission.to_csv("./submission/submission.csv", index=False)
