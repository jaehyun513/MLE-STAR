
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Configuration (Solution 1)
DATA_DIR = './input/train'  # Path to the training data
TEST_DIR = './input/test'
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Loading and Preprocessing (Solution 1)
def load_data(data_dir):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            try:
                img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
                label = 1 if 'dog' in filename else 0
                images.append(img)
                labels.append(label)
            except (FileNotFoundError, OSError) as e:
                print(f"Error loading image {filename}: {e}")
                continue
    return images, labels

def load_test_data(data_dir):
    images = []
    image_ids = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            try:
                img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
                image_id = int(filename.split('.')[0])
                images.append(img)
                image_ids.append(image_id)
            except (FileNotFoundError, OSError) as e:
                print(f"Error loading image {filename}: {e}")
                continue
    return images, image_ids

images, labels = load_data(DATA_DIR)

# Transformations (Solution 1)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, images, image_ids, transform=None):
        self.images = images
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image_id = self.image_ids[idx]
        if self.transform:
            image = self.transform(image)
        return image, image_id

# Split into training and validation sets (Solution 1)
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

train_dataset = CustomDataset(train_images, train_labels, data_transforms['train'])
val_dataset = CustomDataset(val_images, val_labels, data_transforms['val'])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Definitions (Solution 1)
class ResNet50BinaryClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50BinaryClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        # Freeze the ResNet layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Modify the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)

class VGG16Transfer(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16Transfer, self).__init__()
        self.vgg16 = models.vgg16(pretrained=pretrained)
        # Freeze layers
        for param in self.vgg16.parameters():
            param.requires_grad = False

        # Modify classifier
        num_features = self.vgg16.classifier[0].in_features
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vgg16.features(x)
        x = self.vgg16.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg16.classifier(x)
        return x

# Initialize models (Solution 1)
model_resnet = ResNet50BinaryClassifier().to(DEVICE)
model_vgg = VGG16Transfer().to(DEVICE)

# Loss function and optimizers (Solution 1)
criterion = nn.BCELoss()
optimizer_resnet = optim.Adam(model_resnet.resnet.fc.parameters(), lr=LEARNING_RATE)
optimizer_vgg = optim.Adam(model_vgg.parameters(), lr=LEARNING_RATE)

# Training Loop (Combined) (Solution 1)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)

                val_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss

# Train the models (Solution 1)
best_validation_loss_resnet = train_model(model_resnet, train_loader, val_loader, criterion, optimizer_resnet, NUM_EPOCHS, DEVICE)
best_validation_loss_vgg = train_model(model_vgg, train_loader, val_loader, criterion, optimizer_vgg, NUM_EPOCHS, DEVICE)


# EfficientNetB0 Model and Training (Solution 2 - Modified)
class DogCatDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True, val_size=0.2, random_state=42, test_dir=None):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.filepaths = []
        self.labels = []
        self.test_dir = test_dir

        if train:
            for filename in os.listdir(data_dir):
                if filename.endswith('.jpg'):
                    self.filepaths.append(os.path.join(data_dir, filename))
                    self.labels.append(1 if 'dog' in filename else 0)

            # Split into training and validation sets
            self.filepaths, _, self.labels, _ = train_test_split(
                self.filepaths, self.labels, test_size=val_size, random_state=random_state, stratify=self.labels
            )
        elif test_dir is None:
            for filename in os.listdir(data_dir):
                if filename.endswith('.jpg'):
                    self.filepaths.append(os.path.join(data_dir, filename))
                    self.labels.append(1 if 'dog' in filename else 0)
            _, self.filepaths, _, self.labels = train_test_split(
                self.filepaths, self.labels, test_size=val_size, random_state=random_state, stratify=self.labels
            )
        else:
            self.filepaths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]
            self.labels = [-1] * len(self.filepaths)  # Dummy labels for test data


    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image_path = self.filepaths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetB0, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.sigmoid(x)
        return x

# Training parameters (Solution 2)
batch_size = 32
learning_rate = 0.0001
num_epochs = 5
val_size = 0.2

train_transform = data_transforms['train']
val_transform = data_transforms['val']
test_transform = data_transforms['test']

train_dataset_eff = DogCatDataset(DATA_DIR, transform=train_transform, train=True, val_size=val_size)
val_dataset_eff = DogCatDataset(DATA_DIR, transform=val_transform, train=False, val_size=val_size)

train_loader_eff = DataLoader(train_dataset_eff, batch_size=batch_size, shuffle=True)
val_loader_eff = DataLoader(val_dataset_eff, batch_size=batch_size, shuffle=False)

model_eff = EfficientNetB0().to(DEVICE)
criterion_eff = nn.BCELoss()
optimizer_eff = optim.Adam(model_eff.parameters(), lr=learning_rate)

def train_model_eff(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return best_val_loss

best_validation_loss_eff = train_model_eff(model_eff, train_loader_eff, val_loader_eff, criterion_eff, optimizer_eff, num_epochs, DEVICE)

# Ensemble and Evaluate (Modified)
model_resnet.eval()
model_vgg.eval()
model_eff.eval()

all_labels = []
all_predictions = []

resnet_vgg_preds = []
effnet_preds = []
true_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs_resnet = model_resnet(inputs).squeeze().cpu().numpy()
        outputs_vgg = model_vgg(inputs).squeeze().cpu().numpy()
        outputs_eff = model_eff(inputs).squeeze().cpu().numpy()

        resnet_vgg_preds.extend((outputs_resnet + outputs_vgg) / 2.0)
        effnet_preds.extend(outputs_eff)
        true_labels.extend(labels.cpu().numpy())

# Calculate Prediction Concordance
concordance_scores = []
for i in range(len(true_labels)):
    # Binary concordance: 1 if both correct or both incorrect, 0 otherwise
    resnet_vgg_correct = (resnet_vgg_preds[i] > 0.5) == true_labels[i]
    effnet_correct = (effnet_preds[i] > 0.5) == true_labels[i]
    concordance_scores.append(1 if resnet_vgg_correct == effnet_correct else 0)

# Adaptive Weighting and Validation
all_labels = true_labels
all_predictions = []

for i in range(len(true_labels)):
    concordance = concordance_scores[i]
    resnet_vgg_pred = resnet_vgg_preds[i]
    effnet_pred = effnet_preds[i]

    if concordance == 1:  # Models agree
        if best_validation_loss_resnet < best_validation_loss_eff:
            ensemble_prediction = 0.6 * resnet_vgg_pred + 0.4 * effnet_pred
        else:
            ensemble_prediction = 0.4 * resnet_vgg_pred + 0.6 * effnet_pred
    else:  # Models disagree
        ensemble_prediction = 0.5 * resnet_vgg_pred + 0.5 * effnet_pred  # Equal weight

    all_predictions.append(ensemble_prediction)

# Calculate log loss
logloss = log_loss(all_labels, all_predictions)

print(f"ResNet Validation Loss: {best_validation_loss_resnet:.4f}")
print(f"VGG Validation Loss: {best_validation_loss_vgg:.4f}")
print(f"EfficientNet Validation Loss: {best_validation_loss_eff:.4f}")
print(f"Log Loss: {logloss:.4f}")

Final_Validation_Performance = logloss
print(f'Final Validation Performance: {Final_Validation_Performance}')

# Test Data Prediction
test_images, test_image_ids = load_test_data(TEST_DIR)
test_dataset = TestDataset(test_images, test_image_ids, data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Make predictions on test data
model_resnet.eval()
model_vgg.eval()
model_eff.eval()
test_predictions = []
test_ids = []

with torch.no_grad():
    for inputs, image_ids in test_loader:
        inputs = inputs.to(DEVICE)

        outputs_resnet = model_resnet(inputs).squeeze().cpu().numpy()
        outputs_vgg = model_vgg(inputs).squeeze().cpu().numpy()
        outputs_eff = model_eff(inputs).squeeze().cpu().numpy()
        resnet_vgg_preds = (outputs_resnet + outputs_vgg) / 2.0


        #Apply adaptive weighting for test predictions
        for i in range(len(resnet_vgg_preds)):
            resnet_vgg_pred = resnet_vgg_preds[i]
            effnet_pred = outputs_eff[i]

            #Simulate concordance (since we don't have true labels)
            resnet_vgg_agree = resnet_vgg_pred > 0.5
            effnet_agree = effnet_pred > 0.5

            concordance = 1 if resnet_vgg_agree == effnet_agree else 0

            if concordance == 1:  # Models agree
                if best_validation_loss_resnet < best_validation_loss_eff:
                    ensemble_prediction = 0.6 * resnet_vgg_pred + 0.4 * effnet_pred
                else:
                    ensemble_prediction = 0.4 * resnet_vgg_pred + 0.6 * effnet_pred
            else:  # Models disagree
                ensemble_prediction = 0.5 * resnet_vgg_pred + 0.5 * effnet_pred  # Equal weight
            test_predictions.append(ensemble_prediction)

        test_ids.extend(image_ids)

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'label': test_predictions})
submission = submission.sort_values('id')
submission['id'] = submission['id'].astype(int) # Ensure 'id' is integer

# Create ./final directory if it doesn't exist
if not os.path.exists('./final'):
    os.makedirs('./final')

submission.to_csv('./final/submission.csv', index=False)

print("Submission file created successfully!")
