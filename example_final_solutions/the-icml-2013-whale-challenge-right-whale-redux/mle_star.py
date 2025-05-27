
import librosa
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio
from scipy.interpolate import interp1d

# Solution 1 Dataset
class WhaleDataset1(Dataset):
    def __init__(self, data_dir, file_list, labels=None, transform=None, target_length=64, is_test=False):
        self.data_dir = data_dir
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        self.target_length = target_length
        self.is_test = is_test

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        try:
            audio, sr = torchaudio.load(file_path) # Use torchaudio
            audio = audio.squeeze().numpy() # Remove channel dimension if present
        except Exception as e:
            print(f"Error loading audio file {file_name}: {e}")
            return None, None

        # Create spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=64)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Pad or truncate spectrogram to a fixed length
        if spectrogram.shape[1] < self.target_length:
            padding_width = self.target_length - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, padding_width)), mode='constant')
        elif spectrogram.shape[1] > self.target_length:
            spectrogram = spectrogram[:, :self.target_length]

        spectrogram = np.expand_dims(spectrogram, axis=0)  # Add channel dimension
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32) # Convert to tensor
        
        if self.labels is not None:
            label = self.labels[idx]
            return spectrogram, torch.tensor(label, dtype=torch.float32)
        else:
            return spectrogram, file_name

# Solution 2 Dataset
class WhaleDataset2(Dataset):
    def __init__(self, data_dir, file_list, labels=None, is_test=False):
        self.data_dir = data_dir
        self.file_list = file_list
        self.labels = labels
        self.is_test = is_test
        self.target_length = 128  # Define a target length for spectrograms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        try:
            y, sr = librosa.load(file_path, sr=2000)  # Load at 2kHz
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
            spectrogram = librosa.power_to_db(spectrogram)
            spectrogram = np.expand_dims(spectrogram, axis=0)  # Add channel dimension
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

            # Pad or truncate spectrogram to the target length
            current_length = spectrogram.shape[2]
            if current_length < self.target_length:
                padding_size = self.target_length - current_length
                padding = torch.zeros((1, 128, padding_size), dtype=torch.float32)
                spectrogram = torch.cat((spectrogram, padding), dim=2)
            elif current_length > self.target_length:
                spectrogram = spectrogram[:, :, :self.target_length]
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            # Handle the error gracefully, e.g., return a zero tensor and a default label
            spectrogram = torch.zeros((1, 128, self.target_length), dtype=torch.float32)  # Example shape, adjust if needed
            if self.is_test:
                 return spectrogram, file_path.split('/')[-1]
            else:
                label = torch.tensor(0, dtype=torch.float32)
                return spectrogram, label
        
        if self.is_test:
            return spectrogram, file_path.split('/')[-1]
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return spectrogram, label

# Solution 1 Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class WhaleCNN1(nn.Module):
    def __init__(self):
        super(WhaleCNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.se1 = SEBlock(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.se2 = SEBlock(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # Adjusted for 64x16 spectrograms
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def mish(self, x):
        return x * torch.tanh(F.softplus(x))

    def forward(self, x):
        x = self.conv1(x)
        x = self.mish(x)
        x = self.se1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.mish(x)
        x = self.se2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.mish(x)
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Solution 2 Model
class WhaleCNN2(nn.Module):
    def __init__(self):
        super(WhaleCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # Adjusted input size after 3 conv layers and max pooling
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and preprocessing
train_data_dir = './input/train'
train_file_list = os.listdir(train_data_dir)
train_labels = [1 if file.endswith('_1.aif') else 0 for file in train_file_list]

# Split data into training and validation sets
train_files, val_files, train_labels, val_labels = train_test_split(
    train_file_list, train_labels, test_size=0.2, random_state=42
)

# Create datasets
target_length_1 = 64
train_dataset_1 = WhaleDataset1(train_data_dir, train_files, train_labels, target_length=target_length_1)
val_dataset_1 = WhaleDataset1(train_data_dir, val_files, val_labels, target_length=target_length_1)

train_dataset_2 = WhaleDataset2(train_data_dir, train_files, train_labels)
val_dataset_2 = WhaleDataset2(train_data_dir, val_files, val_labels)

# Create data loaders
batch_size = 32
train_loader_1 = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True, collate_fn=lambda x: list(zip(*[i for i in x if i[0] is not None])))
val_loader_1 = DataLoader(val_dataset_1, batch_size=batch_size, shuffle=False, collate_fn=lambda x: list(zip(*[i for i in x if i[0] is not None])))

train_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True)
val_loader_2 = DataLoader(val_dataset_2, batch_size=batch_size, shuffle=False)

# Model instantiation
model_1 = WhaleCNN1().to(device)
model_2 = WhaleCNN2().to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer_1 = optim.Adam(model_1.parameters(), lr=0.001)
optimizer_2 = optim.Adam(model_2.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model_1.train()
    train_loss_1 = 0.0
    for spectrograms, labels in tqdm(train_loader_1, desc=f"Epoch {epoch+1}/{num_epochs} - Training Model 1"):
        spectrograms = torch.stack(spectrograms).to(device)
        labels = torch.stack(labels).to(device)

        optimizer_1.zero_grad()
        outputs = model_1(spectrograms)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer_1.step()

        train_loss_1 += loss.item() * spectrograms.size(0)
    
    train_loss_1 = train_loss_1 / len(train_dataset_1)
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss Model 1: {train_loss_1:.4f}")

    model_2.train()
    for spectrograms, labels in tqdm(train_loader_2, desc=f"Epoch {epoch+1}/{num_epochs} - Training Model 2"):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        optimizer_2.zero_grad()
        outputs = model_2(spectrograms)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer_2.step()
    
# Validation loop
model_1.eval()
val_loss_1 = 0.0
val_preds_1 = []
val_targets_1 = []
with torch.no_grad():
    for spectrograms, labels in tqdm(val_loader_1, desc="Validation Model 1"):
        spectrograms = torch.stack(spectrograms).to(device)
        labels = torch.stack(labels).to(device)

        outputs = model_1(spectrograms)
        loss = criterion(outputs.squeeze(), labels)

        val_loss_1 += loss.item() * spectrograms.size(0)

        val_preds_1.extend(outputs.squeeze().cpu().numpy())
        val_targets_1.extend(labels.cpu().numpy())

val_loss_1 = val_loss_1 / len(val_dataset_1)
val_auc_1 = roc_auc_score(val_targets_1, val_preds_1)
print(f"Validation Loss Model 1: {val_loss_1:.4f}, Validation AUC Model 1: {val_auc_1:.4f}")

model_2.eval()
val_preds_2 = []
val_targets_2 = []
with torch.no_grad():
    for spectrograms, labels in val_loader_2:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        outputs = model_2(spectrograms)
        val_preds_2.extend(outputs.squeeze().cpu().numpy())
        val_targets_2.extend(labels.cpu().numpy())

# Calculate AUC
auc_2 = roc_auc_score(val_targets_2, val_preds_2)
print(f'Validation AUC Model 2: {auc_2}')

# Confidence Recalibration
def calibrate_confidence(predictions, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    calibrated_probs = np.zeros_like(predictions, dtype=float)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
        if np.any(in_bin):
            fraction_in_bin = np.mean(labels[in_bin])
            calibrated_probs[in_bin] = fraction_in_bin
    return calibrated_probs

calibrated_val_preds_1 = calibrate_confidence(np.array(val_preds_1), np.array(val_targets_1))
calibrated_val_preds_2 = calibrate_confidence(np.array(val_preds_2), np.array(val_targets_2))

# Diversity Analysis (identify where only one model is correct)
agree_correct = 0
disagree_correct_1 = 0
disagree_correct_2 = 0
total = len(val_targets_1)

for i in range(total):
  pred_1 = round(calibrated_val_preds_1[i])
  pred_2 = round(calibrated_val_preds_2[i])
  target = val_targets_1[i]

  if pred_1 == target and pred_2 == target:
      agree_correct += 1
  elif pred_1 == target and pred_2 != target:
      disagree_correct_1 += 1
  elif pred_1 != target and pred_2 == target:
      disagree_correct_2 += 1

# Simple Robustness Metric (Placeholder, replace with actual implementation)
def estimate_robustness(model, spectrogram, device):
    # This is a placeholder. Implement a proper robustness metric.
    return 0.5

# Test data loading and prediction
test_data_dir = './input/test'
test_file_list = os.listdir(test_data_dir)

test_dataset_1 = WhaleDataset1(test_data_dir, test_file_list, labels=None, target_length=target_length_1, is_test=True)
test_loader_1 = DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False, collate_fn=lambda x: list(zip(*[i for i in x if i[0] is not None])))

test_dataset_2 = WhaleDataset2(test_data_dir, test_file_list, is_test=True)
test_loader_2 = DataLoader(test_dataset_2, batch_size=batch_size, shuffle=False)

model_1.eval()
test_predictions_1 = []
test_clips_1 = []

model_2.eval()
test_predictions_2 = []
test_clips_2 = []

with torch.no_grad():
    for spectrograms, clips in tqdm(test_loader_1, desc="Testing Model 1"):
        spectrograms = torch.stack(spectrograms).to(device)
        outputs = model_1(spectrograms)
        test_predictions_1.extend(outputs.squeeze().cpu().numpy())
        test_clips_1.extend(clips)

with torch.no_grad():
    for spectrograms, clips in test_loader_2:
        spectrograms = spectrograms.to(device)
        outputs = model_2(spectrograms)
        test_predictions_2.extend(outputs.squeeze().cpu().numpy())
        test_clips_2.extend(clips)

# Ensure clips are the same for both models (important for ensembling)
if test_clips_1 != test_clips_2:
    raise ValueError("Clip lists are not identical between the two models!")

# Ensemble predictions
ensemble_predictions = []
for i in range(len(test_predictions_1)):
    clip = test_clips_1[i]
    
    # Calibrate test predictions
    test_pred_1_calibrated = interp1d(np.linspace(0, 1, len(np.unique(calibrated_val_preds_1))), sorted(np.unique(calibrated_val_preds_1)))(test_predictions_1[i])
    test_pred_2_calibrated = interp1d(np.linspace(0, 1, len(np.unique(calibrated_val_preds_2))), sorted(np.unique(calibrated_val_preds_2)))(test_predictions_2[i])

    # Placeholder for robustness scores (replace with your actual scores)
    robustness_1 = estimate_robustness(model_1, test_predictions_1[i], device)
    robustness_2 = estimate_robustness(model_2, test_predictions_2[i], device)

    if test_pred_1_calibrated > 0.5 and test_pred_2_calibrated > 0.5: #Both agree
        w_1 = robustness_1 / (robustness_1 + robustness_2)
        w_2 = robustness_2 / (robustness_1 + robustness_2)
        ensemble_predictions.append(w_1 * test_pred_1_calibrated + w_2 * test_pred_2_calibrated)
    elif test_pred_1_calibrated < 0.5 and test_pred_2_calibrated < 0.5: #Both agree
        w_1 = robustness_1 / (robustness_1 + robustness_2)
        w_2 = robustness_2 / (robustness_1 + robustness_2)
        ensemble_predictions.append(w_1 * test_pred_1_calibrated + w_2 * test_pred_2_calibrated)
    else:  # Models disagree, use robustness-weighted average
        w_1 = robustness_1 / (robustness_1 + robustness_2)
        w_2 = robustness_2 / (robustness_1 + robustness_2)
        ensemble_predictions.append(w_1 * test_pred_1_calibrated + w_2 * test_pred_2_calibrated)

# Create submission DataFrame
submission = pd.DataFrame({'clip': test_clips_1, 'probability': ensemble_predictions})

# Create the final directory if it doesn't exist
output_dir = './final'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save submission file
submission_path = os.path.join(output_dir, 'submission.csv')
submission.to_csv(submission_path, index=False)

print(f"Submission file saved to {submission_path}")

# Calculate final validation score (using a simple average of the two models for now)
final_val_preds = (np.array(val_preds_1) + np.array(val_preds_2)) / 2
final_validation_score = roc_auc_score(val_targets_1, final_val_preds)

print(f'Final Validation Performance: {final_validation_score}')
