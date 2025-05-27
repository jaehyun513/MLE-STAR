import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# Define function to extract MFCC features
def extract_mfcc_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type="kaiser_fast")
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None
    return mfccs_processed


# Define function to extract Mel-Spectrogram features
def extract_mel_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type="kaiser_fast")
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_features_processed = np.mean(mel_spectrogram_db.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None
    return mel_features_processed


# Set paths
train_dir = "./input/train2/"
test_dir = "./input/test2/"
submission_dir = "./submission/"
if not os.path.exists(submission_dir):
    os.makedirs(submission_dir)

# Load training data and extract labels
train_files = [f for f in os.listdir(train_dir) if f.endswith(".aif")]
train_labels = [1 if f.endswith("_1.aif") else 0 for f in train_files]

# Extract features from training data
train_features = []
for i, file in enumerate(train_files):
    file_path = os.path.join(train_dir, file)
    mfcc_features = extract_mfcc_features(file_path)
    mel_features = extract_mel_features(file_path)

    if mfcc_features is not None and mel_features is not None:
        combined_features = np.concatenate((mfcc_features, mel_features))
        train_features.append(combined_features)
    else:
        train_files.pop(i)
        train_labels.pop(i)

train_features = np.array(train_features)

# Define parameter grid for GridSearchCV
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
}

# Train a Gradient Boosting Classifier model with cross-validation and hyperparameter tuning
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_scores = []
models = []

for fold, (train_index, val_index) in enumerate(
    skf.split(train_features, train_labels)
):
    X_train, X_val = train_features[train_index], train_features[val_index]
    y_train, y_val = [train_labels[i] for i in train_index], [
        train_labels[i] for i in val_index
    ]

    # Use GridSearchCV to find the best hyperparameters
    model = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, scoring="roc_auc", cv=3, n_jobs=-1
    )  # Use 3-fold CV within GridSearchCV
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred)
    cv_scores.append(score)
    models.append(best_model)
    print(f"Fold {fold+1} AUC: {score}")
    print(f"Fold {fold+1} Best parameters: {grid_search.best_params_}")

print(f"Mean AUC: {np.mean(cv_scores)}")

# Load test data
test_files = [f for f in os.listdir(test_dir) if f.endswith(".aif")]

# Extract features from test data
test_features = []
for file in test_files:
    file_path = os.path.join(test_dir, file)
    mfcc_features = extract_mfcc_features(file_path)
    mel_features = extract_mel_features(file_path)

    if mfcc_features is not None and mel_features is not None:
        combined_features = np.concatenate((mfcc_features, mel_features))
        test_features.append(combined_features)
    else:
        test_features.append(
            np.zeros(train_features.shape[1])
        )  # Pad with zeros if feature extraction fails

test_features = np.array(test_features)

# Predict probabilities on the test set using the average of predictions from all folds
test_predictions = np.zeros(len(test_files))
for model in models:
    test_predictions += model.predict_proba(test_features)[:, 1] / len(models)

# Create submission file
submission = pd.DataFrame({"clip": test_files, "probability": test_predictions})
submission.to_csv(os.path.join(submission_dir, "submission.csv"), index=False)
