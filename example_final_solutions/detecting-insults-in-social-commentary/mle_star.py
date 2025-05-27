
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Model 1: RoBERTa
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Model 2: Logistic Regression with TF-IDF
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Model 3: Naive Bayes with CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Model 4: SVM with TF-IDF
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
import codecs
import os

# Solution 1 imports
from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVectorizerSol1, CountVectorizer as CountVectorizerSol1
from sklearn.linear_model import LogisticRegression as LogisticRegressionSol1
from sklearn.svm import LinearSVC as LinearSVCSol1
from sklearn.naive_bayes import MultinomialNB as MultinomialNBSol1
from sklearn.ensemble import GradientBoostingClassifier
import re

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)
os.makedirs("./final", exist_ok=True)

# Load data
try:
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
except FileNotFoundError:
    # Fallback for debug environment if necessary
    print("Using debug input path")
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")


# Preprocessing (handle missing values and unicode escaping)
train_df = train_df.fillna('')
test_df = test_df.fillna('')

def preprocess_text(text):
    if not isinstance(text, str): # Handle potential non-string types just in case
        return ''
    try:
        # Attempt to decode unicode escapes, ignore errors for non-escaped strings
        text = codecs.decode(text, 'unicode_escape', 'ignore')
    except Exception as e:
        # In case of unexpected errors during decoding, keep the original text
        # print(f"Could not decode: {text} due to {e}") # Optional: for debugging
        pass
    return text.strip()

train_df['Comment'] = train_df['Comment'].apply(preprocess_text)
test_df['Comment'] = test_df['Comment'].apply(preprocess_text)


# Split data
X = train_df['Comment']
y = train_df['Insult'].astype(int)
# Use stratification to handle potential class imbalance
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- Model Training (Solution 1 models as feature extractors) ---

# Model 1: Logistic Regression with TF-IDF (Character n-grams)
print("Training Solution 1 Model 1: LR Char TF-IDF...")
tfidf_vectorizer_char = TfidfVectorizerSol1(analyzer='char_wb', ngram_range=(2, 5))
X_train_tfidf_char = tfidf_vectorizer_char.fit_transform(X_train)
X_val_tfidf_char = tfidf_vectorizer_char.transform(X_val)
X_test_tfidf_char = tfidf_vectorizer_char.transform(test_df['Comment'])

lr_classifier_char = LogisticRegressionSol1(solver='liblinear', random_state=42, C=1.0) # Added C parameter
lr_classifier_char.fit(X_train_tfidf_char, y_train)
val_predictions_lr_char = lr_classifier_char.predict_proba(X_val_tfidf_char)[:, 1]
test_predictions_lr_char = lr_classifier_char.predict_proba(X_test_tfidf_char)[:, 1]
roc_auc_lr_char = roc_auc_score(y_val, val_predictions_lr_char)
print(f"Logistic Regression AUC (Char n-grams): {roc_auc_lr_char}")

# Model 2: Linear SVM with TF-IDF (Character n-grams)
print("Training Solution 1 Model 2: SVM Char TF-IDF...")
svm_classifier_char = LinearSVCSol1(random_state=42, C=1.0) # Added C parameter and changed to 1.0
svm_classifier_char.fit(X_train_tfidf_char, y_train)
svm_predictions_char_raw = svm_classifier_char.decision_function(X_val_tfidf_char)
svm_predictions_char_test_raw = svm_classifier_char.decision_function(X_test_tfidf_char)

# Scale decision function to [0, 1] range for AUC and ensembling
scaler_char = MinMaxScaler()
val_predictions_svm_char = scaler_char.fit_transform(svm_predictions_char_raw.reshape(-1, 1)).flatten()
test_predictions_svm_char = scaler_char.transform(svm_predictions_char_test_raw.reshape(-1, 1)).flatten()

svm_auc_char = roc_auc_score(y_val, val_predictions_svm_char) # Use scaled predictions for AUC
print(f"Linear SVM AUC (Char n-grams): {svm_auc_char}")


# Model 3: Logistic Regression with TF-IDF (Word bi-grams)
print("Training Solution 1 Model 3: LR Word TF-IDF...")
tfidf_vectorizer_word = TfidfVectorizerSol1(ngram_range=(1, 2), min_df=3, max_df=0.9) # Added min/max df
X_train_tfidf_word = tfidf_vectorizer_word.fit_transform(X_train)
X_val_tfidf_word = tfidf_vectorizer_word.transform(X_val)
X_test_tfidf_word = tfidf_vectorizer_word.transform(test_df['Comment'])


lr_classifier_word = LogisticRegressionSol1(solver='liblinear', random_state=42, C=1.0) # Added C parameter
lr_classifier_word.fit(X_train_tfidf_word, y_train)
val_predictions_lr_word = lr_classifier_word.predict_proba(X_val_tfidf_word)[:, 1]
test_predictions_lr_word = lr_classifier_word.predict_proba(X_test_tfidf_word)[:, 1]

lr_auc_word = roc_auc_score(y_val, val_predictions_lr_word)
print(f"Logistic Regression AUC (Word bi-grams): {lr_auc_word}")


# Model 4: Linear SVM with TF-IDF (Word bi-grams)
print("Training Solution 1 Model 4: SVM Word TF-IDF...")
svm_classifier_word = LinearSVCSol1(random_state=42, C=0.1) # Added C parameter
svm_classifier_word.fit(X_train_tfidf_word, y_train)

svm_predictions_word_raw = svm_classifier_word.decision_function(X_val_tfidf_word)
svm_predictions_word_test_raw = svm_classifier_word.decision_function(X_test_tfidf_word)

scaler_word = MinMaxScaler()
val_predictions_svm_word = scaler_word.fit_transform(svm_predictions_word_raw.reshape(-1, 1)).flatten()
test_predictions_svm_word = scaler_word.transform(svm_predictions_word_test_raw.reshape(-1, 1)).flatten()


svm_auc_word = roc_auc_score(y_val, val_predictions_svm_word) # Use scaled predictions for AUC
print(f"Linear SVM AUC (Word bi-grams): {svm_auc_word}")


# Model 5: Naive Bayes with CountVectorizer (Word uni-grams)
print("Training Solution 1 Model 5: NB Word CountVec...")
count_vectorizer = CountVectorizerSol1(ngram_range=(1, 1)) # Explicitly use uni-grams
X_train_count = count_vectorizer.fit_transform(X_train)
X_val_count = count_vectorizer.transform(X_val)
X_test_count = count_vectorizer.transform(test_df['Comment'])

nb_classifier = MultinomialNBSol1(alpha=1.0) # Added alpha
nb_classifier.fit(X_train_count, y_train)
val_predictions_nb_count = nb_classifier.predict_proba(X_val_count)[:, 1]
test_predictions_nb_count = nb_classifier.predict_proba(X_test_count)[:, 1]
nb_auc = roc_auc_score(y_val, val_predictions_nb_count)
print(f"Naive Bayes AUC: {nb_auc}")

# --- Solution 1 Model Selection ---
solution1_models = {
    'lr_char': val_predictions_lr_char,
    'svm_char': val_predictions_svm_char,
    'lr_word': val_predictions_lr_word,
    'svm_word': val_predictions_svm_word,
    'nb_count': val_predictions_nb_count
}
solution1_test_models = {
    'lr_char': test_predictions_lr_char,
    'svm_char': test_predictions_svm_char,
    'lr_word': test_predictions_lr_word,
    'svm_word': test_predictions_svm_word,
    'nb_count': test_predictions_nb_count
}

solution1_aucs = {
    'lr_char': roc_auc_lr_char,
    'svm_char': svm_auc_char,
    'lr_word': lr_auc_word,
    'svm_word': svm_auc_word,
    'nb_count': nb_auc
}

# Select top N models from Solution 1 based on validation AUC
N = 3 # Number of top models to select
top_solution1_models = sorted(solution1_aucs, key=solution1_aucs.get, reverse=True)[:N]
print(f"Top {N} Solution 1 Models: {top_solution1_models}")

# --- Model Training (Solution 2 with Solution 1 features) ---

# Use CUDA if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Model 1: RoBERTa (Simplified Training)
model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model_roberta = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2) # Specify num_labels
model_roberta.to(device)

# Tokenize data
MAX_LEN = 128 # Or adjust based on your data/memory
BATCH_SIZE = 8 # Adjust based on GPU memory

# Simple Dataset and DataLoader
class InsultDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Concatenate Solution 1 predictions to the input text
X_train_combined = X_train.copy()
X_val_combined = X_val.copy()
X_test_combined = test_df['Comment'].copy()

for model_name in top_solution1_models:
    X_train_combined = X_train_combined.astype(str) + ' ' + pd.Series(solution1_models[model_name], index=X_val.index).reindex(X_train.index, fill_value=0).astype(str)
    X_val_combined = X_val_combined.astype(str) + ' ' + pd.Series(solution1_models[model_name], index=X_val.index).astype(str)
    X_test_combined = X_test_combined.astype(str) + ' ' + pd.Series(solution1_test_models[model_name]).astype(str)

X_train_list = X_train_combined.tolist()
X_val_list = X_val_combined.tolist()
X_test_list = X_test_combined.tolist()


print("Tokenizing training data for RoBERTa...")
X_train_encoded = tokenizer(X_train_list, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
print("Tokenizing validation data for RoBERTa...")
X_val_encoded = tokenizer(X_val_list, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
print("Tokenizing test data for RoBERTa...")
X_test_encoded = tokenizer(X_test_list, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')

train_dataset = InsultDataset({key: val.tolist() for key, val in X_train_encoded.items()}, y_train.tolist())
val_dataset = InsultDataset({key: val.tolist() for key, val in X_val_encoded.items()}, y_val.tolist())
test_dataset = InsultDataset({key: val.tolist() for key, val in X_test_encoded.items()}, [0] * len(X_test_list))


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)



from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

# Train RoBERTa (with learning rate scheduler and gradient accumulation)
optimizer = AdamW(model_roberta.parameters(), lr=2e-5)
num_epochs = 3 # Increased epochs for better fine-tuning
gradient_accumulation_steps = 2 # Simulate larger batch size

total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps #Steps considering grad accumulation
warmup_steps = int(0.1 * total_steps) # 10% warmup

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

print("Training RoBERTa...")
model_roberta.train()
for epoch in range(num_epochs):
    print(f"RoBERTa Epoch {epoch+1}/{num_epochs}")
    for batch_num, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model_roberta(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps # Scale loss for accumulation
        loss.backward()

        if (batch_num + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        # if batch_num % 50 == 0: # Print progress
        #     print(f"  Batch {batch_num}/{len(train_loader)}, Loss: {loss.item()}")


# Evaluate RoBERTa
print("Evaluating RoBERTa...")
model_roberta.eval()
val_predictions_roberta = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model_roberta(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        val_predictions_roberta.extend(probabilities[:, 1].cpu().numpy()) # Probability of class 1

val_predictions_roberta = np.array(val_predictions_roberta)
roc_auc_roberta = roc_auc_score(y_val, val_predictions_roberta)
print(f"RoBERTa Validation AUC: {roc_auc_roberta}")


# Test RoBERTa
print("Testing RoBERTa...")
model_roberta.eval()
test_predictions_roberta = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model_roberta(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        test_predictions_roberta.extend(probabilities[:, 1].cpu().numpy())
test_predictions_roberta = np.array(test_predictions_roberta)


# Model 2: Logistic Regression with TF-IDF
print("Training Logistic Regression...")

# Concatenate Solution 1 predictions to the input text
X_train_combined = X_train.copy()
X_val_combined = X_val.copy()

for model_name in top_solution1_models:
    X_train_combined = X_train_combined.astype(str) + ' ' + pd.Series(solution1_models[model_name], index=X_val.index).reindex(X_train.index, fill_value=0).astype(str)
    X_val_combined = X_val_combined.astype(str) + ' ' + pd.Series(solution1_models[model_name], index=X_val.index).astype(str)


model_lr = Pipeline([('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))), # Added some common params
                     ('clf', LogisticRegression(solver='liblinear', C=1.0, random_state=42))]) # Specify solver, C
model_lr.fit(X_train_combined, y_train)
val_predictions_lr = model_lr.predict_proba(X_val_combined)[:, 1]
roc_auc_lr = roc_auc_score(y_val, val_predictions_lr)
print(f"Logistic Regression Validation AUC: {roc_auc_lr}")

# Transform test data
X_test_combined = test_df['Comment'].copy()
for model_name in top_solution1_models:
    X_test_combined = X_test_combined.astype(str) + ' ' + pd.Series(solution1_test_models[model_name]).astype(str)
test_predictions_lr = model_lr.predict_proba(X_test_combined)[:, 1]


# Model 3: Naive Bayes with CountVectorizer
print("Training Naive Bayes...")

# Concatenate Solution 1 predictions to the input text
X_train_combined = X_train.copy()
X_val_combined = X_val.copy()

for model_name in top_solution1_models:
    X_train_combined = X_train_combined.astype(str) + ' ' + pd.Series(solution1_models[model_name], index=X_val.index).reindex(X_train.index, fill_value=0).astype(str)
    X_val_combined = X_val_combined.astype(str) + ' ' + pd.Series(solution1_models[model_name], index=X_val.index).astype(str)

model_nb = Pipeline([('vect', CountVectorizer(max_features=10000, ngram_range=(1, 2))), # Added some common params
                     ('clf', MultinomialNB(alpha=1.0))]) # Specify alpha
model_nb.fit(X_train_combined, y_train)
val_predictions_nb = model_nb.predict_proba(X_val_combined)[:, 1]
roc_auc_nb = roc_auc_score(y_val, val_predictions_nb)
print(f"Naive Bayes Validation AUC: {roc_auc_nb}")

# Transform test data
X_test_combined = test_df['Comment'].copy()
for model_name in top_solution1_models:
    X_test_combined = X_test_combined.astype(str) + ' ' + pd.Series(solution1_test_models[model_name]).astype(str)
test_predictions_nb = model_nb.predict_proba(X_test_combined)[:, 1]


# Model 4: SVM model
print("Training SVM...")

# Concatenate Solution 1 predictions to the input text
X_train_combined = X_train.copy()
X_val_combined = X_val.copy()
X_test_combined_svm = test_df['Comment'].copy() # Store test comments separately

for model_name in top_solution1_models:
    X_train_combined = X_train_combined.astype(str) + ' ' + pd.Series(solution1_models[model_name], index=X_val.index).reindex(X_train.index, fill_value=0).astype(str)
    X_val_combined = X_val_combined.astype(str) + ' ' + pd.Series(solution1_models[model_name], index=X_val.index).astype(str)
    X_test_combined_svm = X_test_combined_svm.astype(str) + ' ' + pd.Series(solution1_test_models[model_name]).astype(str)


tfidf_vectorizer_svm = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf_svm = tfidf_vectorizer_svm.fit_transform(X_train_combined)
X_val_tfidf_svm = tfidf_vectorizer_svm.transform(X_val_combined)
X_test_tfidf_svm_transformed = tfidf_vectorizer_svm.transform(X_test_combined_svm)

model_svm = SVC(probability=True, kernel='linear', C=0.1, random_state=42) # Adjusted C, might need tuning
model_svm.fit(X_train_tfidf_svm, y_train)

val_predictions_svm = model_svm.predict_proba(X_val_tfidf_svm)[:, 1]
roc_auc_svm = roc_auc_score(y_val, val_predictions_svm)
print(f"SVM Validation AUC: {roc_auc_svm}")

test_predictions_svm = model_svm.predict_proba(X_test_tfidf_svm_transformed)[:, 1]


# --- Ensemble Optimization ---
print("Optimizing ensemble weights...")

# Ensure all prediction arrays are numpy arrays
val_predictions_roberta = np.array(val_predictions_roberta)
val_predictions_lr = np.array(val_predictions_lr)
val_predictions_nb = np.array(val_predictions_nb)
val_predictions_svm = np.array(val_predictions_svm)

def ensemble_predictions(weights, preds_list):
    weighted_preds = np.zeros_like(preds_list[0])
    for weight, preds in zip(weights, preds_list):
        weighted_preds += weight * preds
    return weighted_preds

def neg_roc_auc(weights, preds_list, y_true):
    predictions = ensemble_predictions(weights, preds_list)
     # Clip predictions to avoid issues with log loss or perfect separation if AUC is 1.0
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
    try:
        auc = roc_auc_score(y_true, predictions)
    except ValueError: # Handle cases where AUC might be undefined (e.g., all predictions same)
        return 0.5 # Return a neutral score instead of crashing
    return -auc  # Negate for minimization

# List of validation predictions
validation_preds_list = [val_predictions_roberta, val_predictions_lr, val_predictions_nb, val_predictions_svm]

# Initial guess for weights
initial_weights = np.array([0.25] * len(validation_preds_list))

# Constraints for weights (sum to 1)
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
# Bounds for weights (0 to 1)
bounds = [(0, 1)] * len(validation_preds_list)

# Optimization using SLSQP (often better for constrained optimization)
result = minimize(neg_roc_auc, initial_weights, args=(validation_preds_list, y_val),
                  method='SLSQP', bounds=bounds, constraints=constraints) # Changed method to SLSQP

if result.success:
    optimal_weights = result.x
else:
    print(f"Optimization failed: {result.message}. Using equal weights.")
    optimal_weights = initial_weights / initial_weights.sum() # Fallback to equal weights

print(f"Optimal Weights: RoBERTa={optimal_weights[0]:.4f}, LR={optimal_weights[1]:.4f}, NB={optimal_weights[2]:.4f}, SVM={optimal_weights[3]:.4f}")

val_predictions_ensemble = ensemble_predictions(optimal_weights, validation_preds_list)
roc_auc_ensemble = roc_auc_score(y_val, val_predictions_ensemble)
print(f"Ensemble Validation AUC: {roc_auc_ensemble}")


# --- Select Best Model and Make Predictions ---

model_performances = {
    'roberta': roc_auc_roberta,
    'logistic_regression': roc_auc_lr,
    'naive_bayes': roc_auc_nb,
    'svm': roc_auc_svm,
    'ensemble': roc_auc_ensemble
}

# Find the best model based on validation AUC
best_model_name = max(model_performances, key=model_performances.get)
best_roc_auc = model_performances[best_model_name]

print(f"Best Model: {best_model_name} with Validation AUC: {best_roc_auc}")
Final_Validation_Performance = best_roc_auc
print(f'Final Validation Performance: {Final_Validation_Performance}') # Required output format


# Make predictions on test data using the best approach
print(f"Generating test predictions using {best_model_name}...")

# Generate predictions for all models on the test set
# Select final predictions based on best model/ensemble
if best_model_name == 'roberta':
    test_predictions = test_predictions_roberta
elif best_model_name == 'logistic_regression':
    test_predictions = test_predictions_lr
elif best_model_name == 'naive_bayes':
    test_predictions = test_predictions_nb
elif best_model_name == 'svm':
     test_predictions = test_predictions_svm
elif best_model_name == 'ensemble':
    test_preds_list = [test_predictions_roberta, test_predictions_lr, test_predictions_nb, test_predictions_svm]
    test_predictions = ensemble_predictions(optimal_weights, test_preds_list)
else:
    # Fallback in case something unexpected happens
    print("Warning: Best model name not recognized, defaulting to ensemble.")
    test_preds_list = [test_predictions_roberta, test_predictions_lr, test_predictions_nb, test_predictions_svm]
    test_predictions = ensemble_predictions(optimal_weights, test_preds_list)


import os # Needed for directory creation
import pandas as pd # Assuming pandas (pd) is used and defined earlier

# Create submission file DataFrame
# Assuming test_predictions is defined earlier, e.g.:
# test_predictions = model.predict(X_test)
submission_df = pd.DataFrame({'Insult': test_predictions})
# Optionally include original columns if needed by submission format (commented out as in original)
# Assuming test_df is defined earlier if the line below is uncommented, e.g.:
# test_df = pd.read_csv('test.csv')
# submission_df = pd.DataFrame({'Insult': test_predictions, 'Date': test_df['Date'], 'Comment': test_df['Comment']})

# Define the desired output path in the './final' directory
submission_path = "./final/submission.csv"
output_dir = os.path.dirname(submission_path) # Get directory path: ./final

# Create the output directory if it doesn't exist to prevent errors
os.makedirs(output_dir, exist_ok=True)

# Save the submission file to the new path
submission_df[['Insult']].to_csv(submission_path, index=False) # Keep only required column 'Insult'

# Update the print statement to reflect the correct path
print(f"Submission file created at: {submission_path}")
