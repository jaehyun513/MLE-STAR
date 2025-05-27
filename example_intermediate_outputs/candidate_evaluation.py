
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import numpy as np

# Define the model name
model_name = 'roberta-base'

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=6)
model.to(device)

# Load the training data
train_df = pd.read_csv('./input/train.csv')

# Subsample the training data if it's too large
if len(train_df) > 10000:
    train_df = train_df.sample(n=10000, random_state=42)

# Define the labels
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['comment_text'], train_df[labels], test_size=0.2, random_state=42
)

# Define a custom dataset
class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts.tolist()
        self.labels = labels.values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Create the datasets
train_dataset = ToxicityDataset(train_texts, train_labels, tokenizer)
val_dataset = ToxicityDataset(val_texts, val_labels, tokenizer)

# Create the dataloaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete.")

# Evaluation
model.eval()
val_preds = []
val_targets = []
with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        val_preds.extend(probs)
        val_targets.extend(labels.cpu().numpy())

# Calculate the ROC AUC score for each column
val_preds = np.array(val_preds)
val_targets = np.array(val_targets)
auc_scores = []
for i in range(val_targets.shape[1]):
    auc = roc_auc_score(val_targets[:, i], val_preds[:, i])
    auc_scores.append(auc)

# Calculate the mean column-wise ROC AUC
mean_auc = np.mean(auc_scores)

print(f'Final Validation Performance: {mean_auc}')
