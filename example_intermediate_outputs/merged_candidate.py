
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, BertForSequenceClassification
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import numpy as np

# Define the models name
model_name_roberta = 'roberta-base'
model_name_bert = 'bert-base-uncased'

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the tokenizers and models
tokenizer_roberta = RobertaTokenizer.from_pretrained(model_name_roberta)
model_roberta = RobertaForSequenceClassification.from_pretrained(model_name_roberta, num_labels=6)
model_roberta.to(device)

tokenizer_bert = BertTokenizer.from_pretrained(model_name_bert)
model_bert = BertForSequenceClassification.from_pretrained(model_name_bert, num_labels=6)
model_bert.to(device)

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
max_length = 128
train_dataset_roberta = ToxicityDataset(train_texts, train_labels, tokenizer_roberta, max_length)
val_dataset_roberta = ToxicityDataset(val_texts, val_labels, tokenizer_roberta, max_length)

train_dataset_bert = ToxicityDataset(train_texts, train_labels, tokenizer_bert, max_length)
val_dataset_bert = ToxicityDataset(val_texts, val_labels, tokenizer_bert, max_length)

# Create the dataloaders
batch_size = 16
train_dataloader_roberta = DataLoader(train_dataset_roberta, batch_size=batch_size, shuffle=True)
val_dataloader_roberta = DataLoader(val_dataset_roberta, batch_size=batch_size)

train_dataloader_bert = DataLoader(train_dataset_bert, batch_size=batch_size, shuffle=True)
val_dataloader_bert = DataLoader(val_dataset_bert, batch_size=batch_size)

# Define the optimizers
optimizer_roberta = torch.optim.AdamW(model_roberta.parameters(), lr=1e-5)
optimizer_bert = torch.optim.AdamW(model_bert.parameters(), lr=2e-5)

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    model_roberta.train()
    for batch in train_dataloader_roberta:
        optimizer_roberta.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model_roberta(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer_roberta.step()
    print(f"Roberta Epoch {epoch+1} complete.")

    model_bert.train()
    for batch in train_dataloader_bert:
        optimizer_bert.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model_bert(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer_bert.step()
    print(f"Bert Epoch {epoch+1} complete.")

# Evaluation
model_roberta.eval()
val_preds_roberta = []
val_targets_roberta = []
with torch.no_grad():
    for batch in val_dataloader_roberta:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model_roberta(input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        val_preds_roberta.extend(probs)
        val_targets_roberta.extend(labels.cpu().numpy())

model_bert.eval()
val_preds_bert = []
val_targets_bert = []
with torch.no_grad():
    for batch in val_dataloader_bert:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model_bert(input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        val_preds_bert.extend(probs)
        val_targets_bert.extend(labels.cpu().numpy())

# Ensemble the predictions
val_preds_roberta = np.array(val_preds_roberta)
val_preds_bert = np.array(val_preds_bert)
val_preds = (val_preds_roberta + val_preds_bert) / 2

# Calculate the ROC AUC score for each column
val_targets = np.array(val_targets_roberta)
auc_scores = []
for i in range(val_targets.shape[1]):
    auc = roc_auc_score(val_targets[:, i], val_preds[:, i])
    auc_scores.append(auc)

# Calculate the mean column-wise ROC AUC
mean_auc = np.mean(auc_scores)

print(f'Final Validation Performance: {mean_auc}')
