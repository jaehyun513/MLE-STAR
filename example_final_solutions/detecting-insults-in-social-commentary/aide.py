import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
import numpy as np
from datasets import Dataset


# Load the datasets
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")

# Handle missing values
train_df["Comment"].fillna("", inplace=True)
test_df["Comment"].fillna("", inplace=True)

# Define the model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)


# Prepare data for the Hugging Face Dataset format
def prepare_dataset(df, is_train=True):
    dataset = df[["Comment"]].copy()
    if is_train and "Insult" in df.columns:
        dataset["labels"] = df["Insult"].astype(int)
    else:
        dataset["labels"] = 0  # Dummy labels, will not be used for test prediction
    dataset = dataset.rename(columns={"Comment": "text"})
    dataset = dataset.to_dict(orient="list")
    dataset = {key: dataset[key] for key in ["text", "labels"]}
    dataset = Dataset.from_dict(dataset)
    return dataset


train_dataset = prepare_dataset(train_df)
test_dataset = prepare_dataset(test_df)


# Tokenize the text
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    try:
        auc = roc_auc_score(labels, probabilities[:, 1])
    except ValueError:
        auc = 0.5  # Return 0.5 if only one class is present
    return {"auc": auc}


# Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
test_predictions = np.zeros(len(test_df))

for fold, (train_index, val_index) in enumerate(kf.split(train_df, train_df["Insult"])):
    print(f"Fold {fold + 1}")

    # Split the training data into training and validation sets
    train_fold = train_df.iloc[train_index]
    val_fold = train_df.iloc[val_index]

    train_dataset_fold = prepare_dataset(train_fold)
    val_dataset_fold = prepare_dataset(val_fold)

    tokenized_train_dataset_fold = train_dataset_fold.map(
        tokenize_function, batched=True
    )
    tokenized_val_dataset_fold = val_dataset_fold.map(tokenize_function, batched=True)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=f"./results_fold_{fold}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="auc",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=1,  # Only keep the best checkpoint
        fp16=True,  # Enable mixed precision training for faster training
    )

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset_fold,
        eval_dataset=tokenized_val_dataset_fold,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate on the validation set
    val_predictions = trainer.predict(tokenized_val_dataset_fold)
    val_probabilities = torch.nn.functional.softmax(
        torch.tensor(val_predictions.predictions), dim=-1
    ).numpy()
    val_auc = roc_auc_score(val_fold["Insult"], val_probabilities[:, 1])
    aucs.append(val_auc)
    print(f"Fold {fold + 1} AUC: {val_auc}")

    # Make predictions on the test set
    test_predictions_fold = trainer.predict(tokenized_test_dataset)
    test_probabilities = torch.nn.functional.softmax(
        torch.tensor(test_predictions_fold.predictions), dim=-1
    ).numpy()
    test_predictions += test_probabilities[:, 1] / kf.get_n_splits()

print(f"Average AUC: {np.mean(aucs)}")

# Create the submission file
submission = pd.read_csv("input/sample_submission_null.csv")
submission["Insult"] = test_predictions
submission.to_csv("submission/submission.csv", index=False)

print("Submission file created successfully!")
