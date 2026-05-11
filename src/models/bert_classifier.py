#here i'm coding a BERT model --> already explained them i'm pretty sure, we used the same for the first submission's baseline

#might have to run this in your terminal: pip install transformers torch scikit-learn pandas

import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from torch.utils.data import Dataset


# LOAD DATASET --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv("data/processed/formality_dataset.csv")

print(df.head())
print("Dataset size:", len(df))


# TRAIN / DEV / TEST SPLIT --------------------------------------

X = df["text"]
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

print("Train size:", len(X_train))
print("Dev size:", len(X_dev))
print("Test size:", len(X_test))


# TOKENIZER -----------------------------------------------------

print("Loading tokenizer...")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


train_encodings = tokenizer(
    list(X_train),
    truncation=True,
    padding=True
)

dev_encodings = tokenizer(
    list(X_dev),
    truncation=True,
    padding=True
)

test_encodings = tokenizer(
    list(X_test),
    truncation=True,
    padding=True
)


# DATASET CLASS -------------------------------------------------

class FormalityDataset(Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):

        return len(self.labels)


train_dataset = FormalityDataset(
    train_encodings,
    list(y_train)
)

dev_dataset = FormalityDataset(
    dev_encodings,
    list(y_dev)
)

test_dataset = FormalityDataset(
    test_encodings,
    list(y_test)
)


# MODEL ---------------------------------------------------------

print("Loading BERT model...")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)


# METRICS -------------------------------------------------------

def compute_metrics(pred):

    labels = pred.label_ids

    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# TRAINING ARGUMENTS --------------------------------------------

training_args = TrainingArguments(
    output_dir="outputs/models/bert_formality",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="outputs/logs",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8
)


# TRAINER -------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)


# TRAIN ---------------------------------------------------------

print("Starting training...")

trainer.train()


# EVALUATE ------------------------------------------------------

print("Evaluating on TEST set...")

results = trainer.evaluate(test_dataset)

print(results)


# SAVE RESULTS --------------------------------------------------

with open("outputs/results/bert_results.txt", "w") as f:

    for key, value in results.items():

        f.write(f"{key}: {value}\n")

print("Results saved.")