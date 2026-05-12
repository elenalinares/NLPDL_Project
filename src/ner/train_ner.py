# Here we have the tokenizer, label encoding, alignment, dataset class, DataLoader, model and training loop
# it's a lot but it's actually pretty readable i'd say, nothing too crazy

# important to mention that we need pytorch for this, it just makes this so much easier --> converts everything into numbers stored in a special structure called tensor
# + it's a train engine and a batching system --> really useful for this project

import os
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data_loader import read_iob2

# Utilize the GPU (supporting CUDA for NVIDIA, MPS for Apple M-Series, and CPU fallback)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

def align_labels(tokenized_inputs, encoded_labels):
    """Helper function to align labels with subword tokens."""
    aligned_labels = []
    for i, sentence_labels in enumerate(encoded_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # ignore padding & special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(sentence_labels[word_idx])
            else:
                # same word split into subwords - assign same label
                label_ids.append(sentence_labels[word_idx])
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    return aligned_labels

class NERDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def save_predictions(sentences, predictions, output_file):
    """Helper function to save predictions back into standard IOB2 format"""
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence, preds in zip(sentences, predictions):
            for i, (word, label) in enumerate(zip(sentence, preds)):
                f.write(f"{i+1}\t{word}\t{label}\t-\t-\n")
            f.write("\n")

def main():
    # Load and process training data
    train_path = "data/processed/en_ewt-ud-train.iob2"
    sentences, labels = read_iob2(train_path)

    all_labels = [label for sentence in labels for label in sentence]
    unique_labels = sorted(list(set(all_labels)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    encoded_labels = [[label2id[label] for label in sentence] for sentence in labels]
    
    # max_length=128 added to prevent memory bottleneck
    tokenized_inputs = tokenizer(sentences, is_split_into_words=True, padding=True, truncation=True, max_length=128)
    aligned_labels = align_labels(tokenized_inputs, encoded_labels)
    dataset = NERDataset(tokenized_inputs, aligned_labels)
    
    # Train batch size MUST be small (8) to prevent OOM errors and SSD swapping
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Load and process dev data
    dev_path = "data/processed/en_ewt-ud-dev.iob2"
    dev_sentences, dev_labels = read_iob2(dev_path)
    dev_encoded_labels = [[label2id[label] for label in sentence] for sentence in dev_labels]
    dev_tokenized = tokenizer(dev_sentences, is_split_into_words=True, padding=True, truncation=True, max_length=128)
    dev_aligned_labels = align_labels(dev_tokenized, dev_encoded_labels)
    dev_dataset = NERDataset(dev_tokenized, dev_aligned_labels)
    
    # Eval batch size can be larger (32) since gradients aren't tracked
    dev_dataloader = DataLoader(dev_dataset, batch_size=32)

    # Load and process test data
    test_path = "data/processed/en_ewt-ud-test-masked.iob2"
    test_sentences, _ = read_iob2(test_path)
    test_tokenized = tokenizer(test_sentences, is_split_into_words=True, padding=True, truncation=True, max_length=128)
    test_dataset = NERDataset(test_tokenized)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    # Initialize model
    model = BertForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # TRAINING LOOP ----------------------------------------------------
    model.train()
    for epoch in range(3):
        print(f"\nEpoch {epoch}")
        total_loss = 0
        
        for batch in tqdm(dataloader):
            # 1. Use set_to_none=True to completely delete old gradients
            optimizer.zero_grad(set_to_none=True)
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            # 2. Forcefully delete the tensors from Python's memory
            del input_ids, attention_mask, batch_labels, outputs, loss 
            
            # 3. Force the GPU to instantly empty its cache safely
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    # EVALUATION ON DEV SET --------------------------------------------
    model.eval()
    print("\nEvaluating Dev Set...")
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dev_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            predictions += preds.tolist()
            true_labels += batch["labels"].tolist()

    # Clean dev predictions
    clean_preds = []
    for sentence_preds, sentence_labels in zip(predictions, true_labels):
        clean_sentence = []
        for pred, label in zip(sentence_preds, sentence_labels):
            if label != -100:
                clean_sentence.append(id2label[pred])
        clean_preds.append(clean_sentence)

    save_predictions(dev_sentences, clean_preds, "outputs/predictions/dev_predictions.iob2")
    print("Dev predictions saved to outputs/predictions/dev_predictions.iob2")

    # TEST PREDICTIONS -------------------------------------------------
    print("Generating Test Predictions...")
    test_predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).tolist()
            test_predictions += preds

    # Clean test predictions using word_ids for better accuracy
    clean_test_preds = []
    for i, (sentence, preds) in enumerate(zip(test_sentences, test_predictions)):
        word_ids = test_tokenized.word_ids(batch_index=i)
        clean_sentence = []
        previous_word_idx = None
        for j, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                # We take the prediction of the first subword of each word
                clean_sentence.append(id2label[preds[j]])
            previous_word_idx = word_idx
        clean_test_preds.append(clean_sentence)

    save_predictions(test_sentences, clean_test_preds, "outputs/predictions/test_predictions.iob2")
    print("Test predictions saved to outputs/predictions/test_predictions.iob2")

    # SAVE MODEL AND TOKENIZER -----------------------------------------
    os.makedirs("outputs/models/ner_model", exist_ok=True)
    model.save_pretrained("outputs/models/ner_model")
    tokenizer.save_pretrained("outputs/models/ner_model")
    print("Model and tokenizer saved successfully.")

    # RESULT ANALYSIS:
    # recall: 0.232, precision: 0.406, slot-f1: 0.296

if __name__ == "__main__":
    main()