#Here we have the tokenizer, label encoding, alignment, dataset class, DataLoader, model and trianing loop
#it's a lot but it's actually pretty readable i'd say, nnothing too crazy

#important to mention that we need pytorch for this, it just makes this so much easer --> converts everything into numbers stored in a special strucutre called tensor
# + it's a train engine and a batching system --> really useful for this project

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
print(tokenizer.tokenize("playing football"))


from data_loader import read_iob2


train_path = "data/en_ewt-ud-train.iob2"
sentences, labels = read_iob2(train_path)


dev_path = "data/en_ewt-ud-dev.iob2"
dev_sentences, dev_labels = read_iob2(dev_path)

print("Number of sentences:", len(sentences))
print("First sentence:", sentences[0])
print("First labels:", labels[0])


# to flatten all labels into one list
all_labels = [label for sentence in labels for label in sentence]

# we get unique labels
unique_labels = sorted(list(set(all_labels)))

print("Unique labels:", unique_labels)

label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

print(label2id)

encoded_labels = [
    [label2id[label] for label in sentence]
    for sentence in labels
]

print("Example encoded labels:", encoded_labels[0])

# TOKENIZE YOUR SENTENCES
tokenized_inputs = tokenizer(
    sentences,
    is_split_into_words=True,
    padding=True,
    truncation=True
)

print("Keys:", list(tokenized_inputs.keys())[:5])
print("First input_ids:", tokenized_inputs["input_ids"][0])

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
            # same word split into subwords
            label_ids.append(sentence_labels[word_idx])

        previous_word_idx = word_idx

    aligned_labels.append(label_ids)

print("Tokenized input_ids:", tokenized_inputs["input_ids"][0])
print("Aligned labels:", aligned_labels[0])


import torch

import torch

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

dataset = NERDataset(tokenized_inputs, aligned_labels)
# ===== DEV DATA PROCESSING =====

# encode dev labels
dev_encoded_labels = [
    [label2id[label] for label in sentence]
    for sentence in dev_labels
]

# tokenize dev sentences
dev_tokenized = tokenizer(
    dev_sentences,
    is_split_into_words=True,
    padding=True,
    truncation=True
)

# align dev labels
dev_aligned_labels = []

for i, sentence_labels in enumerate(dev_encoded_labels):
    word_ids = dev_tokenized.word_ids(batch_index=i)

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(sentence_labels[word_idx])
        else:
            label_ids.append(sentence_labels[word_idx])

        previous_word_idx = word_idx

    dev_aligned_labels.append(label_ids)

# create dev dataset + dataloader
dev_dataset = NERDataset(dev_tokenized, dev_aligned_labels)

from torch.utils.data import DataLoader
dev_dataloader = DataLoader(dev_dataset, batch_size=8)


print(len(dataset))
print(dataset[0])

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=8, shuffle=True) #this now gives us 8 sentences at a time

#HERE THE MODEL
#The model makes predictions, compares with the true labels, computes loss and updates itself --> real basic ML model

#It's a BERT + a classifier on top (NN) and it predict s alabel for each token --> BertForTokenClassification
#BERT is a very deep neural network that already learned language form a huge text data so it understands grammar, meaning and context, 
#the calssifier is a small neural network layer

from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)


#optimizer
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

#TRAINING LOOP!!!!

from tqdm import tqdm

model.train()

for epoch in range(1):  # start simple
    print(f"Epoch {epoch}")

    for i, batch in enumerate(tqdm(dataloader)):
        if i > 50:   
            break        
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

model.eval()


# encode dev labels
dev_encoded_labels = [
    [label2id[label] for label in sentence]
    for sentence in dev_labels
]


# tokenize dev
dev_tokenized = tokenizer(
    dev_sentences,
    is_split_into_words=True,
    padding=True,
    truncation=True
)


# align dev labels
dev_aligned_labels = []

for i, sentence_labels in enumerate(dev_encoded_labels):
    word_ids = dev_tokenized.word_ids(batch_index=i)

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(sentence_labels[word_idx])
        else:
            label_ids.append(sentence_labels[word_idx])

        previous_word_idx = word_idx

    dev_aligned_labels.append(label_ids)



#Here's the evaluation -> DEV set

predictions = []
true_labels = []

with torch.no_grad():
    for batch in dev_dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        predictions += preds.tolist()
        true_labels += batch["labels"].tolist()

#now we convert the predicitons back to labels

decoded_preds = [
    [id2label[p] for p in sentence]
    for sentence in predictions
]


#TEST 
test_path = "data/en_ewt-ud-test-masked.iob2"
test_sentences, _ = read_iob2(test_path)


#SAVE PREDICITONS REALLY IMPORTANT!!!!

# clean predictions (remove padding)
clean_preds = []

for sentence_preds, sentence_labels in zip(predictions, true_labels):
    clean_sentence = []

    for pred, label in zip(sentence_preds, sentence_labels):
        if label != -100:
            clean_sentence.append(id2label[pred])

    clean_preds.append(clean_sentence)

def save_predictions(sentences, predictions, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for sentence, preds in zip(sentences, predictions):
            for i, (word, label) in enumerate(zip(sentence, preds)):
                f.write(f"{i+1}\t{word}\t{label}\t-\t-\n")
            f.write("\n")

            
save_predictions(dev_sentences, clean_preds, "dev_predictions.iob2")