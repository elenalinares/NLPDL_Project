from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
print(tokenizer.tokenize("playing football"))


from data_loader import read_iob2

train_path = "data/en_ewt-ud-train.iob2"

sentences, labels = read_iob2(train_path)

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

print("Keys:", list(tokenized_inputs.keys())[:10])
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