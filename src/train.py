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