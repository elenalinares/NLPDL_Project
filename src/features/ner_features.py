import numpy as np
import torch

from transformers import (
    BertTokenizerFast,
    BertForTokenClassification
)


# LOAD TRAINED NER MODEL ----------------------------

print("Loading trained NER model...")

tokenizer = BertTokenizerFast.from_pretrained(
    "outputs/models/ner_model"
)

model = BertForTokenClassification.from_pretrained(
    "outputs/models/ner_model"
)

model.eval()


# EXTRACT NER FEATURES ----------------------------

def extract_ner_features(texts):

    features = []

    for text in texts:

        words = text.split()

        # tokenize
        inputs = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True
        )

        # predict
        with torch.no_grad():

            outputs = model(**inputs)

        predictions = torch.argmax(
            outputs.logits,
            dim=-1
        )

        pred_ids = predictions[0].tolist()

        labels = [
            model.config.id2label[p]
            for p in pred_ids
        ]

        # remove special tokens
        labels = labels[1:len(words)+1]

        # entity count
        entity_count = sum(
            1 for label in labels
            if label != "O"
        )

        # PERSON entities
        person_count = sum(
            1 for label in labels
            if "PER" in label
        )

        # ORG entities
        org_count = sum(
            1 for label in labels
            if "ORG" in label
        )

        # LOC entities
        loc_count = sum(
            1 for label in labels
            if "LOC" in label
        )

        # entity density
        entity_density = entity_count / max(len(words), 1)

        # unique entity labels
        unique_entities = len(set([
            label for label in labels
            if label != "O"
        ]))

        features.append([
            entity_count,
            person_count,
            org_count,
            loc_count,
            entity_density,
            unique_entities
        ])

    return np.array(features)