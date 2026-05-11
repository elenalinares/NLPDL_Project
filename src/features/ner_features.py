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

#BEWARE: This file takes forever to run so run it at your own risk

#just so that you don't have to run it here's the results i got:

#---------------------------------------------------------------------------------------------------------
# FEATURE_SET = "tfidf"
#---------------------------------------------------------------------------------------------------------

#Evaluating on DEV set...

#DEV RESULTS
#Accuracy: 0.9231950844854071
#Precision: 0.9607843137254902
#Recall: 0.8855421686746988
#F1: 0.9216300940438872

#Evaluating on TEST set...

#TEST RESULTS
#Accuracy: 0.9493865030674846
#Precision: 0.9602446483180428
#Recall: 0.9401197604790419
#F1: 0.9500756429652042

#Classification Report:

#              precision    recall  f1-score   support
#
#           0       0.94      0.96      0.95       318
#           1       0.96      0.94      0.95       334

#    accuracy                           0.95       652
#   macro avg       0.95      0.95      0.95       652
#weighted avg       0.95      0.95      0.95       652

#Predictions saved.
#Classifier saved.

#---------------------------------------------------------------------------------------------------------
# FEATURE_SET = "linguistic"
#---------------------------------------------------------------------------------------------------------

#Evaluating on DEV set...

#DEV RESULTS
#Accuracy: 0.8986175115207373
#Precision: 0.9554794520547946
#Recall: 0.8403614457831325
#F1: 0.8942307692307693

#Evaluating on TEST set...

#TEST RESULTS
#Accuracy: 0.9125766871165644
#Precision: 0.9694915254237289
#Recall: 0.8562874251497006
#F1: 0.9093799682034976

#Classification Report:

#              precision    recall  f1-score   support

#           0       0.87      0.97      0.92       318
#           1       0.97      0.86      0.91       334

#    accuracy                           0.91       652
#   macro avg       0.92      0.91      0.91       652
#weighted avg       0.92      0.91      0.91       652

#Predictions saved.
#Classifier saved.

#---------------------------------------------------------------------------------------------------------
# FEATURE_SET = "ner"
#---------------------------------------------------------------------------------------------------------

#Evaluating on DEV set...

#DEV RESULTS
#Accuracy: 0.8064516129032258
#Precision: 0.912
#Recall: 0.6867469879518072
#F1: 0.7835051546391752

#Evaluating on TEST set...

#TEST RESULTS
#Accuracy: 0.8067484662576687
#Precision: 0.9
#Recall: 0.7005988023952096
#F1: 0.7878787878787878

#Classification Report:

#              precision    recall  f1-score   support

#           0       0.74      0.92      0.82       318
#           1       0.90      0.70      0.79       334

#    accuracy                           0.81       652
#   macro avg       0.82      0.81      0.81       652
#weighted avg       0.82      0.81      0.80       652

#Predictions saved.
#Classifier saved.