#This file right here loads the trained formality classifier, the TF-IDF vectorizer and the NER test sentences
#with that it classifies each sentence as formal/informal and saves the rpedictions 

#This is our bridge between formality classificationa nd NER evaluation



import pandas as pd
import joblib

from scipy.sparse import hstack
from src.features.linguistic_features import extract_linguistic_features

from src.ner.data_loader import read_iob2


# LOAD TRAINED CLASSIFIER --------------------------------

print("Loading classifier...")

model = joblib.load(
    "outputs/models/formality_classifier.pkl"
)

vectorizer = joblib.load(
    "outputs/models/tfidf_vectorizer.pkl"
)


# LOAD NER TEST SENTENCES --------------------------------

print("Loading NER sentences...")

test_path = "data/processed/en_ewt-ud-test-masked.iob2"

sentences, _ = read_iob2(test_path)

texts = [
    " ".join(sentence)
    for sentence in sentences
]

print("Number of sentences:", len(texts))


# CREATE FEATURES --------------------------------

print("Creating features...")

# TF-IDF
X_tfidf = vectorizer.transform(texts)

# linguistic features
X_linguistic = extract_linguistic_features(texts)

# combine
X_features = hstack([
    X_tfidf,
    X_linguistic
])


# PREDICT FORMALITY --------------------------------

print("Predicting formality...")

predictions = model.predict(X_features)


# SAVE RESULTS --------------------------------

results_df = pd.DataFrame({
    "sentence": texts,
    "predicted_formality": predictions
})

results_df.to_csv(
    "outputs/results/ner_sentence_formality.csv",
    index=False
)

print(results_df.head())

print("Done.")