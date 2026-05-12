#This file right here loads the trained formality classifier, the TF-IDF vectorizer and the NER test sentences
#with that it classifies each sentence as formal/informal and saves the rpedictions 

#This is our bridge between formality classificationa nd NER evaluation



import pandas as pd
import joblib

from scipy.sparse import hstack
from src.features.linguistic_features import extract_linguistic_features
from src.features.ner_features import extract_ner_features

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

test_path = "data/processed/en_ewt-ud-dev.iob2"
sentences, _ = read_iob2(test_path)

texts = [
    " ".join(sentence)
    for sentence in sentences
]

print("Number of sentences:", len(texts))


# CREATE FEATURES --------------------------------

print("Creating features...")

# TF-IDF
print("- Extracting TF-IDF...")
X_tfidf = vectorizer.transform(texts)

# linguistic features
print("- Extracting linguistic features...")
X_linguistic = extract_linguistic_features(texts)

# NER features
print("- Extracting NER features...")
X_ner = extract_ner_features(texts)

# combine
print("Combining features...")
X_features = hstack([
    X_tfidf,
    X_linguistic,
    X_ner
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

print("\nFormality distribution:")

print(
    results_df["predicted_formality"].value_counts()
)

print("Done.")

#The output when u run this

#                                            sentence  predicted_formality
#0                             What is this Miramar ?                    0
#1                     It is a place in Argentina lol                    0
#2  what is a good slogan for an Argentinian resta...                    0
#3  " In Argentina , beef is revered , respected ,...                    0
#4        Come see how we continue this tradition . "                    0

#Formality distribution:
#predicted_formality
#0    1814
#1     263
#Name: count, dtype: int64
#Done.