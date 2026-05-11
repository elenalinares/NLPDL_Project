# for this file to run u may need to run this in your terminal: pip install scikit-learn

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# load dataset ----------------------------------------
print('Loading dataset...')

df = pd.read_csv('data/processed/formality_dataset.csv')
print(df.head())

print(f'Dataset size: {len(df)}')


# inputs and labels -----------------------------------

X = df['text']

y = df['label']

#train / dev / test split ------------------------------

# first split: train + temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# second split: dev + test
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)
print("Train size:", len(X_train))
print("Dev size:", len(X_dev))
print("Test size:", len(X_test))

# TF-IDF vectorization

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_dev_tfidf = vectorizer.transform(X_dev)

X_test_tfidf = vectorizer.transform(X_test)


# MODEL ----------------------------------

print("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

# dev evaluation ------------------------------------

print("\nEvaluating on DEV set...")

dev_preds = model.predict(X_dev_tfidf)

accuracy = accuracy_score(y_dev, dev_preds)

precision = precision_score(y_dev, dev_preds)

recall = recall_score(y_dev, dev_preds)

f1 = f1_score(y_dev, dev_preds)

print("\nDEV RESULTS")

print("Accuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)

print("F1:", f1)


# test evaluation ------------------------------------

print("\nEvaluating on TEST set...")

test_preds = model.predict(X_test_tfidf)

test_accuracy = accuracy_score(y_test, test_preds)

test_precision = precision_score(y_test, test_preds)

test_recall = recall_score(y_test, test_preds)

test_f1 = f1_score(y_test, test_preds)

print("\nTEST RESULTS")

print("Accuracy:", test_accuracy)

print("Precision:", test_precision)

print("Recall:", test_recall)

print("F1:", test_f1)



# ---------------------------------------------------
# FULL CLASSIFICATION REPORT
# ---------------------------------------------------

print("\nClassification Report:\n")

print(classification_report(y_test, test_preds))

#Ok, some obersvations here

# In the baselien classifier we've vectorized text using TF-IDF, trained Logistic Regression and evaluatied on dev/test splits
# This is a very standard NLP baseline setup 

# TF-IDF is a way to convert text intoi numbers so our models can understand language. In our LR model we use it for this reason
# The whole thing means: Term Frequency-Inverse Documen Frequency



#These are the results we get:
#Classification Report:

#              precision    recall  f1-score   support

#           0       0.92      1.00      0.96       328
#           1       1.00      0.91      0.95       324

#    accuracy                           0.96       652
#   macro avg       0.96      0.96      0.96       652
#weighted avg       0.96      0.96      0.96       652

#Also, we can see that the inromal dataset has character spacin issues, this prolly makes classification easier than it should be 
# --> The model would learn that the spacing means inromal

