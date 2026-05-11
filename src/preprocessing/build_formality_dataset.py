####  pip install datasets pandas You maybe have to install this, so if thats the case just run this in your terminal

from datasets import load_dataset
import pandas as pd
import random


# LOAD WIKITEXT (FORMAL) ---------------------------------------------
# This dataset is great bc it's:
#   + clean english
#   + formal-ish encyclopedia style
#   + widely used in NLP world
#   + easy to load
#   + already split into train/validation/test



print("Loading WikiText...")

wiki = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1"
)

formal_texts = []

for text in wiki["train"]["text"]:

    text = text.strip()

    # remove empty lines
    if len(text) == 0:
        continue

    # remove weird formatting lines
    if text.startswith("="):
        continue

    formal_texts.append(text)

print("Number of formal examples:", len(formal_texts))



# LOAD TWEEBANK (INFORMAL) ---------------------------------------------


print("Loading Tweebank...")

tweebank = load_dataset("tweet_eval", "emotion")

informal_texts = []

for sentence in tweebank["train"]["text"]:

    text = " ".join(sentence)

    text = text.strip()

    if len(text) == 0:
        continue

    informal_texts.append(text)

print("Number of informal examples:", len(informal_texts))



# BALANCE DATASET ---------------------------------------------

# take same amount from both datasets
min_size = min(len(formal_texts), len(informal_texts))

formal_texts = formal_texts[:min_size]
informal_texts = informal_texts[:min_size]

print("Balanced size:", min_size)



# CREATE LABELS ---------------------------------------------

data = []

# formal = 1
for text in formal_texts:
    data.append({
        "text": text,
        "label": 1
    })

# informal = 0
for text in informal_texts:
    data.append({
        "text": text,
        "label": 0
    })


# shuffle dataset
random.shuffle(data)


# CREATE DATAFRAME ---------------------------------------------



df = pd.DataFrame(data)

print(df.head())

print("Dataset size:", len(df))



# SAVE DATASET ---------------------------------------------


output_path = "data/processed/formality_dataset.csv"

df.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")