
from datasets import load_dataset
import pandas as pd
import random
import re




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

def clean_tweet(text):

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove hashtags
    text = re.sub(r"#\w+", "", text)

    # remove mentions
    text = re.sub(r"@\w+", "", text)

    # remove emojis/non-ascii
    text = text.encode("ascii", "ignore").decode()

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("Loading Informal Tweets (tweet_eval)...")

# Using tweet_eval sentiment as a proxy for informal text
# This is a widely used dataset for twitter-related tasks
tweets_ds = load_dataset("tweet_eval", "sentiment")

informal_texts = []

for text in tweets_ds["train"]["text"]:

    # remove weird spacing between characters
    text = clean_tweet(text)

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