import numpy as np
import string
import re


def extract_linguistic_features(texts):

    features = []

    contractions = [
        "n't", "'re", "'s", "'m", "'ll", "'ve", "'d"
    ]

    for text in texts:

        words = text.split()

        # avoid division by zero
        if len(words) == 0:
            words = [""]

        # average sentence length
        sentence_length = len(words)

        # average word length
        avg_word_length = np.mean([
            len(word) for word in words
        ])

        # punctuation count
        punctuation_count = sum(
            1 for char in text
            if char in string.punctuation
        )

        # uppercase ratio
        uppercase_ratio = sum(
            1 for char in text
            if char.isupper()
        ) / max(len(text), 1)

        # contraction count
        contraction_count = sum(
            text.count(c)
            for c in contractions
        )

        # lexical diversity
        lexical_diversity = len(set(words)) / len(words)

        features.append([
            sentence_length,
            avg_word_length,
            punctuation_count,
            uppercase_ratio,
            contraction_count,
            lexical_diversity
        ])

    return np.array(features)