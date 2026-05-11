#here we just evaluate by formality the predictions we did

import pandas as pd
import subprocess


# LOAD FORMALITY LABELS --------------------------------

print("Loading formality predictions...")

formality_df = pd.read_csv(
    "outputs/results/ner_sentence_formality.csv"
)

formality_labels = formality_df[
    "predicted_formality"
].tolist()


# READ IOB2 FILES --------------------------------

def read_iob2_sentences(filepath):

    sentences = []

    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line == "":

                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []

            else:
                current_sentence.append(line)

    return sentences


# LOAD GOLD + PREDICTIONS -----------------------------

gold_sentences = read_iob2_sentences(
    "data/processed/en_ewt-ud-dev.iob2"
)

pred_sentences = read_iob2_sentences(
    "outputs/predictions/dev_predictions.iob2"
)


print("Gold sentences:", len(gold_sentences))

print("Prediction sentences:", len(pred_sentences))

print("Formality labels:", len(formality_labels))


# SPLIT DATA --------------------------------

formal_gold = []
formal_pred = []

informal_gold = []
informal_pred = []


for label, gold, pred in zip(
    formality_labels,
    gold_sentences,
    pred_sentences
):

    if label == 1:

        formal_gold.append(gold)
        formal_pred.append(pred)

    else:

        informal_gold.append(gold)
        informal_pred.append(pred)


# SAVE SPLIT FILES --------------------------------

def save_iob2(sentences, filepath):

    with open(filepath, "w", encoding="utf-8") as f:

        for sentence in sentences:

            for line in sentence:
                f.write(line + "\n")

            f.write("\n")


save_iob2(
    formal_gold,
    "outputs/results/formal_gold.iob2"
)

save_iob2(
    formal_pred,
    "outputs/results/formal_pred.iob2"
)

save_iob2(
    informal_gold,
    "outputs/results/informal_gold.iob2"
)

save_iob2(
    informal_pred,
    "outputs/results/informal_pred.iob2"
)


print("\nSaved split datasets.")

print("Formal sentences:", len(formal_gold))

print("Informal sentences:", len(informal_gold))


#After running f1 in our predictions - formal and informal, we get that the resutls are actually opposite of the original hypothesis

#The NER performs better onthe informal subset:
#   + formal F1 ~ 0.21
#   + informal F1 ~ 0.36

#This still makes sense when you thingk aobut the data as en_ext is mostly web/conversational text so our model prolly understand short conersational sentences as informal and longer descriptive sentences as formal

#However the NER model was trained on this same domain, so hte model is naturally better adapted to controversional web anguage, short QA-like senteces and casual syntax
# and struggles more on longer, dense rand more edescriptive sentences especially becasue the baseline mdoel is still relatively weak overall



#--------------------------------------------------------------------------------------------------------------------------------------------------------
#The NER model demonstrated higher robustness on conversational/informal web text than on more formal descriptive text within the EWT domain.
#--------------------------------------------------------------------------------------------------------------------------------------------------------