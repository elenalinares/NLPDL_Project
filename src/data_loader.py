def read_iob2(file_path):
    sentences = []
    labels = []

    current_sentence = []
    current_labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []

            elif line.startswith("#"):
                continue

            else:
                parts = line.split("\t")

                if len(parts) < 3:
                    continue

                word = parts[1]
                tag = parts[2]

                current_sentence.append(word)
                current_labels.append(tag)

    #print(f'Sentences: {sentences[:10]}\nLabels: {labels[10]}') #to see what we have created and like how our data loosk like
    return sentences, labels