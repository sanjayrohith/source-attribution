import numpy as np
from textblob import TextBlob

def extract_style_features(doc):
    sentences = list(doc.sents)
    words = [t for t in doc if t.is_alpha]

    if len(sentences) == 0 or len(words) == 0:
        return np.zeros(9)

    sent_lengths = [len([t for t in s if t.is_alpha]) for s in sentences]

    avg_sent_len = np.mean(sent_lengths)
    std_sent_len = np.std(sent_lengths)

    pos_counts = {
        "NOUN": 0,
        "VERB": 0,
        "ADJ": 0,
        "ADV": 0,
        "PRON": 0
    }

    for token in words:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1

    total_words = len(words)
    pos_ratios = [pos_counts[p] / total_words for p in pos_counts]

    polarity = TextBlob(doc.text).sentiment.polarity
    subjectivity = TextBlob(doc.text).sentiment.subjectivity

    features = [
        avg_sent_len,
        std_sent_len,
        polarity,
        subjectivity,
        *pos_ratios
    ]

    return np.array(features)
