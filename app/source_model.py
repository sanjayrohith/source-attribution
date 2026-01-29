import joblib
import numpy as np

source_model = joblib.load("models/source_classifier.pkl")

def predict_source(style_vector: np.ndarray):
    probs = source_model.predict_proba([style_vector])[0]
    idx = probs.argmax()

    return {
        "predicted_source": source_model.classes_[idx],
        "confidence": round(float(probs[idx]), 3)
    }
