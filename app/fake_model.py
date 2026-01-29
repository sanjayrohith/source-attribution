import joblib

fake_model = joblib.load("models/fake_news.pkl")
tfidf = joblib.load("models/tfidf.pkl")

def predict_fake(text: str):
    vector = tfidf.transform([text])
    prob = fake_model.predict_proba(vector)[0][1]
    label = "Fake" if prob > 0.5 else "Real"

    return {
        "label": label,
        "confidence": round(float(prob), 3)
    }
