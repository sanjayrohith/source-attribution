import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load datasets
fake_df = pd.read_csv("data/fake_news/Fake.csv")
true_df = pd.read_csv("data/fake_news/True.csv")

fake_df["label"] = 1
true_df["label"] = 0

df = pd.concat([fake_df, true_df])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title and text
df["content"] = df["title"] + " " + df["text"]

X = df["content"]
y = df["label"]

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_vec = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fake_news.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")

print("Fake news model saved")