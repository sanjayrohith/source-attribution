import pandas as pd
import numpy as np
import spacy
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from app.style_features import extract_style_features

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Load dataset (JSON Lines!)
df = pd.read_json(
    "data/news_category/News_Category_Dataset_v3.json",
    lines=True
)

# Select categories to use
selected_categories = [
    "POLITICS",
    "WORLD NEWS",
    "BUSINESS",
    "TECH",
    "ENTERTAINMENT"
]

df = df[df["category"].isin(selected_categories)]

# Combine text fields
df["content"] = (
    df["headline"].fillna("") + " " +
    df["short_description"].fillna("")
)

print("Total samples:", len(df))

X = []
y = []

# Extract style features
for _, row in df.iterrows():
    doc = nlp(row["content"])
    features = extract_style_features(doc)
    X.append(features)
    y.append(row["category"])

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/source_classifier.pkl")

print("Style-based category model saved")
