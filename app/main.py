from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .preprocessing import clean_text, spacy_doc
from .style_features import extract_style_features
from .fake_model import predict_fake
from .source_model import predict_source
from .impersonation import check_impersonation

app = FastAPI(title="Fake News & Source Impersonation API")

# CORS configuration (required for React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsRequest(BaseModel):
    title: str
    content: str
    claimed_source: str

@app.post("/analyze")
def analyze_news(req: NewsRequest):
    # Combine title and content for fake news detection
    full_text = req.title + " " + req.content
    cleaned = clean_text(full_text)

    # Style analysis uses content only
    doc = spacy_doc(req.content)
    style_vector = extract_style_features(doc)

    # Predictions
    fake_result = predict_fake(cleaned)
    source_result = predict_source(style_vector)

    # Impersonation check
    impersonation = check_impersonation(
        source_result["predicted_source"],
        req.claimed_source,
        source_result["confidence"]
    )

    return {
        "fake_news": fake_result,
        "style_analysis": source_result,
        "impersonation_detected": impersonation,
        "claimed_source": req.claimed_source
    }
