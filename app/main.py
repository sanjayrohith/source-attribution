from fastapi import FastAPI
from pydantic import BaseModel

from .preprocessing import clean_text, spacy_doc
from .style_features import extract_style_features
from .fake_model import predict_fake
from .source_model import predict_source
from .impersonation import check_impersonation

app = FastAPI(title="Fake News & Source Impersonation API")

class NewsRequest(BaseModel):
    title: str
    content: str
    claimed_source: str

@app.post("/analyze")
def analyze_news(req: NewsRequest):
    full_text = req.title + " " + req.content
    cleaned = clean_text(full_text)

    doc = spacy_doc(req.content)
    style_vector = extract_style_features(doc)

    fake_result = predict_fake(cleaned)
    source_result = predict_source(style_vector)

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
