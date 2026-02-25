from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .preprocessing import clean_text, spacy_doc
from .style_features import extract_style_features
from .fake_model import predict_fake
from .source_model import predict_source
from .impersonation import check_impersonation
from .scraper import scrape_verify
from .headlines import fetch_headlines

app = FastAPI(title="Fake News & Source Impersonation API")

# CORS configuration 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsRequest(BaseModel):
    title: str
    content: str
    claimed_source: str

class ScrapeRequest(BaseModel):
    content: str

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

@app.post("/scrape-verify")
async def scrape_verify_news(req: ScrapeRequest):
    result = await scrape_verify(req.content)
    return result

@app.get("/headlines")
async def get_headlines():
    headlines = await fetch_headlines()
    return {"headlines": headlines}


