# 🔍 FalseFind — Backend API

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn)

**AI-powered fake news detection, source attribution, impersonation detection, and multi-API web verification**

</div>

---

## 🎯 What This Does

FalseFind is a backend API that helps users determine whether a news article is **real or fake** using two complementary approaches:

1. **ML-Based Analysis** — Trained machine learning models analyze the text for fake news patterns, predict the likely source from writing style, and flag impersonation attempts.
2. **Web Verification** — Searches the internet across multiple APIs (GNews, Google Fact Check, DuckDuckGo) to cross-reference the claim against real news sources, then produces a **REAL / FAKE / UNVERIFIED** verdict with a confidence score.
3. **Live Headlines** — Fetches trending headlines by category (Politics, Tech, Business, Entertainment, World) for the frontend ticker.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Fake News Detection** | Binary classifier (TF-IDF + trained model) determines if an article is fake or genuine |
| **Source Attribution** | Predicts the likely source based on writing style and linguistic patterns |
| **Impersonation Detection** | Flags when the claimed source doesn't match the detected writing style |
| **Multi-API Web Scraping** | Searches GNews, Google Fact Check, and DuckDuckGo in parallel |
| **Cross-Reference Verdict** | Aggregates results from all APIs to produce REAL / FAKE / UNVERIFIED with confidence % |
| **Live Headlines** | Fetches one headline per category from GNews for the frontend news ticker |

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app — all endpoints
│   ├── scraper.py            # Multi-API web scraping & cross-referencing
│   ├── headlines.py          # Live headline fetching per category
│   ├── fake_model.py         # Fake news detection model
│   ├── source_model.py       # Source attribution model
│   ├── preprocessing.py      # Text cleaning & spaCy tokenization
│   ├── style_features.py     # Linguistic feature extraction
│   ├── impersonation.py      # Impersonation detection logic
│   └── __init__.py
├── data/
│   ├── fake_news/            # Training datasets (Fake.csv, True.csv)
│   └── news_category/        # News category dataset
├── scripts/
│   ├── train_fake_news.py    # Train the fake news classifier
│   └── train_source_model_from_category.py  # Train the source attribution model
├── models/                   # Trained model artifacts (.pkl)
├── .env                      # API keys (not committed)
├── requirements.txt          # Python dependencies
└── README.md
```

## 🚀 Setup Guide

### Prerequisites

- **Python 3.11+**
- **pip** (comes with Python)

### Step 1 — Clone & Create Virtual Environment

```bash
git clone https://github.com/sanjayrohith/source-attribution.git
cd source-attribution

python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download NLP Models

```bash
python -m spacy download en_core_web_sm
```

### Step 4 — Configure API Keys (Optional but Recommended)

Create a `.env` file in the project root:

```env
GNEWS_API_KEY=your_gnews_api_key_here
GOOGLE_FACTCHECK_API_KEY=your_google_factcheck_key_here
```

| API | How to Get a Key | Free Tier |
|-----|-----------------|-----------|
| **GNews** | [gnews.io](https://gnews.io) — sign up and copy your key | 100 requests/day |
| **Google Fact Check** | [Google Cloud Console](https://console.cloud.google.com/apis/library/factchecktools.googleapis.com) — enable the API | Free, unlimited |

> **Without API keys**, the system falls back to DuckDuckGo search only. With keys, you get much richer results from news articles and existing fact-checks.

### Step 5 — Start the Server

```bash
python -m uvicorn app.main:app --reload
```

The API will be running at **http://localhost:8000**

### Step 6 — Verify It Works

```bash
# Test the analyze endpoint
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"title": "Test", "content": "Scientists discover new planet", "claimed_source": "UNKNOWN"}'

# Test the scrape endpoint
curl -X POST http://localhost:8000/scrape-verify \
  -H "Content-Type: application/json" \
  -d '{"content": "NASA discovers water on Mars"}'

# Test the headlines endpoint
curl http://localhost:8000/headlines
```

## 📡 API Endpoints

### `POST /analyze` — ML-Based Fake News Analysis

Analyzes text using trained machine learning models.

**Request:**
```json
{
  "title": "Breaking News: Major Discovery",
  "content": "Scientists announced a groundbreaking discovery today...",
  "claimed_source": "Reuters"
}
```

**Response:**
```json
{
  "fake_news": { "label": "REAL", "confidence": "92.3%" },
  "style_analysis": { "predicted_source": "Reuters", "confidence": "87.1%" },
  "impersonation_detected": false,
  "claimed_source": "Reuters"
}
```

---

### `POST /scrape-verify` — Multi-API Web Verification

Searches the internet across multiple APIs to verify a claim.

**Request:**
```json
{
  "content": "NASA discovers water on Mars"
}
```

**Response:**
```json
{
  "query_used": "NASA discovers water Mars",
  "verdict": "REAL",
  "confidence": 0.71,
  "explanation": "Found 8 source(s) including 3 reputable outlets...",
  "providers_used": ["gnews", "duckduckgo"],
  "fact_checks": [],
  "sources": [
    {
      "title": "NASA Confirms Water on Mars...",
      "url": "https://...",
      "snippet": "...",
      "domain": "reuters.com",
      "provider": "gnews"
    }
  ],
  "sources_found": 8
}
```

**How the verdict works:**

| Priority | Signal | Result |
|----------|--------|--------|
| 1st | Google Fact Check finds existing fact-checks rated "False" | **FAKE** |
| 2nd | Google Fact Check finds existing fact-checks rated "True" | **REAL** |
| 3rd | 3+ reputable news outlets (Reuters, AP, BBC…) report the claim | **REAL** |
| 4th | Some sources found but no reputable outlets | **UNVERIFIED** |
| 5th | No sources found at all | **UNVERIFIED** |

---

### `GET /headlines` — Live News Headlines

Returns one live headline per category for the frontend ticker.

**Response:**
```json
{
  "headlines": [
    { "headline": "...", "category": "POLITICS", "url": "...", "source": "CNN", "time_ago": "2 hours ago" },
    { "headline": "...", "category": "TECH", "url": "...", "source": "The Verge", "time_ago": "1 hour ago" },
    { "headline": "...", "category": "BUSINESS", ... },
    { "headline": "...", "category": "ENTERTAINMENT", ... },
    { "headline": "...", "category": "WORLD", ... }
  ]
}
```

## 🏋️ Model Training

```bash
# Train the fake news detection model
python scripts/train_fake_news.py

# Train the source attribution model
python scripts/train_source_model_from_category.py
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Framework** | FastAPI |
| **Server** | Uvicorn |
| **ML/NLP** | scikit-learn, spaCy, NLTK |
| **Data** | pandas, NumPy |
| **Web Scraping** | requests, BeautifulSoup, DuckDuckGo Search |
| **News APIs** | GNews API, Google Fact Check API |
| **Config** | python-dotenv |

## 📡 API Docs (Interactive)

Once the server is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## ⚠️ Disclaimer

This tool is for **educational and demonstration purposes only**. Always verify news through multiple reputable sources before sharing.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
