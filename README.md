# Fake News Detection & Source Attribution

A comprehensive machine learning system that detects fake news articles and attributes sources while identifying impersonation attempts. This project combines multiple NLP techniques to analyze news credibility and writing style.

## 🎯 Features

- **Fake News Detection**: Binary classification model that determines whether a news article is fake or genuine
- **Source Attribution**: Identifies the likely news source based on writing style and linguistic patterns
- **Impersonation Detection**: Detects when an article's claimed source doesn't match its detected source
- **Style Analysis**: Extracts linguistic and stylistic features from news content for source identification
- **REST API**: FastAPI-based API for easy integration with other applications

## 📋 Project Structure

```
├── app/
│   ├── main.py                      # FastAPI application and endpoints
│   ├── fake_model.py                # Fake news detection model
│   ├── source_model.py              # Source attribution model
│   ├── preprocessing.py             # Text cleaning and preprocessing
│   ├── style_features.py            # Feature extraction for style analysis
│   ├── impersonation.py             # Impersonation detection logic
│   └── __init__.py
├── data/
│   ├── fake_news/
│   │   ├── Fake.csv                 # Fake news dataset
│   │   └── True.csv                 # Genuine news dataset
│   └── news_category/
│       └── News_Category_Dataset_v3.json  # News category dataset
├── scripts/
│   ├── train_fake_news.py           # Training script for fake news model
│   ├── train_source_model_from_category.py  # Training script for source model
│   └── __pycache__/
├── models/                          # Directory for trained model artifacts
├── requirements.txt                 # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   cd "Fake news and source attribution"
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required NLP models**
   ```bash
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt stopwords
   ```

### Running the API Server

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📡 API Usage

### Analyze News Article

**Endpoint**: `POST /analyze`

**Request Body**:
```json
{
  "title": "Breaking News: Major Discovery",
  "content": "Scientists announced a groundbreaking discovery today...",
  "claimed_source": "Reuters"
}
```

**Response**:
```json
{
  "fake_news": {
    "is_fake": false,
    "confidence": 0.92
  },
  "style_analysis": {
    "predicted_source": "Reuters",
    "confidence": 0.87,
    "style_features": {...}
  },
  "impersonation": {
    "is_impersonating": false,
    "confidence": 0.95
  }
}
```

## 🏋️ Model Training

### Train Fake News Detection Model

```bash
python scripts/train_fake_news.py
```

This script trains a binary classifier using the provided fake/true news datasets.

### Train Source Attribution Model

```bash
python scripts/train_source_model_from_category.py
```

This script trains a multi-class classifier to predict the news source based on stylistic features.

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI |
| **Server** | Uvicorn |
| **ML/NLP** | scikit-learn, spacy, nltk |
| **Data Processing** | pandas, numpy |
| **Text Analysis** | TextBlob |
| **Model Serialization** | joblib |

## 📊 Datasets

- **Fake News Dataset**: Balanced dataset of fake and genuine news articles
- **News Category Dataset**: Articles categorized by source and news type

## 🔧 Core Modules

### `preprocessing.py`
- Text cleaning and normalization
- Stop word removal
- Tokenization using spaCy

### `style_features.py`
- Extracts linguistic features from text
- Analyzes writing patterns
- Generates feature vectors for source prediction

### `fake_model.py`
- Binary classifier for fake news detection
- Uses TF-IDF vectorization and trained classifier

### `source_model.py`
- Multi-class classifier for source attribution
- Predicts news source based on style features

### `impersonation.py`
- Compares claimed source with predicted source
- Detects impersonation attempts
- Calculates confidence scores

## 📝 Example Workflow

```python
from app.preprocessing import clean_text
from app.style_features import extract_style_features
from app.fake_model import predict_fake
from app.source_model import predict_source

# Prepare article
title = "Breaking News"
content = "Article content here..."

# Process text
cleaned_text = clean_text(title + " " + content)

# Get predictions
fake_prediction = predict_fake(cleaned_text)
source_prediction = predict_source(extract_style_features(content))

# Analyze results
print(f"Is Fake: {fake_prediction['is_fake']}")
print(f"Predicted Source: {source_prediction['predicted_source']}")
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.

## 📞 Contact

For questions or feedback, please open an issue in the repository.
