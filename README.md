# News_Summarization_and_Text_to_Speech_Application
---
title: News Summarization and Text-to-Speech Application
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: false
license: mit
short_description: Web Scrapping, Gradio, HuggingFace, Sentiment analysis, BeautifulSoup, TTS
---

🚀 Try the demo on Hugging Face Spaces:  
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Try%20it%20now-yellow)](https://huggingface.co/spaces/Srishtipriya/news-summarization-tts-app_2)

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---
![image](https://github.com/user-attachments/assets/d17b6c52-f24f-4e04-8d1a-2ea0dce0a1c2)
![image](https://github.com/user-attachments/assets/54b7e3f2-0ed5-4393-9f37-fb0bd625637e)

This application fetches news articles, summarizes them, analyzes their sentiment, and provides translations of the summaries. It uses the Hugging Face Transformers library for natural language processing tasks and Gradio for the user interface.
This project has been implemented and tested in Google Colab for seamless execution and efficient model inference.
## Features

- Summarizes news articles using a BART model.
- Analyzes sentiment of the summaries using a fine-tuned DistilRoBERTa model.
- Translates summaries from English to Hindi.
- Displays a sentiment distribution report for multiple articles.

## Requirements

- Python 3.9 or higher
- Libraries:
  - `gradio` - `transformers`
  - `requests`
  - `beautifulsoup4`
  - `nltk`
  - 'gradio'
  - 'transformers'
  - 'torch'
  - 'requests'
  - 'beautifulsoup4'
  - 'nltk'

  
You can install the required libraries using pip:

```bash
pip install gradio transformers requests beautifulsoup4 nltk


export HF_TOKEN='your_hugging_face_token'

import nltk
nltk.download('stopwords')

# News Summarization and Sentiment Analysis Application

## Overview
This project is a **News Summarization and Sentiment Analysis Application** that:
- Scrapes news articles from predefined URLs
- Extracts relevant content (title, body, keywords, and publication date)
- Summarizes the articles using **BART (facebook/bart-large-cnn)**
- Analyzes sentiment using **DistilRoBERTa (mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)**
- Translates the summary to Hindi using **NLLB-200 (facebook/nllb-200-distilled-600M)**
- Performs comparative sentiment analysis across multiple articles
- Presents results in a Gradio-based UI

---

## Technologies and Models Used

| Component                  | Model / Library                                         | Purpose |
|----------------------------|---------------------------------------------------------|---------|
| Summarization              | `facebook/bart-large-cnn`                               | Summarizes long articles into concise summaries |
| Sentiment Analysis         | `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | Analyzes sentiment (positive, negative, neutral) of the summary |
| Translation (English-Hindi)| `facebook/nllb-200-distilled-600M`                     | Translates English summary to Hindi |
| Web Scraping               | `requests`, `BeautifulSoup`                            | Extracts news content from websites |
| Keyword Extraction         | `nltk`                                                 | Extracts relevant keywords from the text |
| UI                         | `Gradio`                                               | Provides an interactive interface |

---

## Installation
Ensure you have **Python 3.7+** installed. Then, install the required libraries:

```bash
pip install gradio transformers requests beautifulsoup4 nltk
```

Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

---

## How It Works

### 1. Web Scraping
- The script fetches news articles from predefined URLs using `requests`.
- `BeautifulSoup` is used to parse and extract the **title**, **content**, **publication date**, and **keywords**.
- Random **user-agents** are used to avoid getting blocked.

### 2. Summarization
- Uses `facebook/bart-large-cnn` to summarize the article.
- Ensures the summary has **40-200 words**.

### 3. Sentiment Analysis
- Uses `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`.
- Analyzes the sentiment of the **summarized text**.
- Outputs **positive, negative, or neutral** sentiment with a confidence score.

### 4. Hindi Translation
- Uses `facebook/nllb-200-distilled-600M` to translate the summary into Hindi.

### 5. Keyword Extraction
- Extracts metadata keywords from HTML.
- Falls back to **text-based keyword extraction** using `nltk` and `Counter`.

### 6. Comparative Sentiment Analysis
- Calculates the **distribution of sentiment** across multiple articles.
- Outputs percentage of **positive, negative, and neutral** articles.

### 7. User Interface (Gradio)
- Allows users to select a company (currently only **Tesla**).
- Displays the summarized news, sentiment, Hindi translation, and keyword analysis.
- Presents a comparative sentiment distribution.

---

## Code Walkthrough

### 1. Importing Libraries
```python
import gradio as gr
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import re
import random
import time
from nltk.corpus import stopwords
import nltk
from collections import Counter
```

### 2. Model Loading
```python
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
translator = pipeline("translation_en_to_hi", model="facebook/nllb-200-distilled-600M")
```

### 3. Web Scraping with User-Agent Rotation
```python
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

headers = {"User-Agent": random.choice(user_agents)}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')
```

### 4. Extracting and Processing News Data
```python
def extract_news_data(url):
    title = soup.find('title').get_text(strip=True)
    article_body = ' '.join([p.get_text(strip=True) for p in soup.find_all('p')])
    summary = summarizer(article_body, max_length=200, min_length=40, do_sample=False)[0]['summary_text']
    sentiment = sentiment_analyzer(summary)[0]
    hindi_summary = translator(summary)[0]['translation_text']
    return { 'title': title, 'summary': summary, 'sentiment': sentiment, 'hindi_summary': hindi_summary }
```

### 5. Comparative Sentiment Analysis
```python
def comparative_sentiment_analysis(articles):
    sentiment_counts = Counter(article['sentiment']['label'].lower() for article in articles)
    total = len(articles)
    return {
        "positive": (sentiment_counts["positive"] / total) * 100,
        "negative": (sentiment_counts["negative"] / total) * 100,
        "neutral": (sentiment_counts["neutral"] / total) * 100,
    }
```

### 6. Gradio UI
```python
iface = gr.Interface(
    fn=fetch_news_and_sentiment,
    inputs=gr.Dropdown(label="Select Company", choices=["Tesla"], value="Tesla"),
    outputs=gr.Textbox(label="Sentiment Report"),
    title="News Summarization and Sentiment Analysis",
    description="Select a company to fetch news articles and generate a sentiment report."
)
iface.launch()
```

---

## Expected Output

**Example Output:**
```
Title: Tesla's New Model Unveiled
Summary: Tesla has unveiled its latest model, revolutionizing the EV market...
Summary (Hindi): टेस्ला ने अपने नवीनतम मॉडल का अनावरण किया है...
Sentiment: Positive (Confidence: 0.95)
Publication Date: 2025-03-21
Keywords: Tesla, electric, vehicle, model, unveil
-----------------------------------------------------
Sentiment Distribution:
Positive Articles: 75.00%
Negative Articles: 15.00%
Neutral Articles: 10.00%
```

---

## Future Enhancements
- Add more companies and dynamic news scraping
- Improve keyword extraction with advanced NLP techniques
- Integrate Text-to-Speech (TTS) for Hindi summaries
- Deploy on a cloud-based system

---

## License
This project is for educational purposes and is open-source. Contributions are welcome!
Title: Tesla's New Model Unveiled
Summary: Tesla has unveiled its latest model, which promises to revolutionize the electric vehicle market...
Summary (Hindi): टेस्ला ने अपने नवीनतम मॉडल का अनावरण किया है, जो इलेक्ट्रिक वाहन बाजार में क्रांति लाने का वादा करता है...
Sentiment: Positive (Confidence: 0.95)
Publication Date: 2025-03-21
Keywords: Tesla, electric, vehicle, model, unveil
----------------------------------------------------------------------------------------------------
Sentiment Distribution:
Positive Articles: 75.00%
Negative Articles: 15.00%
Neutral Articles: 10.00%
