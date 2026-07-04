# 📰 News Bias & Fact Clustering Engine

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/news-bias-clustering-engine/blob/main/News_Bias_Fact_Clustering_Engine.ipynb)

An AI system that pulls the **same real-world news event as covered by multiple outlets**, groups those articles together with semantic clustering, and compares how differently each outlet framed the story — tone, sentiment, subjectivity, and keyword emphasis.

This project does **not** attempt to label anything as true or false. It answers a narrower, evidence-based question:

> *Different outlets covered the same event — how did their language and framing differ?*

> ⚠️ Update the Colab badge link above with your actual GitHub username/repo name after you push, so it opens directly in Colab for anyone visiting the repo.

---

## Why I Built This

News coverage of the same event can vary a lot in tone and emphasis depending on the outlet, but that's hard to see at a glance from any single article. I wanted to build a pipeline that automatically finds articles about the *same* event across sources and quantifies how the language actually differs — as a hands-on project combining semantic search, unsupervised clustering, and NLP-based text analysis on real, live data instead of a static/toy dataset.

## Skills Demonstrated
- **NLP & semantic embeddings** — `sentence-transformers` for meaning-based text representation
- **Unsupervised machine learning** — Agglomerative Clustering with cosine distance to auto-discover event groups without knowing cluster count in advance
- **Sentiment & subjectivity analysis** — quantifying language tone with `TextBlob`
- **Keyword/keyphrase extraction** — `KeyBERT` for automatic framing analysis
- **Data engineering** — live RSS ingestion, cleaning, and deduplication pipeline
- **Data visualization** — comparative charts and word clouds
- **Applied ML product thinking** — interactive Gradio demo, plus explicit limitations/ethics section

## Features
- Pulls live articles from multiple news outlets via RSS (no paid API key required)
- Groups articles about the same underlying event using sentence embeddings + clustering
- Scores each article's sentiment polarity & subjectivity
- Extracts framing keywords per outlet with KeyBERT
- Generates a side-by-side "event report" comparing outlets on the same story
- Interactive Gradio demo to explore clusters with a shareable link

## Tech Stack
| Component | Library |
|---|---|
| Data ingestion | `feedparser` (RSS) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Clustering | `scikit-learn` (Agglomerative Clustering, cosine distance) |
| Sentiment / subjectivity | `TextBlob` |
| Keyword extraction | `KeyBERT` |
| Visualization | `matplotlib`, `wordcloud` |
| UI | `gradio` |

## How It Works
```
RSS Feeds (multiple outlets)
        │
        ▼
  Fetch + Clean Text
        │
        ▼
 Sentence Embeddings (MiniLM)
        │
        ▼
 Agglomerative Clustering  ──►  groups articles about the SAME event
        │
        ▼
 Sentiment + Subjectivity (per article)
 Keyword Extraction (per article)
        │
        ▼
   Event Comparison Report  +  Gradio UI
```

## Project Structure
```
news-bias-clustering-engine/
│
├── News_Bias_Fact_Clustering_Engine.ipynb   # main notebook — run this
├── assets/
│   ├── sample_sentiment_chart.png           # example output chart
│   └── sample_gradio_ui.png                 # example Gradio demo screenshot
├── requirements.txt                         # for local/non-Colab use
├── LICENSE
└── README.md
```

## Run it in Google Colab
1. Click the **Open in Colab** badge at the top of this README (or open [Google Colab](https://colab.research.google.com/) and upload the notebook manually)
2. Run all cells top to bottom (`Runtime → Run all`)
3. The last cell launches a Gradio demo with a public shareable link

No API keys, no local setup, no GPU required (runs fine on CPU).

## Run it locally
```bash
git clone https://github.com/YOUR-USERNAME/news-bias-clustering-engine.git
cd news-bias-clustering-engine
pip install -r requirements.txt
jupyter notebook News_Bias_Fact_Clustering_Engine.ipynb
```

## Sample Output

**Sentiment comparison across outlets for the same event:**

![Sample sentiment chart](assets/sample_sentiment_chart.png)

**Interactive Gradio demo output:**

![Sample Gradio UI](assets/sample_gradio_ui.png)

```
EVENT CLUSTER #7  |  Covered by 4 outlet(s)

[BBC News]  (Center)
Headline    : Central bank raises interest rates amid inflation concerns
Polarity    : -0.05   (-1 negative ... +1 positive)
Subjectivity: 0.20   (0 factual ... 1 opinionated)
Keywords    : interest rates, inflation concerns, central bank
------------------------------------------------------------------------

[Fox News]  (Right)
Headline    : Fed hikes rates again as Americans struggle with prices
Polarity    : -0.31
Subjectivity: 0.48
Keywords    : struggle prices, fed hikes, rising costs
------------------------------------------------------------------------

Tone spread across outlets -> polarity range: 0.41 | subjectivity range: 0.35
```

## Limitations (read before presenting this project)
- **Bias labels are static, simplified outlet-level tags** (based on commonly cited public media-bias categorizations), not a measurement of the specific article's content.
- **Sentiment is not the same as bias.** Polarity/subjectivity describe language tone, not political slant or factual accuracy.
- **Clustering quality depends on the distance threshold**, which is tunable in the notebook.
- RSS feed URLs can change over time — swap in alternatives if one stops working.

## Possible Extensions
- Swap TextBlob for a transformer sentiment model (e.g. `cardiffnlp/twitter-roberta-base-sentiment-latest`)
- Track the same event over multiple days to see framing drift as a story develops
- Add a stance/entailment model to detect actual factual disagreement, not just tone
- Swap RSS for NewsAPI or GDELT for broader/historical coverage

## License
This project is licensed under the [MIT License](LICENSE) — free to use, modify, and share.
