# sentiment
AI sentiment pipeline that pulls live market news + social media + fear/greed and outputs a normalized score for your bot in real-time

python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn[standard] feedparser requests python-dotenv pydantic "transformers>=4.40" torch --extra-index-url https://download.pytorch.org/whl/cpu
# Optional extras
pip install praw pymysql SQLAlchemy

How to use it in your bots
Kraken/IBKR entry filter:
Go long only if score >= 0.65
Reduce size if 0.55 <= score < 0.65
Block new longs if score < 0.5 or Fear & Greed < 30
Force‑trim / tighten trailing stop if score drops by ≥0.15 within last N minutes

API you’ll call from the bot
GET /sentiment?symbol=BTC/USD →

json
Copy
Edit
{
  "symbol":"BTC/USD",
  "score":0.71,
  "components":{
    "news":0.76,
    "social":0.62,
    "fear_greed":0.66
  },
  "ts":"2025-08-08T12:00:00Z",
  "n_articles":24,
  "n_social":180,
  "fg_value":66
}
GET /heartbeat → health/status

GET /latest → last full snapshot for all symbols

import requests
def get_sentiment(symbol):
    r = requests.get("http://127.0.0.1:8000/sentiment", params={"symbol": symbol}, timeout=5)
    r.raise_for_status()
    return r.json()

s = get_sentiment("BTC/USD")
if s["score"] >= 0.65:
    # allow entries / full size
elif 0.55 <= s["score"] < 0.65:
    # half size
else:
    # block new entries, maybe tighten stops

Notes / Tweaks
Add per‑symbol social counts: keep simple rolling counts in _social_scores() if you want that exposed in /sentiment.
Better ticker mapping: extend SYMBOL_MAP with more aliases (e.g., “$BTC”, “BTC-USD”).
More RSS feeds: Coindesk, Cointelegraph, Bloomberg Markets RSS, Yahoo Finance per‑ticker RSS, etc.
Performance: FinBERT on CPU is fine for headline‑scale. If you push thousands/min, consider GPU or micro‑batching.
