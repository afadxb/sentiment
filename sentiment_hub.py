import os, re, time, math, json, asyncio, logging, hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import requests
import feedparser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Optional deps
try:
    import praw  # Reddit
except Exception:
    praw = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None

# ------------ Config ------------
load_dotenv()

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USD,ETH/USD,TSLA").split(",") if s.strip()]
REFRESH_SECONDS = int(os.getenv("REFRESH_SECONDS", "60"))

USE_FINBERT = os.getenv("USE_FINBERT", "true").lower() == "true"
WEIGHT_NEWS = float(os.getenv("WEIGHT_NEWS", "0.5"))
WEIGHT_SOCIAL = float(os.getenv("WEIGHT_SOCIAL", "0.3"))
WEIGHT_FEAR_GREED = float(os.getenv("WEIGHT_FEAR_GREED", "0.2"))

ENABLE_STOCKTWITS = os.getenv("ENABLE_STOCKTWITS", "true").lower() == "true"
ENABLE_REDDIT = os.getenv("ENABLE_REDDIT", "false").lower() == "true"

RSS_FEEDS = [u.strip() for u in os.getenv("RSS_FEEDS","").split(",") if u.strip()]

MYSQL_URI = os.getenv("MYSQL_URI", "")
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY", "")
PUSHOVER_API_TOKEN = os.getenv("PUSHOVER_API_TOKEN", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Symbol aliases (very light). Extend as needed.
SYMBOL_MAP = {
    "BTC/USD": ["BTC", "Bitcoin"],
    "ETH/USD": ["ETH", "Ethereum"],
    "XRP/USD": ["XRP", "Ripple"],
    "DOGE/USD":["DOGE","Dogecoin"],
    "TSLA": ["TSLA", "Tesla"],
    "AMD":  ["AMD", "Advanced Micro Devices"],
    "SPY":  ["SPY", "S&P 500", "S&P500"]
}

# ------------ Utilities ------------
def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def simple_text_match(text: str, aliases: List[str]) -> bool:
    t = text.lower()
    return any(a.lower() in t for a in aliases)

def softmax(x):
    e = [math.exp(i - max(x)) for i in x]
    s = sum(e)
    return [i/s for i in e]

def normalize_minmax(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    return clamp01((v - lo) / (hi - lo))

# ------------ FinBERT / Fallback ------------
class SentimentModel:
    def __init__(self, enabled=True):
        self.enabled = enabled and AutoTokenizer is not None
        self.model = None
        self.tokenizer = None
        if self.enabled:
            try:
                # ProsusAI/finbert (finance-tuned)
                mname = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(mname)
                self.model = AutoModelForSequenceClassification.from_pretrained(mname)
                self.model.eval()
                logging.info("FinBERT loaded.")
            except Exception as e:
                logging.warning(f"FinBERT load failed, falling back. {e}")
                self.enabled = False

    def score_text(self, text: str) -> float:
        """
        Returns sentiment in [0..1], where >0.5 is positive.
        FinBERT labels: [negative, neutral, positive]
        """
        if self.enabled and self.model and self.tokenizer:
            try:
                inputs = self.tokenizer(text[:512], return_tensors="pt", truncation=True)
                with torch.no_grad():
                    logits = self.model(**inputs).logits[0].tolist()
                probs = softmax(logits)
                neg, neu, pos = probs
                return clamp01(pos + 0.5*neu)  # reward positive; neutral nudges to mid
            except Exception:
                pass
        # Fallback: simple rule-based on keywords
        text_l = text.lower()
        pos_kw = ["beats", "surge", "rally", "bull", "breakout", "upgrade", "partnership", "profit", "gain"]
        neg_kw = ["miss", "plunge", "bear", "downgrade", "lawsuit", "loss", "hack", "breach", "bankrupt"]
        score = 0.5
        if any(k in text_l for k in pos_kw): score += 0.2
        if any(k in text_l for k in neg_kw): score -= 0.2
        return clamp01(score)

SENTI = SentimentModel(enabled=USE_FINBERT)

# ------------ Sources ------------
class FearGreed:
    URL = "https://api.alternative.me/fng/?limit=1&format=json"
    cache_val = None
    cache_ts = 0

    @classmethod
    def get(cls) -> Dict[str, Any]:
        try:
            if time.time() - cls.cache_ts < 30 and cls.cache_val:
                return cls.cache_val
            r = requests.get(cls.URL, timeout=10)
            r.raise_for_status()
            data = r.json()["data"][0]
            v = int(data["value"])
            cls.cache_val = {"value": v, "value_classification": data["value_classification"], "timestamp": data["timestamp"]}
            cls.cache_ts = time.time()
            return cls.cache_val
        except Exception as e:
            logging.warning(f"FearGreed fetch failed: {e}")
            # neutral fallback
            return {"value": 50, "value_classification":"Neutral", "timestamp": int(time.time())}

class NewsRSS:
    @staticmethod
    def fetch_articles(feeds: List[str], max_items=200) -> List[Dict[str, Any]]:
        items = []
        for url in feeds:
            try:
                parsed = feedparser.parse(url)
                for e in parsed.entries[:100]:
                    items.append({
                        "title": e.get("title",""),
                        "summary": e.get("summary",""),
                        "link": e.get("link",""),
                        "published": e.get("published","")
                    })
            except Exception as e:
                logging.warning(f"RSS fetch failed ({url}): {e}")
        # de-dup by link hash
        seen = set()
        unique = []
        for it in items:
            h = hashlib.md5(it["link"].encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(it)
            if len(unique) >= max_items:
                break
        return unique

class StockTwits:
    @staticmethod
    def fetch_stream(symbol_alias: str, max_items=200) -> List[str]:
        """
        StockTwits symbol format: $BTC.X (crypto) or $TSLA
        We'll attempt both formats.
        """
        texts = []
        base = "https://api.stocktwits.com/api/2/streams/symbol/{}.json"
        candidates = [symbol_alias, f"{symbol_alias}.X"]
        for c in candidates:
            try:
                url = base.format(c)
                r = requests.get(url, timeout=10)
                if r.status_code != 200:
                    continue
                data = r.json()
                for m in data.get("messages", [])[:max_items]:
                    body = m.get("body","")
                    if body:
                        texts.append(body)
                if texts:
                    break
            except Exception as e:
                logging.warning(f"StockTwits fetch failed for {symbol_alias}: {e}")
        return texts

class RedditFeed:
    def __init__(self):
        cid = os.getenv("REDDIT_CLIENT_ID","")
        sec = os.getenv("REDDIT_CLIENT_SECRET","")
        ua  = os.getenv("REDDIT_USER_AGENT","sentiment-hub/1.0")
        self.enabled = ENABLE_REDDIT and praw is not None and cid and sec and ua
        self.reddit = None
        if self.enabled:
            try:
                self.reddit = praw.Reddit(client_id=cid, client_secret=sec, user_agent=ua)
            except Exception as e:
                logging.warning(f"Reddit init failed: {e}")
                self.enabled = False

    def fetch(self, queries: List[str], subreddits=("CryptoCurrency","stocks","wallstreetbets"), limit=100) -> List[str]:
        if not self.enabled:
            return []
        texts = []
        try:
            for q in queries:
                for sub in subreddits:
                    for s in self.reddit.subreddit(sub).search(q, sort="new", limit=limit):
                        if s.title:
                            texts.append(s.title)
                        if s.selftext:
                            texts.append(s.selftext[:500])
        except Exception as e:
            logging.warning(f"Reddit fetch failed: {e}")
        return texts

REDDIT = RedditFeed()

# ------------ Aggregation ------------
@dataclass
class ComponentScores:
    news: float
    social: float
    fear_greed: float

class Aggregator:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.snapshot: Dict[str, Dict[str, Any]] = {}
        self.last_full: Dict[str, Any] = {}

    def _news_scores(self, articles: List[Dict[str,Any]]) -> Dict[str,float]:
        # per symbol scan → score headlines with FinBERT → average
        out = {}
        for sym in self.symbols:
            aliases = SYMBOL_MAP.get(sym, [sym])
            sym_articles = [a for a in articles if simple_text_match(a["title"] + " " + a["summary"], aliases)]
            if not sym_articles:
                # fallback: general market news
                sym_articles = articles[:10]
            scores = [SENTI.score_text(a["title"] + " " + a["summary"]) for a in sym_articles]
            out[sym] = sum(scores)/len(scores) if scores else 0.5
        return out

    def _social_scores(self) -> Dict[str,float]:
        out = {}
        for sym in self.symbols:
            aliases = SYMBOL_MAP.get(sym, [sym])
            # Choose best StockTwits ticker alias heuristic
            primary = re.sub(r"[^A-Z]", "", aliases[0].upper()) or aliases[0].upper()
            texts = []
            if ENABLE_STOCKTWITS:
                texts += StockTwits.fetch_stream(primary, max_items=120)
            if REDDIT.enabled:
                texts += REDDIT.fetch(aliases, limit=50)
            # Light noise filter: drop posts shorter than 12 chars
            texts = [t for t in texts if len(t) >= 12]
            if not texts:
                out[sym] = 0.5
                continue
            vals = [SENTI.score_text(t) for t in texts[:300]]
            # robust mean
            vals.sort()
            k = max(1, int(0.1*len(vals)))
            trimmed = vals[k:-k] if len(vals) > 2*k else vals
            out[sym] = sum(trimmed)/len(trimmed) if trimmed else 0.5
        return out

    def _fear_greed_score(self) -> float:
        fg = FearGreed.get()
        # normalize 0..100 to 0..1
        return normalize_minmax(float(fg["value"]), 0.0, 100.0)

    def compute(self, articles: List[Dict[str,Any]]) -> Dict[str, Any]:
        news = self._news_scores(articles)
        social = self._social_scores()
        fg_score = self._fear_greed_score()

        result = {}
        for sym in self.symbols:
            score = (
                WEIGHT_NEWS   * news.get(sym, 0.5) +
                WEIGHT_SOCIAL * social.get(sym, 0.5) +
                WEIGHT_FEAR_GREED * fg_score
            )
            result[sym] = {
                "symbol": sym,
                "score": round(clamp01(score), 4),
                "components": {
                    "news": round(news.get(sym,0.5),4),
                    "social": round(social.get(sym,0.5),4),
                    "fear_greed": round(fg_score,4)
                }
            }
        self.last_full = {
            "ts": utcnow(),
            "result": result,
            "fg_value": FearGreed.get()["value"]
        }
        return self.last_full

AGG = Aggregator(SYMBOLS)

# ------------ Storage / Alerts (optional) ------------
def mysql_write(snapshot: Dict[str, Any]):
    if not MYSQL_URI:
        return
    try:
        from sqlalchemy import create_engine, text
        eng = create_engine(MYSQL_URI, pool_pre_ping=True)
        with eng.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sentiment_snapshot (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    ts TIMESTAMP NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    score DECIMAL(6,4) NOT NULL,
                    news DECIMAL(6,4) NOT NULL,
                    social DECIMAL(6,4) NOT NULL,
                    fear_greed DECIMAL(6,4) NOT NULL,
                    fg_value INT NULL,
                    UNIQUE KEY uk_ts_symbol (ts, symbol)
                ) ENGINE=InnoDB;
            """))
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            for sym, row in snapshot["result"].items():
                conn.execute(text("""
                    INSERT IGNORE INTO sentiment_snapshot
                    (ts, symbol, score, news, social, fear_greed, fg_value)
                    VALUES (:ts,:symbol,:score,:news,:social,:fg,:fgv)
                """), dict(
                    ts=ts,
                    symbol=sym,
                    score=row["score"],
                    news=row["components"]["news"],
                    social=row["components"]["social"],
                    fg=row["components"]["fear_greed"],
                    fgv=snapshot.get("fg_value")
                ))
    except Exception as e:
        logging.warning(f"MySQL write failed: {e}")

def pushover(title: str, message: str):
    if not (PUSHOVER_API_TOKEN and PUSHOVER_USER_KEY):
        return
    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": PUSHOVER_API_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "title": title,
            "message": message
        }, timeout=10)
    except Exception as e:
        logging.warning(f"Pushover failed: {e}")

# ------------ Scheduler ------------
LATEST_ARTICLES: List[Dict[str,Any]] = []
RUNNING = True

async def refresh_loop():
    global LATEST_ARTICLES
    while RUNNING:
        try:
            if RSS_FEEDS:
                LATEST_ARTICLES = NewsRSS.fetch_articles(RSS_FEEDS, max_items=300)
            snapshot = AGG.compute(LATEST_ARTICLES)
            mysql_write(snapshot)
            # Optional alert: extreme greed/fear swing
            fg_val = snapshot.get("fg_value", 50)
            if fg_val is not None and (fg_val >= 80 or fg_val <= 20):
                pushover("Market Sentiment Extreme", f"Fear & Greed at {fg_val}")
            logging.info("Snapshot updated.")
        except Exception as e:
            logging.error(f"Refresh loop error: {e}")
        await asyncio.sleep(REFRESH_SECONDS)

# ------------ API ------------
app = FastAPI(title="Sentiment Hub", version="1.0.0")

class SentimentResp(BaseModel):
    symbol: str
    score: float
    components: Dict[str,float]
    ts: str
    n_articles: int
    n_social: int
    fg_value: Optional[int] = None

@app.get("/heartbeat")
def heartbeat():
    return {"ok": True, "ts": utcnow(), "symbols": SYMBOLS, "finbert": SENTI.enabled}

@app.get("/latest")
def latest():
    if not AGG.last_full:
        raise HTTPException(status_code=503, detail="Not ready")
    return AGG.last_full

@app.get("/sentiment")
def sentiment(symbol: str):
    symbol = symbol.strip()
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Unknown symbol {symbol}. Configure in .env SYMBOLS.")
    if not AGG.last_full:
        raise HTTPException(status_code=503, detail="Not ready")
    row = AGG.last_full["result"][symbol]
    n_articles = len(LATEST_ARTICLES)
    # This is approximate; for simplicity we don't store per-source counts
    n_social = 0  # can be extended to track
    return {
        "symbol": symbol,
        "score": row["score"],
        "components": row["components"],
        "ts": AGG.last_full["ts"],
        "n_articles": n_articles,
        "n_social": n_social,
        "fg_value": AGG.last_full.get("fg_value")
    }

def main():
    import uvicorn
    loop = asyncio.get_event_loop()
    loop.create_task(refresh_loop())
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
