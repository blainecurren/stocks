import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import uvicorn
import logging
import json

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model variables
sentiment_pipeline = None

# Request/Response models
class NewsAnalysisRequest(BaseModel):
    ticker: Optional[str] = None
    days_back: Optional[int] = None
    analysis_type: str = "summary"

class CompanySentiment(BaseModel):
    ticker: str
    company_name: str
    total_articles: int
    sentiment: str
    sentiment_score: float
    key_insights: List[str]
    recent_headlines: List[str]
    summary: str

class NewsAnalysisResponse(BaseModel):
    analysis_date: str
    period_analyzed: str
    total_articles_analyzed: int
    companies: List[CompanySentiment]
    market_overview: str

def load_model():
    """Load the FinBERT model"""
    global sentiment_pipeline
    logger.info("="*50)
    logger.info("Loading FinBERT Model...")
    logger.info("="*50)
    
    try:
        model_name = "ProsusAI/finbert"
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info("✅ FinBERT model loaded successfully!")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise

def get_news_from_db(ticker: Optional[str] = None, days_back: Optional[int] = None) -> List[Dict]:
    """Fetch news from SQLite database with detailed logging"""
    logger.info("\n" + "="*50)
    logger.info("📊 FETCHING NEWS FROM DATABASE")
    logger.info("="*50)
    logger.info(f"Parameters:")
    logger.info(f"  - Ticker filter: {ticker if ticker else 'ALL TICKERS'}")
    logger.info(f"  - Days back: {days_back if days_back else 'ALL TIME'}")
    
    # Try multiple possible database locations
    db_paths = [
        os.path.join('..', 'data', 'polygon_data.db'),
        os.path.join('data', 'polygon_data.db'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'polygon_data.db'),
        'polygon_data.db'
    ]
    
    db_path = None
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            logger.info(f"✅ Found database at: {path}")
            break
    
    if not db_path:
        logger.error(f"❌ Database not found. Searched paths: {db_paths}")
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Log table structure
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    logger.info(f"Available tables: {tables}")
    
    # Build and log query
    query = """
    SELECT 
        ticker,
        title,
        description,
        published_utc,
        article_url
    FROM news
    WHERE ticker IS NOT NULL AND ticker != ''
    """
    
    params = []
    
    if ticker:
        query += " AND ticker LIKE ?"
        params.append(f"%{ticker}%")
    
    if days_back:
        cutoff_date = datetime.now() - timedelta(days=days_back)
        query += " AND datetime(published_utc) >= ?"
        params.append(cutoff_date.isoformat())
    
    query += " ORDER BY published_utc DESC LIMIT 1000"
    
    logger.info(f"SQL Query: {query}")
    logger.info(f"Parameters: {params}")
    
    try:
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        logger.info(f"✅ Raw articles fetched: {len(results)}")
        
        # Log sample of raw data
        if results:
            logger.info("\n📰 SAMPLE RAW ARTICLE:")
            sample = results[0]
            for key, value in sample.items():
                logger.info(f"  {key}: {value[:100] if value and isinstance(value, str) else value}")
        
        # Parse comma-separated tickers
        parsed_results = []
        ticker_counts = {}
        
        for article in results:
            if article['ticker']:
                tickers = article['ticker'].split(',')
                for t in tickers:
                    t = t.strip()
                    if t:
                        parsed_results.append({
                            **article,
                            'ticker': t,
                            'company_name': t
                        })
                        ticker_counts[t] = ticker_counts.get(t, 0) + 1
        
        conn.close()
        
        logger.info(f"\n📊 DATABASE FETCH SUMMARY:")
        logger.info(f"  Total parsed articles: {len(parsed_results)}")
        logger.info(f"  Unique tickers: {len(ticker_counts)}")
        logger.info(f"\n  Top 10 tickers by article count:")
        for ticker, count in sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"    {ticker}: {count} articles")
        
        return parsed_results
        
    except Exception as e:
        logger.error(f"❌ Error querying database: {str(e)}")
        conn.close()
        return []

def analyze_sentiment_finbert(articles: List[Dict], ticker: str) -> Dict:
    """Analyze sentiment using FinBERT with detailed logging"""
    logger.info(f"\n{'='*50}")
    logger.info(f"🤖 ANALYZING SENTIMENT FOR: {ticker}")
    logger.info(f"{'='*50}")
    logger.info(f"Total articles to analyze: {len(articles)}")
    
    if not articles:
        logger.warning(f"⚠️ No articles found for {ticker}")
        return {
            'ticker': ticker,
            'company_name': ticker,
            'total_articles': 0,
            'sentiment': 'neutral',
            'sentiment_score': 0.0,
            'key_insights': ['No articles found for analysis'],
            'recent_headlines': [],
            'summary': 'No news articles available for this ticker.'
        }
    
    # Prepare texts for analysis
    texts_to_analyze = []
    articles_to_analyze = articles[:20]  # Analyze up to 20 most recent articles
    
    logger.info(f"\n📄 PREPARING TEXTS FOR FINBERT:")
    logger.info(f"Analyzing {len(articles_to_analyze)} most recent articles")
    
    for i, article in enumerate(articles_to_analyze):
        # Combine title and description
        text = article['title']
        if article.get('description'):
            text += f" {article['description'][:200]}"
        
        # Truncate to FinBERT max length
        text = text[:512]
        texts_to_analyze.append(text)
        
        # Log what FinBERT will see
        logger.info(f"\n  Article {i+1}:")
        logger.info(f"    Date: {article.get('published_utc', 'Unknown')}")
        logger.info(f"    Title: {article['title'][:100]}")
        logger.info(f"    TEXT SENT TO FINBERT: {text[:200]}...")
    
    try:
        logger.info(f"\n🔍 Running FinBERT sentiment analysis...")
        results = sentiment_pipeline(texts_to_analyze)
        
        # Log raw FinBERT results
        logger.info(f"\n📊 FINBERT RAW RESULTS:")
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiment_scores = []
        
        for i, (text, result) in enumerate(zip(texts_to_analyze, results)):
            label = result['label'].lower()
            score = result['score']
            
            logger.info(f"\n  Article {i+1}:")
            logger.info(f"    Text preview: {text[:100]}...")
            logger.info(f"    FinBERT says: {label.upper()} (confidence: {score:.3f})")
            
            # Count sentiments
            if label in sentiment_counts:
                sentiment_counts[label] += 1
            
            # Convert to score between -1 and 1
            if label == 'positive':
                sentiment_scores.append(score)
            elif label == 'negative':
                sentiment_scores.append(-score)
            else:  # neutral
                sentiment_scores.append(0)
        
        # Calculate overall sentiment
        avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        logger.info(f"\n📈 SENTIMENT SUMMARY FOR {ticker}:")
        logger.info(f"  Positive articles: {sentiment_counts['positive']}")
        logger.info(f"  Negative articles: {sentiment_counts['negative']}")
        logger.info(f"  Neutral articles: {sentiment_counts['neutral']}")
        logger.info(f"  Average sentiment score: {avg_score:.3f}")
        
        # Determine overall sentiment
        if sentiment_counts['positive'] > sentiment_counts['negative'] * 1.5:
            overall_sentiment = 'bullish'
        elif sentiment_counts['negative'] > sentiment_counts['positive'] * 1.5:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        logger.info(f"  OVERALL SENTIMENT: {overall_sentiment.upper()}")
        
        # Generate insights
        insights = [
            f"Analyzed {len(articles_to_analyze)} recent articles",
            f"Positive: {sentiment_counts['positive']}, Negative: {sentiment_counts['negative']}, Neutral: {sentiment_counts['neutral']}",
            f"Average sentiment score: {avg_score:.3f}"
        ]
        
        summary = f"FinBERT analysis of {len(articles_to_analyze)} articles shows {overall_sentiment} sentiment. "
        summary += f"Distribution: {sentiment_counts['positive']} positive, {sentiment_counts['negative']} negative, {sentiment_counts['neutral']} neutral."
        
    except Exception as e:
        logger.error(f"❌ Error in FinBERT analysis: {e}")
        overall_sentiment = 'neutral'
        avg_score = 0.0
        insights = ["Analysis error occurred"]
        summary = "Unable to complete sentiment analysis"
    
    return {
        'ticker': ticker,
        'company_name': ticker,
        'total_articles': len(articles),
        'sentiment': overall_sentiment,
        'sentiment_score': avg_score,
        'key_insights': insights,
        'recent_headlines': [art['title'] for art in articles[:5]],
        'summary': summary
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("\n" + "="*60)
    logger.info("🚀 FINBERT FINANCIAL ANALYSIS SERVICE STARTING")
    logger.info("="*60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Service will run without model - health check only")
    
    yield
    
    # Shutdown
    logger.info("\n🛑 Service shutting down...")

app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "FinBERT Financial News Analysis Service",
        "status": "running",
        "model_loaded": sentiment_pipeline is not None,
        "docs": "http://localhost:8000/docs"
    }

@app.post("/analyze_news", response_model=NewsAnalysisResponse)
async def analyze_news(request: NewsAnalysisRequest):
    """Analyze news articles using FinBERT"""
    
    logger.info("\n" + "="*60)
    logger.info("📨 NEW ANALYSIS REQUEST RECEIVED")
    logger.info("="*60)
    logger.info(f"Request parameters: {request.dict()}")
    
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Fetch news from database
        news_articles = get_news_from_db(request.ticker, request.days_back)
        
        if not news_articles:
            logger.warning("⚠️ No news articles found in database")
            return NewsAnalysisResponse(
                analysis_date=datetime.now().isoformat(),
                period_analyzed=f"Last {request.days_back} days" if request.days_back else "All time",
                total_articles_analyzed=0,
                companies=[],
                market_overview="No news articles found in database."
            )
        
        # Group articles by ticker
        articles_by_ticker = {}
        for article in news_articles:
            ticker = article['ticker']
            if ticker not in articles_by_ticker:
                articles_by_ticker[ticker] = []
            articles_by_ticker[ticker].append(article)
        
        logger.info(f"\n📊 ARTICLES GROUPED BY TICKER:")
        logger.info(f"Total unique tickers: {len(articles_by_ticker)}")
        
        # Sort by number of articles
        sorted_tickers = sorted(articles_by_ticker.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Analyze top companies
        company_analyses = []
        for ticker, articles in sorted_tickers[:10]:  # Limit to top 10
            analysis = analyze_sentiment_finbert(articles, ticker)
            company_analyses.append(CompanySentiment(**analysis))
        
        # Market overview
        bullish = sum(1 for c in company_analyses if c.sentiment == 'bullish')
        bearish = sum(1 for c in company_analyses if c.sentiment == 'bearish')
        neutral = sum(1 for c in company_analyses if c.sentiment == 'neutral')
        
        logger.info(f"\n📊 FINAL MARKET ANALYSIS:")
        logger.info(f"  Companies analyzed: {len(company_analyses)}")
        logger.info(f"  Bullish: {bullish}")
        logger.info(f"  Bearish: {bearish}")
        logger.info(f"  Neutral: {neutral}")
        
        avg_sentiment = sum(c.sentiment_score for c in company_analyses) / len(company_analyses) if company_analyses else 0
        
        market_overview = f"FinBERT analyzed {len(company_analyses)} companies. "
        market_overview += f"Market breakdown: {bullish} bullish, {bearish} bearish, {neutral} neutral. "
        
        if avg_sentiment > 0.2:
            market_overview += "Overall market sentiment is positive."
        elif avg_sentiment < -0.2:
            market_overview += "Overall market sentiment is negative."
        else:
            market_overview += "Overall market sentiment is neutral."
        
        logger.info(f"  Average market sentiment: {avg_sentiment:.3f}")
        logger.info(f"\n✅ ANALYSIS COMPLETE")
        
        return NewsAnalysisResponse(
            analysis_date=datetime.now().isoformat(),
            period_analyzed=f"Last {request.days_back} days" if request.days_back else "All time",
            total_articles_analyzed=len(news_articles),
            companies=company_analyses,
            market_overview=market_overview
        )
        
    except Exception as e:
        logger.error(f"❌ Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-sentiment")
async def test_sentiment():
    """Test FinBERT with sample texts"""
    logger.info("\n🧪 TESTING FINBERT WITH SAMPLE TEXTS")
    
    if sentiment_pipeline is None:
        return {"error": "Model not loaded"}
    
    test_texts = [
        "Apple stock surges to record high on strong earnings",
        "Tesla shares plummet amid production concerns",
        "Microsoft maintains steady growth in cloud division",
        "Market crash fears as inflation rises",
        "Amazon beats earnings expectations"
    ]
    
    results = []
    for text in test_texts:
        logger.info(f"\nAnalyzing: '{text}'")
        sentiment = sentiment_pipeline(text)[0]
        logger.info(f"Result: {sentiment['label']} (confidence: {sentiment['score']:.3f})")
        
        results.append({
            "text": text,
            "sentiment": sentiment['label'],
            "score": sentiment['score']
        })
    
    return {"test_results": results}

if __name__ == "__main__":
    logger.info("Starting FinBERT Financial Analysis service on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
