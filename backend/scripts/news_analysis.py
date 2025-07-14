import os
import sys
import platform
import psutil
import cpuinfo
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import uvicorn
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
current_pipeline = None
current_model_name = None
system_info = {}

# Available models
AVAILABLE_MODELS = {
    "finbert": {
        "name": "ProsusAI/finbert",
        "description": "FinBERT - Financial Sentiment Analysis",
        "task": "sentiment-analysis",
        "size_mb": 440
    },
    "finbert-tone": {
        "name": "yiyanghkust/finbert-tone",
        "description": "FinBERT-Tone - More nuanced financial sentiment",
        "task": "sentiment-analysis", 
        "size_mb": 440
    },
    "distilbert-financial": {
        "name": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        "description": "DistilRoBERTa - Faster, smaller financial sentiment",
        "task": "sentiment-analysis",
        "size_mb": 320
    },
    "twitter-roberta": {
        "name": "cardiffnlp/twitter-roberta-base-sentiment",
        "description": "RoBERTa - Good for social media financial sentiment",
        "task": "sentiment-analysis",
        "size_mb": 480
    }
}

# Request/Response models
class NewsAnalysisRequest(BaseModel):
    ticker: Optional[str] = None
    days_back: Optional[int] = None
    analysis_type: str = "summary"
    model: str = "finbert"  # Allow model selection

class ModelSwitchRequest(BaseModel):
    model: str

class SystemInfo(BaseModel):
    hostname: str
    cpu: str
    cpu_cores: int
    ram_gb: float
    gpu: Optional[str]
    gpu_memory_gb: Optional[float]
    cuda_available: bool
    pytorch_device: str
    os_info: str

def detect_system():
    """Detect system specifications"""
    global system_info
    
    logger.info("="*60)
    logger.info("🖥️  DETECTING SYSTEM SPECIFICATIONS")
    logger.info("="*60)
    
    info = {
        "hostname": platform.node(),
        "os_info": f"{platform.system()} {platform.release()}",
        "cpu": cpuinfo.get_cpu_info()['brand_raw'],
        "cpu_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1)
    }
    
    # Detect GPU
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        info["pytorch_device"] = "cuda"
        
        # Identify GPU type
        gpu_name = info["gpu"].lower()
        if "4090" in gpu_name:
            logger.info("🎮 NVIDIA RTX 4090 detected - Excellent for all models!")
        elif "4070" in gpu_name:
            logger.info("🎮 NVIDIA RTX 4070 detected - Great for all models!")
        elif any(amd in gpu_name for amd in ["6900", "radeon", "amd"]):
            logger.info("🎮 AMD GPU detected - Will use CPU or DirectML")
            info["pytorch_device"] = "cpu"  # Default to CPU for AMD
    else:
        info["cuda_available"] = False
        info["gpu"] = None
        info["gpu_memory_gb"] = None
        info["pytorch_device"] = "cpu"
    
    # Try DirectML for AMD GPUs
    try:
        import torch_directml
        if torch_directml.is_available():
            info["directml_available"] = True
            info["pytorch_device"] = "directml"
            logger.info("✅ DirectML available for AMD GPU acceleration")
    except:
        info["directml_available"] = False
    
    # CPU identification
    cpu_name = info["cpu"].lower()
    if "5800x3d" in cpu_name:
        logger.info("🚀 AMD 5800X3D detected - Excellent CPU performance!")
    elif "7600x" in cpu_name:
        logger.info("🚀 AMD 7600X detected - Great CPU performance!")
    elif "7800x3d" in cpu_name:
        logger.info("🚀 AMD 7800X3D detected - Top-tier CPU performance!")
    
    # Log system info
    logger.info(f"\n📊 SYSTEM SUMMARY:")
    logger.info(f"  Hostname: {info['hostname']}")
    logger.info(f"  OS: {info['os_info']}")
    logger.info(f"  CPU: {info['cpu']} ({info['cpu_cores']} cores)")
    logger.info(f"  RAM: {info['ram_gb']} GB")
    if info['gpu']:
        logger.info(f"  GPU: {info['gpu']} ({info['gpu_memory_gb']} GB)")
    logger.info(f"  PyTorch Device: {info['pytorch_device']}")
    
    system_info = info
    return info

def get_optimal_device():
    """Get optimal device based on system"""
    if system_info.get("cuda_available"):
        return 0  # CUDA GPU
    elif system_info.get("directml_available"):
        try:
            import torch_directml
            return torch_directml.default_device()
        except:
            return -1  # CPU
    else:
        return -1  # CPU

def load_model(model_key: str = "finbert"):
    """Load specified model with auto-device selection"""
    global current_pipeline, current_model_name
    
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_key} not found. Available: {list(AVAILABLE_MODELS.keys())}")
    
    model_info = AVAILABLE_MODELS[model_key]
    logger.info(f"\n{'='*50}")
    logger.info(f"📦 LOADING MODEL: {model_info['description']}")
    logger.info(f"{'='*50}")
    
    try:
        device = get_optimal_device()
        device_name = "GPU" if device >= 0 else "CPU"
        if system_info.get("directml_available") and device != -1:
            device_name = "DirectML"
        
        logger.info(f"Loading on {device_name}...")
        
        # Load model
        current_pipeline = pipeline(
            model_info['task'],
            model=model_info['name'],
            device=device
        )
        
        current_model_name = model_key
        
        # Test performance
        logger.info("Testing model performance...")
        test_text = ["Testing inference speed for financial sentiment"]
        import time
        start = time.time()
        _ = current_pipeline(test_text)
        elapsed = time.time() - start
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"  Model: {model_info['name']}")
        logger.info(f"  Device: {device_name}")
        logger.info(f"  Test inference: {elapsed*1000:.1f}ms")
        
        # Memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"  GPU memory used: {allocated:.2f} GB")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise

def get_company_name(ticker: str, db_path: str) -> str:
    """Fetch company name from tickers table."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM tickers WHERE ticker = ?", (ticker.upper(),))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else ticker
    except:
        logger.error(f"Error fetching company name for {ticker}")
        return ticker

def analyze_sentiment_universal(articles: List[Dict], ticker: str) -> Dict:
    """Analyze sentiment with current model and provide detailed output."""
    if not articles:
        return {
            'ticker': ticker,
            'company_name': get_company_name(ticker, db_path='data/polygon_market_data.db'),
            'total_articles': 0,
            'sentiment': 'neutral',
            'sentiment_score': 0.0,
            'key_insights': ['No articles found for analysis'],
            'recent_headlines': [],
            'summary': 'No news articles available for this ticker.',
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'confidence_scores': []
        }

    # Prepare texts (truncate to avoid model input length limits)
    texts_to_analyze = []
    articles_to_analyze = articles[:20]  # Limit to 20 articles for performance
    for article in articles_to_analyze:
        text = article.get('title', '')
        if article.get('description'):
            text += f" {article['description'][:200]}"
        texts_to_analyze.append(text[:512])  # Truncate to 512 characters

    try:
        # Run sentiment analysis
        results = current_pipeline(texts_to_analyze, truncation=True, max_length=512)

        # Process results
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiment_scores = []
        confidence_scores = []

        for result in results:
            label = result['label'].lower()
            score = result['score']
            confidence_scores.append(score)

            # Normalize labels from different models
            if 'pos' in label or 'positive' in label:
                sentiment_counts['positive'] += 1
                sentiment_scores.append(score)
            elif 'neg' in label or 'negative' in label:
                sentiment_counts['negative'] += 1
                sentiment_scores.append(-score)
            else:
                sentiment_counts['neutral'] += 1
                sentiment_scores.append(0)

        # Calculate overall sentiment
        avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        if sentiment_counts['positive'] > sentiment_counts['negative'] * 1.5:
            overall_sentiment = 'bullish'
        elif sentiment_counts['negative'] > sentiment_counts['positive'] * 1.5:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'

        # Generate insights
        insights = [
            f"Model: {AVAILABLE_MODELS[current_model_name]['description']}",
            f"Analyzed {len(articles_to_analyze)} articles",
            f"Sentiment distribution: +{sentiment_counts['positive']} / -{sentiment_counts['negative']} / ={sentiment_counts['neutral']}",
            f"Average confidence: {sum(confidence_scores)/len(confidence_scores):.2f}"
        ]

        # Generate summary
        summary = (f"Analyzed {len(articles_to_analyze)} articles for {ticker}. "
                   f"Sentiment: {overall_sentiment} (Score: {avg_score:.2f}). "
                   f"Positive: {sentiment_counts['positive']}, Negative: {sentiment_counts['negative']}, Neutral: {sentiment_counts['neutral']}.")

        return {
            'ticker': ticker,
            'company_name': ticker,  # TODO: Fetch from database
            'total_articles': len(articles),
            'sentiment': overall_sentiment,
            'sentiment_score': avg_score,
            'key_insights': insights,
            'recent_headlines': [art.get('title', 'No title') for art in articles[:5]],
            'summary': summary,
            'sentiment_distribution': sentiment_counts,
            'confidence_scores': confidence_scores
        }

    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return {
            'ticker': ticker,
            'company_name': ticker,
            'total_articles': len(articles),
            'sentiment': 'neutral',
            'sentiment_score': 0.0,
            'key_insights': [f"Analysis error: {str(e)}"],
            'recent_headlines': [art.get('title', 'No title') for art in articles[:5]],
            'summary': 'Unable to complete analysis due to an error.',
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'confidence_scores': []
        }

def get_news_from_db(ticker: Optional[str] = None, days_back: Optional[int] = None) -> List[Dict]:
    """Fetch news from SQLite database across date-suffixed tables."""
    db_paths = [
        os.path.join('..', 'data', 'polygon_market_data.db'),
        os.path.join('data', 'polygon_market_data.db'),
        'polygon_market_data.db'
    ]

    db_path = None
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            logger.info(f"Found database at: {path}")
            break

    if not db_path:
        logger.error("Database not found! Run fetch.py first to collect data.")
        return []

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get available news tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE 'news_%'
            ORDER BY name
        """)
        news_tables = [row['name'] for row in cursor.fetchall()]
        
        if not news_tables:
            logger.error("No news tables found in database.")
            conn.close()
            return []

        # Filter tables based on date range
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            news_tables = [
                table for table in news_tables
                if table.replace('news_', '') >= cutoff_date.strftime('%Y%m%d')
            ]

        # Build UNION query across relevant news tables
        articles = []
        for table in news_tables:
            if ticker:
                query = f"""
                    SELECT DISTINCT n.* 
                    FROM {table} n
                    JOIN news_tickers_{table.replace('news_', '')} nt ON n.id = nt.news_id
                    WHERE nt.ticker = ?
                """
                params = [ticker.upper()]
            else:
                query = f"SELECT * FROM {table}"
                params = []

            if days_back:
                query += " AND n.published_utc >= ?"
                params.append(cutoff_date.isoformat())

            query += " ORDER BY n.published_utc DESC LIMIT 100"
            cursor.execute(query, params)
            rows = cursor.fetchall()

            for row in rows:
                article = dict(row)
                # Parse JSON fields
                for field in ['tickers', 'keywords']:
                    if article.get(field):
                        try:
                            article[field] = json.loads(article[field])
                        except:
                            article[field] = []
                articles.append(article)

        logger.info(f"Retrieved {len(articles)} articles from database for ticker {ticker or 'all'}")
        return articles

    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()

def check_database_status():
    """Check database status and print summary"""
    db_path = 'polygon_complete_data.db'
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check record counts
        cursor.execute("SELECT COUNT(*) FROM news")
        news_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT ticker) FROM news_tickers")
        ticker_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(published_utc), MAX(published_utc) FROM news")
        date_range = cursor.fetchone()
        
        logger.info(f"\n📊 Database Status:")
        logger.info(f"  Total news articles: {news_count:,}")
        logger.info(f"  Unique tickers with news: {ticker_count}")
        if date_range[0]:
            logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error checking database: {str(e)}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("\n" + "="*60)
    logger.info("🚀 UNIVERSAL FINANCIAL SENTIMENT SERVICE")
    logger.info("="*60)
    
    # Detect system
    detect_system()
    
    # Load default model
    try:
        load_model("finbert")
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
    
    yield
    
    # Shutdown
    logger.info("\n🛑 Service shutting down...")

app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Universal Financial Sentiment Analysis",
        "system": system_info.get("hostname", "Unknown"),
        "current_model": current_model_name,
        "available_models": list(AVAILABLE_MODELS.keys()),
        "device": system_info.get("pytorch_device", "Unknown")
    }

@app.get("/system-info", response_model=SystemInfo)
async def get_system_info():
    """Get detailed system information"""
    return SystemInfo(**system_info)

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "current_model": current_model_name,
        "available_models": AVAILABLE_MODELS,
        "can_use_gpu": system_info.get("cuda_available", False) or system_info.get("directml_available", False)
    }

@app.post("/switch-model")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model"""
    try:
        load_model(request.model)
        return {
            "status": "success",
            "message": f"Switched to {request.model}",
            "model_info": AVAILABLE_MODELS[request.model]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze_news")
async def analyze_news(request: NewsAnalysisRequest):
    """Analyze news with selected or current model."""
    # Switch model if requested
    if request.model != current_model_name:
        try:
            load_model(request.model)
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to switch model: {str(e)}")

    # Fetch news articles
    articles = get_news_from_db(ticker=request.ticker, days_back=request.days_back)

    # Perform sentiment analysis
    analysis_result = analyze_sentiment_universal(articles, request.ticker or "ALL")

    # Add analysis type handling
    if request.analysis_type == "detailed":
        # Add more detailed analysis (e.g., keyword extraction, trend analysis)
        analysis_result['detailed_analysis'] = {
            'keywords': [keyword for article in articles[:5] for keyword in article.get('keywords', [])],
            'publishers': list(set(article.get('publisher_name', 'Unknown') for article in articles))
        }
    elif request.analysis_type != "summary":
        raise HTTPException(status_code=400, detail="Invalid analysis_type. Use 'summary' or 'detailed'.")

    return {
        "status": "success",
        "model_used": current_model_name,
        "system": system_info.get("hostname"),
        "device": system_info.get("pytorch_device"),
        "analysis": analysis_result
    }
    
@app.get("/benchmark")
async def benchmark_models():
    """Benchmark all models on current system"""
    results = {}
    test_texts = ["Financial sentiment test"] * 10
    
    for model_key in AVAILABLE_MODELS:
        try:
            load_model(model_key)
            
            import time
            start = time.time()
            _ = current_pipeline(test_texts)
            elapsed = time.time() - start
            
            results[model_key] = {
                "status": "success",
                "total_time": elapsed,
                "per_text_ms": (elapsed / len(test_texts)) * 1000,
                "device": system_info.get("pytorch_device")
            }
        except Exception as e:
            results[model_key] = {
                "status": "failed",
                "error": str(e)
            }
    
    return results