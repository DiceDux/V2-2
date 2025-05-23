from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import re
import logging
import unicodedata
from functools import lru_cache
from typing import Union, List
import numpy as np
from unidecode import unidecode
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from data.data_manager import get_connection
import pandas as pd
from sqlalchemy import text
from collections import Counter
import gc

# دانلود منابع NLTK
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"خطا در دانلود منابع NLTK: {e}")

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# دیکشنری اولیه کلمات کلیدی
KEYWORD_WEIGHTS = {
    'positive': {
        'bullish': 0.8, 'surge': 0.7, 'rally': 0.7, 'adoption': 0.6, 
        'partnership': 0.6, 'listing': 0.5, 'upgrade': 0.5, 'etf': 0.6
    },
    'negative': {
        'bearish': -0.8, 'crash': -0.7, 'scam': -0.7, 'hack': -0.6, 
        'ban': -0.6, 'delisting': -0.5, 'regulation': -0.5, 'lawsuit': -0.6
    }
}

# لود مدل FinBERT
try:
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
except Exception as e:
    logger.error(f"خطا در بارگذاری FinBERT: {e}")
    tokenizer = None
    model = None
    sentiment_pipeline = None

# لود تحلیل‌گر VADER
try:
    vader_analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    logger.error(f"خطا در بارگذاری VADER: {e}")
    vader_analyzer = None

# تنظیمات stopwords
STOPWORDS = set(stopwords.words('english')).union({'crypto', 'cryptocurrency', 'blockchain'})

@lru_cache(maxsize=10000)
def clean_text(text: str) -> str:
    """پاک‌سازی پیشرفته متن با پشتیبانی از زبان‌های مختلف"""
    try:
        if not isinstance(text, str):
            logger.warning(f"متن ورودی نامعتبر است: {type(text)}")
            return ""
        
        text = unicodedata.normalize('NFKC', text)
        text = unidecode(text)
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'#\w+|\@\w+', '', text)
        text = re.sub(r"[^\w\s.,!?']", "", text, flags=re.UNICODE)
        text = ' '.join(text.split())
        return text.strip()
    except Exception as e:
        logger.error(f"خطا در پاک‌سازی متن: {e}")
        return ""

def update_keyword_weights(texts: List[str], scores: List[float]):
    """به‌روزرسانی پویا دیکشنری کلمات کلیدی"""
    try:
        positive_texts = [t for t, s in zip(texts, scores) if s > 0.3]
        negative_texts = [t for t, s in zip(texts, scores) if s < -0.3]
        
        def extract_keywords(texts):
            words = []
            for text in texts:
                tokenized = word_tokenize(clean_text(text).lower())
                words.extend([w for w in tokenized if w not in STOPWORDS and len(w) > 3])
            return Counter(words).most_common(10)
        
        positive_keywords = extract_keywords(positive_texts)
        negative_keywords = extract_keywords(negative_texts)
        
        new_weights = {'positive': {}, 'negative': {}}
        for word, count in positive_keywords:
            if word not in KEYWORD_WEIGHTS['positive'] and word not in KEYWORD_WEIGHTS['negative']:
                new_weights['positive'][word] = min(0.4 * (count / 10), 0.4)
        for word, count in negative_keywords:
            if word not in KEYWORD_WEIGHTS['positive'] and word not in KEYWORD_WEIGHTS['negative']:
                new_weights['negative'][word] = max(-0.4 * (count / 10), -0.4)
        
        with get_connection() as conn:
            for category, keywords in new_weights.items():
                for word, weight in keywords.items():
                    query = """
                        INSERT INTO keyword_weights (word, category, weight, updated_at)
                        VALUES (:word, :category, :weight, NOW())
                        ON DUPLICATE KEY UPDATE weight = :weight, updated_at = NOW()
                    """
                    conn.execute(text(query), {'word': word, 'category': category, 'weight': weight})
        logger.info(f"کلمات کلیدی جدید ذخیره شدند: {new_weights}")
    except Exception as e:
        logger.error(f"خطا در به‌روزرسانی کلمات کلیدی: {e}")

def save_sentiment_stats(scores: List[float], influential_keywords: List[List[str]]):
    """ذخیره آمار توزیع امتیازات"""
    try:
        with get_connection() as conn:
            stats = {
                'positive': len([s for s in scores if s > 0.1]),
                'negative': len([s for s in scores if s < -0.1]),
                'neutral': len([s for s in scores if -0.1 <= s <= 0.1]),
                'mean_score': float(np.mean(scores)) if scores else 0.0,
                'top_keywords': ', '.join(set([kw for sublist in influential_keywords for kw in sublist])),
                'timestamp': pd.Timestamp.now()
            }
            df_stats = pd.DataFrame([stats])
            df_stats.to_sql('sentiment_stats', con=conn, if_exists='append', index=False)
        logger.info(f"آمار احساسات ذخیره شد: {stats}")
    except Exception as e:
        logger.error(f"خطا در ذخیره آمار احساسات: {e}")

def get_keyword_score(text: str) -> tuple[float, List[str]]:
    """محاسبه امتیاز بر اساس کلمات کلیدی"""
    try:
        text_lower = text.lower()
        score = 0.0
        influential_keywords = []
        for word, weight in KEYWORD_WEIGHTS['positive'].items():
            if word in text_lower:
                score += weight
                influential_keywords.append(word)
        for word, weight in KEYWORD_WEIGHTS['negative'].items():
            if word in text_lower:
                score += weight
                influential_keywords.append(word)
        return min(max(score, -1.0), 1.0), influential_keywords
    except Exception as e:
        logger.error(f"خطا در محاسبه امتیاز کلمات کلیدی: {e}")
        return 0.0, []

def analyze_sentiment(texts: Union[str, List[str]], batch_size: int = 32) -> Union[float, List[float]]:
    """تحلیل احساسات متن با FinBERT و VADER"""
    try:
        is_single = isinstance(texts, str)
        texts = [texts] if is_single else texts
        
        if not texts or not all(isinstance(t, str) for t in texts):
            logger.warning("متن‌های ورودی خالی یا نامعتبر هستند.")
            return 0.0 if is_single else [0.0] * len(texts)
        
        cleaned_texts = [clean_text(t) for t in texts]
        cleaned_texts = [t if t else "" for t in cleaned_texts]
        
        if not any(cleaned_texts):
            logger.warning("همه متن‌ها پس از پاک‌سازی خالی هستند.")
            return 0.0 if is_single else [0.0] * len(texts)
        
        scores = []
        influential_keywords = []
        
        if not sentiment_pipeline or not vader_analyzer:
            logger.error("یکی از مدل‌های FinBERT یا VADER لود نشده است.")
            return 0.0 if is_single else [0.0] * len(texts)
        
        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i:i + batch_size]
            valid_batch = [t[:512] for t in batch if t]
            
            if not valid_batch:
                scores.extend([0.0] * len(batch))
                influential_keywords.extend([[]] * len(batch))
                continue
            
            roberta_results = sentiment_pipeline(valid_batch)
            if not all(isinstance(r, dict) and 'label' in r and 'score' in r for r in roberta_results):
                logger.error(f"خروجی FinBERT نامعتبر است: {roberta_results}")
                scores.extend([0.0] * len(batch))
                influential_keywords.extend([[]] * len(batch))
                continue
            
            vader_scores = [vader_analyzer.polarity_scores(t)['compound'] for t in valid_batch]
            
            batch_idx = 0
            for j, (roberta_result, vader_score, orig_text) in enumerate(zip(roberta_results, vader_scores, batch)):
                if not orig_text:
                    scores.append(0.0)
                    influential_keywords.append([])
                    continue
                
                label = roberta_result['label']
                roberta_score = roberta_result['score']
                
                if label == 'positive':
                    score = roberta_score
                elif label == 'negative':
                    score = -roberta_score
                else:
                    keyword_score, keywords = get_keyword_score(orig_text)
                    score = 0.5 * vader_score + 0.3 * keyword_score
                    influential_keywords.append(keywords)
                
                if any('regulation' in t.lower() or 'lawsuit' in t.lower() for t in orig_text.split()):
                    score *= 1.2
                
                scores.append(round(min(max(score, -1.0), 1.0), 2))
                if len(influential_keywords) <= j + i:
                    influential_keywords.append([])
                
                logger.debug(f"متن: {orig_text[:50]}..., FinBERT: {label} ({roberta_score:.2f}), VADER: {vader_score:.2f}, نهایی: {scores[-1]:.2f}")
                
                batch_idx += 1
            
            gc.collect()
        
        while len(scores) < len(texts):
            scores.append(0.0)
            influential_keywords.append([])
        
        update_keyword_weights(texts, scores)
        save_sentiment_stats(scores, influential_keywords)
        
        return scores[0] if is_single else scores
    
    except Exception as e:
        logger.error(f"خطا در تحلیل احساسات: {e}")
        return 0.0 if is_single else [0.0] * len(texts)