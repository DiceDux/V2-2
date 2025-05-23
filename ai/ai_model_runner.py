import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from data.data_manager import get_features_from_db, get_connection
from sqlalchemy import text
import logging
import json
from datetime import datetime, timedelta
import joblib
from config import SYMBOLS
from concurrent.futures import ThreadPoolExecutor, as_completed

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_features_list():
    """لود لیست ویژگی‌های مورد انتظار مدل"""
    try:
        with open('models/catboost_pro_features.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        logger.error("فایل ویژگی‌های مدل یافت نشد: models/catboost_pro_features.txt")
        return []

def calculate_scores(X: pd.DataFrame, technical_features: list, fundamental_features: list, embedding_features: list):
    """محاسبه امتیازهای تکنیکال، فاندامنتال و خبری"""
    try:
        tech_score = X[technical_features].mean(axis=1) if technical_features else pd.Series(0.0, index=X.index)
        tech_score = (tech_score - tech_score.min()) / (tech_score.max() - tech_score.min() + 1e-10)
        fund_score = X[fundamental_features].mean(axis=1) if fundamental_features else pd.Series(0.0, index=X.index)
        fund_score = (fund_score - fund_score.min()) / (fund_score.max() - fund_score.min() + 1e-10)
        news_score = X['news_score'] if 'news_score' in X.columns else pd.Series(0.0, index=X.index)
        news_score = (news_score - news_score.min()) / (news_score.max() - news_score.min() + 1e-10)
        return tech_score, fund_score, news_score
    except Exception as e:
        logger.error(f"خطا در محاسبه امتیازها: {e}")
        return pd.Series(0.0, index=X.index), pd.Series(0.0, index=X.index), pd.Series(0.0, index=X.index)

def save_prediction(symbol: str, timestamp: datetime, prediction: str, probability: float, news_impact: float, technical_score: float, fundamental_score: float):
    """ذخیره پیش‌بینی در دیتابیس"""
    try:
        with get_connection() as conn:
            query = """
                INSERT INTO predictions (
                    symbol, timestamp, prediction, probability, news_impact, technical_score, fundamental_score
                ) VALUES (:symbol, :timestamp, :prediction, :probability, :news_impact, :technical_score, :fundamental_score)
            """
            conn.execute(text(query), {
                'symbol': symbol,
                'timestamp': timestamp,
                'prediction': prediction,
                'probability': float(probability),
                'news_impact': float(news_impact),
                'technical_score': float(technical_score),
                'fundamental_score': float(fundamental_score)
            })
        logger.info(f"پیش‌بینی برای {symbol} در {timestamp} ذخیره شد.")
    except Exception as e:
        logger.error(f"خطا در ذخیره پیش‌بینی برای {symbol} در {timestamp}: {e}")

def process_symbol(symbol: str, model: CatBoostClassifier, expected_features: list, technical_features: list, fundamental_features: list, embedding_features: list, interval: str):
    """پردازش یک نماد به صورت مستقل"""
    try:
        logger.info(f"پردازش نماد {symbol}...")
        latest = get_features_from_db(symbol, interval)
        if latest.empty:
            logger.warning(f"داده‌ای برای {symbol} یافت نشد.")
            return {"symbol": symbol, "status": "no_data"}

        available_features = [f for f in expected_features if f in latest.columns]
        missing_features = [f for f in expected_features if f not in latest.columns]
        if missing_features:
            logger.warning(f"ویژگی‌های غایب برای {symbol}: {missing_features}")
            for f in missing_features:
                latest[f] = 0.0

        if not available_features:
            logger.warning(f"هیچ ویژگی معتبری برای {symbol} یافت نشد.")
            return {"symbol": symbol, "status": "no_valid_features"}

        latest = latest[available_features]
        latest.replace([np.inf, -np.inf], np.nan, inplace=True)
        latest.fillna(latest.mean(), inplace=True)

        for col in latest.columns:
            if latest[col].dtype not in [np.float64, np.int64]:
                try:
                    latest[col] = pd.to_numeric(latest[col], errors='coerce').fillna(0.0)
                except Exception as e:
                    logger.warning(f"ستون {col} به عدد تبدیل نشد: {e}. مقدار پیش‌فرض 0.0 استفاده می‌شود.")
                    latest[col] = 0.0

        available_technical = [f for f in technical_features if f in latest.columns]
        available_fundamental = [f for f in fundamental_features if f in latest.columns]
        available_embedding = [f for f in embedding_features if f in latest.columns]
        tech_score, fund_score, news_score = calculate_scores(latest, available_technical, available_fundamental, available_embedding)

        X = latest.to_numpy()
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        results = []
        for idx, (pred, prob, t_score, f_score, n_score) in enumerate(zip(predictions, probabilities, tech_score, fund_score, news_score)):
            timestamp = latest.index[idx] if latest.index.name == 'timestamp' else latest['timestamp'].iloc[idx]
            prediction = {0: 'Sell', 1: 'Hold', 2: 'Buy'}[pred]
            probability = float(prob.max())

            logger.info(
                f"پیش‌بینی برای {symbol} در {timestamp}: {prediction}, "
                f"اطمینان: {probability:.2f}, "
                f"امتیاز تکنیکال: {t_score:.2f}, "
                f"امتیاز فاندامنتال: {f_score:.2f}, "
                f"امتیاز خبری: {n_score:.2f}"
            )

            save_prediction(
                symbol=symbol,
                timestamp=timestamp,
                prediction=prediction,
                probability=probability,
                news_impact=n_score,
                technical_score=t_score,
                fundamental_score=f_score
            )
            results.append({
                "symbol": symbol,
                "timestamp": timestamp,
                "prediction": prediction,
                "probability": probability,
                "features": {
                    "news_score": float(n_score),
                    "technical_score": float(t_score),
                    "fundamental_score": float(f_score)
                }
            })

        return {"symbol": symbol, "status": "success", "results": results}

    except Exception as e:
        logger.error(f"خطا در پردازش {symbol}: {e}")
        return {"symbol": symbol, "status": "error", "error": str(e)}

def predict_signal_from_model(df: pd.DataFrame, symbol: str, interval: str, verbose: bool = False):
    """تولید سیگنال پیش‌بینی برای یک نماد"""
    try:
        model = joblib.load('models/catboost_pro_model.pkl')
        logger.info("مدل CatBoost با موفقیت لود شد.")
    except Exception as e:
        logger.error(f"خطا در لود مدل: {e}")
        return {"action": "NO_SIGNAL", "confidence": 0.0, "features": {}}

    expected_features = load_features_list()
    if not expected_features:
        logger.error("هیچ ویژگی‌ای برای مدل پیدا نشد.")
        return {"action": "NO_SIGNAL", "confidence": 0.0, "features": {}}

    fundamental_features = [
        'spx_index', 'dxy_index', 'btc_dominance', 'usdt_dominance', 'volume_score',
        'fundamental_score', 'news_score', 'market_sentiment',
        'funding_rate',
        'current_price', 'price_change',
        'btc_d', 'usdt_d', 'spx', 'dxy'
    ]
    trade_features = ['trade_success_rate', 'avg_profit', 'avg_confidence', 'trade_count']
    embedding_features = [f'pca_emb_{i}' for i in range(50)]
    technical_features = [
        f for f in expected_features
        if f not in fundamental_features + trade_features + embedding_features
    ]

    result = process_symbol(symbol, model, expected_features, technical_features, fundamental_features, embedding_features, interval)

    if result["status"] == "success" and result["results"]:
        latest_result = result["results"][-1]
        return {
            "action": latest_result["prediction"].lower(),
            "confidence": latest_result["probability"],
            "features": latest_result["features"]
        }
    return {"action": "NO_SIGNAL", "confidence": 0.0, "features": {}}

def main():
    """اجرای پیش‌بینی‌های مدل برای نمادها"""
    logger.info("شروع اجرای AI Model Runner...")
    try:
        model = joblib.load('models/catboost_pro_model.pkl')
        logger.info("مدل CatBoost با موفقیت لود شد.")
    except Exception as e:
        logger.error(f"خطا در لود مدل: {e}")
        return

    expected_features = load_features_list()
    if not expected_features:
        logger.error("هیچ ویژگی‌ای برای مدل پیدا نشد.")
        return

    fundamental_features = [
        'spx_index', 'dxy_index', 'btc_dominance', 'usdt_dominance', 'volume_score',
        'fundamental_score', 'news_score', 'market_sentiment',
        'funding_rate',
        'current_price', 'price_change',
        'btc_d', 'usdt_d', 'spx', 'dxy'
    ]
    trade_features = ['trade_success_rate', 'avg_profit', 'avg_confidence', 'trade_count']
    embedding_features = [f'pca_emb_{i}' for i in range(50)]
    technical_features = [
        f for f in expected_features
        if f not in fundamental_features + trade_features + embedding_features
    ]

    interval = '4h'
    max_workers = min(len(SYMBOLS), 8)  # تنظیم پویا workerها
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                process_symbol, symbol, model, expected_features, technical_features,
                fundamental_features, embedding_features, interval
            ): symbol
            for symbol in SYMBOLS
        }
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                logger.info(f"نتیجه برای {symbol}: {result}")
            except Exception as e:
                logger.error(f"خطا در پردازش {symbol}: {e}")