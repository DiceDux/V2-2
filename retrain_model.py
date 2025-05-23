import schedule
import time
import logging
from datetime import datetime, timedelta
from train_ai_model_multi_advanced import main
from data.data_manager import get_connection
from sqlalchemy import text
import pandas as pd
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
import joblib
import os
from config import SYMBOLS
import numpy as np
import smtplib
from email.mime.text import MIMEText

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# تنظیمات ایمیل (برای اطلاع‌رسانی خطا)
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your_email@gmail.com',
    'sender_password': 'your_app_password',
    'receiver_email': 'receiver_email@gmail.com'
}

def send_error_notification(error_message: str):
    """ارسال ایمیل اطلاع‌رسانی در صورت خطا"""
    try:
        msg = MIMEText(f"خطا در فرآیند بازآموزش:\n{error_message}")
        msg['Subject'] = 'خطا در بازآموزش مدل'
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = EMAIL_CONFIG['receiver_email']
        
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)
        logger.info("ایمیل اطلاع‌رسانی خطا ارسال شد.")
    except Exception as e:
        logger.error(f"خطا در ارسال ایمیل اطلاع‌رسانی: {e}")

def save_training_history(success: bool, warm_start: bool, accuracy: float = None):
    """ذخیره تاریخچه آموزش در دیتابیس"""
    try:
        with get_connection() as conn:
            query = """
                INSERT INTO training_history (timestamp, success, warm_start, accuracy)
                VALUES (:timestamp, :success, :warm_start, :accuracy)
            """
            conn.execute(text(query), {
                'timestamp': datetime.utcnow(),
                'success': success,
                'warm_start': warm_start,
                'accuracy': accuracy
            })
        logger.info("تاریخچه آموزش ذخیره شد.")
    except Exception as e:
        logger.error(f"خطا در ذخیره تاریخچه آموزش: {e}")

def check_last_training(max_days: int = 1) -> bool:
    """چک کردن تاریخ آخرین آموزش موفق"""
    try:
        with get_connection() as conn:
            query = """
                SELECT timestamp
                FROM training_history
                WHERE success = TRUE
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = conn.execute(text(query)).fetchone()
        if not result:
            logger.warning("هیچ آموزش موفقی یافت نشد. بازآموزش شروع می‌شود...")
            return True
        last_training_date = result[0]
        days_passed = (datetime.utcnow() - last_training_date).days
        if days_passed > max_days:
            logger.warning(f"بیش از {days_passed} روز از آخرین آموزش گذشته. بازآموزش شروع می‌شود...")
            return True
        return False
    except Exception as e:
        logger.error(f"خطا در چک کردن تاریخ آموزش: {e}")
        return True

def check_feature_changes():
    """چک کردن تغییرات در ساختار فیچرها"""
    try:
        with open('models/catboost_pro_features.txt', 'r') as f:
            current_features = set(line.strip() for line in f.readlines())
        
        with get_connection() as conn:
            query = """
                SELECT features
                FROM training_history
                WHERE success = TRUE
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = conn.execute(text(query)).fetchone()
        
        if not result:
            logger.warning("هیچ فیچر قبلی یافت نشد. آموزش با warm_start=False انجام می‌شود...")
            return True
        
        previous_features = set(result[0].split(','))
        if current_features != previous_features:
            logger.warning(f"تغییر در فیچرها: اضافه‌شده={current_features - previous_features}, حذف‌شده={previous_features - current_features}")
            return True
        return False
    except Exception as e:
        logger.error(f"خطا در چک کردن تغییرات فیچرها: {e}")
        return True

def evaluate_model_performance() -> bool:
    """ارزیابی عملکرد مدل فعلی روی داده‌های جدید"""
    try:
        model = joblib.load('models/catboost_pro_model.pkl')
        df = pd.DataFrame()
        for symbol in SYMBOLS[:1]:  # فقط یک نماد برای سرعت
            from train_ai_model_multi_advanced import get_candle_data, extract_features_full, label_data
            df_symbol = get_candle_data(symbol=symbol, limit=100)
            if df_symbol.empty:
                continue
            df_symbol['symbol'] = symbol
            df_features = extract_features_full(df_symbol, symbol=symbol.replace("USDT", ""))
            df_labeled = label_data(df_features)
            df = pd.concat([df, df_labeled])
        
        if df.empty:
            logger.warning("داده‌ای برای ارزیابی عملکرد یافت نشد.")
            return True
        
        X = df.drop(columns=['label', 'future_return', 'symbol', 'close', 'open', 'high', 'low', 'volume', 'timestamp', 'interval'])
        y = df['label'].map({'Sell': 0, 'Hold': 1, 'Buy': 2})
        
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.mean(), inplace=True)
        
        test_pool = Pool(X, y)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        logger.info(f"دقت مدل فعلی: {accuracy:.4f}")
        if accuracy < 0.6:  # آستانه دقت
            logger.warning(f"دقت مدل ({accuracy:.4f}) کمتر از 0.6 است. بازآموزش شروع می‌شود...")
            return True
        return False
    except Exception as e:
        logger.error(f"خطا در ارزیابی عملکرد مدل: {e}")
        return True

def retrain_job(max_retries: int = 3):
    """اجرای فرآیند بازآموزش با retry"""
    logger.info("شروع فرآیند retraining...")
    attempt = 0
    while attempt < max_retries:
        try:
            # تصمیم‌گیری درباره بازآموزش
            need_retrain = check_last_training() or check_feature_changes() or evaluate_model_performance()
            if not need_retrain:
                logger.info("مدل به‌روز است. نیازی به retraining نیست.")
                return
            
            warm_start = not check_feature_changes()  # warm_start فقط اگر فیچرها تغییر نکرده باشن
            main(warm_start=warm_start)
            
            # ذخیره فیچرهای جدید در دیتابیس
            with open('models/catboost_pro_features.txt', 'r') as f:
                features = ','.join(line.strip() for line in f.readlines())
            with get_connection() as conn:
                query = """
                    UPDATE training_history
                    SET features = :features
                    WHERE timestamp = (SELECT MAX(timestamp) FROM training_history)
                """
                conn.execute(text(query), {'features': features})
            
            # ارزیابی دقت مدل جدید
            model = joblib.load('models/catboost_pro_model.pkl')
            accuracy = evaluate_model_performance()
            
            save_training_history(success=True, warm_start=warm_start, accuracy=accuracy)
            logger.info(f"Retraining با موفقیت انجام شد (warm_start={warm_start}, accuracy={accuracy:.4f}).")
            return
        
        except Exception as e:
            attempt += 1
            logger.error(f"خطا در تلاش {attempt}/{max_retries} برای retraining: {e}")
            if attempt == max_retries:
                send_error_notification(str(e))
                save_training_history(success=False, warm_start=False)
                logger.error("حداکثر تلاش‌ها انجام شد. بازآموزش ناموفق بود.")
            time.sleep(60)  # انتظار قبل از تلاش بعدی

# زمان‌بندی برای اجرا هر 12 ساعت
schedule.every(12).hours.do(retrain_job)

if __name__ == "__main__":
    logger.info("شروع برنامه retraining...")
    retrain_job()  # اجرای اولیه
    while True:
        schedule.run_pending()
        time.sleep(60)  # چک کردن هر دقیقه