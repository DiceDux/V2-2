from ai.fundamental_analyzer import get_fundamental_scores_batch
import logging
from sqlalchemy import create_engine, text
from datetime import datetime
from config import MYSQL_CONFIG

# تنظیم لاگ‌گیری
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """ایجاد اتصال به دیتابیس"""
    try:
        connection_string = (
            f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
            f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
        )
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        logger.error(f"خطا در اتصال به دیتابیس: {e}")
        return None

def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT']
    # پیدا کردن بازه زمانی از fundamental_data
    engine = get_db_connection()
    if not engine:
        logger.error("نمی‌توان بازه زمانی را تعیین کرد.")
        return

    try:
        with engine.connect() as conn:
            query = "SELECT MIN(timestamp), MAX(timestamp) FROM fundamental_data WHERE symbol IN ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT')"
            result = conn.execute(text(query)).fetchone()
            start_time = result[0] if result[0] else int((datetime(2017, 9, 29).timestamp()) * 1000)
            end_time = result[1] if result[1] else int(datetime.utcnow().timestamp() * 1000)
    except Exception as e:
        logger.error(f"خطا در دریافت بازه زمانی: {e}")
        return

    # محاسبه امتیازات برای کل بازه
    lookback_days = (end_time - start_time) / (1000 * 60 * 60 * 24)  # تبدیل به روز
    results = get_fundamental_scores_batch(symbols, lookback_days=int(lookback_days))
    logger.info(f"امتیازات فاندامنتال برای {len(results)} ردیف محاسبه شد.")

if __name__ == "__main__":
    main()