import sys
import os
# اضافه کردن مسیر والد به sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3
import pandas as pd
import logging
from data_manager import save_candles_to_db

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_data():
    try:
        # اتصال به SQLite
        sqlite_conn = sqlite3.connect("candles_database.db")
        sqlite_cursor = sqlite_conn.cursor()

        # گرفتن همه نمادها
        sqlite_cursor.execute("SELECT DISTINCT symbol FROM candles")
        symbols = [row[0] for row in sqlite_cursor.fetchall()]
        logger.info(f"نمادهای موجود در SQLite: {symbols}")

        # انتقال داده‌ها برای هر نماد
        for symbol in symbols:
            logger.info(f"انتقال داده‌ها برای نماد {symbol}")

            # گرفتن داده‌ها برای نماد فعلی
            query = "SELECT symbol, timestamp, open, high, low, close, volume FROM candles WHERE symbol = ?"
            df = pd.read_sql_query(query, sqlite_conn, params=(symbol,))

            if df.empty:
                logger.warning(f"هیچ داده‌ای برای {symbol} در SQLite یافت نشد.")
                continue

            logger.info(f"تعداد ردیف‌ها برای {symbol}: {len(df)}")

            # انتقال داده‌ها به MySQL با استفاده از تابع save_candles_to_db
            save_candles_to_db(symbol, df, batch_size=1000)

        logger.info("انتقال داده‌ها به MySQL به پایان رسید.")

    except Exception as e:
        logger.error(f"خطا در انتقال داده‌ها: {str(e)}")
    finally:
        if sqlite_conn:
            sqlite_cursor.close()
            sqlite_conn.close()
            logger.info("اتصال به SQLite بسته شد.")

if __name__ == "__main__":
    migrate_data()