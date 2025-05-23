import logging
from datetime import datetime, timedelta
import time
import requests
from sqlalchemy import create_engine, text
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

def fetch_funding_rate(symbol, start_time, end_time, limit=100):
    funding_data = {}
    current_time = start_time
    while current_time < end_time:
        next_time = min(current_time + 7 * 24 * 60 * 60 * 1000, end_time)  # بازه 7 روزه
        funding_url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime={current_time}&endTime={next_time}&limit={limit}"
        try:
            response = requests.get(funding_url)
            response.raise_for_status()
            data = response.json()
            for fr in data:
                funding_data[int(fr['fundingTime'])] = float(fr['fundingRate'])
            logger.info(f"دریافت {len(data)} داده Funding Rate برای {symbol} از {current_time} تا {next_time}")
            current_time = next_time + 1
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"خطای HTTP در Funding Rate: {http_err}")
            break
        except Exception as e:
            logger.error(f"خطا در Funding Rate: {e}")
            break
    return funding_data

def fetch_ohlcv_data(symbol, start_time, end_time, limit=1000):
    """دریافت داده‌های OHLCV به‌صورت تکه‌تکه"""
    volume_data = {}
    current_time = start_time
    while current_time < end_time:
        ohlcv_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&startTime={current_time}&endTime={end_time}&limit={limit}"
        try:
            response = requests.get(ohlcv_url)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            for candle in data:
                volume_data[int(candle[0])] = float(candle[5])
            last_timestamp = max([int(candle[0]) for candle in data])
            logger.info(f"دریافت {len(data)} داده حجم برای {symbol} از {current_time} تا {last_timestamp}")
            current_time = last_timestamp + 3600 * 1000  # یه ساعت بعد
            time.sleep(0.5)  # جلوگیری از Rate Limit
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"خطای HTTP در OHLCV: {http_err}")
            break
        except Exception as e:
            logger.error(f"خطا در OHLCV: {e}")
            break
    return volume_data

def fetch_fundamental_data(symbol, start_time, end_time):
    try:
        start_time_ms = start_time
        end_time_ms = end_time
        logger.info(f"دریافت داده‌ها برای {symbol} از {start_time_ms} تا {end_time_ms}")

        # دریافت داده‌های OHLCV (حجم)
        volume_data = fetch_ohlcv_data(symbol, start_time_ms, end_time_ms)
        logger.info(f"دریافت {len(volume_data)} داده حجم برای {symbol}, از {min(volume_data.keys(), default=0)} تا {max(volume_data.keys(), default=0)}")

        # دریافت داده‌های Funding Rate
        funding_data = fetch_funding_rate(symbol, start_time_ms, end_time_ms)
        logger.info(f"دریافت {len(funding_data)} داده Funding Rate برای {symbol}, از {min(funding_data.keys(), default=0)} تا {max(funding_data.keys(), default=0)}")

        # Open Interest (موقتاً غیرفعال)
        oi_data = {}
        logger.info("داده‌های Open Interest غیرفعال شدند")

        # دریافت داده‌های Dominance
        dominance_url = "https://api.coingecko.com/api/v3/global"
        dominance_response = requests.get(dominance_url)
        dominance_response.raise_for_status()
        dominance_data = dominance_response.json()['data']['market_cap_percentage']
        btc_dominance = dominance_data.get('btc', 0.0)
        usdt_dominance = dominance_data.get('usdt', 0.0)
        logger.info(f"دریافت Dominance: btc={btc_dominance}, usdt={usdt_dominance}")

        # ترکیب داده‌ها
        timestamps = sorted(set(volume_data.keys()))
        data = []
        funding_times = sorted(funding_data.keys())
        for ts in timestamps:
            if ts < start_time_ms or ts > end_time_ms:
                continue
            # پیدا کردن نزدیک‌ترین Funding Rate
            funding_rate = 0.0
            if funding_times:
                closest_ft = min(funding_times, key=lambda x: abs(x - ts))
                if abs(closest_ft - ts) <= 8 * 60 * 60 * 1000:  # حداکثر 8 ساعت اختلاف
                    funding_rate = funding_data[closest_ft]
            record = {
                'symbol': symbol,
                'timestamp': ts,
                'volume_score': volume_data.get(ts, 0.0),
                'funding_rate': funding_rate,
                'open_interest': oi_data.get(ts, 0.0),
                'btc_dominance': btc_dominance,
                'usdt_dominance': usdt_dominance
            }
            data.append(record)

        logger.info(f"دریافت {len(data)} ردیف داده فاندامنتال برای {symbol}")
        return data
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"خطای HTTP در دریافت داده‌های فاندامنتال برای {symbol}: {http_err}")
        return []
    except Exception as e:
        logger.error(f"خطا در دریافت داده‌های فاندامنتال برای {symbol}: {e}")
        return []

def save_fundamental_data_raw(data):
    """ذخیره داده‌های خام فاندامنتال در دیتابیس"""
    if not data:
        logger.warning("داده‌ای برای ذخیره وجود ندارد.")
        return

    engine = get_db_connection()
    if not engine:
        return

    try:
        with engine.connect() as conn:
            query = """
                INSERT INTO fundamental_data (symbol, timestamp, volume_score, funding_rate, open_interest, btc_dominance, usdt_dominance)
                VALUES (:symbol, :timestamp, :volume_score, :funding_rate, :open_interest, :btc_dominance, :usdt_dominance)
                ON DUPLICATE KEY UPDATE
                volume_score = VALUES(volume_score),
                funding_rate = VALUES(funding_rate),
                open_interest = VALUES(open_interest),
                btc_dominance = VALUES(btc_dominance),
                usdt_dominance = VALUES(usdt_dominance)
            """
            for record in data:
                conn.execute(text(query), record)
            conn.commit()
        logger.info(f"{len(data)} ردیف داده فاندامنتال خام ذخیره شد.")
    except Exception as e:
        logger.error(f"خطا در ذخیره داده‌های فاندامنتال: {e}")
        if engine:
            with engine.connect() as conn:
                conn.rollback()

def main():
    """پر کردن جدول fundamental_data برای نمادها"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT']
    # پیدا کردن قدیمی‌ترین تایم‌استمپ از candle_data
    engine = get_db_connection()
    if not engine:
        logger.error("نمی‌توان بازه زمانی را تعیین کرد.")
        return

    try:
        with engine.connect() as conn:
            query = "SELECT MIN(timestamp) FROM candle_data WHERE symbol IN ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT')"
            result = conn.execute(text(query)).fetchone()
            start_time = result[0] if result[0] else int((datetime(2017, 9, 29).timestamp()) * 1000)  # از 2017-09-29
    except Exception as e:
        logger.error(f"خطا در دریافت قدیمی‌ترین تایم‌استمپ: {e}")
        start_time = int((datetime(2017, 9, 29).timestamp()) * 1000)  # پیش‌فرض: شروع اخبار

    end_time = int(datetime.utcnow().timestamp() * 1000)

    for symbol in symbols:
        logger.info(f"جمع‌آوری داده‌های فاندامنتال برای {symbol}...")
        data = fetch_fundamental_data(symbol, start_time, end_time)
        if data:
            save_fundamental_data_raw(data)
            logger.info(f"داده‌های فاندامنتال برای {symbol} ذخیره شدند.")
        else:
            logger.error(f"هیچ داده‌ای برای {symbol} دریافت نشد.")
        time.sleep(1)  # جلوگیری از Rate Limit

if __name__ == "__main__":
    main()