import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from data.data_manager import get_connection
from sqlalchemy import text
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INDEXES = {
    "BTC.D": "https://api.coingecko.com/api/v3/global",
    "USDT.D": "https://api.alternative.me/fng/",
    "SPX": "SPY",
    "DXY": "UUP"
}

# کلید API برای Alpha Vantage
ALPHA_VANTAGE_API_KEY = "XVJ5HCY0XJO1OR5Q"  # کلید خود را وارد کنید

def create_session():
    """ایجاد جلسه درخواست با تلاش مجدد"""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def fetch_btc_dominance():
    """دریافت داده‌های لحظه‌ای Bitcoin Dominance از Coingecko"""
    try:
        session = create_session()
        r = session.get(INDEXES["BTC.D"], timeout=10)
        r.raise_for_status()
        data = r.json()
        btc_dominance = data["data"]["market_cap_percentage"]["btc"]
        return float(btc_dominance)
    except Exception as e:
        logger.error(f"❌ خطا در دریافت BTC.D لحظه‌ای: {e}")
        return None

def fetch_historical_btc_dominance(start_date, end_date):
    """دریافت داده‌های تاریخی Bitcoin Dominance (تقریبی)"""
    try:
        btc_dominance = fetch_btc_dominance()
        if btc_dominance is None:
            logger.warning("داده BTC.D لحظه‌ای در دسترس نیست.")
            return []
        historical_data = []
        current_date = datetime.utcfromtimestamp(start_date)
        end = datetime.utcfromtimestamp(end_date)
        while current_date < end:
            timestamp = int(current_date.timestamp())
            created_at = current_date.strftime('%Y-%m-%d %H:%M:%S')
            historical_data.append((timestamp, btc_dominance, created_at))
            current_date += timedelta(days=1)
        logger.warning("⚠️ داده‌های تاریخی BTC.D تقریبی هستند (مقدار لحظه‌ای برای کل بازه).")
        return historical_data
    except Exception as e:
        logger.error(f"❌ خطا در دریافت داده‌های تاریخی BTC.D: {e}")
        return []

def fetch_usdt_dominance():
    """دریافت داده‌های لحظه‌ای Fear & Greed Index برای USDT.D"""
    try:
        session = create_session()
        r = session.get(INDEXES["USDT.D"], timeout=10)
        r.raise_for_status()
        return float(r.json()["data"][0]["value"])
    except Exception as e:
        logger.error(f"❌ خطا در دریافت USDT.D: {e}")
        return None

def fetch_historical_usdt_dominance(start_date, end_date):
    """دریافت داده‌های تاریخی Fear & Greed Index"""
    try:
        session = create_session()
        days = (datetime.utcfromtimestamp(end_date) - datetime.utcfromtimestamp(start_date)).days
        url = f"https://api.alternative.me/fng/?limit={days}"
        r = session.get(url, timeout=10)
        r.raise_for_status()
        historical_data = r.json()["data"]
        data = []
        for entry in historical_data:
            value = float(entry["value"])
            timestamp = int(entry["timestamp"])
            if start_date <= timestamp <= end_date:
                created_at = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                data.append((timestamp, value, created_at))
        return sorted(data, key=lambda x: x[0])
    except Exception as e:
        logger.error(f"❌ خطا در دریافت داده‌های تاریخی USDT.D: {e}")
        return []

def fetch_alpha_vantage_index(symbol, start_date, end_date):
    """دریافت داده‌های تاریخی از Alpha Vantage و پر کردن روزهای غیرمعاملاتی"""
    try:
        session = create_session()
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        logger.info(f"دریافت داده‌های {symbol} از {start_date} تا {end_date}")
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
        r = session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "Time Series (Daily)" not in data:
            logger.error(f"❌ خطای API Alpha Vantage برای {symbol}: {data.get('Error Message', 'Unknown error')}")
            return []

        # تبدیل داده‌ها به DataFrame
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame([
            {
                'date': datetime.strptime(date_str, '%Y-%m-%d'),
                'value': float(values["4. close"])
            }
            for date_str, values in time_series.items()
        ])
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        df = df.sort_values('date')

        if df.empty:
            logger.warning(f"⚠️ هیچ داده‌ای برای {symbol} در بازه زمانی مشخص‌شده دریافت نشد.")
            return []

        # تولید تمام روزهای تقویمی در بازه
        all_dates = pd.date_range(start=start, end=end, freq='D')
        full_df = pd.DataFrame(all_dates, columns=['date'])

        # ادغام داده‌ها
        merged_df = pd.merge(full_df, df, on='date', how='left')

        # پر کردن مقادیر NaN با استفاده از ffill و bfill
        if merged_df['value'].isna().all():
            logger.warning(f"⚠️ هیچ داده معتبری برای {symbol} دریافت نشد.")
            return []
        
        # ابتدا با bfill پر می‌کنیم تا مقادیر اولیه NaN با اولین مقدار معتبر پر شوند
        merged_df['value'] = merged_df['value'].bfill()
        # سپس با ffill پر می‌کنیم تا مقادیر باقی‌مانده NaN پر شوند
        merged_df['value'] = merged_df['value'].ffill()

        # تبدیل به فرمت مورد نظر
        historical_data = []
        for _, row in merged_df.iterrows():
            timestamp = int(row['date'].timestamp())
            value = float(row['value'])  # مطمئن می‌شویم که NaN وجود ندارد
            created_at = row['date'].strftime('%Y-%m-%d %H:%M:%S')
            historical_data.append({
                'timestamp': timestamp,
                'value': value,
                'created_at': created_at
            })
        
        return historical_data
    except Exception as e:
        logger.error(f"❌ خطا در دریافت داده‌های {symbol} از Alpha Vantage: {e}")
        return []

def save_index(index_name, value, timestamp, created_at=None):
    """ذخیره تک شاخص در دیتابیس"""
    conn = None
    try:
        if created_at is None:
            created_at = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        conn = get_connection()
        if not conn:
            logger.error("❌ خطا در اتصال به دیتابیس")
            return
        query = """
            INSERT INTO market_indices (index_name, timestamp, value, created_at)
            VALUES (:index_name, :timestamp, :value, :created_at)
            ON DUPLICATE KEY UPDATE value = VALUES(value)
        """
        conn.execute(
            text(query),
            {
                'index_name': index_name,
                'timestamp': timestamp,
                'value': value,
                'created_at': created_at
            }
        )
        conn.commit()
    except Exception as e:
        logger.error(f"❌ خطا در ذخیره {index_name}: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def save_index_batch(index_name, records):
    """ذخیره دسته‌ای شاخص‌ها در دیتابیس"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            logger.error("❌ خطا در اتصال به دیتابیس")
            return
        query = """
            INSERT INTO market_indices (index_name, timestamp, value, created_at)
            VALUES (:index_name, :timestamp, :value, :created_at)
            ON DUPLICATE KEY UPDATE value = VALUES(value)
        """
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            params = [
                {
                    'index_name': index_name,
                    'timestamp': record['timestamp'],
                    'value': record['value'],
                    'created_at': record['created_at']
                }
                for record in batch
            ]
            conn.execute(text(query), params)
            conn.commit()
        logger.info(f"✅ ذخیره شد {index_name}: {len(records)} رکورد")
    except Exception as e:
        logger.error(f"❌ خطا در ذخیره دسته‌ای {index_name}: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def check_stored_data():
    """بررسی تعداد رکوردهای ذخیره‌شده برای هر شاخص"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            logger.error("❌ خطا در اتصال به دیتابیس")
            return
        query = """
            SELECT index_name, COUNT(*), MIN(created_at), MAX(created_at)
            FROM market_indices
            GROUP BY index_name
        """
        df = pd.read_sql(query, con=conn)
        logger.info("📊 وضعیت داده‌های ذخیره‌شده:")
        for _, row in df.iterrows():
            logger.info(f"شاخص: {row['index_name']}, تعداد: {row['COUNT(*)']}, بازه: {row['MIN(created_at)']} تا {row['MAX(created_at)']}")
    except Exception as e:
        logger.error(f"❌ خطا در بررسی داده‌ها: {e}")
    finally:
        if conn:
            conn.close()

def run_initial():
    """حالت initial: جمع‌آوری داده‌های تاریخی از 2017 تا 2025"""
    logger.info("🚀 شروع حالت initial: جمع‌آوری داده‌های 2017 تا 2025")
    start_date = "2017-01-03"  # تاریخ شروع به اولین روز معاملاتی تغییر کرد
    end_date = "2025-05-16"
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

    # SPX
    logger.info("دریافت داده‌های SPX...")
    spx_data = fetch_alpha_vantage_index(INDEXES["SPX"], start_date, end_date)
    logger.info(f"تعداد رکوردهای SPX: {len(spx_data)}")
    save_index_batch("SPX", spx_data)
    time.sleep(12)  # تأخیر برای محدودیت نرخ Alpha Vantage

    # DXY
    logger.info("دریافت داده‌های DXY...")
    dxy_data = fetch_alpha_vantage_index(INDEXES["DXY"], start_date, end_date)
    logger.info(f"تعداد رکوردهای DXY: {len(dxy_data)}")
    save_index_batch("DXY", dxy_data)

    # BTC.D
    logger.info("دریافت داده‌های BTC.D...")
    btc_d_historical = fetch_historical_btc_dominance(start_ts, end_ts)
    logger.info(f"تعداد رکوردهای BTC.D: {len(btc_d_historical)}")
    save_index_batch("BTC.D", [{'timestamp': t, 'value': v, 'created_at': c} for t, v, c in btc_d_historical])

    # USDT.D
    logger.info("دریافت داده‌های USDT.D...")
    usdt_d_historical = fetch_historical_usdt_dominance(start_ts, end_ts)
    logger.info(f"تعداد رکوردهای USDT.D: {len(usdt_d_historical)}")
    save_index_batch("USDT.D", [{'timestamp': t, 'value': v, 'created_at': c} for t, v, c in usdt_d_historical])

def run_recovery():
    """حالت recovery: جمع‌آوری داده‌های تاریخی از 2017 تا 2025"""
    logger.info("🔄 شروع حالت recovery: جمع‌آوری داده‌های 2017 تا 2025")
    run_initial()

def run_live():
    """حالت live: به‌روزرسانی مداوم داده‌ها"""
    logger.info("🌐 شروع حالت live: به‌روزرسانی مداوم")
    while True:
        try:
            timestamp = int(time.time())
            # BTC.D
            btc_d = fetch_btc_dominance()
            if btc_d is not None:
                save_index("BTC.D", btc_d, timestamp)

            # USDT.D
            usdt_d = fetch_usdt_dominance()
            if usdt_d is not None:
                save_index("USDT.D", usdt_d, timestamp)

            # SPX و DXY
            today = datetime.utcnow().strftime('%Y-%m-%d')
            yesterday = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
            for index_name, symbol in [("SPX", INDEXES["SPX"]), ("DXY", INDEXES["DXY"])]:
                time.sleep(12)
                data = fetch_alpha_vantage_index(symbol, yesterday, today)
                if data:
                    latest = data[-1]
                    save_index(index_name, latest['value'], latest['timestamp'], latest['created_at'])

            logger.info("⏳ انتظار برای به‌روزرسانی بعدی...")
            time.sleep(86400)  # روزانه
        except Exception as e:
            logger.error(f"❌ خطا در حالت live: {e}")
            time.sleep(3600)  # 1 ساعت صبر

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Index Updater")
    parser.add_argument("--mode", choices=["initial", "recovery", "live"], default="live",
                        help="حالت اجرا: initial (2017-2025), recovery (2017-2025), live (به‌روزرسانی مداوم)")
    args = parser.parse_args()

    if args.mode == "initial":
        run_initial()
    elif args.mode == "recovery":
        run_recovery()
    elif args.mode == "live":
        run_live()

    check_stored_data()
    logger.info("✅ به‌روزرسانی شاخص‌ها به پایان رسید.")