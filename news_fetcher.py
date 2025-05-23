import requests
from data.data_manager import insert_news, get_connection
from datetime import datetime, timedelta
import re
from ai.news_sentiment_ai import analyze_sentiment
import pandas as pd
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

SYMBOLS = {
    "SOL": ["solana", "SOL", "Sol"],
    "BTC": ["bitcoin", "BTC", "Btc"],
    "ETH": ["ethereum", "ETH", "Eth"],
    "DOGE": ["dogecoin", "DOGE", "Doge"]
}

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
if not API_KEY:
    logger.error("کلید API CryptoPanic در متغیرهای محیطی یافت نشد.")
    exit(1)

# شمارشگر برای محدود کردن لاگ‌های خطا
error_count = 0
MAX_ERROR_LOGS = 5

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    reraise=True
)
def fetch_news_from_api(url):
    """دریافت اخبار از API با مکانیزم retry"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def check_duplicate_news(symbol, title, published_at):
    """چک کردن اینکه آیا خبر قبلاً ذخیره شده است یا خیر"""
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False
        query = """
            SELECT COUNT(*) FROM news
            WHERE symbol = %s AND title = %s AND published_at = %s
        """
        df = pd.read_sql(query, con=conn, params=(symbol, title, published_at))
        return df.iloc[0, 0] > 0
    except Exception as e:
        logger.error(f"خطا در چک کردن خبر تکراری: {e}")
        return False
    finally:
        if conn:
            conn.close()

def fetch_news():
    global error_count
    logger.info("📰 در حال دریافت اخبار...")
    # فقط اخبار ۷ روز اخیر رو بگیر
    since_date = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={API_KEY}&public=true&kind=news&published_since={since_date}"

    try:
        data = fetch_news_from_api(url)
        logger.info(f"اخبار دریافت‌شده: {len(data.get('results', []))} خبر")
        
        if "results" not in data:
            logger.warning("⚠️ هیچ نتیجه‌ای یافت نشد.")
            return

        for item in data["results"]:
            title = item.get("title", "")
            content = item.get("content", title)
            if not title:
                continue

            source = item.get("domain", "unknown")
            published_at = item.get("published_at", datetime.utcnow().isoformat())
            # تبدیل published_at به فرمت datetime
            published_at_dt = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%S%z')
            published_at = published_at_dt.strftime('%Y-%m-%d %H:%M:%S')

            # جستجو در تمام نمادها
            for symbol, symbol_variants in SYMBOLS.items():
                base_symbol = symbol
                pattern = r'\b(' + '|'.join(re.escape(v) for v in symbol_variants) + r')\b'
                if re.search(pattern, title.upper()) or re.search(pattern, content.upper()):
                    if check_duplicate_news(base_symbol, title, published_at):
                        logger.info(f"⚠️ خبر تکراری برای {base_symbol} - {title[:40]}...")
                        continue

                    # محاسبه sentiment_score
                    sentiment_score = 0.0
                    if content and content.strip():
                        sentiment_score = analyze_sentiment(content)
                    else:
                        if error_count < MAX_ERROR_LOGS:
                            logger.warning(f"⚠️ محتوا یا عنوان خالی برای خبر: {title[:40]}...")
                            error_count += 1
                        elif error_count == MAX_ERROR_LOGS:
                            logger.warning("حداکثر تعداد لاگ‌های خطا رسیده است. لاگ‌های بعدی محدود می‌شوند.")
                            error_count += 1

                    # ذخیره خبر
                    insert_news(base_symbol, title, source, published_at, content, sentiment_score)

                    # فیلتر اخبار مهم
                    important_keywords = ['partnership', 'regulation', 'adoption', 'upgrade', 'halving']
                    is_important = any(keyword in title.lower() for keyword in important_keywords)
                    logger.info(f"✅ خبر ذخیره شد برای {base_symbol} - {title[:40]}... {'(مهم)' if is_important else ''}")

    except requests.exceptions.RequestException as e:
        if error_count < MAX_ERROR_LOGS:
            logger.error(f"❌ خطا در دریافت اخبار: {e}")
            error_count += 1
        elif error_count == MAX_ERROR_LOGS:
            logger.error("حداکثر تعداد لاگ‌های خطا رسیده است. لاگ‌های بعدی محدود می‌شوند.")
            error_count += 1
    except Exception as e:
        if error_count < MAX_ERROR_LOGS:
            logger.error(f"❌ خطای غیرمنتظره در fetch_news: {e}")
            error_count += 1
        elif error_count == MAX_ERROR_LOGS:
            logger.error("حداکثر تعداد لاگ‌های خطا رسیده است. لاگ‌های بعدی محدود می‌شوند.")
            error_count += 1

if __name__ == "__main__":
    fetch_news()