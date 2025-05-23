import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from config import MYSQL_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    try:
        connection_string = (
            f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
            f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
        )
        engine = create_engine(connection_string)
        logger.info("Engine دیتابیس با موفقیت ایجاد شد.")
        return engine
    except Exception as e:
        logger.error(f"خطا در اتصال به دیتابیس: {e}")
        return None

def get_fundamental_scores_batch(symbols, lookback_days=7):
    engine = get_db_connection()
    if not engine:
        return []

    # محدود کردن بازه زمانی به داده‌های کندل‌ها
    max_candle_ts = 1747252800  # 14 May 2025
    start_time = (datetime.fromtimestamp(max_candle_ts - lookback_days * 24 * 3600)).strftime('%Y-%m-%d %H:%M:%S')
    end_time = datetime.fromtimestamp(max_candle_ts).strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"بازه زمانی خبری: {start_time} تا {end_time}")
    results = []

    for symbol in symbols:
        logger.info(f"پردازش داده‌های فاندامنتال برای {symbol}...")
        news_symbol = symbol.replace('USDT', '')
        try:
            with engine.connect() as conn:
                # دریافت داده‌های خبری
                news_query = """
                    SELECT symbol, title, content, source, published_at
                    FROM news
                    WHERE symbol = :symbol
                    AND published_at BETWEEN :start_time AND :end_time
                """
                news_result = conn.execute(
                    text(news_query),
                    {"symbol": news_symbol, "start_time": start_time, "end_time": end_time}
                ).fetchall()
                news_data = [
                    {
                        "symbol": row[0],
                        "title": row[1],
                        "content": row[2],
                        "source": row[3],
                        "published_at": row[4]
                    }
                    for row in news_result
                ]

                if not news_data:
                    logger.warning("داده‌های خبری خالی هستند.")
                    news_score = 0.0
                else:
                    news_score = sum(len(row["content"] or "") for row in news_data) / len(news_data) / 1000.0
                    logger.info(f"امتیاز خبری برای {symbol}: {news_score}")

                # دریافت داده‌های بازار
                market_query = """
                    SELECT volume_score, funding_rate, btc_dominance, usdt_dominance, timestamp
                    FROM fundamental_data
                    WHERE symbol = :symbol
                    AND timestamp BETWEEN UNIX_TIMESTAMP(:start_time) * 1000 AND UNIX_TIMESTAMP(:end_time) * 1000
                """
                market_result = conn.execute(
                    text(market_query),
                    {"symbol": symbol, "start_time": start_time, "end_time": end_time}
                ).fetchall()
                market_data = [
                    {
                        "volume_score": row[0],
                        "funding_rate": row[1],
                        "btc_dominance": row[2],
                        "usdt_dominance": row[3],
                        "timestamp": row[4]
                    }
                    for row in market_result
                ]

                logger.info(f"دریافت {len(market_data)} داده بازار برای {symbol} از fundamental_data")

                # محاسبه و ذخیره امتیازات
                if not market_data and not news_data:
                    logger.warning("داده‌های خبری و بازار خالی هستند.")
                    result = {
                        "symbol": symbol,
                        "fundamental_score": 0.0,
                        "news_score": 0.0,
                        "volume_score": 0.0,
                        "funding_score": 0.0,
                        "btc_dominance": 0.0,
                        "usdt_dominance": 0.0,
                        "timestamp": max_candle_ts * 1000
                    }
                    results.append(result)
                else:
                    for market_row in market_data:
                        volume_score = market_row["volume_score"]
                        funding_score = market_row["funding_rate"] * 10000
                        btc_dominance = market_row["btc_dominance"]
                        usdt_dominance = market_row["usdt_dominance"]
                        fundamental_score = (news_score + volume_score / 1000000 + funding_score) / 3

                        result = {
                            "symbol": symbol,
                            "fundamental_score": fundamental_score,
                            "news_score": news_score,
                            "volume_score": volume_score,
                            "funding_score": funding_score,
                            "btc_dominance": btc_dominance,
                            "usdt_dominance": usdt_dominance,
                            "timestamp": market_row["timestamp"]
                        }
                        results.append(result)

                # ذخیره نتایج فقط اگه داده جدیدی داریم
                if news_data or market_data:
                    save_query = """
                        INSERT INTO fundamental_data (
                            symbol, fundamental_score, news_score, volume_score,
                            funding_score, btc_dominance, usdt_dominance, timestamp
                        )
                        VALUES (
                            :symbol, :fundamental_score, :news_score, :volume_score,
                            :funding_score, :btc_dominance, :usdt_dominance, :timestamp
                        )
                        ON DUPLICATE KEY UPDATE
                            fundamental_score = VALUES(fundamental_score),
                            news_score = VALUES(news_score),
                            volume_score = VALUES(volume_score),
                            funding_score = VALUES(funding_score),
                            btc_dominance = VALUES(btc_dominance),
                            usdt_dominance = VALUES(usdt_dominance)
                    """
                    for result in results:
                        conn.execute(text(save_query), result)
                    conn.commit()

        except Exception as e:
            logger.error(f"خطا در پردازش داده‌های فاندامنتال برای {symbol}: {e}")

    logger.debug(f"تعداد نتایج فاندامنتال برای {symbols}: {len(results)}")
    logger.info(f"امتیازات فاندامنتال: {len(results)} نتیجه برای {symbols}")
    return results