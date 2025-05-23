import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from datetime import datetime
from config import MYSQL_CONFIG
import logging
import json
import time
import requests
from decimal import Decimal
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from functools import lru_cache
from feature_engineering_full_ultra_v2 import generate_sql_query

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ایجاد engine با connection pool
engine = None

def init_engine():
    global engine
    try:
        db_url = f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}"
        engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600
        )
        logger.info("Engine دیتابیس با موفقیت ایجاد شد.")
    except Exception as e:
        logger.error(f"خطا در ایجاد Engine دیتابیس: {e}")
        engine = None

def get_connection():
    global engine
    if engine is None:
        init_engine()
    if engine is None:
        logger.error("Engine دیتابیس ایجاد نشد.")
        return None
    try:
        conn = engine.connect()
        logger.debug("اتصال به دیتابیس برقرار شد.")
        return conn
    except Exception as e:
        logger.error(f"خطا در اتصال به دیتابیس: {e}")
        return None

from sqlalchemy import text
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def save_candles_to_db(symbol, df: pd.DataFrame, batch_size=1000):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            logger.error("اتصال به دیتابیس برقرار نشد.")
            return

        # تبدیل DataFrame به فرمت مناسب
        df = df.copy()
        df['symbol'] = symbol
        df = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # کوئری برای درج یا به‌روزرسانی
        update_query = """
            INSERT INTO candles (symbol, timestamp, open, high, low, close, volume)
            VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
            ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                volume = VALUES(volume)
        """

        # تبدیل DataFrame به لیست دیکشنری
        records = df.to_dict('records')
        total_records = len(records)

        # ارسال داده‌ها به صورت دسته‌ای
        for i in range(0, total_records, batch_size):
            batch = records[i:i + batch_size]
            with conn.begin():  # استفاده از تراکنش
                conn.execute(text(update_query), batch)
            logger.info(f"دسته {i // batch_size + 1} از {total_records // batch_size + 1} برای {symbol} ذخیره شد ({len(batch)} ردیف).")

        logger.info(f"داده‌های کندل {symbol} ذخیره شدند (کل: {total_records} ردیف).")
    except Exception as e:
        logger.error(f"خطا در ذخیره کندل: {e}")
    finally:
        if conn:
            conn.close()
            
def get_features_from_db(symbol, interval):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return pd.DataFrame()
        query = """
            SELECT * FROM features
            WHERE symbol = %s AND `interval` = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        df = pd.read_sql(query, conn, params=(symbol, interval))
        return df
    except Exception as e:
        logger.error(f"خطا در گرفتن ویژگی‌ها برای {symbol}: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    reraise=True
)
def fetch_binance_data(symbol="BTCUSDT"):
    try:
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        funding_rate = float(data['lastFundingRate'])

        url_oi = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        response_oi = requests.get(url_oi)
        response_oi.raise_for_status()
        data_oi = response_oi.json()
        open_interest = float(data_oi['openInterest'])

        logger.info(f"داده‌های بایننس برای {symbol} دریافت شد.")
        return {
            "funding_rate": funding_rate,
            "open_interest": open_interest,
            "timestamp": data['time']
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"خطا در دریافت داده از بایننس برای {symbol}: {e}")
        raise

def save_features(symbol, features):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        if 'timestamp' not in features:
            features['timestamp'] = int(datetime.utcnow().timestamp())

        features_dict = {k: float(v) if isinstance(v, Decimal) else v for k, v in features.items()}
        extra_features = {k: v for k, v in features_dict.items() if k not in [
            'symbol', 'timestamp', 'interval', 'open', 'high', 'low', 'close', 'volume', 'ema20', 'rsi', 'atr', 'ch',
            'trade_success_rate', 'avg_profit', 'avg_confidence', 'trade_count'
        ]}
        extra_features_json = json.dumps(extra_features, cls=DecimalEncoder)

        query = """
        INSERT INTO features (
            symbol, timestamp, `interval`, open, high, low, close, volume, ema20, rsi, atr, ch,
            trade_success_rate, avg_profit, avg_confidence, trade_count, extra_features
        )
        VALUES (:symbol, :timestamp, :`interval`, :open, :high, :low, :close, :volume, :ema20, :rsi, :atr, :ch,
                :trade_success_rate, :avg_profit, :avg_confidence, :trade_count, :extra_features)
        ON DUPLICATE KEY UPDATE
            `interval` = VALUES(`interval`),
            open = VALUES(open),
            high = VALUES(high),
            low = VALUES(low),
            close = VALUES(close),
            volume = VALUES(volume),
            ema20 = VALUES(ema20),
            rsi = VALUES(rsi),
            atr = VALUES(atr),
            ch = VALUES(ch),
            trade_success_rate = VALUES(trade_success_rate),
            avg_profit = VALUES(avg_profit),
            avg_confidence = VALUES(avg_confidence),
            trade_count = VALUES(trade_count),
            extra_features = VALUES(extra_features)
        """
        conn.execute(
            text(query),
            {
                'symbol': symbol,
                'timestamp': features_dict.get('timestamp', 0),
                'interval': features_dict.get('interval', None),
                'open': features_dict.get('open', 0.0),
                'high': features_dict.get('high', 0.0),
                'low': features_dict.get('low', 0.0),
                'close': features_dict.get('close', 0.0),
                'volume': features_dict.get('volume', 0.0),
                'ema20': features_dict.get('ema20', 0.0),
                'rsi': features_dict.get('rsi', 0.0),
                'atr': features_dict.get('atr', 0.0),
                'ch': features_dict.get('ch', 0.0),
                'trade_success_rate': features_dict.get('trade_success_rate', 0.0),
                'avg_profit': features_dict.get('avg_profit', 0.0),
                'avg_confidence': features_dict.get('avg_confidence', 0.0),
                'trade_count': features_dict.get('trade_count', 0.0),
                'extra_features': extra_features_json
            }
        )
        logger.info(f"ویژگی‌ها برای {symbol} با موفقیت در جدول features ذخیره شد.")
    except Exception as e:
        logger.error(f"خطا در ذخیره ویژگی‌ها برای {symbol}: {e}")
    finally:
        if conn:
            conn.close()

def insert_balance_to_db(symbol, balance):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        now = datetime.utcnow().isoformat()
        query = """
            INSERT INTO balance (symbol, balance, updated_at)
            VALUES (:symbol, :balance, :updated_at)
            ON DUPLICATE KEY UPDATE
                balance = VALUES(balance),
                updated_at = VALUES(updated_at)
        """
        conn.execute(
            text(query),
            {'symbol': "WALLET", 'balance': balance, 'updated_at': now}
        )
        logger.info(f"موجودی برای {symbol} ذخیره شد.")
    except Exception as e:
        logger.error(f"خطا در ذخیره موجودی: {e}")
    finally:
        if conn:
            conn.close()

def save_trade_record(symbol, action, price, balance, confidence, profit=None):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        now = datetime.utcnow().isoformat()
        query = """
            INSERT INTO trades (symbol, action, entry_price, confidence, balance, profit, timestamp)
            VALUES (:symbol, :action, :entry_price, :confidence, :balance, :profit, :timestamp)
        """
        conn.execute(
            text(query),
            {
                'symbol': symbol,
                'action': action.upper(),
                'entry_price': price,
                'confidence': confidence,
                'balance': balance,
                'profit': profit,
                'timestamp': now
            }
        )
        logger.info(f"ترید برای {symbol} ذخیره شد.")
    except Exception as e:
        logger.error(f"خطا در save_trade_record: {e}")
    finally:
        if conn:
            conn.close()

def has_open_trade(symbol):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return None
        query = "SELECT * FROM open_trades WHERE symbol = %s"
        df = pd.read_sql(query, con=conn, params=(symbol,))
        return df.to_dict('records')[0] if not df.empty else None
    except Exception as e:
        logger.error(f"خطا در چک کردن ترید باز برای {symbol}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def insert_open_trade(symbol, action, price):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        query = """
            INSERT INTO open_trades (symbol, action, entry_price, timestamp)
            VALUES (:symbol, :action, :entry_price, :timestamp)
        """
        conn.execute(
            text(query),
            {
                'symbol': symbol,
                'action': action,
                'entry_price': price,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        logger.info(f"ترید باز برای {symbol} ذخیره شد.")
    except Exception as e:
        logger.error(f"خطا در ذخیره ترید باز: {e}")
    finally:
        if conn:
            conn.close()

def close_open_trade(symbol):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        query = "DELETE FROM open_trades WHERE symbol = %s"
        conn.execute(text(query), {'symbol': symbol})
        logger.info(f"ترید باز برای {symbol} بسته شد.")
    except Exception as e:
        logger.error(f"خطا در بستن ترید باز: {e}")
    finally:
        if conn:
            conn.close()

def has_trade_in_current_candle(symbol, current_candle_time):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False
        query = """
            SELECT COUNT(*) as count FROM trades
            WHERE symbol = %s AND candle_time = %s
        """
        df = pd.read_sql(query, con=conn, params=(symbol, current_candle_time))
        return df['count'].iloc[0] > 0
    except Exception as e:
        logger.error(f"خطا در چک کردن ترید در کندل فعلی: {e}")
        return False
    finally:
        if conn:
            conn.close()

def insert_trade_with_candle(symbol, action, price, confidence, candle_time, mode, profit=None, exit_price=None, quantity=None):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        timestamp = datetime.utcnow().isoformat()
        query = """
            INSERT INTO trades (
                symbol, action, entry_price, confidence, timestamp, mode,
                candle_time, profit, exit_price, quantity
            )
            VALUES (:symbol, :action, :entry_price, :confidence, :timestamp, :mode,
                    :candle_time, :profit, :exit_price, :quantity)
        """
        conn.execute(
            text(query),
            {
                'symbol': symbol,
                'action': action,
                'entry_price': price,
                'confidence': confidence,
                'timestamp': timestamp,
                'mode': mode,
                'candle_time': candle_time,
                'profit': profit,
                'exit_price': exit_price,
                'quantity': quantity
            }
        )
        logger.info(f"ترید با کندل برای {symbol} ذخیره شد.")
    except Exception as e:
        logger.error(f"خطا در ذخیره ترید با کندل: {e}")
    finally:
        if conn:
            conn.close()

def insert_position(symbol, action, entry_price, quantity, tp_price, sl_price, tp_step=1, last_price=None):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        query = """
            INSERT INTO positions (symbol, action, entry_price, quantity, tp_price, sl_price, tp_step, last_price)
            VALUES (:symbol, :action, :entry_price, :quantity, :tp_price, :sl_price, :tp_step, :last_price)
        """
        conn.execute(
            text(query),
            {
                'symbol': symbol,
                'action': action,
                'entry_price': entry_price,
                'quantity': quantity,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'tp_step': tp_step,
                'last_price': last_price
            }
        )
        logger.info(f"پوزیشن برای {symbol} ذخیره شد.")
    except Exception as e:
        logger.error(f"خطا در ذخیره پوزیشن: {e}")
    finally:
        if conn:
            conn.close()

def get_position(symbol):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return None
        query = "SELECT * FROM positions WHERE symbol = %s LIMIT 1"
        df = pd.read_sql(query, con=conn, params=(symbol,))
        return df.to_dict('records')[0] if not df.empty else None
    except Exception as e:
        logger.error(f"خطا در گرفتن پوزیشن برای {symbol}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def delete_position(symbol):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        query = "DELETE FROM positions WHERE symbol = %s"
        conn.execute(text(query), {'symbol': symbol})
        logger.info(f"پوزیشن برای {symbol} حذف شد.")
    except Exception as e:
        logger.error(f"خطا در حذف پوزیشن: {e}")
    finally:
        if conn:
            conn.close()

def get_wallet_balance():
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return 0.0
        query = "SELECT balance FROM wallet LIMIT 1"
        df = pd.read_sql(query, con=conn)
        return df['balance'].iloc[0] if not df.empty else 0.0
    except Exception as e:
        logger.error(f"خطا در گرفتن موجودی کیف‌پول: {e}")
        return 0.0
    finally:
        if conn:
            conn.close()

def update_wallet_balance(new_balance):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        now = datetime.utcnow().isoformat()
        query = """
            UPDATE wallet SET balance = :balance, updated_at = :updated_at WHERE id = 1
        """
        conn.execute(
            text(query),
            {'balance': new_balance, 'updated_at': now}
        )
        logger.info(f"موجودی کیف‌پول به‌روزرسانی شد: {new_balance}")
    except Exception as e:
        logger.error(f"خطا در به‌روزرسانی موجودی کیف‌پول: {e}")
    finally:
        if conn:
            conn.close()

def update_position_trailing(symbol, new_tp, new_sl, new_step, last_price):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        query = """
            UPDATE positions
            SET tp_price = :tp_price, sl_price = :sl_price, tp_step = :tp_step, last_price = :last_price
            WHERE symbol = :symbol
        """
        conn.execute(
            text(query),
            {
                'tp_price': new_tp,
                'sl_price': new_sl,
                'tp_step': new_step,
                'last_price': last_price,
                'symbol': symbol
            }
        )
        logger.info(f"پوزیشن ترایلینگ برای {symbol} به‌روزرسانی شد.")
    except Exception as e:
        logger.error(f"خطا در به‌روزرسانی پوزیشن ترایلینگ: {e}")
    finally:
        if conn:
            conn.close()

def insert_news(symbol: str, title: str, source: str, published_at: str, content: str, sentiment_score: float):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        query = """
            INSERT INTO news (symbol, title, source, published_at, content, sentiment_score)
            VALUES (:symbol, :title, :source, :published_at, :content, :sentiment_score)
        """
        conn.execute(
            text(query),
            {
                'symbol': symbol,
                'title': title,
                'source': source,
                'published_at': published_at,
                'content': content,
                'sentiment_score': sentiment_score
            }
        )
        conn.commit() 
        logger.info(f"خبر برای {symbol} ذخیره شد: {title[:40]}...")
    except Exception as e:
        logger.error(f"خطا در ذخیره خبر: {e}")
    finally:
        if conn:
            conn.close()

def get_sentiment_from_db(symbol, timestamp):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return 0.0
        query = """
            SELECT AVG(sentiment_score) as avg_sentiment
            FROM news
            WHERE symbol = %s
              AND published_at <= FROM_UNIXTIME(%s)
              AND published_at >= FROM_UNIXTIME(%s - 60*60*24)
        """
        df = pd.read_sql(query, con=conn, params=(symbol.replace("USDT", ""), timestamp, timestamp))
        return df['avg_sentiment'].iloc[0] if not df.empty and df['avg_sentiment'].iloc[0] is not None else 0.0
    except Exception as e:
        logger.error(f"خطا در گرفتن sentiment برای {symbol}: {e}")
        return 0.0
    finally:
        if conn:
            conn.close()

def get_index_value(index_name, timestamp):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return None
        query = """
            SELECT value FROM market_indices
            WHERE index_name = %s AND timestamp <= %s
            ORDER BY timestamp DESC LIMIT 1
        """
        df = pd.read_sql(query, con=conn, params=(index_name, timestamp))
        return df['value'].iloc[0] if not df.empty else None
    except Exception as e:
        logger.error(f"خطا در گرفتن مقدار شاخص برای {index_name}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def insert_backtest_result(symbol: str, total_trades: int, win_rate: float, total_profit: float, final_balance: float):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        query = """
            INSERT INTO backtest_results (symbol, total_trades, win_rate, total_profit, final_balance)
            VALUES (:symbol, :total_trades, :win_rate, :total_profit, :final_balance)
        """
        conn.execute(
            text(query),
            {
                'symbol': symbol,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'final_balance': final_balance
            }
        )
        logger.info(f"نتیجه بک‌تست برای {symbol} ذخیره شد: Win Rate={win_rate:.2%}, Profit={total_profit:.2f}")
    except Exception as e:
        logger.error(f"خطا در ذخیره نتیجه بک‌تست: {e}")
    finally:
        if conn:
            conn.close()

def fetch_candles_from_binance(symbol="BTCUSDT", interval="1h", limit=1000):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        df['timestamp'] = df['timestamp'].astype(int) // 1000
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)  # فقط حجم معاملاتی (تعداد واحدها)
        
        logger.info(f"داده‌های کندل برای {symbol} دریافت شد: {len(df)} ردیف")
        return df
    except Exception as e:
        logger.error(f"خطا در دریافت داده‌های کندل از بایننس برای {symbol}: {e}")
        return pd.DataFrame()
    
def fetch_and_save_candles(symbol="BTCUSDT", interval="1h", limit=1000):
    df = fetch_candles_from_binance(symbol, interval, limit)
    if not df.empty:
        save_candles_to_db(symbol, df)
    else:
        logger.warning(f"هیچ داده‌ای برای {symbol} دریافت نشد.")

def insert_strategy(symbol: str, title: str, source: str, published_at: str, content: str, strategy_score: float, trader_sentiment: float):
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return
        query = """
            INSERT INTO strategies (symbol, title, source, published_at, content, strategy_score, trader_sentiment)
            VALUES (:symbol, :title, :source, :published_at, :content, :strategy_score, :trader_sentiment)
        """
        conn.execute(
            text(query),
            {
                'symbol': symbol,
                'title': title,
                'source': source,
                'published_at': published_at,
                'content': content,
                'strategy_score': strategy_score,
                'trader_sentiment': trader_sentiment
            }
        )
        logger.info(f"استراتژی برای {symbol} ذخیره شد: {title[:40]}...")
    except Exception as e:
        logger.error(f"خطا در ذخیره استراتژی: {e}")
    finally:
        if conn:
            conn.close()

def get_latest_candle_timestamp(symbol: str) -> int:
    conn = None
    try:
        conn = get_connection()
        if not conn:
            logger.error(f"اتصال به دیتابیس برای {symbol} برقرار نشد.")
            return 0
        query = """
            SELECT MAX(timestamp)
            FROM candles
            WHERE symbol = :symbol
        """
        result = conn.execute(text(query), {"symbol": symbol}).fetchone()
        return int(result[0]) if result[0] else 0
    except Exception as e:
        logger.error(f"خطا در دریافت آخرین timestamp برای {symbol}: {e}")
        return 0
    finally:
        if conn:
            conn.close()

def get_candle_data(symbol: str, limit: int = 1000) -> pd.DataFrame:
    try:
        conn = get_connection()
        if not conn:
            return pd.DataFrame()
        query = f"""
            SELECT timestamp, open, high, low, close, volume, symbol
            FROM candles
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        df = pd.read_sql(query, con=conn, params=(symbol, limit))
        return df
    except Exception as e:
        logger.error(f"خطا در دریافت داده‌های کندل برای {symbol}: {e}")
        return pd.DataFrame()

# تابع جدید برای دریافت کندل‌های مرتبط با اخبار
def get_candles_for_news(symbol: str, hours_window: int = 24) -> pd.DataFrame:
    try:
        conn = get_connection()
        if not conn:
            logger.error("اتصال به دیتابیس برقرار نشد.")
            return pd.DataFrame()

        # دریافت تمام اخبار برای نماد موردنظر
        symbol_db = symbol.replace("USDT", "")  # تبدیل BTCUSDT به BTC
        query_news = """
            SELECT published_at
            FROM news
            WHERE symbol = %s
            ORDER BY published_at
        """
        news_df = pd.read_sql(query_news, con=conn, params=(symbol_db,))
        if news_df.empty:
            logger.warning(f"هیچ خبری برای {symbol_db} یافت نشد.")
            return pd.DataFrame()

        # محاسبه بازه‌های زمانی (24 ساعت قبل و بعد از هر خبر)
        time_windows = []
        for _, row in news_df.iterrows():
            news_time = row['published_at']
            news_timestamp = int(news_time.timestamp())
            start_time = news_timestamp - (hours_window * 3600)  # 24 ساعت قبل
            end_time = news_timestamp + (hours_window * 3600)   # 24 ساعت بعد
            time_windows.append((start_time, end_time))

        # ادغام بازه‌های زمانی که همپوشانی دارند
        time_windows.sort()
        merged_windows = []
        current_start, current_end = time_windows[0]
        for start, end in time_windows[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_windows.append((current_start, current_end))
                current_start, current_end = start, end
        merged_windows.append((current_start, current_end))

        # دریافت کندل‌ها برای بازه‌های زمانی ادغام‌شده
        all_candles = []
        for start_ts, end_ts in merged_windows:
            logger.info(f"دریافت کندل‌ها برای {symbol} از {start_ts} تا {end_ts}")
            query_candles = """
                SELECT timestamp, open, high, low, close, volume, symbol
                FROM candles
                WHERE symbol = %s
                AND timestamp BETWEEN %s AND %s
                ORDER BY timestamp
            """
            candles = pd.read_sql(query_candles, con=conn, params=(symbol, start_ts, end_ts))
            if not candles.empty:
                all_candles.append(candles)

        if not all_candles:
            logger.warning(f"هیچ کندلی برای {symbol} در بازه‌های زمانی اخبار یافت نشد.")
            return pd.DataFrame()

        # ادغام تمام کندل‌ها و حذف موارد تکراری
        result_df = pd.concat(all_candles).drop_duplicates(subset=['timestamp']).sort_values(by='timestamp')
        logger.info(f"تعداد کندل‌های مرتبط با اخبار برای {symbol}: {len(result_df)}")
        return result_df

    except Exception as e:
        logger.error(f"خطا در دریافت کندل‌های مرتبط با اخبار برای {symbol}: {e}")
        return pd.DataFrame()

# تابع برای دریافت متن اخبار (برای استفاده در جاهای دیگر)
@lru_cache(maxsize=1000)
def get_recent_news_texts(symbol: str, timestamp: int, hours: int = 720) -> str:
    try:
        conn = get_connection()
        if not conn:
            logger.warning("اتصال به دیتابیس برقرار نشد.")
            return "no_news_available"
        time_window = hours * 3600
        from_ts = timestamp - time_window
        to_ts = timestamp + time_window
        query = """
            SELECT COALESCE(content, '') as text
            FROM news
            WHERE symbol = %s 
            AND published_at BETWEEN FROM_UNIXTIME(%s) AND FROM_UNIXTIME(%s)
            ORDER BY published_at DESC
            LIMIT 20
        """
        df = pd.read_sql(query, conn, params=(symbol, from_ts, to_ts))
        conn.close()
        if not df.empty:
            logger.debug(f"برای {symbol} در بازه {from_ts} تا {to_ts}: {len(df)} خبر یافت شد")
            return "...".join(df['text'].tolist())
        else:
            logger.warning(f"هیچ داده‌ای برای {symbol} در بازه {from_ts} تا {to_ts} یافت نشد.")
            return "no_news_available"
    except Exception as e:
        logger.error(f"خطا در دریافت متن اخبار برای {symbol}: {e}")
        return "no_news_available"
    
def get_db_connection():
    from config import MYSQL_CONFIG  # فرض می‌کنیم config.env لود شده
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

def add_news_to_candles(df, symbol, hours_window=720):
    engine = get_db_connection()
    if not engine:
        logger.error("اتصال به دیتابیس برقرار نشد.")
        return df

    df = df.copy()
    df['news_count'] = 0
    df['avg_content_length'] = 0.0
    df['news_text'] = ""  # اطمینان از مقدار اولیه رشته
    df['news_text'] = df['news_text'].astype(str)  # تبدیل صریح به رشته

    for idx, row in df.iterrows():
        timestamp = int(row['timestamp'])
        news_text = get_recent_news_texts(symbol.replace("USDT", ""), timestamp, hours=hours_window)
        print(f"News text for timestamp {timestamp}: {news_text}")  # دیباگ
        if news_text:
            df.at[idx, 'news_text'] = str(news_text)  # تبدیل صریح به رشته
            news_items = news_text.split('...') if '...' in news_text else [news_text]
            df.at[idx, 'news_count'] = len([item for item in news_items if item])
            df.at[idx, 'avg_content_length'] = np.mean([len(item) for item in news_items if item]) if news_items else 0.0
        else:
            logger.warning(f"هیچ متنی برای {symbol} در timestamp {timestamp} یافت نشد.")
            df.at[idx, 'news_text'] = ""  # رشته خالی در صورت عدم وجود داده

    return df

def save_features_to_db(features, batch_size=1000):
    """ذخیره فیچرها در دیتابیس با پشتیبانی از همه فیچرها"""
    conn = get_connection()
    try:
        if not conn:
            logger.error("اتصال به دیتابیس برقرار نشد.")
            return False
        
        if not isinstance(features, list):
            features = [features] if isinstance(features, dict) else []
        
        if not features:
            logger.warning("هیچ فیچری برای ذخیره ارسال نشده است.")
            return False
        
        # تولید کوئری SQL داینامیک
        query, columns = generate_sql_query(num_embeddings=50)
        
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            with conn.begin():
                for feature in batch:
                    # تبدیل نوع داده‌ها
                    feature_dict = {k: float(v) if isinstance(v, (Decimal, np.floating)) else v for k, v in feature.items()}
                    # مدیریت امبدینگ‌ها
                    for emb_col in [f'news_emb_{i}' for i in range(50)]:
                        if emb_col in feature_dict:
                            if isinstance(feature_dict[emb_col], (list, tuple)):
                                feature_dict[emb_col] = float(feature_dict[emb_col][0]) if feature_dict[emb_col] else 0.0
                            else:
                                feature_dict[emb_col] = float(feature_dict.get(emb_col, 0.0))
                    # پر کردن مقادیر پیش‌فرض برای همه ستون‌ها
                    record = {}
                    for col in columns:
                        if col in ['symbol', 'interval']:
                            record[col] = str(feature_dict.get(col, ''))
                        elif col == 'timestamp':
                            record[col] = int(feature_dict.get(col, 0))
                        else:
                            record[col] = float(feature_dict.get(col, 0.0))
                    conn.execute(text(query), record)
        
        logger.info(f"فیچرها با موفقیت در دیتابیس ذخیره شدند. تعداد: {len(features)}")
        return True
    except Exception as e:
        logger.error(f"خطا در ذخیره فیچرها در دیتابیس: {e}", exc_info=True)
        return False
    finally:
        conn.close() 

def get_batch_indices(timestamps, index_name, batch_size=100):
    conn = get_connection()
    try:
        if not timestamps:
            logger.warning(f"لیست timestamps برای {index_name} خالی است.")
            return {}
        # حذف None و مقادیر تکراری
        timestamps = list(set([ts for ts in timestamps if ts is not None]))
        results = {}
        for i in range(0, len(timestamps), batch_size):
            batch = timestamps[i:i + batch_size]
            placeholders = ', '.join(['%s'] * len(batch))
            query = f"""
                SELECT timestamp, value
                FROM market_indices
                WHERE index_name = %s AND timestamp IN ({placeholders})
            """
            params = [index_name] + batch
            df = pd.read_sql(query, conn, params=params)
            batch_results = df.set_index('timestamp')['value'].to_dict()
            results.update(batch_results)
            logger.debug(f"Batch {i//batch_size + 1} برای {index_name}: {len(batch_results)} مقدار دریافت شد.")
        if not results:
            logger.warning(f"هیچ داده‌ای برای {index_name} در جدول market_indices یافت نشد.")
        return results
    except Exception as e:
        logger.error(f"خطا در get_batch_indices: {e}", exc_info=True)
        return {}
    finally:
        conn.close()


def get_batch_sentiments(symbol, timestamps, batch_size=100):
    conn = get_connection()
    try:
        if not timestamps:
            logger.warning(f"لیست timestamps برای {symbol} خالی است.")
            return {}
        # حذف None و مقادیر تکراری
        timestamps = list(set([ts for ts in timestamps if ts is not None]))
        if not timestamps:
            logger.warning(f"هیچ timestamp معتبری برای {symbol} یافت نشد.")
            return {}
        
        results = {}
        symbol_clean = symbol.replace("USDT", "")
        for i in range(0, len(timestamps), batch_size):
            batch = timestamps[i:i + batch_size]
            # تبدیل timestamps به فرمت TIMESTAMP
            batch_timestamps = [datetime.fromtimestamp(ts) for ts in batch]
            placeholders = ', '.join(['%s'] * len(batch))
            query = f"""
                SELECT published_at, sentiment_score
                FROM news
                WHERE symbol = %s
                AND published_at IN ({placeholders})
            """
            params = [symbol_clean] + batch_timestamps
            df = pd.read_sql(query, conn, params=params)
            # تبدیل published_at به epoch برای سازگاری با timestamps
            if not df.empty:
                df['timestamp'] = df['published_at'].apply(lambda x: int(x.timestamp()))
                batch_results = df.set_index('timestamp')['sentiment_score'].to_dict()
                results.update(batch_results)
            logger.debug(f"Batch {i//batch_size + 1} برای {symbol}: {len(batch_results)} مقدار دریافت شد.")
        
        if not results:
            logger.warning(f"هیچ داده احساسی برای {symbol} در جدول news یافت نشد.")
        return results
    except Exception as e:
        logger.error(f"خطا در get_batch_sentiments برای {symbol}: {e}", exc_info=True)
        return {}
    finally:
        conn.close()