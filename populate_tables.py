import pandas as pd
import numpy as np
import mysql.connector
import requests
from config import MYSQL_CONFIG
from datetime import datetime, timedelta
import logging

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)

def fetch_binance_symbols():
    """دریافت لیست نمادهای معتبر از بایننس"""
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # فقط نمادهایی که با USDT هستند و فعال‌اند
        symbols = [s['symbol'] for s in data['symbols'] if s['symbol'].endswith("USDT") and s['status'] == "TRADING"]
        return symbols
    except Exception as e:
        logger.error(f"خطا در دریافت لیست نمادها از بایننس: {e}")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "THETAUSDT"]  # پیش‌فرض

def fetch_candles(symbol, interval="4h", limit=1000):
    """دریافت داده‌های کندل از بایننس"""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url)
        response.raise_for_status()
        candles = response.json()
        return candles
    except Exception as e:
        logger.error(f"خطا در دریافت داده‌های {symbol}: {e}")
        return []

def populate_trades():
    """پر کردن جدول trades با داده‌های شبیه‌سازی‌شده اما واقعی‌تر"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # دریافت نمادهای معتبر
    SYMBOLS = fetch_binance_symbols()
    
    for symbol in SYMBOLS:
        logger.info(f"پر کردن trades برای {symbol}...")
        candles = fetch_candles(symbol)
        if not candles:
            logger.warning(f"داده‌ای برای {symbol} یافت نشد.")
            continue
        
        trades = []
        balance = 10000  # موجودی اولیه
        position_open = False
        entry_price = None
        entry_timestamp = None
        trade_counter = 0
        
        for i, candle in enumerate(candles):
            timestamp = int(candle[0] / 1000)  # تبدیل به ثانیه
            open_price = float(candle[1])
            close_price = float(candle[4])
            
            # تصمیم ترید بر اساس یک استراتژی ساده (مثلاً RSI فرضی)
            rsi = np.random.uniform(0, 100)  # فرضی، می‌تونید از داده‌های واقعی استفاده کنید
            confidence = np.random.uniform(0.6, 0.95)
            
            # تنظیم تایم‌استمپ برای جلوگیری از تکرار
            adjusted_timestamp = timestamp + trade_counter
            adjusted_candle_time = timestamp + trade_counter
            
            if rsi > 70 and position_open:  # Sell Signal
                action = "sell"
                exit_price = close_price
                profit_pct = (exit_price - entry_price) / entry_price * 100  # سود به درصد
                trades.append((
                    symbol, str(entry_timestamp), action, entry_price, exit_price, profit_pct, confidence, "live", str(adjusted_candle_time)
                ))
                position_open = False
                trade_counter += 1
            elif rsi < 30 and not position_open:  # Buy Signal
                action = "buy"
                entry_price = close_price
                entry_timestamp = adjusted_timestamp
                position_open = True
                trades.append((
                    symbol, str(adjusted_timestamp), action, entry_price, None, 0.0, confidence, "live", str(adjusted_candle_time)
                ))
                trade_counter += 1
            elif position_open and i == len(candles) - 1:  # بستن پوزیشن باز در انتها
                action = "sell"
                exit_price = close_price
                profit_pct = (exit_price - entry_price) / entry_price * 100
                trades.append((
                    symbol, str(entry_timestamp), action, entry_price, exit_price, profit_pct, confidence, "live", str(adjusted_candle_time)
                ))
                trade_counter += 1
        
        # ذخیره در جدول trades
        if trades:
            for trade in trades:
                try:
                    cursor.execute("""
                        INSERT INTO trades (symbol, timestamp, action, entry_price, exit_price, profit, confidence, mode, candle_time)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, trade)
                    conn.commit()
                except mysql.connector.errors.IntegrityError as e:
                    logger.warning(f"ترید تکراری رد شد: {trade[0]}-{trade[1]}")
                    continue
            logger.info(f"{len(trades)} ترید برای {symbol} ذخیره شد.")
    
    conn.close()

def populate_backtest_results():
    """اجرای بک‌تست و پر کردن جدول backtest_results"""
    conn = get_connection()
    cursor = conn.cursor()
    
    SYMBOLS = fetch_binance_symbols()
    
    for symbol in SYMBOLS:
        logger.info(f"اجرای بک‌تست برای {symbol}...")
        cursor.execute("""
            SELECT action, profit, confidence, timestamp
            FROM trades
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT 1000
        """, (symbol,))
        trades = cursor.fetchall()
        
        if not trades:
            logger.info(f"هیچ تریدی برای {symbol} یافت نشد.")
            continue
        
        total_trades = len(trades)
        win_rate = len([t for t in trades if t[1] > 0]) / total_trades if total_trades > 0 else 0
        total_profit = sum(t[1] for t in trades if t[1] is not None)
        final_balance = 10000 + sum(t[1] for t in trades if t[1] is not None)  # فرضاً سود به درصد به بالانس اضافه می‌شه
        
        cursor.execute("""
            INSERT INTO backtest_results (symbol, total_trades, win_rate, total_profit, final_balance, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (symbol, total_trades, win_rate, total_profit, final_balance, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        logger.info(f"نتایج بک‌تست برای {symbol} ذخیره شد.")
    
    conn.close()

if __name__ == "__main__":
    populate_trades()
    populate_backtest_results()