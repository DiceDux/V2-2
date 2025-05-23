import time
import asyncio
import websockets
import threading
from data.data_manager import get_connection
from sqlalchemy import text
import pandas as pd
import logging
from config import TRADE_EXECUTION_MODE
from trading.simulation import simulate_trade
from trading.real_trading import real_trade

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_prices():
    """به‌روزرسانی قیمت‌ها و مدیریت TP/SL"""
    while True:
        try:
            with get_connection() as conn:
                query = "SELECT * FROM positions"
                df = pd.read_sql(query, con=conn)
                positions = df.to_dict('records')
                
                for pos in positions:
                    symbol = pos['symbol']
                    last_tp = float(pos['tp_price'])
                    last_sl = float(pos['sl_price'])
                    step = int(pos['tp_step'])
                    entry = float(pos['entry_price'])
                    
                    # گرفتن آخرین قیمت
                    query = """
                        SELECT close, timestamp FROM candles
                        WHERE symbol = :symbol
                        ORDER BY timestamp DESC LIMIT 1
                    """
                    df = pd.read_sql(text(query), conn, params={'symbol': symbol})
                    if df.empty:
                        logger.warning(f"کندلی برای {symbol} یافت نشد.")
                        continue
                    
                    current_price = float(df['close'].iloc[0])
                    latest_timestamp = df['timestamp'].iloc[0]
                    profit_percent = ((current_price - entry) / entry) * 100
                    
                    logger.info(f"🟡 بررسی {symbol} | قیمت: {current_price} | SL: {last_sl} | TP: {last_tp} | Timestamp: {latest_timestamp}")
                    
                    # به‌روزرسانی قیمت و سود
                    update_query = """
                        UPDATE positions SET last_price = :last_price, live_profit = :live_profit WHERE symbol = :symbol
                    """
                    conn.execute(
                        text(update_query),
                        {'last_price': current_price, 'live_profit': profit_percent, 'symbol': symbol}
                    )
                    
                    # بررسی Trailing TP/SL
                    if current_price >= last_tp:
                        new_tp = round(last_tp * 1.03, 8)  # حداقل 3% افزایش
                        new_sl = last_tp
                        new_step = step + 1
                        
                        update_query = """
                            UPDATE positions
                            SET tp_price = :tp_price, sl_price = :sl_price, tp_step = :tp_step
                            WHERE symbol = :symbol
                        """
                        conn.execute(
                            text(update_query),
                            {'tp_price': new_tp, 'sl_price': new_sl, 'tp_step': new_step, 'symbol': symbol}
                        )
                        
                        logger.info(f"[TP STEP] {symbol} reached TP. New TP: {new_tp}, New SL: {new_sl}")
        
        except Exception as e:
            logger.error(f"خطا در price_updater: {e}")
        
        time.sleep(0.5)  # تأخیر 0.5 ثانیه

async def check_prices_and_notify():
    """چک کردن قیمت‌ها و ارسال دستور فروش برای SL"""
    while True:
        try:
            with get_connection() as conn:
                query = "SELECT * FROM positions"
                df = pd.read_sql(query, con=conn)
                positions = df.to_dict('records')
                
                trade_func = simulate_trade if TRADE_EXECUTION_MODE == "demo" else real_trade
                
                for pos in positions:
                    symbol = pos['symbol']
                    sl = float(pos['sl_price'])
                    
                    # گرفتن آخرین قیمت
                    query = """
                        SELECT close FROM candles
                        WHERE symbol = :symbol
                        ORDER BY timestamp DESC LIMIT 1
                    """
                    df = pd.read_sql(text(query), conn, params={'symbol': symbol})
                    if df.empty:
                        continue
                    
                    current_price = float(df['close'].iloc[0])
                    
                    if current_price <= sl:
                        logger.info(f"🟥 SL فعال شد برای {symbol} | قیمت: {current_price} <= SL: {sl} | زمان: {time.time()}")
                        async with websockets.connect("ws://localhost:5678") as websocket:
                            await websocket.send(f"SELL:{symbol}")
                            logger.info(f"📤 ارسال SELL برای SL {symbol}")
        
        except Exception as e:
            logger.error(f"خطا در check_prices_and_notify: {e}")
        
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    threading.Thread(target=update_prices, daemon=True).start()
    asyncio.run(check_prices_and_notify())