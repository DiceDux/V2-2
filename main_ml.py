import time
import asyncio
import websockets
import threading
import socket
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from ai.ai_model_runner import predict_signal_from_model
from trading.simulation import simulate_trade
from trading.real_trading import real_trade
from data.data_manager import (
    get_candle_data, save_features, get_position, insert_position, delete_position,
    has_trade_in_current_candle, insert_trade_with_candle, insert_balance_to_db,
    update_position_trailing
)
from config import SYMBOLS, TRADE_MODE, TIMEFRAME_MINUTES, TRADE_EXECUTION_MODE
from trading.trade_status import get_trade_status
import logging
import numpy as np

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def websocket_handler(websocket):
    """مدیریت پیام‌های WebSocket برای فروش فوری"""
    try:
        async for message in websocket:
            logger.info(f"پیام WebSocket دریافت شد: {message}")
            if message.startswith("SELL:"):
                symbol = message.split(":")[1]
                logger.info(f"دستور فروش فوری دریافت شد برای {symbol}")
                position = get_position(symbol)
                if position:
                    entry_price = float(position["entry_price"])
                    price = float(position["last_price"])
                    profit = price - entry_price
                    profit_percent = (profit / entry_price) * 100
                    trade_func = simulate_trade if TRADE_EXECUTION_MODE == "demo" else real_trade
                    result = trade_func(symbol, "sell", price, confidence=0.0)
                    if "error" not in result:
                        delete_position(symbol)
                        candle_time = int((int(time.time()) // (TIMEFRAME_MINUTES * 60)) * (TIMEFRAME_MINUTES * 60))
                        insert_trade_with_candle(
                            symbol, "sell", price, 0.0, candle_time, TRADE_MODE,
                            profit_percent, exit_price=result.get("exit_price"), quantity=result.get("quantity")
                        )
                        insert_balance_to_db("WALLET", result["balance"])
                        logger.info(f"فروش فوری انجام شد برای {symbol} | سود: {profit_percent:.2f}% | موجودی: {result['balance']}")
                    else:
                        logger.error(f"خطا در فروش فوری برای {symbol}: {result['error']}")
                else:
                    logger.warning(f"پوزیشنی برای {symbol} وجود ندارد. فروش انجام نشد.")
    except Exception as e:
        logger.error(f"خطا در WebSocket: {e}")

def is_port_in_use(port):
    """چک کردن اشغال بودن پورت"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

async def start_ws_server_async():
    """راه‌اندازی سرور WebSocket"""
    async with websockets.serve(websocket_handler, "0.0.0.0", 5678):
        logger.info("WebSocket server فعال شد روی پورت 5678")
        await asyncio.Future()

def start_ws_server():
    asyncio.run(start_ws_server_async())

# راه‌اندازی سرور WebSocket
if not is_port_in_use(5678):
    ws_thread = threading.Thread(target=start_ws_server, daemon=True)
    ws_thread.start()
    logger.info("WebSocket server فعال شد روی پورت 5678")
else:
    logger.warning("WebSocket server قبلاً فعال شده. از راه‌اندازی مجدد جلوگیری شد.")

def check_model_status():
    """چک کردن وضعیت مدل"""
    try:
        from retrain_model import check_last_training
        if check_last_training():
            logger.warning("مدل نیاز به بازآموزش دارد. توسط retrain_model.py انجام می‌شود.")
        else:
            logger.info("مدل به‌روز است.")
    except Exception as e:
        logger.error(f"خطا در چک کردن وضعیت مدل: {e}")

def calculate_dynamic_tp_sl(entry_price: float, volatility: float, confidence: float) -> tuple:
    """محاسبه TP/SL پویا بر اساس نوسانات و اعتماد مدل"""
    base_tp = 0.10  # 10% پایه
    base_sl = 0.10
    volatility_factor = min(max(volatility, 0.5), 2.0)  # محدود کردن تأثیر نوسانات
    confidence_factor = min(max(confidence, 0.5), 1.0)
    
    tp_percent = base_tp * volatility_factor * confidence_factor
    sl_percent = base_sl * volatility_factor / confidence_factor
    
    tp_price = entry_price * (1 + tp_percent)
    sl_price = entry_price * (1 - sl_percent)
    
    return round(tp_price, 8), round(sl_price, 8)

def process_symbol(symbol: str) -> dict:
    """پردازش یک نماد به صورت مستقل"""
    try:
        logger.info(f"تحلیل ML برای {symbol}")
        df = get_candle_data(symbol)
        
        if df.empty or len(df) < 100:
            logger.warning(f"داده کافی برای {symbol} وجود ندارد.")
            return {"symbol": symbol, "status": "no_data"}
        
        # محاسبه نوسانات (ATR)
        atr = ((df['high'] - df['low']) / df['close']).mean()
        
        signal = predict_signal_from_model(df, symbol=symbol, interval=f"{TIMEFRAME_MINUTES}min", verbose=True)
        signal["action"] = str(signal["action"]).lower().strip("[]' ")
        price = df["close"].iloc[-1]
        logger.info(f"سیگنال مدل: {signal['action'].upper()} | قیمت: {price} | اعتماد: {signal['confidence']} | اخبار: {signal['features'].get('news_score', 0)}")
        
        if signal["confidence"] < 0.70:
            logger.warning("اعتماد کافی وجود ندارد. ترید انجام نمی‌شود.")
            return {"symbol": symbol, "status": "low_confidence"}
        
        position = get_position(symbol)
        save_features(symbol, signal['features'])
        
        candle_time = int((int(time.time()) // (TIMEFRAME_MINUTES * 60)) * (TIMEFRAME_MINUTES * 60))
        
        if has_trade_in_current_candle(symbol, candle_time):
            logger.warning(f"تریدی برای {symbol} در همین کندل ثبت شده. عبور می‌کنیم.")
            return {"symbol": symbol, "status": "trade_exists"}
        
        trade_func = simulate_trade if TRADE_EXECUTION_MODE == "demo" else real_trade
        
        if signal["action"] == "buy":
            if position:
                logger.warning(f"پوزیشن باز برای {symbol} وجود دارد. خرید مجدد مجاز نیست.")
                return {"symbol": symbol, "status": "position_exists"}
            else:
                result = trade_func(symbol, "buy", price, signal["confidence"])
                if "error" not in result:
                    # محاسبه TP/SL پویا
                    tp_price, sl_price = calculate_dynamic_tp_sl(price, atr, signal["confidence"])
                    insert_position(
                        symbol, "buy", price, result.get("quantity", 0),
                        tp_price, sl_price, tp_step=1, last_price=price
                    )
                    insert_trade_with_candle(symbol, "buy", price, signal["confidence"], candle_time, TRADE_MODE)
                    insert_balance_to_db("WALLET", result["balance"])
                    logger.info(f"خرید انجام شد | {symbol} | موجودی: {result['balance']}")
                return {"symbol": symbol, "status": "buy_executed", "result": result}
        
        if position:
            entry_price = float(position["entry_price"])
            quantity = float(position["quantity"])
            tp_price = float(position["tp_price"])
            sl_price = float(position["sl_price"])
            tp_step = int(position["tp_step"])
            last_price = float(position["last_price"])
            profit = price - entry_price
            profit_percent = (profit / entry_price) * 100
            
            if signal["action"] == "sell" and signal["confidence"] >= 0.70:
                result = trade_func(symbol, "sell", price, signal["confidence"])
                if "error" not in result:
                    delete_position(symbol)
                    insert_trade_with_candle(
                        symbol, "sell", price, signal["confidence"], candle_time, TRADE_MODE,
                        profit_percent, exit_price=result.get("exit_price"), quantity=result.get("quantity")
                    )
                    insert_balance_to_db("WALLET", result["balance"])
                    logger.info(f"فروش انجام شد با سیگنال مدل | سود: {profit_percent:.2f}% | موجودی: {result['balance']}")
                return {"symbol": symbol, "status": "sell_executed", "result": result}
            
            # مدیریت TP/SL پله‌ای
            if price >= tp_price:
                new_tp_step = tp_step + 1
                new_tp_price, new_sl_price = calculate_dynamic_tp_sl(entry_price, atr, signal["confidence"])
                new_tp_price = max(new_tp_price, price * 1.03)  # حداقل 3% افزایش
                update_position_trailing(symbol, new_tp_price, new_sl_price, new_tp_step, price)
                logger.info(f"TP رسید | مرحله بعدی TP: {new_tp_price:.2f}, SL جدید: {new_sl_price:.2f}")
                return {"symbol": symbol, "status": "tp_updated"}
            
            elif price <= sl_price:
                result = trade_func(symbol, "sell", price, confidence=0.0)
                if "error" not in result:
                    delete_position(symbol)
                    insert_trade_with_candle(
                        symbol, "sell", price, 0.0, candle_time, TRADE_MODE,
                        profit_percent, exit_price=result.get("exit_price"), quantity=result.get("quantity")
                    )
                    insert_balance_to_db("WALLET", result["balance"])
                    logger.info(f"SL فعال شد | فروش با قیمت: {price:.2f} | موجودی: {result['balance']:.2f}")
                return {"symbol": symbol, "status": "sl_triggered", "result": result}
            
            else:
                logger.info(f"فروش انجام نشد. سود فعلی: {profit_percent:.2f}% | اعتماد مدل: {signal['confidence']}")
                return {"symbol": symbol, "status": "no_action"}
        
        return {"symbol": symbol, "status": "no_action"}
    
    except Exception as e:
        logger.error(f"خطا در پردازش {symbol}: {e}")
        return {"symbol": symbol, "status": "error", "error": str(e)}

def run_with_ml():
    """اجرای ربات با مدل ML"""
    logger.info(f"SYMBOLS: {SYMBOLS}")
    if not SYMBOLS:
        logger.error("خطا: SYMBOLS خالی است!")
        return
    
    check_model_status()
    
    if get_trade_status() == "paused":
        logger.info("ربات در حالت توقف قرار دارد.")
        return
    
    logger.info(f"DiceDux ML در حال اجرا با مدل یادگیرنده | حالت: {TRADE_EXECUTION_MODE}")
    
    with ThreadPoolExecutor(max_workers=min(len(SYMBOLS), 10)) as executor:
        future_to_symbol = {executor.submit(process_symbol, symbol): symbol for symbol in SYMBOLS}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                logger.info(f"نتیجه برای {symbol}: {result}")
            except Exception as e:
                logger.error(f"خطا در پردازش {symbol}: {e}")

if __name__ == "__main__":
    while True:
        try:
            run_with_ml()
        except Exception as e:
            logger.error(f"خطای غیرمنتظره در اجرای run_with_ml: {e}")
        logger.info("در حال صبر برای تحلیل بعدی...")
        time.sleep(60 * 5)