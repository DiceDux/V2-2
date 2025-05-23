from data.data_manager import (
    get_wallet_balance, update_wallet_balance,
    get_position, insert_position, delete_position,
    save_trade_record
)
from ccxt import coinex
import logging
from config import COINEX_API_KEY, COINEX_API_SECRET, TRADE_PERCENT

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# تنظیمات کوینکس
coinex_client = coinex({
    'apiKey': COINEX_API_KEY,
    'secret': COINEX_API_SECRET,
    'enableRateLimit': True
})

def real_trade(symbol: str, action: str, price: float, confidence: float = 0.0) -> dict:
    """اجرای ترید واقعی با API کوینکس"""
    logger.info(f"⚙️ [real_trade] شروع ترید | symbol={symbol}, action={action}, price={price}, confidence={confidence}")
    
    try:
        # نرمال‌سازی نماد برای کوینکس (مثل BTC/USDT)
        symbol_coinex = symbol.replace("USDT", "/USDT")
        
        # دریافت موجودی
        balance = float(get_wallet_balance())
        account = coinex_client.fetch_balance()
        usdt_balance = float(account['free']['USDT'])
        
        if action == "buy":
            trade_amount = balance * TRADE_PERCENT
            if trade_amount > usdt_balance:
                logger.error(f"موجودی کافی نیست. موجودی USDT: {usdt_balance}, مورد نیاز: {trade_amount}")
                return {"balance": balance, "error": "insufficient_balance"}
            
            # محاسبه مقدار (quantity)
            quantity = trade_amount / price
            
            # ارسال سفارش خرید
            order = coinex_client.create_market_buy_order(symbol_coinex, quantity)
            executed_price = float(order['price']) if order['price'] else price
            executed_quantity = float(order['amount'])
            
            # تنظیم TP/SL اولیه
            tp_price = executed_price * 1.10  # 10% بالاتر
            sl_price = executed_price * 0.90  # 10% پایین‌تر
            
            # به‌روزرسانی موجودی و پوزیشن
            update_wallet_balance(balance - trade_amount)
            insert_position(
                symbol, "buy", executed_price, executed_quantity,
                tp_price, sl_price, tp_step=1, last_price=executed_price
            )
            
            logger.info(f"خرید انجام شد | {symbol} | قیمت: {executed_price}, مقدار: {executed_quantity}")
            return {"balance": balance - trade_amount, "executed_price": executed_price, "quantity": executed_quantity}
        
        elif action == "sell":
            position = get_position(symbol)
            if not position:
                logger.warning(f"هیچ پوزیشنی برای {symbol} یافت نشد.")
                return {
                    "balance": balance,
                    "quantity": 0,
                    "exit_price": price,
                    "profit_percent": 0,
                    "error": "no_position"
                }
            
            entry_price = float(position["entry_price"])
            quantity = float(position["quantity"])
            
            # ارسال سفارش فروش
            order = coinex_client.create_market_sell_order(symbol_coinex, quantity)
            executed_price = float(order['price']) if order['price'] else price
            sell_amount = quantity * executed_price
            profit_percent = ((executed_price - entry_price) / entry_price) * 100
            
            # به‌روزرسانی موجودی و حذف پوزیشن
            update_wallet_balance(balance + sell_amount)
            delete_position(symbol)
            
            logger.info(f"فروش انجام شد | {symbol} | قیمت: {executed_price}, سود: {profit_percent:.2f}%")
            return {
                "balance": balance + sell_amount,
                "quantity": quantity,
                "exit_price": executed_price,
                "profit_percent": profit_percent
            }
        
        else:
            logger.error(f"اکشن نامعتبر: {action}")
            return {"balance": balance, "error": "invalid_action"}
    
    except Exception as e:
        logger.error(f"خطا در اجرای ترید واقعی برای {symbol}: {e}")
        return {"balance": balance, "error": str(e)}