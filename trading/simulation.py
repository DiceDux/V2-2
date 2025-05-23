from data.data_manager import (
    get_wallet_balance, update_wallet_balance,
    get_position, insert_position, delete_position,
    save_trade_record
)
import logging

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRADE_PERCENT = 0.2

def simulate_trade(symbol: str, action: str, price: float, confidence: float = 0.0) -> dict:
    """شبیه‌سازی ترید برای حالت دمو"""
    logger.info(f"⚙️ [simulate] شروع ترید | symbol={symbol}, action={action}, price={price}, confidence={confidence}")
    
    balance = float(get_wallet_balance())
    
    if action == "buy":
        trade_amount = balance * TRADE_PERCENT
        if trade_amount > balance:
            logger.error("موجودی کافی برای خرید وجود ندارد.")
            return {"balance": balance, "error": "insufficient_balance"}
        
        quantity = trade_amount / price
        tp_price = price * 1.10
        sl_price = price * 0.90
        update_wallet_balance(balance - trade_amount)
        insert_position(symbol, "buy", price, quantity, tp_price, sl_price, tp_step=1, last_price=price)
        logger.info(f"خرید شبیه‌سازی شد | {symbol} | مقدار: {quantity}")
        return {"balance": balance - trade_amount, "quantity": quantity}
    
    elif action == "sell":
        position = get_position(symbol)
        if not position:
            logger.warning("هیچ پوزیشنی برای فروش پیدا نشد.")
            return {
                "balance": balance,
                "quantity": 0,
                "exit_price": price,
                "profit_percent": 0,
                "error": "no_position"
            }
        
        entry_price = float(position["entry_price"])
        quantity = float(position["quantity"])
        sell_amount = quantity * price
        profit_percent = ((price - entry_price) / entry_price) * 100
        
        update_wallet_balance(balance + sell_amount)
        delete_position(symbol)
        
        logger.info(f"فروش شبیه‌سازی شد | {symbol} | سود: {profit_percent:.2f}%")
        return {
            "balance": balance + sell_amount,
            "quantity": quantity,
            "exit_price": price,
            "profit_percent": profit_percent
        }
    
    else:
        logger.error(f"اکشن نامعتبر: {action}")
        return {"balance": balance, "error": "invalid_action"}