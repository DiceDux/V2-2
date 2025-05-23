from data.data_manager import get_connection
from sqlalchemy import text
import logging
import threading

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# قفل برای thread-safety
status_lock = threading.Lock()

def get_trade_status() -> str:
    """دریافت وضعیت ربات از دیتابیس"""
    try:
        with get_connection() as conn:
            query = "SELECT status FROM trade_status ORDER BY updated_at DESC LIMIT 1"
            result = conn.execute(text(query)).fetchone()
            status = result[0] if result else "running"
        logger.info(f"وضعیت ربات دریافت شد: {status}")
        return status
    except Exception as e:
        logger.error(f"خطا در دریافت وضعیت ربات: {e}")
        return "running"

def set_trade_status(status: str):
    """تنظیم وضعیت ربات در دیتابیس"""
    try:
        with status_lock:
            with get_connection() as conn:
                query = """
                    INSERT INTO trade_status (status, updated_at)
                    VALUES (:status, NOW())
                """
                conn.execute(text(query), {'status': status})
        logger.info(f"وضعیت ربات تنظیم شد: {status}")
    except Exception as e:
        logger.error(f"خطا در تنظیم وضعیت ربات: {e}")