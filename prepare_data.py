import pandas as pd
import numpy as np
import logging
import traceback

from config import SYMBOLS
from data.data_manager import get_candle_data
from feature_engineering_full_ultra_v2 import extract_features_full

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

raw_data = {}
features = {}

# پردازش هر نماد بدون وابستگی به df_btc, df_eth, df_doge
for symbol in SYMBOLS:
    df = get_candle_data(symbol)
    df["symbol"] = symbol
    raw_data[symbol] = df
    try:
        logger.info(f"⚙️ استخراج ویژگی‌ها برای {symbol}")
        result = extract_features_full(df)
        if result.empty:
            logger.warning(f"⚠️ ویژگی‌های خالی برای {symbol}")
            continue
        features[symbol] = result
        logger.info(f"✅ ویژگی‌ها آماده {symbol} → {result.shape}")
    except Exception as e:
        logger.error(f"❌ خطا در ویژگی‌گیری {symbol}: {e}")
        traceback.print_exc()

# تست: چاپ ستون احساس برای یکی از ارزها
sample_symbol = SYMBOLS[0]
if sample_symbol in features:
    print(f"🔍 نمونه news_sentiment برای {sample_symbol}")
    print(features[sample_symbol][['timestamp', 'news_sentiment']].tail(5))
else:
    print("❌ هیچ ویژگی برای symbol اول یافت نشد.")
