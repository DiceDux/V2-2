import pandas as pd
import numpy as np
from data.data_manager import get_candle_data, insert_backtest_result
from train_ai_model_multi_advanced import predict_with_confidence, extract_trade_features
from feature_engineering_full_ultra_v2 import extract_features_full
from ai.fundamental_analyzer import analyze_fundamentals
import joblib
import logging
from datetime import datetime, timedelta
from config import SYMBOLS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backtest_strategy(symbol: str, start_timestamp: int, end_timestamp: int, model_path: str = 'models/catboost_pro_model.pkl'):
    """اجرای بک‌تست برای استراتژی‌های مدل"""
    try:
        model = joblib.load(model_path)
        logger.info(f"مدل از {model_path} لود شد.")

        df = get_candle_data(symbol, limit=1000)
        print("ستون‌های دیتافریم کندل:", df.columns)
        assert 'timestamp' in df.columns, "ستون timestamp وجود ندارد!"
        df = df[(df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)]
        if df.empty:
            logger.error(f"هیچ داده‌ای برای {symbol} در بازه زمانی مشخص یافت نشد.")
            return None

        df['symbol'] = symbol
        df_features = extract_features_full(df, symbol=symbol)
        df_features = analyze_fundamentals(df_features)

        trade_features = [extract_trade_features(symbol, row['timestamp']) for _, row in df_features.iterrows()]
        trade_df = pd.DataFrame(trade_features, index=df_features.index)
        df_features = pd.concat([df_features, trade_df], axis=1)

        with open('models/catboost_pro_features.txt', 'r') as f:
            model_features = [line.strip() for line in f.readlines()]
        missing_features = [col for col in model_features if col not in df_features.columns]
        if missing_features:
            logger.warning(f"ویژگی‌های گمشده برای {symbol}: {missing_features}")
            for col in missing_features:
                df_features[col] = 0.0  # پر کردن با صفر
        X = df_features[model_features]
        X.fillna(X.mean(), inplace=True)

        predictions, _, probabilities = predict_with_confidence(model, X)
        df_features['prediction'] = predictions
        df_features['confidence'] = np.max(probabilities, axis=1)

        balance = 10000
        position = None
        trades = []
        for i, row in df_features.iterrows():
            confidence = row['confidence']

            if row['prediction'] == 2 and not position and confidence > 0.75:
                position = {
                    'entry_price': df['close'].iloc[i],
                    'timestamp': row['timestamp'],
                    'confidence': confidence
                }
            elif row['prediction'] == 0 and position and confidence > 0.75:
                exit_price = df['close'].iloc[i]
                profit = (exit_price - position['entry_price']) / position['entry_price'] * balance
                balance += profit
                trades.append({
                    'symbol': symbol,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'entry_time': position['timestamp'],
                    'exit_time': row['timestamp'],
                    'confidence': position['confidence']
                })
                position = None
            elif row['prediction'] == 1:
                continue

        total_trades = len(trades)
        win_rate = len([t for t in trades if t['profit'] > 0]) / total_trades if total_trades > 0 else 0
        total_profit = sum(t['profit'] for t in trades)
        logger.info(f"بک‌تست برای {symbol}:")
        logger.info(f"تعداد تریدها: {total_trades}")
        logger.info(f"نرخ برد: {win_rate:.2%}")
        logger.info(f"سود کل: {total_profit:.2f}")

        # ذخیره نتایج بک‌تست
        insert_backtest_result(symbol, total_trades, win_rate, total_profit, balance)

        return {
            'trades': trades,
            'total_profit': total_profit,
            'win_rate': win_rate,
            'final_balance': balance
        }

    except Exception as e:
        logger.error(f"خطا در بک‌تست برای {symbol}: {e}")
        return None

if __name__ == "__main__":
    start_time = int((datetime.utcnow() - timedelta(days=30)).timestamp())
    end_time = int(datetime.utcnow().timestamp())
    for symbol in SYMBOLS:
        result = backtest_strategy(symbol, start_time, end_time)
        if result:
            logger.info(f"نتایج بک‌تست برای {symbol}: {result}")