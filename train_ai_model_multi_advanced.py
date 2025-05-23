import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import joblib
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.stats.mstats import winsorize
from datetime import datetime, timedelta
from data.data_manager import get_candle_data, save_features_to_db, get_connection, get_batch_indices, get_batch_sentiments
from feature_engineering_full_ultra_v2 import extract_features_full
from config import SYMBOLS, MYSQL_CONFIG
from concurrent.futures import ThreadPoolExecutor
import mysql.connector

# تنظیمات اولیه
os.makedirs('notebooks', exist_ok=True)
os.makedirs('models', exist_ok=True)

# تنظیم لاگینگ
log_format = '%(asctime)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)
file_handler = logging.FileHandler('notebooks/training_log.txt', encoding='utf-8')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# تنظیمات ثابت
TIMEFRAME = '4h'
WINDOW_SIZE = 100
LABEL_LOOKAHEAD = 12
RFE_UPDATE_DAYS = 30
MIN_ROWS = 100

# فقط بخش اصلاح‌شده تابع label_data نشان داده شده است
def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """لیبل‌گذاری داده‌ها بر اساس بازده آتی"""
    try:
        logger.info("شروع لیبل‌گذاری داده‌ها...")
        df['future_return'] = df['close'].shift(-LABEL_LOOKAHEAD) / df['close'] - 1
        df['future_return'].fillna(0.0, inplace=True)  # پر کردن NaN با 0.0
        df['label'] = 'Hold'
        df.loc[df['future_return'] > 0.002, 'label'] = 'Buy'
        df.loc[df['future_return'] < -0.002, 'label'] = 'Sell'
        logger.info("لیبل‌گذاری داده‌ها انجام شد.")
        return df
    except Exception as e:
        logger.error(f"خطا در لیبل‌گذاری: {e}")
        return df.assign(future_return=0.0, label='Hold')  # پیش‌فرض در صورت خطا

def replace_outliers(df: pd.DataFrame, columns: list, threshold: float = 3) -> pd.DataFrame:
    """مدیریت مقادیر پرت با روش Winsorizing و Z-score"""
    try:
        logger.info("جایگزینی مقادیر پرت...")
        numeric_columns = [col for col in columns if df[col].dtype in [np.float64, np.int64]]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = winsorize(df[col], limits=[0.05, 0.05])
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_count = (z_scores > threshold).sum()
                if outliers_count > 0:
                    logger.info(f"تعداد مقادیر پرت در {col}: {outliers_count}")
        logger.info("مقادیر پرت مدیریت شدند.")
        return df
    except Exception as e:
        logger.error(f"خطا در مدیریت پرت‌ها: {e}")
        return df.fillna(0.0)  # پر کردن با صفر در صورت خطا

def ensure_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """اطمینان از نوع داده‌های مناسب"""
    try:
        logger.info("اطمینان از نوع داده‌های مناسب...")
        for col in df.columns:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(float)
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except:
                    logger.warning(f"ستون {col} به عدد تبدیل نشد. مقدار پیش‌فرض 0.0 استفاده می‌شود.")
                    df[col] = 0.0
            else:
                df[col] = df[col].astype(float)
        logger.info("نوع داده‌ها اصلاح شد.")
        return df
    except Exception as e:
        logger.error(f"خطا در اطمینان از نوع داده‌ها: {e}")
        return df.fillna(0.0).astype(float)

def plot_confusion_matrix(y_true, y_pred, model_name):
    """رسم و ذخیره ماتریس درهم‌ریختگی"""
    try:
        logger.info(f"رسم ماتریس درهم‌ریختگی برای {model_name}...")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'notebooks/confusion_matrix_{model_name}.png')
        plt.close()
        logger.info(f"ماتریس درهم‌ریختگی ذخیره شد: notebooks/confusion_matrix_{model_name}.png")
    except Exception as e:
        logger.error(f"خطا در رسم ماتریس درهم‌ریختگی: {e}")

def custom_rfe_logging(estimator, X, y, n_features_to_select=50):
    """اجرای RFE با لاگینگ و ذخیره ویژگی‌های انتخاب‌شده"""
    try:
        logger.info("شروع فرآیند RFE...")
        rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
        rfe.fit(X, y)
        
        feature_ranking = pd.DataFrame({
            'Feature': X.columns,
            'Ranking': rfe.ranking_
        })
        feature_ranking = feature_ranking.sort_values(by='Ranking')
        logger.info("رتبه‌بندی ویژگی‌ها:")
        logger.info(feature_ranking.to_string(index=False))
        
        selected_features = X.columns[rfe.support_].tolist()
        eliminated_features = X.columns[~rfe.support_].tolist()
        logger.info(f"ویژگی‌های انتخاب‌شده ({len(selected_features)}): {selected_features}")
        logger.info(f"ویژگی‌های حذف‌شده ({len(eliminated_features)}): {eliminated_features}")
        
        with open('models/selected_technical_features.txt', 'w') as f:
            for feature in selected_features:
                f.write(feature + '\n')
        logger.info("ویژگی‌های تکنیکال انتخاب‌شده در models/selected_technical_features.txt ذخیره شدند.")
        
        return rfe, selected_features
    except Exception as e:
        logger.error(f"خطا در اجرای RFE: {e}")
        return None, []

def predict_with_confidence(model, X):
    """پیش‌بینی با درصد اعتماد"""
    try:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        confidence_scores = np.max(probabilities, axis=1)
        label_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
        results = []
        for pred, conf in zip(predictions, confidence_scores):
            label = label_map[pred]
            results.append(f"{label} with {conf*100:.2f}% confidence")
        return results, predictions, probabilities
    except Exception as e:
        logger.error(f"خطا در پیش‌بینی با اعتماد: {e}")
        return [f"Hold with 0.00% confidence"] * len(X), np.zeros(len(X)), np.zeros((len(X), 3))

def extract_trade_features(symbol: str, timestamp: int) -> dict:
    """استخراج فیچرهای ترید از جدول trades"""
    try:
        with mysql.connector.connect(**MYSQL_CONFIG) as conn:
            cursor = conn.cursor(dictionary=True)
            time_window = 24 * 3600
            from_ts = timestamp - time_window
            cursor.execute("""
                SELECT action, profit, confidence, timestamp
                FROM trades
                WHERE symbol = %s AND timestamp >= FROM_UNIXTIME(%s)
            """, (symbol, from_ts))
            trades = cursor.fetchall()

            trade_success_rate = 0.0
            avg_profit = 0.0
            avg_confidence = 0.0
            trade_count = 0.0

            if trades:
                successful_trades = len([t for t in trades if t['profit'] and t['profit'] > 0])
                trade_count = len(trades)
                trade_success_rate = successful_trades / trade_count if trade_count > 0 else 0.0
                avg_profit = np.mean([t['profit'] for t in trades if t['profit'] is not None]) if trade_count > 0 else 0.0
                avg_confidence = np.mean([t['confidence'] for t in trades if t['confidence'] is not None]) if trade_count > 0 else 0.0

            return {
                'trade_success_rate': float(trade_success_rate),
                'avg_profit': float(avg_profit),
                'avg_confidence': float(avg_confidence),
                'trade_count': float(trade_count)
            }
    except Exception as e:
        logger.error(f"خطا در استخراج فیچرهای ترید برای {symbol}: {e}")
        return {
            'trade_success_rate': 0.0,
            'avg_profit': 0.0,
            'avg_confidence': 0.0,
            'trade_count': 0.0
        }

def validate_features(X, y, features_to_test, symbol, start_timestamp, end_timestamp):
    """اعتبارسنجی فیچرهای جدید با بک‌تست"""
    try:
        logger.info(f"اعتبارسنجی فیچرها: {features_to_test}")
        model = CatBoostClassifier(verbose=0, thread_count=-1)
        X_train, X_test, y_train, y_test = train_test_split(X[features_to_test], y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        
        df = get_candle_data(symbol=symbol, limit=1000)
        print("ستون‌های دیتافریم کندل:", df.columns)
        assert 'timestamp' in df.columns, "ستون timestamp وجود ندارد!"
        df = df[(df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)]
        if df.empty:
            logger.warning(f"هیچ داده‌ای برای {symbol} در بازه زمانی مشخص یافت نشد.")
            return False
        
        df['symbol'] = symbol
        df_features = extract_features_full(df, symbol=symbol)
        for feat in ['spx_index', 'dxy_index', 'btc_dominance', 'usdt_dominance'] + [f'news_emb_{i}' for i in range(50)]:
            if feat not in df_features.columns:
                df_features[feat] = 0.0
                logger.warning(f"فیچر {feat} غایب بود، با 0.0 پر شد.")
            elif df_features[feat].isna().any():
                df_features[feat].fillna(0.0, inplace=True)
                logger.warning(f"مقادیر NaN در {feat} با 0.0 پر شد.")

        X_backtest = df_features[features_to_test].fillna(0.0)
        predictions, _, probabilities = predict_with_confidence(model, X_backtest)
        df_features['prediction'] = predictions
        df_features['confidence'] = np.max(probabilities, axis=1)
        
        balance = 10000
        trades = []
        entry_price = None
        for i, row in df_features.iterrows():
            if row['prediction'] == 2 and row['confidence'] > 0.7:
                entry_price = df['close'].iloc[i]
            elif row['prediction'] == 0 and row['confidence'] > 0.7 and entry_price is not None:
                exit_price = df['close'].iloc[i]
                profit = (exit_price - entry_price) / entry_price * balance
                balance += profit
                trades.append({'profit': profit})
                entry_price = None
        
        total_trades = len(trades)
        win_rate = len([t for t in trades if t['profit'] > 0]) / total_trades if total_trades > 0 else 0
        total_profit = sum(t['profit'] for t in trades)
        
        logger.info(f"نتایج اعتبارسنجی فیچرها: Win Rate={win_rate:.2%}, Total Profit={total_profit:.2f}")
        
        if win_rate > 0.4 and total_profit > 0:
            logger.info("فیچرها مورد قبول هستند.")
            return True
        else:
            logger.warning("فیچرها عملکرد مناسبی نداشتند.")
            return False
    except Exception as e:
        logger.error(f"خطا در اعتبارسنجی فیچرها: {e}")
        return False

def check_feature_compatibility(X, required_features):
    """چک کردن سازگاری فیچرها با مدل"""
    try:
        logger.info("چک کردن سازگاری فیچرها...")
        missing_features = [f for f in required_features if f not in X.columns]
        extra_features = [f for f in X.columns if f not in required_features]
        
        if missing_features:
            logger.warning(f"فیچرهای غایب: {missing_features}")
            for f in missing_features:
                X[f] = 0.0
        if extra_features:
            logger.info(f"فیچرهای اضافی: {extra_features}")
            X = X.drop(columns=extra_features, errors='ignore')
        
        logger.info("سازگاری فیچرها برقرار شد.")
        return X
    except Exception as e:
        logger.error(f"خطا در چک کردن سازگاری فیچرها: {e}")
        return X.fillna(0.0)

def should_update_rfe():
    """چک کردن اینکه آیا نیاز به آپدیت RFE هست"""
    try:
        with open('models/last_rfe_update.txt', 'r') as f:
            last_update = datetime.strptime(f.read().strip(), '%Y-%m-%d')
        days_since_update = (datetime.now() - last_update).days
        return days_since_update >= RFE_UPDATE_DAYS
    except (FileNotFoundError, ValueError):
        return True

def save_rfe_update_date():
    """ذخیره تاریخ آخرین آپدیت RFE"""
    try:
        with open('models/last_rfe_update.txt', 'w') as f:
            f.write(datetime.now().strftime('%Y-%m-%d'))
    except Exception as e:
        logger.error(f"خطا در ذخیره تاریخ آپدیت RFE: {e}")

def load_selected_features():
    """لود فیچرهای منتخب از فایل"""
    try:
        with open('models/selected_technical_features.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        logger.warning("فایل ویژگی‌های انتخاب‌شده یافت نشد.")
        return None

def process_symbol(symbol):
    """پردازش داده‌ها برای یک نماد"""
    conn = get_connection()
    try:
        df = get_candle_data(symbol=symbol)
        print("ستون‌های دیتافریم کندل:", df.columns)
        assert 'timestamp' in df.columns, "ستون timestamp وجود ندارد!"
        if df.empty or len(df) < MIN_ROWS:
            logger.warning(f"داده کافی برای {symbol} دریافت نشد (تعداد ردیف‌ها: {len(df)}).")
            return None
        logger.info(f"تعداد ردیف‌های داده برای {symbol}: {len(df)}")
        
        df['symbol'] = symbol
        logger.info(f"شروع استخراج فیچرها برای {symbol}...")
        base_symbol = symbol.replace("USDT", "")
        df_feat = extract_features_full(df, symbol=symbol)
        
        # گرفتن دسته‌ای شاخص‌ها و احساسات
        timestamps = df_feat['timestamp'].tolist()
        spx_values = get_batch_indices(timestamps, 'SPX', batch_size=100)
        dxy_values = get_batch_indices(timestamps, 'DXY', batch_size=100)
        sentiment_values = get_batch_sentiments(symbol, timestamps, batch_size=100)
        
        with ThreadPoolExecutor(max_workers=28) as executor:
            for idx, ts in enumerate(df_feat['timestamp']):
                df_feat.at[idx, 'spx_index'] = spx_values.get(ts, 0.0)
                df_feat.at[idx, 'dxy_index'] = dxy_values.get(ts, 0.0)
                df_feat.at[idx, 'market_sentiment'] = sentiment_values.get(ts, 0.0)
        
        for feat in ['spx_index', 'dxy_index', 'btc_dominance', 'usdt_dominance', 'market_sentiment'] + [f'news_emb_{i}' for i in range(50)]:
            if feat not in df_feat.columns:
                df_feat[feat] = 0.0
                logger.warning(f"فیچر {feat} غایب بود، با 0.0 پر شد.")
            elif df_feat[feat].isna().any():
                df_feat[feat].fillna(0.0, inplace=True)
                logger.warning(f"مقادیر NaN در {feat} با 0.0 پر شد.")

        logger.info(f"فیچرها برای {symbol} استخراج شدند: {df_feat.shape}")
        
        df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_feat.fillna(0, inplace=True)
        
        batch_size = 1000
        for start in range(0, len(df_feat), batch_size):
            batch = df_feat.iloc[start:start + batch_size]
            batch_records = batch.to_dict(orient='records')
            if batch_records:
                with conn.begin():
                    save_features_to_db(batch_records)
                logger.debug(f"Batch {start//batch_size + 1}: تعداد ردیف‌ها={len(batch_records)}")
        
        logger.info(f"فیچرها برای {symbol} در دیتابیس ذخیره شدند.")
        
        df_labeled = label_data(df_feat)
        logger.info(f"لیبل‌گذاری برای {symbol} انجام شد: {df_labeled.shape}")
        
        return df_labeled
    except Exception as e:
        logger.error(f"خطا در پردازش {symbol}: {e}", exc_info=True)
        return None
    finally:
        conn.close()

def main(warm_start: bool = False):
    """تابع اصلی برای آموزش مدل"""
    try:
        logger.info("شروع آموزش مدل CatBoost Pro...")
        
        if not SYMBOLS:
            logger.error("لیست SYMBOLS خالی است. لطفاً فایل config.py را چک کنید.")
            return

        # پردازش موازی نمادها
        logger.info("پردازش موازی داده‌های نمادها...")
        with ThreadPoolExecutor() as executor:
            all_data = list(executor.map(process_symbol, SYMBOLS))
        all_data = [df for df in all_data if df is not None]
        
        if not all_data:
            logger.error("هیچ داده‌ای برای آموزش پیدا نشد.")
            return

        logger.info("ترکیب داده‌های تمام نمادها...")
        df_final = pd.concat(all_data)
        df_final.dropna(inplace=True)
        logger.info(f"داده‌های نهایی ساخته شدند: {df_final.shape}")

        # حذف ستون‌های تکراری
        if df_final.columns.duplicated().any():
            logger.warning(f"ستون‌های تکراری در df_final: {df_final.columns[df_final.columns.duplicated()].tolist()}")
            df_final = df_final.loc[:, ~df_final.columns.duplicated()]

        # استخراج فیچرهای ترید
        trade_features = []
        for _, row in df_final.iterrows():
            trade_feats = extract_trade_features(row['symbol'], row['timestamp'])
            trade_features.append(trade_feats)
        trade_df = pd.DataFrame(trade_features, index=df_final.index)
        common_columns = df_final.columns.intersection(trade_df.columns)
        trade_df = trade_df.drop(columns=common_columns, errors='ignore')
        df_final = pd.concat([df_final, trade_df], axis=1)

        # ذخیره دسته‌ای فیچرهای ترید
        batch_size = 100
        for start in range(0, len(trade_df), batch_size):
            batch = trade_df.iloc[start:start + batch_size]
            for _, row in batch.iterrows():
                try:
                    save_features_to_db(df_final['symbol'].iloc[0], row.to_dict())
                except Exception as e:
                    logger.error(f"خطا در ذخیره فیچرهای ترید: {e}")
        logger.info("فیچرهای ترید در دیتابیس ذخیره شدند.")

        # تعریف دسته‌های ویژگی‌ها
        fundamental_features = [
            'spx_index', 'dxy_index', 'btc_dominance', 'usdt_dominance', 'volume_score',
            'fundamental_score', 'news_score', 'market_sentiment', 'funding_rate'
        ]
        trade_features_list = ['trade_success_rate', 'avg_profit', 'avg_confidence', 'trade_count']
        # تعریف صریح embedding_features
        embedding_features = [f'news_emb_{i}' for i in range(50)]
        technical_features = [
            col for col in df_final.columns
            if col not in fundamental_features + trade_features_list + embedding_features +
            ['label', 'future_return', 'symbol', 'close', 'open', 'high', 'low', 'volume', 'timestamp', 'interval']
        ]

        # چک کردن وجود امبدینگ‌ها و پر کردن با صفر در صورت غایب بودن
        logger.info("چک کردن وجود ستون‌های امبدینگ...")
        for feat in embedding_features:
            if feat not in df_final.columns:
                df_final[feat] = 0.0
                logger.warning(f"ستون {feat} غایب بود، با 0.0 پر شد.")
            elif df_final[feat].isna().any():
                df_final[feat].fillna(0.0, inplace=True)
                logger.warning(f"مقادیر NaN در {feat} با 0.0 پر شدند.")

        # آماده‌سازی داده‌های امبدینگ
        logger.info("آماده‌سازی امبدینگ‌های خبری...")
        X_emb = df_final[embedding_features].fillna(0.0)

        # مدیریت مقادیر پرت
        df_final = replace_outliers(df_final, technical_features + fundamental_features + trade_features_list + embedding_features)

        # آماده‌سازی داده‌های آموزشی
        logger.debug(f"ستون‌های df_final: {df_final.columns.tolist()}")
        X = df_final[technical_features + fundamental_features + trade_features_list + embedding_features]
        X = X.select_dtypes(include=[np.number])
        y = df_final['label'].map({'Sell': 0, 'Hold': 1, 'Buy': 2})

        # چک کردن NaN/Inf
        logger.info("چک کردن NaN/Inf در X...")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.mean(), inplace=True)
        if X.isna().any().any():
            logger.error("هنوز مقادیر NaN در X وجود داره!")
            X.fillna(0.0, inplace=True)  # پر کردن با صفر و ادامه
            logger.warning("مقادیر NaN با 0.0 پر شدند.")

        # اعتبارسنجی کلی فیچرها
        new_features = ['trade_success_rate', 'avg_profit', 'avg_confidence', 'trade_count']
        start_timestamp = int((datetime.utcnow() - timedelta(days=30)).timestamp())
        end_timestamp = int(datetime.utcnow().timestamp())
        
        if not validate_features(X, y, new_features, SYMBOLS[0], start_timestamp, end_timestamp):
            logger.warning("حذف فیچرهای جدید به دلیل عملکرد ضعیف...")
            X = X.drop(columns=new_features, errors='ignore')
            trade_features_list = [f for f in trade_features_list if f not in new_features]

        # بررسی سازگاری فیچرها با مدل قبلی
        try:
            with open('models/catboost_pro_features.txt', 'r') as f:
                required_features = [line.strip() for line in f.readlines()]
            X = check_feature_compatibility(X, required_features)
        except FileNotFoundError:
            logger.warning("فایل فیچرهای مدل یافت نشد. ادامه با فیچرهای فعلی...")

        # جداسازی دسته‌های ویژگی‌ها
        X_tech = X[[col for col in technical_features if col in X.columns]]
        X_fund = X[[col for col in fundamental_features if col in X.columns]]
        X_trade = X[[col for col in trade_features_list if col in X.columns]]
        X_emb = X[[col for col in embedding_features if col in X.columns]]

        logger.info(f"ویژگی‌های تکنیکال: {X_tech.columns.tolist()}")
        logger.info(f"ویژگی‌های فاندامنتال: {X_fund.columns.tolist()}")
        logger.info(f"ویژگی‌های ترید: {X_trade.columns.tolist()}")
        logger.info(f"ویژگی‌های امبدینگ: {X_emb.columns.tolist()}")

        # اجرای RFE برای ویژگی‌های تکنیکال
        selected_tech_features = load_selected_features()
        if should_update_rfe() or selected_tech_features is None:
            logger.info("اجرای RFE برای فیچرهای تکنیکال...")
            estimator = CatBoostClassifier(verbose=0, thread_count=-1)
            rfe, selected_tech_features = custom_rfe_logging(estimator, X_tech, y, n_features_to_select=50)
            if selected_tech_features:
                save_rfe_update_date()
                joblib.dump(rfe, 'models/rfe_model.pkl')
            else:
                logger.error("RFE هیچ ویژگی‌ای انتخاب نکرد. استفاده از همه فیچرها.")
                selected_tech_features = X_tech.columns.tolist()
        else:
            logger.info(f"استفاده از فیچرهای قبلی: {selected_tech_features}")

        # ترکیب ویژگی‌های نهایی
        X = pd.concat([X_tech[selected_tech_features], X_fund, X_trade, X_emb], axis=1)
        logger.info(f"ابعاد داده‌های آموزشی (X): {X.shape}")

        if X.empty:
            logger.error("داده‌های آموزشی خالی هستند.")
            return

        # تقسیم داده‌ها
        logger.info("تقسیم داده‌ها به آموزشی و آزمایشی...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"تقسیم داده‌ها انجام شد: X_train={X_train.shape}, X_test={X_test.shape}")

        # حذف ستون‌های تکراری
        if X_train.columns.duplicated().any():
            logger.warning(f"ستون‌های تکراری در X_train: {X_train.columns[X_train.columns.duplicated()].tolist()}")
            X_train = X_train.loc[:, ~X_train.columns.duplicated()]
            X_test = X_test.loc[:, ~X_train.columns.duplicated()]

        # چک کردن مقادیر نامناسب
        logger.info("چک کردن مقادیر نامناسب (NaN/Inf) در داده‌ها...")
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_train.fillna(X_train.mean(), inplace=True)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test.fillna(X_test.mean(), inplace=True)
        logger.info("مقادیر نامناسب اصلاح شدند.")

        # اصلاح نوع داده‌ها
        X_train = ensure_dtypes(X_train)
        X_test = ensure_dtypes(X_test)

        # متعادل‌سازی کلاس‌ها با SMOTE
        logger.info("اعمال SMOTE برای متعادل‌سازی کلاس‌ها...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"ابعاد داده‌های متعادل‌شده: X_train_resampled={X_train_resampled.shape}")

        # محاسبه وزن کلاس‌ها
        class_counts = pd.Series(y_train_resampled).value_counts()
        total_samples = len(y_train_resampled)
        class_weights = {i: total_samples / (len(class_counts) * count) for i, count in class_counts.items()}
        logger.info(f"وزن‌های محاسبه‌شده برای کلاس‌ها: {class_weights}")

        # آموزش مدل با GridSearchCV
        logger.info("شروع بهینه‌سازی CatBoost Pro...")
        catboost = CatBoostClassifier(verbose=0, thread_count=-1, class_weights=class_weights)
        
        if warm_start and os.path.exists('models/catboost_pro_model.pkl'):
            logger.info("لود مدل قبلی برای warm-start...")
            catboost = joblib.load('models/catboost_pro_model.pkl')
            catboost.fit(X_train_resampled, y_train_resampled, init_model=catboost)
        else:
            if warm_start:
                logger.warning("فایل مدل قبلی (catboost_pro_model.pkl) یافت نشد. مدل از ابتدا ساخته می‌شود.")
            
            catboost_params = {
                'iterations': [500, 1000],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [6, 8, 10],
                'l2_leaf_reg': [3, 5, 7],
                'bagging_temperature': [0.5, 1.0]
            }
            
            grid_search = GridSearchCV(
                estimator=catboost,
                param_grid=catboost_params,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            logger.info("اجرای GridSearchCV برای یافتن بهترین پارامترها...")
            grid_search.fit(X_train_resampled, y_train_resampled)
            
            logger.info(f"بهترین پارامترها: {grid_search.best_params_}")
            logger.info(f"بهترین امتیاز: {grid_search.best_score_:.4f}")
            
            catboost = grid_search.best_estimator_
            
            # ذخیره موقت مدل
            joblib.dump(catboost, 'models/catboost_pro_model_temp.pkl')
            logger.info("مدل موقت ذخیره شد: models/catboost_pro_model_temp.pkl")

        # ارزیابی مدل
        logger.info("ارزیابی مدل روی داده‌های آزمایشی...")
        test_pool = Pool(X_test, y_test)
        y_pred = catboost.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"دقت مدل: {accuracy:.4f}")
        
        logger.info("گزارش طبقه‌بندی:")
        report = classification_report(y_test, y_pred, target_names=['Sell', 'Hold', 'Buy'])
        logger.info(report)
        
        # رسم ماتریس درهم‌ریختگی
        plot_confusion_matrix(y_test, y_pred, 'CatBoost_Pro')
        
        # ذخیره مدل و فیچرها
        logger.info("ذخیره مدل و فیچرها...")
        joblib.dump(catboost, 'models/catboost_pro_model.pkl')
        with open('models/catboost_pro_features.txt', 'w') as f:
            for feature in X.columns:
                f.write(feature + '\n')
        logger.info("مدل و فیچرها ذخیره شدند: models/catboost_pro_model.pkl, models/catboost_pro_features.txt")

    except Exception as e:
        logger.error(f"خطا در اجرای فرآیند آموزش: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main(warm_start=False)