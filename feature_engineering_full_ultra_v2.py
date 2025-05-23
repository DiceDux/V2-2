import pandas as pd
import numpy as np
import ta
from data.market_index_manager import get_index
from ai.fundamental_analyzer import get_fundamental_scores_batch
from transformers import AutoTokenizer, AutoModel
import torch
from config import MYSQL_CONFIG
from datetime import datetime
import logging
from sqlalchemy.sql import text
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import polars as pl
from sklearn.decomposition import PCA

# دقیق‌ترین لاگینگ
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_eng_debug.log", 'w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# تنظیم مدل FinBERT برای امبدینگ‌های خبری (فقط هشدار و خطا)
try:
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    model.eval()
except Exception as e:
    logger.error(f"خطا در بارگذاری مدل FinBERT: {e}")
    tokenizer, model = None, None

def validate_input_df(df: pd.DataFrame) -> bool:
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f"ستون‌های موردنیاز یافت نشدند: {missing}")
        return False
    return True

def calculate_ichimoku_cloud(df: pd.DataFrame) -> dict:
    logger.debug("محاسبه اندیکاتور Ichimoku Cloud")
    high = df['high']
    low = df['low']
    close = df['close']
    tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou_span = close.shift(-26)
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    logger.debug("محاسبه VWAP")
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    logger.debug("محاسبه OBV")
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def calculate_chaikin_oscillator(df: pd.DataFrame) -> pd.Series:
    logger.debug("محاسبه Chaikin Oscillator")
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_multiplier = money_flow_multiplier.fillna(0)
    money_flow_volume = money_flow_multiplier * df['volume']
    ad = money_flow_volume.cumsum()
    short_ema = ad.ewm(span=3, adjust=False).mean()
    long_ema = ad.ewm(span=10, adjust=False).mean()
    return short_ema - long_ema

@lru_cache(maxsize=1000)
def fetch_news_embeddings(symbol: str, timestamp: int) -> tuple:
    from data.data_manager import get_recent_news_texts
    try:
        if not symbol:
            logger.warning("نماد (symbol) خالی یا None است.")
            return tuple([0.0] * 50)
        base_symbol = symbol.replace("USDT", "")
        news_text = get_recent_news_texts(base_symbol, timestamp, hours=720)
        if not news_text:
            logger.warning(f"هیچ خبری برای {base_symbol} در timestamp {timestamp} یافت نشد.")
            return tuple([0.0] * 50)
        if not tokenizer or not model:
            logger.warning("مدل FinBERT لود نشده است.")
            return tuple([0.0] * 50)
        keywords = [base_symbol.lower(), "crypto", "bitcoin", "blockchain", "cryptocurrency"]
        if not any(keyword in news_text.lower() for keyword in keywords):
            logger.warning(f"خبر غیرمرتبط برای {base_symbol}: {news_text[:100]}...")
            return tuple([0.0] * 50)
        inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        pca = PCA(n_components=50)
        reduced_embeddings = pca.fit_transform(embeddings.reshape(1, -1))[0]
        return [float(e) if pd.notna(e) else 0.0 for e in reduced_embeddings]
    except Exception as e:
        logger.error(f"خطا در تولید امبدینگ برای {symbol}: {e}", exc_info=True)
        return tuple([0.0] * 50)

def fetch_embeddings_parallel(df, symbol):
    with ThreadPoolExecutor(max_workers=28) as executor:
        embeddings = list(executor.map(lambda ts: fetch_news_embeddings(symbol, int(ts)), df['timestamp']))
    return embeddings

def generate_sql_query(num_embeddings=50) -> tuple:
    from data.data_manager import get_connection
    base_columns = [
        'symbol', 'timestamp', 'interval', 'open', 'high', 'low', 'close', 'volume',
        'ema20', 'ema50', 'ema200', 'rsi', 'atr', 'tema20', 'dema20', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'bb_mid', 'bb_width', 'bb_squeeze', 'keltner_upper', 'keltner_lower',
        'donchian_upper', 'donchian_lower', 'obv', 'vwap', 'adx', 'breakout', 'breakdown', 'volume_spike',
        'vwap_buy_signal', 'rsi_slope', 'macd_slope', 'rsi_macd_converge', 'stoch_rsi', 'cci', 'willr', 'mfi',
        'roc', 'momentum', 'psar', 'ult_osc', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
        'chikou_span', 'daily_return', 'candle_length_pct', 'candle_length', 'candle_body', 'upper_wick',
        'lower_wick', 'body_to_range', 'relative_volume', 'ema_cross', 'trend_strength', 'trend_age',
        'range_pct', 'range_spike', 'ha_close', 'ha_open', 'gap_up', 'gap_down', 'ema_spread',
        'ema_compression', 'bullish_candles', 'bullish_streak', 'avg_true_body', 'div_rsi', 'div_macd',
        'div_obv', 'confirmed_rsi_div', 'volatility_14', 'z_score', 'doji', 'hammer', 'inv_hammer',
        'hanging_man', 'engulfing_bull', 'engulfing_bear', 'morning_star', 'evening_star', 'harami_bull',
        'harami_bear', 'piercing_line', 'three_white_soldiers', 'spinning_top', 'marubozu', 'three_black_crows',
        'combo_signal', 'hour', 'session_asia', 'session_europe', 'session_us', 'day_of_week', 'month',
        'price_to_30d_mean', 'double_top', 'head_shoulders', 'support_zone', 'resistance_zone', 'support_bounce',
        'resistance_reject', 'cup', 'handle', 'cup_and_handle', 'higher_highs', 'lower_lows', 'diamond_top',
        'flag_pole', 'flag_body', 'flag_pattern', 'fibo_0_5_bounce', 'fibo_0_618_bounce', 'spx_index',
        'dxy_index', 'btc_dominance', 'usdt_dominance', 'low_volatility', 'volume_mean', 'low_volume',
        'weak_trend', 'low_adx', 'low_z_score', 'chaikin_osc', 'fundamental_score', 'news_score',
        'volume_score', 'market_sentiment', 'funding_rate'
    ]
    embedding_columns = [f'news_emb_{i}' for i in range(num_embeddings)]
    all_columns = base_columns + embedding_columns

    def quote_if_reserved(col):
        return f'`{col}`' if col.lower() == 'interval' else col

    insert_columns = ', '.join([quote_if_reserved(col) for col in all_columns])
    placeholders = ', '.join([f':{col}' for col in all_columns])
    update_statements = ', '.join([f"{quote_if_reserved(col)} = VALUES({quote_if_reserved(col)})" for col in all_columns if col not in ['symbol', 'timestamp']])

    query = f"""
        INSERT INTO features ({insert_columns})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_statements}
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("DESCRIBE features")
            db_columns = {row[0] for row in cursor.fetchall()}
            missing_columns = set(all_columns) - db_columns
            if missing_columns:
                logger.warning(f"ستون‌های زیر در جدول features وجود ندارند: {missing_columns}")
    except Exception as e:
        logger.error(f"خطا در بررسی ستون‌های دیتابیس: {e}", exc_info=True)
    finally:
        if 'conn' in locals() and conn:
            conn.close()
    return query, all_columns

def calculate_technical_features(df: pd.DataFrame) -> dict:
    import traceback
    try:
        logger.info("START calculate_technical_features")
        logger.debug(f"Input df shape: {df.shape}")
        logger.debug(f"Input df columns: {df.columns.tolist()}")
        logger.debug(f"Input df dtypes: {df.dtypes.to_dict()}")
        logger.debug(f"Input df head:\n{df.head(3)}")
        for col in df.columns:
            logger.debug(f"Type counts for column [{col}]: {df[col].apply(type).value_counts().to_dict()}")
            logger.debug(f"Null count for [{col}]: {df[col].isnull().sum()}")

        required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"ستون‌های موردنیاز غایبند: {missing_cols}")
            return {}

        logger.info("تبدیل دیتافریم به Polars...")
        df_pl = pl.from_pandas(df)
        logger.debug(f"Polars columns after from_pandas: {df_pl.columns}")

        cast_cols = ['open', 'high', 'low', 'close', 'volume']
        logger.info("تبدیل نوع داده‌های ستون‌ها به Float64...")
        df_pl = df_pl.with_columns([
            pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0).alias(col) for col in cast_cols
        ])
        logger.debug(f"Polars columns after casting: {df_pl.columns}")

        df_pl = df_pl.with_columns(
            pl.col('timestamp').cast(pl.Float64, strict=False).fill_null(0.0).alias('timestamp')
        )
        logger.debug(f"Polars columns after timestamp cast: {df_pl.columns}")

        features = {}

        # EMA
        df_pl = df_pl.with_columns([
            pl.col('close').ewm_mean(span=20).fill_null(0.0).alias('ema20'),
            pl.col('close').ewm_mean(span=50).fill_null(0.0).alias('ema50'),
            pl.col('close').ewm_mean(span=200).fill_null(0.0).alias('ema200')
        ])
        features['ema20'] = df_pl['ema20'].to_pandas()
        features['ema50'] = df_pl['ema50'].to_pandas()
        features['ema200'] = df_pl['ema200'].to_pandas()

        # RSI
        df_pl = df_pl.with_columns([
            pl.col('close').diff().fill_null(0.0).alias('delta')
        ])
        df_pl = df_pl.with_columns([
            pl.when(pl.col('delta') > 0).then(pl.col('delta')).otherwise(0).alias('gain'),
            pl.when(pl.col('delta') < 0).then(-pl.col('delta')).otherwise(0).alias('loss')
        ])
        df_pl = df_pl.with_columns([
            pl.col('gain').ewm_mean(span=14).fill_null(0.0).alias('avg_gain'),
            pl.col('loss').ewm_mean(span=14).fill_null(0.0).alias('avg_loss')
        ])
        df_pl = df_pl.with_columns([
            (pl.col('avg_gain') / pl.col('avg_loss').replace(0, 1)).alias('rs')
        ])
        df_pl = df_pl.with_columns([
            (100 - (100 / (1 + pl.col('rs')))).fill_null(0.0).alias('rsi')
        ])
        features['rsi'] = df_pl['rsi'].to_pandas()

        # ATR
        logger.info("محاسبه ATR")
        df_pl = df_pl.with_columns([
            (pl.col('high') - pl.col('low')).abs().alias('tr1'),
            (pl.col('high') - pl.col('close').shift(1)).abs().fill_null(0.0).alias('tr2'),
            (pl.col('low') - pl.col('close').shift(1)).abs().fill_null(0.0).alias('tr3')
        ])
        logger.debug(f"ATR intermediate columns: {df_pl.columns}")
        for col in ['tr1', 'tr2', 'tr3']:
            logger.debug(f"tr-check: {col} in columns? {col in df_pl.columns}")
            logger.debug(f"{col} null count: {df_pl[col].null_count()}")
            logger.debug(f"{col} dtype: {df_pl[col].dtype}")
        df_pl = df_pl.with_columns([
            pl.max_horizontal([pl.col('tr1'), pl.col('tr2'), pl.col('tr3')]).alias('tr')
        ])
        logger.debug(f"ATR after max_horizontal: {df_pl.columns}")
        df_pl = df_pl.with_columns([
            pl.col('tr').ewm_mean(span=14).fill_null(0.0).alias('atr')
        ])
        features['atr'] = df_pl['atr'].to_pandas()
        logger.debug("ATR feature done")

        # TEMA / DEMA (نمونه)
        df_pl = df_pl.with_columns([
            pl.col('ema20').ewm_mean(span=20).fill_null(0.0).alias('ema20_2')
        ])
        df_pl = df_pl.with_columns([
            pl.col('ema20_2').ewm_mean(span=20).fill_null(0.0).alias('ema20_3')
        ])
        df_pl = df_pl.with_columns([
            (3 * pl.col('ema20') - 3 * pl.col('ema20_2') + pl.col('ema20_3')).fill_null(0.0).alias('tema20'),
            (2 * pl.col('ema20') - pl.col('ema20_2')).fill_null(0.0).alias('dema20')
        ])
        features['tema20'] = df_pl['tema20'].to_pandas()
        features['dema20'] = df_pl['dema20'].to_pandas()

        # MACD
        df_pl = df_pl.with_columns([
            pl.col('close').ewm_mean(span=12).fill_null(0.0).alias('ema12'),
            pl.col('close').ewm_mean(span=26).fill_null(0.0).alias('ema26')
        ])
        df_pl = df_pl.with_columns([
            (pl.col('ema12') - pl.col('ema26')).fill_null(0.0).alias('macd')
        ])
        df_pl = df_pl.with_columns([
            pl.col('macd').ewm_mean(span=9).fill_null(0.0).alias('macd_signal')
        ])
        features['macd'] = df_pl['macd'].to_pandas()
        features['macd_signal'] = df_pl['macd_signal'].to_pandas()

        # Bollinger Bands
        df_pl = df_pl.with_columns([
            pl.col('close').rolling_mean(20).fill_null(0.0).alias('rolling_mean'),
            pl.col('close').rolling_std(20).fill_null(0.0).alias('rolling_std')
        ])
        df_pl = df_pl.with_columns([
            (pl.col('rolling_mean') + 2 * pl.col('rolling_std')).fill_null(0.0).alias('bb_upper'),
            (pl.col('rolling_mean') - 2 * pl.col('rolling_std')).fill_null(0.0).alias('bb_lower'),
            pl.col('rolling_mean').fill_null(0.0).alias('bb_mid')
        ])
        features['bb_upper'] = df_pl['bb_upper'].to_pandas()
        features['bb_lower'] = df_pl['bb_lower'].to_pandas()
        features['bb_mid'] = df_pl['bb_mid'].to_pandas()
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']).fillna(0.0)
        features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(window=20).mean() * 0.8).astype(float).fillna(0.0)

        # Keltner Channels
        features['keltner_upper'] = (features['ema20'] + 2 * features['atr']).fillna(0.0)
        features['keltner_lower'] = (features['ema20'] - 2 * features['atr']).fillna(0.0)

        # Donchian Channels
        df_pl = df_pl.with_columns([
            pl.col('high').rolling_max(20).fill_null(0.0).alias('donchian_upper'),
            pl.col('low').rolling_min(20).fill_null(0.0).alias('donchian_lower')
        ])
        features['donchian_upper'] = df_pl['donchian_upper'].to_pandas()
        features['donchian_lower'] = df_pl['donchian_lower'].to_pandas()

        # OBV
        features['obv'] = calculate_obv(df).fillna(0.0)
        df_pl = df_pl.with_columns(pl.Series('obv', features['obv'].values).cast(pl.Float64).fill_null(0.0).alias('obv'))

        # VWAP
        features['vwap'] = calculate_vwap(df).fillna(0.0)
        df_pl = df_pl.with_columns(pl.Series('vwap', features['vwap'].values).cast(pl.Float64).fill_null(0.0).alias('vwap'))

        # ADX
        features['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx().fillna(0.0)

        # Breakout/Breakdown
        df_pl = df_pl.with_columns([
            (pl.col('close') > pl.col('donchian_upper')).cast(pl.Float64).fill_null(0.0).alias('breakout'),
            (pl.col('close') < pl.col('donchian_lower')).cast(pl.Float64).fill_null(0.0).alias('breakdown')
        ])
        features['breakout'] = df_pl['breakout'].to_pandas()
        features['breakdown'] = df_pl['breakdown'].to_pandas()

        # Volume Spike
        df_pl = df_pl.with_columns([
            (pl.col('volume') > pl.col('volume').rolling_mean(20) * 1.5).cast(pl.Float64).fill_null(0.0).alias('volume_spike')
        ])
        features['volume_spike'] = df_pl['volume_spike'].to_pandas()

        # VWAP Buy Signal
        df_pl = df_pl.with_columns([
            ((pl.col('close') > pl.col('vwap')) & (pl.col('volume_spike') > 0)).cast(pl.Float64).fill_null(0.0).alias('vwap_buy_signal')
        ])
        features['vwap_buy_signal'] = df_pl['vwap_buy_signal'].to_pandas()

        # شیب RSI و MACD
        features['rsi_slope'] = features['rsi'].diff().fillna(0.0)
        features['macd_slope'] = features['macd'].diff().fillna(0.0)
        features['rsi_macd_converge'] = ((features['rsi_slope'] > 0) & (features['macd_slope'] > 0)).astype(float).fillna(0.0)

        # سایر اندیکاتورها
        features['stoch_rsi'] = ta.momentum.StochRSIIndicator(close=df['close']).stochrsi().fillna(0.0)
        features['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci().fillna(0.0)
        features['willr'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14).williams_r().fillna(0.0)
        features['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).money_flow_index().fillna(0.0)
        features['roc'] = ta.momentum.ROCIndicator(close=df['close'], window=10).roc().fillna(0.0)
        df_pl = df_pl.with_columns([
            (pl.col('close') - pl.col('close').shift(10)).fill_null(0.0).alias('momentum')
        ])
        features['momentum'] = df_pl['momentum'].to_pandas()
        features['psar'] = ta.trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close']).psar().fillna(0.0)

        # Ultimate Oscillator
        df_pl = df_pl.with_columns([
            (pl.col('close') - pl.min_horizontal([pl.col('low'), pl.col('close').shift(1)])).fill_null(0.0).alias('bp'),
            pl.max_horizontal([
                (pl.col('high') - pl.col('low')).abs(),
                (pl.col('high') - pl.col('close').shift(1)).abs(),
                (pl.col('low') - pl.col('close').shift(1)).abs()
            ]).fill_null(0.0).alias('tr_uo')
        ])
        df_pl = df_pl.with_columns([
            (pl.col('bp').rolling_sum(7) / pl.col('tr_uo').rolling_sum(7)).fill_null(0.0).alias('avg7'),
            (pl.col('bp').rolling_sum(14) / pl.col('tr_uo').rolling_sum(14)).fill_null(0.0).alias('avg14'),
            (pl.col('bp').rolling_sum(28) / pl.col('tr_uo').rolling_sum(28)).fill_null(0.0).alias('avg28')
        ])
        df_pl = df_pl.with_columns([
            (100 * (4 * pl.col('avg7') + 2 * pl.col('avg14') + pl.col('avg28')) / 7).fill_null(0.0).alias('ult_osc')
        ])
        features['ult_osc'] = df_pl['ult_osc'].to_pandas()

        # ویژگی‌های کندل
        df_pl = df_pl.with_columns([
            pl.col('close').pct_change().fill_null(0.0).alias('daily_return'),
            ((pl.col('high') - pl.col('low')) / pl.col('open').replace(0, 1)).fill_null(0.0).alias('candle_length_pct'),
            (pl.col('high') - pl.col('low')).fill_null(0.0).alias('candle_length'),
            (pl.col('close') - pl.col('open')).abs().fill_null(0.0).alias('candle_body'),
            (pl.col('high') - pl.max_horizontal([pl.col('close'), pl.col('open')])).fill_null(0.0).alias('upper_wick'),
            (pl.min_horizontal([pl.col('close'), pl.col('open')]) - pl.col('low')).fill_null(0.0).alias('lower_wick'),
            (pl.col('volume') / pl.col('volume').rolling_mean(20)).fill_null(0.0).alias('relative_volume'),
            ((pl.col('open') + pl.col('high') + pl.col('low') + pl.col('close')) / 4).fill_null(0.0).alias('ha_close'),
            ((pl.col('open').shift(1) + pl.col('close').shift(1)) / 2).fill_null(0.0).alias('ha_open'),
            (pl.col('open') > pl.col('close').shift(1) * 1.01).cast(pl.Float64).fill_null(0.0).alias('gap_up'),
            (pl.col('open') < pl.col('close').shift(1) * 0.99).cast(pl.Float64).fill_null(0.0).alias('gap_down'),
            (pl.col('close') > pl.col('open')).cast(pl.Float64).fill_null(0.0).alias('bullish_candles')
        ])
        features['daily_return'] = df_pl['daily_return'].to_pandas()
        features['candle_length_pct'] = df_pl['candle_length_pct'].to_pandas()
        features['candle_length'] = df_pl['candle_length'].to_pandas()
        features['candle_body'] = df_pl['candle_body'].to_pandas()
        features['upper_wick'] = df_pl['upper_wick'].to_pandas()
        features['lower_wick'] = df_pl['lower_wick'].to_pandas()
        features['body_to_range'] = (features['candle_body'] / features['candle_length'].replace(0, 1)).fillna(0.0)
        features['relative_volume'] = df_pl['relative_volume'].to_pandas()
        features['ha_close'] = df_pl['ha_close'].to_pandas()
        features['ha_open'] = df_pl['ha_open'].to_pandas()
        features['gap_up'] = df_pl['gap_up'].to_pandas()
        features['gap_down'] = df_pl['gap_down'].to_pandas()
        features['bullish_candles'] = df_pl['bullish_candles'].to_pandas()
        features['ema_cross'] = (features['ema20'] > features['ema50']).astype(float).fillna(0.0)
        features['trend_strength'] = ((features['ema20'] - features['ema50']) / features['ema50'].replace(0, 1)).fillna(0.0)
        features['trend_age'] = features['ema_cross'].groupby((features['ema_cross'] != features['ema_cross'].shift()).cumsum()).cumcount() + 1
        features['range_pct'] = (features['candle_length'] / features['candle_length'].rolling(20).mean().replace(0, 1)).fillna(0.0)
        features['range_spike'] = (features['range_pct'] > 1.5).astype(float).fillna(0.0)
        features['ema_spread'] = ((features['ema200'] - features['ema20']) / features['ema20'].replace(0, 1)).fillna(0.0)
        features['ema_compression'] = (features['ema_spread'].abs() < features['ema_spread'].rolling(20).std()).astype(float).fillna(0.0)
        features['bullish_streak'] = features['bullish_candles'].groupby((features['bullish_candles'] != features['bullish_candles'].shift()).cumsum()).cumcount() + 1
        features['avg_true_body'] = features['candle_body'].rolling(window=14).mean().fillna(0.0)
        df_pl = df_pl.with_columns([
            (pl.col('close').diff() - pl.col('rsi').diff()).fill_null(0.0).alias('div_rsi'),
            (pl.col('close').diff() - pl.col('macd').diff()).fill_null(0.0).alias('div_macd'),
            (pl.col('close').diff() - pl.col('obv').diff()).fill_null(0.0).alias('div_obv')
        ])
        features['div_rsi'] = df_pl['div_rsi'].to_pandas()
        features['div_macd'] = df_pl['div_macd'].to_pandas()
        features['div_obv'] = df_pl['div_obv'].to_pandas()
        features['confirmed_rsi_div'] = ((features['div_rsi'] > 0) & (features['adx'] < 25)).astype(float).fillna(0.0)
        df_pl = df_pl.with_columns([
            pl.col('close').rolling_std(14).fill_null(0.0).alias('volatility_14'),
            ((pl.col('close') - pl.col('close').rolling_mean(20)) / pl.col('close').rolling_std(20).replace(0, 1)).fill_null(0.0).alias('z_score')
        ])
        features['volatility_14'] = df_pl['volatility_14'].to_pandas()
        features['z_score'] = df_pl['z_score'].to_pandas()

        logger.info("پایان محاسبه فیچرها. کلیدهای features:")
        logger.debug(f"feature keys: {list(features.keys())}")

        return features

    except Exception as e:
        logger.error(f"EXCEPTION in calculate_technical_features: {e}\n{traceback.format_exc()}")
        return {}

def extract_features_full(df: pd.DataFrame, symbol: str = None, interval: str = None) -> pd.DataFrame:
    from data.data_manager import add_news_to_candles, save_features_to_db
    if not validate_input_df(df):
        logger.error("دیتافریم ورودی نامعتبر است.")
        return df
    df = df.copy()
    logger.info(f"شروع استخراج فیچر. df shape: {df.shape}")
    logger.debug(f"df head:\n{df.head(3)}")
    df = add_news_to_candles(df, symbol, hours_window=720)
    if symbol:
        df['symbol'] = symbol
    else:
        logger.warning("نماد مشخص نشده است.")
        symbol = df.get('symbol', 'BTC').iloc[0].replace("USDT", "") if 'symbol' in df.columns else 'BTC'
        df['symbol'] = symbol
    if interval:
        df['interval'] = interval
    features = calculate_technical_features(df)
    if not features:
        logger.error("هیچ فیچری محاسبه نشد.")
        return df
    if 'timestamp' in df.columns:
        embedding_df = pd.DataFrame(index=df.index, columns=[f'news_emb_{i}' for i in range(50)])
        embeddings = fetch_embeddings_parallel(df, symbol)
        for i, emb in enumerate(embeddings):
            if not isinstance(emb, (list, tuple)) or len(emb) != 50:
                logger.warning(f"امبدینگ نامعتبر برای ردیف {i}: {emb}")
                emb = [0.0] * 50
            for j, val in enumerate(emb):
                embedding_df.iloc[i, j] = float(val) if pd.notna(val) else 0.0
        embedding_df = embedding_df.astype(float).fillna(0.0)
        for col in embedding_df.columns:
            features[col] = embedding_df[col].astype(float)
    features_df = pd.DataFrame(features, index=df.index)
    common_columns = df.columns.intersection(features_df.columns)
    features_df = features_df.drop(columns=common_columns, errors='ignore')
    result_df = pd.concat([df, features_df], axis=1)
    result_df = result_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in result_df.columns:
        if col not in ['symbol', 'interval']:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0.0)
    if result_df.columns.duplicated().any():
        logger.warning(f"ستون‌های تکراری یافت شدند: {result_df.columns[result_df.columns.duplicated()].tolist()}")
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    logger.debug(f"ستون‌های result_df: {result_df.columns.tolist()}")
    records = result_df.to_dict(orient='records')
    logger.debug(f"تعداد ردیف‌های تبدیل‌شده به دیکشنری: {len(records)}")
    save_features_to_db(records)
    logger.info(f"پایان استخراج فیچر. result_df shape: {result_df.shape}")
    return result_df