import re
import pandas as pd
from datetime import datetime
from data.data_manager import get_connection
from sqlalchemy import text
import logging

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

log_file_path = "logs/ai_decisions.log"

def save_sentiment_analysis_to_db(df):
    """ذخیره نتایج تحلیل احساسات در دیتابیس"""
    conn = get_connection()
    if not conn:
        logger.error("نمی‌توان به دیتابیس متصل شد.")
        return
    
    query = """
        INSERT INTO news_sentiment_analysis (symbol, action, confidence, news_score, timestamp)
        VALUES (:symbol, :action, :confidence, :news_score, :timestamp)
    """
    
    for _, row in df.iterrows():
        try:
            timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                text(query),
                {
                    'symbol': row['symbol'],
                    'action': row['action'],
                    'confidence': row['confidence'],
                    'news_score': row['news_score'] if pd.notnull(row['news_score']) else None,
                    'timestamp': timestamp
                }
            )
            logger.info(f"تحلیل احساسات برای {row['symbol']} ذخیره شد.")
        except Exception as e:
            logger.error(f"خطا در ذخیره تحلیل احساسات برای {row['symbol']}: {e}")
    
    conn.close()
    logger.info("نتایج تحلیل احساسات با موفقیت در دیتابیس ذخیره شدند.")

data = []
with open(log_file_path, encoding="utf-8") as f:
    lines = f.readlines()

    for i, line in enumerate(lines):
        # فقط خط تصمیم مدل
        if "تصمیم مدل:" in line:
            match = re.search(r"\[(.*?)\].*?تصمیم مدل: (\w+).*?اعتماد: ([\d\.]+)", line)
            if match:
                symbol = match.group(1)
                action = match.group(2).upper()
                confidence = float(match.group(3))

                # سعی کن خط قبلی که شامل news_score است را پیدا کنی
                news_score = None
                for j in range(i - 1, max(0, i - 10), -1):
                    if "features_df_columns" in lines[j]:  # لاگ columns شامل news_score است
                        if "news_score" in lines[j]:
                            # خط بعدی ممکن است شامل مقدار news_score باشد
                            for k in range(j, min(len(lines), j + 10)):
                                if "features:" in lines[k].lower():
                                    score_match = re.search(r"news_score\s+([-\d\.eE]+)", lines[k])
                                    if score_match:
                                        news_score = float(score_match.group(1))
                                        break
                            break

                data.append({
                    "symbol": symbol,
                    "action": action,
                    "confidence": confidence,
                    "news_score": news_score,
                })

# ساخت دیتافریم نهایی
df = pd.DataFrame(data)

if df.empty:
    logger.warning("⛔ هیچ داده‌ای قابل استخراج نیست.")
else:
    logger.info("[+] آمار خلاصه news_score بر اساس خروجی مدل:")
    logger.info(df.groupby("action")["news_score"].describe().to_string())

    # ذخیره نتایج در دیتابیس
    save_sentiment_analysis_to_db(df)

    import matplotlib.pyplot as plt

    df.boxplot(column="news_score", by="action")
    plt.title("News Score Distribution by Model Action")
    plt.suptitle("")
    plt.ylabel("news_score")
    plt.xlabel("Model Action")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('notebooks/news_score_distribution.png')
    plt.close()
    logger.info("نمودار توزیع news_score ذخیره شد: notebooks/news_score_distribution.png")