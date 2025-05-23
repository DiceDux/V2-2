import time
import subprocess
from datetime import datetime

def run_news_and_market():
    """اجرای news_fetcher و market_index_updater"""
    print("📰 اجرای news_fetcher.py ...")
    subprocess.run(["python", "news_fetcher.py"])

    print("📈 اجرای market_index_updater.py ...")
    subprocess.run(["python", "market_index_updater.py"])

def run_retrain_model():
    """اجرای retrain_model.py"""
    print("🤖 اجرای retrain_model.py ...")
    subprocess.run(["python", "retrain_model.py"])

def run_backtest():
    """اجرای backtest.py"""
    print("📊 اجرای backtest.py ...")
    subprocess.run(["python", "backtest.py"])

last_retrain_time = 0
last_backtest_time = 0
retrain_interval = 24 * 3600  # 24 ساعت
backtest_interval = 24 * 3600  # 24 ساعت

while True:
    current_time = time.time()
    
    # اجرای news_fetcher و market_index_updater هر دقیقه
    run_news_and_market()
    
    # اجرای retrain_model هر 24 ساعت
    if current_time - last_retrain_time >= retrain_interval:
        run_retrain_model()
        last_retrain_time = current_time
    
    # اجرای backtest هر 24 ساعت (بعد از retrain)
    if current_time - last_backtest_time >= backtest_interval:
        run_backtest()
        last_backtest_time = current_time
    
    print("⏳ صبر ۶۰ ثانیه...")
    time.sleep(60)