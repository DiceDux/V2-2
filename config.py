import os
from dotenv import load_dotenv

# مسیر فایل config.env
config_path = os.path.join(os.path.dirname(__file__), "config.env")
print("Config file path:", config_path)
print("Config file exists:", os.path.exists(config_path))
load_dotenv(config_path)

# دیباگ متغیرهای محیطی
print("MYSQL_HOST:", os.getenv("MYSQL_HOST"))
print("MYSQL_PORT:", os.getenv("MYSQL_PORT"))
print("MYSQL_USER:", os.getenv("MYSQL_USER"))
print("MYSQL_DATABASE:", os.getenv("MYSQL_DATABASE"))
print("COINEX_API_KEY:", os.getenv("COINEX_API_KEY"))

TRADE_MODE = "simulation"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]

BASE_SYMBOLS = ["BTC", "ETH", "DOGE", "SOL"]

TRADE_EXECUTION_MODE = "demo"  # یا "real" برای ترید واقعی
TIMEFRAME_MINUTES = 240
COINEX_BASE_URL = "https://api.coinex.com"
COINEX_KLINE_ENDPOINT = "/market/kline"
COINEX_API_KEY = os.getenv("COINEX_API_KEY")
COINEX_API_SECRET = os.getenv("COINEX_API_SECRET")
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECTRET')
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY")
INITIAL_BALANCE = 1000
CANDLE_HISTORY_LIMIT = 200
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "port": int(os.getenv("MYSQL_PORT")),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE")
}
API_RATE_LIMIT = 10
API_TIMEOUT = 5