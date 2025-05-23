from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import mysql.connector
from config import MYSQL_CONFIG, TRADE_MODE
from fastapi.responses import PlainTextResponse
import os
from data.data_manager import get_connection, insert_balance_to_db, save_trade_record
from datetime import datetime
from trading.trade_status import get_trade_status, set_trade_status

app = FastAPI(title="DiceDux Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
LOG_FILE_PATH = "logs/ai_decisions.log"

def get_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)

@app.get("/", response_class=HTMLResponse)
def dashboard_view(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/trades")
def get_trades():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, symbol, action, entry_price, confidence, profit, timestamp, mode
            FROM trades ORDER BY timestamp DESC LIMIT 50
        """)
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn.is_connected():
            conn.close()

@app.get("/api/balances")
def get_balances():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT symbol, balance, updated_at FROM balance")
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn.is_connected():
            conn.close()

@app.get("/api/logs", response_class=PlainTextResponse)
def get_logs():
    if not os.path.exists(LOG_FILE_PATH):
        return "No logs available."
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return "".join(lines[-50:]) if lines else "No logs available."
    except Exception as e:
        return f"Error reading logs: {e}"

@app.get("/api/mode")
def get_mode():
    return {"mode": TRADE_MODE}

@app.post("/api/close_all_trades")
def close_all_trades():
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM open_trades")
        open_positions = cursor.fetchall()

        if not open_positions:
            return {"message": "❗ هیچ پوزیشن بازی وجود ندارد."}

        for position in open_positions:
            symbol = position["symbol"]
            entry_price = float(position["entry_price"])
            opened_at = position["opened_at"]

            cursor.execute("""
                SELECT close FROM candles
                WHERE symbol = %s
                ORDER BY timestamp DESC LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            if not row:
                continue

            exit_price = float(row["close"])
            profit = ((exit_price - entry_price) / entry_price) * 100

            cursor.execute("""
                INSERT INTO trades (symbol, action, entry_price, exit_price, confidence, timestamp, mode, profit)
                VALUES (%s, 'sell', %s, %s, %s, %s, %s, %s)
            """, (
                symbol,
                entry_price,
                exit_price,
                0.0,
                datetime.utcnow().isoformat(),
                "simulation",
                profit
            ))

            cursor.execute("DELETE FROM open_trades WHERE symbol = %s", (symbol,))

            cursor.execute("""
                INSERT INTO balance (symbol, balance, updated_at)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    balance = VALUES(balance),
                    updated_at = VALUES(updated_at)
            """, (symbol, 0.0, datetime.utcnow().isoformat()))

        conn.commit()
        return {"message": "✅ تمام معاملات باز بسته شدند."}

    except Exception as e:
        print("❌ خطا:", e)
        return {"message": "❌ خطا در بستن معاملات"}

    finally:
        if conn.is_connected():
            conn.close()

@app.get("/api/trade-status")
def get_status():
    return {"status": get_trade_status()}

@app.post("/api/trade-status/{status}")
def set_status(status: str):
    if status not in ["running", "paused"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    set_trade_status(status)
    return {"message": f"Trade status set to {status}"}

@app.get("/api/candles")
def get_candles(symbol: str):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT timestamp, open, high, low, close
            FROM candles
            WHERE symbol = %s
            ORDER BY timestamp ASC
            LIMIT 200
        """, (symbol,))
        return cursor.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn.is_connected():
            conn.close()

@app.get("/api/chart-data")
def get_chart_data(symbol: str = "BTCUSDT"):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT timestamp, open, high, low, close
            FROM candles
            WHERE symbol = %s
            ORDER BY timestamp ASC
            LIMIT 100
        """, (symbol,))
        candles = cursor.fetchall()

        cursor.execute("""
            SELECT action, entry_price, timestamp
            FROM trades
            WHERE symbol = %s
            ORDER BY timestamp ASC
        """, (symbol,))
        signals = cursor.fetchall()

        return {"candles": candles, "signals": signals}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn.is_connected():
            conn.close()

@app.get("/api/open_trades")
def get_open_trades():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT symbol, action, entry_price, created_at as timestamp,
                   tp_price, sl_price, tp_step, last_price,
            quantity, live_profit FROM positions
        """)
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn.is_connected():
            conn.close()

@app.get("/api/performance-chart")
def get_balance_trend():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT updated_at, balance
            FROM balance
            ORDER BY updated_at ASC
        """)

        rows = cursor.fetchall()
        return rows

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn.is_connected():
            conn.close()

@app.get("/api/backtest-results")
def get_backtest_results():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT symbol, total_trades, win_rate, total_profit, final_balance, executed_at
            FROM backtest_results
            ORDER BY executed_at DESC
            LIMIT 50
        """)
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn.is_connected():
            conn.close()

if __name__ == "__main__":
    uvicorn.run("dashboard:app", host="0.0.0.0", port=8080, reload=True)