from fastapi import FastAPI
import yfinance as yf
import talib
import numpy as np

app = FastAPI()

@app.get("/signals/{symbol}")
def get_signals(symbol: str):
    data = yf.download(symbol, period="6mo", interval="1d")
    close = data["Close"].values

    rsi = talib.RSI(close, timeperiod=14)
    ma50 = talib.SMA(close, timeperiod=50)
    ma200 = talib.SMA(close, timeperiod=200)

    signal = "BUY" if ma50[-1] > ma200[-1] and rsi[-1] < 70 else "SELL"

    return {
        "symbol": symbol,
        "latest_price": float(close[-1]),
        "signal": signal,
        "rsi": float(rsi[-1]),
        "ma50": float(ma50[-1]),
        "ma200": float(ma200[-1])
    }
