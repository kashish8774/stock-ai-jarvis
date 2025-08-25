# main.py
"""
Stock AI Jarvis – Pro Backend
FastAPI service that provides:
- Real-time-ish quotes & technical indicators
- Smart BUY/SELL/HOLD signals with reasons and risk metrics
- Multi-symbol scanner
- Portfolio suggestions (rebalance, stops/targets)
- Options chain fetch + basic strategy builders (US tickers best supported)

NOTE:
- Yahoo Finance (yfinance) provides delayed data for most markets. For true real-time and execution,
  integrate a broker/data feed (e.g., Zerodha/Upstox/Dhan in India; Alpaca/Polygon in US) later.
- NSE options via yfinance can be limited; US options coverage is better.
"""

from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yfinance as yf
import pandas as pd
import numpy as np
import talib
import math
import threading
import time

app = FastAPI(title="Stock AI Jarvis – Pro", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Helpers
# ---------------------------

def normalize_symbol(symbol: str, market: Optional[str] = None) -> str:
    s = symbol.strip().upper()
    if market and market.upper() in ["IN", "IND", "INDIA"]:
        # If user passes plain NSE root ticker, append .NS
        if not s.endswith(".NS") and not s.endswith(".BO"):
            s = s + ".NS"
    return s


def fetch_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna().copy()
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    vol = df["Volume"].values if "Volume" in df.columns else np.zeros(len(df))

    # Trend / momentum
    df["SMA20"] = talib.SMA(close, timeperiod=20)
    df["SMA50"] = talib.SMA(close, timeperiod=50)
    df["SMA200"] = talib.SMA(close, timeperiod=200)
    df["RSI14"] = talib.RSI(close, timeperiod=14)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    # Volatility
    df["ATR14"] = talib.ATR(high, low, close, timeperiod=14)
    # Volume trend (OBV)
    try:
        df["OBV"] = talib.OBV(close, vol)
    except Exception:
        df["OBV"] = np.nan

    return df


def signal_engine(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate signal, score and reasons from latest row."""
    if df.empty or len(df) < 50:
        return {"signal": "HOLD", "score": 0.0, "reasons": ["Insufficient data"], "risk": None}

    row = df.iloc[-1]
    price = float(row["Close"]) if not math.isnan(row["Close"]) else None

    score = 0.0
    reasons = []

    # Trend filter
    if row["SMA50"] and row["SMA200"] and not math.isnan(row["SMA50"]) and not math.isnan(row["SMA200"]):
        if row["SMA50"] > row["SMA200"]:
            score += 1.5
            reasons.append("SMA50 > SMA200 (long-term uptrend)")
        else:
            score -= 1.5
            reasons.append("SMA50 < SMA200 (long-term downtrend)")

    # Short-term momentum
    if row["SMA20"] and row["SMA50"] and not math.isnan(row["SMA20"]) and not math.isnan(row["SMA50"]):
        if row["SMA20"] > row["SMA50"]:
            score += 1.0
            reasons.append("SMA20 > SMA50 (short-term uptrend)")
        else:
            score -= 0.5
            reasons.append("SMA20 < SMA50 (short-term weakness)")

    # RSI regime
    if row["RSI14"] and not math.isnan(row["RSI14"]):
        if row["RSI14"] < 30:
            score += 0.7
            reasons.append("RSI < 30 (oversold bounce potential)")
        elif row["RSI14"] > 70:
            score -= 0.7
            reasons.append("RSI > 70 (overbought risk)")

    # MACD trend confirmation
    if row["MACD"] and row["MACD_SIGNAL"] and not math.isnan(row["MACD"]) and not math.isnan(row["MACD_SIGNAL"]):
        if row["MACD"] > row["MACD_SIGNAL"]:
            score += 0.5
            reasons.append("MACD above signal (bullish momentum)")
        else:
            score -= 0.4
            reasons.append("MACD below signal (bearish momentum)")

    # Risk metrics from ATR
    atr = float(row["ATR14"]) if row.get("ATR14") is not None and not math.isnan(row["ATR14"]) else None
    stop_loss = target = None
    if atr and price:
        stop_loss = round(price - 1.5 * atr, 2)
        target = round(price + 3.0 * atr, 2)

    label = "HOLD"
    if score >= 1.5:
        label = "BUY"
    elif score <= -1.2:
        label = "SELL"

    return {
        "signal": label,
        "score": round(score, 2),
        "reasons": reasons,
        "risk": {
            "atr": round(atr, 2) if atr else None,
            "stop_loss": stop_loss,
            "target": target,
        },
    }


def suggest_position_size(capital: float, price: float, atr: Optional[float], target_portfolio_vol: float = 0.12) -> Dict[str, Any]:
    """Volatility-scaled position sizing. Very simple heuristic.
    target_portfolio_vol ~ annualized (e.g., 12%).
    """
    if price <= 0 or capital <= 0 or atr is None or atr <= 0:
        units = max(1, int(capital * 0.05 / max(price, 1)))
        return {"units": units, "capital_used": round(units * price, 2)}

    # daily vol proxy from ATR (~ATR/price)
    daily_vol = atr / price
    if daily_vol <= 0:
        daily_vol = 0.02

    # Convert target annual vol to daily (~ sqrt(252))
    target_daily_vol = target_portfolio_vol / math.sqrt(252)
    weight = min(0.15, max(0.01, target_daily_vol / daily_vol))  # cap 1–15%
    capital_alloc = capital * weight
    units = max(1, int(capital_alloc // price))
    return {"units": units, "capital_used": round(units * price, 2), "weight": round(weight, 4)}


# ---------------------------
# Models
# ---------------------------
class Holding(BaseModel):
    symbol: str
    qty: float
    avg_price: float
    market: Optional[str] = None  # "IN" or "US"

class PortfolioRequest(BaseModel):
    capital: float
    holdings: List[Holding]

class ScanRequest(BaseModel):
    symbols: List[str]
    market: Optional[str] = None

# ---------------------------
# Endpoints
# ---------------------------

@app.get("/")
def home():
    return {"message": "Welcome to Stock AI Jarvis Pro 🚀", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/quote")
def quote(symbol: str = Query(...), market: Optional[str] = Query(None)):
    sym = normalize_symbol(symbol, market)
    t = yf.Ticker(sym)
    info = t.fast_info if hasattr(t, "fast_info") else {}
    last_price = None
    try:
        last_price = float(info.get("last_price")) if info else None
    except Exception:
        pass

    return {
        "symbol": sym,
        "last_price": last_price,
        "currency": info.get("currency") if info else None,
        "exchange": info.get("exchange") if info else None,
    }


@app.get("/recommend")
def recommend(symbol: str = Query(...), market: Optional[str] = Query(None), period: str = Query("6mo"), interval: str = Query("1d")):
    sym = normalize_symbol(symbol, market)
    df = fetch_history(sym, period=period, interval=interval)
    if df.empty:
        return {"error": f"No data for {sym}"}
    df = compute_indicators(df)
    sig = signal_engine(df)
    last = df.iloc[-1]

    size = suggest_position_size(capital=100000.0, price=float(last["Close"]), atr=float(last.get("ATR14") or 0))

    return {
        "symbol": sym,
        "latest": {
            "price": round(float(last["Close"]), 2),
            "rsi14": round(float(last["RSI14"]), 2) if not math.isnan(last["RSI14"]) else None,
            "sma20": round(float(last["SMA20"]), 2) if not math.isnan(last["SMA20"]) else None,
            "sma50": round(float(last["SMA50"]), 2) if not math.isnan(last["SMA50"]) else None,
            "sma200": round(float(last["SMA200"]), 2) if not math.isnan(last["SMA200"]) else None,
            "atr14": round(float(last["ATR14"]), 2) if not math.isnan(last["ATR14"]) else None,
        },
        "signal": sig,
        "suggested_size_on_1L_capital": size,
    }


@app.post("/scan")
def scan(req: ScanRequest):
    results = []
    for s in req.symbols:
        sym = normalize_symbol(s, req.market)
        try:
            df = fetch_history(sym, period="6mo", interval="1d")
            if df.empty:
                results.append({"symbol": sym, "error": "no_data"})
                continue
            df = compute_indicators(df)
            sig = signal_engine(df)
            last = df.iloc[-1]
            results.append({
                "symbol": sym,
                "price": round(float(last["Close"]), 2),
                "score": sig["score"],
                "signal": sig["signal"],
                "reasons": sig["reasons"],
                "risk": sig["risk"],
            })
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})

    # Rank by score desc
    results_sorted = sorted(results, key=lambda x: x.get("score", -999), reverse=True)
    return {"count": len(results), "results": results_sorted}


@app.post("/portfolio/suggest")
def portfolio_suggest(req: PortfolioRequest):
    portfolio_value = 0.0
    enriched = []

    for h in req.holdings:
        sym = normalize_symbol(h.symbol, h.market)
        df = fetch_history(sym, period="6mo", interval="1d")
        if df.empty:
            enriched.append({"symbol": sym, "error": "no_data"})
            continue
        df = compute_indicators(df)
        sig = signal_engine(df)
        last = df.iloc[-1]
        mkt_price = float(last["Close"])
        value = mkt_price * h.qty
        portfolio_value += value

        # Basic action suggestion
        action = "HOLD"
        if sig["signal"] == "SELL":
            action = "TRIM" if h.qty > 0 else "HOLD"
        elif sig["signal"] == "BUY":
            action = "ADD" if h.qty > 0 else "BUY"

        enriched.append({
            "symbol": sym,
            "qty": h.qty,
            "avg_price": h.avg_price,
            "market_price": round(mkt_price, 2),
            "unrealized_pnl": round((mkt_price - h.avg_price) * h.qty, 2),
            "signal": sig,
            "suggested_action": action,
        })

    # Position sizing suggestions for new capital deployment (if any)
    per_symbol_allocations = []
    if portfolio_value > 0:
        for item in enriched:
            if "market_price" in item and item.get("signal", {}).get("risk", {}).get("atr"):
                size = suggest_position_size(
                    capital=req.capital,
                    price=item["market_price"],
                    atr=item["signal"]["risk"]["atr"],
                )
                per_symbol_allocations.append({
                    "symbol": item["symbol"],
                    "units": size.get("units"),
                    "capital_used": size.get("capital_used"),
                    "weight": size.get("weight"),
                })

    return {
        "portfolio_value": round(portfolio_value, 2),
        "positions": enriched,
        "new_capital_sizing": per_symbol_allocations,
    }


@app.get("/options/expiries")
def options_expiries(symbol: str = Query(...), market: Optional[str] = Query(None)):
    sym = normalize_symbol(symbol, market)
    t = yf.Ticker(sym)
    try:
        exps = t.options
        return {"symbol": sym, "expiries": exps}
    except Exception as e:
        return {"symbol": sym, "error": str(e)}


@app.get("/options/chain")
def options_chain(symbol: str = Query(...), expiry: str = Query(...), market: Optional[str] = Query(None), strikes_around: int = Query(5)):
    sym = normalize_symbol(symbol, market)
    t = yf.Ticker(sym)
    try:
        chain = t.option_chain(expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        # Trim around ATM
        spot = t.fast_info.get("last_price") if hasattr(t, "fast_info") else None
        if spot:
            calls["dist"] = (calls["strike"] - float(spot)).abs()
            puts["dist"] = (puts["strike"] - float(spot)).abs()
            calls = calls.sort_values("dist").head(strikes_around)
            puts = puts.sort_values("dist").head(strikes_around)
            calls = calls.drop(columns=["dist"]) if "dist" in calls.columns else calls
            puts = puts.drop(columns=["dist"]) if "dist" in puts.columns else puts
        return {
            "symbol": sym,
            "expiry": expiry,
            "spot": float(spot) if spot else None,
            "calls": calls.to_dict(orient="records"),
            "puts": puts.to_dict(orient="records"),
        }
    except Exception as e:
        return {"symbol": sym, "error": str(e)}


@app.get("/options/strategy/straddle")
def options_straddle(symbol: str = Query(...), expiry: str = Query(...), market: Optional[str] = Query(None)):
    sym = normalize_symbol(symbol, market)
    t = yf.Ticker(sym)
    try:
        spot = t.fast_info.get("last_price") if hasattr(t, "fast_info") else None
        chain = t.option_chain(expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        # Pick nearest strike to ATM
        if spot is None or calls.empty or puts.empty:
            return {"error": "Spot or chain unavailable"}
        calls["dist"] = (calls["strike"] - float(spot)).abs()
        puts["dist"] = (puts["strike"] - float(spot)).abs()
        call_atm = calls.sort_values("dist").iloc[0]
        put_atm = puts.sort_values("dist").iloc[0]
        total_premium = float(call_atm.get("lastPrice", np.nan)) + float(put_atm.get("lastPrice", np.nan))
        breakeven_high = float(spot) + total_premium
        breakeven_low = float(spot) - total_premium
        return {
            "symbol": sym,
            "expiry": expiry,
            "spot": float(spot),
            "call": {
                "strike": float(call_atm["strike"]),
                "lastPrice": float(call_atm.get("lastPrice", np.nan)),
            },
            "put": {
                "strike": float(put_atm["strike"]),
                "lastPrice": float(put_atm.get("lastPrice", np.nan)),
            },
            "total_premium": round(total_premium, 2),
            "breakeven_low": round(breakeven_low, 2),
            "breakeven_high": round(breakeven_high, 2),
            "note": "For live Greeks/IV and Indian options depth, use a dedicated options API later.",
        }
    except Exception as e:
        return {"symbol": sym, "error": str(e)}


# ---------------------------
# Background cache (optional): simple example that refreshes a small symbol set periodically.
# NOTE: Render's free tier may kill background threads; use a Worker service for reliability.
# ---------------------------
_cached_signals: Dict[str, Dict[str, Any]] = {}


def background_scanner(symbols: List[str]):
    while True:
        try:
            for s in symbols:
                df = fetch_history(s, period="6mo", interval="1d")
                if df.empty:
                    continue
                df = compute_indicators(df)
                _cached_signals[s] = signal_engine(df)
        except Exception:
            pass
        time.sleep(900)  # refresh every 15 min


@app.get("/cache/signals")
def cached_signals():
    return _cached_signals


# Start a lightweight scanner for a tiny default watchlist (US + IN examples)
def _start_bg():
    watch = ["AAPL", "NVDA", "MSFT", "RELIANCE.NS", "TCS.NS"]
    th = threading.Thread(target=background_scanner, args=(watch,), daemon=True)
    th.start()


_start_bg()
