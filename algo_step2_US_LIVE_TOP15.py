# algo_step2_US_LIVE_TOP15.py
# STEP-2 LIVE (US) signal generator + Orders (AL/TUT/SAT)
# Yahoo (yfinance) robust: chunk + retry/backoff + safe no-trade fallback
# Outputs (US-names):
#   live_signal_us.csv, orders_us.csv, orders_us.txt, equity_curve_us.csv, report_us.csv

import warnings
warnings.filterwarnings("ignore")

import datetime as dt
import time
import random
from typing import Optional, List, Dict, Set

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor


# =========================
# UNIVERSE: DOW-ish (embedded)
# =========================
US_TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA",
    "JPM","V","MA","UNH","JNJ","PG","XOM","CVX",
    "HD","KO","PEP","MRK","LLY","ABBV","AVGO",
    "COST","WMT","MCD","DIS","CSCO","IBM","BA","NKE"
]
TICKERS_ALL = sorted(list(dict.fromkeys(US_TICKERS)))

# Market proxy
MARKET_PROXY = "SPY"


# =========================
# PARAMS
# =========================
MAX_LOOKBACK_YEARS = 8

HORIZON = 5
TRAIN_DAYS = 252 * 3
TEST_DAYS  = 63
STEP_DAYS  = 63

TOP_N = 15
POS_CAP = 0.10
GROSS_CAP = 1.00
LAMBDA_VOL = 0.35

COST_BPS = 2
TURNOVER_CAP_DAILY = 0.40
POS_EMA_ALPHA = 0.35

VOL_TARGET_ANNUAL = 0.10
PORT_VOL_LOOKBACK = 60

USE_LIQ_FILTER = False
MIN_AVG_DV20 = 20_000_000

VOL_WINDOW = 252
VOL_MIN_PERIODS = 60
VOL_Q_GRID = [0.55, 0.65, 0.75]

INIT_CAPITAL_USD = 10_000
FRESH_MAX_AGE_DAYS = 2

COOLDOWN_DAYS = 5

DL_CHUNK_SIZE = 10
DL_MAX_RETRIES = 6


# OUTPUT NAMES (US)
ORDERS_CSV = "orders_us.csv"
ORDERS_TXT = "orders_us.txt"
LIVE_SIGNAL_CSV = "live_signal_us.csv"
EQUITY_CSV = "equity_curve_us.csv"
REPORT_CSV = "report_us.csv"


# =========================
# UTIL
# =========================
def today_utc() -> pd.Timestamp:
    return pd.Timestamp(dt.date.today()).normalize()

def safe_print(msg: str) -> None:
    print(msg, flush=True)

def last_date_from_raw(raw: pd.DataFrame) -> Optional[pd.Timestamp]:
    try:
        if raw is None or raw.empty:
            return None
        idx = pd.to_datetime(pd.Index(raw.index), errors="coerce").dropna()
        if len(idx) == 0:
            return None
        return idx.max().normalize()
    except Exception:
        return None


def write_no_trade_outputs(reason: str, data_date: Optional[pd.Timestamp] = None) -> None:
    t0 = today_utc()
    if isinstance(data_date, pd.Timestamp):
        dd_str = str(pd.to_datetime(data_date).normalize().date())
    else:
        dd_str = "unknown"

    orders_df = pd.DataFrame([{
        "date": str(t0.date()),
        "side": "TUT",
        "ticker": "N/A",
        "target_weight_%": 0.0,
        "target_alloc_USD": 0,
        "note": reason,
        "data_date": dd_str,
        "fresh": 0,
        "fresh_note": f"⛔ {reason} → NO TRADE TODAY"
    }])
    orders_df.to_csv(ORDERS_CSV, index=False)

    with open(ORDERS_TXT, "w", encoding="utf-8") as f:
        f.write(reason + "\n")

    pd.DataFrame({"date":[t0], "equity_scaled":[1.0]}).to_csv(EQUITY_CSV, index=False)
    pd.DataFrame([{"variant":"NO_TRADE","return_%":0.0,"sharpe":0.0,"maxDD_%":0.0,"days":0}]).to_csv(REPORT_CSV, index=False)
    pd.DataFrame(columns=["date","ticker","weight_%","alloc_USD"]).to_csv(LIVE_SIGNAL_CSV, index=False)

    safe_print("✅ US Fallback outputs written (no-trade).")


# =========================
# DOWNLOAD
# =========================
def yf_download_chunked(tickers: List[str], start: str, end: Optional[str], chunk_size: int, max_retries: int) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
    parts: List[pd.DataFrame] = []

    for ci, ch in enumerate(chunks, start=1):
        time.sleep(1.0 + random.random() * 1.0)
        ok = False
        last_err = None

        for r in range(max_retries):
            try:
                df = yf.download(
                    tickers=ch,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    group_by="column",
                    threads=False,
                    progress=False
                )
                if df is not None and not df.empty:
                    parts.append(df)
                    ok = True
                    break
                last_err = "empty"
            except Exception as e:
                last_err = str(e)

            wait = min(120, (2 ** r)) + random.random() * 3.0
            safe_print(f"⚠️ US chunk {ci}/{len(chunks)} retry {r+1}/{max_retries} wait {wait:.1f}s err={last_err}")
            time.sleep(wait)

        if not ok:
            safe_print(f"❌ US chunk {ci}/{len(chunks)} failed err={last_err}")

    if not parts:
        return pd.DataFrame()

    try:
        return pd.concat(parts, axis=1)
    except Exception:
        raw = parts[0]
        for p in parts[1:]:
            raw = pd.concat([raw, p], axis=1)
        return raw


def load_market_series(start: str, end: Optional[str]) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(MARKET_PROXY, start=start, end=end, auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.reset_index().rename(columns={df.reset_index().columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.rename(columns={c: str(c).lower() for c in df.columns})
        if "close" not in df.columns and "adj close" in df.columns:
            df["close"] = df["adj close"]
        if "close" not in df.columns:
            return None
        df["mkt_close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date","mkt_close"]).sort_values("date")
        safe_print(f"Market proxy: {MARKET_PROXY}")
        return df[["date","mkt_close"]].copy()
    except Exception:
        return None


# =========================
# PANEL
# =========================
def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    if "close" not in df.columns and "adj close" in df.columns:
        df["close"] = df["adj close"]

    needed = ["open","high","low","close","volume"]
    if not all(c in df.columns for c in needed):
        return None

    df = df[needed].copy()
    df = df.reset_index().rename(columns={df.reset_index().columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = ticker

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date","close"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if df.empty:
        return None
    return df[["date","ticker","open","high","low","close","volume"]]


def extract_one_ticker(raw: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        for level in [1, 0]:
            try:
                sub = raw.xs(ticker, axis=1, level=level)
                out = _normalize_ohlcv(sub, ticker)
                if out is not None:
                    return out
            except Exception:
                continue
        return None
    return _normalize_ohlcv(raw.copy(), ticker)


def build_panel(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    frames = []
    for t in tickers:
        sub = extract_one_ticker(raw, t)
        if sub is not None:
            frames.append(sub)
    if not frames:
        raise ValueError("No OHLCV data for any US ticker.")
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["ticker","date"]).reset_index(drop=True)


# =========================
# FEATURES
# =========================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))


FEATURES = [
    "ret_1","ret_2","ret_5","ret_10","ret_20",
    "vol_10","vol_20","vol_chg",
    "ma_diff","rsi_14","z_50",
    "hl_spread",
    "mkt_ret_1","mkt_ret_5","mkt_ret_20","trend_flag",
    "rel_5","rel_20",
    "beta_60",
    "cs_rank_rel20","cs_rank_ret5","cs_rank_rsi","cs_rank_z50",
    "dv20",
]


def add_features(panel: pd.DataFrame, mkt: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["date","ticker","close"]).copy()

    g = out.groupby("ticker", group_keys=False)

    out["ret_1"]  = g["close"].apply(lambda s: np.log(s / s.shift(1)))
    out["ret_2"]  = g["close"].apply(lambda s: np.log(s / s.shift(2)))
    out["ret_5"]  = g["close"].apply(lambda s: np.log(s / s.shift(5)))
    out["ret_10"] = g["close"].apply(lambda s: np.log(s / s.shift(10)))
    out["ret_20"] = g["close"].apply(lambda s: np.log(s / s.shift(20)))

    out["vol_10"] = g["ret_1"].apply(lambda s: s.rolling(10).std())
    out["vol_20"] = g["ret_1"].apply(lambda s: s.rolling(20).std())
    out["vol_chg"] = g["vol_20"].apply(lambda s: s / (s.shift(5) + 1e-12) - 1.0)

    ma10 = g["close"].apply(lambda s: s.rolling(10).mean())
    ma20 = g["close"].apply(lambda s: s.rolling(20).mean())
    out["ma_diff"] = (ma10 - ma20) / (ma20 + 1e-12)

    out["rsi_14"] = g["close"].apply(lambda s: rsi(s, 14))

    m50 = g["close"].apply(lambda s: s.rolling(50).mean())
    sd50 = g["close"].apply(lambda s: s.rolling(50).std())
    out["z_50"] = (out["close"] - m50) / (sd50 + 1e-12)

    out["hl_spread"] = (out["high"] - out["low"]) / (out["close"] + 1e-12)

    out["dv"] = (out["close"].abs() * out["volume"]).astype(float)
    out["dv20"] = g["dv"].apply(lambda s: s.rolling(20).mean())

    out["y_fwd"] = g["close"].apply(lambda s: np.log(s.shift(-HORIZON) / s))

    if mkt is not None:
        mdf = mkt.copy()
        mdf["date"] = pd.to_datetime(mdf["date"], errors="coerce").dt.normalize()
        mdf = mdf.sort_values("date").dropna(subset=["mkt_close"])
        mdf["mkt_ret_1"]  = np.log(mdf["mkt_close"] / mdf["mkt_close"].shift(1))
        mdf["mkt_ret_5"]  = np.log(mdf["mkt_close"] / mdf["mkt_close"].shift(5))
        mdf["mkt_ret_20"] = np.log(mdf["mkt_close"] / mdf["mkt_close"].shift(20))
        mdf["trend_flag"] = (mdf["mkt_close"].rolling(50).mean() > mdf["mkt_close"].rolling(200).mean()).astype(int)
        out = out.merge(mdf[["date","mkt_ret_1","mkt_ret_5","mkt_ret_20","trend_flag"]], on="date", how="left")
    else:
        uni = out.groupby("date")["ret_1"].mean().rename("mkt_ret_1").to_frame()
        uni["mkt_ret_5"]  = uni["mkt_ret_1"].rolling(5).sum()
        uni["mkt_ret_20"] = uni["mkt_ret_1"].rolling(20).sum()
        uni["trend_flag"] = (uni["mkt_ret_20"] > 0).astype(int)
        out = out.merge(uni.reset_index(), on="date", how="left")

    out["rel_5"]  = out["ret_5"]  - out["mkt_ret_5"]
    out["rel_20"] = out["ret_20"] - out["mkt_ret_20"]

    out["cs_rank_rel20"] = out.groupby("date")["rel_20"].rank(pct=True)
    out["cs_rank_ret5"]  = out.groupby("date")["ret_5"].rank(pct=True)
    out["cs_rank_rsi"]   = out.groupby("date")["rsi_14"].rank(pct=True)
    out["cs_rank_z50"]   = out.groupby("date")["z_50"].rank(pct=True)

    def rolling_beta(df_t: pd.DataFrame) -> pd.Series:
        x = df_t["mkt_ret_1"]
        y = df_t["ret_1"]
        cov = (x*y).rolling(60).mean() - x.rolling(60).mean()*y.rolling(60).mean()
        var = x.rolling(60).var()
        return cov / (var + 1e-12)

    out = out.sort_values(["ticker","date"]).reset_index(drop=True)
    out["beta_60"] = out.groupby("ticker", group_keys=False).apply(rolling_beta)
    return out


# =========================
# WALK + BACKTEST
# =========================
def walk_forward_dates(dates: np.ndarray, train_days: int, test_days: int, step_days: int):
    i, n = 0, len(dates)
    while True:
        tr_end = i + train_days
        te_end = tr_end + test_days
        if te_end > n:
            break
        yield dates[i:tr_end], dates[tr_end:te_end]
        i += step_days

def compute_vol_thr_from_context(context_df: pd.DataFrame, vol_q: float) -> pd.Series:
    ctx = context_df.copy()
    ctx["date"] = pd.to_datetime(ctx["date"], errors="coerce").dt.normalize()
    ctx["vol_20"] = pd.to_numeric(ctx["vol_20"], errors="coerce")
    daily_vol = ctx.groupby("date")["vol_20"].mean()
    thr = daily_vol.rolling(VOL_WINDOW, min_periods=VOL_MIN_PERIODS).quantile(vol_q).shift(1)
    thr.index = pd.to_datetime(thr.index).normalize()
    thr.name = "vol_thr"
    return thr

def portfolio_vol_target_scale(daily_pnl: pd.Series, target_annual: float, lookback: int) -> pd.Series:
    target_daily = target_annual / np.sqrt(252)
    vol = daily_pnl.rolling(lookback, min_periods=max(20, lookback//3)).std()
    scale = (target_daily / (vol + 1e-12)).clip(upper=3.0)
    return scale.shift(1).fillna(1.0)

def ema_by_ticker(df: pd.DataFrame, col: str, alpha: float) -> pd.Series:
    out = np.empty(len(df), dtype=float)
    last_t = None
    last = 0.0
    tick = df["ticker"].to_numpy()
    val = df[col].to_numpy(float)
    for i, (t, x) in enumerate(zip(tick, val)):
        if t != last_t:
            last_t = t
            last = x
        else:
            last = alpha * x + (1 - alpha) * last
        out[i] = last
    return pd.Series(out, index=df.index)

def backtest(preds: pd.DataFrame, vol_thr: pd.Series, top_n: int):
    out = preds.copy().reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    for c in ["ret_1","vol_20","dv20","mu_hat"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["ret_1"] = out["ret_1"].fillna(0.0)
    out["mu_hat"] = out["mu_hat"].fillna(0.0)

    out = out.merge(vol_thr.to_frame(), left_on="date", right_index=True, how="left")

    trade_ok = (out["vol_20"] <= out["vol_thr"]).fillna(False)
    liq_ok = (out["dv20"] >= MIN_AVG_DV20).fillna(False) if USE_LIQ_FILTER else True
    ok = trade_ok & liq_ok

    out["score"] = out["mu_hat"] - (LAMBDA_VOL * out["vol_20"].fillna(out["vol_20"].median()))
    out["score_ok"] = out["score"].where(ok, np.nan)

    out["rank_day"] = out.groupby("date")["score_ok"].rank(method="first", ascending=False)
    selected = (out["rank_day"] <= top_n) & ok

    out["w_raw"] = 0.0
    out.loc[selected, "w_raw"] = 1.0

    sum_w = out.groupby("date")["w_raw"].transform("sum").replace(0, np.nan)
    out["w_raw"] = (out["w_raw"] / sum_w).fillna(0.0) * GROSS_CAP
    out["w_raw"] = out["w_raw"].clip(0.0, POS_CAP)

    out = out.sort_values(["ticker","date"]).reset_index(drop=True)
    out["w"] = ema_by_ticker(out, "w_raw", POS_EMA_ALPHA)

    gross = out.groupby("date")["w"].transform("sum").replace(0, np.nan)
    out["w"] = (out["w"] / gross).fillna(0.0) * GROSS_CAP
    out["w"] = out["w"].clip(0.0, POS_CAP)

    out["w_lag"] = out.groupby("ticker")["w"].shift(1).fillna(0.0)
    out["turnover"] = out.groupby("ticker")["w"].diff().abs().fillna(0.0)

    if TURNOVER_CAP_DAILY is not None:
        day_turn = out.groupby("date")["turnover"].mean()
        tscale = (TURNOVER_CAP_DAILY / (day_turn + 1e-12)).clip(upper=1.0)
        out = out.merge(tscale.rename("turn_scale"), left_on="date", right_index=True, how="left")
        out["w"] = out["w"] * out["turn_scale"].fillna(1.0)
        out["w_lag"] = out.groupby("ticker")["w"].shift(1).fillna(0.0)
        out["turnover"] = out.groupby("ticker")["w"].diff().abs().fillna(0.0)

    cost = (COST_BPS / 1e4) * out["turnover"]
    out["pnl"] = out["w_lag"] * out["ret_1"] - cost

    daily = out.groupby("date")["pnl"].sum().to_frame("pnl")
    scale = portfolio_vol_target_scale(daily["pnl"], VOL_TARGET_ANNUAL, PORT_VOL_LOOKBACK)
    daily["scale"] = scale
    daily["pnl_scaled"] = daily["pnl"] * daily["scale"].fillna(1.0)

    daily["equity"] = np.exp(daily["pnl"].cumsum())
    daily["equity_scaled"] = np.exp(daily["pnl_scaled"].cumsum())

    out = out.merge(scale.rename("port_scale"), left_on="date", right_index=True, how="left")
    out["w_scaled"] = out["w"] * out["port_scale"].fillna(1.0)

    return out, daily

def sharpe(x: pd.Series) -> float:
    r = x.to_numpy(float)
    return float((r.mean() / (r.std() + 1e-12)) * np.sqrt(252))

def max_dd(eq: pd.Series) -> float:
    peak = eq.cummax()
    return float((eq / peak - 1.0).min())


def main():
    t0 = today_utc()
    start_dt = (t0 - pd.Timedelta(days=365 * MAX_LOOKBACK_YEARS)).date()
    START = str(start_dt)
    END = None

    safe_print("1) Download US universe (Yahoo)...")
    raw = yf_download_chunked(TICKERS_ALL, START, END, DL_CHUNK_SIZE, DL_MAX_RETRIES)
    raw_last = last_date_from_raw(raw)
    safe_print(f"DEBUG raw_last: {raw_last}")

    if raw is None or raw.empty or raw_last is None:
        write_no_trade_outputs("NO DATA (Yahoo rate-limit / access)", data_date=raw_last)
        return

    age_days = int((t0 - raw_last).days)
    if age_days > FRESH_MAX_AGE_DAYS:
        write_no_trade_outputs(f"Data not fresh ({age_days} days old)", data_date=raw_last)
        return

    mkt = load_market_series(START, END)

    safe_print("2) Panel + features...")
    try:
        panel = build_panel(raw, TICKERS_ALL)
    except Exception as e:
        write_no_trade_outputs(f"PANEL FAILED ({type(e).__name__})", data_date=raw_last)
        return

    panel = add_features(panel, mkt)
    need = ["date","ticker","ret_1","vol_20","dv20","y_fwd"] + FEATURES
    panel = panel.dropna(subset=need).sort_values(["date","ticker"]).reset_index(drop=True)

    dates = np.array(sorted(panel["date"].unique()))
    if len(dates) < (TRAIN_DAYS + TEST_DAYS + 50):
        write_no_trade_outputs(f"NOT ENOUGH DAYS ({len(dates)})", data_date=raw_last)
        return

    safe_print("3) Walk-forward...")
    trades_chunks = []
    daily_chunks = []

    for fold, (train_dates, test_dates) in enumerate(walk_forward_dates(dates, TRAIN_DAYS, TEST_DAYS, STEP_DAYS), start=1):
        train = panel[panel["date"].isin(train_dates)].copy()
        test  = panel[panel["date"].isin(test_dates)].copy()
        if train.empty or test.empty:
            continue

        tr_dates = np.array(sorted(train["date"].unique()))
        split = int(len(tr_dates) * 0.8)
        trd, vad = tr_dates[:split], tr_dates[split:]
        tr_df = train[train["date"].isin(trd)].copy()
        va_df = train[train["date"].isin(vad)].copy()
        if tr_df.empty or va_df.empty:
            continue

        model = HistGradientBoostingRegressor(learning_rate=0.05, max_depth=6, max_iter=800, random_state=42)
        model.fit(tr_df[FEATURES], tr_df["y_fwd"].astype(float))

        va_pred = va_df[["date","ticker","ret_1","vol_20","dv20"]].copy()
        va_pred["mu_hat"] = model.predict(va_df[FEATURES])

        best = (-1e18, None)
        for vol_q in VOL_Q_GRID:
            thr_va = compute_vol_thr_from_context(pd.concat([tr_df[["date","vol_20"]], va_pred[["date","vol_20"]]]), vol_q)
            _, d_va = backtest(va_pred, thr_va, TOP_N)
            s = sharpe(d_va["pnl_scaled"])
            if s > best[0]:
                best = (s, vol_q)
        vol_q_best = best[1]
        if vol_q_best is None:
            continue

        model.fit(train[FEATURES], train["y_fwd"].astype(float))

        te_pred = test[["date","ticker","ret_1","vol_20","dv20"]].copy()
        te_pred["mu_hat"] = model.predict(test[FEATURES])

        thr_te = compute_vol_thr_from_context(pd.concat([train[["date","vol_20"]], te_pred[["date","vol_20"]]]), vol_q_best)
        bt, d = backtest(te_pred, thr_te, TOP_N)
        bt["fold"] = fold

        trades_chunks.append(bt)
        daily_chunks.append(d)

    if not trades_chunks or not daily_chunks:
        write_no_trade_outputs("MODEL PRODUCED NO OUTPUT", data_date=raw_last)
        return

    trades = pd.concat(trades_chunks, ignore_index=True)

    all_daily = pd.concat(daily_chunks)
    all_daily = all_daily.reset_index().rename(columns={"index":"date"}) if "date" not in all_daily.columns else all_daily
    all_daily["date"] = pd.to_datetime(all_daily["date"], errors="coerce").dt.normalize()

    daily = all_daily.groupby("date")[["pnl","pnl_scaled","scale"]].mean().sort_index()
    daily["equity"] = np.exp(daily["pnl"].cumsum())
    daily["equity_scaled"] = np.exp(daily["pnl_scaled"].cumsum())
    daily.to_csv(EQUITY_CSV, index=True)

    rep = pd.DataFrame([{
        "variant":"VOL_TARGET",
        "end_capital_USD": INIT_CAPITAL_USD * float(daily["equity_scaled"].iloc[-1]),
        "pnl_USD": INIT_CAPITAL_USD * float(daily["equity_scaled"].iloc[-1]) - INIT_CAPITAL_USD,
        "return_%": (float(daily["equity_scaled"].iloc[-1]) - 1.0) * 100,
        "sharpe": sharpe(daily["pnl_scaled"]),
        "maxDD_%": max_dd(daily["equity_scaled"]) * 100,
        "days": len(daily)
    }])
    rep.to_csv(REPORT_CSV, index=False)

    last_date = pd.to_datetime(trades["date"].dropna().max()).normalize()
    if pd.isna(last_date):
        write_no_trade_outputs("NO SIGNAL DATE", data_date=raw_last)
        return

    data_age_days = int((t0 - last_date).days)
    data_is_fresh = (data_age_days <= FRESH_MAX_AGE_DAYS)
    freshness_note = "✅ FRESH" if data_is_fresh else f"⚠️ STALE ({data_age_days}d)"

    if not data_is_fresh:
        write_no_trade_outputs(f"Data not fresh ({data_age_days} days old)", data_date=last_date)
        return

    # LIVE signal
    tdf = trades[trades["date"] == last_date].copy()
    tdf["w_final"] = pd.to_numeric(tdf["w_scaled"], errors="coerce").fillna(0.0)
    tdf = tdf[tdf["w_final"] > 0].sort_values("w_final", ascending=False)

    tdf["alloc_USD"] = (tdf["w_final"] * INIT_CAPITAL_USD).round(0).astype(int)
    tdf["weight_%"] = (tdf["w_final"] * 100).round(3)

    live = tdf[["date","ticker","weight_%","alloc_USD"]].reset_index(drop=True)
    live.to_csv(LIVE_SIGNAL_CSV, index=False)

    # ORDERS
    all_dates = sorted(pd.to_datetime(trades["date"].dropna()).dt.normalize().unique())
    prev_date = all_dates[-2] if len(all_dates) >= 2 else None
    today_date = all_dates[-1]

    MIN_TRADE_PCT = 0.20

    today_df = trades[trades["date"] == today_date].copy()
    today_df["w_final"] = pd.to_numeric(today_df["w_scaled"], errors="coerce").fillna(0.0)
    today_df = today_df[today_df["w_final"] > 0].sort_values("w_final", ascending=False)

    today_top = today_df.head(TOP_N).copy()
    today_top = today_top[(today_top["w_final"] * 100) >= MIN_TRADE_PCT].copy()
    today_top["alloc_USD"] = (today_top["w_final"] * INIT_CAPITAL_USD).round(0).astype(int)
    today_top["weight_%"] = (today_top["w_final"] * 100).round(3)
    today_set = set(today_top["ticker"].tolist())

    if prev_date is not None:
        prev_df = trades[trades["date"] == prev_date].copy()
        prev_df["w_prev"] = pd.to_numeric(prev_df["w_scaled"], errors="coerce").fillna(0.0)
        prev_df = prev_df[prev_df["w_prev"] > 0].sort_values("w_prev", ascending=False)

        prev_top = prev_df.head(TOP_N).copy()
        prev_top = prev_top[(prev_top["w_prev"] * 100) >= MIN_TRADE_PCT].copy()
        prev_set = set(prev_top["ticker"].tolist())
    else:
        prev_set = set()

    to_buy  = today_set - prev_set
    to_sell = prev_set - today_set

    # cooldown
    recent_dates = all_dates[-(COOLDOWN_DAYS + 3):]
    top_sets: Dict[pd.Timestamp, Set[str]] = {}
    for d0 in recent_dates:
        df_d = trades[trades["date"] == d0].copy()
        df_d["w_d"] = pd.to_numeric(df_d["w_scaled"], errors="coerce").fillna(0.0)
        df_d = df_d[df_d["w_d"] > 0].sort_values("w_d", ascending=False).head(TOP_N)
        top_sets[d0] = set(df_d["ticker"].tolist())

    cooldown_block: Set[str] = set()
    for i in range(1, len(recent_dates)):
        cooldown_block |= (top_sets.get(recent_dates[i-1], set()) - top_sets.get(recent_dates[i], set()))
    to_buy = {t for t in to_buy if t not in cooldown_block}

    orders = []
    for _, r in today_top.iterrows():
        t = r["ticker"]
        side = "AL" if t in to_buy else "TUT"
        orders.append({
            "date": str(pd.to_datetime(today_date).date()),
            "side": side,
            "ticker": t,
            "target_weight_%": float(r["weight_%"]),
            "target_alloc_USD": int(r["alloc_USD"]),
            "note": "T+1 open / first liquid",
            "data_date": str(last_date.date()),
            "fresh": 1,
            "fresh_note": freshness_note
        })

    for t in sorted(list(to_sell)):
        orders.append({
            "date": str(pd.to_datetime(today_date).date()),
            "side": "SAT",
            "ticker": t,
            "target_weight_%": 0.0,
            "target_alloc_USD": 0,
            "note": "Dropped from list → T+1 open",
            "data_date": str(last_date.date()),
            "fresh": 1,
            "fresh_note": freshness_note
        })

    orders_df = pd.DataFrame(orders).drop_duplicates(subset=["date","side","ticker"], keep="first").reset_index(drop=True)
    orders_df.to_csv(ORDERS_CSV, index=False)

    with open(ORDERS_TXT, "w", encoding="utf-8") as f:
        f.write(f"ORDERS (US TOP{TOP_N})\n")
        f.write(f"Signal date: {pd.to_datetime(today_date).date()} | Execute: next session open\n")
        f.write(f"Data date: {last_date.date()} | {freshness_note}\n\n")
        f.write(orders_df.to_string(index=False))

    safe_print("✅ US Outputs written.")
    safe_print(f"Signal date: {pd.to_datetime(today_date).date()} | Data date: {last_date.date()} | {freshness_note}")


if __name__ == "__main__":
    main()
