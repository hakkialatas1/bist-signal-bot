# algo_step2_LIVE_TOP15.py
# BIST100 LIVE stabilized: Yahoo rate-limit resistant + parquet cache + tz-safe + LIVE signal on latest feature date
# Outputs:
#  - orders_today.csv
#  - live_signal_today.csv
#  - equity_curve_live.csv
#  - report_live.csv
#
# NOTE: This is a research/demo tool, not investment advice.

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor

# =========================
# UNIVERSE (embedded list)
# =========================
BIST100_CODES = [
    "AKBNK","ALARK","ARCLK","ASELS","BIMAS","BRSAN","CCOLA","DOHOL","ECILC","EGEEN",
    "EKGYO","ENJSA","ENKAI","EREGL","FROTO","GARAN","GUBRF","HEKTS","ISCTR","KCHOL",
    "KOZAA","KOZAL","KRDMD","MGROS","ODAS","OTKAR","PETKM","PGSUS","SAHOL","SASA",
    "SISE","SKBNK","SMRTG","SOKM","TAVHL","TCELL","THYAO","TKFEN","TOASO","TRGYO",
    "TSKB","TTKOM","TUPRS","ULKER","VAKBN","VESBE","YKBNK","ZOREN",
    "AEFES","ANHYT","ASTOR","BAGFS","BERA","BOBET","BRYAT","CANTE","CIMSA","ENERY",
    "GENIL","GESAN","GWIND","KCAER","KONTR","MAVI","NTHOL","OYAKC","QUAGR",
    "SELEC","TTRAK","TUKAS","YATAS"
]
TICKERS_ALL = sorted(list(dict.fromkeys([c + ".IS" for c in BIST100_CODES])))

# =========================
# PARAMS
# =========================
START = "2016-01-01"
END = None

HORIZON = 5
TRAIN_DAYS = 252 * 3
TEST_DAYS  = 63
STEP_DAYS  = 63

TOP_N = 15
POS_CAP = 0.08
GROSS_CAP = 1.00
LAMBDA_VOL = 0.35

COST_BPS = 8
TURNOVER_CAP_DAILY = 0.40
POS_EMA_ALPHA = 0.35

VOL_TARGET_ANNUAL = 0.10
PORT_VOL_LOOKBACK = 60

USE_LIQ_FILTER = True
MIN_AVG_DV20 = 5_000_000

VOL_WINDOW = 252
VOL_MIN_PERIODS = 60
VOL_Q_GRID = [0.55, 0.65, 0.75]

INIT_CAPITAL_TL = 100_000
MARKET_CANDIDATES = ["XU100.IS", "^XU100"]

# freshness / kill-switch
MAX_STALENESS_DAYS = 2
MIN_TRADE_PCT = 0.20
COOLDOWN_DAYS = 5

# cache
DATA_DIR = Path("data")
CACHE_FILE = DATA_DIR / "bist_ohlcv.parquet"
INCREMENTAL_LOOKBACK_DAYS = 10

# yahoo download stability
CHUNK_SIZE = 6          # smaller = fewer blocks
MAX_RETRIES = 7
SLEEP_BASE = 2.0

# =========================
# Time helpers (tz-safe)
# =========================
def today_utc_naive() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None).normalize()

def normalize_ts(x) -> pd.Timestamp | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    ts = pd.to_datetime(x, errors="coerce")
    if ts is pd.NaT:
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()

def safe_print(msg: str) -> None:
    print(msg, flush=True)

# =========================
# Yahoo download (chunk + backoff)
# =========================
def yf_download_chunked(tickers, start, end, chunk_size=CHUNK_SIZE, max_retries=MAX_RETRIES, sleep_base=SLEEP_BASE):
    all_raw = None
    tickers = list(tickers)

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        last_err = None

        for r in range(max_retries):
            try:
                raw = yf.download(
                    tickers=chunk,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    group_by="column",
                    threads=False,      # IMPORTANT: reduce rate-limit pressure
                    progress=True,
                )
                if all_raw is None:
                    all_raw = raw
                else:
                    all_raw = pd.concat([all_raw, raw], axis=1)
                last_err = None
                break
            except Exception as e:
                last_err = e
                wait = sleep_base * (2 ** r)
                safe_print(f"Download retry {r+1}/{max_retries} err={type(e).__name__} wait={wait:.1f}s")
                time.sleep(wait)

        if last_err is not None:
            safe_print(f"Chunk failed: {chunk[:3]}... err={last_err}")

        # gentle pacing between chunks
        time.sleep(0.8)

    return all_raw if all_raw is not None else pd.DataFrame()

# =========================
# Raw -> Panel
# =========================
def _normalize_one(df: pd.DataFrame, ticker: str):
    if df is None or df.empty:
        return None

    cols_lower = [str(c).lower() for c in df.columns]
    idx_lower  = [str(i).lower() for i in df.index]
    need_any = {"open","high","low","close","adj close","volume"}

    if not (need_any & set(cols_lower)) and (need_any & set(idx_lower)):
        df = df.T

    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    if "close" not in df.columns and "adj close" in df.columns:
        df["close"] = df["adj close"]

    needed = ["open","high","low","close","volume"]
    if not all(c in df.columns for c in needed):
        return None

    out = df[needed].copy()
    out = out.reset_index().rename(columns={out.reset_index().columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["ticker"] = ticker

    for c in needed:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["date","close"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if out.empty:
        return None
    return out[["date","ticker","open","high","low","close","volume"]]

def raw_to_panel(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","volume"])

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            sub = None
            for level in [1, 0]:
                try:
                    sub = raw.xs(t, axis=1, level=level)
                    break
                except Exception:
                    sub = None
            if sub is None:
                continue
            one = _normalize_one(sub, t)
            if one is not None:
                frames.append(one)
    else:
        one = _normalize_one(raw.copy(), tickers[0] if tickers else "NA")
        if one is not None:
            frames.append(one)

    if not frames:
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","volume"])

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel = panel.dropna(subset=["date","ticker","close"]).drop_duplicates(subset=["date","ticker"], keep="last")
    return panel.sort_values(["ticker","date"]).reset_index(drop=True)

def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","volume"])
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date","ticker","close"]).drop_duplicates(subset=["date","ticker"], keep="last")
        return df.sort_values(["ticker","date"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","volume"])

def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def update_cache(tickers: list[str], cache_path: Path, start_fallback: str, end):
    cache = load_cache(cache_path)

    if not cache.empty:
        last_dt = normalize_ts(cache["date"].max())
        dl_start = str((last_dt - pd.Timedelta(days=INCREMENTAL_LOOKBACK_DAYS)).date())
        safe_print(f"Cache var. Son tarih={last_dt.date()} → incremental start={dl_start}")
    else:
        dl_start = start_fallback
        safe_print(f"Cache yok. Full download start={dl_start}")

    raw = yf_download_chunked(tickers, dl_start, end)
    new_panel = raw_to_panel(raw, tickers)

    if new_panel.empty:
        if cache.empty:
            return cache, None, "⛔ VERİ YOK (Yahoo rate-limit / erişim)"
        data_date = normalize_ts(cache["date"].max())
        return cache, data_date, "⚠️ Yahoo erişim yok → CACHE ile devam"

    merged = pd.concat([cache, new_panel], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.normalize()
    merged = merged.dropna(subset=["date","ticker","close"]).drop_duplicates(subset=["date","ticker"], keep="last")
    merged = merged.sort_values(["ticker","date"]).reset_index(drop=True)

    save_cache(merged, cache_path)
    data_date = normalize_ts(merged["date"].max())
    return merged, data_date, "✅ Yahoo güncel/cached"

def load_market_series(start: str, end):
    for sym in MARKET_CANDIDATES:
        try:
            df = yf.download(sym, start=start, end=end, auto_adjust=False, progress=False, threads=False)
            if df is None or df.empty:
                continue
            df = df.reset_index().rename(columns={df.reset_index().columns[0]: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            df = df.rename(columns={c: str(c).lower() for c in df.columns})
            if "close" not in df.columns and "adj close" in df.columns:
                df["close"] = df["adj close"]
            if "close" not in df.columns:
                continue
            df["mkt_close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["date","mkt_close"])
            if df.empty:
                continue
            safe_print(f"Market proxy: {sym}")
            return df[["date", "mkt_close"]].copy()
        except Exception:
            continue
    safe_print("Market proxy yok. Fallback: universe ortalama getirisi.")
    return None

# =========================
# FEATURES
# =========================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def add_features(panel: pd.DataFrame, mkt: pd.DataFrame | None) -> pd.DataFrame:
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["date", "ticker", "close"]).copy()
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
        out = out.merge(mdf[["date","mkt_ret_1","mkt_ret_5","mkt_ret_20","trend_flag"]],
                        on="date", how="left")
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

# =========================
# Core (backtest)
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
    xs = df[col].to_numpy(float)
    ts = df["ticker"].to_numpy()
    for i, (t, x) in enumerate(zip(ts, xs)):
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

# =========================
# LIVE signal on latest feature date (fix)
# =========================
def build_live_signal(panel_feat: pd.DataFrame, top_n: int) -> pd.DataFrame:
    train_df = panel_feat.dropna(subset=FEATURES + ["y_fwd"]).copy()
    dates = np.array(sorted(train_df["date"].unique()))
    if len(dates) < (TRAIN_DAYS + 60):
        return pd.DataFrame()

    train_dates = dates[-TRAIN_DAYS:]
    train_slice = train_df[train_df["date"].isin(train_dates)].copy()

    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=6, max_iter=800, random_state=42
    )
    model.fit(train_slice[FEATURES], train_slice["y_fwd"].astype(float))

    sig_df = panel_feat.dropna(subset=FEATURES).copy()
    last_date = sig_df["date"].max()
    day = sig_df[sig_df["date"] == last_date].copy()
    day["mu_hat"] = model.predict(day[FEATURES])

    day["vol_20"] = pd.to_numeric(day["vol_20"], errors="coerce")
    day["score"] = day["mu_hat"] - (LAMBDA_VOL * day["vol_20"].fillna(day["vol_20"].median()))

   day = day.sort_values("score", ascending=False).head(top_n).copy()

# --- score -> weight (softmax) ---
scores = day["score"].astype(float).to_numpy()
scores = scores - np.nanmax(scores)  # stabilize
temp = 1.0  # istersen 0.7 daha agresif yapar, 1.5 daha yumuşak
w = np.exp(scores / temp)
w = np.where(np.isfinite(w), w, 0.0)

if w.sum() <= 0:
    w = np.ones_like(w)

w = w / w.sum()
w = w * GROSS_CAP

# position cap uygula, sonra tekrar normalize et
w = np.minimum(w, POS_CAP)
if w.sum() > 0:
    w = (w / w.sum()) * GROSS_CAP

day["w_final"] = w
day["weight_%"] = (day["w_final"] * 100).round(3)
day["alloc_TL"] = (day["w_final"] * INIT_CAPITAL_TL).round(0).astype(int)

# =========================
# MAIN
# =========================
def main():
    safe_print("1) Download (Yahoo) + Cache (Parquet)...")
    panel, data_date, fetch_note = update_cache(TICKERS_ALL, CACHE_FILE, START, END)

    today = today_utc_naive()

    # kill-switch if no data
    if panel is None or panel.empty or data_date is None:
        orders_df = pd.DataFrame([{
            "date": str(today.date()),
            "side": "NONE",
            "ticker": "NA",
            "target_weight_%": 0.0,
            "target_alloc_TL": 0,
            "note": "VERİ YOK",
            "data_date": "unknown",
            "fresh": 0,
            "fresh_note": "⛔ VERİ YOK (Yahoo rate-limit / erişim) → BUGÜN İŞLEM YOK"
        }])
        orders_df.to_csv("orders_today.csv", index=False)
        safe_print("Panel yok → orders_today.csv yazıldı (kill-switch).")
        return

    data_date = normalize_ts(data_date)
    staleness = int((today - data_date).days)
    fresh = 1 if staleness <= MAX_STALENESS_DAYS else 0
    fresh_note = "✅ GÜNCEL" if fresh else f"⚠️ GÜNCEL DEĞİL ({staleness} gün eski)"
    safe_print(f"Data date: {data_date.date()} | {fetch_note} | {fresh_note}")

    mkt = load_market_series(START, END)

    safe_print("2) Features...")
    panel_feat = add_features(panel, mkt)
    panel_feat = panel_feat.sort_values(["date","ticker"]).reset_index(drop=True)

    # optional: walk-forward performance files
    safe_print("3) Walk-forward backtest...")
    wf_df = panel_feat.dropna(subset=["date","ticker"] + FEATURES + ["y_fwd","ret_1","vol_20","dv20"]).copy()
    dates = np.array(sorted(wf_df["date"].unique()))
    trades_chunks = []
    daily_chunks = []

    if len(dates) >= (TRAIN_DAYS + TEST_DAYS + 50):
        for fold, (train_dates, test_dates) in enumerate(
            walk_forward_dates(dates, TRAIN_DAYS, TEST_DAYS, STEP_DAYS), start=1
        ):
            train = wf_df[wf_df["date"].isin(train_dates)].copy()
            test  = wf_df[wf_df["date"].isin(test_dates)].copy()
            if train.empty or test.empty:
                continue

            tr_dates = np.array(sorted(train["date"].unique()))
            split = int(len(tr_dates) * 0.8)
            trd, vad = tr_dates[:split], tr_dates[split:]
            tr_df = train[train["date"].isin(trd)].copy()
            va_df = train[train["date"].isin(vad)].copy()
            if tr_df.empty or va_df.empty:
                continue

            model = HistGradientBoostingRegressor(
                learning_rate=0.05, max_depth=6, max_iter=800, random_state=42
            )
            model.fit(tr_df[FEATURES], tr_df["y_fwd"].astype(float))

            va_pred = va_df[["date","ticker","ret_1","vol_20","dv20"]].copy()
            va_pred["mu_hat"] = model.predict(va_df[FEATURES])

            best = (-1e9, VOL_Q_GRID[0])
            for vol_q in VOL_Q_GRID:
                thr_va = compute_vol_thr_from_context(
                    pd.concat([tr_df[["date","vol_20"]], va_pred[["date","vol_20"]]]),
                    vol_q
                )
                _, d_va = backtest(va_pred, thr_va, TOP_N)
                s = sharpe(d_va["pnl_scaled"])
                if s > best[0]:
                    best = (s, vol_q)
            vol_q_best = best[1]

            model.fit(train[FEATURES], train["y_fwd"].astype(float))

            te_pred = test[["date","ticker","ret_1","vol_20","dv20"]].copy()
            te_pred["mu_hat"] = model.predict(test[FEATURES])

            thr_te = compute_vol_thr_from_context(
                pd.concat([train[["date","vol_20"]], te_pred[["date","vol_20"]]]),
                vol_q_best
            )
            bt, d = backtest(te_pred, thr_te, TOP_N)
            bt["fold"] = fold
            trades_chunks.append(bt)
            daily_chunks.append(d)

    if daily_chunks:
        all_daily = pd.concat(daily_chunks)
        if "date" not in all_daily.columns:
            all_daily = all_daily.reset_index().rename(columns={"index":"date"})
        all_daily["date"] = pd.to_datetime(all_daily["date"], errors="coerce").dt.normalize()
        daily = all_daily.groupby("date")[["pnl","pnl_scaled","scale"]].mean().sort_index()
        daily["equity"] = np.exp(daily["pnl"].cumsum())
        daily["equity_scaled"] = np.exp(daily["pnl_scaled"].cumsum())
        daily.to_csv("equity_curve_live.csv")

        rep = pd.DataFrame([{
            "end_capital_TL": INIT_CAPITAL_TL * float(daily["equity_scaled"].iloc[-1]),
            "pnl_TL": INIT_CAPITAL_TL * float(daily["equity_scaled"].iloc[-1]) - INIT_CAPITAL_TL,
            "return_%": (float(daily["equity_scaled"].iloc[-1]) - 1.0) * 100,
            "sharpe": sharpe(daily["pnl_scaled"]),
            "maxDD_%": max_dd(daily["equity_scaled"]) * 100,
            "days": len(daily)
        }])
        rep.to_csv("report_live.csv", index=False)

    # LIVE signal (always latest feature date)
    safe_print("4) LIVE signal on latest date (fix stale signal)...")
    live = build_live_signal(panel_feat, TOP_N)

    if live.empty:
        orders_df = pd.DataFrame([{
            "date": str(today.date()),
            "side": "NONE",
            "ticker": "NA",
            "target_weight_%": 0.0,
            "target_alloc_TL": 0,
            "note": "SİNYAL ÜRETİLEMEDİ",
            "data_date": str(data_date.date()),
            "fresh": int(fresh),
            "fresh_note": fresh_note
        }])
        orders_df.to_csv("orders_today.csv", index=False)
        return

    live.to_csv("live_signal_today.csv", index=False)

    signal_date = normalize_ts(live["date"].max())
    today_set = set(live["ticker"].tolist())

    # build prev set (yesterday) via same live builder on truncated data
    all_dates = sorted(panel_feat["date"].dropna().unique())
    all_dates = [normalize_ts(d) for d in all_dates if normalize_ts(d) is not None]
    all_dates = sorted(list(dict.fromkeys(all_dates)))
    prev_date = all_dates[-2] if len(all_dates) >= 2 else None

    if prev_date is not None:
        prev_feat = panel_feat[panel_feat["date"] <= prev_date].copy()
        prev_live = build_live_signal(prev_feat, TOP_N)
        prev_set = set(prev_live["ticker"].tolist()) if not prev_live.empty else set()
    else:
        prev_set = set()

    to_buy  = today_set - prev_set
    to_sell = prev_set - today_set

    # cooldown block (avoid flip-flop)
    recent_dates = all_dates[-(COOLDOWN_DAYS + 3):]
    top_sets = {}
    for d in recent_dates:
        sub = panel_feat[panel_feat["date"] <= d].copy()
        ls = build_live_signal(sub, TOP_N)
        top_sets[d] = set(ls["ticker"].tolist()) if not ls.empty else set()

    cooldown_block = set()
    for i in range(1, len(recent_dates)):
        cooldown_block |= (top_sets.get(recent_dates[i-1], set()) - top_sets.get(recent_dates[i], set()))
    to_buy = {t for t in to_buy if t not in cooldown_block}

    orders = []
    for _, r in live.iterrows():
        t = r["ticker"]
        if float(r["weight_%"]) < MIN_TRADE_PCT:
            continue
        side = "AL" if t in to_buy else "TUT"
        orders.append({
            "date": str(signal_date.date()),
            "side": side,
            "ticker": t,
            "target_weight_%": float(r["weight_%"]),
            "target_alloc_TL": int(r["alloc_TL"]),
            "note": "T+1 açılış/ilk likit",
            "data_date": str(data_date.date()),
            "fresh": int(fresh),
            "fresh_note": fresh_note
        })

    for t in sorted(list(to_sell)):
        orders.append({
            "date": str(signal_date.date()),
            "side": "SAT",
            "ticker": t,
            "target_weight_%": 0.0,
            "target_alloc_TL": 0,
            "note": "Listeden çıktı → T+1 açılış/ilk likit",
            "data_date": str(data_date.date()),
            "fresh": int(fresh),
            "fresh_note": fresh_note
        })

    orders_df = pd.DataFrame(orders)
    if orders_df.empty:
        orders_df = pd.DataFrame([{
            "date": str(signal_date.date()),
            "side": "NONE",
            "ticker": "NA",
            "target_weight_%": 0.0,
            "target_alloc_TL": 0,
            "note": "NO TRADE",
            "data_date": str(data_date.date()),
            "fresh": int(fresh),
            "fresh_note": fresh_note
        }])

    orders_df.to_csv("orders_today.csv", index=False)
    safe_print(f"✅ BIST signal date: {signal_date.date()} | data_date={data_date.date()} | {fresh_note}")
    safe_print("✅ Files: live_signal_today.csv, orders_today.csv, equity_curve_live.csv, report_live.csv")

if __name__ == "__main__":
    main()
