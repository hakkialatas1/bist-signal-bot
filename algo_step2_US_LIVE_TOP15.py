# algo_step2_US_LIVE_TOP15.py
# FINAL CLEAN US LIVE (TOP15)
# - Yahoo rate-limit resistant (chunk + backoff, threads=False)
# - Parquet cache (data/us_ohlcv.parquet)
# - tz-safe dates
# - LIVE signal on latest feature date
# - Softmax weights (not equal weights)
# - Kill-switch outputs side=NONE if data missing / stale policy

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor

# =========================
# UNIVERSE (DOW30-ish + mega)
# =========================
US_TICKERS = [
    # DOW 30 core
    "AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS",
    "DOW","GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD",
    "MMM","MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WMT",
    # a few extra mega/liquid (optional)
    "META","NVDA","AVGO","XOM","BLK","MA","WFC","T"
]
TICKERS_ALL = sorted(list(dict.fromkeys(US_TICKERS)))

# =========================
# PARAMS
# =========================
START = "2016-01-01"
END = None

HORIZON = 5
TRAIN_DAYS_LIVE = 252 * 3

TOP_N = 15
INIT_CAPITAL_USD = 10_000

LAMBDA_VOL = 0.25
GROSS_CAP = 1.00
POS_CAP = 0.12
SOFTMAX_TEMP = 1.0

# liquidity filter (US: use dollar volume)
USE_LIQ_FILTER = True
MIN_AVG_DV20 = 50_000_000

# kill-switch
MAX_STALENESS_DAYS = 2
MIN_TRADE_PCT = 0.20
COOLDOWN_DAYS = 5

# cache
DATA_DIR = Path("data")
CACHE_FILE = DATA_DIR / "us_ohlcv.parquet"
INCREMENTAL_LOOKBACK_DAYS = 10

# yahoo stability
CHUNK_SIZE = 8
MAX_RETRIES = 7
SLEEP_BASE = 2.0

MARKET_PROXY = "^GSPC"  # S&P 500 proxy


# =========================
# UTIL (tz-safe)
# =========================
def today_utc_naive() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None).normalize()

def norm_date(x) -> pd.Timestamp | None:
    ts = pd.to_datetime(x, errors="coerce")
    if ts is pd.NaT:
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()

def log(msg: str) -> None:
    print(msg, flush=True)


# =========================
# YFINANCE DOWNLOAD (chunk + backoff)
# =========================
def yf_download_chunked(tickers, start, end):
    tickers = list(tickers)
    all_raw = None

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        last_err = None

        for r in range(MAX_RETRIES):
            try:
                raw = yf.download(
                    tickers=chunk,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    group_by="column",
                    threads=False,
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
                wait = SLEEP_BASE * (2 ** r)
                log(f"Download retry {r+1}/{MAX_RETRIES} err={type(e).__name__} wait={wait:.1f}s")
                time.sleep(wait)

        if last_err is not None:
            log(f"Chunk failed: {chunk[:3]}... err={last_err}")

        time.sleep(0.6)

    return all_raw if all_raw is not None else pd.DataFrame()


# =========================
# RAW -> PANEL
# =========================
def _normalize_one(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    cols_lower = [str(c).lower() for c in df.columns]
    idx_lower = [str(i).lower() for i in df.index]
    need_any = {"open", "high", "low", "close", "adj close", "volume"}

    if not (need_any & set(cols_lower)) and (need_any & set(idx_lower)):
        df = df.T

    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    if "close" not in df.columns and "adj close" in df.columns:
        df["close"] = df["adj close"]

    needed = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in needed):
        return None

    out = df[needed].copy()
    out = out.reset_index().rename(columns={out.reset_index().columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["ticker"] = ticker

    for c in needed:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    if out.empty:
        return None

    return out[["date", "ticker", "open", "high", "low", "close", "volume"]]

def raw_to_panel(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

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

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel = panel.dropna(subset=["date", "ticker", "close"])
    panel = panel.drop_duplicates(subset=["date", "ticker"], keep="last")
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


# =========================
# CACHE
# =========================
def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date", "ticker", "close"]).drop_duplicates(subset=["date", "ticker"], keep="last")
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def update_cache(tickers: list[str], cache_path: Path):
    cache = load_cache(cache_path)

    if not cache.empty:
        last_dt = norm_date(cache["date"].max())
        assert last_dt is not None
        dl_start = str((last_dt - pd.Timedelta(days=INCREMENTAL_LOOKBACK_DAYS)).date())
        log(f"Cache var. Son tarih={last_dt.date()} → incremental start={dl_start}")
    else:
        dl_start = START
        log(f"Cache yok. Full download start={dl_start}")

    raw = yf_download_chunked(tickers, dl_start, END)
    new_panel = raw_to_panel(raw, tickers)

    if new_panel.empty:
        if cache.empty:
            return cache, None, "⛔ VERİ YOK (Yahoo rate-limit / erişim)"
        data_date = norm_date(cache["date"].max())
        return cache, data_date, "⚠️ Yahoo erişim yok → CACHE ile devam"

    merged = pd.concat([cache, new_panel], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.normalize()
    merged = merged.dropna(subset=["date", "ticker", "close"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    save_cache(merged, cache_path)
    data_date = norm_date(merged["date"].max())
    return merged, data_date, "✅ Yahoo güncel/cached"


# =========================
# MARKET SERIES (S&P500)
# =========================
def load_market_series(start: str, end):
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
        df = df.dropna(subset=["date", "mkt_close"])
        if df.empty:
            return None
        log(f"Market proxy: {MARKET_PROXY}")
        return df[["date", "mkt_close"]].copy()
    except Exception:
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

    out["ret_1"] = g["close"].apply(lambda s: np.log(s / s.shift(1)))
    out["ret_2"] = g["close"].apply(lambda s: np.log(s / s.shift(2)))
    out["ret_5"] = g["close"].apply(lambda s: np.log(s / s.shift(5)))
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
        mdf["mkt_ret_1"] = np.log(mdf["mkt_close"] / mdf["mkt_close"].shift(1))
        mdf["mkt_ret_5"] = np.log(mdf["mkt_close"] / mdf["mkt_close"].shift(5))
        mdf["mkt_ret_20"] = np.log(mdf["mkt_close"] / mdf["mkt_close"].shift(20))
        mdf["trend_flag"] = (mdf["mkt_close"].rolling(50).mean() > mdf["mkt_close"].rolling(200).mean()).astype(int)
        out = out.merge(
            mdf[["date", "mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "trend_flag"]],
            on="date",
            how="left"
        )
    else:
        uni = out.groupby("date")["ret_1"].mean().rename("mkt_ret_1").to_frame()
        uni["mkt_ret_5"] = uni["mkt_ret_1"].rolling(5).sum()
        uni["mkt_ret_20"] = uni["mkt_ret_1"].rolling(20).sum()
        uni["trend_flag"] = (uni["mkt_ret_20"] > 0).astype(int)
        out = out.merge(uni.reset_index(), on="date", how="left")

    out["rel_5"] = out["ret_5"] - out["mkt_ret_5"]
    out["rel_20"] = out["ret_20"] - out["mkt_ret_20"]

    out["cs_rank_rel20"] = out.groupby("date")["rel_20"].rank(pct=True)
    out["cs_rank_ret5"] = out.groupby("date")["ret_5"].rank(pct=True)
    out["cs_rank_rsi"] = out.groupby("date")["rsi_14"].rank(pct=True)
    out["cs_rank_z50"] = out.groupby("date")["z_50"].rank(pct=True)

    def rolling_beta(df_t: pd.DataFrame) -> pd.Series:
        x = df_t["mkt_ret_1"]
        y = df_t["ret_1"]
        cov = (x * y).rolling(60).mean() - x.rolling(60).mean() * y.rolling(60).mean()
        var = x.rolling(60).var()
        return cov / (var + 1e-12)

    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
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
# LIVE SIGNAL
# =========================
def build_live_signal(panel_feat: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = panel_feat.copy()
    train_df = df.dropna(subset=FEATURES + ["y_fwd"]).copy()

    dates = np.array(sorted(train_df["date"].unique()))
    if len(dates) < (TRAIN_DAYS_LIVE + 60):
        return pd.DataFrame()

    train_dates = dates[-TRAIN_DAYS_LIVE:]
    train_slice = train_df[train_df["date"].isin(train_dates)].copy()

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=700,
        random_state=42
    )
    model.fit(train_slice[FEATURES], train_slice["y_fwd"].astype(float))

    sig_df = df.dropna(subset=FEATURES).copy()
    last_date = sig_df["date"].max()
    day = sig_df[sig_df["date"] == last_date].copy()
    if day.empty:
        return pd.DataFrame()

    day["mu_hat"] = model.predict(day[FEATURES])
    day["vol_20"] = pd.to_numeric(day["vol_20"], errors="coerce")
    day["dv20"] = pd.to_numeric(day["dv20"], errors="coerce")

    if USE_LIQ_FILTER:
        day = day[day["dv20"] >= MIN_AVG_DV20].copy()
        if day.empty:
            return pd.DataFrame()

    day["score"] = day["mu_hat"] - (LAMBDA_VOL * day["vol_20"].fillna(day["vol_20"].median()))
    day = day.sort_values("score", ascending=False).head(top_n).copy()

    scores = day["score"].astype(float).to_numpy()
    scores = scores - np.nanmax(scores)
    w = np.exp(scores / SOFTMAX_TEMP)
    w = np.where(np.isfinite(w), w, 0.0)

    if w.sum() <= 0:
        w = np.ones_like(w)

    w = w / w.sum()
    w = w * GROSS_CAP

    w = np.minimum(w, POS_CAP)
    if w.sum() > 0:
        w = (w / w.sum()) * GROSS_CAP

    day["w_final"] = w
    day["weight_%"] = (day["w_final"] * 100).round(3)
    day["alloc_USD"] = (day["w_final"] * INIT_CAPITAL_USD).round(0).astype(int)

    return day[["date", "ticker", "w_final", "weight_%", "alloc_USD"]].reset_index(drop=True)


def compute_cooldown_block(panel_feat: pd.DataFrame, all_dates: list[pd.Timestamp]) -> set[str]:
    recent = all_dates[-(COOLDOWN_DAYS + 3):]
    top_sets: dict[pd.Timestamp, set[str]] = {}

    for d in recent:
        sub = panel_feat[panel_feat["date"] <= d].copy()
        ls = build_live_signal(sub, TOP_N)
        top_sets[d] = set(ls["ticker"].tolist()) if not ls.empty else set()

    cooldown = set()
    for i in range(1, len(recent)):
        prev_set = top_sets.get(recent[i - 1], set())
        cur_set = top_sets.get(recent[i], set())
        cooldown |= (prev_set - cur_set)

    return cooldown


# =========================
# MAIN
# =========================
def main():
    log("1) Download (Yahoo) + Cache (Parquet)...")
    panel, data_date, fetch_note = update_cache(TICKERS_ALL, CACHE_FILE)
    today = today_utc_naive()

    if panel is None or panel.empty or data_date is None:
        orders_df = pd.DataFrame([{
            "date": str(today.date()),
            "side": "NONE",
            "ticker": "NA",
            "target_weight_%": 0.0,
            "target_alloc_USD": 0,
            "note": "VERİ YOK",
            "data_date": "unknown",
            "fresh": 0,
            "fresh_note": "⛔ VERİ YOK (Yahoo rate-limit / erişim) → BUGÜN İŞLEM YOK"
        }])
        orders_df.to_csv("orders_us.csv", index=False)
        log("Panel yok → orders_us.csv (NONE) yazıldı.")
        return

    data_date = norm_date(data_date)
    assert data_date is not None

    staleness = int((today - data_date).days)
    fresh = 1 if staleness <= MAX_STALENESS_DAYS else 0
    fresh_note = "✅ GÜNCEL" if fresh else f"⚠️ GÜNCEL DEĞİL ({staleness} gün eski)"
    log(f"Data date: {data_date.date()} | {fetch_note} | {fresh_note}")

    log("2) Features...")
    mkt = load_market_series(START, END)
    panel_feat = add_features(panel, mkt)
    panel_feat = panel_feat.sort_values(["date", "ticker"]).reset_index(drop=True)

    log("3) LIVE signal...")
    live = build_live_signal(panel_feat, TOP_N)

    if live.empty:
        orders_df = pd.DataFrame([{
            "date": str(today.date()),
            "side": "NONE",
            "ticker": "NA",
            "target_weight_%": 0.0,
            "target_alloc_USD": 0,
            "note": "SİNYAL ÜRETİLEMEDİ",
            "data_date": str(data_date.date()),
            "fresh": int(fresh),
            "fresh_note": fresh_note
        }])
        orders_df.to_csv("orders_us.csv", index=False)
        log("Sinyal üretilemedi → orders_us.csv (NONE).")
        return

    live.to_csv("live_signal_us_today.csv", index=False)

    signal_date = norm_date(live["date"].max())
    assert signal_date is not None
    today_set = set(live["ticker"].tolist())

    all_dates = sorted([norm_date(d) for d in panel_feat["date"].dropna().unique() if norm_date(d) is not None])
    prev_date = all_dates[-2] if len(all_dates) >= 2 else None

    if prev_date is not None:
        prev_feat = panel_feat[panel_feat["date"] <= prev_date].copy()
        prev_live = build_live_signal(prev_feat, TOP_N)
        prev_set = set(prev_live["ticker"].tolist()) if not prev_live.empty else set()
    else:
        prev_set = set()

    to_buy = today_set - prev_set
    to_sell = prev_set - today_set

    cooldown_block = compute_cooldown_block(panel_feat, all_dates) if len(all_dates) >= 3 else set()
    to_buy = {t for t in to_buy if t not in cooldown_block}

    orders = []
    for _, r in live.iterrows():
        if float(r["weight_%"]) < MIN_TRADE_PCT:
            continue
        t = str(r["ticker"])
        side = "AL" if t in to_buy else "TUT"
        orders.append({
            "date": str(signal_date.date()),
            "side": side,
            "ticker": t,
            "target_weight_%": float(r["weight_%"]),
            "target_alloc_USD": int(r["alloc_USD"]),
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
            "target_alloc_USD": 0,
            "note": "Listeden çıktı → T+1 açılış/ilk likit",
            "data_date": str(data_date.date()),
            "fresh": int(fresh),
            "fresh_note": fresh_note
        })

    if not orders:
        orders = [{
            "date": str(signal_date.date()),
            "side": "NONE",
            "ticker": "NA",
            "target_weight_%": 0.0,
            "target_alloc_USD": 0,
            "note": "NO TRADE",
            "data_date": str(data_date.date()),
            "fresh": int(fresh),
            "fresh_note": fresh_note
        }]

    orders_df = pd.DataFrame(orders)
    orders_df = orders_df.drop_duplicates(subset=["date", "side", "ticker"], keep="first").reset_index(drop=True)
    orders_df.to_csv("orders_us.csv", index=False)

    log("\n✅ Üretilen dosyalar:")
    log(" - live_signal_us_today.csv")
    log(" - orders_us.csv")
    log(f"\nSignal date: {signal_date.date()}")
    log(f"Data date: {data_date.date()} | {fresh_note}")

if __name__ == "__main__":
    main()