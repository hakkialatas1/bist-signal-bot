# algo_step2_LIVE_TOP15.py
# STEP-2 LIVE signal generator + AL/TUT/SAT orders (BIST100 TOP15)
# Uses FINAL-FIXED logic: aggregate pnl, rebuild equity; generate last-day signal

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor


# =========================
# UNIVERSE (embedded BIST100-ish list you used)
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
# FINAL PARAMS (grid winner)
# =========================
START = "2016-01-01"
END = None

HORIZON = 5
TRAIN_DAYS = 252 * 3
TEST_DAYS  = 63
STEP_DAYS  = 63

TOP_N = 15                 # ✅ grid winner
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


# =========================
# DATA helpers
# =========================
def yf_download(tickers, start, end):
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        threads=True,
        progress=True,
    )

def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    cols_lower = [str(c).lower() for c in df.columns]
    idx_lower  = [str(i).lower() for i in df.index]
    need_any = {"open","high","low","close","adj close","volume"}

    # if data is transposed
    if not (need_any & set(cols_lower)) and (need_any & set(idx_lower)):
        df = df.T

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

    df = df.dropna(subset=["date","close"])
    if df.empty:
        return None

    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df[["date","ticker","open","high","low","close","volume"]]

def extract_one_ticker(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
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

def build_panel(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    frames = []
    for t in tickers:
        sub = extract_one_ticker(raw, t)
        if sub is not None:
            frames.append(sub)
    if not frames:
        raise ValueError("Hiçbir ticker için OHLCV alınamadı.")
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)

def load_market_series(start: str, end):
    for sym in MARKET_CANDIDATES:
        try:
            df = yf.download(sym, start=start, end=end, auto_adjust=False, progress=False)
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
            print(f"Market proxy: {sym}")
            return df[["date", "mkt_close"]].copy()
        except Exception:
            continue
    print("Market proxy yok. Fallback: universe ortalama getirisi.")
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

    # label
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
# WALK FORWARD + RISK
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
    for i, (t, x) in enumerate(zip(df["ticker"].to_numpy(), df[col].to_numpy(float))):
        if t != last_t:
            last_t = t
            last = x
        else:
            last = alpha * x + (1 - alpha) * last
        out[i] = last
    return pd.Series(out, index=df.index)


# =========================
# BACKTEST CORE
# =========================
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
# MAIN
# =========================
def main():
    print("1) Download BIST100 (Yahoo)...")
    raw = yf_download(TICKERS_ALL, START, END)

    # Drop missing tickers
    if isinstance(raw.columns, pd.MultiIndex):
        got = set(raw.columns.get_level_values(1).unique())
        tickers_ok = [t for t in TICKERS_ALL if t in got]
        missing = [t for t in TICKERS_ALL if t not in got]
        if missing:
            print(f"Uyarı: {len(missing)} ticker yok (çıkarıldı). Örn: {missing[:10]}")
    else:
        tickers_ok = TICKERS_ALL

    mkt = load_market_series(START, END)

    print("2) Panel + features...")
    panel = build_panel(raw, tickers_ok)
    panel = add_features(panel, mkt)

    need = ["date","ticker","ret_1","vol_20","dv20","y_fwd"] + FEATURES
    panel = panel.dropna(subset=need).copy()
    panel = panel.sort_values(["date","ticker"]).reset_index(drop=True)

    dates = np.array(sorted(panel["date"].unique()))
    if len(dates) < (TRAIN_DAYS + TEST_DAYS + 50):
        raise ValueError(f"Yetersiz gün: {len(dates)}")

    print("3) Walk-forward + collect last signal...")
    trades_chunks = []
    daily_chunks = []

    for fold, (train_dates, test_dates) in enumerate(
        walk_forward_dates(dates, TRAIN_DAYS, TEST_DAYS, STEP_DAYS), start=1
    ):
        train = panel[panel["date"].isin(train_dates)].copy()
        test  = panel[panel["date"].isin(test_dates)].copy()
        if train.empty or test.empty:
            continue

        # 80/20 split inside train for vol_q selection
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

        # choose vol_q
        best = (-1e9, None)
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

        # refit on full train
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

    trades = pd.concat(trades_chunks, ignore_index=True)

    # Aggregate daily pnl correctly, then rebuild equity
    all_daily = pd.concat(daily_chunks)
    all_daily = all_daily.reset_index().rename(columns={"index":"date"}) if "date" not in all_daily.columns else all_daily
    all_daily["date"] = pd.to_datetime(all_daily["date"], errors="coerce").dt.normalize()
    daily = all_daily.groupby("date")[["pnl","pnl_scaled","scale"]].mean().sort_index()
    daily["equity"] = np.exp(daily["pnl"].cumsum())
    daily["equity_scaled"] = np.exp(daily["pnl_scaled"].cumsum())

    daily.to_csv("equity_curve_live.csv")
    rep = pd.DataFrame([{
        "variant":"VOL_TARGET",
        "end_capital_TL": INIT_CAPITAL_TL * float(daily["equity_scaled"].iloc[-1]),
        "pnl_TL": INIT_CAPITAL_TL * float(daily["equity_scaled"].iloc[-1]) - INIT_CAPITAL_TL,
        "return_%": (float(daily["equity_scaled"].iloc[-1]) - 1.0) * 100,
        "sharpe": sharpe(daily["pnl_scaled"]),
        "maxDD_%": max_dd(daily["equity_scaled"]) * 100,
        "days": len(daily)
    }])
    rep.to_csv("report_live.csv", index=False)

    # ===== LIVE SIGNAL (last available date) =====
    last_date = trades["date"].dropna().max()
    if pd.isna(last_date):
        raise ValueError("Sinyal üretilemedi: trades boş.")

    tdf = trades[trades["date"] == last_date].copy()
    tdf["w_final"] = pd.to_numeric(tdf["w_scaled"], errors="coerce").fillna(0.0)
    tdf = tdf[tdf["w_final"] > 0].copy()
    tdf = tdf.sort_values("w_final", ascending=False)

    tdf["alloc_TL"] = (tdf["w_final"] * INIT_CAPITAL_TL).round(0).astype(int)
    tdf["weight_%"] = (tdf["w_final"] * 100).round(3)

    live = tdf[["date","ticker","weight_%","alloc_TL"]].reset_index(drop=True)
    live.to_csv("live_signal_today.csv", index=False)

    # ===== ORDERS (AL / TUT / SAT) =====
    all_dates = sorted(trades["date"].dropna().unique())
    prev_date = all_dates[-2] if len(all_dates) >= 2 else None
    today_date = all_dates[-1]

    today_df = trades[trades["date"] == today_date].copy()
    today_df["w_final"] = pd.to_numeric(today_df["w_scaled"], errors="coerce").fillna(0.0)
    today_df = today_df[today_df["w_final"] > 0].copy()
    today_df = today_df.sort_values("w_final", ascending=False)
    today_df["alloc_TL"] = (today_df["w_final"] * INIT_CAPITAL_TL).round(0).astype(int)
    today_df["weight_%"] = (today_df["w_final"] * 100).round(3)

    if prev_date is not None:
        prev_df = trades[trades["date"] == prev_date].copy()
        prev_df["w_prev"] = pd.to_numeric(prev_df["w_scaled"], errors="coerce").fillna(0.0)
        prev_df = prev_df[prev_df["w_prev"] > 0].copy()
    else:
        prev_df = pd.DataFrame(columns=["ticker","w_prev"])

    today_set = set(today_df["ticker"].tolist())
    prev_set  = set(prev_df["ticker"].tolist())

    to_buy  = today_set - prev_set
    to_sell = prev_set - today_set

    orders = []

    for _, r in today_df.iterrows():
        t = r["ticker"]
        side = "AL" if t in to_buy else "TUT"
        orders.append({
            "date": str(pd.to_datetime(today_date).date()),
            "side": side,
            "ticker": t,
            "target_weight_%": float(r["weight_%"]),
            "target_alloc_TL": int(r["alloc_TL"]),
            "note": "T+1 açılış/ilk likit"
        })

    for t in sorted(list(to_sell)):
        orders.append({
            "date": str(pd.to_datetime(today_date).date()),
            "side": "SAT",
            "ticker": t,
            "target_weight_%": 0.0,
            "target_alloc_TL": 0,
            "note": "Listeden çıktı → T+1 açılış/ilk likit"
        })

    orders_df = pd.DataFrame(orders)
    orders_df.to_csv("orders_today.csv", index=False)

    with open("orders_today.txt", "w", encoding="utf-8") as f:
        f.write(f"ORDERS (BIST100 TOP{TOP_N})\n")
        f.write(f"Signal date: {pd.to_datetime(today_date).date()} | Execute: next session open/first liquid\n\n")
        f.write(orders_df.to_string(index=False))

    print("\n✅ Üretilen dosyalar:")
    print(" - live_signal_today.csv")
    print(" - orders_today.csv")
    print(" - orders_today.txt")
    print(" - equity_curve_live.csv")
    print(" - report_live.csv")
    print(f"\nSignal date: {pd.to_datetime(today_date).date()}")
    print("\nİlk 10 emir:")
    print(orders_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
