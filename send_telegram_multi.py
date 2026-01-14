import os
import pandas as pd
import requests

TOKEN = os.getenv("TG_BOT_TOKEN")
CHAT_ID = os.getenv("TG_CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise SystemExit("TG_BOT_TOKEN veya TG_CHAT_ID yok. GitHub Secrets ekle.")

MIN_PRINT_PCT = 0.20

def send_message(text: str) -> None:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": CHAT_ID, "text": text})
    if resp.status_code != 200:
        raise SystemExit(f"Telegram error: {resp.status_code} {resp.text}")

def build_msg(orders_path: str, title: str, top_n: int = 15) -> str:
    if not os.path.exists(orders_path):
        return f"📉 {title}\n⛔ orders dosyası yok: {orders_path}"

    orders = pd.read_csv(orders_path)
    if orders.empty:
        return f"📉 {title}\n⛔ orders dosyası boş: {orders_path}"

    weight_col = "target_weight_%"
    alloc_col = "target_alloc_TL" if "target_alloc_TL" in orders.columns else (
        "target_alloc_USD" if "target_alloc_USD" in orders.columns else None
    )

    orders[weight_col] = pd.to_numeric(orders.get(weight_col, 0), errors="coerce").fillna(0.0)
    if alloc_col:
        orders[alloc_col] = pd.to_numeric(orders.get(alloc_col, 0), errors="coerce").fillna(0).astype(int)

    data_date = str(orders["data_date"].iloc[0]) if "data_date" in orders.columns else "unknown"
    fresh_note = str(orders["fresh_note"].iloc[0]) if "fresh_note" in orders.columns else ""
    fresh = int(pd.to_numeric(orders["fresh"].iloc[0], errors="coerce")) if "fresh" in orders.columns else 0

    date = str(orders["date"].iloc[0]) if "date" in orders.columns else "unknown"

    buy  = orders[orders["side"] == "AL"].copy()
    hold = orders[orders["side"] == "TUT"].copy()
    sell = orders[orders["side"] == "SAT"].copy()

    buy  = buy.sort_values(weight_col, ascending=False)
    hold = hold.sort_values(weight_col, ascending=False)

    buy  = buy[buy[weight_col] >= MIN_PRINT_PCT].head(top_n)
    hold = hold[hold[weight_col] >= MIN_PRINT_PCT].head(top_n)
    sell = sell.drop_duplicates(subset=["ticker"]).head(15)

    lines = []
    lines.append(f"📈 {title} (TOP{top_n})")
    lines.append(f"Sinyal: {date}")
    lines.append("Uygulama: T+1 açılış / ilk likit")

    meta = f"📅 Veri tarihi: {data_date}"
    if fresh_note.strip():
        meta = f"{meta}  {fresh_note.strip()}"
    lines.append(meta)
    lines.append("")

    # veri taze değilse emir basma (sadece bilgi)
    if fresh == 0:
        lines.append("⛔ Veri güncel değil → BUGÜN İŞLEM YOK (güvenlik kilidi)")
        return "\n".join(lines).strip()

    if len(buy):
        lines.append("🟢 AL (yeni):")
        for _, r in buy.iterrows():
            if alloc_col:
                unit = "TL" if alloc_col == "target_alloc_TL" else "USD"
                alloc_txt = f" (~{int(r[alloc_col])} {unit})"
            else:
                alloc_txt = ""
            lines.append(f"• {r['ticker']}  %{float(r[weight_col]):.3f}{alloc_txt}")
        lines.append("")
    else:
        lines.append("🟢 AL (yeni): Yok\n")

    if len(hold):
        lines.append("🟡 TUT (devam):")
        for _, r in hold.iterrows():
            if alloc_col:
                unit = "TL" if alloc_col == "target_alloc_TL" else "USD"
                alloc_txt = f" (~{int(r[alloc_col])} {unit})"
            else:
                alloc_txt = ""
            lines.append(f"• {r['ticker']}  %{float(r[weight_col]):.3f}{alloc_txt}")
        lines.append("")
    else:
        lines.append("🟡 TUT (devam): Yok\n")

    if len(sell):
        lines.append("🔴 SAT (çıkan):")
        for _, r in sell.iterrows():
            lines.append(f"• {r['ticker']}")
        lines.append("")
    else:
        lines.append("🔴 SAT (çıkan): Yok\n")

    return "\n".join(lines).strip()

def main():
    bist_msg = build_msg("orders_bist.csv", "BIST100 SİNYAL", top_n=15)
    send_message(bist_msg)

    us_msg = build_msg("orders_us.csv", "US (DOW30) SİNYAL", top_n=15)
    send_message(us_msg)

    print("OK: sent both messages")

if __name__ == "__main__":
    main()