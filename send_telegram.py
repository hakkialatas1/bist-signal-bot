import os
import pandas as pd
import requests

TOKEN = os.getenv("TG_BOT_TOKEN")
CHAT_ID = os.getenv("TG_CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise SystemExit("TG_BOT_TOKEN veya TG_CHAT_ID yok. GitHub Secrets ekle.")

# Ayarlar
TOP_N = 15
MIN_PRINT_PCT = 0.20  # %0.20 altını Telegram'da gösterme
KILL_SWITCH_IF_NOT_FRESH = True  # veri güncel değilse emir basma

orders = pd.read_csv("orders_today.csv")

# Güvenli dönüşümler
orders["target_weight_%"] = pd.to_numeric(orders.get("target_weight_%", 0), errors="coerce").fillna(0.0)
orders["target_alloc_TL"] = pd.to_numeric(orders.get("target_alloc_TL", 0), errors="coerce").fillna(0).astype(int)

# Data freshness bilgisi (yoksa boş geç)
data_date = str(orders["data_date"].iloc[0]) if "data_date" in orders.columns and len(orders) else "unknown"
fresh_note = str(orders["fresh_note"].iloc[0]) if "fresh_note" in orders.columns and len(orders) else ""
fresh = int(pd.to_numeric(orders["fresh"].iloc[0], errors="coerce")) if "fresh" in orders.columns and len(orders) else 0

date = str(orders["date"].iloc[0]) if "date" in orders.columns and len(orders) else "unknown"

buy  = orders[orders["side"] == "AL"].copy()
hold = orders[orders["side"] == "TUT"].copy()
sell = orders[orders["side"] == "SAT"].copy()

# Sırala + filtrele + limit
buy  = buy.sort_values("target_weight_%", ascending=False)
hold = hold.sort_values("target_weight_%", ascending=False)

buy  = buy[buy["target_weight_%"] >= MIN_PRINT_PCT].head(TOP_N)
hold = hold[hold["target_weight_%"] >= MIN_PRINT_PCT].head(TOP_N)

sell = sell.drop_duplicates(subset=["ticker"]).head(15)

lines = []
lines.append(f"📈 BIST100 SİNYAL (TOP{TOP_N})")
lines.append(f"Sinyal: {date}")
lines.append("Uygulama: T+1 açılış / ilk likit")
lines.append(f"📅 Veri tarihi: {data_date}  {fresh_note}".strip())
lines.append("")

# Kill-switch: veri güncel değilse emirleri basma
if KILL_SWITCH_IF_NOT_FRESH and fresh == 0:
    lines.append("⛔ Veri güncel değil → BUGÜN İŞLEM YOK (güvenlik kilidi)")
    msg = "\n".join(lines).strip()

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": CHAT_ID, "text": msg})
    if resp.status_code != 200:
        raise SystemExit(f"Telegram error: {resp.status_code} {resp.text}")
    print(resp.status_code, resp.text)
    raise SystemExit(0)

# Normal mesaj
if len(buy):
    lines.append("🟢 AL (yeni):")
    for _, r in buy.iterrows():
        lines.append(f"• {r['ticker']}  %{float(r['target_weight_%']):.3f}  (~{int(r['target_alloc_TL'])} TL)")
    lines.append("")
else:
    lines.append("🟢 AL (yeni): Yok\n")

if len(hold):
    lines.append("🟡 TUT (devam):")
    for _, r in hold.iterrows():
        lines.append(f"• {r['ticker']}  %{float(r['target_weight_%']):.3f}  (~{int(r['target_alloc_TL'])} TL)")
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

msg = "\n".join(lines).strip()

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
resp = requests.post(url, json={"chat_id": CHAT_ID, "text": msg})

if resp.status_code != 200:
    raise SystemExit(f"Telegram error: {resp.status_code} {resp.text}")

print(resp.status_code, resp.text)