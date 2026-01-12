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

orders = pd.read_csv("orders_today.csv")

# Güvenli dönüşümler
orders["target_weight_%"] = pd.to_numeric(orders["target_weight_%"], errors="coerce").fillna(0.0)
orders["target_alloc_TL"] = pd.to_numeric(orders["target_alloc_TL"], errors="coerce").fillna(0).astype(int)

date = str(orders["date"].iloc[0])

buy  = orders[orders["side"] == "AL"].copy()
hold = orders[orders["side"] == "TUT"].copy()
sell = orders[orders["side"] == "SAT"].copy()
TOP_N = 15
buy = buy.sort_values("target_weight_%", ascending=False).head(TOP_N)
hold = hold.sort_values("target_weight_%", ascending=False).head(TOP_N)

# 0 veya çok küçükleri filtrele (güzel görünmesi için)
MIN_PRINT_PCT = 0.20
buy = buy[buy["target_weight_%"] >= MIN_PRINT_PCT]
hold = hold[hold["target_weight_%"] >= MIN_PRINT_PCT]

# Sırala + filtrele + ilk TOP_N
buy  = buy.sort_values("target_weight_%", ascending=False)
hold = hold.sort_values("target_weight_%", ascending=False)

buy  = buy[buy["target_weight_%"] >= MIN_PRINT_PCT].head(TOP_N)
hold = hold[hold["target_weight_%"] >= MIN_PRINT_PCT].head(TOP_N)

# SAT zaten liste dışı olduğu için filtre gerekmiyor (istersen ilk 20 ile sınırlayabiliriz)
sell = sell.head(30)

lines = []
lines.append(f"📈 BIST100 SİNYAL (TOP{TOP_N})")
lines.append(f"Sinyal: {date}")
lines.append("Uygulama: T+1 açılış / ilk likit")
lines.append("")

if len(buy):
    lines.append("🟢 AL (yeni):")
    for _, r in buy.iterrows():
        lines.append(f"• {r['ticker']}  %{r['target_weight_%']:.3f}  (~{int(r['target_alloc_TL'])} TL)")
    lines.append("")
else:
    lines.append("🟢 AL (yeni): Yok")
    lines.append("")

if len(hold):
    lines.append("🟡 TUT (devam):")
    for _, r in hold.iterrows():
        lines.append(f"• {r['ticker']}  %{r['target_weight_%']:.3f}  (~{int(r['target_alloc_TL'])} TL)")
    lines.append("")
else:
    lines.append("🟡 TUT (devam): Yok")
    lines.append("")

if len(sell):
    lines.append("🔴 SAT (çıkan):")
    for _, r in sell.iterrows():
        lines.append(f"• {r['ticker']}")
    lines.append("")
else:
    lines.append("🔴 SAT (çıkan): Yok")
    lines.append("")

msg = "\n".join(lines).strip()

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
resp = requests.post(url, json={"chat_id": CHAT_ID, "text": msg})

# Hata olursa workflow kırmızı yapsın
if resp.status_code != 200:
    raise SystemExit(f"Telegram error: {resp.status_code} {resp.text}")

print(resp.status_code, resp.text)
