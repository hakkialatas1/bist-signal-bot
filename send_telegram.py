import os
import pandas as pd
import requests

TOKEN = os.getenv("TG_BOT_TOKEN")
CHAT_ID = os.getenv("TG_CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise SystemExit("TG_BOT_TOKEN veya TG_CHAT_ID yok. GitHub Secrets ekle.")

orders = pd.read_csv("orders_today.csv")

date = str(orders["date"].iloc[0])
buy  = orders[orders["side"] == "AL"].copy()
hold = orders[orders["side"] == "TUT"].copy()
sell = orders[orders["side"] == "SAT"].copy()

lines = []
lines.append(f"📈 BIST100 SİNYAL (TOP15)\nSinyal: {date}\nUygulama: T+1 açılış / ilk likit\n")

if len(buy):
    lines.append("🟢 AL (yeni):")
    for _, r in buy.iterrows():
        lines.append(f"• {r['ticker']}  %{r['target_weight_%']}  (~{int(r['target_alloc_TL'])} TL)")
    lines.append("")

if len(hold):
    lines.append("🟡 TUT (devam):")
    for _, r in hold.iterrows():
        lines.append(f"• {r['ticker']}  %{r['target_weight_%']}  (~{int(r['target_alloc_TL'])} TL)")
    lines.append("")

if len(sell):
    lines.append("🔴 SAT (çıkan):")
    for _, r in sell.iterrows():
        lines.append(f"• {r['ticker']}")
    lines.append("")

msg = "\n".join(lines).strip()

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
resp = requests.post(url, json={"chat_id": CHAT_ID, "text": msg})
print(resp.status_code, resp.text)
