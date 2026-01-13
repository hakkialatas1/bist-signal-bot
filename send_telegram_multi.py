- name: Send Telegram (both)
  run: python send_telegram_multi.py
  env:
    TG_BOT_TOKEN: ${{ secrets.TG_BOT_TOKEN }}
    TG_CHAT_ID: ${{ secrets.TG_CHAT_ID }}