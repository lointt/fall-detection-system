"""
sender.py — Telegram Alert Sender
===================================
Input : image_path (str), reason (str)
Output: Gửi tin nhắn cảnh báo + ảnh té ngã tới người thân qua Telegram Bot API

Cấu hình .env:
  TELEGRAM_BOT_TOKEN=<your_bot_token>
  TELEGRAM_CHAT_ID=<your_chat_id>
"""

import os
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────
logger = logging.getLogger("fall_detection")

# ─────────────────────────────────────────────
# NẠP BIẾN MÔI TRƯỜNG
# ─────────────────────────────────────────────
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")
TELEGRAM_API_BASE  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


# ─────────────────────────────────────────────
# KIỂM TRA CẤU HÌNH
# ─────────────────────────────────────────────
def _check_config() -> bool:
    if not TELEGRAM_BOT_TOKEN:
        logger.error("SENDER | TELEGRAM_BOT_TOKEN chưa được đặt trong .env")
        return False
    if not TELEGRAM_CHAT_ID:
        logger.error("SENDER | TELEGRAM_CHAT_ID chưa được đặt trong .env")
        return False
    return True


# ─────────────────────────────────────────────
# GỬI TIN NHẮN VĂN BẢN
# ─────────────────────────────────────────────
def _send_message(text: str) -> bool:
    url     = f"{TELEGRAM_API_BASE}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("SENDER | Đã gửi tin nhắn văn bản Telegram.")
        return True
    except requests.RequestException as e:
        logger.error(f"SENDER | Gửi tin nhắn thất bại: {e}")
        return False


# ─────────────────────────────────────────────
# GỬI ẢNH KÈM CAPTION
# ─────────────────────────────────────────────
def _send_photo(image_path: str, caption: str) -> bool:
    url = f"{TELEGRAM_API_BASE}/sendPhoto"

    if not os.path.isfile(image_path):
        logger.warning(f"SENDER | Không tìm thấy ảnh: {image_path} → chỉ gửi text")
        return False

    try:
        with open(image_path, "rb") as photo_file:
            resp = requests.post(
                url,
                data={
                    "chat_id":    TELEGRAM_CHAT_ID,
                    "caption":    caption,
                    "parse_mode": "HTML",
                },
                files={"photo": photo_file},
                timeout=20,
            )
        resp.raise_for_status()
        logger.info("SENDER | Đã gửi ảnh té ngã lên Telegram thành công.")
        return True
    except requests.RequestException as e:
        logger.error(f"SENDER | Gửi ảnh thất bại: {e}")
        return False


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def send_alert(image_path: str, reason: str = "không có phản hồi") -> None:
    """
    Gửi cảnh báo té ngã qua Telegram: ảnh + caption.

    Args:
        image_path : đường dẫn ảnh té ngã (PNG/JPG)
        reason     : lý do cảnh báo
    """
    if not _check_config():
        return

    timestamp = datetime.now().strftime("%H:%M:%S ngày %d/%m/%Y")
    caption   = (
        f"⚠️ <b>Phát hiện té ngã lúc {timestamp}</b>\n"
        f"📋 Lý do: {reason}\n"
        f"🆘 Vui lòng kiểm tra ngay!"
    )

    logger.info(f"SENDER | Đang gửi cảnh báo Telegram | lý do: {reason}")
    success = _send_photo(image_path, caption)

    if not success:
        fallback_text = (
            f"⚠️ <b>Phát hiện té ngã lúc {timestamp}</b>\n"
            f"📋 Lý do: {reason}\n"
            f"🖼️ (Không gửi được ảnh)\n"
            f"🆘 Vui lòng kiểm tra ngay!"
        )
        success_text = _send_message(fallback_text)
        status = "thành công (text fallback)" if success_text else "thất bại"
        logger.info(f"SENDER | Gửi fallback text: {status}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    send_alert("fall_snapshots/fall_test.png", reason="không có phản hồi")
