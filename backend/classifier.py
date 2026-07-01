"""
classifier.py — Voice Interaction & NLP Classification
=======================================================
Input : fall_image_path (str) — đường dẫn ảnh té ngã từ detector.py
Output:
  - Nếu "safe"        → TTS "Vâng, hãy cẩn thận"
  - Nếu "danger"      → gọi sender.send_alert(fall_image_path)
  - Nếu "no_response" → gọi sender.send_alert(fall_image_path)

Pipeline:
  1. TTS hỏi "Bạn có ổn không?"
  2. STT lắng nghe 10 giây
  3. NLP classify → "safe" | "danger" | "no_response"
  4. Xử lý theo kết quả + ghi log
"""

import logging
import numpy as np
import pyttsx3
import speech_recognition as sr
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# LOGGER (dùng chung logger của project)
# ─────────────────────────────────────────────
logger = logging.getLogger("fall_detection")

# ─────────────────────────────────────────────
# KHỞI TẠO NLP MODEL
# ─────────────────────────────────────────────
_nlp_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

SAFE_PHRASES = [
    "tôi ổn", "không sao", "tôi không sao", "bình thường",
    "ổn rồi", "không cần giúp", "tôi tự đứng được",
]

DANGER_PHRASES = [
    "giúp tôi", "cứu tôi", "tôi bị ngã", "đau quá",
    "không đứng được", "gọi giúp", "cần giúp đỡ",
]

_safe_embeddings   = _nlp_model.encode(SAFE_PHRASES)
_danger_embeddings = _nlp_model.encode(DANGER_PHRASES)

NLP_THRESHOLD = 0.45


# ─────────────────────────────────────────────
# TTS — TEXT TO SPEECH
# ─────────────────────────────────────────────
def speak(text: str) -> None:
    """Đọc văn bản thành giọng nói."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.warning(f"TTS lỗi: {e}")


# ─────────────────────────────────────────────
# STT — SPEECH TO TEXT
# ─────────────────────────────────────────────
def listen(timeout: int = 10) -> str:
    """
    Lắng nghe microphone vật lý và trả về văn bản (tiếng Việt).
    Trả về chuỗi rỗng nếu timeout hoặc không nhận ra giọng nói.
    """
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        logger.error("PyAudio chưa được cài. Chạy: pip install pyaudio")
        return ""

    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logger.info("CLASSIFIER | Đang lắng nghe ... (nói trong vòng 10 giây)")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=timeout, phrase_time_limit=8)

        text   = r.recognize_google(audio, language="vi-VN")
        result = text.lower().strip()
        logger.info(f"CLASSIFIER | STT nhận được: '{result}'")
        return result

    except sr.WaitTimeoutError:
        logger.info("CLASSIFIER | Không có phản hồi (timeout 10s)")
        return ""
    except sr.UnknownValueError:
        logger.info("CLASSIFIER | Không nhận ra giọng nói")
        return ""
    except sr.RequestError as e:
        logger.warning(f"CLASSIFIER | Google STT lỗi kết nối: {e}")
        return ""
    except OSError as e:
        logger.warning(f"CLASSIFIER | Không mở được microphone: {e}")
        return ""
    except Exception as e:
        logger.warning(f"CLASSIFIER | STT lỗi không xác định: {e}")
        return ""


# ─────────────────────────────────────────────
# NLP CLASSIFY
# ─────────────────────────────────────────────
def classify_response(text: str) -> str:
    """
    Phân loại phản hồi của người dùng.

    Returns:
        "safe"        — người dùng cho biết họ ổn
        "danger"      — người dùng cần giúp đỡ
        "no_response" — không có phản hồi hoặc không rõ ý
    """
    if not text or len(text.strip()) < 2:
        return "no_response"

    emb          = _nlp_model.encode([text])
    safe_score   = float(np.max(np.dot(_safe_embeddings,   emb.T)))
    danger_score = float(np.max(np.dot(_danger_embeddings, emb.T)))

    logger.info(f"CLASSIFIER | NLP scores — safe: {safe_score:.3f} | danger: {danger_score:.3f}")

    if safe_score > danger_score and safe_score > NLP_THRESHOLD:
        return "safe"
    elif danger_score > NLP_THRESHOLD:
        return "danger"
    else:
        return "no_response"


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def run_classifier(fall_image_path: str) -> None:
    """
    Luồng chính:
      1. TTS hỏi thăm
      2. STT lắng nghe
      3. NLP classify
      4. Phản hồi phù hợp + ghi log

    Args:
        fall_image_path: đường dẫn ảnh té ngã để gửi kèm nếu cần cảnh báo
    """
    logger.info("CLASSIFIER | Bắt đầu xử lý phản hồi sau té ngã ...")

    # Bước 1 — hỏi thăm
    speak("Bạn có ổn không?")

    # Bước 2 — lắng nghe
    response_text = listen(timeout=10)

    # Bước 3 — phân loại
    result = classify_response(response_text)
    logger.info(f"CLASSIFIER | Kết quả phân loại: {result} | Câu trả lời: '{response_text}'")

    # Bước 4 — xử lý
    if result == "safe":
        speak("Vâng, hãy cẩn thận")
        logger.info("CLASSIFIER | Người dùng ổn. Không gửi cảnh báo.")
    else:
        reason = "có yêu cầu giúp đỡ" if result == "danger" else "không có phản hồi"
        logger.info(f"CLASSIFIER | {reason.capitalize()} → Gửi cảnh báo Telegram ...")
        try:
            from sender import send_alert
            send_alert(fall_image_path, reason=reason)
        except Exception as e:
            logger.error(f"CLASSIFIER | Không gửi được cảnh báo: {e}")


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # Setup logger standalone
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_classifier("fall_snapshots/fall_test.png")
