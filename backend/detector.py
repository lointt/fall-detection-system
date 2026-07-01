"""
detector.py — Fall Detection Pipeline (Webcam Real-time)
=========================================================
Input : Webcam laptop (cv2.VideoCapture(0))
Output:
  - MJPEG frames → streaming qua FastAPI
  - fall_snapshots/fall_YYYYMMDD_HHMMSS.png  (khi phát hiện té ngã)
  - predicted_videos/YYYYMMDD_HHMMSS.mp4     (ghi lại toàn bộ session)

Pipeline:
  1. Đọc từng frame từ webcam
  2. YOLOv8n-Pose → trích xuất 17 keypoints (x,y) → chuẩn hóa
  3. Gom thành sliding window (30 frames) → TCN dự đoán FALL / NORMAL
  4. Nếu FALL và classifier chưa chạy: lưu ảnh + spawn thread classifier
     - Thread classifier: TTS → STT 10s → NLP → Telegram nếu cần
     - Pipeline tiếp tục xử lý frame bình thường trong 10s đó
     - Mọi FALL trigger trong khi classifier đang chạy bị bỏ qua
  5. Encode frame thành JPEG → đưa vào queue cho FastAPI stream
  6. Ghi frame ra video file (predicted_videos/)
"""

import os
import cv2
import time
import queue
import threading
import logging
import numpy as np
from datetime import datetime
from collections import deque
from ultralytics import YOLO
import onnxruntime as ort

# ─────────────────────────────────────────────
# CẤU HÌNH ĐƯỜNG DẪN
# ─────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH     = os.path.join(BASE_DIR, "models",           "yolov8n-pose.pt")
TCN_MODEL_PATH      = os.path.join(BASE_DIR, "models",           "tcn_model.onnx")
SNAPSHOT_DIR        = os.path.join(BASE_DIR, "fall_snapshots")
PREDICTED_VIDEO_DIR = os.path.join(BASE_DIR, "predicted_videos")
LOG_PATH            = os.path.join(BASE_DIR, "logs",             "info.log")

# ─────────────────────────────────────────────
# THAM SỐ MÔ HÌNH
# ─────────────────────────────────────────────
WINDOW_SIZE         = 30      # số frame mỗi chuỗi đầu vào TCN
NUM_KP              = 17      # số keypoints COCO
NUM_FEATURES        = 34      # 17 kp × 2 (x,y)
FALL_THRESHOLD      = 0.8     # ngưỡng sigmoid để kết luận FALL
CONF_THRESHOLD      = 0.3     # ngưỡng confidence của YOLOv8
TRAIN_FPS           = 25      # FPS lúc training
WEBCAM_INDEX        = 0       # index webcam (0 = webcam mặc định)
FRAME_QUEUE_SIZE    = 5       # buffer MJPEG stream
SEGMENT_DURATION_SEC = 300    # 5 phút mỗi segment video
                              # (mp4v codec chỉ ghi moov atom khi release(),
                              #  nên phải chia nhỏ để xem lại được ngay)

# ── Tối ưu tốc độ pipeline real-time ──────────
YOLO_IMGSZ            = 320   # giảm từ mặc định 640 → YOLO infer nhanh hơn
                              # nhiều (vẫn đủ chính xác cho pose 1 người)
PROCESS_EVERY_N       = 2     # chỉ chạy YOLO+TCN mỗi N frame; frame còn lại
                              # dùng lại skeleton/nhãn gần nhất (vẫn stream
                              # và ghi video bình thường) → tăng FPS thực tế

# Chỉ số keypoints COCO
KP_LEFT_SHOULDER    = 5
KP_RIGHT_SHOULDER   = 6
KP_LEFT_HIP         = 11
KP_RIGHT_HIP        = 12
EPSILON             = 1e-6

# ─────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────
def _setup_logger() -> logging.Logger:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logger = logging.getLogger("fall_detection")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        # File handler
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

logger = _setup_logger()

# ─────────────────────────────────────────────
# GLOBAL STATE (dùng chung với FastAPI)
# ─────────────────────────────────────────────
frame_queue   = queue.Queue(maxsize=FRAME_QUEUE_SIZE)   # MJPEG frames
status_lock   = threading.Lock()
current_status = {
    "label":   "Starting...",
    "prob":    0.0,
    "is_fall": False,
}

# Path của video session đang ghi (để history xem real-time)
current_session_path: str = ""

# Danh sách callbacks để notify WebSocket clients
_status_callbacks: list = []

def register_status_callback(cb):
    _status_callbacks.append(cb)

def unregister_status_callback(cb):
    _status_callbacks.discard(cb) if hasattr(_status_callbacks, 'discard') else None
    try:
        _status_callbacks.remove(cb)
    except ValueError:
        pass

def _notify_status(data: dict):
    with status_lock:
        current_status.update(data)
    for cb in list(_status_callbacks):
        try:
            cb(data)
        except Exception:
            pass

# Stop flag
_stop_event = threading.Event()


# ─────────────────────────────────────────────
# CAPTURE THREAD (tách đọc webcam khỏi xử lý)
# ─────────────────────────────────────────────
# Trước đây cap.read() và YOLO+TCN chạy tuần tự trong CÙNG 1 vòng lặp:
# nếu 1 frame xử lý mất 300ms thì suốt 300ms đó vòng lặp không đọc thêm
# frame mới nào — webcam driver phải giữ frame cũ trong buffer nội bộ,
# khiến stream MJPEG bị "trễ" so với thực tế và tổng số frame xử lý
# được mỗi giây rất thấp.
# Giải pháp: 1 thread riêng liên tục cap.read() và chỉ giữ FRAME MỚI
# NHẤT (ghi đè, không xếp hàng đợi) → vòng lặp xử lý chính luôn lấy được
# ảnh mới nhất hiện có thay vì phải xếp hàng chờ frame cũ.
class _LatestFrameGrabber:
    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._ts = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="webcam-grabber")

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._ts = time.time()

    def read(self):
        """Trả về (ret, frame) của frame mới nhất hiện có."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

# ─────────────────────────────────────────────
# TIỀN XỬ LÝ KEYPOINTS
# ─────────────────────────────────────────────
def normalize_keypoints(kp_xy: np.ndarray) -> np.ndarray:
    kp            = kp_xy.copy().astype(np.float32)
    hip_mid       = (kp[KP_LEFT_HIP] + kp[KP_RIGHT_HIP]) / 2.0
    shoulder_vec  = kp[KP_LEFT_SHOULDER] - kp[KP_RIGHT_SHOULDER]
    shoulder_dist = np.linalg.norm(shoulder_vec) + EPSILON
    return (kp - hip_mid) / shoulder_dist


# ─────────────────────────────────────────────
# VẼ KẾT QUẢ
# ─────────────────────────────────────────────
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

def draw_skeleton(frame: np.ndarray, kp_raw: np.ndarray, color=(0,255,0)) -> None:
    for x, y in kp_raw:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    for a, b in COCO_SKELETON:
        xa, ya = kp_raw[a]; xb, yb = kp_raw[b]
        if xa > 0 and ya > 0 and xb > 0 and yb > 0:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)


def draw_overlay(frame: np.ndarray, label: str, prob: float,
                 is_fall: bool, frame_idx: int, fps: float) -> None:
    h, w  = frame.shape[:2]
    color = (0, 0, 220) if is_fall else (0, 200, 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, label,
                (15, 45), cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3, cv2.LINE_AA)
    cv2.putText(frame, f"Prob: {prob:.2f}",
                (w - 200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

    time_sec = frame_idx / fps if fps > 0 else 0
    cv2.putText(frame, f"Frame {frame_idx}  ({time_sec:.1f}s)",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    bar_x, bar_y, bar_w, bar_h = 15, h - 40, 250, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (80,80,80), -1)
    fill = int(bar_w * prob)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), color, -1)
    cv2.putText(frame, "FALL risk",
                (bar_x+bar_w+8, bar_y+bar_h-1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)


# ─────────────────────────────────────────────
# LOAD TCN MODEL (ONNX)
# ─────────────────────────────────────────────
def _load_tcn_model(model_path: str) -> ort.InferenceSession:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp  = session.get_inputs()[0].name
    out  = session.get_outputs()[0].name
    logger.info(f"ONNX model loaded | input: '{inp}' | output: '{out}'")
    return session


# ─────────────────────────────────────────────
# RESAMPLE WINDOW
# ─────────────────────────────────────────────
def resample_window(window: np.ndarray, source_fps: float,
                    target_fps: float = TRAIN_FPS) -> np.ndarray:
    if abs(source_fps - target_fps) < 0.5:
        return window
    src_len     = window.shape[0]
    src_indices = np.linspace(0, src_len - 1, src_len)
    dst_indices = np.linspace(0, src_len - 1, WINDOW_SIZE)
    resampled   = np.zeros((WINDOW_SIZE, window.shape[1]), dtype=np.float32)
    for feat in range(window.shape[1]):
        resampled[:, feat] = np.interp(dst_indices, src_indices, window[:, feat])
    return resampled


# ─────────────────────────────────────────────
# TẠO SEGMENT VIDEO MỚI
# ─────────────────────────────────────────────
# Thứ tự ưu tiên codec: avc1 (H.264) chuẩn nhất, hầu hết build OpenCV
# trên Windows không có sẵn H.264 encoder (do bản quyền), nên fallback
# dần xuống các codec khác mà Windows Media Player / browser còn đọc được.
# LƯU Ý QUAN TRỌNG: KHÔNG đưa "mp4v" vào danh sách này.
# mp4v (MPEG-4 Part 2) ghi "thành công" trên OpenCV/FFMPEG ở mọi OS —
# isOpened()=True, file có dữ liệu, thậm chí cv2.VideoCapture đọc lại
# được trên CHÍNH máy đó (vì cùng dùng FFMPEG backend để decode).
# Nhưng Windows Media Player / Movies & TV và phần lớn trình duyệt lại
# KHÔNG có decoder cho mp4v → mở file báo lỗi 0xC00D36C4. Vì việc tự
# probe bằng VideoCapture không phát hiện được lỗi tương thích này, ta
# loại trừ thẳng mp4v khỏi danh sách ứng viên thay vì dựa vào probe.
_CODEC_CANDIDATES = [
    ("avc1", ".mp4"),   # H.264 — chuẩn nhất, mọi browser/player đọc được
    ("H264", ".mp4"),   # tên FourCC khác của H.264 trên 1 số build OpenCV
    ("MJPG", ".avi"),   # Motion JPEG — luôn có sẵn, mọi player đọc được
                         # (file to hơn H.264 nhưng KHÔNG bao giờ lỗi codec)
]

_working_codec = None   # cache codec đã verify chạy được, tránh thử lại mỗi segment


def _probe_codec(fourcc_str: str, ext: str, width: int, height: int, fps: float) -> bool:
    """
    Thử ghi VÀ đọc lại thật 1 file test để xác nhận codec hoạt động
    end-to-end trên máy hiện tại.

    isOpened()=True hay file_size>0 KHÔNG đủ để kết luận codec dùng được:
    OpenCV (qua FFMPEG) có thể ghi "thành công" container mp4v trên Linux
    server, nhưng player trên Windows (Movies & TV, WMP) lại không có
    decoder cho mp4v và báo lỗi 0xC00D36C4 khi mở. Do đó phải verify
    bằng cách mở lại file vừa ghi bằng chính cv2.VideoCapture và đọc
    được ít nhất 1 frame hợp lệ.
    """
    test_path = os.path.join(PREDICTED_VIDEO_DIR, f"_codec_test{ext}")
    try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            return False

        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(5):
            writer.write(dummy)
        writer.release()

        if not (os.path.isfile(test_path) and os.path.getsize(test_path) > 0):
            return False

        # Đọc lại để chắc chắn codec decode được (không chỉ ghi được)
        cap = cv2.VideoCapture(test_path)
        if not cap.isOpened():
            cap.release()
            return False
        ret, frame = cap.read()
        cap.release()

        return bool(ret) and frame is not None
    except Exception:
        return False
    finally:
        try:
            if os.path.isfile(test_path):
                os.remove(test_path)
        except OSError:
            pass


def _new_writer(width: int, height: int, fps: float):
    """
    Tạo VideoWriter mới cho 1 segment, tự dò codec nào thực sự ghi
    được trên máy hiện tại (OpenCV build có thể thiếu encoder H.264,
    isOpened()=True không đảm bảo codec hoạt động đúng — phải probe
    bằng cách ghi thử frame thật).
    Trả về (writer, output_path).
    """
    global _working_codec

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if _working_codec is not None:
        fourcc_str, ext = _working_codec
        output_path = os.path.join(PREDICTED_VIDEO_DIR, f"{ts}{ext}")
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (width, height)
        )
        if writer.isOpened():
            logger.info(f"Bắt đầu segment video mới: {output_path}")
            return writer, output_path
        # Codec đã từng hoạt động nhưng giờ lỗi (hiếm) → dò lại từ đầu
        writer.release()
        logger.warning("Codec đã cache không còn hoạt động, dò lại từ đầu...")
        _working_codec = None

    for fourcc_str, ext in _CODEC_CANDIDATES:
        if not _probe_codec(fourcc_str, ext, width, height, fps):
            logger.warning(f"Codec '{fourcc_str}' không ghi được trên máy này, thử codec khác...")
            continue

        output_path = os.path.join(PREDICTED_VIDEO_DIR, f"{ts}{ext}")
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (width, height)
        )
        if writer.isOpened():
            _working_codec = (fourcc_str, ext)
            logger.info(f"Codec video được chọn: {fourcc_str} ({ext})")
            logger.info(f"Bắt đầu segment video mới: {output_path}")
            return writer, output_path
        writer.release()

    raise RuntimeError(
        "Không tìm thấy codec video nào khả dụng trên hệ thống này. "
        "Cài đặt K-Lite Codec Pack hoặc dùng OpenCV build có hỗ trợ H.264."
    )


# ─────────────────────────────────────────────
# LƯU ẢNH TÉ NGÃ
# ─────────────────────────────────────────────
def save_fall_snapshot(frame: np.ndarray) -> str:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(SNAPSHOT_DIR, f"fall_{ts}.png")
    cv2.imwrite(image_path, frame)
    logger.info(f"DETECTOR | Đã lưu ảnh té ngã: {image_path}")
    return image_path


# ─────────────────────────────────────────────
# PIPELINE CHÍNH (chạy trong thread riêng)
# ─────────────────────────────────────────────
def run_detection():
    """
    Vòng lặp chính: đọc webcam → YOLO → TCN → stream frames.
    Gọi hàm này trong một thread daemon từ main.py.
    """
    # ── Kiểm tra model files ──────────────────
    for path, name in [
        (YOLO_MODEL_PATH, "YOLOv8-Pose model"),
        (TCN_MODEL_PATH,  "TCN model"),
    ]:
        if not os.path.isfile(path):
            logger.error(f"Không tìm thấy {name}: {path}")
            return

    os.makedirs(PREDICTED_VIDEO_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # ── Nạp mô hình ──────────────────────────
    logger.info("Đang nạp YOLOv8-Pose ...")
    pose_model = YOLO(YOLO_MODEL_PATH)

    logger.info("Đang nạp TCN model ...")
    tcn_model  = _load_tcn_model(TCN_MODEL_PATH)

    # ── Mở webcam ─────────────────────────────
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        logger.error(f"Không mở được webcam (index={WEBCAM_INDEX})")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Webcam: {fps:.1f} FPS (danh nghĩa) | {width}×{height}")

    grabber = _LatestFrameGrabber(cap).start()

    # ── Khởi tạo VideoWriter segment đầu tiên ──
    # Lưu ý: fps danh nghĩa của webcam CHỈ dùng cho segment đầu tiên (chưa
    # có số liệu đo thực tế). Từ segment thứ 2 trở đi, fps sẽ được thay
    # bằng fps THỰC TẾ đo được ở segment trước đó (xem đoạn xoay segment
    # bên dưới) để thời lượng video khớp với thời gian ghi thật.
    writer, output_path = _new_writer(width, height, fps)
    segment_start_time   = time.time()
    segment_frame_count  = 0
    measured_fps         = fps

    # ── Sliding window buffer ─────────────────
    window_buf = deque(maxlen=WINDOW_SIZE)
    raw_kp_buf = deque(maxlen=WINDOW_SIZE)

    current_label = "Starting..."
    current_prob  = 0.0
    current_fall  = False

    # ── Classifier thread state ───────────────
    classifier_running = threading.Event()

    frame_idx = 0
    _stop_event.clear()

    logger.info("Bắt đầu xử lý webcam real-time ...")

    # Kết quả keypoints gần nhất, tái dùng cho các frame bị bỏ qua YOLO
    last_kp_raw  = np.zeros((NUM_KP, 2), dtype=np.float32)
    last_kp_norm = np.zeros((NUM_KP, 2), dtype=np.float32)
    # Dấu thời gian thực của WINDOW_SIZE frame gần nhất → đo fps thực tế
    loop_times = deque(maxlen=WINDOW_SIZE)

    while not _stop_event.is_set():
        ret, frame = grabber.read()
        if not ret:
            # Capture thread chưa có frame nào (mới khởi động) — chờ ngắn
            time.sleep(0.01)
            continue

        # ── 1. Trích xuất keypoints ────────────
        # Chỉ chạy YOLO (bước nặng nhất) mỗi PROCESS_EVERY_N frame để tăng
        # FPS thực tế; các frame còn lại tái dùng skeleton gần nhất — vẫn
        # đủ mượt cho mắt người và không ảnh hưởng độ chính xác TCN vì
        # cửa sổ trượt vẫn nhận đủ 30 mẫu, chỉ là vài mẫu liên tiếp giống
        # nhau thay vì suy luận lại từ đầu.
        run_pose = (frame_idx % PROCESS_EVERY_N == 0)

        kp_raw  = last_kp_raw
        kp_norm = last_kp_norm

        if run_pose:
            results = pose_model(frame, verbose=False, conf=CONF_THRESHOLD, imgsz=YOLO_IMGSZ)

            kp_raw  = np.zeros((NUM_KP, 2), dtype=np.float32)
            kp_norm = np.zeros((NUM_KP, 2), dtype=np.float32)

            if (results[0].keypoints is not None
                    and results[0].keypoints.xy is not None
                    and len(results[0].keypoints.xy) > 0):

                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 1:
                    areas  = (boxes.xywh[:, 2] * boxes.xywh[:, 3]).cpu().numpy()
                    best_i = int(np.argmax(areas))
                else:
                    best_i = 0

                kp_raw_candidate = results[0].keypoints.xy[best_i].cpu().numpy()

                # Đảm bảo có đúng NUM_KP keypoints (17) trước khi dùng
                if kp_raw_candidate.shape[0] == NUM_KP:
                    kp_raw = kp_raw_candidate
                    critical = [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
                                KP_LEFT_HIP, KP_RIGHT_HIP]
                    if not any((kp_raw[i] == 0).all() for i in critical):
                        kp_norm = normalize_keypoints(kp_raw)

            # Lưu lại để các frame bị skip (không chạy YOLO) dùng tạm
            last_kp_raw, last_kp_norm = kp_raw, kp_norm

        window_buf.append(kp_norm)
        raw_kp_buf.append(kp_raw)
        loop_times.append(time.time())

        # ── 2. Dự đoán khi đủ 30 frames ────────
        if len(window_buf) == WINDOW_SIZE:
            # FPS thực tế của WINDOW_SIZE frame gần nhất (không dùng fps
            # danh nghĩa webcam) — vì tốc độ vòng lặp thực tế bị chi phối
            # bởi thời gian YOLO+TCN xử lý, không phải bởi tốc độ webcam.
            # Dùng đúng khoảng thời gian thực giữa các mẫu để resample về
            # TRAIN_FPS chính xác hơn, tránh méo tốc độ chuyển động đưa
            # vào TCN.
            if len(loop_times) >= 2 and (loop_times[-1] - loop_times[0]) > 0:
                effective_fps = (len(loop_times) - 1) / (loop_times[-1] - loop_times[0])
            else:
                effective_fps = fps

            seq = np.array(window_buf, dtype=np.float32)
            seq = seq.reshape(WINDOW_SIZE, NUM_FEATURES)
            seq = resample_window(seq, source_fps=effective_fps)
            seq = seq.reshape(1, WINDOW_SIZE, NUM_FEATURES)

            input_name   = tcn_model.get_inputs()[0].name
            prob         = float(tcn_model.run(None, {input_name: seq})[0][0][0])
            current_prob = prob
            current_fall = prob >= FALL_THRESHOLD
            current_label = "[!] FALL DETECTED" if current_fall else "[OK] NORMAL"

            # Notify WebSocket clients
            _notify_status({
                "label":   current_label,
                "prob":    current_prob,
                "is_fall": current_fall,
            })

            # ── 3. Xử lý khi phát hiện té ngã ──
            if current_fall and not classifier_running.is_set():
                classifier_running.set()

                snapshot_frame = frame.copy()
                image_path     = save_fall_snapshot(snapshot_frame)

                # Log thời gian phát hiện
                logger.info(f"DETECTOR | TÉ NGÃ tại frame {frame_idx} "
                            f"(prob={current_prob:.4f}) | snapshot: {image_path}")

                def _classifier_thread(img_path: str) -> None:
                    try:
                        from classifier import run_classifier
                        run_classifier(img_path)
                    except Exception as e:
                        logger.warning(f"Không gọi được classifier: {e}")
                    finally:
                        classifier_running.clear()
                        logger.info("Classifier hoàn tất — sẵn sàng nhận FALL mới")

                t = threading.Thread(
                    target=_classifier_thread,
                    args=(image_path,),
                    daemon=True,
                )
                t.start()
                logger.info("Đã spawn classifier thread (pipeline tiếp tục)")

        # ── 4. Vẽ skeleton ────────────────────
        if len(raw_kp_buf) > 0:
            skel_color = (0, 0, 220) if current_fall else (0, 220, 0)
            draw_skeleton(frame, raw_kp_buf[-1], color=skel_color)

        # ── 5. Vẽ overlay ─────────────────────
        draw_overlay(frame, current_label, current_prob,
                     current_fall, frame_idx, fps)

        # ── 6. Ghi video output ───────────────
        writer.write(frame)
        segment_frame_count += 1
        # Cập nhật path session đang ghi (history có thể xem real-time)
        global current_session_path
        current_session_path = output_path

        # ── 6b. Xoay segment mới sau mỗi SEGMENT_DURATION_SEC ──
        # (mp4v chỉ ghi moov atom lúc release() — phải release định kỳ
        #  để các đoạn cũ luôn xem lại được ngay trong lúc detector
        #  vẫn đang chạy real-time)
        #
        # SỬA LỖI "video 5 phút chỉ dài 19s": nguyên nhân là VideoWriter
        # được tạo với fps DANH NGHĨA của webcam (vd 25-30fps), nhưng vì
        # YOLO+TCN chạy đồng bộ trên từng frame nên số frame THỰC SỰ ghi
        # được trong 300 giây đồng hồ thực tế thấp hơn nhiều (vd chỉ ~2fps
        # do suy luận chậm). Trình phát tính thời lượng = số_frame / fps
        # khai báo trong file → 300s thực tế nhưng chỉ có ít frame → video
        # bị "dồn" lại còn vài chục giây. Cách sửa: đo fps THỰC TẾ đạt được
        # trong segment vừa đóng (số frame đã ghi / thời gian thực đã trôi
        # qua) và dùng fps đo được đó cho segment kế tiếp, để thời lượng
        # video khớp với thời gian ghi thật.
        if time.time() - segment_start_time >= SEGMENT_DURATION_SEC:
            elapsed = time.time() - segment_start_time
            if segment_frame_count > 0 and elapsed > 0:
                measured_fps = segment_frame_count / elapsed
                # Kẹp trong khoảng hợp lý để tránh giá trị bất thường
                measured_fps = max(1.0, min(measured_fps, fps))

            writer.release()
            logger.info(
                f"Đã đóng segment: {output_path} | "
                f"{segment_frame_count} frame trong {elapsed:.1f}s "
                f"(fps thực đo: {measured_fps:.2f}, fps danh nghĩa webcam: {fps:.1f})"
            )
            writer, output_path = _new_writer(width, height, measured_fps)
            segment_start_time   = time.time()
            segment_frame_count  = 0
            current_session_path = output_path

        # ── 7. Encode JPEG → queue cho stream ─
        ret_enc, jpeg = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        if ret_enc:
            try:
                frame_queue.put_nowait(jpeg.tobytes())
            except queue.Full:
                # Drop oldest frame nếu queue đầy
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(jpeg.tobytes())
                except queue.Empty:
                    pass

        frame_idx += 1

    # ── Dọn dẹp ───────────────────────────────
    grabber.stop()
    cap.release()
    writer.release()
    logger.info(f"Detector dừng. Video đã lưu: {output_path}")


def stop_detection():
    """Gọi từ bên ngoài để dừng vòng lặp detection."""
    _stop_event.set()


def generate_frames():
    """
    Generator cho MJPEG streaming.
    FastAPI/Starlette dùng hàm này cho endpoint /video_feed.
    """
    while True:
        try:
            jpeg_bytes = frame_queue.get(timeout=2.0)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )
        except queue.Empty:
            # Gửi frame trống nếu chưa có dữ liệu
            continue
