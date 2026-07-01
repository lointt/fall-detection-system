# Fall Detection System

A real-time fall detection system using YOLOv8-Pose + TCN, with a FastAPI backend and a web-based frontend (live view + history playback).

## Project Structure

```
fall_detection_system/
├── backend/
│   ├── fall_snapshots/       # Auto-created: saved images when a fall is detected
│   ├── logs/                 # Auto-created: application logs
│   ├── models/                # YOLOv8-Pose + TCN model files (required)
│   ├── predicted_videos/     # Auto-created: recorded video segments
│   ├── .env                  # Telegram bot credentials (see Configuration below)
│   ├── classifier.py
│   ├── detector.py
│   ├── main.py                # FastAPI entry point
│   └── sender.py
├── frontend/
│   ├── history.css / history.html / history.js
│   └── index.css / index.html / index.js
└── requirements.txt
```

## Prerequisites

- Python 3.10+ installed
- A webcam connected to your machine
- Git installed

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/fall_detection_system.git
cd fall_detection_system
```

> Replace `<your-username>/fall_detection_system` with the actual repository URL.

## 2. Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Once activated, your terminal prompt should be prefixed with `(venv)`.

## 3. Install dependencies

With the virtual environment active, install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 4. Configuration

The backend sends fall alerts via Telegram. Create/edit the `.env` file inside `backend/` with your bot credentials:

```
TELEGRAM_BOT_TOKEN=<your_bot_token>
TELEGRAM_CHAT_ID=<your_chat_id>
```

Also make sure the `models/` folder inside `backend/` contains the required model files:
- `yolov8n-pose.pt`
- `tcn_model.onnx`

## 5. Run the application

Navigate into the `backend/` folder (where `main.py` is located) and run the server:

```bash
cd backend
python main.py
```

The server will start at `http://localhost:8000`.

## 6. Open the app

Open your browser and go to:

- **Live view:** `http://localhost:8000/`
- **History view:** `http://localhost:8000/history`

## Notes

- The `fall_snapshots/`, `logs/`, and `predicted_videos/` folders inside `backend/` are created automatically on first run if they don't already exist.
- To stop the server, press `Ctrl + C` in the terminal.
- To deactivate the virtual environment when you're done, run `deactivate`.
