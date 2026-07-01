/**
 * index.js — Live webcam feed + fall detection status
 */

const STREAM_URL = 'http://localhost:8000/video_feed';
const STATUS_WS  = 'ws://localhost:8000/ws/status';

// DOM refs
const liveStream         = document.getElementById('liveStream');
const thumbStream        = document.getElementById('thumbStream');
const thumbOffline       = document.getElementById('thumbOffline');
const feedOffline        = document.getElementById('feedOffline');
const fallBadge          = document.getElementById('fallBadge');
const detectionIndicator = document.getElementById('detectionIndicator');
const detectionDot       = document.getElementById('detectionDot');
const liveDateEl         = document.getElementById('liveDate');
const liveTimeEl         = document.getElementById('liveTime');
const btnFullscreen      = document.getElementById('btnFullscreen');
const btnMute            = document.getElementById('btnMute');
const iconSoundOn        = document.getElementById('iconSoundOn');
const iconSoundOff       = document.getElementById('iconSoundOff');
const feedContainer      = document.getElementById('feedContainer');

// ── Clock ─────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const pad = n => String(n).padStart(2, '0');
  liveDateEl.textContent = `${now.getFullYear()} - ${pad(now.getMonth()+1)} - ${pad(now.getDate())}`;
  liveTimeEl.textContent = `${pad(now.getHours())} : ${pad(now.getMinutes())} : ${pad(now.getSeconds())}`;
}
updateClock();
setInterval(updateClock, 1000);

// ── MJPEG stream ──────────────────────────────────────
let streamRetryTimer = null;

function startStream() {
  const url = STREAM_URL + '?t=' + Date.now();

  // Main feed
  liveStream.src = url;
  liveStream.onload = onStreamOk;
  liveStream.onerror = onStreamErr;

  // Mini thumb — cùng nguồn
  thumbStream.src = url;
  thumbStream.onload = () => {
    thumbOffline.style.display = 'none';
    thumbStream.style.display  = 'block';
  };
  thumbStream.onerror = () => {
    thumbOffline.style.display = 'block';
    thumbStream.style.display  = 'none';
  };
}

function onStreamOk() {
  feedOffline.style.display = 'none';
  liveStream.style.display  = 'block';
}

function onStreamErr() {
  feedOffline.style.display = 'flex';
  liveStream.style.display  = 'none';
  clearTimeout(streamRetryTimer);
  streamRetryTimer = setTimeout(startStream, 3000);
}

startStream();

// ── Mute toggle ───────────────────────────────────────
let muted = false;
btnMute.addEventListener('click', () => {
  muted = !muted;
  iconSoundOn.style.display  = muted ? 'none'  : 'block';
  iconSoundOff.style.display = muted ? 'block' : 'none';
});

// ── Fullscreen ────────────────────────────────────────
btnFullscreen.addEventListener('click', () => {
  if (!document.fullscreenElement) {
    feedContainer.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen();
  }
});

// ── WebSocket status ──────────────────────────────────
let ws;
function connectWS() {
  ws = new WebSocket(STATUS_WS);
  ws.onmessage = ev => {
    try { updateDetection(JSON.parse(ev.data)); } catch {}
  };
  ws.onclose = () => setTimeout(connectWS, 3000);
  ws.onerror = () => ws.close();
}
connectWS();

function updateDetection({ is_fall }) {
  fallBadge.style.display = is_fall ? 'block' : 'none';
  detectionIndicator.classList.toggle('is-fall', !!is_fall);
}
