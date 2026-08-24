/**
 * history.js — Xem lại video lịch sử
 *
 * Logic:
 *  - Chọn ngày → GET /api/history?date=YYYY-MM-DD
 *    → backend trả danh sách các segment video (~5 phút/segment) đã ghi
 *      xong trong ngày đó (segment đang ghi dở bị backend tự loại trừ
 *      vì file mp4 chưa flush xong metadata, mở lên sẽ lỗi)
 *  - Click session → phát video tương ứng từ /api/video/<filename>
 *  - Vì video được chia segment ngắn, đoạn vài phút gần nhất luôn
 *    xuất hiện trong danh sách chỉ sau tối đa ~5 phút kể từ lúc quay
 *    (không cần tắt detector mới xem được)
 *
 * Timeline:
 *  - Thanh ngang màu xanh: kéo trái = tua về trước, kéo phải = xem cũ hơn
 *  - Red dots = vị trí té ngã trong video
 */

const API = 'http://localhost:8000';

// DOM
const histDateEl     = document.getElementById('histDate');
const histTimeEl     = document.getElementById('histTime');
const histVideo      = document.getElementById('histVideo');
const feedPlaceholder= document.getElementById('feedPlaceholder');
const feedMeta       = document.getElementById('feedMeta');
const fallBadge      = document.getElementById('fallBadge');
const sessionList    = document.getElementById('sessionList');
const timelineFill   = document.getElementById('timelineFill');
const timelineRange  = document.getElementById('timelineRange');
const tlCurrent      = document.getElementById('tlCurrent');
const tlTotal        = document.getElementById('tlTotal');
const btnPlay        = document.getElementById('btnPlay');
const iconPlay       = document.getElementById('iconPlay');
const iconPause      = document.getElementById('iconPause');
const btnSpeed       = document.getElementById('btnSpeed');
const btnMute        = document.getElementById('btnMute');
const iconSoundOn    = document.getElementById('iconSoundOn');
const iconSoundOff   = document.getElementById('iconSoundOff');
const btnFullscreen  = document.getElementById('btnFullscreen');
const btnCalendar    = document.getElementById('btnCalendar');
const datePicker     = document.getElementById('datePicker');
const dateStrip      = document.getElementById('dateStrip');

// State
let selectedDate = todayStr();
let fallTimes    = [];
let speeds       = [1, 1.5, 2, 0.5];
let speedIdx     = 0;
let muted        = false;

// ── Helpers ───────────────────────────────────────────
function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2,'0')}:${String(Math.floor(s%60)).padStart(2,'0')}`;
}

// ── Date strip (last 10 days) ─────────────────────────
const DAY_NAMES = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];

function buildDateStrip() {
  dateStrip.innerHTML = '';
  const today = new Date();
  for (let i = 9; i >= 0; i--) {
    const d  = new Date(today);
    d.setDate(today.getDate() - i);
    const ds = d.toISOString().slice(0,10);
    const pill = document.createElement('button');
    pill.className = 'date-pill' + (ds === selectedDate ? ' date-pill--active' : '');
    pill.dataset.date = ds;
    const mo  = d.getMonth()+1;
    const day = d.getDate();
    pill.innerHTML =
      `<span class="date-pill__day">${DAY_NAMES[d.getDay()]}</span>` +
      `<span class="date-pill__num">${mo}-${String(day).padStart(2,'0')}</span>`;
    pill.addEventListener('click', () => selectDate(ds));
    dateStrip.appendChild(pill);
  }
}

function selectDate(ds) {
  selectedDate = ds;
  dateStrip.querySelectorAll('.date-pill').forEach(p => {
    p.classList.toggle('date-pill--active', p.dataset.date === ds);
  });
  datePicker.value = ds;
  loadSessions(ds);
}

// ── Calendar ──────────────────────────────────────────
btnCalendar.addEventListener('click', () => {
  datePicker.value = selectedDate;
  datePicker.showPicker ? datePicker.showPicker() : datePicker.click();
});
datePicker.addEventListener('change', () => {
  if (datePicker.value) selectDate(datePicker.value);
});

// ── Load sessions ─────────────────────────────────────
async function loadSessions(ds) {
  sessionList.innerHTML = '<p class="session-empty">Đang tải…</p>';
  resetPlayer();
  try {
    const res  = await fetch(`${API}/api/history?date=${ds}`);
    const list = await res.json();
    if (!Array.isArray(list) || list.length === 0) {
      sessionList.innerHTML = '<p class="session-empty">Không có video</p>';
      return;
    }
    sessionList.innerHTML = '';
    list.forEach(item => {
      const d   = new Date(item.timestamp);
      const pad = n => String(n).padStart(2,'0');
      const ts  = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
      const el  = document.createElement('div');
      el.className = 'session-item';
      el.dataset.filename = item.filename;
      el.innerHTML =
        `<div class="session-item__time">${ts}</div>` +
        `<div class="session-item__label">${item.duration || '--:--'}</div>`;
      el.addEventListener('click', () => {
        document.querySelectorAll('.session-item').forEach(e => e.classList.remove('session-item--active'));
        el.classList.add('session-item--active');
        openVideo(item, d);
      });
      sessionList.appendChild(el);
    });
  } catch {
    sessionList.innerHTML = '<p class="session-empty">Lỗi kết nối</p>';
  }
}

// ── Open video ────────────────────────────────────────
async function openVideo(item, startDate) {
  const pad = n => String(n).padStart(2,'0');
  histDateEl.textContent =
    `${startDate.getFullYear()} - ${pad(startDate.getMonth()+1)} - ${pad(startDate.getDate())}`;
  histTimeEl.textContent =
    `${pad(startDate.getHours())} : ${pad(startDate.getMinutes())} : ${pad(startDate.getSeconds())}`;

  feedMeta.style.display       = 'flex';
  feedPlaceholder.style.display = 'none';

  histVideo.src = `${API}/api/video/${item.filename}?t=${Date.now()}`;
  histVideo.load();
  histVideo.play().catch(() => {});

  // Fall times
  try {
    const r = await fetch(`${API}/api/fall_times/${item.filename}`);
    fallTimes = await r.json();
  } catch { fallTimes = []; }
}

function resetPlayer() {
  histVideo.pause();
  histVideo.src = '';
  feedMeta.style.display        = 'none';
  feedPlaceholder.style.display = 'flex';
  fallBadge.style.display       = 'none';
  iconPlay.style.display  = 'block';
  iconPause.style.display = 'none';
  tlCurrent.textContent = '00:00';
  tlTotal.textContent   = '00:00';
  timelineFill.style.width = '0%';
  timelineRange.value      = 0;
  fallTimes = [];
}

// ── Video events ──────────────────────────────────────
histVideo.addEventListener('error', () => {
  feedPlaceholder.style.display = 'flex';
  feedPlaceholder.querySelector('p').textContent =
    'Không phát được video này — có thể đang ghi dở, thử lại sau ít phút.';
  feedMeta.style.display = 'none';
});

histVideo.addEventListener('loadedmetadata', () => {
  const dur = histVideo.duration;
  timelineRange.max = dur;
  tlTotal.textContent = fmtTime(dur);
});

histVideo.addEventListener('timeupdate', () => {
  const cur = histVideo.currentTime;
  const dur = histVideo.duration || 1;
  tlCurrent.textContent = fmtTime(cur);
  const pct = (cur / dur * 100).toFixed(2);
  timelineFill.style.width = pct + '%';
  timelineRange.value = cur;

  // Fall badge
  const isFall = fallTimes.some(ft => Math.abs(cur - ft) < 1.0);
  fallBadge.style.display = isFall ? 'block' : 'none';
});

histVideo.addEventListener('play',  () => { iconPlay.style.display='none';  iconPause.style.display='block'; });
histVideo.addEventListener('pause', () => { iconPlay.style.display='block'; iconPause.style.display='none';  });
histVideo.addEventListener('ended', () => { iconPlay.style.display='block'; iconPause.style.display='none';  });

// ── Timeline drag ─────────────────────────────────────
// Kéo phải = tua về trước (xem cũ hơn), kéo trái = nhảy tới (tua nhanh)
timelineRange.addEventListener('input', () => {
  histVideo.currentTime = parseFloat(timelineRange.value);
});

// ── Controls ──────────────────────────────────────────
btnPlay.addEventListener('click', () => {
  if (!histVideo.src) return;
  histVideo.paused ? histVideo.play() : histVideo.pause();
});

btnSpeed.addEventListener('click', () => {
  speedIdx = (speedIdx + 1) % speeds.length;
  const s = speeds[speedIdx];
  histVideo.playbackRate = s;
  btnSpeed.textContent = `x${s}`;
});

btnMute.addEventListener('click', () => {
  muted = !muted;
  histVideo.muted = muted;
  iconSoundOn.style.display  = muted ? 'none'  : 'block';
  iconSoundOff.style.display = muted ? 'block' : 'none';
});

btnFullscreen.addEventListener('click', () => {
  const c = document.getElementById('feedContainer');
  document.fullscreenElement ? document.exitFullscreen() : c.requestFullscreen().catch(()=>{});
});

// ── Init ──────────────────────────────────────────────
buildDateStrip();
loadSessions(selectedDate);
