from __future__ import annotations

import os
import re
import uuid
import time
import wave
import threading
from pathlib import Path
from typing import List, Dict, Optional

from flask import (
    Flask, request, send_from_directory, render_template_string, jsonify, url_for
)
from pydub import AudioSegment
from TTS.api import TTS

DOCS_DIR   = Path.home() / "Documents" / "Text2Voice"
VOICE_DIR  = DOCS_DIR / "voices"
OUT_DIR    = DOCS_DIR / "outputs"
TMP_DIR    = DOCS_DIR / "tmp"
for p in (VOICE_DIR, OUT_DIR, TMP_DIR):
    p.mkdir(parents=True, exist_ok=True)

MODEL_DIR  = DOCS_DIR / "model" 
os.environ["TTS_CACHE_DIR"] = str(MODEL_DIR)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
WANT_GPU   = False

LANGS = {
    "ru": "Русский",
    "en": "English",
    "de": "Deutsch",
    "es": "Español",
    "fr": "Français",
}
DEFAULT_LANG = "ru"
RECENT_VOICES = 12

def safe_name(name: str) -> str:
    return Path(name).name.replace(" ", "_").replace("\\", "_").replace("/", "_")

def ensure_wav_24k_mono(src: Path) -> Path:
    """
    Гарантируем mono/24k WAV (быстро, если уже ок).
    """
    if src.suffix.lower() == ".wav":
        try:
            with wave.open(str(src), "rb") as w:
                if w.getnchannels() == 1 and w.getframerate() == 24000:
                    return src
        except wave.Error:
            pass
    dst = VOICE_DIR / f"{src.stem}_24k.wav"
    seg = AudioSegment.from_file(str(src))
    seg = seg.set_frame_rate(24000).set_channels(1)
    seg.export(str(dst), format="wav")
    return dst

def list_recent_voices(n: int = RECENT_VOICES) -> List[Path]:
    files = [p for p in VOICE_DIR.glob("*.*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:n]

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\…])\s+")
_WS = re.compile(r"\s+")

def normalize_text(txt: str, hard: bool = True) -> str:
    """
    Мягкая чистка текста: убираем мусорные пробелы, гарантируем точки.
    'hard=True' дополнительно схлопывает многоточия, тройные знаки и т.п.
    """
    t = txt.replace("\r", " ").replace("\n", " ").strip()
    t = _WS.sub(" ", t)
    if hard:
        t = re.sub(r"[!！]{2,}", "!", t)
        t = re.sub(r"[?？]{2,}", "?", t)
        t = re.sub(r"[.。]{3,}", "…", t)
        t = re.sub(r"[;]{2,}", ";", t)
    if t and t[-1] not in ".!?…":
        t += "."
    return t

def split_into_blocks(txt: str, max_len: int) -> List[str]:
    """
    Рубим по предложениям на блоки ≤ max_len, соблюдая границы.
    """
    sents = [s.strip() for s in _SENT_SPLIT.split(txt) if s.strip()]
    blocks: List[str] = []
    cur = ""
    for s in sents:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_len:
            cur = f"{cur} {s}"
        else:
            blocks.append(cur)
            cur = s
    if cur:
        blocks.append(cur)
    return blocks

print("[XTTS] loading model… (first run may take a minute)")
TTS_MODEL = TTS(model_name=MODEL_NAME, gpu=WANT_GPU)
print("[XTTS] ready ✓")


PROGRESS: Dict[str, Dict] = {}

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="ru" data-theme="dark">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Text → Voice · XTTS-v2</title>
<style>
:root{
  --bg:#0e0f13; --panel:#171923; --muted:#8a8fa3; --text:#e7eaf3; --accent:#34d399; --accent-dim:#1f6f55;
  --warn:#f59e0b; --err:#ef4444; --border:#232736;
}
:root[data-theme="light"]{
  --bg:#f7f8fc; --panel:#ffffff; --muted:#5b6070; --text:#12141a; --accent:#059669; --accent-dim:#146a55;
  --warn:#9a6b00; --err:#b91c1c; --border:#e8ebf2;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji";
  background:var(--bg); color:var(--text);
  display:flex; align-items:flex-start; justify-content:center; padding:24px;
}
.container{width:clamp(320px, 100%, 980px)}
.card{background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:20px; box-shadow:0 10px 30px rgba(0,0,0,.15)}
.row{display:grid; grid-template-columns:1.1fr .9fr; gap:20px}
@media (max-width: 860px){ .row{grid-template-columns:1fr} }

h1{margin:0 0 12px; font-size:24px; display:flex; align-items:center; gap:10px}
.sub{color:var(--muted); font-size:14px; margin-bottom:18px}

label{font-size:14px; color:var(--muted); display:block; margin:.25rem 0 .35rem}
textarea{width:100%; min-height:160px; resize:vertical; padding:12px 14px; border-radius:12px; border:1px solid var(--border);
  background:transparent; color:var(--text); font-size:15px; line-height:1.5}
select,input[type="number"],input[type="file"]{
  width:100%; background:transparent; color:var(--text); border:1px solid var(--border); border-radius:12px; padding:10px 12px; font-size:14px
}
.small{font-size:13px; color:var(--muted)}

.btn{
  display:inline-flex; align-items:center; gap:8px; background:var(--accent); color:#fff; border:none; border-radius:12px; padding:10px 14px;
  font-size:15px; cursor:pointer; transition:filter .15s ease
}
.btn.secondary{ background:transparent; color:var(--text); border:1px solid var(--border) }
.btn:disabled{opacity:.6; cursor:not-allowed}
.btn:hover{ filter:brightness(1.05) }

.grid{display:grid; gap:12px}
.panel{background:rgba(255,255,255,.02); border:1px dashed var(--border); border-radius:12px; padding:12px}

.kpis{display:flex; gap:16px; flex-wrap:wrap}
.kpis .k{background:rgba(255,255,255,.03); border:1px solid var(--border); border-radius:12px; padding:10px 12px; min-width:120px}
.k .t{color:var(--muted); font-size:12px}
.k .v{font-weight:600; font-size:16px}

.progressWrap{display:flex; gap:18px; align-items:center; justify-content:flex-start; margin:6px 0 4px}
.ring{position:relative; width:140px; height:140px}
svg{display:block}
.ring .pct{
  position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:20px
}
.ring .bar{transition:stroke-dashoffset .35s ease}

.audio{margin-top:14px; display:flex; align-items:center; gap:10px}
hr{border:none; border-top:1px solid var(--border); margin:14px 0}

.themeToggle{float:right}
.muted{color:var(--muted)}

.bad{color:var(--err)}
.warn{color:var(--warn)}
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h1>🗣️ Text → Voice (XTTS-v2) <button id="themeBtn" class="btn secondary themeToggle" type="button">Тема</button></h1>
    <div class="sub">Многоязычный синтез речи на основе эталонного голоса. Длинные тексты режутся на блоки и склеиваются с умными паузами.</div>

    <div class="row">
      <div class="grid">
        <div>
          <label>Текст</label>
          <textarea id="text" placeholder="Вставь длинный текст здесь…"></textarea>
        </div>

        <div class="grid" style="grid-template-columns: 1fr 1fr;">
          <div>
            <label>Язык</label>
            <select id="lang">
              {% for code,label in langs.items() %}
                <option value="{{code}}" {{'selected' if code==cur_lang else ''}}>{{label}}</option>
              {% endfor %}
            </select>
          </div>
          <div>
            <label>Размер блока (симв.)</label>
            <input id="block" type="number" min="160" max="900" step="20" value="360"/>
            <div class="small">Оптимально 280–420 для стабильности</div>
          </div>
        </div>

        <div class="grid">
          <div class="panel">
            <div class="grid" style="grid-template-columns: 1fr 1fr;">
              <div>
                <label>Пауза между блоками (мс)</label>
                <input id="pause" type="number" min="0" max="1000" step="10" value="120"/>
              </div>
              <div>
                <label>Нормализовать текст</label>
                <select id="norm">
                  <option value="1" selected>Да</option>
                  <option value="0">Нет</option>
                </select>
              </div>
            </div>
            <div class="small muted">Совет: оставляй естественную пунктуацию. Перед точкой — пробелов не нужно.</div>
          </div>
        </div>

        <div>
          <button id="runBtn" class="btn" type="button">Синтезировать</button>
        </div>
      </div>

      <div class="grid">
        <div class="panel">
          <label>Файл эталонного голоса (wav/mp3)</label>
          <input id="voice" type="file" accept="audio/*"/>

          {% if recent %}
          <div style="margin-top:10px">
            <label>…или выбрать недавний</label>
            <select id="recent">
              <option value="">— выбор —</option>
              {% for v in recent %}
                <option value="{{v.name}}">{{v.name}}</option>
              {% endfor %}
            </select>
          </div>
          {% endif %}
          <div class="small muted" style="margin-top:8px">
            Эталон лучше 15–60 сек, без шумов. Для наилучшего качества — 1–3 мин с разной интонацией.
          </div>
        </div>

        <div class="panel">
          <div class="progressWrap">
            <div class="ring">
              <svg width="140" height="140" viewBox="0 0 140 140">
                <circle cx="70" cy="70" r="58" stroke="rgba(255,255,255,0.09)" stroke-width="12" fill="none" />
                <circle id="bar" class="bar" cx="70" cy="70" r="58" stroke="var(--accent)" stroke-width="12" fill="none"
                        stroke-linecap="round" transform="rotate(-90 70 70)" />
              </svg>
              <div class="pct" id="pct">0%</div>
            </div>
            <div class="grid" style="gap:6px">
              <div class="kpis">
                <div class="k"><div class="t">Блоки</div><div class="v" id="blocks">0 / 0</div></div>
                <div class="k"><div class="t">Прошло</div><div class="v" id="elapsed">—</div></div>
                <div class="k"><div class="t">Осталось</div><div class="v" id="eta">—</div></div>
              </div>
              <div class="small muted" id="statusLine">Готов к работе.</div>
            </div>
          </div>
          <hr/>
          <div id="result" class="audio" style="display:none">
            <audio id="player" controls></audio>
            <a id="download" class="btn secondary" download>Скачать WAV</a>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const $ = (q)=>document.querySelector(q);
const R = 58, CIRC = 2*Math.PI*R;
const bar = $("#bar");
bar.setAttribute("stroke-dasharray", CIRC);
bar.setAttribute("stroke-dashoffset", CIRC);

function fmtSec(s){
  if (!isFinite(s)) return "—";
  s = Math.max(0, Math.round(s));
  const m = Math.floor(s/60), ss = String(s%60).padStart(2,'0');
  return m ? `${m} мин ${ss} с` : `${ss} с`;
}
function setProgress(p, done, total, eta, elapsed){
  p = Math.max(0, Math.min(1, p || 0));
  const off = Math.max(0, CIRC - CIRC*p);
  bar.setAttribute("stroke-dashoffset", off);
  $("#pct").textContent = Math.round(p*100) + "%";
  $("#blocks").textContent = `${done} / ${total}`;
  $("#elapsed").textContent = fmtSec(elapsed);
  $("#eta").textContent = fmtSec(eta);
}
function setReady(url){
  if (!url) return;
  $("#result").style.display = "flex";
  $("#player").src = url;
  $("#download").href = url;
}

let polling = null, jobId = null;

async function poll(){
  if (!jobId) return;
  try{
    const r = await fetch(`/progress/${jobId}`);
    if (!r.ok) throw new Error("progress http "+r.status);
    const j = await r.json();
    setProgress(j.progress, j.done, j.total, j.eta_sec, j.elapsed_sec);
    if (j.url){
      setReady(j.url);
      $("#statusLine").textContent = "Готово ✓";
      clearInterval(polling); polling = null; jobId = null;
      $("#runBtn").disabled = false;
    }
  }catch(e){
    console.error(e);
    $("#statusLine").textContent = "Ошибка связи с сервером";
  }
}

$("#runBtn").addEventListener("click", async ()=>{
  const text  = $("#text").value.trim();
  if (!text){ alert("Введите текст."); return; }
  const lang  = $("#lang").value;
  const block = +$("#block").value || 360;
  const pause = +$("#pause").value || 120;
  const norm  = $("#norm").value === "1";

  const fd = new FormData();
  fd.append("text", text);
  fd.append("lang", lang);
  fd.append("block", String(block));
  fd.append("pause", String(pause));
  fd.append("norm",  norm ? "1" : "0");

  const vfile = $("#voice").files[0];
  const recent = $("#recent") ? $("#recent").value : "";
  if (vfile) fd.append("voice_upload", vfile);
  else if (recent) fd.append("voice_choice", recent);

  $("#runBtn").disabled = true;
  $("#statusLine").textContent = "Запуск синтеза…";
  setProgress(0, 0, 0, Infinity, 0);
  $("#result").style.display = "none";

  try{
    const r = await fetch("/synthesize", {method:"POST", body: fd});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || ("http "+r.status));
    jobId = j.job_id;
    $("#statusLine").textContent = "Синтез идёт…";
    if (polling) clearInterval(polling);
    polling = setInterval(poll, 600);
  }catch(e){
    console.error(e);
    alert("Ошибка: " + e.message);
    $("#runBtn").disabled = false;
  }
});

// тема
$("#themeBtn").addEventListener("click", ()=>{
  const root = document.documentElement;
  const cur = root.getAttribute("data-theme") || "dark";
  root.setAttribute("data-theme", cur==="dark" ? "light":"dark");
});
</script>
</body>
</html>
"""

def do_synth(job_id: str, text: str, lang: str, voice_path: Path, block_len: int, pause_ms: int):
    """
    Фоновая сборка итогового WAV: блоки -> tts_to_file -> склейка pydub (+тихие паузы).
    Обновляет PROGRESS[job_id] на каждом шаге, чтобы фронт показывал проценты и ETA.
    """
    try:
        t0 = time.time()
        txt = normalize_text(text, hard=True)
        blocks = split_into_blocks(txt, max_len=block_len)
        PROGRESS[job_id].update(
            total_blocks=len(blocks),
            total_chars=sum(len(b) for b in blocks),
            job_started=t0
        )

        tmp_files: List[Path] = []
        for i, b in enumerate(blocks, 1):
            PROGRESS[job_id]["cur_block_len"] = len(b)
            PROGRESS[job_id]["cur_block_started"] = time.time()

            tmp_wav = TMP_DIR / f"{job_id}_{i:04d}.wav"
            # Сам tts:
            t1 = time.time()
            TTS_MODEL.tts_to_file(text=b, speaker_wav=str(voice_path), language=lang, file_path=str(tmp_wav))
            t2 = time.time()

            dt = max(0.001, t2 - t1)
            rate_now = len(b) / dt
            ema = PROGRESS[job_id].get("ema_rate", 18.0)
            PROGRESS[job_id]["ema_rate"] = 0.80 * ema + 0.20 * rate_now

            tmp_files.append(tmp_wav)
            PROGRESS[job_id]["done_blocks"] += 1
            PROGRESS[job_id]["chars_done"]  += len(b)
            PROGRESS[job_id]["cur_block_len"] = 0
            PROGRESS[job_id]["cur_block_started"] = None

        final = AudioSegment.silent(duration=0)
        pad = AudioSegment.silent(duration=max(0, int(pause_ms)))
        for p in tmp_files:
            seg = AudioSegment.from_file(str(p))
            final += seg
            if pad.duration_seconds > 0:
                final += pad

        out_path = OUT_DIR / f"{uuid.uuid4().hex}.wav"
        final = final.set_frame_rate(24000).set_channels(1)
        final.export(str(out_path), format="wav")

        
        PROGRESS[job_id]["url"] = out_path.name

        for p in tmp_files:
            try: p.unlink(missing_ok=True)
            except: pass

    except Exception as e:
        PROGRESS[job_id]["error"] = f"{e}"

@app.route("/")
def home():
    return render_template_string(HTML, langs=LANGS, cur_lang=DEFAULT_LANG, recent=list_recent_voices())

@app.route("/synthesize", methods=["POST"])
def synth_route():
    text = request.form.get("text", "").strip()
    lang = request.form.get("lang", DEFAULT_LANG)
    try:
        block_len = int(request.form.get("block", "360"))
        pause_ms  = int(request.form.get("pause", "120"))
        norm      = request.form.get("norm", "1") == "1"
    except:
        return jsonify({"error":"Неверные параметры"}), 400

    if not text:
        return jsonify({"error":"Пустой текст"}), 400
    if lang not in LANGS:
        return jsonify({"error":"Неверный язык"}), 400

    voice_path: Optional[Path] = None
    up = request.files.get("voice_upload")
    if up and up.filename:
        dest = VOICE_DIR / safe_name(up.filename)
        up.save(str(dest))
        voice_path = dest
    else:
        choice = request.form.get("voice_choice", "")
        if choice:
            cand = VOICE_DIR / safe_name(choice)
            if cand.exists():
                voice_path = cand

    if not voice_path:
        return jsonify({"error":"Загрузите или выберите эталонный голос"}), 400

    if norm:
        text = normalize_text(text, hard=True)
    try:
        voice_wav = ensure_wav_24k_mono(voice_path)
    except Exception as e:
        return jsonify({"error": f"Ошибка конвертации эталона: {e}"}), 500

    job_id = uuid.uuid4().hex
    PROGRESS[job_id] = dict(
        done_blocks=0, total_blocks=0,
        chars_done=0, total_chars=0,
        cur_block_len=0, cur_block_started=None,
        ema_rate=18.0, url=None, error=None,
        job_started=time.time()
    )

    th = threading.Thread(target=do_synth, args=(job_id, text, lang, voice_wav, block_len, pause_ms), daemon=True)
    th.start()

    return jsonify({"job_id": job_id})

@app.route("/progress/<job_id>")
def progress(job_id):
    p = PROGRESS.get(job_id)
    if not p:
        return jsonify({"error":"no such job"}), 404

    if p.get("error"):
        return jsonify({"error": p["error"]}), 500

    started = float(p.get("job_started") or time.time())
    elapsed = max(0.0, time.time() - started)

    est_chars = float(p.get("chars_done") or 0.0)
    rate = max(6.0, float(p.get("ema_rate") or 18.0))

    cur_len = int(p.get("cur_block_len") or 0)
    cur_started = p.get("cur_block_started")
    if cur_started and cur_len > 0:
        elapsed_cur = max(0.0, time.time() - float(cur_started))
        est_chars += min(cur_len, elapsed_cur * rate)

    total_chars = max(1.0, float(p.get("total_chars") or 1.0))
    prog = min(1.0, est_chars / total_chars)

    remain_chars = max(0.0, total_chars - est_chars)
    eta = remain_chars / rate

    url = None
    if p.get("url"):
        url = url_for("audio", fname=p["url"])

    return jsonify({
        "done": int(p.get("done_blocks", 0)),
        "total": int(p.get("total_blocks", 0)),
        "progress": float(prog),
        "elapsed_sec": float(elapsed),
        "eta_sec": float(eta),
        "url": url
    })

@app.route("/audio/<path:fname>")
def audio(fname):
    return send_from_directory(OUT_DIR, fname, as_attachment=False)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
