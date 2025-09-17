# rt_bootstrap.py
import os, sys, shutil
from pathlib import Path

MEI = Path(getattr(sys, "_MEIPASS", Path.cwd()))

# 1) глушим typeguard (иначе он лезет в исходники и падает в onefile)
os.environ.setdefault("TYPEGUARD_DISABLE", "1")

# 2) pydub -> наш встроенный ffmpeg.exe (лежит в MEI)
ffmpeg = MEI / "ffmpeg.exe"
if ffmpeg.exists():
    os.environ["FFMPEG_BINARY"] = str(ffmpeg)
    try:
        from pydub import AudioSegment
        AudioSegment.converter = str(ffmpeg)
        AudioSegment.ffmpeg = str(ffmpeg)
    except Exception:
        pass

# 3) готовим оффлайн HuggingFace (копируем модель из бандла в пользовательский кэш)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
src_hf = MEI / "assets" / "hf" / "models--coqui-ai--XTTS-v2"
dst_root = Path.home() / ".cache" / "huggingface" / "hub"
dst = dst_root / "models--coqui-ai--XTTS-v2"
try:
    if src_hf.exists():
        dst_root.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or not any(dst.rglob("*")):
            shutil.copytree(src_hf, dst, dirs_exist_ok=True)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(dst_root))
except Exception as e:
    print(f"[bootstrap] HF cache copy skipped: {e}")
