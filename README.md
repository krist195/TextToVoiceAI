<h1 align="center">Text → Voice (XTTS-v2)</h1>
<p align="center">
  One-page web UI for Coqui XTTS-v2 with progress ring, ETA, voice cloning, and WAV export.
  <p align="center">
  A minimal Flask web app for multi-lingual TTS using Coqui XTTS-v2, with block-wise synthesis, a live progress ring, ETA, and WAV export.
  <br/>
  <sub>🇬🇧 English below · 🇷🇺 Русский — внизу</sub>


## Features
- **Coqui XTTS-v2** multilingual TTS (ru/en/de/es/fr by default)
- **Reference voice cloning** (upload WAV/MP3 or pick a recent file)
- **Smart block splitting** with configurable size and pause
- **Live progress ring**: percent, blocks, elapsed, ETA
- **Dark/Light** theme toggle
- One-click **WAV** download



## Pack offline EXE
1) Build-machine dependencies (Python 3.11, Windows PowerShell)
```sh
py -3.11 -m pip install --upgrade pip wheel setuptools
py -3.11 -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
py -3.11 -m pip install coqui-tts==0.27.0 flask pydub soundfile pyinstaller
py -3.11 -m pip install transformers huggingface_hub safetensors sentencepiece tokenizers regex tqdm inflect typeguard packaging filelock pyyaml requests
```
2)Download XTTS-v2 snapshot into assets/
```sh
py -3.11 -c "from huggingface_hub import snapshot_download; snapshot_download('coqui/XTTS-v2', local_dir_use_symlinks=False)"
```
```sh
$SRC = (py -3.11 -c "from huggingface_hub import snapshot_download; import pathlib; p=pathlib.Path(snapshot_download('coqui/XTTS-v2', local_dir_use_symlinks=False)); print(p.parents[1])").Trim()
New-Item -ItemType Directory -Force -Path .\assets\hf\models--coqui-ai--XTTS-v2 | Out-Null
robocopy $SRC .\assets\hf\models--coqui-ai--XTTS-v2 /E
```
3) Put FFmpeg
Download a static x64 [ffmpeg.exe](https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z) → place at:

   **tools\ffmpeg.exe**

4) Build (single command)
```sh
py -3.11 -m PyInstaller synth.py `
  --name Text2Voice `
  --onefile --console --clean `
  --runtime-hook rt_bootstrap.py `
  --add-binary "tools\ffmpeg.exe;." `
  --add-data "assets\hf\models--coqui-ai--XTTS-v2;assets\hf\models--coqui-ai--XTTS-v2" `
  --collect-all torch `
  --collect-all torchaudio `
  --collect-all TTS `
  --collect-all transformers `
  --collect-all huggingface_hub `
  --collect-all tqdm `
  --collect-all safetensors `
  --collect-all sentencepiece `
  --collect-all tokenizers `
  --collect-all regex `
  --collect-all inflect `
  --collect-all typeguard `
  --collect-data gruut `
  --copy-metadata tqdm `
  --copy-metadata transformers `
  --copy-metadata huggingface_hub `
  --copy-metadata packaging `
  --copy-metadata regex `
  --copy-metadata tokenizers `
  --copy-metadata filelock `
  --copy-metadata pyyaml `
  --copy-metadata requests
```

## Project layout after
```text

├─ synth.py                 # the app (unchanged)
├─ rt_bootstrap.py          # runtime hook for PyInstaller (ffmpeg + offline model)
├─ assets/
│  └─ hf/
│     └─ models--coqui-ai--XTTS-v2/    # HF snapshot (snapshots/, refs/, etc.)
├─ tools/
│  └─ ffmpeg.exe            # static x64 ffmpeg (bundled)
└─ dist/
   └─ Text2Voice.exe        # final single-file build
```

## Usage

_**1. Run Text2Voice.exe → browser at http://127.0.0.1:5000**_

_**2. Paste text, choose language, set block size/pause, optional normalization**_

_**3. Upload reference voice (WAV/MP3) or pick a recent one**_

_**4. Click Synthesize → watch progress ring → download final WAV**_


