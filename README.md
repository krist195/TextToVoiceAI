# TextToVoiceAI
A program that converts text into MP3/WAV with the voice you give it


A minimal Flask web app for multi-lingual TTS using Coqui XTTS-v2, with block-wise synthesis, a live progress ring, ETA, and WAV export.


## Features
- **Coqui XTTS-v2** multilingual TTS (ru/en/de/es/fr by default)
- **Reference voice cloning** (upload WAV/MP3 or pick a recent file)
- **Smart block splitting** with configurable size and pause
- **Live progress ring**: percent, blocks, elapsed, ETA
- **Dark/Light** theme toggle
- One-click **WAV** download
