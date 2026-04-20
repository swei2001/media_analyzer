# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
bash scripts/setup.sh          # Creates .venv, installs deps, checks ffmpeg
python scripts/download_models.py  # Downloads Qwen3-VL-2B-Instruct (~4 GB) and Whisper large-v3 (~3 GB)
```

## Running

```bash
python analyze.py <file_path> [options]
# Options: --config, --output-dir, --whisper-model, --device, --no-save, --quiet
```

Optional vLLM backend for batch processing:
```bash
bash scripts/start_vllm.sh
```

## Architecture

Single-file CLI (`analyze.py`) delegates to `MediaAnalyzer` in `analyzer/pipeline.py`, which dispatches based on media type and duration:

- **Images** → `VisionModel` directly
- **Short video** (≤180s) → `VisionModel` on native video
- **Long video** (>180s) → `MediaPreprocessor` extracts frames via ffmpeg → `VisionModel` on frame list + optional `AudioModel` transcript
- **Audio only** → `AudioModel` (Whisper) transcription → `VisionModel` reasoning on text

```
analyze.py
  └─ analyzer/pipeline.py  (MediaAnalyzer — orchestrator)
      ├─ analyzer/preprocessor.py  (MediaPreprocessor — ffmpeg wrapper)
      ├─ analyzer/vision.py        (VisionModel — Qwen3-VL-2B-Instruct inference)
      └─ analyzer/audio.py         (AudioModel — Whisper transcription)
```

**Config:** `config.yaml` is the source of truth for model names, device, thresholds, and output paths. CLI args override config values. Relative paths in config are resolved relative to the script location.

**Output:** Each run prints JSON to stdout and saves to `results/<filename>_result.json`. The JSON keys are in Chinese (`思考过程`, `事件`, `解读`).

**Hardware:** Auto-detects CUDA → MPS (Apple Silicon) → CPU. Supports 4-bit quantization for limited VRAM. `flash-attention-2` is optional for speed/memory gains.

## Rules for Claude Code

- **Do not run experimental or model-inference commands via local Bash.** This includes `python analyze.py`, `python scripts/download_models.py`, `bash scripts/start_vllm.sh`, and any command that trains, evaluates, or runs the pipeline. Instead, print the command to the conversation and let the user run it on their server.
