# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Scope

This repository implements only **Function 1: single-file media analysis**. It analyzes one video, audio file, or image at a time and returns structured JSON with Chinese keys:

```json
{
  "file": "example.mp4",
  "思考过程": "...",
  "事件": ["..."],
  "解读": "..."
}
```

Do not implement Function 2 (multi-modal file collection analysis) or Function 3 (media authenticity detection) in this project. Mentions of those functions in `功能接口定义.md` are background only.

## Environment

Use the `media` conda environment. Do not assume `.venv`.

```bash
bash scripts/setup.sh
conda activate media
```

`scripts/setup.sh` creates or reuses the `media` conda environment with Python 3.12. It installs PyTorch first using the CUDA 12.8 command copied from `requirements.txt`, then installs the rest of `requirements.txt`.

## Models

Model downloads are handled by ModelScope:

```bash
python scripts/download_models.py
```

The script downloads these models to `/data/models`:

- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `openai-mirror/whisper-large-v3`

Recommended local paths after download:

- Vision: `/data/models/Qwen/Qwen3-VL-2B-Instruct`
- Whisper: `/data/models/openai-mirror/whisper-large-v3`

## Running

Prefer interactive mode by default because it loads models once and then handles multiple files:

```bash
python analyze.py --continue \
  --vision-model /data/models/Qwen/Qwen3-VL-2B-Instruct \
  --whisper-model /data/models/openai-mirror/whisper-large-v3
```

Single-file one-shot mode is also supported:

```bash
python analyze.py path/to/file.mp4
```

Common options:

- `--continue`: interactive mode.
- `--vision-model`: vision model path or model ID.
- `--whisper-model`: Whisper model path or model ID.
- `--device`: `auto`, `cuda`, `cpu`, or `mps`.
- `--dtype`: `bfloat16`, `float16`, or `float32`.
- `--no-audio`: skip video audio extraction.
- `--no-save`: print only, do not save result JSON.
- `--quiet`: reduce logs.

There is no `--config` argument and no active `config.yaml` entry point. `analyze.py` builds its config from CLI arguments.

## Optional vLLM

`scripts/start_vllm.sh` starts an optional vLLM service. It activates the `media` conda environment, defaults to `/data/models/Qwen/Qwen3-VL-2B-Instruct`, and supports a model path argument:

```bash
bash scripts/start_vllm.sh
bash scripts/start_vllm.sh /data/models/Qwen/Qwen3-VL-4B-Instruct
PORT=8001 bash scripts/start_vllm.sh
```

The script expects `vllm` to already be installed in the `media` environment. It does not install `vllm` automatically and does not use AWQ quantization.

`analyze.py` currently uses the transformers backend. Switching analysis to vLLM requires changing inference calls in `analyzer/vision.py`.

## Architecture

Single-file CLI (`analyze.py`) delegates to `MediaAnalyzer` in `analyzer/pipeline.py`, which dispatches based on media type and duration:

- Images -> `VisionModel.analyze_image()`
- Short videos (`<=180s`) -> `VisionModel.analyze_video_native()`
- Long videos (`>180s`) -> `MediaPreprocessor.extract_frames()` via ffmpeg -> `VisionModel.analyze_frames()`
- Audio only -> `AudioModel.transcribe()` -> `VisionModel.analyze_audio_text()`

```text
analyze.py
  └─ analyzer/pipeline.py  (MediaAnalyzer orchestrator)
      ├─ analyzer/preprocessor.py  (MediaPreprocessor: ffmpeg/ffprobe wrapper)
      ├─ analyzer/vision.py        (VisionModel: Qwen3-VL inference and JSON extraction)
      └─ analyzer/audio.py         (AudioModel: Whisper transcription)
```

Output is printed to stdout and, by default, saved to `results/<filename>_result.json`. Temporary frames/audio are stored under `tmp/` and cleaned up unless `--no-cleanup` is used.

## Rules for Claude Code

- Do not run heavy model commands locally. This includes `python analyze.py ...`, `python scripts/download_models.py`, `bash scripts/start_vllm.sh`, training, evaluation, or anything that loads/downloads large models.
- If runtime validation requires a heavy command, print the command for the user to run on their server.
- Lightweight static checks, syntax checks, file reads, and documentation updates are allowed.
- Preserve the Chinese output keys (`思考过程`, `事件`, `解读`) unless the user explicitly asks to change the interface.
- When adding tests, mock model loading and ffmpeg/ffprobe calls. Do not make tests load large models or process large media files.
- Keep generated outputs, temporary media, model files, and demo media out of version control.
