# AGENT.md

本文件用于指导自动化编码代理在本仓库中工作。

## 项目概览

这是一个 Python 媒体分析 CLI，目标是对单个图片、视频或音频文件做多模态分析，并输出结构化 JSON。本项目只负责“功能一：单文件解析”，不负责“功能二：多模态文件集合解析”和“功能三：媒体文件真伪鉴别”。`功能接口定义.md` 中的功能二、功能三仅作为背景资料，不应在本项目中实现。

核心输出字段使用中文键：

```json
{
  "file": "example.mp4",
  "思考过程": "...",
  "事件": ["..."],
  "解读": "..."
}
```

## 代码结构

- `analyze.py`：CLI 入口，解析命令行参数，构造配置字典，实例化 `MediaAnalyzer`。
- `analyzer/pipeline.py`：主流程编排。按媒体类型分派到图片、视频或音频流程，并负责保存结果。
- `analyzer/preprocessor.py`：媒体类型识别、`ffprobe` 获取视频时长、`ffmpeg` 提帧和提取音频、临时文件清理。
- `analyzer/vision.py`：视觉/文本推理封装，使用 `AutoModelForImageTextToText`、`AutoProcessor` 和 `qwen-vl-utils`，并从模型输出中提取 JSON。
- `analyzer/audio.py`：Whisper ASR 封装，使用 Hugging Face `automatic-speech-recognition` pipeline。
- `scripts/setup.sh`：创建或复用 `media` conda 环境，安装 ffmpeg、PyTorch 和 Python 依赖。
- `scripts/download_models.py`：使用 ModelScope 批量下载模型到 `/data/models`。
- `scripts/start_vllm.sh`：激活 `media` conda 环境并启动可选 vLLM 服务。
- `vllm_client.py`：vLLM HTTP 客户端，支持普通文本对话，以及图片/视频/音频路径解析。
- `demo_media/`：示例媒体文件，已在 `.gitignore` 中忽略。

## 当前实现流程

`analyze.py` 调用 `analyzer.MediaAnalyzer`：

1. `MediaPreprocessor.detect_type()` 通过扩展名识别 `image`、`video` 或 `audio`。
2. 图片直接调用 `VisionModel.analyze_image()`。
3. 视频先用 `ffprobe` 获取时长，并可用 `ffmpeg` 提取音轨转写。
4. 短视频（默认 `<=180s`）调用 `VisionModel.analyze_video_native()`。
5. 长视频提帧后调用 `VisionModel.analyze_frames()`。
6. 音频先转写，再调用 `VisionModel.analyze_audio_text()` 对文本做结构化分析。
7. 默认保存到 `results/<stem>_result.json`，并在结束后清理 `tmp/<stem>/`。

## 常用命令

初始化环境：

```bash
bash scripts/setup.sh
conda activate media
```

本项目使用 `media` conda 环境，Python 版本为 3.12。运行代码、安装依赖或做轻量检查时应先 `conda activate media`，不要默认使用 `.venv`。

安装 PyTorch 时注意 `requirements.txt` 的注释：CUDA 12.8 版本需要先使用 PyTorch 官方 index 单独安装：

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

`scripts/setup.sh` 已经按这个顺序处理：先执行上面的 PyTorch 安装命令，再安装 `requirements.txt` 中剩余依赖。

下载模型：

```bash
python scripts/download_models.py
```

该脚本使用 ModelScope 下载到 `/data/models`，当前模型列表包括：

- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `openai-mirror/whisper-large-v3`

运行 CLI 的形态：

```bash
python analyze.py --continue \
  --vision-model /data/models/Qwen3-VL-2B-Instruct \
  --whisper-model /data/models/whisper-large-v3
python analyze.py <file_path> [options]
```

默认推荐使用 `--continue` 交互模式，因为它会加载一次模型后持续处理多个文件。

常见参数：

- `--vision-model`：视觉模型路径或 Hugging Face / 本地模型 ID。当前代码默认 `/data/models/Qwen3-VL-2B-Instruct`。
- `--whisper-model`：Whisper 模型路径或 Hugging Face / 本地模型 ID。当前代码默认 `/data/models/large-v3`，常用本地路径为 `/data/models/whisper-large-v3`。
- `--device`：`auto`、`cuda`、`cpu` 或 `mps`，当前 CLI 默认 `cuda`。
- `--dtype`：`bfloat16`、`float16` 或 `float32`。
- `--max-new-tokens`：当前默认 `16384`。
- `--short-video-threshold`、`--extract-fps`、`--max-frames`、`--max-pixels`：视频处理参数。
- `--language`、`--no-audio`：音频转写参数。
- `--output-dir`、`--no-save`、`--quiet`、`--tmp-dir`、`--no-cleanup`：输出和临时文件参数。

## 重要注意事项

- 不要在本地 Bash 中直接运行重型或实验性模型推理命令，包括：
  - `python analyze.py ...`
  - `python scripts/download_models.py`
  - `bash scripts/start_vllm.sh`
  - 任何训练、评测或实际加载大模型的命令
- 如需验证推理流程，给出要运行的命令，让用户在服务器上执行。
- 允许运行轻量级静态检查、语法检查或不加载模型的代码检查命令。
- 当前仓库根目录没有 `config.yaml`。`analyze.py` 使用命令行参数构造内联配置，不读取 `config.yaml`，CLI 也没有 `--config` 参数。
- `scripts/download_models.py` 依赖 `modelscope`，如果环境中缺少该包，需要在 `media` 环境中安装。
- `requirements.txt` 固定 `vllm==0.19.1`，匹配 `torch==2.10.0` / CUDA 12.8；不要用未固定版本的 `pip install vllm` 覆盖。
- `scripts/start_vllm.sh` 依赖 `media` 环境中已安装 `vllm`，不会自动安装 `vllm`。
- `scripts/start_vllm.sh` 默认模型为 `/data/models/Qwen3-VL-4B-Instruct`，默认监听 `127.0.0.1:8011`；也可通过第一个参数指定模型路径，通过 `HOST=` 和 `PORT=` 覆盖监听地址。
- `vllm_client.py` 默认连接 `http://127.0.0.1:8011/v1`，默认模型为 `/data/models/Qwen3-VL-4B-Instruct`，音频路径会先用本地 Whisper 转写再提交给 vLLM。
- `MediaPreprocessor.get_video_duration()` 没有检查 `ffprobe` 返回码；修改相关逻辑时应考虑异常处理。
- `VisionModel._load()` 将 `device_map` 直接设为配置中的 `device` 字符串；如调整设备逻辑，注意兼容 `auto`、`cuda`、`cpu`、`mps`。
- `AudioModel` 当前用 `transformers` ASR pipeline，而不是 `requirements.txt` 注释中的 faster-whisper 主路径。
- 不使用 4-bit / AWQ 自动量化路径；显存不足时优先降低 `--max-frames`、降低 `--max-pixels`、改用更小模型或使用 `float16`。

## 开发约定

- 保持输出 JSON 的中文键名不变，除非用户明确要求变更接口。
- 修改 pipeline 时同时考虑图片、短视频、长视频、纯音频四条路径。
- 与模型提示词相关的修改集中在 `analyzer/vision.py`。
- 与媒体预处理和临时文件相关的修改集中在 `analyzer/preprocessor.py`。
- 避免把生成结果、临时帧、模型文件或示例媒体加入版本控制；这些路径和大文件类型已在 `.gitignore` 中忽略。
- 如果新增测试，优先 mock 掉模型加载和 `ffmpeg`/`ffprobe` 调用，避免测试实际加载大模型或处理大文件。

## 已知待办方向

本项目待办仅限功能一范围：

- 对比更多模型，例如 8B 级模型。
- 统计图片、短视频、长视频解析耗时。
- 寻找偏政治事件的数据集并统计整体指标。
