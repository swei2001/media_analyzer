# 媒体分析器 - 功能一：单文件解析

对单个视频、音频、图片文件进行深度分析，输出结构化 JSON，包括思考过程、事件和解读。

本项目只负责功能一：单文件解析。不负责功能二：多模态文件集合解析，也不负责功能三：媒体文件真伪鉴别。

---

## 目录结构

```text
media_analyzer/
├── analyze.py              # CLI 入口
├── requirements.txt        # Python 依赖
├── analyzer/
│   ├── pipeline.py         # 主流程编排
│   ├── vision.py           # Qwen3-VL 推理封装
│   ├── audio.py            # Whisper 转写封装
│   └── preprocessor.py     # ffmpeg/ffprobe 预处理工具
├── scripts/
│   ├── setup.sh            # conda 环境初始化
│   ├── download_models.py  # 使用 ModelScope 批量下载模型
│   └── start_vllm.sh       # 启动 vLLM 服务（可选）
├── vllm_client.py          # vLLM HTTP 客户端（对话/媒体解析）
├── results/                # 输出 JSON（自动创建）
└── tmp/                    # 临时帧/音轨（自动清理）
```

---

## 快速开始

### 第一步：初始化环境

```bash
cd /data/media_analyzer
bash scripts/setup.sh
conda activate media
```

`scripts/setup.sh` 会创建或复用 `media` conda 环境，Python 版本为 3.12，并安装 ffmpeg 和 Python 依赖。

PyTorch 会先按 `requirements.txt` 第 10 行的 CUDA 12.8 命令安装：

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

随后再安装 `requirements.txt` 中剩余依赖。当前 `requirements.txt` 固定 `vllm==0.19.1`，该版本匹配 `torch==2.10.0`、`torchvision==0.25.0`、`torchaudio==2.10.0` 和 CUDA 12.8；不要直接使用未固定版本的 `pip install vllm` 覆盖它。

### 第二步：下载模型

```bash
python scripts/download_models.py
```

当前脚本使用 ModelScope 下载模型到 `/data/models`。脚本会打印每个模型的实际保存路径：

| ModelScope 模型 ID | 用途 |
|------|------|
| `Qwen/Qwen3-VL-2B-Instruct` | 2B 视觉/视频/文本推理模型 |
| `Qwen/Qwen3-VL-4B-Instruct` | 更大视觉模型备选 |
| `Qwen/Qwen3-VL-8B-Instruct` | 更大视觉模型备选 |
| `openai-mirror/whisper-large-v3` | 音频转写 |

当前 CLI 代码默认使用的本地路径是 `/data/models/Qwen3-VL-2B-Instruct` 和 `/data/models/large-v3`。如果你的实际模型保存路径不同，请在命令行显式传入 `--vision-model` 和 `--whisper-model`。

### 第三步：运行分析

推荐默认使用交互模式：

```bash
python analyze.py --continue \
  --vision-model /data/models/Qwen3-VL-2B-Instruct \
  --whisper-model /data/models/whisper-large-v3
```

进入交互模式后，输入待分析文件路径：

```text
文件路径> /data/media_analyzer/demo_media/BigBuckBunny.mp4
文件路径> /data/media_analyzer/demo_media/people.jpg
文件路径> /data/media_analyzer/demo_media/speech_audio.mp3
```

交互模式会加载一次模型后持续处理多个文件，适合实际使用。输入 `q`、`quit`、`exit` 或按 `Ctrl+C` 退出。

单次分析也仍然可用：

```bash
python analyze.py path/to/file.mp4
```

---

## 输出格式

结果会打印到终端，并在默认情况下保存到 `results/<文件名>_result.json`：

```json
{
  "file": "example.mp4",
  "思考过程": "1. 逐帧分析...; 2. 音频分析...; 3. 综合判断...",
  "事件": [
    "0-8秒：...",
    "9-15秒：..."
  ],
  "解读": "..."
}
```

---

## 常用参数

```text
python analyze.py [file] [options]

位置参数:
  file                         待分析的媒体文件路径；交互模式下可省略

模式:
  --continue                   交互模式：加载一次模型后持续等待输入

模型:
  --vision-model MODEL         视觉模型路径或 Hugging Face / 本地模型 ID
                               默认: /data/models/Qwen3-VL-2B-Instruct
  --whisper-model MODEL        Whisper 模型路径或 Hugging Face / 本地模型 ID
                               默认: /data/models/large-v3
  --device DEVICE              auto/cuda/cpu/mps，默认: cuda
  --dtype DTYPE                bfloat16/float16/float32，默认: bfloat16
  --max-new-tokens N           最大生成 token 数，默认: 16384

视频:
  --short-video-threshold N    短视频阈值秒数，默认: 180
  --extract-fps FPS            长视频提帧帧率，默认: 1.0
  --max-frames N               最大提帧数，默认: 64
  --max-pixels N               每帧最大像素数，默认: 151200

音频:
  --language LANG              Whisper 转写语言，留空自动检测
  --no-audio                   不提取视频音轨转写

输出:
  --output-dir DIR             结果目录，默认: results
  --no-save                    不保存结果文件
  --quiet                      减少日志输出
  --tmp-dir DIR                临时文件目录，默认: tmp
  --no-cleanup                 运行结束后保留临时文件
```

示例：

```bash
# 默认推荐：交互模式
python analyze.py --continue \
  --vision-model /data/models/Qwen3-VL-2B-Instruct \
  --whisper-model /data/models/whisper-large-v3

# 指定 4B 模型
python analyze.py --continue \
  --vision-model /data/models/Qwen3-VL-4B-Instruct \
  --whisper-model /data/models/whisper-large-v3

# 仅打印结果，不保存文件
python analyze.py image.png --no-save --quiet

# 不转写视频音轨
python analyze.py video.mp4 --no-audio
```

---

## 实现流程

`analyze.py` 构造配置字典并调用 `MediaAnalyzer`。当前没有 `config.yaml` 配置入口。

处理逻辑：

1. `MediaPreprocessor.detect_type()` 通过文件扩展名识别图片、视频或音频。
2. 图片直接调用 `VisionModel.analyze_image()`。
3. 视频先用 `ffprobe` 获取时长，并按参数决定是否提取音轨转写。
4. 短视频（默认不超过 180 秒）直接交给视觉模型原生处理。
5. 长视频通过 `ffmpeg` 提帧后交给视觉模型处理。
6. 纯音频先用 Whisper 转写，再把文本交给视觉/文本模型做结构化分析。
7. 默认保存 JSON 到 `results/`，并清理 `tmp/` 下的临时文件。

---

## 硬件需求

| 硬件 | 可用性 | 备注 |
|------|--------|------|
| H200 / A100 / RTX 4090 24GB+ | 推荐 | 适合 `bfloat16` 和较大模型 |
| 12GB-24GB GPU | 可用 | 建议使用 `float16` 或较小模型 |
| Mac M 系列 | 可用 | 可尝试 `--device mps --dtype float32` |
| 纯 CPU | 可用 | 速度会很慢，仅建议调试小文件 |

显存不足时，优先降低 `--max-frames`、降低 `--max-pixels`，或改用更小的视觉模型。

---

## 高吞吐模式（vLLM，可选）

本项目使用的 vLLM 版本：

```bash
pip install vllm==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu128
```

该版本与项目推荐的 PyTorch 版本匹配：

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

如果 `nvidia-smi` 能看到 GPU，但 Python 看不到 CUDA，先检查当前环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.backends.cuda.is_built()); print(torch.cuda.is_available())"
```

批量处理场景下可以启动 vLLM 服务：

```bash
bash scripts/start_vllm.sh
```

默认模型路径：

```text
/data/models/Qwen3-VL-4B-Instruct
```

也可以指定模型路径：

```bash
bash scripts/start_vllm.sh /data/models/Qwen3-VL-4B-Instruct
```

默认地址为 `http://127.0.0.1:8011`，可通过环境变量覆盖：

```bash
PORT=8001 bash scripts/start_vllm.sh
```

服务启动后，可通过 OpenAI 兼容接口调用：`http://127.0.0.1:8011`。

也可以使用项目内的 vLLM 客户端脚本。交互模式下，输入普通文本会进行连续对话；输入存在的图片/视频/音频路径会进行结构化解析。音频会先用本地 Whisper 转写，再把转写文本提交给 vLLM 分析：

```bash
python vllm_client.py --continue
```

`vllm_client.py` 的默认配置与 vLLM 启动脚本保持一致：

```text
base_url: http://127.0.0.1:8011/v1
model: /data/models/Qwen3-VL-4B-Instruct
whisper_model: /data/models/whisper-large-v3
```

单次对话：

```bash
python vllm_client.py "帮我写个小的科幻故事"
```

单次解析图片、视频或音频，文件路径建议使用绝对路径：

```bash
python vllm_client.py /data/media_analyzer/demo_media/people.jpg
python vllm_client.py /data/media_analyzer/demo_media/holding_phone.mp4
python vllm_client.py /data/media_analyzer/demo_media/speech_audio.mp3
```

`analyze.py` 当前仍使用 transformers 后端；如需切换到 vLLM，需要在 `analyzer/vision.py` 中替换推理调用。

---

## 常见问题

**Q: CUDA out of memory**  
A: 降低 `--max-frames`，降低 `--max-pixels`，改用更小的视觉模型，或把 `--dtype` 改为 `float16`。

**Q: 视频无音轨怎么办？**  
A: 正常。音轨提取失败时 pipeline 会跳过音频步骤，仅做视觉分析。

**Q: 输出 JSON 格式错误怎么办？**  
A: 偶发于模型输出不规范时。相关容错逻辑在 `analyzer/vision.py` 的 `_extract_json()`。

**Q: `download_models.py` 找不到 `modelscope` 怎么办？**  
A: 在 `media` 环境中安装：`pip install modelscope`，然后重新运行下载脚本。
