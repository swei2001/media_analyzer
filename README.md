# 媒体分析器 — 功能一：单文件解析

对单个视频、音频、图片文件进行深度分析，输出结构化 JSON（思考过程 / 事件 / 解读）。

---

## 目录结构

```
media_analyzer/
├── analyze.py              # CLI 入口
├── config.yaml             # 所有可调参数
├── requirements.txt        # Python 依赖
├── analyzer/
│   ├── pipeline.py         # 主流程编排
│   ├── vision.py           # Qwen2.5-VL 推理封装
│   ├── audio.py            # Whisper 转写封装
│   └── preprocessor.py     # ffmpeg 预处理工具
├── scripts/
│   ├── setup.sh            # 一键初始化环境
│   ├── download_models.py  # 预下载模型
│   └── start_vllm.sh       # 启动 vLLM 服务（可选）
├── results/                # 输出 JSON（自动创建）
└── tmp/                    # 临时帧/音轨（自动清理）
```

---

## 快速开始

### 第一步：初始化环境

```bash
cd media_analyzer
bash scripts/setup.sh
source .venv/bin/activate
```

> 需要 Python ≥ 3.10、ffmpeg。setup.sh 会自动安装 ffmpeg（支持 Homebrew / apt / yum）。

### 第二步：下载模型

```bash
python scripts/download_models.py
```

首次运行会下载：

| 模型 | 大小 | 用途 |
|------|------|------|
| Qwen2.5-VL-7B-Instruct | ~15 GB | 视频/图片/文本推理 |
| Whisper large-v3 | ~3 GB | 音频转写 |

> 模型保存在 `~/.cache/huggingface/`，下次直接从缓存加载。

### 第三步：运行分析

```bash
# 分析视频
python analyze.py /data/media_analyzer/demo_media/BigBuckBunny.mp4

# 分析图片
python analyze.py path/to/image.jpg

# 分析音频
python analyze.py path/to/audio.wav
```

---

## 输出格式

结果同时打印到终端并保存到 `results/<文件名>_result.json`：

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

```
python analyze.py <文件> [选项]

选项:
  --config CONFIG         配置文件路径（默认: config.yaml）
  --output-dir DIR        结果目录（默认: results）
  --whisper-model MODEL   Whisper 大小: tiny/base/small/medium/large-v3
  --device DEVICE         推理设备: auto/cuda/cpu/mps
  --no-save               不保存结果文件
  --quiet                 减少日志输出
```

**示例：**

```bash
# 使用较小的 Whisper 加快音频转写
python analyze.py video.mp4 --whisper-model medium

# Mac M 系列芯片
python analyze.py video.mp4 --device mps

# 仅打印结果，不保存文件
python analyze.py image.png --no-save --quiet
```

---

## 配置说明（config.yaml）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.vision_model` | `Qwen2.5-VL-7B-Instruct` | 视觉模型 HuggingFace ID |
| `model.whisper_model` | `large-v3` | Whisper 模型大小 |
| `model.device` | `auto` | 推理设备，`auto` 自动选择 GPU |
| `model.torch_dtype` | `bfloat16` | 精度，GPU 用 `bfloat16`，CPU 用 `float32` |
| `model.max_new_tokens` | `2048` | 最大生成 token 数 |
| `video.short_video_threshold` | `180` | 秒，短于此值直接原生推理 |
| `video.extract_fps` | `1.0` | 长视频提帧帧率 |
| `video.max_frames` | `64` | 最大提帧数 |
| `audio.language` | `zh` | 转写语言，`null` 为自动检测 |
| `output.save_results` | `true` | 是否保存 JSON 结果 |
| `tmp.cleanup` | `true` | 分析后自动删除临时文件 |

---

## 硬件需求

| 硬件 | 可用性 | 备注 |
|------|--------|------|
| RTX 4090 24GB | 推荐 | bfloat16 全精度，速度最快 |
| RTX 3080 12GB | 可用 | 建议改 `torch_dtype: float16` |
| RTX 3060 8GB | 勉强 | 需 4-bit 量化（见下文） |
| Mac M2/M3 32GB | 可用 | `device: mps` |
| 纯 CPU | 可用 | `torch_dtype: float32`，极慢 |

### 8GB VRAM 4-bit 量化（可选）

```bash
pip install bitsandbytes
```

修改 `analyzer/vision.py` 的 `_load` 方法，添加：

```python
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)
```

---

## 高吞吐模式（vLLM，可选）

批量处理场景下，启动 vLLM 服务可显著提升吞吐量：

```bash
bash scripts/start_vllm.sh
```

服务启动后，可通过 OpenAI 兼容接口调用（`http://localhost:8000`）。
`analyze.py` 目前使用 transformers 后端；如需切换 vLLM 后端，在 `vision.py` 中替换推理调用为 HTTP 请求即可。

---

## 常见问题

**Q: CUDA out of memory**  
A: 降低 `video.max_frames`（如改为 32），或将 `model.torch_dtype` 改为 `float16`，或开启 4-bit 量化。

**Q: 模型下载太慢**  
A: 设置镜像源：`export HF_ENDPOINT=https://hf-mirror.com`，然后重新运行 `download_models.py`。

**Q: 视频无音轨，Whisper 报错**  
A: 正常，pipeline 会跳过音频步骤，仅做视觉分析。

**Q: 输出 JSON 格式错误**  
A: 偶发于模型输出不规范时。可在 `analyzer/vision.py` 的 `_extract_json` 中增加更强的容错逻辑（如正则提取 `{...}` 块）。
