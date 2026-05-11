# 媒体分析器 - 功能一：单文件解析

对单个图片、视频或音频文件做多模态分析，输出结构化 JSON。本仓库只实现功能一：单文件解析；不实现多文件集合解析和真伪鉴别。

## 目录结构

```text
media_analyzer/
├── analyze.py              # Transformers 后端 CLI
├── analyzer/
│   ├── pipeline.py         # 主流程编排
│   ├── vision.py           # Qwen3-VL 推理封装
│   ├── audio.py            # Whisper 转写封装
│   └── preprocessor.py     # ffmpeg/ffprobe 预处理
├── eval/                   # vLLM 评测脚本
├── scripts/                # 环境、模型下载、vLLM 启动脚本
├── vllm_client.py          # vLLM HTTP 客户端
├── results/                # 默认分析结果目录
└── tmp/                    # 临时帧和音轨
```

## 快速开始

初始化环境：

```bash
cd /data/media_analyzer
bash scripts/setup.sh
conda activate media
```

下载模型：

```bash
python scripts/download_models.py
```

默认推荐交互模式，模型只加载一次：

```bash
python analyze.py --continue \
  --vision-model /data/models/Qwen3-VL-2B-Instruct \
  --whisper-model /data/models/whisper-large-v3
```

进入交互模式后输入文件路径：

```text
文件路径> /data/media_analyzer/demo_media/holding_phone.mp4
文件路径> /data/media_analyzer/demo_media/drug_interactions.jpg
文件路径> /data/media_analyzer/demo_media/speech_audio.mp3
```

单次分析：

```bash
python analyze.py path/to/file.mp4 \
  --vision-model /data/models/Qwen3-VL-2B-Instruct \
  --whisper-model /data/models/whisper-large-v3
```

## 输出格式

结果会打印到终端，并默认保存到 `results/<文件名>_result.json`。

图片没有真实时间轴，`事件` 不带秒数前缀：

```json
{
  "file": "example.jpg",
  "思考过程": "画面中可见人群、标语和街道环境。",
  "事件": [
    "人群在街道上聚集并举起标语",
    "现场呈现公共集会或抗议活动特征"
  ],
  "解读": "这是一张公共政治或社会活动现场图片。"
}
```

视频和音频有时间轴，`事件` 保留秒数或秒数范围：

```json
{
  "file": "example.mp4",
  "思考过程": "结合画面变化、音频转写和时序信息判断。",
  "事件": [
    "0-8秒：人群聚集并面向舞台",
    "9-15秒：演讲者开始发言"
  ],
  "解读": "视频记录了一段公共演讲或集会活动。"
}
```

## 常用命令

```bash
# 指定 4B 视觉模型
python analyze.py --continue \
  --vision-model /data/models/Qwen3-VL-4B-Instruct \
  --whisper-model /data/models/whisper-large-v3

# 仅打印结果，不保存文件
python analyze.py image.png --no-save --quiet

# 视频不提取音轨
python analyze.py video.mp4 --no-audio

# 保留临时帧/音轨，便于排查
python analyze.py video.mp4 --no-cleanup
```

常用参数：

```text
--vision-model MODEL          视觉模型路径，默认 /data/models/Qwen3-VL-2B-Instruct
--whisper-model MODEL         Whisper 模型路径；建议显式传 /data/models/whisper-large-v3
--device DEVICE               auto/cuda/cpu/mps，默认 cuda
--dtype DTYPE                 bfloat16/float16/float32，默认 bfloat16
--max-new-tokens N            最大生成 token 数，默认 16384
--short-video-threshold N     短视频阈值秒数，默认 180
--extract-fps FPS             长视频提帧帧率，默认 1.0
--max-frames N                最大提帧数，默认 64
--max-pixels N                每帧最大像素数，默认 151200
--language LANG               Whisper 转写语言，留空自动检测
--no-audio                    不提取视频音轨
--output-dir DIR              结果目录，默认 results
```

## 处理流程

1. 通过扩展名识别图片、视频或音频。
2. 图片直接交给视觉模型分析，事件不加时间前缀。
3. 视频先取时长，默认提取音轨并用 Whisper 转写；无音轨时跳过音频。
4. 短视频直接传给视觉模型；长视频先用 ffmpeg 抽帧。
5. 音频先转写，再把文本交给模型做结构化分析。

显存不足时优先降低 `--max-frames`、`--max-pixels`，或改用更小模型。

## vLLM 模式

启动 vLLM 服务：

```bash
bash scripts/start_vllm.sh /data/models/Qwen3-VL-4B-Instruct
```

默认地址是 `http://127.0.0.1:8011`，可用环境变量覆盖：

```bash
PORT=8001 bash scripts/start_vllm.sh /data/models/Qwen3-VL-4B-Instruct
```

使用 vLLM 客户端解析媒体或对话：

```bash
python vllm_client.py --continue
python vllm_client.py /data/media_analyzer/demo_media/drug_interactions.jpg
python vllm_client.py /data/media_analyzer/demo_media/holding_phone.mp4
python vllm_client.py /data/media_analyzer/demo_media/speech_audio.mp3
```

`vllm_client.py` 对音频文件会先本地 Whisper 转写；对视频默认也会提取音轨并转写后与视觉内容一起分析。需要关闭视频音轨时传 `--no-audio`。

## 功能一评测

评测脚本使用 vLLM OpenAI-compatible 接口，不走本地 Transformers 视觉模型加载路径。

评测数据清单已在 `eval/manifest.jsonl`。如需重新下载样本：

```bash
python eval/download_eval_media.py
```

一个 vLLM 服务通常只加载一个模型。对比多个模型时，逐个启动服务并运行评测；每个模型的结果会自动进入自己的目录。

```bash
bash scripts/start_vllm.sh /data/models/Qwen3-VL-4B-Instruct

python eval/benchmark_vllm.py \
  --manifest eval/manifest.jsonl \
  --base-url http://127.0.0.1:8011/v1 \
  --models /data/models/Qwen3-VL-4B-Instruct \
  --whisper-model /data/models/whisper-large-v3
```

切换模型后重复运行：

```bash
bash scripts/start_vllm.sh /data/models/Qwen3-VL-8B-Instruct

python eval/benchmark_vllm.py \
  --manifest eval/manifest.jsonl \
  --base-url http://127.0.0.1:8011/v1 \
  --models /data/models/Qwen3-VL-8B-Instruct \
  --whisper-model /data/models/whisper-large-v3
```

默认输出结构：

```text
eval/benchmark_results/
├── Qwen3-VL-4B-Instruct/
│   ├── benchmark_metrics.jsonl
│   ├── benchmark_scored.jsonl
│   └── *_vllm_result.json
├── Qwen3-VL-8B-Instruct/
│   ├── benchmark_metrics.jsonl
│   ├── benchmark_scored.jsonl
│   └── *_vllm_result.json
└── summary/
    ├── benchmark_summary.csv
    └── benchmark_summary.md
```

重复评测同一个模型时，会重写该模型目录下的 `benchmark_metrics.jsonl` 和结果文件。

汇总所有模型目录：

```bash
python eval/summarize_benchmark.py \
  --metrics-dir eval/benchmark_results \
  --manifest eval/manifest.jsonl
```

汇总指标按模型、媒体类型、视频时长分桶聚合；图片检查事件不带时间前缀，视频/音频检查事件带时间前缀。

## 常见问题

**CUDA out of memory**
降低 `--max-frames`、`--max-pixels`，改用更小模型，或把 `--dtype` 改为 `float16`。

**视频无音轨怎么办？**
正常。音轨提取失败时会跳过音频步骤，仅做视觉分析。

**输出 JSON 格式错误怎么办？**
模型偶发输出不规范时会触发 JSON 解析错误；容错逻辑在 `analyzer/vision.py` 的 `_extract_json()`。
