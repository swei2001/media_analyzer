#!/usr/bin/env bash
# vLLM 高吞吐服务端（生产/批量分析场景）
# 启动后可通过 OpenAI 兼容接口调用，地址: http://localhost:8000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_NAME="media"
DEFAULT_MODEL="/data/models/Qwen/Qwen3-VL-2B-Instruct"
MODEL="${1:-$DEFAULT_MODEL}"
PORT="${PORT:-8000}"

# ── 激活 Conda 环境 ──────────────────────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到 conda，请先安装 Miniconda 或 Anaconda"
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

if ! python -c "import vllm" &>/dev/null; then
    echo "错误: 当前 $ENV_NAME 环境未安装 vllm"
    echo "请先执行: conda activate $ENV_NAME && pip install vllm"
    exit 1
fi

if [[ "$MODEL" == /* && ! -e "$MODEL" ]]; then
    echo "错误: 模型路径不存在: $MODEL"
    echo "可先运行: python scripts/download_models.py"
    echo "也可以指定模型路径: bash scripts/start_vllm.sh /path/to/model"
    exit 1
fi

echo "================================================"
echo " 启动 vLLM 服务"
echo " 模型: $MODEL"
echo " Conda 环境: $ENV_NAME"
echo " 地址: http://localhost:$PORT"
echo "================================================"

# ── 检测可用 GPU 显存，自动选择 dtype ────────────────────────────────────────
VRAM=$(python -c "
import torch
if torch.cuda.is_available():
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'{gb:.0f}')
else:
    print('0')
" 2>/dev/null || echo "0")

if [ "$VRAM" -ge 24 ]; then
    DTYPE="bfloat16"
    echo "检测到 ${VRAM}GB VRAM，使用 bfloat16 全精度"
elif [ "$VRAM" -ge 12 ]; then
    DTYPE="float16"
    echo "检测到 ${VRAM}GB VRAM，使用 float16"
else
    echo "VRAM < 12GB 或无 GPU，建议使用 transformers 后端（analyze.py 默认）"
    echo "如需强制启动请手动修改此脚本"
    exit 1
fi

vllm serve "$MODEL" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --max-model-len 32768 \
    --limit-mm-per-prompt "image=64,video=1" \
    --trust-remote-code
