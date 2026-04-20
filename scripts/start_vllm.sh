#!/usr/bin/env bash
# vLLM 高吞吐服务端（生产/批量分析场景）
# 启动后可通过 OpenAI 兼容接口调用，地址: http://localhost:8000
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 读取配置中的模型名称
MODEL=$(python3 -c "
import yaml
with open('$PROJECT_DIR/config.yaml') as f:
    c = yaml.safe_load(f)
print(c['model']['vision_model'])
")

echo "================================================"
echo " 启动 vLLM 服务"
echo " 模型: $MODEL"
echo " 地址: http://localhost:8000"
echo "================================================"

# 检查 vllm 是否安装
if ! python3 -c "import vllm" &>/dev/null; then
    echo "安装 vllm..."
    pip install vllm -q
fi

# 检测可用 GPU 显存，自动选择量化策略
VRAM=$(python3 -c "
import torch
if torch.cuda.is_available():
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'{gb:.0f}')
else:
    print('0')
" 2>/dev/null || echo "0")

if [ "$VRAM" -ge 24 ]; then
    DTYPE="bfloat16"
    QUANTIZE_ARGS=""
    echo "检测到 ${VRAM}GB VRAM，使用 bfloat16 全精度"
elif [ "$VRAM" -ge 12 ]; then
    DTYPE="float16"
    QUANTIZE_ARGS="--quantization awq"
    echo "检测到 ${VRAM}GB VRAM，使用 AWQ 4-bit 量化"
else
    echo "VRAM < 12GB 或无 GPU，建议使用 transformers 后端（analyze.py 默认）"
    echo "如需强制启动请手动修改此脚本"
    exit 1
fi

vllm serve "$MODEL" \
    --port 8000 \
    --dtype "$DTYPE" \
    $QUANTIZE_ARGS \
    --max-model-len 32768 \
    --limit-mm-per-prompt "image=64,video=1" \
    --trust-remote-code
