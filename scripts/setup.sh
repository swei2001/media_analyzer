#!/usr/bin/env bash
# Conda 一键环境初始化脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_NAME="media"
PYTHON_VERSION="3.12"

echo "================================================"
echo " 媒体分析器环境初始化"
echo "================================================"

# ── 检查 Conda ───────────────────────────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "错误: 未检测到 conda，请先安装 Miniconda 或 Anaconda"
    exit 1
fi
echo "[1/6] conda $(conda --version | awk '{print $2}') ✓"

# ── 初始化 Conda shell ───────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"

# ── 创建 Conda 环境 ──────────────────────────────────────────────────────────
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[2/6] Conda 环境 $ENV_NAME 已存在 ✓"
else
    echo "[2/6] 创建 Conda 环境 $ENV_NAME (python=$PYTHON_VERSION)..."
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

# ── 激活环境并检查 Python ───────────────────────────────────────────────────
conda activate "$ENV_NAME"
python_version=$(python --version 2>&1 | awk '{print $2}')
if [[ "$python_version" != "$PYTHON_VERSION"* ]]; then
    echo "错误: $ENV_NAME 环境 Python 版本应为 $PYTHON_VERSION，当前版本: $python_version"
    exit 1
fi
echo "[3/6] Python $python_version ✓"

# ── 安装 ffmpeg ───────────────────────────────────────────────────────────────
if ! command -v ffmpeg &> /dev/null; then
    echo "[4/6] 安装 ffmpeg..."
    conda install -y -c conda-forge ffmpeg
else
    echo "[4/6] ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}') ✓"
fi

# ── 安装依赖 ──────────────────────────────────────────────────────────────────
echo "[5/6] 安装 Python 依赖..."
python -m pip install --upgrade pip -q

echo "      先安装 PyTorch 依赖..."
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

echo "      安装剩余依赖..."
python -m pip install -r "$PROJECT_DIR/requirements.txt"

# ── 检测 GPU ──────────────────────────────────────────────────────────────────
echo "[6/6] 检查推理环境..."
python - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
elif torch.backends.mps.is_available():
    print("  设备: Apple Silicon MPS")
else:
    print("  设备: CPU（推理速度较慢）")
EOF

echo ""
echo "================================================"
echo " 初始化完成！"
echo ""
echo " 激活环境:  conda activate media"
echo " 下载模型:  python scripts/download_models.py"
echo " 运行分析:  python analyze.py <文件路径>"
echo "================================================"
