#!/usr/bin/env bash
# 一键环境初始化脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo " 媒体分析器环境初始化"
echo "================================================"

# ── 检查 Python 版本 ─────────────────────────────────────────────────────────
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required="3.10"
if [[ "$(printf '%s\n' "$required" "$python_version" | sort -V | head -n1)" != "$required" ]]; then
    echo "错误: 需要 Python >= $required，当前版本: $python_version"
    exit 1
fi
echo "[1/5] Python $python_version ✓"

# ── 检查 ffmpeg ───────────────────────────────────────────────────────────────
if ! command -v ffmpeg &> /dev/null; then
    echo "[2/5] 安装 ffmpeg..."
    if command -v brew &> /dev/null; then
        brew install ffmpeg
    elif command -v apt-get &> /dev/null; then
        sudo apt-get install -y ffmpeg
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg
    else
        echo "错误: 无法自动安装 ffmpeg，请手动安装后重试"
        exit 1
    fi
else
    echo "[2/5] ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}') ✓"
fi

# ── 创建虚拟环境 ──────────────────────────────────────────────────────────────
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[3/5] 创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
else
    echo "[3/5] 虚拟环境已存在 ✓"
fi

# 激活
source "$VENV_DIR/bin/activate"

# ── 安装依赖 ──────────────────────────────────────────────────────────────────
echo "[4/5] 安装 Python 依赖..."
pip install --upgrade pip -q
pip install -r "$PROJECT_DIR/requirements.txt"

# ── 检测 GPU ──────────────────────────────────────────────────────────────────
echo "[5/5] 检查推理环境..."
python3 - <<'EOF'
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
echo " 激活环境:  source .venv/bin/activate"
echo " 下载模型:  python scripts/download_models.py"
echo " 运行分析:  python analyze.py <文件路径>"
echo "================================================"
