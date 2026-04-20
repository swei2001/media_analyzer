#!/usr/bin/env python3
"""
预下载所有模型到本地缓存。
首次运行时执行，后续分析直接使用缓存。
"""

import sys
from pathlib import Path

# 加载配置
sys.path.insert(0, str(Path(__file__).parent.parent))
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

vision_model = config["model"]["vision_model"]
whisper_model = config["model"]["whisper_model"]


def download_vision_model():
    print(f"\n[1/2] 下载视觉模型: {vision_model}")
    print("      首次下载约 15-30 GB，请确保磁盘空间充足...")
    from transformers import AutoModelForImageTextToText, AutoProcessor

    AutoProcessor.from_pretrained(vision_model)
    print("      Processor 下载完成")

    AutoModelForImageTextToText.from_pretrained(
        vision_model,
        torch_dtype="auto",
        device_map="cpu",   # 下载时用 CPU，避免 OOM
    )
    print(f"      {vision_model} 下载完成 ✓")


def download_whisper_model():
    print(f"\n[2/2] 下载 Whisper {whisper_model}...")
    import whisper

    model_spec = str(whisper_model).strip()
    model_path = Path(model_spec).expanduser()
    available = set(whisper.available_models())

    if model_path.is_file():
        print(f"      检测到本地 .pt 文件，跳过下载: {model_path}")
        return

    if model_path.is_dir():
        named_pt = model_path / f"{model_path.name}.pt"
        if named_pt.is_file():
            print(f"      检测到本地 .pt 文件，跳过下载: {named_pt}")
            return
        if model_path.name in available:
            whisper.load_model(model_path.name, download_root=str(model_path.parent))
            print(f"      Whisper {model_path.name} 下载完成 ✓")
            return
        raise ValueError(
            "whisper_model 指向目录，但目录下无 .pt 文件，且目录名不是可用模型名"
        )

    if model_spec in available:
        whisper.load_model(model_spec)
        print(f"      Whisper {model_spec} 下载完成 ✓")
        return

    raise ValueError(
        f"不支持的 whisper_model: {model_spec}；可用模型名: {sorted(available)}"
    )


if __name__ == "__main__":
    print("=" * 50)
    print(" 模型预下载工具")
    print("=" * 50)

    try:
        download_vision_model()
    except Exception as e:
        print(f"视觉模型下载失败: {e}")
        sys.exit(1)

    try:
        download_whisper_model()
    except Exception as e:
        print(f"Whisper 下载失败: {e}")
        sys.exit(1)

    print("\n所有模型下载完成，可以开始分析了。")
