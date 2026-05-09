#!/usr/bin/env python3
"""
使用 ModelScope 批量下载模型到 /data/models。

首次运行时执行，后续分析可直接使用本地模型路径。
"""

from modelscope import snapshot_download


MODEL_CACHE_DIR = "/data/models"

models_to_download = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "openai-mirror/whisper-large-v3",
    # 在这里添加更多模型名称
]


def main() -> None:
    print("=" * 50)
    print(" 模型批量下载工具")
    print("=" * 50)

    for model_name in models_to_download:
        print(f"\n开始下载模型: {model_name}")
        try:
            model_dir = snapshot_download(model_name, cache_dir=MODEL_CACHE_DIR)
            print(f"模型 {model_name} 下载完成。")
            print(f"保存路径: {model_dir}")
        except Exception as e:
            print(f"模型 {model_name} 下载失败: {e}")

    print("\n所有模型下载任务完成！")


if __name__ == "__main__":
    main()
