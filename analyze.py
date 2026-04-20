#!/usr/bin/env python3
"""
功能一：单文件媒体解析 CLI 入口

用法:
  python analyze.py <文件路径> [--config config.yaml] [--output-dir results]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    if args.whisper_model:
        config["model"]["whisper_model"] = args.whisper_model
    if args.device:
        config["model"]["device"] = args.device
    if args.no_save:
        config["output"]["save_results"] = False
    if args.quiet:
        config["output"]["verbose"] = False
    return config


def main():
    parser = argparse.ArgumentParser(
        description="单文件媒体解析（视频/音频/图片）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analyze.py video.mp4
  python analyze.py image.jpg --quiet
  python analyze.py audio.wav --output-dir my_results
  python analyze.py video.mp4 --whisper-model medium --device cuda
        """,
    )
    parser.add_argument("file", help="待分析的媒体文件路径")
    parser.add_argument(
        "--config", default="config.yaml", help="配置文件路径（默认: config.yaml）"
    )
    parser.add_argument("--output-dir", help="结果输出目录（覆盖配置文件）")
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper 模型大小（覆盖配置文件）",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        help="推理设备（覆盖配置文件）",
    )
    parser.add_argument("--no-save", action="store_true", help="不保存结果到文件")
    parser.add_argument("--quiet", action="store_true", help="减少日志输出")

    args = parser.parse_args()

    # 检查文件存在
    if not Path(args.file).exists():
        print(f"错误: 文件不存在: {args.file}", file=sys.stderr)
        sys.exit(1)

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(str(config_path))
    config = merge_cli_overrides(config, args)

    # 确保相对路径基于脚本所在目录
    script_dir = Path(__file__).parent
    for key in ("output_dir",):
        val = config["output"].get(key, "")
        if val and not Path(val).is_absolute():
            config["output"][key] = str(script_dir / val)
    for key in ("tmp_dir",):
        val = config["tmp"].get(key, "")
        if val and not Path(val).is_absolute():
            config["tmp"][key] = str(script_dir / val)

    # 延迟导入（避免在参数解析阶段触发模型加载）
    from analyzer import MediaAnalyzer

    analyzer = MediaAnalyzer(config)

    try:
        result = analyzer.analyze(args.file)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)

    # 终端输出 JSON
    print("\n" + "=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()


