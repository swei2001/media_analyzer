#!/usr/bin/env python3
"""
单文件媒体解析 CLI 入口

用法:
  python analyze.py <文件路径> [选项]
  python analyze.py --continue   # 交互模式：加载一次模型，批量分析
"""

import argparse
import json
import sys
from pathlib import Path


def build_config(args: argparse.Namespace) -> dict:
    script_dir = Path(__file__).parent

    def resolve(p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else script_dir / path)

    return {
        "model": {
            "vision_model": args.vision_model,
            "whisper_model": args.whisper_model,
            "device": args.device,
            "torch_dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
        },
        "video": {
            "short_video_threshold": args.short_video_threshold,
            "extract_fps": args.extract_fps,
            "max_frames": args.max_frames,
            "max_pixels": args.max_pixels,
        },
        "audio": {
            "language": args.language or None,
            "extract_audio_from_video": not args.no_audio,
        },
        "output": {
            "save_results": not args.no_save,
            "output_dir": resolve(args.output_dir),
            "verbose": not args.quiet,
        },
        "tmp": {
            "tmp_dir": resolve(args.tmp_dir),
            "cleanup": not args.no_cleanup,
        },
    }


def print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 60)


def run_interactive(analyzer) -> None:
    print("进入交互模式，输入文件路径开始分析。输入 q / quit / exit 或按 Ctrl+C 退出。")
    while True:
        try:
            raw = input("\n文件路径> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not raw or raw.lower() in ("q", "quit", "exit"):
            print("退出。")
            break

        file_path = Path(raw)
        if not file_path.exists():
            print(f"错误: 文件不存在: {raw}", file=sys.stderr)
            continue

        try:
            result = analyzer.analyze(str(file_path))
            print_result(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n错误: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="单文件媒体解析（视频/音频/图片）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analyze.py video.mp4
  python analyze.py image.jpg --quiet
  python analyze.py audio.mp3 --whisper-model /data/models/large-v3
  python analyze.py --continue --vision-model /data/models/Qwen3-VL-2B-Instruct
        """,
    )

    # 位置参数
    parser.add_argument("file", nargs="?", help="待分析的媒体文件路径（交互模式下可省略）")
    parser.add_argument("--continue", dest="interactive", action="store_true",
                        help="交互模式：加载一次模型后持续等待输入")

    # 模型
    parser.add_argument("--vision-model", default="/data/models/Qwen3-VL-2B-Instruct",
                        help="视觉模型路径或 HuggingFace ID")
    parser.add_argument("--whisper-model", default="/data/models/large-v3",
                        help="Whisper 模型路径或 HuggingFace ID")
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu", "mps"],
                        help="推理设备（默认: cuda）")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="模型精度（默认: bfloat16）")
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="最大生成 token 数（默认: 8192）")

    # 视频
    parser.add_argument("--short-video-threshold", type=int, default=180,
                        help="短视频阈值秒数，低于此值原生推理（默认: 180）")
    parser.add_argument("--extract-fps", type=float, default=1.0,
                        help="长视频提帧帧率（默认: 1.0）")
    parser.add_argument("--max-frames", type=int, default=64,
                        help="最大提帧数（默认: 64）")
    parser.add_argument("--max-pixels", type=int, default=151200,
                        help="每帧最大像素数（默认: 151200）")

    # 音频
    parser.add_argument("--language", default="",
                        help="Whisper 转写语言，留空自动检测")
    parser.add_argument("--no-audio", action="store_true",
                        help="不提取视频音轨转写")

    # 输出
    parser.add_argument("--output-dir", default="results",
                        help="结果输出目录（默认: results）")
    parser.add_argument("--no-save", action="store_true", help="不保存结果到文件")
    parser.add_argument("--quiet", action="store_true", help="减少日志输出")

    # 临时文件
    parser.add_argument("--tmp-dir", default="tmp",
                        help="临时文件目录（默认: tmp）")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="运行结束后保留临时文件")

    args = parser.parse_args()

    if not args.interactive and not args.file:
        parser.error("请提供文件路径，或使用 --continue 进入交互模式")

    if not args.interactive and args.file and not Path(args.file).exists():
        print(f"错误: 文件不存在: {args.file}", file=sys.stderr)
        sys.exit(1)

    config = build_config(args)

    from analyzer import MediaAnalyzer
    analyzer = MediaAnalyzer(config)

    if args.interactive:
        run_interactive(analyzer)
        return

    try:
        result = analyzer.analyze(args.file)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)

    print_result(result)


if __name__ == "__main__":
    main()
