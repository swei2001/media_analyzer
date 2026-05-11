#!/usr/bin/env python3
"""vLLM benchmark for feature 1 single-file media analysis."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_WHISPER_MODEL, analyze_media  # noqa: E402


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _model_slug(model: str) -> str:
    name = Path(model.rstrip("/")).name or model
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def _load_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest 第 {line_no} 行不是合法 JSON: {exc}") from exc
            if "file" not in item:
                raise ValueError(f"manifest 第 {line_no} 行缺少 file 字段")
            if "id" not in item:
                item["id"] = Path(item["file"]).stem
            items.append(item)
            if limit is not None and len(items) >= limit:
                break
    return items


def _detect_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}:
        return "image"
    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}:
        return "video"
    if suffix in {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}:
        return "audio"
    raise ValueError(f"不支持的文件格式: {path.suffix}")


def _video_duration(path: Path) -> float | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _duration_bucket(media_type: str, duration: float | None, threshold: int) -> str:
    if media_type != "video":
        return "not_video"
    if duration is None:
        return "unknown"
    return "short" if duration <= threshold else "long"


def _result_path(output_dir: Path, media_path: Path) -> Path:
    return output_dir / f"{media_path.stem}_vllm_result.json"


def _save_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _client_args(args: argparse.Namespace, model: str) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=args.base_url,
        model=model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_dtype=args.whisper_dtype,
        language=args.language,
        extract_audio_from_video=getattr(args, "extract_audio_from_video", True),
        tmp_dir=getattr(args, "tmp_dir", "tmp"),
    )


def run(args: argparse.Namespace) -> int:
    manifest_path = _resolve_path(args.manifest)
    output_root = _resolve_path(args.output_dir)
    metrics_path = _resolve_path(args.metrics_path) if getattr(args, "metrics_path", None) else None
    items = _load_manifest(manifest_path, args.limit)
    if not items:
        raise ValueError(f"manifest 为空: {manifest_path}")

    append = args.append
    written_metrics: list[Path] = []
    for model in args.models:
        slug = _model_slug(model)
        model_output_dir = output_root / slug
        model_metrics_path = metrics_path or model_output_dir / "benchmark_metrics.jsonl"
        client_args = _client_args(args, model)
        rows: list[dict[str, Any]] = []

        for idx, item in enumerate(items, 1):
            media_path = _resolve_path(item["file"])
            media_type = item.get("media_type")
            result_path = _result_path(model_output_dir, media_path)
            row: dict[str, Any] = {
                "id": item.get("id"),
                "backend": "vllm",
                "base_url": args.base_url,
                "model": model,
                "model_name": slug,
                "file": str(media_path),
                "manifest_file": str(manifest_path),
                "media_type": media_type,
                "duration_seconds": None,
                "duration_bucket": "unknown",
                "latency_seconds": None,
                "result_path": str(result_path),
                "success": False,
                "error": "",
            }

            print(f"[Benchmark:vLLM] {slug} [{idx}/{len(items)}] {media_path.name}")
            start = time.perf_counter()
            try:
                if not media_path.exists():
                    raise FileNotFoundError(f"文件不存在: {media_path}")
                media_type = _detect_media_type(media_path)
                row["media_type"] = media_type
                duration = _video_duration(media_path) if media_type == "video" else None
                row["duration_seconds"] = duration
                row["duration_bucket"] = _duration_bucket(media_type, duration, args.short_video_threshold)

                result = analyze_media(media_path, client_args)
                _save_result(result_path, result)
                row["success"] = True
            except Exception as exc:  # noqa: BLE001 - benchmark should continue
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[Benchmark:vLLM] 失败: {row['error']}")
            finally:
                row["latency_seconds"] = round(time.perf_counter() - start, 3)
                rows.append(row)

        _write_jsonl(model_metrics_path, rows, append=append)
        written_metrics.append(model_metrics_path)
        if metrics_path:
            append = True

    for path in written_metrics:
        print(f"[Benchmark:vLLM] 指标已写入: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 vLLM 批量评测功能一单文件解析")
    parser.add_argument("--manifest", default="eval/manifest.jsonl", help="评测样本 JSONL")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL], help="vLLM 服务中的模型名/路径")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM OpenAI API 地址")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--short-video-threshold", type=int, default=180)
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--whisper-device", default="cuda", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--whisper-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--language", default="")
    parser.add_argument("--no-audio", dest="extract_audio_from_video", action="store_false", help="解析视频时不提取音轨转写")
    parser.add_argument("--tmp-dir", default="tmp")
    parser.add_argument("--output-dir", default="eval/benchmark_results")
    parser.add_argument("--metrics-path", default=None, help="兼容旧用法：显式指定单个 metrics JSONL")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true", help="追加写入 metrics，而不是覆盖")
    parser.set_defaults(extract_audio_from_video=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
