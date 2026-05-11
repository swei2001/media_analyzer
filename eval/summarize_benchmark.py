#!/usr/bin/env python3
"""Summarize benchmark JSONL records for feature 1."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_KEYS = ("file", "思考过程", "事件", "解读")
TIME_PREFIX_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?秒|[0-9:.]+(?:-[0-9:.]+)?秒?)[:：]")


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {exc}") from exc
    return rows


def _metrics_sources(metrics: str | None, metrics_dir: str | None) -> list[tuple[Path, list[dict[str, Any]]]]:
    if metrics:
        path = _resolve_path(metrics)
        return [(path, _read_jsonl(path))]

    if not metrics_dir:
        raise ValueError("请提供 --metrics 或 --metrics-dir")

    root = _resolve_path(metrics_dir)
    paths = sorted(root.glob("*/benchmark_metrics.jsonl"))
    if not paths:
        raise FileNotFoundError(f"未找到模型 metrics 文件: {root}/*/benchmark_metrics.jsonl")
    return [(path, _read_jsonl(path)) for path in paths]


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    items = _read_jsonl(path)
    mapping: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = item.get("id") or Path(item.get("file", "")).stem
        mapping[str(item_id)] = item
        if item.get("file"):
            mapping[str(_resolve_path(item["file"]))] = item
    return mapping


def _load_result(path: str | None) -> tuple[dict[str, Any] | None, str]:
    if not path:
        return None, "missing_result_path"
    result_path = Path(path)
    if not result_path.exists():
        return None, "result_file_not_found"
    try:
        with result_path.open("r", encoding="utf-8") as f:
            result = json.load(f)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(result, dict):
        return None, "json_not_object"
    return result, ""


def _schema_pass(result: dict[str, Any] | None) -> bool:
    if not result:
        return False
    if not all(key in result for key in REQUIRED_KEYS):
        return False
    return isinstance(result.get("事件"), list)


def _event_time_rule_rate(result: dict[str, Any] | None, media_type: str) -> float:
    if not result or not isinstance(result.get("事件"), list) or not result["事件"]:
        return 0.0
    events = [str(event) for event in result["事件"]]
    if media_type == "image":
        hits = sum(1 for event in events if not TIME_PREFIX_RE.search(event))
    else:
        hits = sum(1 for event in events if TIME_PREFIX_RE.search(event))
    return hits / len(events)


def _keyword_recall(result: dict[str, Any] | None, expected_keywords: list[str]) -> float | None:
    if not expected_keywords:
        return None
    if not result:
        return 0.0
    text = json.dumps(result, ensure_ascii=False)
    hits = sum(1 for keyword in expected_keywords if str(keyword) in text)
    return hits / len(expected_keywords)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    idx = int(round((len(values) - 1) * 0.95))
    return values[idx]


def _avg(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def score_rows(metrics: list[dict[str, Any]], manifest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in metrics:
        manifest_item = manifest.get(str(row.get("id"))) or manifest.get(str(row.get("file"))) or {}
        expected_keywords = manifest_item.get("expected_keywords") or []
        result, result_error = _load_result(row.get("result_path")) if row.get("success") else (None, row.get("error", ""))
        keyword_recall = _keyword_recall(result, expected_keywords)
        scored.append({
            **row,
            "json_valid": result is not None,
            "result_error": result_error,
            "schema_pass": _schema_pass(result),
            "event_time_rule_rate": round(_event_time_rule_rate(result, str(row.get("media_type", ""))), 4),
            "keyword_recall": None if keyword_recall is None else round(keyword_recall, 4),
            "expected_category": manifest_item.get("expected_category", ""),
            "topic": manifest_item.get("topic", ""),
        })
    return scored


def aggregate_rows(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        key = (
            str(row.get("backend", "")),
            str(row.get("model_name", "")),
            str(row.get("media_type", "")),
            str(row.get("duration_bucket", "")),
        )
        groups[key].append(row)

    summary: list[dict[str, Any]] = []
    for (backend, model_name, media_type, duration_bucket), rows in sorted(groups.items()):
        total = len(rows)
        latencies = [
            float(row["latency_seconds"])
            for row in rows
            if isinstance(row.get("latency_seconds"), (int, float))
        ]
        keyword_values = [
            float(row["keyword_recall"])
            for row in rows
            if isinstance(row.get("keyword_recall"), (int, float))
        ]
        summary.append({
            "backend": backend,
            "model_name": model_name,
            "media_type": media_type,
            "duration_bucket": duration_bucket,
            "count": total,
            "success_rate": round(sum(1 for row in rows if row.get("success")) / total, 4),
            "json_valid_rate": round(sum(1 for row in rows if row.get("json_valid")) / total, 4),
            "schema_pass_rate": round(sum(1 for row in rows if row.get("schema_pass")) / total, 4),
            "avg_event_time_rule_rate": round(_avg([float(row.get("event_time_rule_rate", 0.0)) for row in rows]), 4),
            "avg_keyword_recall": "" if not keyword_values else round(_avg(keyword_values), 4),
            "avg_latency_seconds": round(_avg(latencies), 3),
            "p50_latency_seconds": round(statistics.median(latencies), 3) if latencies else 0.0,
            "p95_latency_seconds": round(_p95(latencies), 3),
        })
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("# Benchmark Summary\n\nNo rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = ["# Benchmark Summary", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    metrics_arg = getattr(args, "metrics", None)
    metrics_dir_arg = getattr(args, "metrics_dir", None)
    sources = _metrics_sources(metrics_arg, metrics_dir_arg)
    manifest = _load_manifest(_resolve_path(args.manifest))
    metrics = [row for _, rows in sources for row in rows]
    scored = score_rows(metrics, manifest)
    summary = aggregate_rows(scored)

    if metrics_arg:
        default_dir = _resolve_path("eval")
    else:
        default_dir = _resolve_path(metrics_dir_arg) / "summary"

    scored_output = getattr(args, "scored_output", None)
    csv_output = getattr(args, "csv_output", None)
    md_output = getattr(args, "md_output", None)
    scored_path = _resolve_path(scored_output) if scored_output else default_dir / "benchmark_scored.jsonl"
    csv_path = _resolve_path(csv_output) if csv_output else default_dir / "benchmark_summary.csv"
    md_path = _resolve_path(md_output) if md_output else default_dir / "benchmark_summary.md"

    if not metrics_arg:
        for source_path, source_rows in sources:
            source_scored = score_rows(source_rows, manifest)
            source_scored_path = source_path.parent / "benchmark_scored.jsonl"
            _write_jsonl(source_scored_path, source_scored)
            print(f"[Summary] 模型逐条评分: {source_scored_path}")

    _write_jsonl(scored_path, scored)
    _write_csv(csv_path, summary)
    _write_markdown(md_path, summary)

    print(f"[Summary] 逐条评分: {scored_path}")
    print(f"[Summary] 汇总 CSV: {csv_path}")
    print(f"[Summary] 汇总 Markdown: {md_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总功能一 benchmark 指标")
    parser.add_argument("--metrics", default=None, help="兼容旧用法：单个 metrics JSONL")
    parser.add_argument("--metrics-dir", default="eval/benchmark_results", help="包含各模型 benchmark_metrics.jsonl 的目录")
    parser.add_argument("--manifest", default="eval/manifest.jsonl")
    parser.add_argument("--scored-output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--md-output", default=None)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
