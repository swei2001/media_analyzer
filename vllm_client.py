#!/usr/bin/env python3
"""
vLLM OpenAI-compatible client.

Usage:
  python vllm_client.py --continue
  python vllm_client.py "你好，介绍一下你自己"
  python vllm_client.py /data/media_analyzer/demo_media/people.jpg
  python vllm_client.py /data/media_analyzer/demo_media/holding_phone.mp4
  python vllm_client.py /data/media_analyzer/demo_media/speech_audio.mp3
"""

import argparse
import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:8011/v1"
DEFAULT_MODEL = "/data/models/Qwen3-VL-4B-Instruct"
DEFAULT_WHISPER_MODEL = "/data/models/whisper-large-v3"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}

SYSTEM_PROMPT = (
    "你是一个专业的媒体内容分析专家，擅长对视频、图片和音频进行深度分析。"
    "请严格按照要求的 JSON 格式输出分析结果，不要输出任何 JSON 之外的内容。"
    "只输出一个合法 JSON 对象；所有字符串必须是单行文本，不能包含未转义的双引号或裸换行。"
)

VISUAL_PROMPT = """请对以下媒体内容进行深度分析，输出严格符合如下格式的 JSON（不加 markdown 代码块）：

{{
  "file": "{filename}",
  "思考过程": "<简明分析依据，涵盖视觉线索、音频线索（如有）、时序信息等>",
  "事件": [
    "<秒数或秒数范围：按时间顺序描述的关键事件1>",
    "<秒数或秒数范围：关键事件2>"
  ],
  "解读": "<综合研判：事件性质、背景、可能影响>"
}}

时间要求：
{timing_rule}

要求：只输出上述 JSON 对象；不要输出 markdown；字符串内容保持简洁，避免换行。"""

AUDIO_PROMPT = """以下是音频文件的完整转写内容，请基于文本内容进行深度分析：

音频转写：
{transcript}

输出严格符合如下格式的 JSON（不加 markdown 代码块）：
{{
  "file": "{filename}",
  "思考过程": "<对音频内容的简明分析依据：说话人、情绪、关键信息等>",
  "事件": [
    "<秒数或秒数范围：关键事件或信息点1>",
    "<秒数或秒数范围：关键事件或信息点2>"
  ],
  "解读": "<综合研判：内容性质、背景、可能影响>"
}}

时间要求：
{timing_rule}

要求：只输出上述 JSON 对象；不要输出 markdown；字符串内容保持简洁，避免换行。"""


def _timing_rule(media_type: str) -> str:
    if media_type == "image":
        return (
            "图片没有真实时间轴；事件字段仍必须以秒数开头。"
            "若只有一个静态场景，使用 \"0秒：...\"；不要编造持续时长。"
        )
    return (
        "事件字段中的每一项都必须以 \"起始秒-结束秒：\" 或 \"某秒：\" 开头，"
        "例如 \"0-8秒：无人机在街道上空盘旋\" 或 \"9秒：车辆发生爆炸\"。"
        "按时间顺序分段，不要输出没有秒数前缀的事件。"
    )


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    obj = _first_json_object(text)
    if obj:
        text = obj
    return json.loads(text)


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _media_content(path: Path) -> dict:
    suffix = path.suffix.lower()
    url = _data_url(path)
    if suffix in IMAGE_EXTS:
        return {"type": "image_url", "image_url": {"url": url}}
    if suffix in VIDEO_EXTS:
        return {"type": "video_url", "video_url": {"url": url}}
    raise ValueError(f"vLLM 客户端目前只支持图片/视频解析: {path.suffix}")


def _audio_config(args: argparse.Namespace) -> dict:
    return {
        "model": {
            "whisper_model": args.whisper_model,
            "device": args.whisper_device,
            "torch_dtype": args.whisper_dtype,
        },
        "audio": {
            "language": args.language or None,
        },
    }


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"连接 vLLM 失败: {exc}") from exc


def _chat_completion(
    messages: list[dict],
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    response = _post_json(
        f"{base_url.rstrip('/')}/chat/completions",
        payload,
        timeout,
    )
    try:
        return response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"无法解析 vLLM 响应: {response}") from exc


def chat_once(text: str, args: argparse.Namespace) -> str:
    messages = [{"role": "user", "content": text}]
    return _chat_completion(
        messages,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )


def analyze_media(path: Path, args: argparse.Namespace) -> dict:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        media_type = "image"
    elif suffix in VIDEO_EXTS:
        media_type = "video"
    elif suffix in AUDIO_EXTS:
        return analyze_audio(path, args)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    prompt = VISUAL_PROMPT.format(
        filename=path.name,
        timing_rule=_timing_rule(media_type),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                _media_content(path),
                {"type": "text", "text": prompt},
            ],
        },
    ]
    raw = _chat_completion(
        messages,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    return _extract_json(raw)


def analyze_audio(path: Path, args: argparse.Namespace) -> dict:
    from analyzer.audio import AudioModel

    print("[vLLM] 音频转写中...")
    transcript = AudioModel(_audio_config(args)).transcribe(str(path))
    if not transcript:
        raise RuntimeError("音频转写失败，无法继续分析")

    print(f"[vLLM] 转写完成（{len(transcript)} 字），提交 vLLM 分析...")
    prompt = AUDIO_PROMPT.format(
        filename=path.name,
        transcript=transcript,
        timing_rule=_timing_rule("audio"),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    raw = _chat_completion(
        messages,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    return _extract_json(raw)


def save_result(result: dict, path: Path, output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_vllm_result.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[vLLM] 结果已保存: {out_path}")


def handle_input(text: str, args: argparse.Namespace, history: list[dict] | None) -> None:
    path = Path(text).expanduser()
    if path.exists():
        result = analyze_media(path.resolve(), args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.save_results:
            save_result(result, path, args.output_dir)
        return

    if history is None:
        print(chat_once(text, args))
        return

    history.append({"role": "user", "content": text})
    reply = _chat_completion(
        history,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    history.append({"role": "assistant", "content": reply})
    print(reply)


def run_interactive(args: argparse.Namespace) -> None:
    print("进入 vLLM 交互模式。输入文字进行对话；输入图片/视频/音频绝对路径进行解析。输入 q / quit / exit 退出。")
    history: list[dict] = []
    while True:
        try:
            raw = input("\n输入> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not raw or raw.lower() in {"q", "quit", "exit"}:
            print("退出。")
            break

        try:
            handle_input(raw, args, history)
        except Exception as exc:
            print(f"错误: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM 对话/图片视频音频解析客户端")
    parser.add_argument("input", nargs="?", help="文本，或图片/视频/音频文件路径")
    parser.add_argument("--continue", dest="interactive", action="store_true",
                        help="交互模式：文本连续对话，文件路径触发解析")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"vLLM OpenAI API 地址（默认: {DEFAULT_BASE_URL}）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"模型名/路径（默认: {DEFAULT_MODEL}）")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="最大输出 token 数（默认: 2048）")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL,
                        help=f"Whisper 模型路径（默认: {DEFAULT_WHISPER_MODEL}）")
    parser.add_argument("--whisper-device", default="cuda",
                        choices=["auto", "cuda", "cpu", "mps"],
                        help="Whisper 推理设备（默认: cuda）")
    parser.add_argument("--whisper-dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Whisper 推理精度（默认: bfloat16）")
    parser.add_argument("--language", default="",
                        help="Whisper 转写语言，留空自动检测")
    parser.add_argument("--timeout", type=int, default=600,
                        help="HTTP 超时时间秒数（默认: 600）")
    parser.add_argument("--output-dir", default="results",
                        help="解析结果保存目录（默认: results）")
    parser.add_argument("--no-save", dest="save_results", action="store_false",
                        help="解析媒体时不保存结果文件")
    parser.set_defaults(save_results=True)
    args = parser.parse_args()

    if args.interactive:
        run_interactive(args)
        return

    if not args.input:
        parser.error("请提供文本/文件路径，或使用 --continue")

    handle_input(args.input, args, history=None)


if __name__ == "__main__":
    main()
