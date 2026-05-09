import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── Prompt 模板 ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "你是一个专业的媒体内容分析专家，擅长对视频、图片和音频进行深度分析。"
    "请严格按照要求的 JSON 格式输出分析结果，不要输出任何 JSON 之外的内容。"
    "只输出一个合法 JSON 对象；所有字符串必须是单行文本，不能包含未转义的双引号或裸换行。"
)

_VISUAL_PROMPT = """请对以下媒体内容进行深度分析，输出严格符合如下格式的 JSON（不加 markdown 代码块）：

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

_VIDEO_WITH_AUDIO_PROMPT = """请结合视觉内容和以下音频转写文本，对视频进行深度分析，输出严格符合如下格式的 JSON：

音频转写：
{transcript}

输出格式（不加 markdown 代码块）：
{{
  "file": "{filename}",
  "思考过程": "<结合视觉与音频的简明分析依据，标注关键时间点>",
  "事件": [
    "<秒数或秒数范围：按时间顺序描述的关键事件1>",
    "<秒数或秒数范围：关键事件2>"
  ],
  "解读": "<综合研判：事件性质、背景、可能影响>"
}}

时间要求：
{timing_rule}

要求：只输出上述 JSON 对象；不要输出 markdown；字符串内容保持简洁，避免换行。"""

_AUDIO_ONLY_PROMPT = """以下是音频文件的完整转写内容，请基于文本内容进行深度分析：

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


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}".rstrip("0").rstrip(".")


def _timing_rule(media_type: str, duration: float | None = None) -> str:
    if media_type == "image":
        return (
            "图片没有真实时间轴；事件字段仍必须以秒数开头。"
            "若只有一个静态场景，使用 \"0秒：...\"；不要编造持续时长。"
        )

    duration_hint = ""
    if duration is not None:
        duration_hint = f"，秒数范围必须在 0 到 {_format_seconds(duration)} 秒内"
    return (
        "事件字段中的每一项都必须以 \"起始秒-结束秒：\" 或 \"某秒：\" 开头，"
        f"例如 \"0-8秒：无人机在街道上空盘旋\" 或 \"9秒：车辆发生爆炸\"{duration_hint}。"
        "按时间顺序分段，不要输出没有秒数前缀的事件。"
    )


def _frame_timing_context(frame_items: list[tuple[str, float]]) -> str:
    lines = [
        f"- {Path(path).name}: {_format_seconds(second)}秒"
        for path, second in frame_items
    ]
    return "以下抽帧按时间顺序给出，文件名对应视频时间点：\n" + "\n".join(lines)


class ModelOutputJSONError(ValueError):
    """模型输出无法解析为目标 JSON。"""


def _strip_model_wrappers(text: str) -> str:
    """剥离模型常见的非 JSON 包裹内容。"""
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    return text


def _first_json_object(text: str) -> str | None:
    """返回输出中的第一个完整 JSON object，忽略字符串里的大括号。"""
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


def _escape_control_chars_in_strings(text: str) -> str:
    """修复模型偶尔在 JSON 字符串中输出的裸换行/制表符。"""
    out = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                out.append(ch)
                escaped = True
            elif ch == '"':
                out.append(ch)
                in_string = False
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            elif ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
            else:
                out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_string = True
    return "".join(out)


def _loads_json_lenient(text: str) -> dict:
    """对常见模型 JSON 小错误做有限修复后解析。"""
    attempts = [
        text,
        re.sub(r",(\s*[}\]])", r"\1", text),
        _escape_control_chars_in_strings(text),
        _escape_control_chars_in_strings(re.sub(r",(\s*[}\]])", r"\1", text)),
    ]

    last_error = None
    for candidate in attempts:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    assert last_error is not None
    raise last_error


def _json_error_message(text: str, exc: json.JSONDecodeError) -> str:
    start = max(0, exc.pos - 240)
    end = min(len(text), exc.pos + 240)
    excerpt = text[start:end]
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt += "..."
    return (
        "模型输出不是合法 JSON，无法解析。"
        f"位置: line {exc.lineno} column {exc.colno} (char {exc.pos})。"
        "这通常是模型输出被截断，或在 JSON 字符串中生成了未转义引号/换行。"
        f"\n原始输出片段:\n{excerpt}"
    )


def _extract_json(text: str) -> dict:
    """从模型输出中提取 JSON 对象，容忍 markdown 包裹和 <think> 推理块。"""
    text = _strip_model_wrappers(text)

    candidates = [text]
    obj = _first_json_object(text)
    if obj and obj != text:
        candidates.append(obj)

    last_error = None
    for candidate in candidates:
        try:
            return _loads_json_lenient(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    if last_error is None:
        raise ModelOutputJSONError("模型输出中没有找到 JSON 对象。")
    raise ModelOutputJSONError(_json_error_message(text, last_error)) from last_error


# ── 主模型类 ─────────────────────────────────────────────────────────────────

class VisionModel:
    def __init__(self, config: dict):
        self.config = config
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return

        model_name = self.config["model"]["vision_model"]
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config["model"]["torch_dtype"], torch.bfloat16)

        print(f"[VisionModel] 加载 {model_name} ...")
        load_kwargs = dict(
            dtype=dtype,
            device_map=self.config["model"]["device"],
        )
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass  # 没装 flash-attn，退回默认 attention
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_name, **load_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(model_name)
        print("[VisionModel] 模型加载完成")

    def _infer(self, messages: list) -> str:
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config["model"]["max_new_tokens"],
                do_sample=False,
            )

        # 只取生成部分（去掉 prompt token）
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def analyze_image(self, file_path: str) -> dict:
        self._load()
        filename = Path(file_path).name
        prompt = _VISUAL_PROMPT.format(
            filename=filename,
            timing_rule=_timing_rule("image"),
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": file_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        raw = self._infer(messages)
        return _extract_json(raw)

    def analyze_video_native(
        self,
        file_path: str,
        transcript: str = "",
        duration: float | None = None,
    ) -> dict:
        """直接传入视频文件（短视频，模型原生处理）。"""
        self._load()
        filename = Path(file_path).name
        max_pixels = self.config["video"]["max_pixels"]
        fps = self.config["video"]["extract_fps"]
        timing_rule = _timing_rule("video", duration)

        if transcript:
            prompt = _VIDEO_WITH_AUDIO_PROMPT.format(
                filename=filename,
                transcript=transcript,
                timing_rule=timing_rule,
            )
        else:
            prompt = _VISUAL_PROMPT.format(
                filename=filename,
                timing_rule=timing_rule,
            )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": file_path,
                        "max_pixels": max_pixels,
                        "fps": fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        raw = self._infer(messages)
        return _extract_json(raw)

    def analyze_frames(
        self,
        frame_items: list[tuple[str, float]],
        transcript: str = "",
        filename: str = "",
        duration: float | None = None,
    ) -> dict:
        """传入提帧图片列表（长视频回退方案）。"""
        self._load()
        timing_rule = _timing_rule("video", duration)
        if transcript:
            prompt = _VIDEO_WITH_AUDIO_PROMPT.format(
                filename=filename,
                transcript=transcript,
                timing_rule=timing_rule,
            )
        else:
            prompt = _VISUAL_PROMPT.format(
                filename=filename,
                timing_rule=timing_rule,
            )

        content = [{"type": "text", "text": _frame_timing_context(frame_items)}]
        for frame_path, second in frame_items:
            content.append(
                {"type": "text", "text": f"时间点：{_format_seconds(second)}秒"}
            )
            content.append({"type": "image", "image": frame_path})
        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        raw = self._infer(messages)
        return _extract_json(raw)

    def analyze_audio_text(self, transcript: str, filename: str) -> dict:
        """纯音频：基于转写文本推理。"""
        self._load()
        prompt = _AUDIO_ONLY_PROMPT.format(
            filename=filename,
            transcript=transcript,
            timing_rule=_timing_rule("audio"),
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = self._infer(messages)
        return _extract_json(raw)
