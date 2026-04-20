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
)

_VISUAL_PROMPT = """请对以下媒体内容进行深度分析，输出严格符合如下格式的 JSON（不加 markdown 代码块）：

{{
  "file": "{filename}",
  "思考过程": "<逐步分析的推理过程，涵盖视觉线索、音频线索（如有）、时序信息等>",
  "事件": [
    "<按时间顺序描述的关键事件1>",
    "<关键事件2>"
  ],
  "解读": "<综合研判：事件性质、背景、可能影响>"
}}"""

_VIDEO_WITH_AUDIO_PROMPT = """请结合视觉内容和以下音频转写文本，对视频进行深度分析，输出严格符合如下格式的 JSON：

音频转写：
{transcript}

输出格式（不加 markdown 代码块）：
{{
  "file": "{filename}",
  "思考过程": "<结合视觉与音频的逐步推理，标注关键时间点>",
  "事件": [
    "<按时间顺序描述的关键事件1>",
    "<关键事件2>"
  ],
  "解读": "<综合研判：事件性质、背景、可能影响>"
}}"""

_AUDIO_ONLY_PROMPT = """以下是音频文件的完整转写内容，请基于文本内容进行深度分析：

音频转写：
{transcript}

输出严格符合如下格式的 JSON（不加 markdown 代码块）：
{{
  "file": "{filename}",
  "思考过程": "<对音频内容的逐步分析：说话人、情绪、关键信息等>",
  "事件": [
    "<关键事件或信息点1>",
    "<关键事件或信息点2>"
  ],
  "解读": "<综合研判：内容性质、背景、可能影响>"
}}"""


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """从模型输出中提取 JSON 对象，容忍 markdown 包裹和 <think> 推理块。"""
    text = text.strip()
    # 剥离 Qwen3 的 <think>...</think> 推理块
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 去掉可能的 ```json ... ``` 包裹
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 回退：提取第一个完整 {...} 块
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise


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
            torch_dtype=dtype,
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
        prompt = _VISUAL_PROMPT.format(filename=filename)

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

    def analyze_video_native(self, file_path: str, transcript: str = "") -> dict:
        """直接传入视频文件（短视频，模型原生处理）。"""
        self._load()
        filename = Path(file_path).name
        max_pixels = self.config["video"]["max_pixels"]
        fps = self.config["video"]["extract_fps"]

        if transcript:
            prompt = _VIDEO_WITH_AUDIO_PROMPT.format(
                filename=filename, transcript=transcript
            )
        else:
            prompt = _VISUAL_PROMPT.format(filename=filename)

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

    def analyze_frames(self, frame_paths: list[str], transcript: str = "", filename: str = "") -> dict:
        """传入提帧图片列表（长视频回退方案）。"""
        self._load()
        if transcript:
            prompt = _VIDEO_WITH_AUDIO_PROMPT.format(
                filename=filename, transcript=transcript
            )
        else:
            prompt = _VISUAL_PROMPT.format(filename=filename)

        content = [{"type": "image", "image": p} for p in frame_paths]
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
        prompt = _AUDIO_ONLY_PROMPT.format(filename=filename, transcript=transcript)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = self._infer(messages)
        return _extract_json(raw)
