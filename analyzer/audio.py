import logging

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline


class _WhisperDuplicateProcessorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "A custom logits processor of type" not in message:
            return True
        return (
            "SuppressTokensLogitsProcessor" not in message
            and "SuppressTokensAtBeginLogitsProcessor" not in message
        )


logging.getLogger("transformers.generation.utils").addFilter(
    _WhisperDuplicateProcessorFilter()
)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("配置了 device=cuda，但当前环境未检测到 CUDA")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("配置了 device=mps，但当前环境不支持 MPS")
    return requested


def _select_dtype(device: str, requested: str) -> torch.dtype:
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(requested, torch.float16)
    # bfloat16 在 CPU/MPS 上部分算子不支持，回退 float32
    if device in ("cpu", "mps") and dtype == torch.bfloat16:
        return torch.float32
    return dtype


class AudioModel:
    def __init__(self, config: dict):
        self.config = config
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return

        model_spec = str(self.config["model"]["whisper_model"]).strip()
        device = _resolve_device(self.config["model"].get("device", "auto"))
        dtype = _select_dtype(device, self.config["model"].get("torch_dtype", "float16"))

        print(f"[AudioModel] 加载 Whisper {model_spec} (device={device}, dtype={dtype}) ...")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_spec,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_spec)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.forced_decoder_ids = None

        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=device,
            chunk_length_s=30,
            ignore_warning=True,
        )
        print("[AudioModel] 模型加载完成")

    @staticmethod
    def _format_transcript(result: dict) -> str:
        chunks = result.get("chunks") or []
        if not chunks:
            return result.get("text", "").strip()

        lines = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            timestamp = chunk.get("timestamp") or ()
            if not text:
                continue
            if len(timestamp) == 2 and timestamp[0] is not None:
                start = float(timestamp[0])
                end = timestamp[1]
                if end is None:
                    lines.append(f"[{start:.1f}秒] {text}")
                else:
                    lines.append(f"[{start:.1f}-{float(end):.1f}秒] {text}")
            else:
                lines.append(text)
        return "\n".join(lines).strip()

    def transcribe(self, audio_path: str) -> str:
        """返回转写文本，失败时返回空字符串。"""
        self._load()
        language = self.config["audio"].get("language")
        try:
            generate_kwargs = {"task": "transcribe"}
            if language:
                generate_kwargs["language"] = language
            result = self._pipe(
                audio_path,
                generate_kwargs=generate_kwargs,
                return_timestamps=True,
            )
            return self._format_transcript(result)
        except Exception as e:
            print(f"[AudioModel] 转写失败: {e}")
            return ""
