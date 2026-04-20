import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline


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
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_spec)

        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
            chunk_length_s=30,
        )
        print("[AudioModel] 模型加载完成")

    def transcribe(self, audio_path: str) -> str:
        """返回转写文本，失败时返回空字符串。"""
        self._load()
        language = self.config["audio"].get("language")
        try:
            kwargs = {"generate_kwargs": {"language": language}} if language else {}
            result = self._pipe(audio_path, **kwargs)
            return result["text"].strip()
        except Exception as e:
            print(f"[AudioModel] 转写失败: {e}")
            return ""
