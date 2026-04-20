import whisper
from pathlib import Path
import warnings


def _resolve_device(requested: str) -> tuple[str, str]:
    import torch

    if requested == "auto":
        cuda_warning = ""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cuda_available = torch.cuda.is_available()
        if caught:
            cuda_warning = str(caught[-1].message)

        if cuda_available:
            return "cuda", "检测到可用 CUDA"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", "未检测到可用 CUDA，回退到 MPS"
        if cuda_warning:
            return "cpu", f"CUDA 不可用（{cuda_warning}），回退到 CPU"
        return "cpu", "未检测到可用 CUDA/MPS，回退到 CPU"

    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("配置了 device=cuda，但当前环境未检测到 CUDA")
    if requested == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            raise ValueError("配置了 device=mps，但当前环境不支持 MPS")

    return requested, f"按配置使用 device={requested}"


class AudioModel:
    def __init__(self, config: dict):
        self.config = config
        self._model = None

    def _load(self):
        if self._model is None:
            model_spec = str(self.config["model"]["whisper_model"]).strip()
            device, reason = _resolve_device(self.config["model"].get("device", "auto"))
            available = set(whisper.available_models())
            model_path = Path(model_spec).expanduser()

            print(f"[AudioModel] 加载 Whisper {model_spec} (device={device}) ...")
            print(f"[AudioModel] 设备选择: {reason}")

            if model_path.is_dir():
                named_pt = model_path / f"{model_path.name}.pt"
                if named_pt.is_file():
                    model_path = named_pt
                else:
                    pt_files = sorted(model_path.glob("*.pt"))
                    if len(pt_files) == 1:
                        model_path = pt_files[0]
                    elif model_path.name in available:
                        self._model = whisper.load_model(
                            model_path.name,
                            device=device,
                            download_root=str(model_path.parent),
                        )
                        return
                    else:
                        raise ValueError(
                            "whisper_model 指向目录，但目录下未找到可用 .pt 文件；"
                            "请改为 Whisper 模型名（如 large-v3）或 .pt 文件路径。"
                        )

            if model_path.is_file():
                import torch

                checkpoint = torch.load(str(model_path), map_location="cpu")
                dims = whisper.ModelDimensions(**checkpoint["dims"])
                self._model = whisper.Whisper(dims)
                self._model.load_state_dict(checkpoint["model_state_dict"])
                self._model = self._model.to(device)
            elif model_spec in available:
                self._model = whisper.load_model(model_spec, device=device)
            else:
                raise ValueError(
                    f"不支持的 whisper_model: {model_spec}；可用模型名: {sorted(available)}"
                )

    def transcribe(self, audio_path: str) -> str:
        """返回转写文本，失败时返回空字符串。"""
        self._load()
        language = self.config["audio"].get("language")
        try:
            result = self._model.transcribe(
                audio_path,
                language=language,
                verbose=False,
            )
            return result["text"].strip()
        except Exception as e:
            print(f"[AudioModel] 转写失败: {e}")
            return ""
