import json
import time
from pathlib import Path

from .audio import AudioModel
from .preprocessor import MediaPreprocessor
from .vision import VisionModel


class MediaAnalyzer:
    def __init__(self, config: dict):
        self.config = config
        self.preprocessor = MediaPreprocessor(config)
        self.vision = VisionModel(config)
        self.audio = AudioModel(config)

    def analyze(self, file_path: str) -> dict:
        file_path = str(Path(file_path).resolve())
        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_type = self.preprocessor.detect_type(file_path)
        verbose = self.config["output"].get("verbose", True)

        if verbose:
            print(f"\n[Pipeline] 开始分析: {Path(file_path).name}（类型: {file_type}）")

        t0 = time.time()
        try:
            if file_type == "image":
                result = self._analyze_image(file_path, verbose)
            elif file_type == "video":
                result = self._analyze_video(file_path, verbose)
            elif file_type == "audio":
                result = self._analyze_audio(file_path, verbose)
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
        finally:
            if self.config["tmp"].get("cleanup", True):
                self.preprocessor.cleanup(file_path)

        elapsed = time.time() - t0
        if verbose:
            print(f"[Pipeline] 分析完成，耗时 {elapsed:.1f}s")

        if self.config["output"].get("save_results", True):
            self._save(result, file_path)

        return result

    # ── 分类处理 ──────────────────────────────────────────────────────────────

    def _analyze_image(self, file_path: str, verbose: bool) -> dict:
        if verbose:
            print("[Pipeline] 图片分析中...")
        return self.vision.analyze_image(file_path)

    def _analyze_video(self, file_path: str, verbose: bool) -> dict:
        duration = self.preprocessor.get_video_duration(file_path)
        threshold = self.config["video"]["short_video_threshold"]

        # 提取音轨转写
        transcript = ""
        if self.config["audio"]["extract_audio_from_video"]:
            if verbose:
                print("[Pipeline] 提取音轨并转写...")
            audio_path = self.preprocessor.extract_audio(file_path)
            if audio_path:
                transcript = self.audio.transcribe(audio_path)
                if verbose and transcript:
                    print(f"[Pipeline] 转写完成（{len(transcript)} 字）")

        if duration <= threshold:
            if verbose:
                print(f"[Pipeline] 短视频（{duration:.0f}s），原生推理中...")
            return self.vision.analyze_video_native(file_path, transcript)
        else:
            if verbose:
                print(f"[Pipeline] 长视频（{duration:.0f}s），提帧推理中...")
            frames = self.preprocessor.extract_frames(file_path)
            if verbose:
                print(f"[Pipeline] 提取 {len(frames)} 帧")
            return self.vision.analyze_frames(
                frames,
                transcript=transcript,
                filename=Path(file_path).name,
            )

    def _analyze_audio(self, file_path: str, verbose: bool) -> dict:
        if verbose:
            print("[Pipeline] 音频转写中...")
        transcript = self.audio.transcribe(file_path)
        if not transcript:
            raise RuntimeError("音频转写失败，无法继续分析")
        if verbose:
            print(f"[Pipeline] 转写完成（{len(transcript)} 字），推理中...")
        return self.vision.analyze_audio_text(transcript, Path(file_path).name)

    # ── 结果存储 ──────────────────────────────────────────────────────────────

    def _save(self, result: dict, file_path: str):
        out_dir = Path(self.config["output"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(file_path).stem
        out_path = out_dir / f"{stem}_result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[Pipeline] 结果已保存: {out_path}")
