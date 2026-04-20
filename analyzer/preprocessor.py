import os
import shutil
import subprocess
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}


class MediaPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.tmp_dir = Path(config["tmp"]["tmp_dir"])
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def detect_type(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext in IMAGE_EXTS:
            return "image"
        if ext in VIDEO_EXTS:
            return "video"
        if ext in AUDIO_EXTS:
            return "audio"
        raise ValueError(f"不支持的文件格式: {ext}")

    def get_video_duration(self, file_path: str) -> float:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())

    def extract_frames(self, file_path: str) -> list[str]:
        """按配置 fps 提帧，返回帧图片路径列表（按时间排序）。"""
        fps = self.config["video"]["extract_fps"]
        max_frames = self.config["video"]["max_frames"]
        out_dir = self.tmp_dir / Path(file_path).stem / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", file_path,
                "-vf", f"fps={fps}",
                "-q:v", "2",
                str(out_dir / "%06d.jpg"),
            ],
            capture_output=True,
            check=True,
        )

        frames = sorted(out_dir.glob("*.jpg"))
        if len(frames) > max_frames:
            # 均匀采样，保留 max_frames 帧
            step = len(frames) / max_frames
            frames = [frames[int(i * step)] for i in range(max_frames)]

        return [str(f) for f in frames]

    def extract_audio(self, file_path: str) -> str | None:
        """从视频中提取音轨，返回 wav 路径；若无音轨返回 None。"""
        if not self.config["audio"]["extract_audio_from_video"]:
            return None

        out_path = self.tmp_dir / Path(file_path).stem / "audio.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", file_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "16000",
                str(out_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            return None
        return str(out_path)

    def cleanup(self, file_path: str):
        tmp_subdir = self.tmp_dir / Path(file_path).stem
        if tmp_subdir.exists():
            shutil.rmtree(tmp_subdir)
