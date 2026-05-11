#!/usr/bin/env python3
"""Download a small political-event evaluation media set."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import subprocess
import struct
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MediaAnalyzerEvalBot/1.0 (single-file parsing benchmark; contact: media-analyzer-local@example.invalid)"
DOWNLOAD_TIMEOUT = 20

TOPICS = [
    {
        "topic": "protest",
        "query": "protest demonstration crowd banner",
        "expected_category": "demonstration",
        "expected_keywords": ["抗议", "人群", "标语"],
    },
    {
        "topic": "anti_war_protest",
        "query": "anti war protest demonstration",
        "expected_category": "demonstration",
        "expected_keywords": ["反战", "抗议", "人群"],
    },
    {
        "topic": "election_rally",
        "query": "election rally political campaign crowd",
        "expected_category": "political_rally",
        "expected_keywords": ["选举", "集会", "人群"],
    },
    {
        "topic": "riot_police",
        "query": "riot police protest demonstration",
        "expected_category": "riot_or_intervention",
        "expected_keywords": ["警察", "抗议", "冲突"],
    },
    {
        "topic": "strike",
        "query": "workers strike protest demonstration",
        "expected_category": "demonstration",
        "expected_keywords": ["罢工", "抗议", "工人"],
    },
    {
        "topic": "parliament_protest",
        "query": "parliament protest demonstration",
        "expected_category": "demonstration",
        "expected_keywords": ["议会", "抗议", "人群"],
    },
    {
        "topic": "military_event",
        "query": "military conflict checkpoint civilians",
        "expected_category": "political_violence",
        "expected_keywords": ["军事", "人员", "现场"],
    },
    {
        "topic": "refugee_crisis",
        "query": "refugee crisis border crowd",
        "expected_category": "civilian_crisis",
        "expected_keywords": ["难民", "边境", "人群"],
    },
]

FALLBACK_TAGS = [
    ("protest", "protest,demonstration"),
    ("rally", "political,rally"),
    ("strike", "strike,protest"),
    ("police", "police,protest"),
    ("crowd", "crowd,demonstration"),
]

VIDEO_TOPICS = [
    {
        "topic": "protest_video",
        "query": 'protest demonstration insource:"video"',
        "expected_category": "demonstration",
        "expected_keywords": ["抗议", "人群", "现场"],
    },
    {
        "topic": "political_rally_video",
        "query": 'political rally campaign speech insource:"video"',
        "expected_category": "political_rally",
        "expected_keywords": ["集会", "演讲", "人群"],
    },
    {
        "topic": "parliament_video",
        "query": 'parliament protest political video',
        "expected_category": "political_event",
        "expected_keywords": ["议会", "政治", "现场"],
    },
    {
        "topic": "election_video",
        "query": 'election campaign political video',
        "expected_category": "political_rally",
        "expected_keywords": ["选举", "活动", "人物"],
    },
    {
        "topic": "press_conference_video",
        "query": 'government press conference political video',
        "expected_category": "official_statement",
        "expected_keywords": ["发布会", "讲话", "人物"],
    },
]

AUDIO_TOPICS = [
    {
        "topic": "political_speech_audio",
        "query": 'political speech audio',
        "expected_category": "speech_or_statement",
        "expected_keywords": ["讲话", "政治", "声音"],
    },
    {
        "topic": "campaign_speech_audio",
        "query": 'campaign speech audio election',
        "expected_category": "speech_or_statement",
        "expected_keywords": ["竞选", "讲话", "声音"],
    },
    {
        "topic": "parliament_speech_audio",
        "query": 'parliament speech audio',
        "expected_category": "speech_or_statement",
        "expected_keywords": ["议会", "讲话", "声音"],
    },
    {
        "topic": "protest_audio",
        "query": 'protest chant audio demonstration',
        "expected_category": "demonstration",
        "expected_keywords": ["抗议", "口号", "声音"],
    },
    {
        "topic": "government_statement_audio",
        "query": 'government statement audio politics',
        "expected_category": "speech_or_statement",
        "expected_keywords": ["政府", "声明", "声音"],
    },
    {
        "topic": "president_speech_audio",
        "query": 'president speech mp3',
        "expected_category": "speech_or_statement",
        "expected_keywords": ["总统", "讲话", "声音"],
    },
    {
        "topic": "minister_speech_audio",
        "query": 'minister speech mp3',
        "expected_category": "speech_or_statement",
        "expected_keywords": ["部长", "讲话", "声音"],
    },
    {
        "topic": "house_representatives_audio",
        "query": 'House of Representatives speech mp3',
        "expected_category": "speech_or_statement",
        "expected_keywords": ["议会", "讲话", "声音"],
    },
    {
        "topic": "freedom_speech_audio",
        "query": 'freedom speech audio mp3',
        "expected_category": "speech_or_statement",
        "expected_keywords": ["自由", "讲话", "声音"],
    },
    {
        "topic": "political_address_audio",
        "query": 'political address mp3',
        "expected_category": "speech_or_statement",
        "expected_keywords": ["政治", "讲话", "声音"],
    },
]

AUDIO_CATEGORY_TOPICS = [
    {
        "topic": "politician_voice_audio",
        "category": "Category:Audio files of politicians",
        "expected_category": "speech_or_statement",
        "expected_keywords": ["政治人物", "讲话", "声音"],
    },
    {
        "topic": "politics_audio",
        "category": "Category:Audio files about politics",
        "expected_category": "speech_or_statement",
        "expected_keywords": ["政治", "讲话", "声音"],
    },
    {
        "topic": "speech_audio",
        "category": "Category:Audio files of speeches",
        "expected_category": "speech_or_statement",
        "expected_keywords": ["演讲", "讲话", "声音"],
    },
    {
        "topic": "political_speech_audio",
        "category": "Category:Political speeches",
        "expected_category": "speech_or_statement",
        "expected_keywords": ["政治", "演讲", "声音"],
    },
]

VIDEO_MIMES = {"video/mp4", "video/webm", "video/ogg", "application/ogg"}
AUDIO_MIMES = {"audio/mpeg", "audio/mp3", "audio/ogg", "audio/wav", "audio/x-wav", "audio/flac"}
MEDIA_EXTENSIONS = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/ogg": ".ogv",
    "application/ogg": ".ogv",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
}


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_").lower() or "item"


def _request_json(params: dict[str, Any]) -> dict[str, Any]:
    url = COMMONS_API + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _search_commons(query: str, limit: int, extra_iiprop: str = "") -> list[dict[str, Any]]:
    iiprop = "url|mime|size|extmetadata"
    if extra_iiprop:
        iiprop += f"|{extra_iiprop}"
    data = _request_json({
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": query,
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": iiprop,
        "iiurlwidth": "640",
    })
    pages = data.get("query", {}).get("pages", {})
    return list(pages.values())


def _category_commons(category: str, limit: int, extra_iiprop: str = "") -> list[dict[str, Any]]:
    iiprop = "url|mime|size|extmetadata"
    if extra_iiprop:
        iiprop += f"|{extra_iiprop}"
    data = _request_json({
        "action": "query",
        "format": "json",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": iiprop,
    })
    pages = data.get("query", {}).get("pages", {})
    return list(pages.values())


def _extension_from_image(info: dict[str, Any]) -> str:
    url = info.get("thumburl") or info.get("url", "")
    ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if ext == ".jpeg" else ext
    guessed = mimetypes.guess_extension(info.get("mime", ""))
    if guessed in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if guessed == ".jpeg" else guessed
    return ".jpg"


def _clean_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse(parsed._replace(query=""))


def _metadata_value(extmetadata: dict[str, Any], key: str) -> str:
    value = extmetadata.get(key, {})
    if isinstance(value, dict):
        return str(value.get("value", "")).strip()
    return ""


def _parse_duration_text(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?", text):
        parts = [float(part) for part in text.split(":")]
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        return seconds
    minutes = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|minute|minutes|分钟)", text, re.I)
    seconds = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds|秒)", text, re.I)
    if minutes or seconds:
        return (float(minutes.group(1)) * 60 if minutes else 0.0) + (float(seconds.group(1)) if seconds else 0.0)
    return None


def _duration_from_commons_metadata(info: dict[str, Any]) -> float | None:
    for metadata in info.get("metadata") or []:
        name = str(metadata.get("name", "")).lower()
        if name in {"duration", "length", "playtime", "runtime"}:
            duration = _parse_duration_text(metadata.get("value"))
            if duration:
                return duration
    extmetadata = info.get("extmetadata") or {}
    for key in ("Duration", "Length", "Playtime"):
        duration = _parse_duration_text(_metadata_value(extmetadata, key))
        if duration:
            return duration
    return None


def _download_file(url: str, out_path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
        with out_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def _extension_from_media(info: dict[str, Any], allowed_mimes: set[str]) -> str | None:
    url = info.get("url", "")
    ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
    mime = info.get("mime", "")
    if mime in MEDIA_EXTENSIONS:
        return MEDIA_EXTENSIONS[mime]
    if mime not in allowed_mimes:
        return None
    guessed = mimetypes.guess_extension(mime)
    if guessed:
        return ".ogv" if guessed == ".oga" and mime.startswith("video/") else guessed
    return None


def _probe_duration_seconds(path: Path) -> float | None:
    if shutil.which("ffprobe"):
        try:
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
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            result = None

        if result and result.returncode == 0:
            duration = _parse_duration_text(result.stdout.strip())
            if duration:
                return duration

    if path.suffix.lower() in {".mp4", ".webm", ".ogv", ".avi", ".mov"}:
        try:
            import cv2  # type: ignore[import-not-found]

            capture = cv2.VideoCapture(str(path))
            frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()
            if frames > 0 and fps > 0:
                return float(frames / fps)
        except Exception:  # noqa: BLE001
            return None

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as audio:
                return float(audio.getnframes() / audio.getframerate())
        except Exception:  # noqa: BLE001
            return None

    if path.suffix.lower() == ".mp3":
        return _probe_mp3_duration(path)

    if path.suffix.lower() == ".ogg":
        return _probe_ogg_duration(path)

    if path.suffix.lower() == ".flac":
        return _probe_flac_duration(path)

    return None


def _probe_mp3_duration(path: Path) -> float | None:
    data = path.read_bytes()
    offset = 0
    if data.startswith(b"ID3") and len(data) >= 10:
        size = 0
        for byte in data[6:10]:
            size = (size << 7) | (byte & 0x7F)
        offset = 10 + size

    bitrates = {
        (3, 1): [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],
        (3, 2): [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0],
        (3, 3): [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],
        (2, 1): [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0],
        (2, 2): [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
        (2, 3): [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
    }
    sample_rates = {
        3: [44100, 48000, 32000, 0],
        2: [22050, 24000, 16000, 0],
        0: [11025, 12000, 8000, 0],
    }
    samples_per_frame = {
        (3, 1): 384,
        (3, 2): 1152,
        (3, 3): 1152,
        (2, 1): 384,
        (2, 2): 1152,
        (2, 3): 576,
        (0, 1): 384,
        (0, 2): 1152,
        (0, 3): 576,
    }

    durations: list[float] = []
    cursor = offset
    while cursor + 4 <= len(data) and len(durations) < 3000:
        header = int.from_bytes(data[cursor:cursor + 4], "big")
        if (header & 0xFFE00000) != 0xFFE00000:
            cursor += 1
            continue
        version = (header >> 19) & 0x3
        layer = (header >> 17) & 0x3
        bitrate_idx = (header >> 12) & 0xF
        sample_rate_idx = (header >> 10) & 0x3
        padding = (header >> 9) & 0x1
        bitrate = bitrates.get((version, layer), [0] * 16)[bitrate_idx] * 1000
        sample_rate = sample_rates.get(version, [0] * 4)[sample_rate_idx]
        frame_samples = samples_per_frame.get((version, layer), 0)
        if bitrate <= 0 or sample_rate <= 0 or frame_samples <= 0:
            cursor += 1
            continue
        if layer == 1:
            frame_size = int(((12 * bitrate / sample_rate) + padding) * 4)
        elif version == 3:
            frame_size = int((144 * bitrate / sample_rate) + padding)
        else:
            frame_size = int((72 * bitrate / sample_rate) + padding)
        if frame_size <= 4:
            cursor += 1
            continue
        durations.append(frame_samples / sample_rate)
        cursor += frame_size

    if durations:
        return float(sum(durations))
    return None


def _probe_ogg_duration(path: Path) -> float | None:
    data = path.read_bytes()
    sample_rate: int | None = None
    pre_skip = 0
    if b"OpusHead" in data[:65536]:
        idx = data.find(b"OpusHead")
        if idx >= 0 and idx + 16 <= len(data):
            pre_skip = int.from_bytes(data[idx + 10:idx + 12], "little")
            sample_rate = 48000
    vorbis_idx = data.find(b"\x01vorbis", 0, 65536)
    if sample_rate is None and vorbis_idx >= 0 and vorbis_idx + 16 <= len(data):
        sample_rate = int.from_bytes(data[vorbis_idx + 12:vorbis_idx + 16], "little")

    last_granule: int | None = None
    cursor = 0
    while True:
        page = data.find(b"OggS", cursor)
        if page < 0 or page + 27 > len(data):
            break
        granule = struct.unpack("<q", data[page + 6:page + 14])[0]
        segments = data[page + 26]
        segment_table_end = page + 27 + segments
        if segment_table_end > len(data):
            break
        body_size = sum(data[page + 27:segment_table_end])
        cursor = segment_table_end + body_size
        if granule >= 0:
            last_granule = granule

    if sample_rate and last_granule is not None and last_granule > pre_skip:
        return float((last_granule - pre_skip) / sample_rate)
    return None


def _probe_flac_duration(path: Path) -> float | None:
    data = path.read_bytes()
    if not data.startswith(b"fLaC") or len(data) < 42:
        return None
    cursor = 4
    while cursor + 4 <= len(data):
        header = data[cursor]
        block_type = header & 0x7F
        length = int.from_bytes(data[cursor + 1:cursor + 4], "big")
        cursor += 4
        block = data[cursor:cursor + length]
        if block_type == 0 and len(block) >= 18:
            value = int.from_bytes(block[10:18], "big")
            sample_rate = (value >> 44) & 0xFFFFF
            total_samples = value & 0xFFFFFFFFF
            if sample_rate and total_samples:
                return float(total_samples / sample_rate)
            return None
        cursor += length
    return None


def _trim_media(path: Path, max_duration: float) -> bool:
    if path.suffix.lower() == ".mp3":
        return _trim_mp3(path, max_duration)
    if not shutil.which("ffmpeg"):
        return False
    tmp_path = path.with_name(f"{path.stem}.trim{path.suffix}")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-t",
            f"{max_duration:.3f}",
            "-map",
            "0",
            "-c",
            "copy",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        return False
    tmp_path.replace(path)
    return True


def _trim_mp3(path: Path, max_duration: float) -> bool:
    data = path.read_bytes()
    offset = 0
    if data.startswith(b"ID3") and len(data) >= 10:
        size = 0
        for byte in data[6:10]:
            size = (size << 7) | (byte & 0x7F)
        offset = 10 + size

    bitrates = {
        (3, 1): [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],
        (3, 2): [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0],
        (3, 3): [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],
        (2, 1): [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0],
        (2, 2): [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
        (2, 3): [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
    }
    sample_rates = {
        3: [44100, 48000, 32000, 0],
        2: [22050, 24000, 16000, 0],
        0: [11025, 12000, 8000, 0],
    }
    samples_per_frame = {
        (3, 1): 384,
        (3, 2): 1152,
        (3, 3): 1152,
        (2, 1): 384,
        (2, 2): 1152,
        (2, 3): 576,
        (0, 1): 384,
        (0, 2): 1152,
        (0, 3): 576,
    }

    output = bytearray(data[:offset])
    duration = 0.0
    cursor = offset
    while cursor + 4 <= len(data) and duration < max_duration:
        header = int.from_bytes(data[cursor:cursor + 4], "big")
        if (header & 0xFFE00000) != 0xFFE00000:
            cursor += 1
            continue
        version = (header >> 19) & 0x3
        layer = (header >> 17) & 0x3
        bitrate_idx = (header >> 12) & 0xF
        sample_rate_idx = (header >> 10) & 0x3
        padding = (header >> 9) & 0x1
        bitrate = bitrates.get((version, layer), [0] * 16)[bitrate_idx] * 1000
        sample_rate = sample_rates.get(version, [0] * 4)[sample_rate_idx]
        frame_samples = samples_per_frame.get((version, layer), 0)
        if bitrate <= 0 or sample_rate <= 0 or frame_samples <= 0:
            cursor += 1
            continue
        if layer == 1:
            frame_size = int(((12 * bitrate / sample_rate) + padding) * 4)
        elif version == 3:
            frame_size = int((144 * bitrate / sample_rate) + padding)
        else:
            frame_size = int((72 * bitrate / sample_rate) + padding)
        if frame_size <= 4 or cursor + frame_size > len(data):
            break
        frame_duration = frame_samples / sample_rate
        if duration + frame_duration > max_duration:
            break
        output.extend(data[cursor:cursor + frame_size])
        duration += frame_duration
        cursor += frame_size

    if duration <= 0 or len(output) >= len(data):
        return False
    path.write_bytes(output)
    return True


def _download_fallback(output_dir: Path, start_idx: int, count: int, sleep: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lock = 1000
    max_lock = lock + count * 4
    while len(rows) < count and lock <= max_lock:
        topic, tags = FALLBACK_TAGS[len(rows) % len(FALLBACK_TAGS)]
        idx = start_idx + len(rows)
        url = f"https://loremflickr.com/640/480/{urllib.parse.quote(tags)}?lock={lock + idx}"
        out_path = output_dir / f"political_{idx:03d}_fallback_{topic}.jpg"
        try:
            _download_file(url, out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[Download] fallback 下载失败 {url}: {exc}")
            lock += 1
            continue
        rows.append({
            "id": out_path.stem,
            "file": str(out_path.relative_to(REPO_ROOT)),
            "media_type": "image",
            "topic": topic,
            "expected_category": "demonstration",
            "expected_keywords": ["抗议", "人群"],
            "source": "LoremFlickr tag query",
            "source_url": url,
            "download_url": url,
            "license": "see source service",
            "attribution": "",
        })
        print(f"[Download] fallback 保存 {out_path}")
        time.sleep(sleep)
    return rows


def download_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    per_topic = max(1, args.per_topic)
    for topic in TOPICS:
        if len(manifest_rows) >= args.count:
            break
        print(f"[Download] 搜索 {topic['topic']}: {topic['query']}")
        try:
            pages = _search_commons(topic["query"], args.search_limit)
        except Exception as exc:  # noqa: BLE001
            print(f"[Download] 搜索失败: {exc}")
            continue

        topic_added = 0
        topic_failures = 0
        for page in pages:
            if len(manifest_rows) >= args.count or topic_added >= per_topic:
                break
            if topic_failures >= args.max_failures_per_topic:
                print(f"[Download] {topic['topic']} 连续失败较多，切换下一个 topic")
                break
            imageinfo = (page.get("imageinfo") or [{}])[0]
            url = _clean_url(imageinfo.get("thumburl") or imageinfo.get("url") or "")
            mime = imageinfo.get("mime", "")
            if not url or url in seen_urls or mime not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            if imageinfo.get("width", 0) < 300 or imageinfo.get("height", 0) < 200:
                continue

            ext = _extension_from_image(imageinfo)
            idx = len(manifest_rows) + 1
            filename = f"political_{idx:03d}_{_slug(topic['topic'])}{ext}"
            out_path = output_dir / filename
            try:
                _download_file(url, out_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[Download] 下载失败 {url}: {exc}")
                topic_failures += 1
                if "HTTP Error 429" in str(exc):
                    time.sleep(max(args.sleep, 1.0))
                continue

            extmetadata = imageinfo.get("extmetadata") or {}
            manifest_rows.append({
                "id": out_path.stem,
                "file": str(out_path.relative_to(REPO_ROOT)),
                "media_type": "image",
                "topic": topic["topic"],
                "expected_category": topic["expected_category"],
                "expected_keywords": topic["expected_keywords"],
                "source": "Wikimedia Commons",
                "source_title": page.get("title", ""),
                "source_url": imageinfo.get("descriptionurl", ""),
                "download_url": url,
                "original_url": imageinfo.get("url", ""),
                "license": _metadata_value(extmetadata, "LicenseShortName"),
                "attribution": _metadata_value(extmetadata, "Attribution"),
            })
            seen_urls.add(url)
            topic_added += 1
            topic_failures = 0
            print(f"[Download] 保存 {out_path}")
            time.sleep(args.sleep)

    if len(manifest_rows) < args.count and not args.no_fallback:
        need = args.count - len(manifest_rows)
        print(f"[Download] Commons 仅下载 {len(manifest_rows)} 个，使用 fallback 补齐 {need} 个")
        manifest_rows.extend(_download_fallback(
            output_dir=output_dir,
            start_idx=len(manifest_rows) + 1,
            count=need,
            sleep=args.sleep,
        ))

    return manifest_rows


def download_commons_timed_media(
    *,
    output_dir: Path,
    media_type: str,
    topics: list[dict[str, Any]],
    count: int,
    start_idx: int,
    search_limit: int,
    sleep: float,
    max_duration: float,
    trim_long_media: bool,
    max_size_mb: float,
    max_failures_per_topic: int,
    skip_urls: set[str] | None = None,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []

    allowed_mimes = VIDEO_MIMES if media_type == "video" else AUDIO_MIMES
    rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set(skip_urls or set())
    topic_cursor = 0

    def try_add_page(page: dict[str, Any], topic: dict[str, Any]) -> bool:
        imageinfo = (page.get("imageinfo") or [{}])[0]
        url = _clean_url(imageinfo.get("url") or "")
        mime = imageinfo.get("mime", "")
        if not url or url in seen_urls or mime not in allowed_mimes:
            return False
        byte_size = float(imageinfo.get("size") or 0)
        if byte_size > max_size_mb * 1024 * 1024:
            return False

        ext = _extension_from_media(imageinfo, allowed_mimes)
        if not ext:
            return False
        metadata_duration = _duration_from_commons_metadata(imageinfo)
        can_trim = trim_long_media and (shutil.which("ffmpeg") or ext == ".mp3")
        if metadata_duration and metadata_duration > max_duration and not can_trim:
            return False
        idx = start_idx + len(rows)
        filename = f"political_{idx:03d}_{_slug(topic['topic'])}{ext}"
        out_path = output_dir / filename
        try:
            _download_file(url, out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[Download] 下载失败 {url}: {exc}")
            if "HTTP Error 429" in str(exc):
                time.sleep(max(sleep, 1.0))
            return False

        source_duration = _probe_duration_seconds(out_path) or metadata_duration
        duration = source_duration
        was_trimmed = False
        if duration is not None and duration > max_duration and trim_long_media:
            if _trim_media(out_path, max_duration):
                duration = _probe_duration_seconds(out_path)
                was_trimmed = True

        if duration is None or duration <= 0 or duration > max_duration + 0.5:
            reason = "无法读取时长" if duration is None else f"时长 {duration:.1f}s 超过 {max_duration:.1f}s"
            print(f"[Download] 跳过 {out_path.name}: {reason}")
            out_path.unlink(missing_ok=True)
            return False

        extmetadata = imageinfo.get("extmetadata") or {}
        rows.append({
            "id": out_path.stem,
            "file": str(out_path.relative_to(REPO_ROOT)),
            "media_type": media_type,
            "duration_seconds": round(duration, 3),
            "source_duration_seconds": round(source_duration, 3) if source_duration else None,
            "trimmed": was_trimmed,
            "topic": topic["topic"],
            "expected_category": topic["expected_category"],
            "expected_keywords": topic["expected_keywords"],
            "source": "Wikimedia Commons",
            "source_title": page.get("title", ""),
            "source_url": imageinfo.get("descriptionurl", ""),
            "download_url": url,
            "original_url": imageinfo.get("url", ""),
            "license": _metadata_value(extmetadata, "LicenseShortName"),
            "attribution": _metadata_value(extmetadata, "Attribution"),
        })
        seen_urls.add(url)
        trim_note = ", trimmed" if was_trimmed else ""
        print(f"[Download] 保存 {out_path} ({duration:.1f}s{trim_note})")
        time.sleep(sleep)
        return True

    while len(rows) < count and topic_cursor < len(topics) * 3:
        topic = topics[topic_cursor % len(topics)]
        topic_cursor += 1
        print(f"[Download] 搜索 {media_type} {topic['topic']}: {topic['query']}")
        try:
            pages = _search_commons(topic["query"], search_limit, extra_iiprop="metadata")
        except Exception as exc:  # noqa: BLE001
            print(f"[Download] 搜索失败: {exc}")
            continue

        topic_failures = 0
        for page in pages:
            if len(rows) >= count:
                break
            if topic_failures >= max_failures_per_topic:
                print(f"[Download] {topic['topic']} 连续失败较多，切换下一个 topic")
                break
            if not try_add_page(page, topic):
                topic_failures += 1
                continue
            topic_failures = 0

    if media_type == "audio" and len(rows) < count:
        for topic in AUDIO_CATEGORY_TOPICS:
            if len(rows) >= count:
                break
            print(f"[Download] 分类补充 audio {topic['topic']}: {topic['category']}")
            try:
                pages = _category_commons(topic["category"], search_limit, extra_iiprop="metadata")
            except Exception as exc:  # noqa: BLE001
                print(f"[Download] 分类搜索失败: {exc}")
                continue
            category_failures = 0
            for page in pages:
                if len(rows) >= count:
                    break
                if category_failures >= max_failures_per_topic:
                    break
                if not try_add_page(page, topic):
                    category_failures += 1
                    continue
                category_failures = 0

    if len(rows) < count:
        print(f"[Download] {media_type} 仅下载 {len(rows)} / {count} 个，可能需要提高 --search-limit 或放宽查询词")
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="下载小型政治事件评测媒体集")
    parser.add_argument("--output-dir", default="demo_media/political_events")
    parser.add_argument("--manifest", default="eval/manifest.jsonl")
    parser.add_argument("--count", type=int, default=10, help="图片默认下载数量")
    parser.add_argument("--video-count", type=int, default=10, help="视频默认下载数量")
    parser.add_argument("--audio-count", type=int, default=10, help="音频默认下载数量")
    parser.add_argument("--max-duration", type=float, default=60.0, help="视频/音频最大时长秒数")
    parser.add_argument("--skip-long-media", action="store_true", help="跳过超过 --max-duration 的远程视频/音频，不裁剪")
    parser.add_argument("--max-video-mb", type=float, default=80.0, help="下载视频候选的最大文件大小")
    parser.add_argument("--max-audio-mb", type=float, default=40.0, help="下载音频候选的最大文件大小")
    parser.add_argument("--per-topic", type=int, default=5)
    parser.add_argument("--search-limit", type=int, default=40)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--max-failures-per-topic", type=int, default=8)
    parser.add_argument("--no-fallback", action="store_true", help="Commons 不足时不使用 LoremFlickr 补齐")
    parser.add_argument("--append", action="store_true", help="追加到已有 manifest，不重写已有行")
    args = parser.parse_args()

    output_dir = _resolve_path(args.output_dir)
    manifest_path = _resolve_path(args.manifest)
    rows = read_manifest(manifest_path) if args.append else []
    existing_urls = {
        _clean_url(str(row.get("download_url") or row.get("original_url") or ""))
        for row in rows
        if row.get("download_url") or row.get("original_url")
    }

    new_rows = download_dataset(args)
    rows.extend(new_rows)
    existing_urls.update(
        _clean_url(str(row.get("download_url") or row.get("original_url") or ""))
        for row in new_rows
        if row.get("download_url") or row.get("original_url")
    )
    video_rows = download_commons_timed_media(
        output_dir=output_dir,
        media_type="video",
        topics=VIDEO_TOPICS,
        count=args.video_count,
        start_idx=len(rows) + 1,
        search_limit=args.search_limit,
        sleep=args.sleep,
        max_duration=args.max_duration,
        trim_long_media=not args.skip_long_media,
        max_size_mb=args.max_video_mb,
        max_failures_per_topic=args.max_failures_per_topic,
        skip_urls=existing_urls,
    )
    rows.extend(video_rows)
    existing_urls.update(
        _clean_url(str(row.get("download_url") or row.get("original_url") or ""))
        for row in video_rows
        if row.get("download_url") or row.get("original_url")
    )
    rows.extend(download_commons_timed_media(
        output_dir=output_dir,
        media_type="audio",
        topics=AUDIO_TOPICS,
        count=args.audio_count,
        start_idx=len(rows) + 1,
        search_limit=args.search_limit,
        sleep=args.sleep,
        max_duration=args.max_duration,
        trim_long_media=not args.skip_long_media,
        max_size_mb=args.max_audio_mb,
        max_failures_per_topic=args.max_failures_per_topic,
        skip_urls=existing_urls,
    ))
    write_manifest(manifest_path, rows)
    print(f"[Download] 下载 {len(rows)} 个媒体文件，manifest 写入: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
