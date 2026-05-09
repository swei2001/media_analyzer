import sys
import types
import unittest


if "torch" not in sys.modules:
    torch = types.ModuleType("torch")
    torch.dtype = object
    torch.bfloat16 = object()
    torch.float16 = object()
    torch.float32 = object()
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace()
    sys.modules["torch"] = torch

if "transformers" not in sys.modules:
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForImageTextToText = object
    transformers.AutoModelForSpeechSeq2Seq = object
    transformers.AutoProcessor = object
    transformers.pipeline = lambda *args, **kwargs: None
    sys.modules["transformers"] = transformers

if "qwen_vl_utils" not in sys.modules:
    qwen_vl_utils = types.ModuleType("qwen_vl_utils")
    qwen_vl_utils.process_vision_info = lambda messages: (None, None)
    sys.modules["qwen_vl_utils"] = qwen_vl_utils

from analyzer.audio import AudioModel


class AudioTranscriptTests(unittest.TestCase):
    def test_formats_timestamped_transcript_chunks(self):
        result = {
            "text": "ignored",
            "chunks": [
                {"timestamp": (0.0, 3.4), "text": "第一句"},
                {"timestamp": (3.4, None), "text": "第二句"},
            ],
        }

        self.assertEqual(
            AudioModel._format_transcript(result),
            "[0.0-3.4秒] 第一句\n[3.4秒] 第二句",
        )

    def test_falls_back_to_plain_text_without_chunks(self):
        self.assertEqual(
            AudioModel._format_transcript({"text": "  plain text  "}),
            "plain text",
        )


if __name__ == "__main__":
    unittest.main()
