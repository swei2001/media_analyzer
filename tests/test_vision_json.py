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

from analyzer.vision import (
    ModelOutputJSONError,
    _extract_json,
    _frame_timing_context,
    _normalize_image_result,
    _timing_rule,
)


class ExtractJSONTests(unittest.TestCase):
    def test_extracts_json_from_wrapped_model_output(self):
        raw = """<think>推理内容</think>
```json
{"file": "a.jpg", "思考过程": "依据", "事件": ["事件1"], "解读": "解读"}
```
"""

        self.assertEqual(_extract_json(raw)["file"], "a.jpg")

    def test_extracts_first_json_object_with_trailing_text(self):
        raw = """{"file": "a.jpg", "思考过程": "依据", "事件": [], "解读": "解读"}
额外说明"""

        self.assertEqual(_extract_json(raw)["解读"], "解读")

    def test_repairs_raw_newline_inside_string(self):
        raw = '{"file": "a.jpg", "思考过程": "第一行\n第二行", "事件": [], "解读": "解读"}'

        self.assertEqual(_extract_json(raw)["思考过程"], "第一行\n第二行")

    def test_raises_diagnostic_error_for_truncated_json(self):
        raw = '{"file": "a.jpg", "思考过程": "没有结束'

        with self.assertRaises(ModelOutputJSONError) as ctx:
            _extract_json(raw)

        self.assertIn("模型输出不是合法 JSON", str(ctx.exception))
        self.assertIn("原始输出片段", str(ctx.exception))

    def test_video_timing_rule_requires_second_prefix(self):
        rule = _timing_rule("video", duration=15.2)

        self.assertIn("起始秒-结束秒", rule)
        self.assertIn("0 到 15.2 秒", rule)

    def test_image_timing_rule_forbids_second_prefix(self):
        rule = _timing_rule("image")

        self.assertIn("不要以", rule)
        self.assertIn("0秒", rule)

    def test_image_result_strips_second_prefix(self):
        result = _normalize_image_result({
            "file": "a.jpg",
            "思考过程": "依据",
            "事件": ["0秒：人群聚集", "10-12秒：举起标语"],
            "解读": "解读",
        })

        self.assertEqual(result["事件"], ["人群聚集", "举起标语"])

    def test_frame_timing_context_lists_frame_seconds(self):
        context = _frame_timing_context([
            ("/tmp/000001.jpg", 0.0),
            ("/tmp/000016.jpg", 15.0),
        ])

        self.assertIn("000001.jpg: 0秒", context)
        self.assertIn("000016.jpg: 15秒", context)


if __name__ == "__main__":
    unittest.main()
