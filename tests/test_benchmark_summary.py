import json
import tempfile
import unittest
from pathlib import Path

from eval.summarize_benchmark import _metrics_sources, aggregate_rows, score_rows


class BenchmarkSummaryTests(unittest.TestCase):
    def test_scores_schema_time_prefix_and_keywords(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "result.json"
            result_path.write_text(
                json.dumps({
                    "file": "a.jpg",
                    "思考过程": "画面中有人群和标语",
                    "事件": ["人群聚集并举起标语"],
                    "解读": "抗议活动现场",
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            metrics = [{
                "id": "sample",
                "model_name": "Qwen3-VL-2B-Instruct",
                "media_type": "image",
                "duration_bucket": "not_video",
                "latency_seconds": 1.5,
                "result_path": str(result_path),
                "success": True,
            }]
            manifest = {
                "sample": {
                    "expected_keywords": ["人群", "标语", "抗议"],
                    "expected_category": "demonstration",
                }
            }

            scored = score_rows(metrics, manifest)

            self.assertTrue(scored[0]["json_valid"])
            self.assertTrue(scored[0]["schema_pass"])
            self.assertEqual(scored[0]["event_time_rule_rate"], 1.0)
            self.assertEqual(scored[0]["keyword_recall"], 1.0)

    def test_aggregates_rates_and_latency(self):
        rows = [
            {
                "model_name": "m",
                "media_type": "video",
                "duration_bucket": "short",
                "success": True,
                "json_valid": True,
                "schema_pass": True,
                "event_time_rule_rate": 1.0,
                "keyword_recall": 0.5,
                "latency_seconds": 10.0,
            },
            {
                "model_name": "m",
                "media_type": "video",
                "duration_bucket": "short",
                "success": False,
                "json_valid": False,
                "schema_pass": False,
                "event_time_rule_rate": 0.0,
                "keyword_recall": 0.0,
                "latency_seconds": 20.0,
            },
        ]

        summary = aggregate_rows(rows)

        self.assertEqual(summary[0]["count"], 2)
        self.assertEqual(summary[0]["success_rate"], 0.5)
        self.assertEqual(summary[0]["avg_latency_seconds"], 15.0)

    def test_loads_metrics_from_model_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "model-a" / "benchmark_metrics.jsonl"
            metrics_path.parent.mkdir()
            metrics_path.write_text('{"id": "sample"}\n', encoding="utf-8")

            sources = _metrics_sources(None, tmp_dir)

            self.assertEqual(sources[0][0], metrics_path)
            self.assertEqual(sources[0][1], [{"id": "sample"}])


if __name__ == "__main__":
    unittest.main()
