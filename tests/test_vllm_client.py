import argparse
import sys
import types
import unittest

import vllm_client


class VllmClientAudioCacheTests(unittest.TestCase):
    def setUp(self):
        self.old_audio_module = sys.modules.get("analyzer.audio")
        self.fake_audio_module = types.ModuleType("analyzer.audio")

        class FakeAudioModel:
            init_count = 0

            def __init__(self, config):
                type(self).init_count += 1
                self.config = config

            def transcribe(self, path):
                return f"transcript:{path}"

        self.fake_audio_model = FakeAudioModel
        self.fake_audio_module.AudioModel = FakeAudioModel
        sys.modules["analyzer.audio"] = self.fake_audio_module
        vllm_client._AUDIO_MODEL_CACHE.clear()

    def tearDown(self):
        vllm_client._AUDIO_MODEL_CACHE.clear()
        if self.old_audio_module is None:
            sys.modules.pop("analyzer.audio", None)
        else:
            sys.modules["analyzer.audio"] = self.old_audio_module

    def test_reuses_audio_model_for_same_config(self):
        args = argparse.Namespace(
            whisper_model="whisper",
            whisper_device="cuda",
            whisper_dtype="bfloat16",
            language="",
        )

        first = vllm_client._get_audio_model(args)
        second = vllm_client._get_audio_model(args)

        self.assertIs(first, second)
        self.assertEqual(self.fake_audio_model.init_count, 1)

    def test_uses_separate_cache_entry_for_different_language(self):
        base_args = argparse.Namespace(
            whisper_model="whisper",
            whisper_device="cuda",
            whisper_dtype="bfloat16",
            language="",
        )
        zh_args = argparse.Namespace(
            whisper_model="whisper",
            whisper_device="cuda",
            whisper_dtype="bfloat16",
            language="zh",
        )

        first = vllm_client._get_audio_model(base_args)
        second = vllm_client._get_audio_model(zh_args)

        self.assertIsNot(first, second)
        self.assertEqual(self.fake_audio_model.init_count, 2)


if __name__ == "__main__":
    unittest.main()
