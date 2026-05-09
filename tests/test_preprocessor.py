import tempfile
import unittest

from analyzer.preprocessor import MediaPreprocessor


class PreprocessorFrameTimestampTests(unittest.TestCase):
    def test_estimates_timestamps_from_extracted_frame_numbers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            preprocessor = MediaPreprocessor({
                "tmp": {"tmp_dir": tmp_dir},
                "video": {"extract_fps": 2.0},
                "audio": {"extract_audio_from_video": True},
            })

            self.assertEqual(
                preprocessor.frame_timestamps([
                    "/tmp/000001.jpg",
                    "/tmp/000011.jpg",
                ]),
                [
                    ("/tmp/000001.jpg", 0.0),
                    ("/tmp/000011.jpg", 5.0),
                ],
            )


if __name__ == "__main__":
    unittest.main()
