import os
import unittest

import main


class TestModeKeys(unittest.TestCase):
    def test_flg_keys_map_to_modes(self):
        self.assertEqual(main.KEY_MODE[ord("f")], "FILTERS")
        self.assertEqual(main.KEY_MODE[ord("l")], "LENSA")
        self.assertEqual(main.KEY_MODE[ord("g")], "GAMBAR")

    def test_routing_lookup_safe_for_all_letters(self):
        for key in range(256):
            if key in main.KEY_MODE:
                self.assertIn(main.KEY_MODE[key], ("FILTERS", "LENSA", "GAMBAR"))

    def test_ffmpeg_low_latency_options_set_on_import(self):
        opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
        self.assertIn("nobuffer", opts, "stream FFmpeg harus dibuka tanpa buffer")
        self.assertIn("low_delay", opts)


if __name__ == "__main__":
    unittest.main()
