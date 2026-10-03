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


if __name__ == "__main__":
    unittest.main()
