import os
import tempfile
import time
import unittest

import numpy as np

import capture


class TestSave(unittest.TestCase):
    def setUp(self):
        capture.reset_count()
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def test_creates_dir_and_returns_path(self):
        target = os.path.join(self.dir.name, "baru")
        img = np.zeros((10, 10, 3), np.uint8)
        path = capture.save(img, "LENS", folder=target)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.startswith(target))
        self.assertEqual(capture.photo_count(), 1)

    def test_unique_names_no_overwrite(self):
        img = np.zeros((10, 10, 3), np.uint8)
        p1 = capture.save(img, "LENS", folder=self.dir.name)
        p2 = capture.save(img, "LENS", folder=self.dir.name)
        self.assertNotEqual(p1, p2)
        self.assertEqual(capture.photo_count(), 2)

    def test_bad_folder_returns_none(self):
        img = np.zeros((10, 10, 3), np.uint8)
        path = capture.save(img, "LENS", folder="/proc/tidak/boleh/x")
        self.assertIsNone(path)
        self.assertEqual(capture.photo_count(), 0)


class TestFlash(unittest.TestCase):
    def test_flash_fades_to_normal(self):
        f = capture.Flash(durasi=0.30)
        frame = np.full((4, 4, 3), 100, np.uint8)
        f.trigger(0.0)
        mid = f.apply(frame.copy(), 0.15)
        self.assertGreater(mid.mean(), frame.mean())
        after = f.apply(frame.copy(), 0.5)
        self.assertEqual(after.mean(), frame.mean())


if __name__ == "__main__":
    unittest.main()
