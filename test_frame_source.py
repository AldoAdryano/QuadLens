import time
import unittest

from frame_source import FrameSource


class FakeCap:
    def __init__(self, count=5):
        self.i = 0
        self.count = count
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.i < self.count:
            f = f"frame{self.i}"
            self.i += 1
            return (True, f)
        while not self.released:
            time.sleep(0.005)
        return (False, None)

    def release(self):
        self.released = True


class TestFrameSource(unittest.TestCase):
    def _wait_until(self, cond, timeout=2.0):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return True
            time.sleep(0.01)
        return False

    def test_initially_no_frame(self):
        src = FrameSource(0, cap=FakeCap(count=0))
        try:
            ok, frame = src.read()
            self.assertFalse(ok)
            self.assertIsNone(frame)
        finally:
            src.release()

    def test_returns_latest_not_queue(self):
        cap = FakeCap(count=5)
        src = FrameSource(0, cap=cap)
        try:
            self.assertTrue(self._wait_until(lambda: cap.i == 5))
            got = None

            def got_latest():
                nonlocal got
                ok, f = src.read()
                if ok:
                    got = f
                    return True
                return False

            self.assertTrue(self._wait_until(got_latest))
            self.assertEqual(got, "frame4")
        finally:
            src.release()

    def test_slow_consumer_still_fresh(self):
        cap = FakeCap(count=20)
        src = FrameSource(0, cap=cap)
        try:
            self.assertTrue(self._wait_until(lambda: cap.i == 20))
            time.sleep(0.05)
            ok, f = src.read()
            self.assertTrue(ok)
            self.assertEqual(f, "frame19")
        finally:
            src.release()

    def test_release_stops_thread(self):
        src = FrameSource(0, cap=FakeCap(count=1))
        time.sleep(0.1)
        src.release()
        self.assertFalse(src.alive)
        self.assertTrue(src.cap.released)

    def test_not_opened(self):
        class ClosedCap(FakeCap):
            def isOpened(self):
                return False

        src = FrameSource(0, cap=ClosedCap())
        try:
            self.assertFalse(src.isOpened())
        finally:
            src.release()


if __name__ == "__main__":
    unittest.main()
