import unittest

from main import parse_source


class TestParseSource(unittest.TestCase):
    def test_camera_index(self):
        self.assertEqual(parse_source("0"), 0)
        self.assertEqual(parse_source("1"), 1)

    def test_http_url(self):
        self.assertEqual(
            parse_source("http://192.168.42.107:8080/video"),
            "http://192.168.42.107:8080/video",
        )

    def test_rtsp_url(self):
        self.assertEqual(
            parse_source("rtsp://192.168.1.5:554/stream"),
            "rtsp://192.168.1.5:554/stream",
        )

    def test_non_numeric_string_stays_string(self):
        self.assertEqual(parse_source("webcam"), "webcam")


if __name__ == "__main__":
    unittest.main()
