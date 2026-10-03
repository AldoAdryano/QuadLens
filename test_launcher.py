import json
import os
import tempfile
import unittest
from unittest import mock

import launcher as L


class TestState(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "state.json")

    def tearDown(self):
        self.dir.cleanup()

    def test_load_missing_returns_empty(self):
        self.assertEqual(L.load_state(self.path), {})

    def test_save_and_load_roundtrip(self):
        L.save_state({"last_source": "http://x/video", "last_wifi": "http://x/video"}, self.path)
        self.assertEqual(L.load_state(self.path)["last_source"], "http://x/video")
        self.assertEqual(L.load_state(self.path)["last_wifi"], "http://x/video")

    def test_load_corrupt_returns_empty(self):
        with open(self.path, "w") as f:
            f.write("{not json")
        self.assertEqual(L.load_state(self.path), {})


class TestProbeUrl(unittest.TestCase):
    @mock.patch("launcher.urllib.request.urlopen")
    def test_success_returns_status(self, m):
        m.return_value.__enter__.return_value.status = 200
        self.assertTrue(L.probe_url("http://1.2.3.4:8080/video", timeout=0.2))

    @mock.patch("launcher.urllib.request.urlopen")
    def test_exception_returns_false(self, m):
        m.side_effect = OSError("refused")
        self.assertFalse(L.probe_url("http://1.2.3.4:8080/video", timeout=0.2))


class TestDetectCamera(unittest.TestCase):
    @mock.patch("cv2.VideoCapture")
    def test_present(self, m):
        m.return_value.isOpened.return_value = True
        self.assertTrue(L.detect_camera())
        m.return_value.release.assert_called_once()

    @mock.patch("cv2.VideoCapture")
    def test_absent(self, m):
        m.return_value.isOpened.return_value = False
        self.assertFalse(L.detect_camera())

    def test_cv2_missing_returns_false(self):
        with mock.patch.dict("sys.modules", {"cv2": None}):
            self.assertFalse(L.detect_camera())


class TestVenvPython(unittest.TestCase):
    @mock.patch("launcher.os.name", "nt")
    def test_windows(self):
        p = L.venv_python("/app")
        self.assertEqual(os.path.basename(p), "python.exe")
        self.assertIn(os.path.join("venv", "Scripts"), p)

    @mock.patch("launcher.os.name", "posix")
    def test_posix(self):
        p = L.venv_python("/app")
        self.assertEqual(os.path.basename(p), "python")
        self.assertIn(os.path.join("venv", "bin"), p)


class TestFindAdb(unittest.TestCase):
    @mock.patch("launcher.os.path.exists", return_value=False)
    def test_falls_back_to_path(self, m):
        self.assertEqual(L.find_adb(), "adb")

    @mock.patch("launcher.os.path.exists", return_value=True)
    def test_home_platform_tools(self, m):
        self.assertTrue(L.find_adb().endswith(os.path.join("platform-tools", "adb")))

    @mock.patch("launcher.os.path.exists", side_effect=[False, True])
    def test_windows_exe(self, m):
        self.assertTrue(L.find_adb().endswith("adb.exe"))


ARP_WINDOWS = """Interface: 192.168.1.7 --- 0x5
  Internet Address      Physical Address      Type
  192.168.1.5           aa-bb-cc-dd-ee-ff     dynamic
  10.0.0.4              11-22-33-44-55-66     dynamic
  192.168.1.5           aa-bb-cc-dd-ee-ff     dynamic
  169.254.3.3           99-88-77-66-55-44     invalid
"""


class TestWifiIps(unittest.TestCase):
    @mock.patch("launcher.subprocess.run")
    def test_linux_ip_neigh(self, m):
        m.return_value = mock.Mock(stdout="192.168.100.6 dev wlp2s0 lladdr aa:bb:cc REACHABLE\n")
        ips = L.wifi_ips()
        self.assertEqual(ips, ["192.168.100.6"])
        self.assertEqual(m.call_count, 1)
        self.assertIn("neigh", m.call_args[0][0])

    @mock.patch("launcher.subprocess.run")
    def test_falls_back_to_arp_a(self, m):
        m.side_effect = [FileNotFoundError("no ip"), mock.Mock(stdout=ARP_WINDOWS)]
        ips = L.wifi_ips()
        self.assertEqual(ips, ["192.168.1.5", "10.0.0.4"])
        self.assertEqual(m.call_count, 2)
        self.assertEqual(m.call_args_list[1][0][0][0], "arp")

    @mock.patch("launcher.subprocess.run", side_effect=FileNotFoundError)
    def test_no_commands_returns_empty(self, m):
        self.assertEqual(L.wifi_ips(), [])


class TestDetectUsb(unittest.TestCase):
    @mock.patch("launcher.probe_url", return_value=True)
    @mock.patch("launcher.run_adb", side_effect=["List of devices attached\nXYZ\tdevice\n", "18080"])
    def test_ready(self, run_adb, probe):
        r = L.detect_usb()
        self.assertEqual(r["status"], "ready")
        self.assertEqual(r["url"], "http://127.0.0.1:18080/video")
        self.assertTrue(any("forward" in a for a in run_adb.call_args_list[1][0][0]))

    @mock.patch("launcher.probe_url", return_value=False)
    @mock.patch("launcher.run_adb", side_effect=["List of devices attached\n\n", ""])
    def test_no_device(self, run_adb, probe):
        self.assertEqual(L.detect_usb()["status"], "no_device")

    @mock.patch("launcher.probe_url", return_value=False)
    @mock.patch("launcher.run_adb", side_effect=["List of devices attached\nXYZ\tdevice\n", "18080"])
    def test_device_but_app_down(self, run_adb, probe):
        r = L.detect_usb()
        self.assertEqual(r["status"], "app_down")
        self.assertEqual(r["url"], "http://127.0.0.1:18080/video")


class TestDetectWifi(unittest.TestCase):
    def test_prefers_saved_and_finds_neighbor(self):
        with mock.patch.object(L, "wifi_ips", return_value=["192.168.100.6"]), \
             mock.patch.object(L, "probe_url", side_effect=lambda u, timeout=0.5: "100.6" in u):
            url, status = L.detect_wifi("http://192.168.100.9:8080/video")
        self.assertEqual(url, "http://192.168.100.6:8080/video")
        self.assertEqual(status, "ready")

    def test_saved_responds(self):
        with mock.patch.object(L, "wifi_ips", return_value=[]), \
             mock.patch.object(L, "probe_url", return_value=True):
            url, status = L.detect_wifi("http://192.168.100.9:8080/video")
        self.assertEqual(url, "http://192.168.100.9:8080/video")
        self.assertEqual(status, "ready")

    def test_nothing_responds_keeps_saved(self):
        with mock.patch.object(L, "wifi_ips", return_value=["192.168.100.6"]), \
             mock.patch.object(L, "probe_url", return_value=False):
            url, status = L.detect_wifi("http://192.168.100.9:8080/video")
        self.assertEqual(url, "http://192.168.100.9:8080/video")
        self.assertEqual(status, "down")

    def test_nothing_at_all(self):
        with mock.patch.object(L, "wifi_ips", return_value=[]), \
             mock.patch.object(L, "probe_url", return_value=False):
            url, status = L.detect_wifi(None)
        self.assertIsNone(url)
        self.assertEqual(status, "none")


class TestDefaultChoice(unittest.TestCase):
    def test_last_wifi_wins(self):
        src = L.choose_default("http://192.168.100.9:8080/video", True, "http://192.168.100.9:8080/video", "http://127.0.0.1:18080/video")
        self.assertEqual(src, "http://192.168.100.9:8080/video")

    def test_last_camera(self):
        src = L.choose_default("0", True, None, None)
        self.assertEqual(src, "0")

    def test_last_usb(self):
        src = L.choose_default("http://127.0.0.1:18080/video", False, None, "http://127.0.0.1:18080/video")
        self.assertEqual(src, "http://127.0.0.1:18080/video")

    def test_none_when_stale(self):
        self.assertIsNone(L.choose_default("http://old:8080/video", True, "http://new:8080/video", None))

    def test_last_camera_any_index(self):
        self.assertEqual(L.choose_default("1", True, None, None), "1")


class TestMenu(unittest.TestCase):
    def test_menu_lines_and_default(self):
        entries = L.build_menu(
            camera_ok=True,
            wifi=("http://192.168.100.6:8080/video", "ready"),
            usb={"status": "ready", "url": "http://127.0.0.1:18080/video"},
            last_source="0",
        )
        labels = [e["label"] for e in entries]
        self.assertEqual(len(labels), 5)
        self.assertTrue(any("laptop" in l for l in labels))
        eksternal = next(e for e in entries if "eksternal" in e["label"])
        self.assertEqual(eksternal["source"], L.CAM_INDEX)
        self.assertNotIn("/dev/video0", entries[0]["status"],
                         "pesan status harus netral lintas-OS")
        default_idx = next(i for i, e in enumerate(entries) if e.get("source") == "0")
        self.assertEqual(default_idx, L.default_index(entries, "0"))


if __name__ == "__main__":
    unittest.main()
