import json
import os
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def venv_python(here=HERE):
    if os.name == "nt":
        return os.path.join(here, "venv", "Scripts", "python.exe")
    return os.path.join(here, "venv", "bin", "python")


VENV_PY = venv_python()
STATE_PATH = os.path.expanduser("~/.config/filters/state.json")
USB_PORT = 18080
USB_URL = f"http://127.0.0.1:{USB_PORT}/video"
CAM_SRC = "0"
CAM_INDEX = "__cam__"
MANUAL = "__manual__"


def load_state(path=STATE_PATH):
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(data, path=STATE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def probe_url(url, timeout=1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def detect_camera():
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ok = bool(cap.isOpened())
        cap.release()
        return ok
    except Exception:
        return False


def find_adb():
    for cand in ("adb", "adb.exe"):
        p = os.path.expanduser(os.path.join("~", "platform-tools", cand))
        if os.path.exists(p):
            return p
    return "adb"


def run_adb(args, timeout=8):
    try:
        out = subprocess.run(
            [find_adb()] + args, capture_output=True, text=True, timeout=timeout
        )
        return out.stdout
    except Exception:
        return ""


def detect_usb():
    out = run_adb(["devices"])
    has_device = any(
        "\tdevice" in line for line in out.splitlines() if not line.startswith("List")
    )
    if not has_device:
        return {"status": "no_device", "url": USB_URL}
    run_adb(["forward", f"tcp:{USB_PORT}", "tcp:8080"])
    status = "ready" if probe_url(USB_URL) else "app_down"
    return {"status": status, "url": USB_URL}


def _priv(ip):
    if ip.startswith(("192.168.", "10.")):
        return True
    if ip.startswith("172."):
        parts = ip.split(".")
        return len(parts) > 1 and parts[1].isdigit() and 16 <= int(parts[1]) <= 31
    return False


def _parse_neigh(text):
    ips = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or "FAILED" in line:
            continue
        if parts[0] not in ips and _priv(parts[0]):
            ips.append(parts[0])
    return ips


def _parse_arp(text):
    ips = []
    for line in text.splitlines():
        parts = line.split()
        if parts and _priv(parts[0]) and parts[0] not in ips:
            ips.append(parts[0])
    return ips


def wifi_ips(dev="wlp2s0"):
    try:
        out = subprocess.run(
            ["ip", "neigh", "show", "dev", dev],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return _parse_neigh(out)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["arp", "-a"], capture_output=True, text=True, timeout=5,
        ).stdout
        return _parse_arp(out)
    except Exception:
        return []


def detect_wifi(saved_url=None):
    candidates = []
    if saved_url:
        candidates.append(saved_url)
    for ip in wifi_ips():
        url = f"http://{ip}:8080/video"
        if url not in candidates:
            candidates.append(url)
    for url in candidates[:6]:
        if probe_url(url, timeout=0.6):
            return (url, "ready")
    if saved_url:
        return (saved_url, "down")
    return (None, "none")


def choose_default(last_source, camera_ok, wifi_url, usb_url):
    if not last_source:
        return None
    if camera_ok and last_source.isdigit():
        return last_source
    if wifi_url and last_source == wifi_url:
        return wifi_url
    if usb_url and last_source == usb_url:
        return usb_url
    return None


def build_menu(camera_ok, wifi, usb, last_source):
    wifi_url, wifi_status = wifi
    entries = [
        {
            "label": "Kamera laptop",
            "source": CAM_SRC if camera_ok else None,
            "status": "siap" if camera_ok else "tidak ada",
        },
        {
            "label": f"HP via WiFi {wifi_url or ''}".strip(),
            "source": wifi_url,
            "status": {"ready": "siap", "down": "tidak merespons",
                       "none": "tidak terdeteksi"}[wifi_status],
        },
        {
            "label": "HP via USB",
            "source": usb["url"] if usb["status"] in ("ready", "app_down") else None,
            "status": {"ready": "siap (adb forward aktif)",
                       "app_down": "HP terdeteksi, IP Webcam tidak merespons",
                       "no_device": "tidak terdeteksi"}[usb["status"]],
        },
        {"label": "Kamera eksternal (indeks)", "source": CAM_INDEX, "status": "isi indeks"},
        {"label": "URL manual", "source": MANUAL, "status": "isi sendiri"},
    ]
    return entries


def default_index(entries, last_source):
    src = choose_default(
        last_source,
        True,
        next((e["source"] for e in entries if e["label"].startswith("HP via WiFi")), None),
        next((e["source"] for e in entries if e["label"] == "HP via USB"), None),
    )
    for i, e in enumerate(entries):
        if src and e["source"] == src:
            return i
    return 0


def run_check():
    state = load_state()
    camera_ok = detect_camera()
    wifi = detect_wifi(state.get("last_wifi"))
    usb = detect_usb()
    print(f"kamera laptop : {'OK' if camera_ok else 'tidak ada'}")
    print(f"HP WiFi       : {wifi[1]} {wifi[0] or ''}")
    print(f"HP USB        : {usb['status']} {usb['url']}")
    print(f"pilihan terakhir: {state.get('last_source') or '(belum ada)'}")


def main(argv=None):
    argv = argv if argv is not None else os.sys.argv[1:]
    if argv and argv[0] == "--check":
        run_check()
        return 0

    state = load_state()
    print("Deteksi sumber video...")
    camera_ok = detect_camera()
    wifi = detect_wifi(state.get("last_wifi"))
    usb = detect_usb()
    entries = build_menu(camera_ok, wifi, usb, state.get("last_source"))
    default = default_index(entries, state.get("last_source"))

    print()
    for i, e in enumerate(entries, 1):
        mark = "OK " if e["status"].startswith("siap") else "-- "
        print(f"  {i}. [{mark}{e['status']}] {e['label']}")
    print()

    while True:
        try:
            raw = input(f"Pilih sumber [{default + 1}]: ").strip()
        except EOFError:
            print("Dibatalkan.")
            return 1
        if raw == "":
            idx = default
        elif raw.isdigit() and 1 <= int(raw) <= len(entries):
            idx = int(raw) - 1
        else:
            print("Input tidak valid.")
            continue
        source = entries[idx]["source"]
        if source == CAM_INDEX:
            raw_idx = input("Indeks kamera [0]: ").strip() or "0"
            if not raw_idx.isdigit():
                continue
            source = raw_idx
        elif source == MANUAL:
            source = input("Masukkan URL (mis. http://IP:8080/video): ").strip()
            if not source:
                continue
        if source is None:
            print("Pilihan itu tidak tersedia, coba lagi.")
            continue
        break

    rc = subprocess.call([VENV_PY, os.path.join(HERE, "main.py"), "--source", source])
    if rc == 0 and source:
        state = load_state()
        state["last_source"] = source
        if source.startswith("http") and source != USB_URL:
            state["last_wifi"] = source
        save_state(state)
    return rc
