import copy
import threading

import cv2


class FrameSource:
    """Membaca frame di thread terpisah; selalu menyimpan frame terbaru.

    Konsumen yang lambat memperoleh frame terkini (frame basi dibuang),
    sehingga latensi stream jaringan tidak menumpuk.
    """

    def __init__(self, source, cap=None):
        self.cap = cap if cap is not None else cv2.VideoCapture(source)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def alive(self):
        return self._thread.is_alive()

    def isOpened(self):
        return self.cap.isOpened()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                break
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            frame = self._frame
        if frame is None:
            return (False, None)
        return (True, copy.copy(frame))

    def release(self):
        self._running = False
        self.cap.release()
        self._thread.join(timeout=2.0)
