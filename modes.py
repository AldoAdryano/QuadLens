# modes.py
def is_L(f):
    return f["thumb"] and f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]


def is_tunjuk(f):
    return f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]


def is_tunjuk_ketat(f):
    return is_tunjuk(f) and not f["thumb"]


def is_telapak(f):
    return all(f.values())


def is_kepal(f):
    return not any(f.values())


def is_peace(f):
    return f["index"] and f["middle"] and not f["ring"] and not f["pinky"]


class HoldTransition:
    def __init__(self, hold_s=2.0, reset_grace_s=0.3, enter_grace_s=0.5):
        self.hold_s = hold_s
        self.reset_grace_s = reset_grace_s
        self.enter_grace_s = enter_grace_s
        self._arm_until = 0.0
        self._start = None
        self._last_active = None
        self._latched = False

    def arm(self, now):
        self._arm_until = now + self.enter_grace_s
        self._start = None
        self._last_active = None
        self._latched = False

    def update(self, active, now):
        if now < self._arm_until:
            return False
        if active:
            self._last_active = now
            if self._start is None:
                self._start = now
            if not self._latched and now - self._start >= self.hold_s:
                self._latched = True
                return True
            return False
        if self._start is not None and now - self._last_active > self.reset_grace_s:
            self._start = None
            self._latched = False
        return False


class TransitionController:
    TABLE = {
        "FILTERS": [("kepal", True, "LENSA"), ("tunjuk_ketat", False, "GAMBAR")],
        "LENSA": [("tunjuk_ketat", False, "GAMBAR"), ("kepal", False, "FILTERS")],
        "GAMBAR": [("telapak", False, "__sebelumnya__")],
    }
    GESTURES = {
        "kepal": is_kepal,
        "tunjuk_ketat": is_tunjuk_ketat,
        "telapak": is_telapak,
    }

    def __init__(self):
        self._holds = {
            mode: [HoldTransition() for _ in rows]
            for mode, rows in self.TABLE.items()
        }

    def arm(self, mode, now):
        for hold in self._holds[mode]:
            hold.arm(now)

    def check(self, mode, hands, now):
        for (gestur, dua_tangan, target), hold in zip(
            self.TABLE[mode], self._holds[mode]
        ):
            if dua_tangan:
                ok = (
                    hands.left is not None
                    and hands.right is not None
                    and is_kepal(hands.left.fingers)
                    and is_kepal(hands.right.fingers)
                )
            else:
                ok = any(self.GESTURES[gestur](x.fingers) for x in hands.all)
            if hold.update(ok, now):
                return target
        return None
