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
