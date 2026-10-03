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
