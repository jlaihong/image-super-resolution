

def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def normalize_01(x):
    return x / 255.0


def denormalize_m11(x):
    return (x + 1) * 127.5
