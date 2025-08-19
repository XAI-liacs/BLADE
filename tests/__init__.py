try:
    import numpy as np

    if not hasattr(np, "byte"):
        np.byte = np.int8
except Exception:  # pragma: no cover - optional dependency
    pass
