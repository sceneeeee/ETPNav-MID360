"""JSON conversion for evaluation diagnostics and forensic reports."""

import numpy as np


def json_safe(value):
    """Recursively convert NumPy values while leaving Python scalars unchanged."""
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {json_safe(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(json_safe(item) for item in value)
    return value
