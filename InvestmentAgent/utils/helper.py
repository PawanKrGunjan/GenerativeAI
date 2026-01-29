# utils/helper.py
from __future__ import annotations

import io
import json
import contextlib
from typing import Any, Optional, Callable


def safe_float(x: Any) -> Optional[float]:
    """Convert to float safely; return None if conversion fails."""
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def safe_json_loads(text: str) -> Optional[dict]:
    """Parse JSON safely; return dict or None."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def quiet_call(fn: Callable[..., Any], *args, **kwargs):
    """Call a function while suppressing stdout (useful for noisy libraries)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)
