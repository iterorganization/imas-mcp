"""Binary data transfer protocol for remote signal extraction.

Provides pack/unpack functions for numpy arrays using msgpack.
Falls back to JSON with tolist() when msgpack is unavailable.

Wire format for numpy arrays:
    {"__ndarray__": True, "shape": [...], "dtype": "...", "data": bytes}

Usage:
    # Client-side (decode remote output)
    from imas_codex.remote.serialization import decode_extraction_output

    results = decode_extraction_output(raw_bytes)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def pack_array(arr: Any) -> dict | Any:
    """Pack a numpy array into a msgpack-compatible dict.

    Args:
        arr: numpy array or scalar value.

    Returns:
        Dict with __ndarray__ marker for arrays, passthrough for scalars.
    """
    try:
        import numpy as np

        if isinstance(arr, np.ndarray):
            return {
                "__ndarray__": True,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "data": arr.tobytes(),
            }
    except ImportError:
        pass
    return arr


def unpack_array(obj: Any) -> Any:
    """Unpack a msgpack-encoded numpy array dict.

    Args:
        obj: Dict with __ndarray__ marker, or any other value.

    Returns:
        numpy array if __ndarray__ dict, otherwise passthrough.
    """
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        import numpy as np

        return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj


def decode_extraction_output(raw: bytes | str) -> dict[str, Any]:
    """Decode extraction script output (msgpack or JSON).

    Auto-detects format based on the first byte:
    - '{' → JSON
    - anything else → msgpack

    Args:
        raw: Raw output bytes or string from extraction script.

    Returns:
        Decoded results dict with signal extraction data.
        Array values in results are converted back to numpy arrays
        when msgpack format is detected.
    """
    if isinstance(raw, str):
        raw = raw.encode()

    if not raw:
        return {"results": {}}

    # Detect format
    first_byte = raw[0:1]
    if first_byte == b"{" or first_byte == b"[":
        # JSON format
        return json.loads(raw)

    # msgpack format
    try:
        import msgpack

        data = msgpack.unpackb(raw, raw=False)
        # Recursively unpack arrays
        return _unpack_recursive(data)
    except ImportError:
        logger.warning(
            "msgpack not available for decoding binary output, attempting JSON fallback"
        )
        return json.loads(raw)


def _unpack_recursive(obj: Any) -> Any:
    """Recursively unpack __ndarray__ dicts in nested structures."""
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            return unpack_array(obj)
        return {k: _unpack_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack_recursive(item) for item in obj]
    return obj
