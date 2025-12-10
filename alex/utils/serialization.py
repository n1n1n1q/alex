"""
Serialization utilities for game state and data structures.

Separated from extractor.py to improve separation of concerns.
"""

from __future__ import annotations

from dataclasses import is_dataclass, fields
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
import json


def to_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable form.
    
    Handles:
    - Dataclasses
    - Numpy arrays and dtypes
    - Nested dictionaries and sequences
    - Primitive types
    """
    if is_dataclass(obj):
        return {f.name: to_serializable(getattr(obj, f.name)) for f in fields(obj)}
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    if isinstance(obj, (np.integer,)):
        return int(obj)
    
    if isinstance(obj, (np.floating,)):
        return float(obj)
    
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(k)
            out[key] = to_serializable(v)
        return out
    
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [to_serializable(x) for x in obj]
    
    return obj


def to_json_str(obj: Any, indent: int = 2) -> str:
    """
    Convert object to JSON string.
    """
    return json.dumps(obj, default=to_serializable, ensure_ascii=False, indent=indent)


def to_json_file(obj: Any, filepath: str, indent: int = 2) -> None:
    """
    Save object to JSON file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=to_serializable, ensure_ascii=False, indent=indent)


__all__ = ['to_serializable', 'to_json_str', 'to_json_file']
