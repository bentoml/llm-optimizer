"""Common utility functions and classes for llm-optimizer."""

import json
import math


class InfinityToNullEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts float('inf') to null."""

    def encode(self, o):
        if self._check_for_inf(o):
            # Convert infinity values to None before encoding
            o = self._replace_inf_with_none(o)
        return super().encode(o)

    def _check_for_inf(self, obj):
        """Recursively check if object contains infinity values."""
        if isinstance(obj, float) and math.isinf(obj):
            return True
        elif isinstance(obj, dict):
            return any(self._check_for_inf(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return any(self._check_for_inf(item) for item in obj)
        return False

    def _replace_inf_with_none(self, obj):
        """Recursively replace infinity values with None."""
        if isinstance(obj, float) and math.isinf(obj):
            return None
        elif isinstance(obj, dict):
            return {k: self._replace_inf_with_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_inf_with_none(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._replace_inf_with_none(item) for item in obj)
        return obj
