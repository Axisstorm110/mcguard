from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .utils import as_1d, check_same_length

@dataclass(frozen=True)
class ResamplingPlan:
    """
    Defines resampling structure.

    unit: cluster label (e.g., patient_id). Resampling happens at unit level.
    strata: optional labels for stratified resampling (e.g., site).
    blocks: optional labels for blocked bootstrap (e.g., time blocks).
    paired: optional pair_id for paired designs (used in permutation).
    """
    unit: np.ndarray | None = None
    strata: np.ndarray | None = None
    blocks: np.ndarray | None = None
    paired: np.ndarray | None = None

    def validate(self, *, y: np.ndarray, group: np.ndarray | None = None) -> None:
        y = as_1d(y)
        arrays = {"y": y}
        if group is not None:
            arrays["group"] = group
        if self.unit is not None:
            arrays["unit"] = self.unit
        if self.strata is not None:
            arrays["strata"] = self.strata
        if self.blocks is not None:
            arrays["blocks"] = self.blocks
        if self.paired is not None:
            arrays["paired"] = self.paired
        check_same_length(**arrays)

    def _group_keys(self, n: int) -> np.ndarray:
        parts = []
        if self.strata is not None:
            parts.append(as_1d(self.strata).astype(object))
        if self.blocks is not None:
            parts.append(as_1d(self.blocks).astype(object))
        if not parts:
            return np.array([None] * n, dtype=object)
        cols = np.stack(parts, axis=1)
        return np.array([tuple(row) for row in cols], dtype=object)

    def bucket_units(self, y: np.ndarray) -> dict[object, list[object]]:
        y = as_1d(y)
        n = len(y)
        self.validate(y=y)

        if self.unit is None:
            unit = np.arange(n, dtype=object)
        else:
            unit = as_1d(self.unit).astype(object)

        keys = self._group_keys(n)
        buckets: dict[object, list[object]] = {}
        for i in range(n):
            k = keys[i]
            buckets.setdefault(k, [])
            buckets[k].append(unit[i])

        for k in list(buckets.keys()):
            buckets[k] = list(dict.fromkeys(buckets[k]))

        return buckets
