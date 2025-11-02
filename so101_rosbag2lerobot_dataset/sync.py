# sync.py
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Stamped:
    """A lightweight container storing timestamped data."""

    t: float
    data: object


class TopicBuffer:
    """Keeps time-sorted messages and supports nearest lookup within tolerance."""

    def __init__(self):
        """Initialize an empty topic buffer."""

        self.t: List[float] = []
        self.d: List[object] = []

    def add(self, t: float, data: object):
        """Append a timestamped message to the buffer."""

        self.t.append(t)
        self.d.append(data)

    def finalize(self):
        """Ensure timestamps are sorted while keeping data aligned."""

        if len(self.t) > 1 and any(self.t[i] > self.t[i + 1] for i in range(len(self.t) - 1)):
            pairs = sorted(zip(self.t, self.d), key=lambda x: x[0])
            self.t, self.d = [p[0] for p in pairs], [p[1] for p in pairs]

    def nearest(self, tref: float, tol: float) -> Optional[Stamped]:
        """Return the closest stamped message within ``tol`` seconds of ``tref``."""

        if not self.t:
            return None
        i = bisect_left(self.t, tref)
        candidates = []
        if i < len(self.t):
            candidates.append((abs(self.t[i] - tref), i))
        if i > 0:
            candidates.append((abs(self.t[i - 1] - tref), i - 1))
        if not candidates:
            return None
        best = min(candidates, key=lambda x: x[0])
        if best[0] <= tol:
            idx = best[1]
            return Stamped(self.t[idx], self.d[idx])
        return None

    def nearest_with_dt(self, tref: float, tol: float) -> Tuple[Optional[Stamped], Optional[float]]:
        """Return the closest stamped message along with its offset from ``tref``."""

        if not self.t:
            return None, None
        i = bisect_left(self.t, tref)
        candidates = []
        if i < len(self.t):
            candidates.append((abs(self.t[i] - tref), i))
        if i > 0:
            candidates.append((abs(self.t[i - 1] - tref), i - 1))
        if not candidates:
            return None, None
        best = min(candidates, key=lambda x: x[0])
        if best[0] <= tol:
            idx = best[1]
            dt = self.t[idx] - tref
            return Stamped(self.t[idx], self.d[idx]), dt
        return None, None


@dataclass
class SyncStats:
    """Accumulate statistics about synchronization success and drift."""

    tried: int = 0
    matched: int = 0
    abs_dts: Optional[List[float]] = None

    def add(self, matched: bool, dt: Optional[float]):
        """Record the outcome of a synchronization attempt."""

        if self.abs_dts is None:
            self.abs_dts = []
        self.tried += 1
        if matched and dt is not None:
            self.matched += 1
            self.abs_dts.append(abs(dt))

    def summary(self) -> Dict[str, float]:
        """Return descriptive statistics for the collected synchronization deltas."""

        if not self.abs_dts:
            return {
                "match_rate": 0.0,
                "median_abs_dt_s": float("nan"),
                "mean_abs_dt_s": float("nan"),
                "p95_abs_dt_s": float("nan"),
                "max_abs_dt_s": float("nan"),
            }
        arr = np.asarray(self.abs_dts, dtype=float)
        return {
            "match_rate": self.matched / max(1, self.tried),
            "median_abs_dt_s": float(np.median(arr)),
            "mean_abs_dt_s": float(np.mean(arr)),
            "p95_abs_dt_s": float(np.percentile(arr, 95)),
            "max_abs_dt_s": float(np.max(arr)),
        }
