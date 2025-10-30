from dataclasses import dataclass
from typing import List, Optional, Dict
from bisect import bisect_left


@dataclass
class Stamped:
    t: float
    data: object


class TopicBuffer:
    """Keeps time-sorted messages and supports nearest lookup within tolerance."""

    def __init__(self):
        self.t: List[float] = []
        self.d: List[object] = []

    def add(self, t: float, data: object):
        # Append, but keep sorted amortized by append + sort on finalize if needed.
        self.t.append(t)
        self.d.append(data)

    def finalize(self):
        if len(self.t) > 1 and any(self.t[i] > self.t[i + 1] for i in range(len(self.t) - 1)):
            pairs = sorted(zip(self.t, self.d), key=lambda x: x[0])
            self.t, self.d = [p[0] for p in pairs], [p[1] for p in pairs]

    def nearest(self, tref: float, tol: float) -> Optional[Stamped]:
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
