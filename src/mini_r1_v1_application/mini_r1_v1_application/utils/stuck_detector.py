"""Stuck detection via displacement window — ported from legacy DisplacementWindowDetector."""

import math
import time


class StuckDetector:
    """Fires when the robot hasn't moved enough within a rolling time window.

    Usage:
        detector = StuckDetector(window_s=5.0, displacement_m=0.05)
        # In control loop:
        if detector.update(x, y, now):
            # robot is stuck
    """

    def __init__(self, window_s: float = 5.0, displacement_m: float = 0.05,
                 cooldown_s: float = 2.0):
        self._window_s = window_s
        self._displacement_m = displacement_m
        self._cooldown_s = cooldown_s
        self._history: list[tuple[float, float, float]] = []  # (x, y, time)
        self._last_trigger_time = 0.0

    def update(self, x: float, y: float, now: float) -> bool:
        """Add a pose sample and return True if stuck is detected."""
        self._history.append((x, y, now))

        # Prune old entries
        self._history = [(hx, hy, ht) for hx, hy, ht in self._history
                         if now - ht <= self._window_s]

        if len(self._history) < 2:
            return False

        # Cooldown check
        if now - self._last_trigger_time < self._cooldown_s:
            return False

        oldest = self._history[0]
        dx = x - oldest[0]
        dy = y - oldest[1]
        displacement = math.sqrt(dx * dx + dy * dy)
        elapsed = now - oldest[2]

        if elapsed >= self._window_s * 0.8 and displacement < self._displacement_m:
            self._last_trigger_time = now
            return True
        return False

    def reset(self):
        """Clear history (e.g. after receiving a new path)."""
        self._history.clear()
