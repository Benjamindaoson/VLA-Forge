"""Best-effort policy execution scheduling (NOT a safety-rated robot controller).

A real machine still needs independent hardware protection, certified motion
limits, a proper ROS2 driver and a human emergency-stop path.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


class ActionRejected(ValueError):
    pass


@dataclass(frozen=True)
class SchedulerConfig:
    action_dimensions: int = 7
    max_abs_command: float = 1.0
    max_observation_age_seconds: float = 0.20
    max_chunk_length: int = 50

    def __post_init__(self) -> None:
        if self.action_dimensions < 1 or self.max_abs_command <= 0:
            raise ValueError("Invalid action dimensions or command limits.")
        if self.max_observation_age_seconds <= 0 or self.max_chunk_length < 1:
            raise ValueError("Invalid scheduler timing/length settings.")


class ActionChunkScheduler:
    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig()
        self._actions: deque[tuple[float, ...]] = deque()
        self._observation_timestamp: float | None = None

    @property
    def remaining(self) -> int:
        return len(self._actions)

    def clear(self) -> None:
        self._actions.clear()
        self._observation_timestamp = None

    def _ensure_fresh(self, now_seconds: float) -> None:
        timestamp = self._observation_timestamp
        if timestamp is None or not math.isfinite(now_seconds):
            self.clear()
            raise ActionRejected("No valid observation timestamp.")
        age = now_seconds - timestamp
        if age < 0 or age > self.config.max_observation_age_seconds:
            self.clear()
            raise ActionRejected("Observation expired or timestamp is in the future.")

    def submit(
        self,
        actions: list[list[float]],
        *,
        observation_timestamp: float,
        now_seconds: float,
    ) -> None:
        """Atomically replace a chunk only after every element passes validation."""
        if not math.isfinite(observation_timestamp):
            raise ActionRejected("Invalid observation timestamp.")
        if not 1 <= len(actions) <= self.config.max_chunk_length:
            raise ActionRejected("Empty or excessive action chunk.")
        validated: list[tuple[float, ...]] = []
        for action in actions:
            if len(action) != self.config.action_dimensions:
                raise ActionRejected("Action dimension mismatch.")
            if any(not math.isfinite(x) or abs(x) > self.config.max_abs_command for x in action):
                raise ActionRejected("Non-finite or out-of-bounds action.")
            validated.append(tuple(action))
        if now_seconds < observation_timestamp or (
            now_seconds - observation_timestamp > self.config.max_observation_age_seconds
        ):
            raise ActionRejected("Stale observation; chunk not accepted.")
        self._actions = deque(validated)
        self._observation_timestamp = observation_timestamp

    def next_action(self, *, now_seconds: float) -> tuple[float, ...]:
        self._ensure_fresh(now_seconds)
        if not self._actions:
            raise ActionRejected("Action queue exhausted; request a new observation.")
        return self._actions.popleft()
