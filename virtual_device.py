"""Offline projector/rotation-stage playback for generated VAM jobs.

This module intentionally has no Qt or hardware dependency.  It validates the
same completed run that the UI consumes and emits deterministic playback
events.  A real projector or rotation-stage adapter can later implement the
same event contract without changing the slicer core.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from frame_io import FrameRepository, GenerationManifest, load_manifest, sha256_file
from validation import ValidationError


class PlaybackCancelled(Exception):
    """Raised when an offline or hardware playback request is cancelled."""


@dataclass(frozen=True)
class FramePlaybackEvent:
    """One frame presentation command in physical playback order."""

    index: int
    path: str
    angle_deg: float
    duration_s: float
    sha256: str


@dataclass(frozen=True)
class VirtualPlaybackReport:
    """Deterministic summary returned by :class:`VirtualPrinter`."""

    resolved_dir: str
    frame_count: int
    total_duration_s: float
    events: tuple[FramePlaybackEvent, ...]


EventCallback = Callable[[FramePlaybackEvent], None]
ProgressCallback = Callable[[float, FramePlaybackEvent], None]
CancelCheck = Callable[[], bool]


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Frame schedule {name} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValidationError(f"Frame schedule {name} must be finite.")
    return number


def _schedule(
    frame_count: int,
    manifest: Optional[GenerationManifest],
) -> tuple[list[float], float]:
    """Read a schedule, retaining compatibility with legacy frame folders."""

    schedule = manifest.frame_schedule if manifest is not None else {}
    if not isinstance(schedule, dict):
        raise ValidationError("Manifest frame_schedule must be an object.")

    raw_angles = schedule.get("angles_deg")
    if raw_angles is None:
        # Legacy runs were generated at 24 FPS with the original 90° offset.
        angles = [(90.0 + 360.0 * index / frame_count) % 360.0 for index in range(frame_count)]
    else:
        if not isinstance(raw_angles, list) or len(raw_angles) != frame_count:
            raise ValidationError("Frame schedule angle count does not match the frame set.")
        angles = [_finite(value, "angles_deg") for value in raw_angles]

    raw_duration = schedule.get("frame_duration_s")
    if raw_duration is None:
        raw_rate = schedule.get("frame_rate_hz", 24.0)
        rate = _finite(raw_rate, "frame_rate_hz")
        if rate <= 0.0:
            raise ValidationError("Frame schedule frame_rate_hz must be greater than zero.")
        duration = 1.0 / rate
    else:
        duration = _finite(raw_duration, "frame_duration_s")
        if duration <= 0.0:
            raise ValidationError("Frame schedule frame_duration_s must be greater than zero.")

    return angles, duration


class VirtualPrinter:
    """Replay a completed run without a projector or rotation stage.

    ``realtime=False`` is the default so tests and development tools run
    instantly.  Set ``realtime=True`` to exercise the same timing path that a
    future hardware adapter will use.
    """

    def __init__(self, realtime: bool = False) -> None:
        self.realtime = realtime

    def play(
        self,
        frames_dir: str,
        on_event: Optional[EventCallback] = None,
        progress_cb: Optional[ProgressCallback] = None,
        is_cancelled: Optional[CancelCheck] = None,
    ) -> VirtualPlaybackReport:
        frame_set = FrameRepository.validate(frames_dir)
        manifest = load_manifest(frame_set.resolved_dir)
        angles, duration = _schedule(frame_set.frame_count, manifest)
        events: list[FramePlaybackEvent] = []

        for index, path in enumerate(frame_set.paths):
            if is_cancelled is not None and is_cancelled():
                raise PlaybackCancelled()

            event = FramePlaybackEvent(
                index=index,
                path=path,
                angle_deg=angles[index],
                duration_s=duration,
                sha256=sha256_file(path),
            )
            if on_event is not None:
                on_event(event)
            if progress_cb is not None:
                progress_cb((index + 1) / frame_set.frame_count, event)
            events.append(event)

            if self.realtime:
                time.sleep(duration)

        return VirtualPlaybackReport(
            resolved_dir=frame_set.resolved_dir,
            frame_count=len(events),
            total_duration_s=duration * len(events),
            events=tuple(events),
        )
