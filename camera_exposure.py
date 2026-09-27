"""Illustrative side-camera long-exposure preview for CAL projection frames.

The camera image assumes weak *uniform* scattering. Real clear liquid can be
almost invisible from the side, while dust, bubbles and vial walls add glare.
The returned image is normalized and is never a radiometric measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image
from scipy.ndimage import rotate
from skimage.transform import resize

from frame_io import FrameRepository, load_manifest


@dataclass(frozen=True, slots=True)
class CameraExposureResult:
    dose_midplane: np.ndarray
    dose_projection: np.ndarray
    camera_image: np.ndarray
    frames_used: int
    frames_available: int
    rotation_fraction: float
    camera_angle_deg: float


def project_side_camera(volume: np.ndarray, angle_deg: float = 0.0) -> np.ndarray:
    """Integrate a (z, y, x) field along the camera line of sight."""

    if volume.ndim != 3 or min(volume.shape) < 2:
        raise ValueError("Для вида камеры нужен трёхмерный массив.")
    if not np.isfinite(volume).all() or not math.isfinite(angle_deg):
        raise ValueError("Поле и угол камеры должны быть конечными.")
    turned = rotate(volume, angle_deg, axes=(1, 2), reshape=False, order=1, mode="constant")
    return np.mean(turned, axis=1, dtype=np.float64).astype(np.float32)


def simulate_camera_long_exposure(
    frames_dir: str,
    *,
    shutter_s: float = 60.0,
    rotation_s: float = 60.0,
    camera_angle_deg: float = 0.0,
    grid_size: int = 80,
    max_frames: int = 120,
    is_cancelled: Callable[[], bool] | None = None,
) -> CameraExposureResult:
    """Contrast backprojected material dose with a synthetic side-camera view.

    Pixel values are relative.  The backprojection omits attenuation,
    diffusion, refraction and photochemistry.  The camera sees the *instantaneous*
    projected frames through a uniformly weakly scattering liquid; it has no
    access to the material's accumulated dose. A projection of that dose is
    returned separately for teaching, not as a prediction of the photograph.
    Exposure shorter than one turn captures only the first corresponding
    portion of the frame sequence.
    """

    if not math.isfinite(shutter_s) or shutter_s <= 0:
        raise ValueError("Выдержка камеры должна быть положительной.")
    if not math.isfinite(rotation_s) or rotation_s <= 0:
        raise ValueError("Время оборота должно быть положительным.")
    if not math.isfinite(camera_angle_deg):
        raise ValueError("Угол камеры должен быть конечным.")
    if not 24 <= grid_size <= 128 or not 2 <= max_frames <= 360:
        raise ValueError("Слишком высокое разрешение предпросмотра.")

    frame_set = FrameRepository.validate(frames_dir)
    if frame_set.frame_count < 2:
        raise ValueError("Для длинной выдержки нужны как минимум два кадра проекций.")
    rotation_fraction = shutter_s / rotation_s
    available = frame_set.frame_count
    captured = min(available, max(1, math.ceil(available * min(1.0, rotation_fraction))))
    indices = np.unique(np.linspace(0, captured - 1, min(captured, max_frames), dtype=int))
    manifest = load_manifest(frame_set.resolved_dir)
    scheduled_angles = (
        manifest.frame_schedule.get("angles_deg") if manifest is not None else None
    )
    if not isinstance(scheduled_angles, list) or len(scheduled_angles) != available:
        scheduled_angles = [
            float(angle)
            for angle in (np.linspace(0, 360, available, endpoint=False) + 90) % 360
        ]

    axis = np.arange(grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(axis - (grid_size - 1) / 2, axis - (grid_size - 1) / 2)
    volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)  # z, y, x
    frame_sum = np.zeros((grid_size, grid_size), dtype=np.float32)  # z, detector coordinate
    for index in indices:
        if is_cancelled is not None and is_cancelled():
            raise RuntimeError("Расчёт снимка отменён.")
        with Image.open(frame_set.paths[int(index)]) as image:
            source = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        frame = resize(
            source, (grid_size, grid_size), order=1, mode="edge",
            anti_aliasing=True, preserve_range=True,
        ).astype(np.float32)
        frame = np.flipud(frame)
        frame_sum += frame
        angle = math.radians(float(scheduled_angles[int(index)]))
        detector = xx * math.cos(angle) + yy * math.sin(angle) + (grid_size - 1) / 2
        low = np.floor(detector).astype(np.intp)
        high = low + 1
        fraction = detector - low
        mask = (low >= 0) & (high < grid_size)
        low = np.clip(low, 0, grid_size - 1)
        high = np.clip(high, 0, grid_size - 1)
        contribution = (
            frame[:, low] * (1 - fraction)[None, :, :]
            + frame[:, high] * fraction[None, :, :]
        )
        volume += contribution * mask[None, :, :]
    volume /= len(indices)

    dose_projection = project_side_camera(volume, camera_angle_deg)
    # The projector illuminates the same laboratory-space cylinder throughout
    # the rotation. The camera integrates that *instantaneous* light after weak
    # uniform scattering. Material dose instead follows each rotating voxel.
    frame_mean = frame_sum / len(indices)
    radius = (grid_size - 1) / 2
    vial_mask = (xx * xx + yy * yy <= radius * radius).astype(np.float32)
    illumination = frame_mean[:, :, None] * vial_mask[None, :, :]
    camera_image = project_side_camera(illumination, camera_angle_deg)
    return CameraExposureResult(
        dose_midplane=volume[grid_size // 2].copy(),
        dose_projection=dose_projection,
        camera_image=camera_image,
        frames_used=len(indices),
        frames_available=available,
        rotation_fraction=rotation_fraction,
        camera_angle_deg=camera_angle_deg,
    )
