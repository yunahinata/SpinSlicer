from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from camera_exposure import project_side_camera, simulate_camera_long_exposure


def _frames(path, count: int = 12) -> None:
    path.mkdir()
    for index in range(count):
        frame = np.zeros((24, 32), dtype=np.uint8)
        frame[4:20, 8 + index % 8:12 + index % 8] = 255
        Image.fromarray(frame, mode="L").save(path / f"frame_{index:04d}.png")


def test_long_exposure_captures_more_frames_as_shutter_increases(tmp_path) -> None:
    path = tmp_path / "frames"
    _frames(path)
    short = simulate_camera_long_exposure(
        str(path), shutter_s=15, rotation_s=60, grid_size=32,
    )
    full = simulate_camera_long_exposure(
        str(path), shutter_s=60, rotation_s=60, grid_size=32,
    )
    assert short.frames_used == 3
    assert full.frames_used == 12
    assert full.dose_midplane.shape == (32, 32)
    assert full.dose_projection.shape == (32, 32)
    assert full.camera_image.shape == (32, 32)
    assert float(full.camera_image.max()) > 0
    assert not np.allclose(full.dose_projection, full.camera_image)


def test_one_camera_image_cannot_distinguish_depth(tmp_path) -> None:
    first = np.zeros((4, 4, 4), dtype=np.float32)
    second = np.zeros_like(first)
    first[1, 0, 2] = 1
    second[1, 3, 2] = 1
    np.testing.assert_array_equal(project_side_camera(first), project_side_camera(second))


def test_invalid_camera_exposure_rejected(tmp_path) -> None:
    path = tmp_path / "frames"
    _frames(path)
    with pytest.raises(ValueError):
        simulate_camera_long_exposure(str(path), shutter_s=0)
