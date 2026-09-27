from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from frame_io import FrameRepository, load_manifest, load_meta
from optical_frame_set import build_simulated_frame_set
from optical_simulation import OpticalScene, make_scene_with_indices, simulate_optical_projection


def _write_source_frames(directory, count: int = 4) -> None:
    directory.mkdir()
    height, width = 24, 48
    x = np.linspace(0.0, 1.0, width, dtype=np.float64)
    for index in range(count):
        frame = np.tile(np.roll(x, index * 3), (height, 1))
        Image.fromarray(np.rint(frame * 255.0).astype(np.uint8), mode="L").save(
            directory / f"frame_{index:04d}.png"
        )


def test_optical_simulation_writes_a_complete_reconstruction_set(tmp_path) -> None:
    source_dir = tmp_path / "frames"
    _write_source_frames(source_dir)
    scene = OpticalScene(ray_count=81)
    result = simulate_optical_projection(scene)

    simulated_dir = build_simulated_frame_set(str(source_dir), result, scene)

    info = FrameRepository.validate(simulated_dir)
    assert info.frame_count == 4
    assert info.width == 48
    assert info.height == 24
    assert load_meta(simulated_dir) is not None
    manifest = load_manifest(simulated_dir)
    assert manifest is not None
    assert manifest.complete
    assert manifest.slice_parameters["pipeline"] == "optical_simulation"
    assert manifest.frame_schedule["angles_deg"] == [90.0, 180.0, 270.0, 0.0]

    with Image.open(info.paths[0]) as image:
        simulated = np.asarray(image.convert("L"), dtype=np.uint8)
    with Image.open(source_dir / "frame_0000.png") as image:
        source = np.asarray(image.convert("L"), dtype=np.uint8)
    assert simulated.shape == source.shape
    assert not np.array_equal(simulated, source)


def test_optical_simulation_requires_multiple_projection_images(tmp_path) -> None:
    source_dir = tmp_path / "frames"
    _write_source_frames(source_dir, count=1)
    scene = OpticalScene(ray_count=41)
    result = simulate_optical_projection(scene)

    with pytest.raises(ValueError, match="как минимум два"):
        build_simulated_frame_set(str(source_dir), result, scene)


def test_water_dry_run_keeps_geometry_without_claiming_resin_dose(tmp_path) -> None:
    source_dir = tmp_path / "frames"
    _write_source_frames(source_dir)
    scene = make_scene_with_indices(
        resin_profile_key="water_dry_run",
        ray_count=81,
        source_efficiency=0.0,
    )
    result = simulate_optical_projection(scene)

    simulated_dir = build_simulated_frame_set(
        str(source_dir), result, scene, apply_resin_dose=False
    )

    manifest = load_manifest(simulated_dir)
    assert manifest is not None
    assert manifest.resin_profile["optical_only"] is True
    assert manifest.resin_profile["critical_exposure_mj_cm2"] is None
    assert manifest.slice_parameters["resin_dose_response_applied"] is False
    with Image.open(FrameRepository.validate(simulated_dir).paths[0]) as image:
        assert np.asarray(image.convert("L"), dtype=np.uint8).max() > 0
