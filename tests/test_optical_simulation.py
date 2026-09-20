"""Numerical checks for the optical compensator model."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from optical_simulation import (
    OpticalScene,
    refract_direction,
    simulate_optical_projection,
    trace_ray,
    warp_projection_frame,
)


def test_normal_incidence_keeps_direction_and_has_fresnel_loss() -> None:
    direction, tir, transmission = refract_direction([1.0, 0.0], [-1.0, 0.0], 1.0, 1.5)

    assert not tir
    np.testing.assert_allclose(direction, [1.0, 0.0], atol=1e-12)
    assert 0.95 < transmission < 1.0


def test_snell_bends_toward_normal_when_entering_higher_index() -> None:
    direction, tir, _transmission = refract_direction([1.0, 0.2], [-1.0, 0.0], 1.0, 1.5)

    assert not tir
    assert direction[1] < 0.2
    assert direction[0] > 0.0


def test_total_internal_reflection_is_reported() -> None:
    direction, tir, transmission = refract_direction([1.0, 2.0], [-1.0, 0.0], 1.5, 1.0)

    assert tir
    assert transmission == pytest.approx(0.0)
    np.testing.assert_allclose(direction, np.array([1.0, 2.0]) / np.sqrt(5.0))


def test_central_ray_stays_on_axis_through_concentric_surfaces() -> None:
    scene = OpticalScene(horizontal_fov_deg=12.0)
    trace = trace_ray(scene, 0.0)

    assert trace.hit_resin
    assert trace.target_y_mm == pytest.approx(0.0, abs=1e-8)
    assert [segment.medium for segment in trace.segments] == [
        "Air",
        "Aquarium glass",
        "Water",
        "Vat glass",
        "Photopolymer",
    ]


def test_water_compensator_reduces_mapping_error_for_default_geometry() -> None:
    scene = OpticalScene(horizontal_fov_deg=12.0)
    with_water = simulate_optical_projection(scene)
    without_water = simulate_optical_projection(scene.without_compensator())

    assert with_water.coverage_pct == pytest.approx(100.0)
    assert without_water.coverage_pct == pytest.approx(100.0)
    assert with_water.mapping_error_mean_mm < without_water.mapping_error_mean_mm


def test_low_water_level_disables_compensator_for_selected_slice() -> None:
    scene = OpticalScene(
        horizontal_fov_deg=12.0,
        optical_slice_height_mm=50.0,
        water_level_mm=20.0,
    )
    assert not scene.compensator_active
    assert scene.warnings()

    result = simulate_optical_projection(scene)
    baseline = simulate_optical_projection(dataclasses.replace(scene, use_water_compensator=False))
    np.testing.assert_allclose(result.target_y_mm, baseline.target_y_mm, equal_nan=True)


def test_finite_aperture_clips_wide_rays() -> None:
    scene = OpticalScene(horizontal_fov_deg=40.0, ray_count=121)
    result = simulate_optical_projection(scene)

    assert result.coverage_pct < 100.0
    assert np.count_nonzero(result.clipped_mask) > 0
    assert np.count_nonzero(~result.hit_mask) > 0


def test_projection_frame_warp_preserves_shape_and_range() -> None:
    scene = OpticalScene(horizontal_fov_deg=12.0, ray_count=121)
    result = simulate_optical_projection(scene)
    frame = np.linspace(0.0, 1.0, 32 * 48, dtype=np.float64).reshape(32, 48)

    warped = warp_projection_frame(frame, result)

    assert warped.shape == frame.shape
    assert np.all(np.isfinite(warped))
    assert float(warped.min()) >= 0.0
    assert float(warped.max()) <= 1.0
