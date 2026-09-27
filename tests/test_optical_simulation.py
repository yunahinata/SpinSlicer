"""Numerical checks for the optical compensator model."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from optical_simulation import (
    OpticalScene,
    make_scene_with_indices,
    make_test_projection_frame,
    refract_direction,
    simulate_optical_projection,
    trace_ray,
    warp_projection_frame,
)
from optical_tab import OpticalSimulationTab


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


def test_external_aquarium_wall_thickness_changes_path_and_mapping() -> None:
    thin_scene = OpticalScene(aquarium_wall_thickness_mm=0.5)
    thick_scene = OpticalScene(aquarium_wall_thickness_mm=15.0)

    thin = trace_ray(thin_scene, 0.04)
    thick = trace_ray(thick_scene, 0.04)

    thin_wall_path = sum(segment.length_mm for segment in thin.segments if segment.medium == "Aquarium glass")
    thick_wall_path = sum(segment.length_mm for segment in thick.segments if segment.medium == "Aquarium glass")
    assert thick_wall_path > thin_wall_path * 10.0
    assert thick.target_y_mm != pytest.approx(thin.target_y_mm)


def test_empty_compensator_keeps_external_aquarium_wall() -> None:
    trace = trace_ray(OpticalScene().without_compensator(), 0.0)

    assert [segment.medium for segment in trace.segments] == [
        "Air",
        "Aquarium glass",
        "Air",
        "Vat glass",
        "Photopolymer",
    ]


def test_material_presets_select_acrylic_and_glycerin() -> None:
    scene = make_scene_with_indices(
        aquarium_wall_material_key="acrylic",
        compensator_material_key="glycerin",
        vat_wall_material_key="acrylic",
    )
    trace = trace_ray(scene, 0.0)

    assert trace.hit_resin
    assert scene.aquarium_wall.refractive_index == pytest.approx(1.500)
    assert scene.water.refractive_index == pytest.approx(1.474)
    assert scene.aquarium_wall.name != scene.vat_wall.name
    assert [segment.medium for segment in trace.segments][1:4] == [
        scene.aquarium_wall.name,
        scene.water.name,
        scene.vat_wall.name,
    ]


def test_source_power_and_resin_working_curve_are_reported() -> None:
    scene = dataclasses.replace(OpticalScene(), exposure_time_s=5.0)
    result = simulate_optical_projection(scene)

    assert result.source_radiant_exposure_mj_cm2 == pytest.approx(222.222222, rel=1e-6)
    assert result.surface_dose_mj_cm2.shape == (scene.ray_count,)
    assert result.target_dose_mj_cm2.shape == (scene.ray_count,)
    assert result.cure_depth_mm.shape == (scene.ray_count,)
    assert result.max_cure_depth_mm > 0.0
    # Ordinary resin has a short 450 nm penetration depth, so the central
    # slice receives essentially no cure dose through a 60 mm vat.
    assert result.cured_fraction_pct == pytest.approx(0.0)


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


def test_default_test_projection_frame_is_a_valid_2d_source() -> None:
    frame = make_test_projection_frame(width=48, height=12)

    assert frame.shape == (12, 48)
    assert np.all(np.isfinite(frame))
    assert float(frame.min()) >= 0.0
    assert float(frame.max()) <= 1.0
    np.testing.assert_allclose(frame[0], frame[-1])


def test_optical_tab_uses_the_last_numbered_frame(tmp_path) -> None:
    for name in ("frame_0000.png", "frame_0002.png", "frame_0001.png"):
        (tmp_path / name).write_bytes(b"frame")

    selected = OpticalSimulationTab._find_latest_frame(str(tmp_path))

    assert selected is not None
    assert selected.endswith("frame_0002.png")
