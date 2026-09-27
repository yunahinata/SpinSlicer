from __future__ import annotations

from dataclasses import replace

import pytest

from development_paths import PathCase, compare_laser_to_baseline, evaluate_path
from light_source_simulation import LightBudgetInput


def test_ordinary_resin_remains_unknown_without_downstream_data() -> None:
    result = evaluate_path(PathCase("ordinary"), LightBudgetInput())
    assert result.ideal_irradiance_mw_cm2 == (100.0, 115.0)
    assert result.plane_irradiance_mw_cm2 is None
    assert result.threshold_ratio is None
    assert result.cal_margin_mj_cm2 is None


def test_hypothetical_transmission_is_not_measured_or_cal_proof() -> None:
    laser = LightBudgetInput(transmission_pct=10)
    result = evaluate_path(
        PathCase("ordinary", exposure_s=60, gel_threshold_mj_cm2=50,
                 threshold_evidence="scenario"), laser,
    )
    assert result.plane_irradiance_mw_cm2 == (10, 11.5)
    assert result.full_field_dose_mj_cm2 == (600, 690)
    assert result.threshold_ratio == (12, 13.8)
    assert result.plane_evidence == "scenario"
    assert result.cal_margin_mj_cm2 is None


def test_gelatin_requires_thermal_control_even_with_assumed_threshold() -> None:
    laser = LightBudgetInput(measured_irradiance_mw_cm2=5)
    case = PathCase(
        "gelatin", gel_threshold_mj_cm2=100, threshold_evidence="measured",
        voxel_inside_mj_cm2=120, voxel_outside_mj_cm2=20,
        voxel_evidence="measured",
    )
    before = evaluate_path(case, laser)
    assert before.threshold_ratio is None
    assert before.cal_margin_mj_cm2 is None
    after = evaluate_path(replace(case, thermal_control_passed=True), laser)
    assert after.cal_margin_mj_cm2 == 20


def test_camera_path_never_uses_phone_as_a_light_source() -> None:
    result = evaluate_path(PathCase("phone", exposure_s=30))
    assert result.plane_irradiance_mw_cm2 is None
    assert result.full_field_dose_mj_cm2 is None
    assert "Камера" in result.status


def test_laser_advantage_reports_only_supported_comparisons() -> None:
    laser = LightBudgetInput(measured_irradiance_mw_cm2=10)
    alternative = PathCase("alternative", strength_mpa=30)
    baseline = PathCase(
        "baseline", plane_irradiance_mw_cm2=5,
        plane_evidence="measured", strength_mpa=20,
    )
    comparison = compare_laser_to_baseline(
        evaluate_path(alternative, laser), evaluate_path(baseline),
        alternative, baseline,
    )
    assert comparison.intensity_ratio == (2, 2)
    assert comparison.strength_change_pct == 50
    assert comparison.evidence == "измерено"


def test_invalid_case_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_path(PathCase("ordinary", exposure_s=0), LightBudgetInput())
    with pytest.raises(ValueError):
        evaluate_path(PathCase("alternative", voxel_evidence="measured"), LightBudgetInput())
