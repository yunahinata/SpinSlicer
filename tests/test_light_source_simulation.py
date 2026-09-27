"""Checks for the measurement boundaries in the CAL light budget."""

from __future__ import annotations

import pytest

from light_source_simulation import LightBudgetInput, evaluate_light_budget


def test_unknown_optical_path_does_not_produce_a_delivered_dose() -> None:
    result = evaluate_light_budget(LightBudgetInput())

    assert result.area_cm2 == 10.0
    assert result.source_irradiance_mw_cm2 == pytest.approx((100.0, 115.0))
    assert result.plane_irradiance_mw_cm2 is None
    assert result.plane_dose_mj_cm2 is None
    assert result.gel_dose_ratio is None


def test_conditional_transmission_uses_area_and_seconds_with_correct_units() -> None:
    result = evaluate_light_budget(
        LightBudgetInput(transmission_pct=10.0, measured_gel_dose_mj_cm2=300.0)
    )

    assert result.plane_irradiance_mw_cm2 == pytest.approx((10.0, 11.5))
    assert result.plane_dose_mj_cm2 == pytest.approx((600.0, 690.0))
    assert result.gel_dose_ratio == pytest.approx((2.0, 2.3))
    assert "Условная" in result.basis


def test_measured_plane_intensity_takes_priority_over_assumed_transmission() -> None:
    result = evaluate_light_budget(
        LightBudgetInput(transmission_pct=25.0, measured_irradiance_mw_cm2=7.0)
    )

    assert result.plane_irradiance_mw_cm2 == pytest.approx((7.0, 7.0))
    assert result.plane_dose_mj_cm2 == pytest.approx((420.0, 420.0))
    assert result.gel_dose_ratio is None
    assert any("приоритет" in warning.lower() for warning in result.warnings)


@pytest.mark.parametrize(
    "invalid",
    [
        LightBudgetInput(power_min_w=2.0, power_max_w=1.0),
        LightBudgetInput(image_width_mm=0),
        LightBudgetInput(transmission_pct=110),
    ],
)
def test_invalid_inputs_are_rejected(invalid: LightBudgetInput) -> None:
    with pytest.raises(ValueError):
        evaluate_light_budget(invalid)
