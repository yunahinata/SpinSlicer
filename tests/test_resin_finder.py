from __future__ import annotations

import pytest

from light_source_simulation import LightBudgetInput
from resin_finder import run_resin_search


def test_search_ranks_cal_reference_and_purchasable_daylight_candidate() -> None:
    result = run_resin_search()

    assert result.laser_band_nm == (440.0, 460.0)
    assert len(result.assessments) == 4
    assert result.assessments[0].candidate.key == "opencal_udma_cq_edab"
    assert result.assessments[0].status == "cal_reference"
    assert result.assessments[1].candidate.key == "photocentric_magna_draft"
    assert result.assessments[1].status == "edge_candidate"
    assert result.assessments[2].candidate.key == "photocentric_rigid_dlfr"
    assert result.assessments[2].status == "edge_candidate"
    assert result.assessments[3].candidate.key == "generic_405_resin"
    assert result.assessments[3].status == "weak_match"


def test_dose_sweep_is_explicitly_hypothetical_and_has_expected_bounds() -> None:
    result = run_resin_search()

    assert result.scenario_count == 30
    scenario = next(
        item for item in result.dose_scenarios
        if item.transmission_pct == 10 and item.exposure_s == 60
    )
    assert scenario.dose_mj_cm2 == pytest.approx((600, 690))
    assert len(result.missing_for_prediction) >= 3


def test_search_uses_changed_source_for_irradiance_and_band_center() -> None:
    result = run_resin_search(LightBudgetInput(wavelength_nm=470, power_min_w=0.5, power_max_w=1.0))

    assert result.laser_band_nm == (460.0, 480.0)
    scenario = next(
        item for item in result.dose_scenarios
        if item.transmission_pct == 10 and item.exposure_s == 1
    )
    assert scenario.dose_mj_cm2 == pytest.approx((5, 10))


def test_invalid_spectrum_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="Нижняя граница"):
        run_resin_search(laser_band_nm=(460, 440))
