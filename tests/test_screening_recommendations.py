from __future__ import annotations

from screening_recommendations import ROUTE_ASSESSMENTS, build_screening_plan


def test_no_input_plan_uses_nameplate_and_explicit_transmission_scenarios() -> None:
    plan = build_screening_plan()
    assert plan.wavelength_nm == 450
    assert plan.power_w == (1, 1.15)
    assert plan.field_cm2 == 10
    assert plan.exposure_s == 60
    assert plan.ideal_irradiance_mw_cm2 == (100, 115)
    assert plan.ideal_full_field_dose_mj_cm2 == (6000, 6900)
    assert tuple(row.transmission_pct for row in plan.scenarios) == (1, 5, 10, 25)
    assert plan.scenarios[2].irradiance_mw_cm2 == (10, 11.5)
    assert plan.scenarios[2].full_field_dose_mj_cm2 == (600, 690)


def test_plan_compares_all_four_hypotheses_without_resin_threshold() -> None:
    assert tuple(row[0] for row in ROUTE_ASSESSMENTS) == (
        "ordinary", "alternative", "gelatin", "phone",
    )
    assert "CAL" in ROUTE_ASSESSMENTS[2][1]
    assert "нельзя" in ROUTE_ASSESSMENTS[3][1]
