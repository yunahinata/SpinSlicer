"""First-order light budget for a proposed CAL projection source.

The calculation deliberately stops at an irradiance/dose envelope.  A source
power specification alone cannot establish a resin's spectral response or the
three-dimensional dose contrast of a tomographic print.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LightBudgetInput:
    """Optical inputs, with unknown downstream measurements left as ``None``."""

    wavelength_nm: float = 450.0
    power_min_w: float = 1.0
    power_max_w: float = 1.15
    image_width_mm: float = 20.0
    image_height_mm: float = 50.0
    exposure_s: float = 60.0
    transmission_pct: float | None = None
    measured_irradiance_mw_cm2: float | None = None
    measured_gel_dose_mj_cm2: float | None = None

    def validate(self) -> None:
        for name in (
            "wavelength_nm", "power_min_w", "power_max_w", "image_width_mm",
            "image_height_mm", "exposure_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.power_min_w > self.power_max_w:
            raise ValueError("minimum power cannot exceed maximum power")
        if not 200 <= self.wavelength_nm <= 1200:
            raise ValueError("wavelength must be between 200 and 1200 nm")
        for name in (
            "transmission_pct", "measured_irradiance_mw_cm2",
            "measured_gel_dose_mj_cm2",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when supplied")
        if self.transmission_pct is not None and self.transmission_pct > 100:
            raise ValueError("transmission cannot exceed 100%")


@dataclass(frozen=True, slots=True)
class LightBudgetResult:
    area_cm2: float
    source_irradiance_mw_cm2: tuple[float, float]
    plane_irradiance_mw_cm2: tuple[float, float] | None
    plane_dose_mj_cm2: tuple[float, float] | None
    gel_dose_ratio: tuple[float, float] | None
    basis: str
    missing: tuple[str, ...]
    warnings: tuple[str, ...]


def evaluate_light_budget(inputs: LightBudgetInput) -> LightBudgetResult:
    """Return an average-field envelope and keep unsupported outputs blank."""

    inputs.validate()
    area_cm2 = inputs.image_width_mm * inputs.image_height_mm / 100.0
    source = (
        inputs.power_min_w * 1000.0 / area_cm2,
        inputs.power_max_w * 1000.0 / area_cm2,
    )
    plane: tuple[float, float] | None = None
    basis = "Паспорт источника: идеальная верхняя граница до потерь"
    if inputs.measured_irradiance_mw_cm2 is not None:
        measured = inputs.measured_irradiance_mw_cm2
        plane = (measured, measured)
        basis = "Измеренная средняя интенсивность в плоскости колбы"
    elif inputs.transmission_pct is not None:
        fraction = inputs.transmission_pct / 100.0
        plane = (source[0] * fraction, source[1] * fraction)
        basis = "Условная оценка по заданному пропусканию тракта"

    dose = (
        (plane[0] * inputs.exposure_s, plane[1] * inputs.exposure_s)
        if plane is not None else None
    )
    ratio = (
        (dose[0] / inputs.measured_gel_dose_mj_cm2,
         dose[1] / inputs.measured_gel_dose_mj_cm2)
        if dose is not None and inputs.measured_gel_dose_mj_cm2 is not None else None
    )

    missing: list[str] = []
    if plane is None:
        missing.append("Пропускание всего тракта или интенсивность в плоскости колбы")
    if inputs.measured_gel_dose_mj_cm2 is None:
        missing.append("Порог гелеобразования конкретного материала при этой длине волны")
    missing.append("Карта неравномерности и фоновой засветки для оценки CAL-контраста")
    warnings: list[str] = []
    if (inputs.measured_irradiance_mw_cm2 is not None
            and inputs.measured_irradiance_mw_cm2 > source[1] * 1.05):
        warnings.append("Измеренная интенсивность выше предела по паспорту и площади: проверьте площадь, мощность и калибровку датчика.")
    if inputs.measured_irradiance_mw_cm2 is not None and inputs.transmission_pct is not None:
        warnings.append("Приоритет отдан измеренной интенсивности; введённое пропускание в расчёте не использовано.")
    return LightBudgetResult(
        area_cm2=area_cm2,
        source_irradiance_mw_cm2=source,
        plane_irradiance_mw_cm2=plane,
        plane_dose_mj_cm2=dose,
        gel_dose_ratio=ratio,
        basis=basis,
        missing=tuple(missing),
        warnings=tuple(warnings),
    )


def what_if_transmission(inputs: LightBudgetInput, percent: float) -> tuple[float, float]:
    """Irradiance envelope for a *hypothetical* total transmission."""

    candidate = LightBudgetInput(
        wavelength_nm=inputs.wavelength_nm,
        power_min_w=inputs.power_min_w,
        power_max_w=inputs.power_max_w,
        image_width_mm=inputs.image_width_mm,
        image_height_mm=inputs.image_height_mm,
        exposure_s=inputs.exposure_s,
        transmission_pct=percent,
    )
    result = evaluate_light_budget(candidate)
    assert result.plane_irradiance_mw_cm2 is not None
    return result.plane_irradiance_mw_cm2
