"""Evidence-gated screening of four possible development paths.

The model deliberately does not infer photochemistry from a matching colour.
Its dose is for a uniformly lit field; a CAL voxel receives a different,
angle-dependent history.  Separate in/out voxel doses are needed for even a
preliminary CAL contrast check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from light_source_simulation import LightBudgetInput, evaluate_light_budget

PATH_TITLES = {
    "ordinary": "1. Лазер + обычная смола",
    "alternative": "2. Лазер + другой материал",
    "gelatin": "3. Желатин + витамин B2",
    "phone": "4. Камера телефона: длинная выдержка",
    "baseline": "Эталон OpenCAL без переделки проектора",
}


@dataclass(frozen=True, slots=True)
class PathCase:
    key: str
    exposure_s: float = 60.0
    plane_irradiance_mw_cm2: float | None = None
    plane_evidence: str = "unknown"  # unknown, scenario, measured
    gel_threshold_mj_cm2: float | None = None
    threshold_evidence: str = "unknown"
    voxel_inside_mj_cm2: float | None = None
    voxel_outside_mj_cm2: float | None = None
    voxel_evidence: str = "unknown"
    strength_mpa: float | None = None
    thermal_control_passed: bool = False

    def validate(self) -> None:
        if self.key not in PATH_TITLES:
            raise ValueError("Неизвестный вариант проекта.")
        if not math.isfinite(self.exposure_s) or not 0 < self.exposure_s <= 86400:
            raise ValueError("Время экспозиции должно быть от 0 до 86400 с.")
        for evidence in (self.plane_evidence, self.threshold_evidence, self.voxel_evidence):
            if evidence not in {"unknown", "scenario", "measured"}:
                raise ValueError("Источник данных: неизвестно, допущение или измерено.")
        for name in (
            "plane_irradiance_mw_cm2", "gel_threshold_mj_cm2",
            "voxel_inside_mj_cm2", "voxel_outside_mj_cm2", "strength_mpa",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name}: требуется конечное неотрицательное число.")
        for name, evidence in (
            ("plane_irradiance_mw_cm2", self.plane_evidence),
            ("gel_threshold_mj_cm2", self.threshold_evidence),
        ):
            if evidence != "unknown" and (getattr(self, name) is None or getattr(self, name) <= 0):
                raise ValueError(f"{name}: задайте положительное число или оставьте неизвестным.")
        if self.voxel_evidence != "unknown" and (
            self.voxel_inside_mj_cm2 is None or self.voxel_outside_mj_cm2 is None
        ):
            raise ValueError("Для CAL нужны обе накопленные дозы: внутри и снаружи детали.")


@dataclass(frozen=True, slots=True)
class PathResult:
    key: str
    ideal_irradiance_mw_cm2: tuple[float, float] | None
    plane_irradiance_mw_cm2: tuple[float, float] | None
    full_field_dose_mj_cm2: tuple[float, float] | None
    threshold_ratio: tuple[float, float] | None
    cal_margin_mj_cm2: float | None
    plane_evidence: str
    status: str
    missing: tuple[str, ...]
    explanation: str


def evaluate_path(case: PathCase, laser: LightBudgetInput | None = None) -> PathResult:
    """Return only quantities supported by supplied measurements or scenarios."""

    case.validate()
    if case.key == "phone":
        return PathResult(
            key="phone",
            ideal_irradiance_mw_cm2=None,
            plane_irradiance_mw_cm2=None,
            full_field_dose_mj_cm2=None,
            threshold_ratio=None,
            cal_margin_mj_cm2=None,
            plane_evidence="unknown",
            status="Камера покажет лишь проекцию света",
            missing=(
                "папка проекционных кадров и синхронизация с оборотом",
                "модель рассеяния жидкости и чувствительности камеры для сравнения с фото",
            ),
            explanation=(
                "Длинная выдержка камеры суммирует попавший в неё свет в двумерном кадре. "
                "Прозрачная вода почти не показывает луч сбоку; по одному снимку нельзя "
                "восстановить трёхмерную дозу или доказать образование детали. "
                "Программа отдельно покажет накопленное поле и условный снимок "
                "рассеянного света при боковом обзоре."
            ),
        )
    laser_route = case.key in {"ordinary", "alternative", "gelatin"}
    ideal = None
    plane = None
    plane_evidence = case.plane_evidence
    if laser_route:
        if laser is None:
            raise ValueError("Для лазерного варианта задайте параметры источника.")
        budget = evaluate_light_budget(replace(laser, exposure_s=case.exposure_s))
        ideal = budget.source_irradiance_mw_cm2
        plane = budget.plane_irradiance_mw_cm2
        plane_evidence = (
            "measured" if laser.measured_irradiance_mw_cm2 is not None else
            "scenario" if laser.transmission_pct is not None else "unknown"
        )
        if case.plane_evidence != "unknown":
            assert case.plane_irradiance_mw_cm2 is not None
            plane = (case.plane_irradiance_mw_cm2,) * 2
            plane_evidence = case.plane_evidence
    elif case.plane_evidence != "unknown":
        assert case.plane_irradiance_mw_cm2 is not None
        plane = (case.plane_irradiance_mw_cm2,) * 2

    dose = (plane[0] * case.exposure_s, plane[1] * case.exposure_s) if plane else None
    threshold = (
        case.gel_threshold_mj_cm2 if case.threshold_evidence != "unknown" else None
    )
    thermal_gate = case.key != "gelatin" or case.thermal_control_passed
    ratio = (
        (dose[0] / threshold, dose[1] / threshold)
        if dose is not None and threshold is not None and thermal_gate else None
    )
    margin = None
    if (
        threshold is not None and thermal_gate and case.voxel_evidence != "unknown"
    ):
        assert case.voxel_inside_mj_cm2 is not None
        assert case.voxel_outside_mj_cm2 is not None
        margin = min(
            case.voxel_inside_mj_cm2 - threshold,
            threshold - case.voxel_outside_mj_cm2,
        )

    missing: list[str] = []
    if plane is None:
        missing.append("свет в нужном спектральном диапазоне на колбе/материале")
    if threshold is None:
        missing.append("порог светового закрепления именно этого материала")
    if case.key == "gelatin" and not case.thermal_control_passed:
        missing.append("контроль без света при той же температуре: желатин может застыть от охлаждения")
    if case.voxel_evidence == "unknown":
        missing.append("накопленная доза внутри и вне детали для CAL")

    if plane is None:
        status = "Недостаточно данных о свете"
    elif threshold is None:
        status = "Неизвестен отклик материала"
    elif not thermal_gate:
        status = "Нужен температурный контроль"
    elif margin is None:
        status = "Есть сравнение дозы; печать не доказана"
    elif margin > 0:
        status = "Есть условный запас CAL-контраста"
    else:
        status = "Нет разделения CAL-доз по порогу"

    explanation = (
        "Вычислена доза равномерного полного поля I×t. Доза отдельной точки при вращении "
        "не равна этому числу. Сравнение с порогом показывает только световой сценарий."
    )
    if case.key == "gelatin":
        explanation += (
            " B2 поглощает синий свет; смесь обычного желатина с B2 не имеет "
            "переносимой рабочей кривой GelMA. Контроль охлаждения обязателен."
        )
    return PathResult(
        key=case.key,
        ideal_irradiance_mw_cm2=ideal,
        plane_irradiance_mw_cm2=plane,
        full_field_dose_mj_cm2=dose,
        threshold_ratio=ratio,
        cal_margin_mj_cm2=margin,
        plane_evidence=plane_evidence,
        status=status,
        missing=tuple(missing),
        explanation=explanation,
    )


@dataclass(frozen=True, slots=True)
class LaserAdvantage:
    intensity_ratio: tuple[float, float] | None
    strength_change_pct: float | None
    evidence: str
    interpretation: str


def compare_laser_to_baseline(
    laser_result: PathResult,
    baseline_result: PathResult,
    laser_case: PathCase,
    baseline_case: PathCase,
) -> LaserAdvantage:
    """Compare the same candidate material and illuminated field under two sources."""

    ratio = None
    if laser_result.plane_irradiance_mw_cm2 and baseline_result.plane_irradiance_mw_cm2:
        lower, upper = laser_result.plane_irradiance_mw_cm2
        reference = baseline_result.plane_irradiance_mw_cm2
        if reference[0] > 0:
            ratio = (lower / reference[1], upper / reference[0])
    strength = None
    if laser_case.strength_mpa is not None and baseline_case.strength_mpa:
        strength = 100 * (laser_case.strength_mpa / baseline_case.strength_mpa - 1)
    evidence = (
        "измерено" if laser_result.plane_evidence == baseline_result.plane_evidence == "measured"
        else "сценарий" if ratio is not None else "нет данных"
    )
    interpretation = (
        "Сравнение освещённости относится к одному материалу и одинаковому полю. "
        "Само по себе оно не доказывает меньшего времени CAL-печати или лучших свойств детали. "
        "Для этого нужны порог, объёмный контраст и одинаковая методика испытаний деталей."
    )
    return LaserAdvantage(ratio, strength, evidence, interpretation)
