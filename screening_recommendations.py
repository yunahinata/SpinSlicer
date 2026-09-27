"""No-input screening numbers for the four proposed development paths."""

from __future__ import annotations

from dataclasses import dataclass, replace

from light_source_simulation import LightBudgetInput, evaluate_light_budget


@dataclass(frozen=True, slots=True)
class LightScenario:
    transmission_pct: float
    irradiance_mw_cm2: tuple[float, float]
    full_field_dose_mj_cm2: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ScreeningPlan:
    wavelength_nm: float
    power_w: tuple[float, float]
    field_cm2: float
    exposure_s: float
    ideal_irradiance_mw_cm2: tuple[float, float]
    ideal_full_field_dose_mj_cm2: tuple[float, float]
    scenarios: tuple[LightScenario, ...]


def build_screening_plan(source: LightBudgetInput | None = None) -> ScreeningPlan:
    """Calculate an explicitly hypothetical light budget without resin inputs.

    The current source profile supplies wavelength, nameplate power, field, and
    exposure. Downstream transmission/measurements are ignored here and swept
    over broad what-if values. These are full-field calculations, not CAL voxel
    dose or cure predictions.
    """

    source = source or LightBudgetInput()
    assumptions = replace(
        source,
        transmission_pct=None,
        measured_irradiance_mw_cm2=None,
        measured_gel_dose_mj_cm2=None,
    )
    ideal = evaluate_light_budget(assumptions)
    assert ideal.plane_irradiance_mw_cm2 is None
    ideal_dose = tuple(value * assumptions.exposure_s for value in ideal.source_irradiance_mw_cm2)
    scenarios: list[LightScenario] = []
    for transmission in (1.0, 5.0, 10.0, 25.0):
        result = evaluate_light_budget(replace(assumptions, transmission_pct=transmission))
        assert result.plane_irradiance_mw_cm2 is not None
        assert result.plane_dose_mj_cm2 is not None
        scenarios.append(
            LightScenario(
                transmission_pct=transmission,
                irradiance_mw_cm2=result.plane_irradiance_mw_cm2,
                full_field_dose_mj_cm2=result.plane_dose_mj_cm2,
            )
        )
    return ScreeningPlan(
        wavelength_nm=assumptions.wavelength_nm,
        power_w=(assumptions.power_min_w, assumptions.power_max_w),
        field_cm2=ideal.area_cm2,
        exposure_s=assumptions.exposure_s,
        ideal_irradiance_mw_cm2=ideal.source_irradiance_mw_cm2,
        ideal_full_field_dose_mj_cm2=ideal_dose,  # type: ignore[arg-type]
        scenarios=tuple(scenarios),
    )


ROUTE_ASSESSMENTS: tuple[tuple[str, str, str, str], ...] = (
    (
        "ordinary",
        "Возможно в принципе, но для вашей смолы неизвестно.",
        "Отдельные фотополимеры реагируют на синий свет; маркировка 405 нм сама по себе не предсказывает отклик при 450 нм. Без отклика конкретной смолы печати не будет.",
        "Пока: симулировать проекции и геометрию. Для печати потребуются фоточувствительная смола и измерение её отклика именно при свете этого источника.",
    ),
    (
        "alternative",
        "Практическая выгода лазера не подтверждена.",
        "Световой бюджет можно оценить, но лазерная яркость не означает лучшую смолу или более прочную деталь. Прямой лазерный тракт добавляет потери, неравномерность и спекл.",
        "Если сравнивать, то один материал и одинаковое поле под лазером и штатным проектором OpenCAL. Без измерений это только цифровая проверка, не доказательство выигрыша.",
    ),
    (
        "gelatin",
        "Да, желатин с рибофлавином можно фотосшивать в плёнку; печать CAL при 450 нм не доказана.",
        "В опубликованной работе желатиновую плёнку облучали 2 часа при 370–405 нм; растворимость снизилась примерно со 100% до 30%. Это не тест лазера 450 нм и не объёмная печать. Рибофлавин поглощает синий свет около 450 нм, поэтому гипотеза правдоподобна, но порог неизвестен.",
        "Перспективно как отдельная проверка материала, вероятно медленно и с водостойкостью как отдельной проблемой. Нельзя переносить свойства плёнки на деталь в колбе.",
    ),
    (
        "phone",
        "Печатать этим способом нельзя; можно только наблюдать свет камерой.",
        "Длинная выдержка записывает двумерную сумму света. Она не создаёт отверждение и не подтверждает 3D-дозу.",
        "Использовать как визуальную демонстрацию проекций. Прозрачная жидкость может почти не быть видна боковой камере.",
    ),
)
