"""Transparent, extensible first-order models for vat photopolymer resins.

This module deliberately keeps chemistry calculations independent of Qt.  The
default model combines an ideal logarithmic viscosity blend, Arrhenius
temperature correction, the Jacobs working curve, and explicitly labelled
mixture-rule estimates for shrinkage and mechanics.  These are screening tools,
not substitutes for a resin datasheet or printer calibration.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

R_GAS_J_MOL_K = 8.314462618
SCHEMA_VERSION = 1


@dataclass(slots=True)
class Component:
    """A resin ingredient. Optional properties stay unknown until supplied."""

    key: str
    name: str
    category: str
    density_g_ml: float | None = None
    viscosity_mpas: float | None = None
    viscosity_activation_kj_mol: float | None = None
    viscosity_modifier_per_wt_pct: float = 0.0
    refractive_index: float | None = None
    penetration_depth_mm: float | None = None
    critical_exposure_mj_cm2: float | None = None
    initiator_peak_nm: float | None = None
    initiator_bandwidth_nm: float | None = None
    initiator_sensitivity_per_wt_pct: float = 0.0
    attenuation_per_wt_pct_mm_inv: float = 0.0
    shrinkage_vol_pct: float | None = None
    modulus_mpa: float | None = None
    tensile_strength_mpa: float | None = None
    elongation_pct: float | None = None
    brittleness_index: float | None = None
    layer_bond_factor: float | None = None
    source: str = "estimate"
    notes: str = ""
    inhibition_per_wt_pct: float | None = None

    def validate(self) -> None:
        if not self.key.strip() or not self.name.strip():
            raise ValueError("У компонента должны быть имя и уникальный ключ.")
        if self.category not in COMPONENT_CATEGORIES:
            raise ValueError(f"Неизвестная категория компонента: {self.category}.")
        if self.source not in {"estimate", "measured"}:
            raise ValueError("Источник свойств должен быть «оценка» или «измерено».")
        positive = (
            "density_g_ml",
            "viscosity_mpas",
            "viscosity_activation_kj_mol",
            "refractive_index",
            "penetration_depth_mm",
            "critical_exposure_mj_cm2",
            "initiator_peak_nm",
            "initiator_bandwidth_nm",
            "modulus_mpa",
            "tensile_strength_mpa",
        )
        for name in positive:
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{self.name}: {name} должен быть положительным числом.")
        modifiers = ("viscosity_modifier_per_wt_pct",)
        for name in modifiers:
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{self.name}: {name} должен быть конечным числом.")
        nonnegative = (
            "initiator_sensitivity_per_wt_pct",
            "attenuation_per_wt_pct_mm_inv",
            "inhibition_per_wt_pct",
            "shrinkage_vol_pct",
            "elongation_pct",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{self.name}: {name} должен быть неотрицательным числом.")
        for name in ("brittleness_index", "layer_bond_factor"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 100):
                raise ValueError(f"{self.name}: {name} должен быть от 0 до 100.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Component:
        allowed = cls.__dataclass_fields__.keys()
        component = cls(**{key: item for key, item in value.items() if key in allowed})
        component.validate()
        return component


COMPONENT_CATEGORIES: dict[str, str] = {
    "base": "Основа смолы",
    "oligomer": "Олигомер",
    "monomer": "Мономер / разбавитель",
    "initiator": "Фотоинициатор",
    "pigment": "Пигмент / краситель",
    "filler": "Наполнитель",
    "plasticizer": "Пластификатор",
    "inhibitor": "Ингибитор",
    "additive": "Другая добавка",
}


def default_components() -> dict[str, Component]:
    """Return editable example ingredients; every starter value is an estimate."""
    items = (
        Component(
            "standard_acrylate", "Акрилатная основа средней вязкости", "base",
            1.10, 800, 34, 0, 1.49, 0.18, 70, None, None, 0, 0,
            5.5, 1450, 38, 6, 64, 62, "estimate",
            "Учебное значение. Замените данными паспорта материала и тест-купона.",
        ),
        Component(
            "flexible_urethane", "Гибкий уретан-акрилат", "base",
            1.08, 2400, 38, 0, 1.49, 0.19, 75, None, None, 0, 0,
            3.5, 180, 23, 85, 18, 78, "estimate",
            "Условный профиль гибкой смолы, не паспорт конкретного продукта.",
        ),
        Component(
            "low_viscosity_base", "Низковязкая смоляная основа", "base",
            1.07, 110, 31, 0, 1.49, 0.17, 65, None, None, 0, 0,
            6.2, 1750, 42, 7, 70, 58, "estimate",
            "Условный профиль для тонких деталей и быстрого стекания.",
        ),
        Component(
            "reactive_diluent", "Реактивный разбавитель (IBOA-подобный)", "monomer",
            0.99, 6, 28, -0.012, 1.48, None, None, None, None, 0, 0,
            2.4, 1150, 35, 9, 48, 60, "estimate",
            "Свойства сильно зависят от конкретного мономера и степени конверсии.",
        ),
        Component(
            "flexible_oligomer", "Гибкий уретан-акрилатный олигомер", "oligomer",
            1.08, 3600, 39, 0.002, 1.49, 0.19, 75, None, None, 0, 0,
            3.2, 260, 28, 70, 24, 76, "estimate",
            "Оценка для демонстрации направления изменения свойств.",
        ),
        Component(
            "tpo_l", "Фотоинициатор TPO-L (пример)", "initiator",
            1.17, 40, 28, 0, 1.52, None, None, 395, 55, 0.52, 0.06,
            0.5, None, None, None, None, None, "estimate",
            "Пиковое поглощение и эффективность зависят от партии и спектра источника.",
        ),
        Component(
            "bap_o", "Фотоинициатор BAPO (пример)", "initiator",
            1.20, 80, 30, 0, 1.55, None, None, 385, 65, 0.42, 0.05,
            0.5, None, None, None, None, None, "estimate",
            "Учебный спектральный отклик; проверьте совместимость с длиной волны.",
        ),
        Component(
            "carbon_black", "Чёрный пигмент (оценка)", "pigment",
            1.80, 5000, 25, 0.16, 1.80, None, None, None, None, 0, 0.90,
            0, None, None, None, None, None, "estimate",
            "Дисперсия и размер частиц обычно доминируют над этим упрощённым вкладом.",
        ),
        Component(
            "silica", "Микрокремнезём (оценка)", "filler",
            2.20, 10000, 20, 0.20, 1.46, None, None, None, None, 0, 0.12,
            0, None, None, None, None, None, "estimate",
            "Не моделируются размер частиц, агломерация и осаждение наполнителя.",
        ),
        Component(
            "plasticizer", "Пластификатор (пример)", "plasticizer",
            1.02, 30, 29, -0.018, 1.46, None, None, None, None, 0, 0,
            1.0, 750, 24, 28, 42, 56, "estimate",
            "Оценка; совместимость и миграцию добавки нужно проверить отдельно.",
        ),
        Component(
            "inhibitor", "Ингибитор (пример)", "inhibitor",
            1.10, 20, 26, 0, 1.50, None, None, None, None, 0, 0,
            0, None, None, None, None, None, "estimate",
            "Влияние ингибитора на Ec и хранение здесь не калибровано.",
            inhibition_per_wt_pct=0.25,
        ),
    )
    return {item.key: item for item in items}


@dataclass(slots=True)
class ProcessConditions:
    wavelength_nm: float = 405.0
    intensity_mw_cm2: float = 5.0
    exposure_s: float = 2.0
    layer_height_um: float = 50.0
    temperature_c: float = 25.0
    mix_rpm: float = 300.0
    mix_minutes: float = 5.0
    rest_minutes: float = 10.0
    wash_minutes: float = 5.0
    post_cure_minutes: float = 10.0
    reference_dimension_mm: float = 20.0
    measured: tuple[str, ...] = ()

    def validate(self) -> None:
        bounds = {
            "wavelength_nm": (200, 1000), "intensity_mw_cm2": (0.01, 5000),
            "exposure_s": (0.001, 3600), "layer_height_um": (1, 1000),
            "temperature_c": (-20, 150), "mix_rpm": (0, 10000),
            "mix_minutes": (0, 1440), "rest_minutes": (0, 10080),
            "wash_minutes": (0, 1440), "post_cure_minutes": (0, 1440),
            "reference_dimension_mm": (0.01, 100000),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name}: ожидается значение от {low:g} до {high:g}.")
        allowed_measured = set(bounds)
        if not set(self.measured).issubset(allowed_measured):
            raise ValueError("В списке измеренных параметров есть неизвестное поле.")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["measured"] = list(self.measured)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProcessConditions:
        allowed = cls.__dataclass_fields__.keys()
        values = {key: item for key, item in value.items() if key in allowed}
        values["measured"] = tuple(values.get("measured", ()))
        conditions = cls(**values)
        conditions.validate()
        return conditions


@dataclass(slots=True)
class ModelParameters:
    viscosity_interaction: float = 1.0
    filler_sensitivity: float = 1.0
    cure_depth_scale: float = 1.0
    critical_exposure_scale: float = 1.0
    shrinkage_scale: float = 1.0
    mechanics_scale: float = 1.0
    exposure_window_overcure_layers: float = 1.5
    temperature_reference_c: float = 25.0

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"Коэффициент {key} должен быть конечным числом.")
        if self.viscosity_interaction < 0 or self.filler_sensitivity < 0:
            raise ValueError("Коэффициенты вязкости не могут быть отрицательными.")
        for key in ("cure_depth_scale", "critical_exposure_scale", "shrinkage_scale", "mechanics_scale"):
            if getattr(self, key) <= 0:
                raise ValueError(f"Коэффициент {key} должен быть положительным.")
        if self.exposure_window_overcure_layers <= 1:
            raise ValueError("Верхняя граница окна должна быть больше одного слоя.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelParameters:
        allowed = cls.__dataclass_fields__.keys()
        result = cls(**{key: item for key, item in value.items() if key in allowed})
        result.validate()
        return result


@dataclass(slots=True)
class Recipe:
    name: str = "Новая рецептура"
    task: str = "detail"
    base_id: str = "standard_acrylate"
    batch_mass_g: float = 100.0
    additives_g: dict[str, float] = field(default_factory=lambda: {"reactive_diluent": 15.0, "tpo_l": 1.5})
    process: ProcessConditions = field(default_factory=ProcessConditions)
    model: ModelParameters = field(default_factory=ModelParameters)
    model_key: str = "screening"

    def validate(self, components: dict[str, Component]) -> None:
        self.process.validate()
        self.model.validate()
        if not self.name.strip():
            raise ValueError("У рецептуры должно быть имя.")
        if not math.isfinite(self.batch_mass_g) or self.batch_mass_g <= 0:
            raise ValueError("Масса партии должна быть больше нуля.")
        if self.base_id not in components or components[self.base_id].category != "base":
            raise ValueError("Выберите существующую основу смолы.")
        components[self.base_id].validate()
        total = 0.0
        for key, mass in self.additives_g.items():
            if key not in components:
                raise ValueError(f"Компонент {key} отсутствует в библиотеке.")
            components[key].validate()
            if components[key].category == "base":
                raise ValueError("Основу задайте отдельно от добавок.")
            if not math.isfinite(mass) or mass < 0:
                raise ValueError(f"Дозировка {components[key].name} должна быть неотрицательной.")
            total += mass
        if total >= self.batch_mass_g:
            raise ValueError("Сумма добавок должна быть меньше массы партии: остаток основы не положителен.")

    def component_masses(self, components: dict[str, Component]) -> dict[str, float]:
        self.validate(components)
        masses = {key: value for key, value in self.additives_g.items() if value > 0}
        masses[self.base_id] = self.batch_mass_g - sum(masses.values())
        return masses

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "task": self.task, "base_id": self.base_id,
            "batch_mass_g": self.batch_mass_g, "additives_g": dict(self.additives_g),
            "process": self.process.to_dict(), "model": self.model.to_dict(),
            "model_key": self.model_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Recipe:
        return cls(
            name=str(value.get("name", "Импортированная рецептура")),
            task=str(value.get("task", "experimental")),
            base_id=str(value.get("base_id", "standard_acrylate")),
            batch_mass_g=float(value.get("batch_mass_g", 100.0)),
            additives_g={str(k): float(v) for k, v in value.get("additives_g", {}).items()},
            process=ProcessConditions.from_dict(value.get("process", {})),
            model=ModelParameters.from_dict(value.get("model", {})),
            model_key=str(value.get("model_key", "screening")),
        )


@dataclass(slots=True)
class Prediction:
    key: str
    name: str
    value: str | None
    low: float | None = None
    high: float | None = None
    unit: str = ""
    confidence: str = "низкая"
    drivers: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    caveat: str = ""


@dataclass(slots=True)
class Sensitivity:
    name: str
    viscosity_pct: float | None
    cure_depth_pct: float | None
    shrinkage_pct: float | None
    printability_points: float | None
    perturbation: str


@dataclass(slots=True)
class SimulationResult:
    predictions: list[Prediction]
    sensitivities: list[Sensitivity]
    composition: list[dict[str, Any]]
    dose_curve: list[tuple[float, float]]
    warnings: list[str]
    assumptions: list[str]
    missing_data: list[str]

    def prediction(self, key: str) -> Prediction:
        return next(item for item in self.predictions if item.key == key)


@dataclass(frozen=True, slots=True)
class ModelPlugin:
    key: str
    name: str
    description: str
    calculate: Callable[..., SimulationResult]


class ModelRegistry:
    """Small extension point: register another calculator without changing UI."""

    def __init__(self) -> None:
        self._models: dict[str, ModelPlugin] = {}

    def register(self, model: ModelPlugin) -> None:
        if not model.key or model.key in self._models:
            raise ValueError("Ключ модели должен быть уникальным и непустым.")
        self._models[model.key] = model

    def get(self, key: str) -> ModelPlugin:
        try:
            return self._models[key]
        except KeyError as exc:
            raise ValueError(f"Модель {key!r} не зарегистрирована.") from exc

    def all(self) -> tuple[ModelPlugin, ...]:
        return tuple(self._models.values())


def _quantile_band(value: float, confidence: str) -> tuple[float, float]:
    # Screening ranges communicate model spread; they are not statistical CIs.
    relative = {"низкая": 0.35, "средняя": 0.20, "выше средней": 0.12}.get(confidence, 0.35)
    return max(0.0, value * (1.0 - relative)), value * (1.0 + relative)


def _confidence(components: list[Component], conditions: ProcessConditions, *, heuristic: bool = True) -> str:
    if all(item.source == "measured" for item in components) and not heuristic and len(conditions.measured) >= 3:
        return "выше средней"
    if all(item.source == "measured" for item in components) and len(conditions.measured) >= 3:
        return "средняя"
    if any(item.source == "measured" for item in components) or conditions.measured:
        return "низкая"
    return "низкая"


def _source_notes(components: list[Component], names: tuple[str, ...]) -> list[str]:
    return [
        f"{item.name}: {_source_label(item)}; {item.notes}".strip()
        for item in components
        if any(getattr(item, key, None) is not None for key in names)
    ]


def _source_label(component: Component) -> str:
    return "измеренные данные" if component.source == "measured" else "приближённые данные библиотеки"


def composition_table(recipe: Recipe, components: dict[str, Component]) -> list[dict[str, Any]]:
    masses = recipe.component_masses(components)
    volumes: dict[str, float] = {}
    for key, mass in masses.items():
        density = components[key].density_g_ml
        if density is not None:
            volumes[key] = mass / density
    total_volume = sum(volumes.values()) if len(volumes) == len(masses) else None
    total_mass = sum(masses.values())
    result = []
    for key, mass in masses.items():
        component = components[key]
        result.append({
            "key": key,
            "name": component.name,
            "category": COMPONENT_CATEGORIES[component.category],
            "mass_g": mass,
            "weight_pct": 100 * mass / total_mass,
            "volume_pct": 100 * volumes[key] / total_volume if total_volume else None,
            "density_g_ml": component.density_g_ml,
        })
    return result


def _calculate_core(recipe: Recipe, components: dict[str, Component]) -> tuple[list[Prediction], list[str], list[str], list[tuple[float, float]], list[str]]:
    recipe.validate(components)
    masses = recipe.component_masses(components)
    fractions = {key: value / recipe.batch_mass_g for key, value in masses.items()}
    wt_pcts = {key: value * 100.0 for key, value in fractions.items()}
    used = [components[key] for key in masses]
    base = components[recipe.base_id]
    p = recipe.process
    model = recipe.model
    predictions: list[Prediction] = []
    warnings: list[str] = []
    missing_all: list[str] = []
    assumptions = [
        "Расчёт — предварительная оценка для планирования, не гарантия печати и не точное предсказание реакции.",
        "Библиотечные значения по умолчанию помечены как оценки; реальные партии могут отличаться.",
        "Идеальное смешение, равномерная температура и однородный свет — упрощения модели.",
        "Промывка и постотверждение показаны как заданные условия, но их химический вклад не калиброван.",
    ]
    if any(components[key].attenuation_per_wt_pct_mm_inv is None for key in masses):
        assumptions.append("Для компонентов без данных поглощения дополнительное оптическое поглощение принято равным нулю.")
    estimated = [item for item in used if item.source == "estimate"]
    if estimated:
        warnings.append(f"Оценочные свойства: {', '.join(item.name for item in estimated)}.")
    unknown_interactions = [
        components[key].name
        for key in masses
        if components[key].category in {"pigment", "filler"}
        and wt_pcts[key] >= 0.1
        and components[key].viscosity_modifier_per_wt_pct is None
    ]
    if unknown_interactions:
        warnings.append(
            "Для вязкости неизвестен вклад наполнителя; используется идеальное смешение: "
            + ", ".join(unknown_interactions)
            + "."
        )
    if not p.measured:
        warnings.append("Все параметры процесса пока считаются значениями по умолчанию.")
    confidence = _confidence(used, p)

    # Viscosity: log-mixture baseline + user-editable non-ideal modifiers and an Arrhenius correction.
    missing = [components[key].name + ": вязкость" for key in masses if components[key].viscosity_mpas is None]
    if abs(p.temperature_c - model.temperature_reference_c) > 0.05:
        missing.extend(
            components[key].name + ": температурная зависимость вязкости"
            for key in masses
            if components[key].viscosity_activation_kj_mol is None
        )
    eta = None
    if not missing:
        log_eta = sum(fractions[k] * math.log(float(components[k].viscosity_mpas or 1.0)) for k in masses)
        nonideal = sum(wt_pcts[k] * float(components[k].viscosity_modifier_per_wt_pct or 0.0) for k in masses)
        activation = sum(fractions[k] * float(components[k].viscosity_activation_kj_mol or 0.0) for k in masses)
        ref_k = model.temperature_reference_c + 273.15
        temp_k = p.temperature_c + 273.15
        temp_factor = math.exp((activation * 1000 / R_GAS_J_MOL_K) * (1 / temp_k - 1 / ref_k))
        eta = math.exp(log_eta * model.viscosity_interaction + nonideal * model.filler_sensitivity) * temp_factor
        low, high = _quantile_band(eta, confidence)
        most_viscous = sorted(
            masses, key=lambda key: wt_pcts[key] * math.log(max(components[key].viscosity_mpas or 1, 1)), reverse=True,
        )[:2]
        predictions.append(Prediction(
            "viscosity", "Вязкость при температуре смеси", f"≈ {eta:.0f} мПа·с",
            low, high, "мПа·с", confidence,
            [components[key].name for key in most_viscous if wt_pcts[key] > 2],
            _source_notes(used, ("viscosity_mpas", "viscosity_activation_kj_mol")),
            caveat="Логарифмическая идеальная смесь + температурная поправка Аррениуса; сдвиговое разжижение, тиксотропия и дисперсность не моделируются.",
        ))
        if eta > 1500:
            warnings.append("Высокая расчётная вязкость: проверьте перемешивание, дегазацию и стекание смолы.")
        temperature_delta_pct = 100.0 * (temp_factor - 1.0)
        predictions.append(Prediction(
            "temperature_effect", "Влияние температуры на вязкость",
            f"{temperature_delta_pct:+.1f}% к значению при {model.temperature_reference_c:g} °C",
            temperature_delta_pct - abs(temperature_delta_pct) * 0.3 - 3,
            temperature_delta_pct + abs(temperature_delta_pct) * 0.3 + 3,
            "%", "низкая", [f"{p.temperature_c:g} °C", f"энергия активации ≈ {activation:.1f} кДж/моль"],
            _source_notes(used, ("viscosity_activation_kj_mol",)),
            caveat="Поправка Аррениуса; у наполненных и неньютоновских смол температурная зависимость может быть иной.",
        ))
    else:
        missing_all.extend(missing)
        predictions.append(Prediction("viscosity", "Вязкость при температуре смеси", None, confidence=confidence, missing=missing, caveat="Добавьте вязкость каждого компонента в библиотеке."))
        predictions.append(Prediction("temperature_effect", "Влияние температуры на вязкость", None, confidence=confidence, missing=missing, caveat="Нужна базовая вязкость компонентов и энергия активации.") )

    # Jacobs working curve with wavelength-dependent initiator sensitivity and added attenuation.
    cure_missing = []
    if base.penetration_depth_mm is None:
        cure_missing.append(base.name + ": глубина проникновения Dp")
    if base.critical_exposure_mj_cm2 is None:
        cure_missing.append(base.name + ": критическая экспозиция Ec")
    initiators = [components[key] for key in masses if components[key].category == "initiator" and wt_pcts[key] > 0]
    inhibitors = [components[key] for key in masses if components[key].category == "inhibitor" and wt_pcts[key] > 0]
    if any(item.initiator_peak_nm is None or item.initiator_bandwidth_nm is None for item in initiators):
        cure_missing.extend(item.name + ": спектр фотоинициатора" for item in initiators if item.initiator_peak_nm is None or item.initiator_bandwidth_nm is None)
    if any(item.initiator_sensitivity_per_wt_pct is None for item in initiators):
        cure_missing.extend(item.name + ": чувствительность фотоинициатора" for item in initiators if item.initiator_sensitivity_per_wt_pct is None)
    if any(item.inhibition_per_wt_pct is None for item in inhibitors):
        cure_missing.extend(item.name + ": влияние ингибитора" for item in inhibitors if item.inhibition_per_wt_pct is None)
    optical_additives = [components[key] for key in masses if components[key].category in {"pigment", "filler"} and wt_pcts[key] >= 0.1]
    if any(item.attenuation_per_wt_pct_mm_inv is None for item in optical_additives):
        cure_missing.extend(item.name + ": поглощение света" for item in optical_additives if item.attenuation_per_wt_pct_mm_inv is None)
    cure_depth = None
    dp = ec = None
    dose = p.intensity_mw_cm2 * p.exposure_s
    wavelength_factor = 0.0
    if not cure_missing:
        initiation_gain = 0.0
        for item in initiators:
            bandwidth = float(item.initiator_bandwidth_nm or 1.0)
            match = math.exp(-0.5 * ((p.wavelength_nm - float(item.initiator_peak_nm or p.wavelength_nm)) / bandwidth) ** 2)
            wavelength_factor = max(wavelength_factor, match)
            initiation_gain += wt_pcts[item.key] * float(item.initiator_sensitivity_per_wt_pct or 0.0) * match
        inhibition = sum(wt_pcts[item.key] * float(item.inhibition_per_wt_pct or 0.0) for item in inhibitors)
        attenuation = sum(wt_pcts[k] * float(components[k].attenuation_per_wt_pct_mm_inv or 0.0) for k in masses)
        dp = model.cure_depth_scale / (1 / float(base.penetration_depth_mm or 1.0) + attenuation)
        ec = model.critical_exposure_scale * float(base.critical_exposure_mj_cm2 or 1.0) * (1 + inhibition) / max(0.15, 1 + initiation_gain)
        cure_depth = max(0.0, dp * math.log(max(dose, 1e-12) / ec))
        min_dose = ec * math.exp((p.layer_height_um / 1000) / dp)
        max_dose = ec * math.exp((p.layer_height_um / 1000 * model.exposure_window_overcure_layers) / dp)
        min_time = min_dose / p.intensity_mw_cm2
        max_time = max_dose / p.intensity_mw_cm2
        current_time = p.exposure_s
        if current_time < min_time:
            warnings.append("Текущая экспозиция ниже расчётного минимума для одного слоя.")
        elif current_time > max_time:
            warnings.append("Текущая экспозиция выше предварительной верхней границы окна.")
        if wavelength_factor < 0.2 and initiators:
            warnings.append("Длина волны далеко от заданного пика поглощения инициатора.")
        depth_label = f"≈ {cure_depth:.3f} мм" if cure_depth > 0 else "ниже порога · E < Ec"
        depth_low, depth_high = _quantile_band(cure_depth, confidence)
        if cure_depth <= 0:
            depth_high = dp * 0.35
        predictions.append(Prediction(
            "cure_depth", "Глубина отверждения по Jacobs", depth_label,
            depth_low, depth_high, "мм", confidence,
            [f"доза {dose:.2f} мДж/см²", f"Dp {dp:.3f} мм", f"Ec {ec:.1f} мДж/см²"],
            _source_notes(used, ("penetration_depth_mm", "critical_exposure_mj_cm2", "initiator_peak_nm")),
            caveat="Кривая Jacobs (Cd = Dp·ln(E/Ec)); дозу считаем как измеренная интенсивность × экспозиция на поверхности смолы.",
        ))
        cure_rate = cure_depth / p.exposure_s
        rate_label = f"≈ {cure_rate:.4f} мм/с" if cure_depth > 0 else "ниже порога · скорость не оценивается"
        rate_low, rate_high = _quantile_band(cure_rate, confidence)
        if cure_depth <= 0:
            rate_high = dp * 0.35 / p.exposure_s
        predictions.append(Prediction(
            "cure_rate", "Скорость роста глубины за экспозицию",
            rate_label, rate_low, rate_high, "мм/с", confidence,
            [f"глубина {cure_depth:.3f} мм", f"экспозиция {p.exposure_s:g} с"],
            _source_notes(used, ("penetration_depth_mm", "critical_exposure_mj_cm2")),
            caveat="Глубина отверждения / время одной экспозиции — кинематический прокси по Jacobs, не скорость химической реакции.",
        ))
        predictions.append(Prediction(
            "exposure_window", "Окно экспозиции для текущего слоя",
            f"{min_time:.2f}–{max_time:.2f} с при {p.intensity_mw_cm2:g} мВт/см²",
            min_time, max_time, "с", confidence, ["Dp, Ec, длина волны", f"слой {p.layer_height_um:g} мкм"],
            _source_notes(used, ("penetration_depth_mm", "critical_exposure_mj_cm2", "initiator_peak_nm")),
            caveat=f"Нижняя граница соответствует глубине 1 слоя, верхняя — {model.exposure_window_overcure_layers:g} слоя; это эвристика, не валидированная граница качества.",
        ))
        layer_ratio = cure_depth / max(p.layer_height_um / 1000, 1e-6)
        bond = base.layer_bond_factor
        if bond is None:
            predictions.append(Prediction(
                "layer_adhesion", "Риск недостаточной связи слоёв", None,
                confidence=confidence, missing=[base.name + ": данные сцепления слоёв"],
                drivers=[f"глубина / слой = {layer_ratio:.2f}"],
                caveat="Глубина засветки сама по себе не предсказывает адгезию между слоями.",
            ))
        else:
            adhesion_score = max(0.0, min(100.0, bond * min(1.0, layer_ratio / 1.25)))
            label = "пониженный" if adhesion_score < 55 else "умеренный" if adhesion_score < 80 else "приемлемый в этой эвристике"
            predictions.append(Prediction(
                "layer_adhesion", "Индикатор связи слоёв", f"{label} · {adhesion_score:.0f}/100",
                max(0.0, adhesion_score - 25), min(100.0, adhesion_score + 15), "балл", confidence,
                [f"глубина / слой = {layer_ratio:.2f}", f"базовый фактор {bond:g}"],
                _source_notes([base], ("layer_bond_factor", "penetration_depth_mm")),
                caveat="Сравнительный индикатор достаточности засветки; реальная адгезия требует теста на расслоение.",
            ))
    else:
        missing_all.extend(cure_missing)
        predictions.append(Prediction("cure_depth", "Глубина отверждения по Jacobs", None, confidence=confidence, missing=cure_missing, caveat="Измерьте Dp/Ec рабочей кривой и укажите спектр инициатора."))
        predictions.append(Prediction("cure_rate", "Скорость роста глубины за экспозицию", None, confidence=confidence, missing=cure_missing, caveat="Нужны Dp и Ec."))
        predictions.append(Prediction("exposure_window", "Окно экспозиции для текущего слоя", None, confidence=confidence, missing=cure_missing, caveat="Без Dp и Ec нельзя оценить окно экспозиции."))
        predictions.append(Prediction("layer_adhesion", "Риск недостаточной связи слоёв", None, confidence=confidence, missing=cure_missing, caveat="Нужны данные Dp/Ec и проверка сцепления тест-купоном."))

    # Volumetric shrinkage and dimensional-change estimate.
    shrink_missing = [components[key].name + ": усадка" for key in masses if components[key].shrinkage_vol_pct is None]
    if not shrink_missing:
        shrink = sum(fractions[k] * float(components[k].shrinkage_vol_pct or 0.0) for k in masses) * model.shrinkage_scale
        linear_pct = shrink / 3.0
        delta = p.reference_dimension_mm * linear_pct / 100
        predictions.append(Prediction(
            "shrinkage", "Объёмная усадка смеси", f"≈ {shrink:.2f}% объёма",
            *_quantile_band(shrink, confidence), "%", confidence,
            sorted(masses, key=lambda k: fractions[k] * float(components[k].shrinkage_vol_pct or 0), reverse=True)[:2],
            _source_notes(used, ("shrinkage_vol_pct",)),
            caveat="Массовое усреднение табличных значений; не учитываются конверсия, геометрия, анизотропия и постотверждение.",
        ))
        predictions.append(Prediction(
            "dimension_change", f"Изменение размера детали {p.reference_dimension_mm:g} мм",
            f"≈ −{delta:.2f} мм по линейному размеру", max(0, delta * 0.55), delta * 1.45, "мм", confidence,
            [f"усадка {shrink:.2f}% объёма", "линейная усадка ≈ объёмная / 3"],
            _source_notes(used, ("shrinkage_vol_pct",)),
            caveat="Изотропное приближение для свободной усадки; не является компенсацией размера модели.",
        ))
    else:
        missing_all.extend(shrink_missing)
        predictions.append(Prediction("shrinkage", "Объёмная усадка смеси", None, confidence=confidence, missing=shrink_missing, caveat="Нужна усадка каждого компонента или измерение смеси.") )
        predictions.append(Prediction("dimension_change", "Изменение размера детали", None, confidence=confidence, missing=shrink_missing, caveat="Без данных усадки размерное изменение не рассчитывается."))

    # Mechanical screening estimates only when every major component provides each property.
    major = [components[key] for key in masses if fractions[key] >= 0.03 or components[key].category in {"base", "oligomer", "monomer", "plasticizer"}]
    metric_defs = (
        ("modulus_mpa", "stiffness", "Модуль упругости (смесительная оценка)", "МПа", "geometric"),
        ("tensile_strength_mpa", "strength", "Прочность на растяжение (оценка)", "МПа", "arithmetic"),
        ("elongation_pct", "flexibility", "Относительное удлинение (индикатор гибкости)", "%", "arithmetic"),
        ("brittleness_index", "brittleness", "Индекс хрупкости смеси", "/100", "arithmetic"),
    )
    mechanical_values: dict[str, float] = {}
    for prop, key, title, unit, mix in metric_defs:
        absent = [item.name + ": " + _property_ru(prop) for item in major if getattr(item, prop) is None]
        if absent:
            missing_all.extend(absent)
            predictions.append(Prediction(key, title, None, confidence=confidence, missing=absent, caveat="Смесительная оценка доступна только при наличии свойства у основы и заметных долей реактивных компонентов."))
            continue
        values = [(fractions[item.key], float(getattr(item, prop))) for item in major]
        if mix == "geometric":
            result = math.exp(sum(weight * math.log(max(value, 1e-6)) for weight, value in values) / sum(weight for weight, _ in values))
        else:
            result = sum(weight * value for weight, value in values) / sum(weight for weight, _ in values)
        if key == "brittleness":
            result = min(100.0, max(0.0, result))
        else:
            result *= model.mechanics_scale
        mechanical_values[key] = result
        lo, hi = _quantile_band(result, "низкая")
        top_components = sorted(major, key=lambda item: fractions[item.key], reverse=True)[:2]
        predictions.append(Prediction(
            key, title, f"≈ {result:.1f} {unit}", lo, hi, unit, "низкая",
            [item.name for item in top_components],
            _source_notes(major, (prop,)),
            caveat="Грубое правило смеси, уровень уверенности низкий; механические свойства следует измерить на напечатанных образцах.",
        ))

    # A coarse processability index is withheld if key physical inputs are missing.
    if eta is not None and cure_depth is not None:
        viscosity_score = max(0.0, 100 - max(0.0, math.log10(max(eta, 1) / 150) * 24))
        cure_ratio = cure_depth / max(p.layer_height_um / 1000, 0.001)
        cure_score = min(100.0, cure_ratio * 100) if cure_ratio < 1.0 else max(0.0, 100 - (cure_ratio - 1.0) * 25)
        shrink_value = sum(fractions[k] * float(components[k].shrinkage_vol_pct or 0.0) for k in masses) if not shrink_missing else None
        shrink_score = 70.0 if shrink_value is None else max(0.0, 100 - float(shrink_value) * 8)
        score = 0.40 * viscosity_score + 0.35 * cure_score + 0.25 * shrink_score
        band = _quantile_band(score, "низкая")
        band = (band[0], min(100.0, band[1]))
        drivers = [f"вязкость {eta:.0f} мПа·с", f"глубина / слой {cure_ratio:.2f}"]
        if shrink_value is not None:
            drivers.append(f"усадка ≈ {shrink_value:.1f}% и выше")
        predictions.append(Prediction(
            "printability", "Технологичность (эвристический индекс)",
            f"{score:.0f}/100 · {('сложная' if score < 45 else 'требует настройки' if score < 72 else 'перспективная для теста')}",
            band[0], band[1], "балл", "низкая", drivers,
            ["вязкость, рабочая кривая, слой, усадка"],
            caveat="Индекс помогает сравнивать рецептуры внутри этой модели; не вероятность успеха и не допуск процесса.",
        ))
        risks: list[str] = []
        if cure_ratio < 1:
            risks.append("слой может не набрать расчётную глубину отверждения")
        elif p.exposure_s > max_time:
            risks.append("возможна избыточная засветка и потеря детализации")
        if eta > 1500:
            risks.append("вязкость затруднит перемешивание и удаление пузырьков")
        if shrink_value is not None and shrink_value > 5:
            risks.append("усадка может вызвать коробление или размерный уход")
        if wavelength_factor < 0.2 and initiators:
            risks.append("спектр источника слабо совпадает с фотоинициатором")
        if p.rest_minutes == 0 and eta > 500:
            risks.append("нет выдержки для удаления пузырьков в вязкой смеси")
        if not risks:
            risks.append("явные флаги этой упрощённой модели не обнаружены; проверьте тест-купон")
        predictions.append(Prediction(
            "failure_risks", "Вероятные причины проблем печати", "; ".join(risks),
            confidence="низкая", drivers=drivers,
            sources=["текущая доза, вязкость, слой и оценка усадки"],
            caveat="Это список направлений для проверки, а не диагноз отказа принтера.",
        ))
    else:
        missing = []
        if eta is None:
            missing.append("вязкость компонентов")
        if cure_depth is None:
            missing.append("Dp/Ec и экспозиция")
        predictions.append(Prediction("printability", "Технологичность (эвристический индекс)", None, confidence="низкая", missing=missing, caveat="Индекс строится только при расчёте вязкости и глубины отверждения."))
        predictions.append(Prediction("failure_risks", "Вероятные причины проблем печати", None, confidence="низкая", missing=missing, caveat="Укажите исходные свойства и параметры процесса."))

    # User-readable qualitative checks that avoid pretending to predict dispersion.
    if eta is not None:
        stirring = "перемешивание обычно несложное" if eta < 300 else "может потребоваться более долгая выдержка и дегазация" if eta < 1500 else "высокая вязкость усложнит перемешивание и удаление пузырьков"
        predictions.append(Prediction("mixing", "Удобство перемешивания", stirring, confidence="низкая", drivers=[f"вязкость ≈ {eta:.0f} мПа·с", f"задано {p.mix_rpm:g} об/мин × {p.mix_minutes:g} мин"], caveat="Не моделируются тип мешалки, сдвиговое разжижение и размер партии."))
    else:
        predictions.append(Prediction("mixing", "Удобство перемешивания", None, confidence="низкая", missing=["вязкость компонентов"]))

    dose_curve: list[tuple[float, float]] = []
    if dp is not None and ec is not None:
        for multiplier in (0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5, 8, 12):
            test_dose = ec * multiplier
            depth = max(0, dp * math.log(multiplier))
            dose_curve.append((test_dose, depth))

    if len(used) > 1 and shrink_missing:
        warnings.append("Усадка не показана: заполните свойство для компонентов без данных.")
    if p.post_cure_minutes > 0:
        warnings.append("Постотверждение задано, но его влияние на конверсию, модуль и усадку не рассчитано.")
    if p.wash_minutes > 0:
        assumptions.append(f"Промывка задана: {p.wash_minutes:g} мин; растворитель и влияние на свойства не моделируются.")
    return predictions, warnings, list(dict.fromkeys(missing_all)), dose_curve, assumptions


def _property_ru(name: str) -> str:
    return {
        "modulus_mpa": "модуль упругости", "tensile_strength_mpa": "прочность",
        "elongation_pct": "удлинение", "brittleness_index": "индекс хрупкости",
    }.get(name, name)


def _build_sensitivities(recipe: Recipe, components: dict[str, Component]) -> list[Sensitivity]:
    base_predictions, *_ = _calculate_core(recipe, components)
    baseline = {item.key: item for item in base_predictions}
    entries: list[Sensitivity] = []

    def metric_change(label: str, candidate: Recipe, changed_name: str, perturbation: str) -> None:
        try:
            preds, *_ = _calculate_core(candidate, components)
        except ValueError:
            return
        current = {item.key: item for item in preds}
        effects: dict[str, float | None] = {}
        numeric = {"viscosity": "viscosity", "cure_depth": "cure_depth", "shrinkage": "shrinkage", "printability": "printability"}
        for key, target in numeric.items():
            first, second = baseline.get(key), current.get(key)
            a, b = (first.value, second.value) if first and second else (None, None)
            number_a = _first_number(a)
            number_b = _first_number(b)
            if number_a is None or number_b is None:
                effects[target] = None
            elif target == "printability":
                effects[target] = number_b - number_a
            else:
                effects[target] = 100 * (number_b - number_a) / abs(number_a) if number_a != 0 else None
        entries.append(Sensitivity(label, effects["viscosity"], effects["cure_depth"], effects["shrinkage"], effects["printability"], perturbation))

    for key, amount in recipe.additives_g.items():
        if amount <= 0:
            continue
        candidate = Recipe.from_dict(recipe.to_dict())
        fixed_other = sum(value for other, value in candidate.additives_g.items() if other != key)
        candidate.additives_g[key] = min(amount * 1.10, candidate.batch_mass_g - fixed_other - 0.001)
        if candidate.additives_g[key] <= amount:
            continue
        metric_change("+10% · " + components[key].name, candidate, key, "масса добавки +10%, масса основы уменьшается")
    base_amount = recipe.component_masses(components)[recipe.base_id]
    candidate = Recipe.from_dict(recipe.to_dict())
    candidate.batch_mass_g += base_amount * 0.10
    metric_change("Основа +10% (добавки без изменений)", candidate, recipe.base_id, "масса основы +10%")
    for key in ("intensity_mw_cm2", "exposure_s", "layer_height_um", "temperature_c", "wavelength_nm"):
        candidate = Recipe.from_dict(recipe.to_dict())
        val = getattr(candidate.process, key)
        setattr(candidate.process, key, val * 1.10 if key != "temperature_c" else val + 2.0)
        metric_change({
            "intensity_mw_cm2": "Интенсивность +10%", "exposure_s": "Экспозиция +10%",
            "layer_height_um": "Толщина слоя +10%", "temperature_c": "Температура +2 °C",
            "wavelength_nm": "Длина волны +10%",
        }[key], candidate, key, "+10%" if key != "temperature_c" else "+2 °C")
    for label, limitation in (
        ("Скорость перемешивания", "модель не описывает влияние оборотов на качество дисперсии"),
        ("Время перемешивания", "модель не описывает влияние времени на дисперсию"),
        ("Выдержка / дегазация", "нет модели пузырьков и осаждения наполнителя"),
        ("Промывка", "нет калибровки растворителя и времени промывки"),
        ("Постотверждение", "нет модели конверсии и механики после постотверждения"),
    ):
        entries.append(Sensitivity(label, None, None, None, None, limitation))
    return entries


def _first_number(value: str | None) -> float | None:
    if value is None:
        return None
    import re
    match = re.search(r"-?\d+(?:[.,]\d+)?", value)
    return float(match.group().replace(",", ".")) if match else None


def _calculate_screening(recipe: Recipe, components: dict[str, Component], *, sensitivity: bool = True) -> SimulationResult:
    """Calculate transparent screening predictions and local sensitivity."""
    predictions, warnings, missing, dose_curve, assumptions = _calculate_core(recipe, components)
    return SimulationResult(
        predictions=predictions,
        sensitivities=_build_sensitivities(recipe, components) if sensitivity else [],
        composition=composition_table(recipe, components),
        dose_curve=dose_curve,
        warnings=warnings,
        assumptions=assumptions + [
            "Диапазоны — инженерный разброс приближённой модели, а не статистический доверительный интервал.",
            "Уровень доверия снижен, пока библиотечные оценки и параметры принтера не заменены измерениями.",
            "Сравнение рецептур корректно только при одинаковых параметрах процесса и одной версии модели.",
        ],
        missing_data=missing,
    )


def starter_recipes() -> dict[str, Recipe]:
    return {
        "Гибкая деталь · пример": Recipe(
            "Гибкая деталь · пример", "flexible", "flexible_urethane", 100,
            {"plasticizer": 8.0, "tpo_l": 1.5},
        ),
        "Жёсткая деталь · пример": Recipe(
            "Жёсткая деталь · пример", "rigid", "standard_acrylate", 100,
            {"reactive_diluent": 8.0, "tpo_l": 1.5},
        ),
        "Детализация · пример": Recipe(
            "Детализация · пример", "detail", "low_viscosity_base", 100,
            {"tpo_l": 1.8, "bap_o": 0.7},
        ),
        "Быстрое отверждение · пример": Recipe(
            "Быстрое отверждение · пример", "fast", "standard_acrylate", 100,
            {"tpo_l": 2.4},
            ProcessConditions(exposure_s=1.0, intensity_mw_cm2=8.0, layer_height_um=50),
        ),
        "Экспериментальная смесь": Recipe(
            "Экспериментальная смесь", "experimental", "standard_acrylate", 100,
            {"reactive_diluent": 12.0, "flexible_oligomer": 10.0, "tpo_l": 1.5},
        ),
    }


def save_library(path: str | Path, components: dict[str, Component], recipes: dict[str, Recipe]) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "components": [item.to_dict() for item in components.values()],
        "recipes": [item.to_dict() for item in recipes.values()],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_library(path: str | Path) -> tuple[dict[str, Component], dict[str, Recipe]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Неизвестная версия библиотеки рецептур.")
    components = {item.key: item for item in (Component.from_dict(value) for value in raw.get("components", []))}
    recipes = {item.name: item for item in (Recipe.from_dict(value) for value in raw.get("recipes", []))}
    if not components:
        raise ValueError("Библиотека компонентов пуста.")
    return components, recipes


def export_recipe(path: str | Path, recipe: Recipe, components: dict[str, Component]) -> None:
    recipe.validate(components)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "recipe": recipe.to_dict(),
        "components": [components[key].to_dict() for key in recipe.component_masses(components)],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


DEFAULT_MODEL_REGISTRY = ModelRegistry()
DEFAULT_MODEL_REGISTRY.register(ModelPlugin(
    "screening", "Предварительная модель смеси", "Смесительные правила и рабочая кривая Jacobs",
    _calculate_screening,
))


def calculate_recipe(recipe: Recipe, components: dict[str, Component], *, sensitivity: bool = True) -> SimulationResult:
    """Dispatch to the selected registered model implementation."""
    recipe.validate(components)
    plugin = DEFAULT_MODEL_REGISTRY.get(recipe.model_key)
    return plugin.calculate(recipe, components, sensitivity=sensitivity)
