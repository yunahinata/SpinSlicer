"""Evidence-based material screening for the 450 nm SpinSlicer laser path.

This is a screening model, not a photopolymer cure simulator.  It ranks known
material evidence and enumerates optical dose scenarios; it deliberately does
not infer gelation when a candidate has no measured threshold at the target
wavelength.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from light_source_simulation import LightBudgetInput, evaluate_light_budget

LASER_BAND_NM = (440.0, 460.0)
EXPOSURE_SWEEP_S = (1.0, 5.0, 10.0, 30.0, 60.0, 120.0)
TRANSMISSION_SWEEP_PCT = (1.0, 5.0, 10.0, 25.0, 100.0)


@dataclass(frozen=True, slots=True)
class ResinCandidate:
    key: str
    name: str
    target_wavelength_nm: float | None
    cal_tested: bool
    purchasable: bool
    evidence: str
    known: tuple[str, ...]
    missing: tuple[str, ...]
    recommendation: str
    source_title: str
    source_url: str


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    candidate: ResinCandidate
    status: str
    status_label: str
    wavelength_gap_nm: float | None
    confidence: str


@dataclass(frozen=True, slots=True)
class DoseScenario:
    transmission_pct: float
    exposure_s: float
    dose_mj_cm2: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ResinSearchResult:
    laser_band_nm: tuple[float, float]
    laser_peak_nm: float
    assessments: tuple[CandidateAssessment, ...]
    dose_scenarios: tuple[DoseScenario, ...]
    scenario_count: int
    missing_for_prediction: tuple[str, ...]


RESIN_CATALOG: tuple[ResinCandidate, ...] = (
    ResinCandidate(
        key="opencal_udma_cq_edab",
        name="OpenCAL UDMA + CQ/EDAB (исследовательский состав)",
        target_wavelength_nm=450.0,
        cal_tested=True,
        purchasable=False,
        evidence=(
            "Опубликована печать на OpenCAL при 450 нм. В работе использовали UDMA "
            "(около 9000 мПа·с), 17 мМ камфорхинона и 8,5 мМ EDAB. Авторы "
            "отмечают, что обычные проверенные смолы почти полностью пропускали "
            "450 нм и потому разработали отдельный состав."
        ),
        known=("проверено для CAL/OpenCAL при 450 нм", "есть опубликованная рецептура и тест геометрии"),
        missing=(
            "наличие готового продукта у поставщика",
            "какая доза начинает отверждать его в вашем проекторе",
        ),
        recommendation=(
            "Лучший научный эталон именно для volumetric CAL. Это не подтверждённая "
            "обычная магазинная смола; не следует самостоятельно смешивать состав без "
            "лабораторного оснащения и подготовки."
        ),
        source_title="Публикация OpenCAL (материалы и результаты CAL)",
        source_url=(
            "https://assets.pubpub.org/5emoccq0/104%20An%20Open-Sourced%2C%20"
            "Community-Driven%20Volumetric%20Additive%20Manufacturing%20Printer%20"
            "and%20Post-Processor-31760729720398.pdf"
        ),
    ),
    ResinCandidate(
        key="photocentric_magna_draft",
        name="Photocentric Magna Draft (Daylight, профиль LC Magna 460 нм)",
        target_wavelength_nm=460.0,
        cal_tested=False,
        purchasable=True,
        evidence=(
            "Готовая Daylight-смола для LC Magna; сам производитель указывает, что "
            "LC Magna работает на 460 нм. Продукт описан как зелёный и полупрозрачный, "
            "но паспорт не даёт кривую пропускания при 450 нм. CAL-печать не проверена."
        ),
        known=(
            "Daylight-смола для послойной печати на LC Magna",
            "зелёный цвет; заявлена полупрозрачность",
            "вязкость 970 мПа·с; модуль 3200 МПа; прочность 84 МПа; удлинение 4,4%",
        ),
        missing=(
            "сколько света 450 нм проходит через слой смолы",
            "какая доза 450 нм начинает отверждать смолу",
            "CAL-контраст, сцепление томографических слоёв и пригодность для volumetric печати",
        ),
        recommendation=(
            "Самый разумный готовый материал, о котором спросить поставщика первым: "
            "совпадает с Daylight-системой и производитель заявляет полупрозрачность. "
            "Это лишь кандидат для измерения отклика, не подтверждение печати на CAL."
        ),
        source_title="Photocentric: Magna Draft TDS",
        source_url="https://photocentricgroup.com/wp-content/uploads/2025/08/TDS-Magna-Draft-2025.pdf",
    ),
    ResinCandidate(
        key="photocentric_rigid_dlfr",
        name="Photocentric Rigid DLFR (Daylight, 460 нм)",
        target_wavelength_nm=460.0,
        cal_tested=False,
        purchasable=True,
        evidence=(
            "В паспорте производителя прямо указан диапазон-профиль 460 нм; смола "
            "предназначена для LC Magna/LC Titan, а не для UV-принтеров. Это "
            "ближайший найденный готовый кандидат под видимый синий источник, "
            "но 450 нм находится на краю паспортного диапазона вашего лазера."
        ),
        known=(
            "производитель: 460 нм, продаётся в бутылках 5 кг",
            "вязкость 580 мПа·с; плотность 1,21 г/мл",
            "модуль 2848 МПа; прочность 66 МПа; удлинение 3,5% после UV-постотверждения",
            "чёрный цвет: без спектра нельзя узнать, пройдёт ли достаточно света через колбу",
        ),
        missing=(
            "сколько света 450 нм проходит через эту смолу и какая доза начинает её отверждать",
            "поведение и контраст дозы в CAL",
            "совместимость с вашей колбой и лазерным проектором",
        ),
        recommendation=(
            "Кандидат для проверки спектрального отклика при наличии оборудованного "
            "стенда; паспортные результаты послойной печати нельзя переносить на CAL."
        ),
        source_title="Photocentric: паспорт Rigid DLFR, редакция 2026",
        source_url="https://photocentricgroup.com/wp-content/uploads/2026/06/TDS-Rigid-DLFR-Black-2026.pdf",
    ),
    ResinCandidate(
        key="generic_405_resin",
        name="Обычная смола с маркировкой 405 нм (без точной марки)",
        target_wavelength_nm=405.0,
        cal_tested=False,
        purchasable=True,
        evidence=(
            "Маркировка 405 нм не означает отклик при 450 нм. Поставщик Liqcreate "
            "пишет, что некоторые 405-нм материалы реагируют до 420–440 нм, а на "
            "более длинной волне могут не отверждаться. Для неизвестной банки это "
            "не характеристика конкретного состава, а основание считать его "
            "низкоприоритетным кандидатом."
        ),
        known=("известна только типичная маркировка 405 нм",),
        missing=(
            "точное название этой смолы",
            "сколько света 450 нм проходит через неё и какая доза её отверждает",
            "контраст света внутри и за пределами модели при CAL",
        ),
        recommendation=(
            "Не выбирать первой для 450 нм. Если позже появится конкретный материал, "
            "его можно добавить в каталог по паспорту или спектральным данным."
        ),
        source_title="Liqcreate: объяснение Daylight и 385/405 нм смол",
        source_url="https://www.liqcreate.com/supportarticles/daylight-405nm-385nm-365nm-resin/",
    ),
)


def _distance_to_band(value: float, band: tuple[float, float]) -> float:
    low, high = band
    if low <= value <= high:
        return 0.0
    return min(abs(value - low), abs(value - high))


def _assess(candidate: ResinCandidate, band: tuple[float, float]) -> CandidateAssessment:
    gap = (
        _distance_to_band(candidate.target_wavelength_nm, band)
        if candidate.target_wavelength_nm is not None else None
    )
    if candidate.cal_tested and gap == 0:
        return CandidateAssessment(candidate, "cal_reference", "Лучший эталон CAL", gap, "высокая для опубликованного стенда")
    if candidate.purchasable and candidate.target_wavelength_nm is not None and gap == 0:
        status = "edge_candidate" if abs(candidate.target_wavelength_nm - sum(band) / 2) > 1 else "spectral_candidate"
        label = "Кандидат у края спектра" if status == "edge_candidate" else "Совпадает по спектру"
        return CandidateAssessment(candidate, status, label, gap, "низкая: CAL не проверен")
    return CandidateAssessment(candidate, "weak_match", "Спектральное несовпадение", gap, "низкая / данных конкретной смолы нет")


def run_resin_search(
    source: LightBudgetInput | None = None,
    *,
    laser_band_nm: tuple[float, float] | None = None,
) -> ResinSearchResult:
    """Rank evidence and sweep light-dose scenarios without inventing cure data."""

    source = source or LightBudgetInput()
    source.validate()
    if laser_band_nm is None:
        shift = source.wavelength_nm - 450.0
        laser_band_nm = (LASER_BAND_NM[0] + shift, LASER_BAND_NM[1] + shift)
    low, high = laser_band_nm
    if low > high:
        raise ValueError("Нижняя граница спектра не может быть выше верхней.")

    assessments = tuple(
        sorted(
            (_assess(item, laser_band_nm) for item in RESIN_CATALOG),
            key=lambda item: (
                0 if item.status == "cal_reference" else 1 if item.status == "edge_candidate" else 2,
                item.wavelength_gap_nm if item.wavelength_gap_nm is not None else 1_000,
            ),
        )
    )

    dose_scenarios: list[DoseScenario] = []
    for transmission_pct in TRANSMISSION_SWEEP_PCT:
        for exposure_s in EXPOSURE_SWEEP_S:
            hypothetical = replace(
                source,
                exposure_s=exposure_s,
                transmission_pct=transmission_pct,
                measured_irradiance_mw_cm2=None,
                measured_gel_dose_mj_cm2=None,
            )
            dose = evaluate_light_budget(hypothetical).plane_dose_mj_cm2
            assert dose is not None
            dose_scenarios.append(DoseScenario(transmission_pct, exposure_s, dose))

    return ResinSearchResult(
        laser_band_nm=laser_band_nm,
        laser_peak_nm=source.wavelength_nm,
        assessments=assessments,
        dose_scenarios=tuple(dose_scenarios),
        scenario_count=len(dose_scenarios),
        missing_for_prediction=(
            "сколько света от вашего лазера доходит до колбы после стёкол и матрицы",
            "сколько синего света проходит через конкретную смолу",
            "какая доза переводит эту смолу из жидкой в твёрдую при 450 нм",
            "хватает ли разницы между освещёнными и тёмными точками внутри колбы",
        ),
    )
