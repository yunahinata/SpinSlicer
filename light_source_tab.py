"""Interactive, measurement-aware light budget for the proposed CAL source."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from light_source_simulation import (
    LightBudgetInput,
    LightBudgetResult,
    evaluate_light_budget,
    what_if_transmission,
)


def _spin(value: float, lower: float, upper: float, decimals: int = 2) -> QDoubleSpinBox:
    control = QDoubleSpinBox()
    control.setRange(lower, upper)
    control.setDecimals(decimals)
    control.setValue(value)
    control.setSingleStep(1 if decimals == 0 else 0.05)
    return control


class LightSourceTab(QWidget):
    """Compare the laser nameplate envelope with measured downstream inputs."""

    scenarioSelected = pyqtSignal(object)
    inputsChanged = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result: LightBudgetResult | None = None
        self._build_ui()
        self._wire_inputs()
        self._refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(10)
        scroll.setWidget(body)

        title = QLabel("Лазер → матрица → колба: проверка гипотезы")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        intro = QLabel(
            "На фото — JAP45160Z-004, лазер класса IV. Паспорт модели: 440–460 нм "
            "(номинал 450) и 1,0–1,15 Вт оптической мощности. Каталог того же "
            "производителя заявляет 1,2–1,6 Вт: мощность вашего экземпляра не измерена. "
            "Показанные ниже числа — бюджет света, а не обещание отверждения или CAL-печати."
        )
        intro.setWordWrap(True)
        intro.setObjectName("hintLabel")
        layout.addWidget(intro)

        self.advanced_toggle = QCheckBox("Подробно: изменить паспортные данные или добавить измерения")
        self.advanced_toggle.setToolTip(
            "Необязательно. Паспорт модели уже заполнен; неизвестные параметры можно оставить пустыми."
        )
        layout.addWidget(self.advanced_toggle)

        self.form_box = QGroupBox("Исходные данные")
        form = QFormLayout(self.form_box)
        self.wavelength = _spin(450, 200, 1000, 0)
        self.power_min = _spin(1.0, 0.001, 100, 3)
        self.power_max = _spin(1.15, 0.001, 100, 3)
        self.field_width = _spin(20, 1, 1000, 1)
        self.field_height = _spin(50, 1, 1000, 1)
        self.exposure = _spin(60, 0.01, 3600, 1)
        for label, control, unit in (
            ("Центральная длина волны", self.wavelength, "нм"),
            ("Мощность источника, минимум", self.power_min, "Вт"),
            ("Мощность источника, максимум", self.power_max, "Вт"),
            ("Ширина засвечиваемого поля", self.field_width, "мм"),
            ("Высота засвечиваемого поля", self.field_height, "мм"),
            ("Время полной засветки", self.exposure, "с"),
        ):
            row = QHBoxLayout()
            row.addWidget(control, 1)
            row.addWidget(QLabel(unit))
            form.addRow(label, row)
        field_hint = QLabel(
            "Поле 20 × 50 мм и 60 с — только пример для сравнения. В CAL отдельные "
            "пиксели включаются по кадрам, поэтому суммарная доза в объёме будет иной."
        )
        field_hint.setWordWrap(True)
        field_hint.setObjectName("hintLabel")
        form.addRow(field_hint)
        layout.addWidget(self.form_box)

        self.measured_box = QGroupBox("Измерения и допущения")
        measured_form = QFormLayout(self.measured_box)
        self.transmission_known = QCheckBox("Задать передачу всего оптического тракта")
        self.transmission = _spin(10, 0.01, 100, 2)
        self.transmission.setEnabled(False)
        self.intensity_known = QCheckBox("Измерена интенсивность в плоскости колбы")
        self.intensity = _spin(5, 0.001, 10000, 3)
        self.intensity.setEnabled(False)
        self.gel_known = QCheckBox("Измерен порог материала при этой длине волны")
        self.gel_dose = _spin(50, 0.001, 100000, 2)
        self.gel_dose.setEnabled(False)
        for check, spin, unit in (
            (self.transmission_known, self.transmission, "%"),
            (self.intensity_known, self.intensity, "мВт/см²"),
            (self.gel_known, self.gel_dose, "мДж/см²"),
        ):
            row = QHBoxLayout()
            row.addWidget(spin, 1)
            row.addWidget(QLabel(unit))
            measured_form.addRow(check, row)
        measured_hint = QLabel(
            "Передача тракта включает рассеиватель, матрицу, объектив и стенки колбы. "
            "Без измерения она неизвестна. Измеренная интенсивность, если задана, имеет приоритет."
        )
        measured_hint.setWordWrap(True)
        measured_hint.setObjectName("hintLabel")
        measured_form.addRow(measured_hint)
        layout.addWidget(self.measured_box)
        self.advanced_toggle.toggled.connect(self._set_advanced)
        self._set_advanced(False)

        results = QGroupBox("Световой бюджет")
        results_layout = QVBoxLayout(results)
        self.upper_bound = QLabel()
        self.plane_estimate = QLabel()
        self.cure_estimate = QLabel()
        self.missing = QLabel()
        for result_label in (self.upper_bound, self.plane_estimate, self.cure_estimate, self.missing):
            result_label.setWordWrap(True)
            results_layout.addWidget(result_label)
        self.missing.setObjectName("hintLabel")
        layout.addWidget(results)

        scenario_box = QGroupBox("Если потери будут такими · примеры, не измерения")
        scenario_layout = QVBoxLayout(scenario_box)
        self.scenarios = QTableWidget(4, 3)
        self.scenarios.setHorizontalHeaderLabels(
            ("Передача", "Интенсивность у колбы, мВт/см²", "Доза за полную засветку, мДж/см²")
        )
        horizontal_header = self.scenarios.horizontalHeader()
        vertical_header = self.scenarios.verticalHeader()
        if horizontal_header is not None:
            horizontal_header.setStretchLastSection(True)
        if vertical_header is not None:
            vertical_header.setVisible(False)
        self.scenarios.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.scenarios.setMaximumHeight(175)
        scenario_layout.addWidget(self.scenarios)
        layout.addWidget(scenario_box)

        dry_run = QGroupBox("Что можно проверить без смолы")
        dry_layout = QVBoxLayout(dry_run)
        dry_text = QLabel(
            "1. Сгенерировать проекции и проверить синхронизацию кадров с вращением в виртуальном устройстве.\n"
            "2. Проверить геометрию, резкость, размеры пикселя и искажения с безопасным "
            "серийным проектором и пустой/водяной колбой.\n"
            "3. Для этого лазера — только в закрытом, межблокированном оптическом тракте "
            "измерить спектр, интенсивность и равномерность на плоскости колбы.\n"
            "4. Отверждение и запас CAL-контраста останутся неизвестными до проверки "
            "конкретного материала при 450 нм."
        )
        dry_text.setWordWrap(True)
        dry_layout.addWidget(dry_text)
        layout.addWidget(dry_run)

        safety = QLabel(
            "Класс IV: прямой и отражённый луч опасны для глаз и кожи; рассеивание "
            "само по себе не делает открытый тракт безопасным. Не включайте снятый "
            "модуль в открытом проекторе."
        )
        safety.setObjectName("hintLabel")
        safety.setWordWrap(True)
        layout.addWidget(safety)
        self.apply_button = QPushButton("Перейти к виртуальной оптике →")
        self.apply_button.setObjectName("generateButton")
        self.apply_button.clicked.connect(lambda: self.scenarioSelected.emit(self.inputs()))
        layout.addWidget(self.apply_button)
        layout.addStretch(1)

    def _set_advanced(self, visible: bool) -> None:
        self.form_box.setVisible(visible)
        self.measured_box.setVisible(visible)

    def _wire_inputs(self) -> None:
        for control in (
            self.wavelength, self.power_min, self.power_max, self.field_width, self.field_height,
            self.exposure, self.transmission, self.intensity, self.gel_dose,
        ):
            control.valueChanged.connect(self._refresh)
        for check, control in (
            (self.transmission_known, self.transmission),
            (self.intensity_known, self.intensity),
            (self.gel_known, self.gel_dose),
        ):
            check.toggled.connect(control.setEnabled)
            check.toggled.connect(self._refresh)

    def inputs(self) -> LightBudgetInput:
        return LightBudgetInput(
            wavelength_nm=self.wavelength.value(),
            power_min_w=self.power_min.value(),
            power_max_w=self.power_max.value(),
            image_width_mm=self.field_width.value(),
            image_height_mm=self.field_height.value(),
            exposure_s=self.exposure.value(),
            transmission_pct=self.transmission.value() if self.transmission_known.isChecked() else None,
            measured_irradiance_mw_cm2=self.intensity.value() if self.intensity_known.isChecked() else None,
            measured_gel_dose_mj_cm2=self.gel_dose.value() if self.gel_known.isChecked() else None,
        )

    @staticmethod
    def _range(values: tuple[float, float], unit: str) -> str:
        return f"{values[0]:.2f}–{values[1]:.2f} {unit}"

    def _refresh(self, *_args: object) -> None:
        inputs = self.inputs()
        try:
            result = evaluate_light_budget(inputs)
        except ValueError as exc:
            self.result = None
            self.upper_bound.setText(f"Проверьте значения: {exc}")
            self.upper_bound.setStyleSheet("color: #e8bd6f")
            self.plane_estimate.clear()
            self.cure_estimate.clear()
            self.missing.clear()
            self.scenarios.setRowCount(0)
            self.apply_button.setEnabled(False)
            return
        self.result = result
        self.inputsChanged.emit(inputs)
        self.apply_button.setEnabled(True)
        self.upper_bound.setStyleSheet("")
        self.upper_bound.setText(
            f"Поле {result.area_cm2:.2f} см²; идеальная средняя освещённость до оптики: "
            f"{self._range(result.source_irradiance_mw_cm2, 'мВт/см²')}. "
            "Это верхний предел при равномерном распределении всей мощности."
        )
        if result.plane_irradiance_mw_cm2 is None or result.plane_dose_mj_cm2 is None:
            self.plane_estimate.setText("На колбе: нет оценки — пропускание и интенсивность не заданы.")
        else:
            self.plane_estimate.setText(
                f"На колбе: {self._range(result.plane_irradiance_mw_cm2, 'мВт/см²')}; "
                f"за {inputs.exposure_s:g} с полной засветки: "
                f"{self._range(result.plane_dose_mj_cm2, 'мДж/см²')}. {result.basis}."
            )
        if result.gel_dose_ratio is None:
            self.cure_estimate.setText("Отверждение: не оценивается без измеренного порога материала и дозы на колбе.")
        else:
            self.cure_estimate.setText(
                f"Доза полной засветки / заданный порог: "
                f"{result.gel_dose_ratio[0]:.2f}–{result.gel_dose_ratio[1]:.2f}×. "
                "Для CAL ещё нужны объёмное накопление дозы, фоновая засветка и влияние кислорода."
            )
        self.missing.setText(
            "Что остаётся неизвестным: " + "; ".join(result.missing)
            + (("\nПроверка: " + " ".join(result.warnings)) if result.warnings else "")
        )
        self.scenarios.setRowCount(4)
        for row, transmission in enumerate((1.0, 5.0, 10.0, 25.0)):
            irradiance = what_if_transmission(inputs, transmission)
            dose = (irradiance[0] * inputs.exposure_s, irradiance[1] * inputs.exposure_s)
            values = (
                f"{transmission:g}% · допущение",
                self._range(irradiance, ""),
                self._range(dose, ""),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setForeground(QColor("#e8bd6f"))
                item.setToolTip("Условный сценарий, а не измерение вашего проектора")
                self.scenarios.setItem(row, column, item)
