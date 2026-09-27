"""Beginner-facing comparison of four proposed CAL development routes."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from camera_exposure import CameraExposureResult, simulate_camera_long_exposure
from constants import APP_ORG, APP_TITLE
from development_paths import (
    PATH_TITLES,
    PathCase,
    compare_laser_to_baseline,
    evaluate_path,
)
from job_controller import JobController
from light_source_simulation import LightBudgetInput
from screening_recommendations import ROUTE_ASSESSMENTS, build_screening_plan

PATH_KEYS = ("ordinary", "alternative", "gelatin", "phone")
_INTRO = {
    "ordinary": (
        "Проверяем, достигает ли свет 450 нм обычной смолы и есть ли у неё отклик "
        "при этой длине волны. Надпись «405 нм» сама по себе ответа не даёт."
    ),
    "alternative": (
        "Проверяем другой материал при том же лазере и сравниваем его с таким же "
        "материалом под готовым проектором OpenCAL. Световая выгода и прочность — разные вопросы."
    ),
    "gelatin": (
        "B2 поглощает синий свет, но пищевой желатин + B2 нельзя считать готовой CAL-смолой. "
        "Сначала отделяем эффект света от обычного застывания при охлаждении."
    ),
    "phone": (
        "Телефон здесь — камера с длинной выдержкой. Предпросмотр показывает, как "
        "двумерный снимок смешивает свет по глубине; прозрачная вода может выглядеть тёмной."
    ),
}
_EVIDENCE = (("Неизвестно", "unknown"), ("Допущение", "scenario"), ("Измерено", "measured"))
_GUIDE = {
    "ordinary": (
        "Пока нельзя сказать, застынет ли обычная смола от этого лазера.",
        "Известно из паспорта модели: 440–460 нм, типично 450 нм и 1–1,15 Вт при штатных условиях. "
        "Мощность именно вашего снятого модуля ещё не измерена.",
        "Неизвестны потери в проекторе и реакция выбранной смолы на 450 нм. "
        "Число «405 нм» на банке не позволяет вычислить их.",
        "Уже сейчас можно проверить проекции и геометрию без смолы. "
        "Позже понадобится название/паспорт конкретной смолы или её проба под светом 450 нм.",
    ),
    "alternative": (
        "Выгода лазера пока не установлена; готовый OpenCAL служит базовым вариантом.",
        "Можно сравнить одинаковую цифровую модель и поле проекции для двух источников. "
        "OpenCAL уже использует готовый лазерный проектор.",
        "Без света у колбы и свойств одного и того же материала нельзя узнать, "
        "станет ли печать быстрее или деталь прочнее.",
        "Начните с виртуальных проекций и эталонного проекта OpenCAL. "
        "Числа для сравнения добавляются позже, если появятся измерения.",
    ),
    "gelatin": (
        "Фотосшивка желатина с рибофлавином опубликована для плёнок; печать CAL при 450 нм не доказана.",
        "В одной работе плёнки облучали 2 часа при 370–405 нм и получили частичное снижение растворимости в воде. "
        "Рибофлавин поглощает синий свет около 450 нм.",
        "Нет порога отверждения и объёмного CAL-теста именно этой смеси при 450 нм. "
        "Обычный желатин также застывает при охлаждении.",
        "Программа не угадывает дозу по другой работе. До печати сравните световой опыт с контролем без света; "
        "плёночные результаты не переносятся на деталь в колбе.",
    ),
    "phone": (
        "Длинная выдержка телефона может показать следы света, но не готовую 3D-деталь.",
        "Телефон здесь работает камерой. По созданным проекционным кадрам программа "
        "строит условный вид света и отдельно показывает накопленное поле.",
        "Реальный снимок зависит от рассеяния в жидкости, стенок колбы и камеры. "
        "Прозрачная вода может почти не показывать луч сбоку.",
        "Создайте проекции в Слайсере или выберите их папку ниже и нажмите "
        "«Рассчитать снимок камеры». Другие параметры менять не нужно.",
    ),
}


def _spin(value: float, maximum: float, *, decimals: int = 2) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.001, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(0.1 if decimals else 1)
    spin.setValue(value)
    return spin


def _evidence_combo() -> QComboBox:
    combo = QComboBox()
    for label, key in _EVIDENCE:
        combo.addItem(label, key)
    return combo


def _range(values: tuple[float, float] | None, unit: str) -> str:
    if values is None:
        return "нет данных"
    if abs(values[0] - values[1]) < 1e-9:
        return f"{values[0]:.3g} {unit}"
    return f"{values[0]:.3g}–{values[1]:.3g} {unit}"


class CameraPreviewWorker(QThread):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path: str, shutter: float, rotation: float, angle: float) -> None:
        super().__init__()
        self.path = path
        self.shutter = shutter
        self.rotation = rotation
        self.angle = angle

    def run(self) -> None:
        try:
            self.ready.emit(simulate_camera_long_exposure(
                self.path, shutter_s=self.shutter, rotation_s=self.rotation,
                camera_angle_deg=self.angle, is_cancelled=self.isInterruptionRequested,
            ))
        except (OSError, ValueError, RuntimeError) as exc:
            self.failed.emit(str(exc))

    def request_cancel(self) -> None:
        self.requestInterruption()


class DevelopmentPathsTab(QWidget):
    """Editable comparison and a camera preview connected to generated frames."""

    sourceRequested = pyqtSignal()
    resinRequested = pyqtSignal()
    slicerRequested = pyqtSignal()

    def __init__(
        self, parent: QWidget | None = None, job_controller: JobController | None = None,
    ) -> None:
        super().__init__(parent)
        self._job_controller = job_controller
        self._settings = QSettings(APP_ORG, APP_TITLE)
        self._laser = LightBudgetInput()
        self._cases = {
            "ordinary": PathCase("ordinary"),
            "alternative": PathCase("alternative"),
            "gelatin": PathCase("gelatin"),
            "phone": PathCase("phone"),
            "baseline": PathCase("baseline"),
        }
        self._rotation_s = 60.0
        self._camera_angle_deg = 0.0
        self._frames_dir = ""
        self._worker: CameraPreviewWorker | None = None
        self._loading = False
        self._restore()
        self._build_ui()
        self._load_case("ordinary")
        self._refresh()

    def _restore(self) -> None:
        raw = self._settings.value("development_paths/v1", "")
        if not isinstance(raw, str) or not raw:
            return
        try:
            data = json.loads(raw)
            for key in self._cases:
                if key in data.get("cases", {}):
                    case = PathCase(**data["cases"][key])
                    case.validate()
                    self._cases[key] = case
            self._rotation_s = float(data.get("rotation_s", 60.0))
            self._camera_angle_deg = float(data.get("camera_angle_deg", 0.0))
            if not 0 < self._rotation_s <= 86400:
                self._rotation_s = 60.0
            if not 0 <= self._camera_angle_deg <= 360:
                self._camera_angle_deg = 0.0
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        payload = {
            "schema": 1,
            "cases": {key: asdict(case) for key, case in self._cases.items()},
            "rotation_s": self._rotation_s,
            "camera_angle_deg": self._camera_angle_deg,
        }
        self._settings.setValue("development_paths/v1", json.dumps(payload, ensure_ascii=False))

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setSpacing(10)

        title = QLabel("Четыре пути проекта")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        hint = QLabel(
            "Выберите один из четырёх путей. Ничего вводить не нужно: программа уже знает "
            "паспорт модели лазера, а неизвестное честно оставит неизвестным. "
            "Ручные числа нужны только позже, если вы их узнаете."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        layout.addWidget(hint)

        self.table = QTableWidget(4, 5)
        self.table.setHorizontalHeaderLabels((
            "Путь", "Свет у колбы", "Доза полной засветки", "Вывод", "Что делать дальше",
        ))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(210)
        header = self.table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        for column, width in enumerate((270, 170, 190, 280)):
            self.table.setColumnWidth(column, width)
        vertical = self.table.verticalHeader()
        if vertical is not None:
            vertical.setVisible(False)
            vertical.setDefaultSectionSize(44)
        self.table.cellClicked.connect(lambda row, _col: self._select_row(row))
        layout.addWidget(self.table)

        quick_box = QGroupBox("Ответ для выбранного пути")
        quick_layout = QVBoxLayout(quick_box)
        self.quick_verdict = QLabel()
        self.quick_known = QLabel()
        self.quick_unknown = QLabel()
        self.quick_next = QLabel()
        for label in (self.quick_verdict, self.quick_known, self.quick_unknown, self.quick_next):
            label.setWordWrap(True)
            quick_layout.addWidget(label)
        self.quick_verdict.setObjectName("fieldLabel")
        self.quick_unknown.setObjectName("hintLabel")
        layout.addWidget(quick_box)

        self.recommend_button = QPushButton("Подобрать параметры")
        self.recommend_button.setObjectName("generateButton")
        self.recommend_button.clicked.connect(self.show_recommendations)
        layout.addWidget(self.recommend_button)

        self.advanced_toggle = QCheckBox("Подробно: ввести свои допущения или результаты измерений")
        self.advanced_toggle.setToolTip(
            "Эти поля необязательны. Если данных пока нет, ничего здесь заполнять не нужно."
        )
        self.advanced_toggle.toggled.connect(self._update_visibility)
        layout.addWidget(self.advanced_toggle)

        self.edit_box = QGroupBox("Ручные данные · необязательно")
        edit_form = QFormLayout(self.edit_box)
        self.route = QComboBox()
        for key in PATH_KEYS:
            self.route.addItem(PATH_TITLES[key], key)
        self.route.currentIndexChanged.connect(lambda _index: self._load_case(str(self.route.currentData())))
        edit_form.addRow("Вариант", self.route)
        self.intro = QLabel()
        self.intro.setWordWrap(True)
        edit_form.addRow(self.intro)
        self.laser_note = QLabel()
        self.laser_note.setWordWrap(True)
        self.laser_note.setObjectName("hintLabel")
        edit_form.addRow(self.laser_note)
        self.exposure = _spin(60, 86400, decimals=1)
        self.exposure.valueChanged.connect(self._changed)
        edit_form.addRow("Выдержка / время полной засветки, с", self.exposure)

        self.plane_row = QWidget()
        plane_layout = QHBoxLayout(self.plane_row)
        plane_layout.setContentsMargins(0, 0, 0, 0)
        self.plane_evidence = _evidence_combo()
        self.plane = _spin(5, 10000, decimals=3)
        plane_layout.addWidget(self.plane_evidence)
        plane_layout.addWidget(self.plane)
        plane_layout.addWidget(QLabel("мВт/см²"))
        edit_form.addRow("Свет нужного диапазона на материале", self.plane_row)
        self.plane_evidence.currentIndexChanged.connect(self._changed)
        self.plane.valueChanged.connect(self._changed)

        self.threshold_row = QWidget()
        threshold_layout = QHBoxLayout(self.threshold_row)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        self.threshold_evidence = _evidence_combo()
        self.threshold = _spin(50, 1_000_000)
        threshold_layout.addWidget(self.threshold_evidence)
        threshold_layout.addWidget(self.threshold)
        threshold_layout.addWidget(QLabel("мДж/см²"))
        edit_form.addRow("Порог закрепления материала", self.threshold_row)
        self.threshold_evidence.currentIndexChanged.connect(self._changed)
        self.threshold.valueChanged.connect(self._changed)

        self.thermal_control = QCheckBox(
            "При той же температуре образец без света не застыл; освещённый сохранил форму после нагрева"
        )
        self.thermal_control.toggled.connect(self._changed)
        edit_form.addRow(self.thermal_control)

        self.voxel_box = QGroupBox("Подробно: накопленная доза CAL внутри и вне детали")
        self.voxel_toggle = QCheckBox("Показать подробные параметры CAL")
        self.voxel_toggle.toggled.connect(self._update_visibility)
        edit_form.addRow(self.voxel_toggle)
        voxel_form = QFormLayout(self.voxel_box)
        self.voxel_evidence = _evidence_combo()
        self.voxel_inside = _spin(100, 1_000_000)
        self.voxel_outside = _spin(10, 1_000_000)
        voxel_form.addRow("Источник этих двух значений", self.voxel_evidence)
        voxel_form.addRow("Внутри детали, мДж/см²", self.voxel_inside)
        voxel_form.addRow("Вне детали, мДж/см²", self.voxel_outside)
        for widget in (self.voxel_evidence, self.voxel_inside, self.voxel_outside):
            signal = widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged
            signal.connect(self._changed)
        edit_form.addRow(self.voxel_box)
        layout.addWidget(self.edit_box)

        self.baseline_box = QGroupBox("Сравнение варианта 2 с готовым проектором OpenCAL")
        baseline_form = QFormLayout(self.baseline_box)
        self.baseline_evidence = _evidence_combo()
        self.baseline_plane = _spin(5, 10000, decimals=3)
        baseline_row = QWidget()
        baseline_layout = QHBoxLayout(baseline_row)
        baseline_layout.setContentsMargins(0, 0, 0, 0)
        baseline_layout.addWidget(self.baseline_evidence)
        baseline_layout.addWidget(self.baseline_plane)
        baseline_layout.addWidget(QLabel("мВт/см²"))
        baseline_form.addRow("Свет проектора у той же колбы", baseline_row)
        self.baseline_evidence.currentIndexChanged.connect(self._changed)
        self.baseline_plane.valueChanged.connect(self._changed)
        self.strength_known = QCheckBox("Есть испытания прочности деталей при одинаковом постотверждении")
        self.strength_known.toggled.connect(self._changed)
        baseline_form.addRow(self.strength_known)
        self.laser_strength = _spin(20, 10000)
        self.baseline_strength = _spin(20, 10000)
        self.laser_strength.valueChanged.connect(self._changed)
        self.baseline_strength.valueChanged.connect(self._changed)
        baseline_form.addRow("Деталь с лазером, МПа", self.laser_strength)
        baseline_form.addRow("Деталь с проектором, МПа", self.baseline_strength)
        baseline_hint = QLabel(
            "Сравнивайте один и тот же материал, размер поля и метод испытания. "
            "Если лазерной выгоды не видно, готовый OpenCAL служит эталоном для проверки модели."
        )
        baseline_hint.setWordWrap(True)
        baseline_hint.setObjectName("hintLabel")
        baseline_form.addRow(baseline_hint)
        self.benefit = QLabel()
        self.benefit.setWordWrap(True)
        baseline_form.addRow(self.benefit)
        layout.addWidget(self.baseline_box)

        self.camera_box = QGroupBox("Камера телефона: условный снимок длинной выдержки")
        camera_form = QFormLayout(self.camera_box)
        self.rotation = _spin(self._rotation_s, 86400, decimals=1)
        self.camera_angle = _spin(self._camera_angle_deg, 360, decimals=0)
        self.camera_angle.setMinimum(0.0)
        self.rotation.valueChanged.connect(self._changed)
        self.camera_angle.valueChanged.connect(self._changed)
        self.camera_settings = QWidget()
        camera_settings_form = QFormLayout(self.camera_settings)
        camera_settings_form.setContentsMargins(0, 0, 0, 0)
        camera_settings_form.addRow("Один оборот колбы, с", self.rotation)
        camera_settings_form.addRow("Угол камеры: 0° сбоку, 90° вдоль луча", self.camera_angle)
        camera_form.addRow(self.camera_settings)
        self.frame_label = QLabel("Кадры ещё не созданы. Сначала откройте Слайсер.")
        self.frame_label.setWordWrap(True)
        camera_form.addRow(self.frame_label)
        camera_buttons = QHBoxLayout()
        self.camera_button = QPushButton("Рассчитать снимок камеры")
        self.camera_button.setEnabled(False)
        self.camera_button.clicked.connect(self._start_camera_preview)
        camera_buttons.addWidget(self.camera_button)
        browse = QPushButton("Выбрать папку кадров")
        browse.clicked.connect(self._browse_frames)
        camera_buttons.addWidget(browse)
        camera_form.addRow(camera_buttons)
        self.camera_status = QLabel(
            "Условная камера суммирует рассеянный свет текущих кадров. "
            "В прозрачной воде она может почти ничего не увидеть. Снимок не равен 3D-дозе."
        )
        self.camera_status.setWordWrap(True)
        self.camera_status.setObjectName("hintLabel")
        camera_form.addRow(self.camera_status)
        previews = QHBoxLayout()
        self.dose_image = QLabel("Срез накопленного поля")
        self.dose_projection_image = QLabel("Вид накопленной 3D-дозы")
        self.camera_image = QLabel("Условный снимок камеры")
        for image in (self.dose_image, self.dose_projection_image, self.camera_image):
            image.setMinimumSize(170, 170)
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image.setStyleSheet("background: #161c27; border: 1px solid #363d4a;")
            previews.addWidget(image, 1)
        camera_form.addRow(previews)
        layout.addWidget(self.camera_box)

        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setObjectName("hintLabel")
        layout.addWidget(self.details)
        actions = QHBoxLayout()
        self.source_button = QPushButton("Открыть источник света →")
        self.source_button.clicked.connect(self.sourceRequested.emit)
        actions.addWidget(self.source_button)
        self.resin_button = QPushButton("Открыть лабораторию смолы →")
        self.resin_button.clicked.connect(self.resinRequested.emit)
        actions.addWidget(self.resin_button)
        self.slicer_button = QPushButton("К созданию проекций →")
        self.slicer_button.clicked.connect(self.slicerRequested.emit)
        actions.addWidget(self.slicer_button)
        self.export_button = QPushButton("Экспорт сравнения JSON")
        self.export_button.clicked.connect(self._export)
        actions.addWidget(self.export_button)
        layout.addLayout(actions)
        layout.addStretch(1)

    def _select_row(self, row: int) -> None:
        if 0 <= row < len(PATH_KEYS):
            self.route.setCurrentIndex(row)

    def _load_case(self, key: str) -> None:
        if key not in PATH_KEYS:
            return
        self._loading = True
        case = self._cases[key]
        self.exposure.setValue(case.exposure_s)
        for combo, evidence in (
            (self.plane_evidence, case.plane_evidence),
            (self.threshold_evidence, case.threshold_evidence),
            (self.voxel_evidence, case.voxel_evidence),
            (self.baseline_evidence, self._cases["baseline"].plane_evidence),
        ):
            combo.setCurrentIndex(max(0, combo.findData(evidence)))
        self.plane.setValue(case.plane_irradiance_mw_cm2 or 5)
        self.threshold.setValue(case.gel_threshold_mj_cm2 or 50)
        self.voxel_inside.setValue(case.voxel_inside_mj_cm2 or 100)
        self.voxel_outside.setValue(case.voxel_outside_mj_cm2 or 10)
        self.thermal_control.setChecked(case.thermal_control_passed)
        baseline = self._cases["baseline"]
        self.baseline_plane.setValue(baseline.plane_irradiance_mw_cm2 or 5)
        self.strength_known.setChecked(case.strength_mpa is not None and baseline.strength_mpa is not None)
        self.laser_strength.setValue(case.strength_mpa or 20)
        self.baseline_strength.setValue(baseline.strength_mpa or 20)
        self.rotation.setValue(self._rotation_s)
        self.camera_angle.setValue(self._camera_angle_deg)
        self._loading = False
        self.table.selectRow(PATH_KEYS.index(key))
        self._update_visibility()
        self._refresh()

    def _update_visibility(self, *_args: object) -> None:
        key = str(self.route.currentData())
        phone = key == "phone"
        advanced = self.advanced_toggle.isChecked()
        self.edit_box.setVisible(advanced)
        self.camera_settings.setVisible(advanced and phone)
        self.details.setVisible(advanced)
        self.intro.setText(_INTRO[key])
        self.laser_note.setText(
            "Лазер общ для путей 1–3. Его паспорт, поле и передача задаются на следующем этапе «Источник света»."
            if not phone else "Экран телефона ничего не освещает: работает камера с длинной выдержкой."
        )
        for widget in (self.plane_row, self.threshold_row, self.voxel_toggle):
            widget.setVisible(not phone)
        self.voxel_box.setVisible(not phone and self.voxel_toggle.isChecked())
        self.thermal_control.setVisible(key == "gelatin")
        self.baseline_box.setVisible(advanced and key == "alternative")
        self.camera_box.setVisible(phone)
        self.source_button.setVisible(not phone)
        self.resin_button.setVisible(key in {"ordinary", "alternative"})
        self.slicer_button.setVisible(phone)
        self.plane.setEnabled(self.plane_evidence.currentData() != "unknown")
        self.threshold.setEnabled(self.threshold_evidence.currentData() != "unknown")
        self.voxel_inside.setEnabled(self.voxel_evidence.currentData() != "unknown")
        self.voxel_outside.setEnabled(self.voxel_evidence.currentData() != "unknown")
        self.baseline_plane.setEnabled(self.baseline_evidence.currentData() != "unknown")
        self.laser_strength.setEnabled(self.strength_known.isChecked())
        self.baseline_strength.setEnabled(self.strength_known.isChecked())

    def _changed(self, *_args: object) -> None:
        if self._loading:
            return
        key = str(self.route.currentData())
        self._cases[key] = PathCase(
            key=key,
            exposure_s=self.exposure.value(),
            plane_irradiance_mw_cm2=(
                self.plane.value() if self.plane_evidence.currentData() != "unknown" else None
            ),
            plane_evidence=str(self.plane_evidence.currentData()),
            gel_threshold_mj_cm2=(
                self.threshold.value() if self.threshold_evidence.currentData() != "unknown" else None
            ),
            threshold_evidence=str(self.threshold_evidence.currentData()),
            voxel_inside_mj_cm2=(
                self.voxel_inside.value() if self.voxel_evidence.currentData() != "unknown" else None
            ),
            voxel_outside_mj_cm2=(
                self.voxel_outside.value() if self.voxel_evidence.currentData() != "unknown" else None
            ),
            voxel_evidence=str(self.voxel_evidence.currentData()),
            strength_mpa=(self.laser_strength.value() if key == "alternative" and self.strength_known.isChecked() else None),
            thermal_control_passed=self.thermal_control.isChecked(),
        )
        if key == "alternative":
            self._cases["baseline"] = PathCase(
                key="baseline",
                exposure_s=self.exposure.value(),
                plane_irradiance_mw_cm2=(
                    self.baseline_plane.value() if self.baseline_evidence.currentData() != "unknown" else None
                ),
                plane_evidence=str(self.baseline_evidence.currentData()),
                gel_threshold_mj_cm2=self._cases[key].gel_threshold_mj_cm2,
                threshold_evidence=self._cases[key].threshold_evidence,
                strength_mpa=self.baseline_strength.value() if self.strength_known.isChecked() else None,
            )
        self._rotation_s = self.rotation.value()
        self._camera_angle_deg = self.camera_angle.value()
        self._update_visibility()
        self._save()
        self._refresh()

    def set_laser_source(self, laser: LightBudgetInput) -> None:
        self._laser = laser
        self._refresh()

    def show_recommendations(self) -> None:
        """Show a complete, no-input assessment of all four proposed paths."""

        plan = build_screening_plan(self._laser)
        dialog = QDialog(self)
        dialog.setWindowTitle("Подбор стартовых параметров и четыре гипотезы")
        dialog.resize(1300, 860)
        layout = QVBoxLayout(dialog)

        summary = QLabel(
            "Ничего вводить не нужно. Без смолы и измерений нельзя честно выбрать одну «оптимальную» выдержку. "
            "Для первого сравнения использованы паспортный пример (450 нм типично, диапазон 440–460 нм; "
            f"мощность {plan.power_w[0]:g}–{plan.power_w[1]:g} Вт), "
            f"поле {plan.field_cm2:g} см², засветка {plan.exposure_s:g} с. "
            "Размер поля и время — пример для арифметики. Передача 1–25% — сценарии, а не измерения вашего проектора."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        scenario_table = QTableWidget(len(plan.scenarios) + 1, 3)
        scenario_table.setHorizontalHeaderLabels(
            ("Передача тракта", "Свет у колбы, мВт/см²", "Доза полной засветки, мДж/см²")
        )
        scenario_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        scenario_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        ideal_row = (
            "100% · идеальный предел",
            _range(plan.ideal_irradiance_mw_cm2, ""),
            _range(plan.ideal_full_field_dose_mj_cm2, ""),
        )
        rows = [ideal_row] + [
            (
                f"{scenario.transmission_pct:g}% · допущение",
                _range(scenario.irradiance_mw_cm2, ""),
                _range(scenario.full_field_dose_mj_cm2, ""),
            )
            for scenario in plan.scenarios
        ]
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip("Верхняя граница или условный сценарий; это не измерение и не прогноз отверждения.")
                if row > 0:
                    item.setForeground(QColor("#e8bd6f"))
                scenario_table.setItem(row, column, item)
        scenario_table.setMaximumHeight(190)
        scenario_header = scenario_table.horizontalHeader()
        if scenario_header is not None:
            scenario_header.setStretchLastSection(True)
        layout.addWidget(scenario_table)

        dose_note = QLabel(
            "Это арифметика полной равномерной засветки I×t. При CAL колба вращается, "
            "кадры меняются, а доза отдельной точки будет другой. Поэтому порог отверждения "
            "без данных смолы не подставлен. Проценты потерь показаны только как диапазон «что если»."
        )
        dose_note.setWordWrap(True)
        dose_note.setObjectName("hintLabel")
        layout.addWidget(dose_note)

        verdict_table = QTableWidget(len(ROUTE_ASSESSMENTS), 4)
        verdict_table.setHorizontalHeaderLabels(
            ("Гипотеза", "Вывод", "Что следует из источников", "Что это означает для печати")
        )
        verdict_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        verdict_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for row, (key, verdict, basis, next_step) in enumerate(ROUTE_ASSESSMENTS):
            for column, value in enumerate((PATH_TITLES[key], verdict, basis, next_step)):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                verdict_table.setItem(row, column, item)
            verdict_table.setRowHeight(row, 106)
        verdict_table.setWordWrap(True)
        verdict_table.setMinimumHeight(420)
        for column, width in enumerate((190, 230, 420, 420)):
            verdict_table.setColumnWidth(column, width)
        layout.addWidget(verdict_table, 1)

        sources = QLabel(
            'Источники: <a href="https://www.bu-lasers.com/wp-content/uploads/2025/09/jap45160z-004-2.pdf">паспорт лазера</a> · '
            '<a href="https://opencal-org.readthedocs.io/en/stable/software/tomo_workflow/">OpenCAL / Tomo</a> · '
            '<a href="https://www.sciencedirect.com/science/article/pii/S0268005X23009177">желатин + рибофлавин</a>'
        )
        sources.setOpenExternalLinks(True)
        sources.setObjectName("hintLabel")
        layout.addWidget(sources)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def set_frames_dir(self, path: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
        self._frames_dir = path
        self.frame_label.setText(f"Кадры: {path}" if path else "Кадры ещё не созданы в Слайсере.")
        self.camera_button.setEnabled(bool(path))
        self._refresh()

    def _refresh(self) -> None:
        results = {
            key: evaluate_path(case, self._laser if key in PATH_KEYS[:3] else None)
            for key, case in self._cases.items()
        }
        for row, key in enumerate(PATH_KEYS):
            result = results[key]
            missing = result.missing[0] if result.missing else "Можно сравнить с тестом"
            if key == "phone":
                missing = (
                    "Рассчитать условный снимок по кадрам"
                    if self._frames_dir else "Создать проекционные кадры"
                )
            values = (
                PATH_TITLES[key],
                "камера, не источник" if key == "phone" else
                _range(result.plane_irradiance_mw_cm2, "мВт/см²"),
                "не применимо" if key == "phone" else
                _range(result.full_field_dose_mj_cm2, "мДж/см²"),
                result.status,
                missing,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(result.explanation)
                if column == 3:
                    item.setForeground(QColor("#8bd1a2" if result.cal_margin_mj_cm2 is not None and result.cal_margin_mj_cm2 > 0 else "#e8bd6f"))
                self.table.setItem(row, column, item)
        key = str(self.route.currentData())
        result = results[key]
        details = [result.explanation]
        if result.ideal_irradiance_mw_cm2 is not None:
            details.append(
                "Идеальный верхний предел до оптики: "
                + _range(result.ideal_irradiance_mw_cm2, "мВт/см²") + "."
            )
        if result.threshold_ratio is not None:
            details.append(
                "Доза полной засветки / порог: " + _range(result.threshold_ratio, "×")
                + ". Это не доза вокселя при CAL."
            )
        if result.cal_margin_mj_cm2 is not None:
            details.append(
                f"Запас до порога для пары «внутри / снаружи»: {result.cal_margin_mj_cm2:+.3g} мДж/см² "
                f"({self._cases[key].voxel_evidence})."
            )
        missing_details = list(result.missing)
        if key == "phone" and self._frames_dir:
            missing_details = missing_details[1:]
        details.append("Не хватает: " + "; ".join(missing_details) + ".")
        self.details.setText("\n".join(details))
        self.details.setVisible(self.advanced_toggle.isChecked())
        verdict, known, unknown, next_step = _GUIDE[key]
        if result.cal_margin_mj_cm2 is not None:
            verdict = result.status + ". Это предварительная проверка, не гарантия печати."
        self.quick_verdict.setText(verdict)
        self.quick_known.setText("Что известно: " + known)
        self.quick_unknown.setText("Чего пока не знаем: " + unknown)
        self.quick_next.setText("Что делать дальше: " + next_step)
        alternative = self._cases["alternative"]
        baseline = replace(
            self._cases["baseline"],
            exposure_s=alternative.exposure_s,
            gel_threshold_mj_cm2=alternative.gel_threshold_mj_cm2,
            threshold_evidence=alternative.threshold_evidence,
        )
        baseline_result = evaluate_path(baseline)
        comparison = compare_laser_to_baseline(
            results["alternative"], baseline_result, alternative, baseline,
        )
        if comparison.intensity_ratio is None:
            benefit_text = "Световая выгода лазера пока неизвестна: нужен свет у колбы для обоих источников."
        else:
            benefit_text = (
                "Отношение освещённости лазер / проектор: "
                + _range(comparison.intensity_ratio, "×")
                + f" ({comparison.evidence})."
            )
        if comparison.strength_change_pct is not None:
            benefit_text += f" Измеренная разница прочности: {comparison.strength_change_pct:+.1f}%."
        benefit_text += " " + comparison.interpretation
        self.benefit.setText(benefit_text)

    def _browse_frames(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка проекционных кадров", self._frames_dir)
        if path:
            self.set_frames_dir(path)

    def _start_camera_preview(self) -> None:
        if not self._frames_dir or self._worker is not None and self._worker.isRunning():
            return
        self.camera_button.setEnabled(False)
        self.camera_status.setText("Суммирую проекции за выдержку камеры...")
        worker = CameraPreviewWorker(
            self._frames_dir, self._cases["phone"].exposure_s,
            self._rotation_s, self._camera_angle_deg,
        )
        self._worker = worker
        if self._job_controller is not None:
            self._job_controller.register(worker)
        worker.ready.connect(self._camera_ready)
        worker.failed.connect(self._camera_failed)
        worker.finished.connect(lambda: self.camera_button.setEnabled(bool(self._frames_dir)))
        worker.start()

    @staticmethod
    def _image_pixmap(values: np.ndarray) -> QPixmap:
        maximum = float(np.percentile(values, 99.5))
        scaled = np.clip(values / max(maximum, 1e-8), 0, 1) ** 0.65
        pixels = np.ascontiguousarray(np.rint(scaled * 255).astype(np.uint8))
        image = QImage(
            pixels.tobytes(), pixels.shape[1], pixels.shape[0], pixels.shape[1],
            QImage.Format.Format_Grayscale8,
        ).copy()
        return QPixmap.fromImage(image).scaled(
            280, 240, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _camera_ready(self, result: CameraExposureResult) -> None:
        self.dose_image.setPixmap(self._image_pixmap(result.dose_midplane))
        self.dose_projection_image.setPixmap(self._image_pixmap(result.dose_projection))
        self.camera_image.setPixmap(self._image_pixmap(result.camera_image))
        self.camera_status.setText(
            f"Использовано {result.frames_used} из {result.frames_available} кадров; "
            f"выдержка охватывает {result.rotation_fraction:.2f} оборота. "
            "Слева — срез накопленного поля в материале; в центре — его вид сбоку. "
            "Справа — условный снимок камеры: свет в объёме колбы при слабом "
            "равномерном рассеянии и выбранном угле. Эти изображения могут различаться. "
            "Яркость нормирована."
        )

    def _camera_failed(self, message: str) -> None:
        self.camera_status.setText("Не удалось построить снимок: " + message)

    def _export(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self, "Экспорт сравнения", "development_paths.json", "JSON (*.json)",
        )
        if not path:
            return
        try:
            payload = {
                "schema": 1,
                "model": "uniform full-field light budget; camera preview uses uniform scattering",
                "laser": asdict(self._laser),
                "cases": {key: asdict(case) for key, case in self._cases.items()},
                "rotation_s": self._rotation_s,
                "camera_angle_deg": self._camera_angle_deg,
                "source_frames": self._frames_dir,
            }
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Экспорт не выполнен", str(exc))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait()
        super().closeEvent(event)
