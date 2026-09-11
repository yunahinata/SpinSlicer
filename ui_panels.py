"""
ui_panels.py
============
Две боковые панели интерфейса:

  ProcessSettingsPanel (левая) — глобальные настройки принтера/процесса:
    диаметр колбы, параметры фотополимера, разрешение сетки, кол-во кадров.

  ObjectPanel (правая) — параметры выбранной модели: точные размеры в мм,
    Uniform Scale, поворот, сдвиг, кнопки центрирования/авто-фита/сброса.

Обе панели ничего не знают про trimesh/PyVista напрямую — они лишь читают
и пишут числа. Всю связь с ModelNode делает главное окно.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from constants import (
    DEFAULT_DIAMETER_MM,
    DEFAULT_GRID_RESOLUTION,
    DEFAULT_NUM_FRAMES,
    DEFAULT_OUTPUT_RESOLUTION,
)
from i18n import tr
from model_node import ModelNode
from widgets import AxisNudgeControl, LabeledSlider


def _scroll_wrap(inner: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(inner)
    return scroll


class ProcessSettingsPanel(QWidget):
    """Левая панель: глобальные настройки принтера и процесса печати."""

    diameterChanged = pyqtSignal(float)
    settingsChanged = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        title = QLabel("Настройки процесса")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        # --- Колба / принтер -----------------------------------------------
        vat_box = QGroupBox("Печатная колба")
        vat_layout = QVBoxLayout(vat_box)
        self.diameter = LabeledSlider("Диаметр колбы, мм", 20.0, 200.0, DEFAULT_DIAMETER_MM, decimals=1)
        vat_layout.addWidget(self.diameter)
        layout.addWidget(vat_box)

        # --- Смола -----------------------------------------------------------
        resin_box = QGroupBox("Параметры фотополимера")
        resin_layout = QVBoxLayout(resin_box)
        self.exposure = LabeledSlider("Базовое время засветки, с", 0.1, 20.0, 1.0, decimals=2)
        self.intensity = LabeledSlider("Интенсивность источника, %", 1.0, 200.0, 100.0, decimals=1)
        self.threshold = LabeledSlider("Порог полимеризации, %", 0.0, 100.0, 0.0, decimals=1)
        resin_layout.addWidget(self.exposure)
        resin_layout.addWidget(self.intensity)
        resin_layout.addWidget(self.threshold)
        hint = QLabel("Порог и интенсивность определяют итоговую\nконтрастность и яркость кадров.")
        hint.setObjectName("hintLabel")
        resin_layout.addWidget(hint)
        layout.addWidget(resin_box)

        # --- Сетка и рендер -----------------------------------------------
        grid_box = QGroupBox("Разрешение и кадры")
        grid_layout = QVBoxLayout(grid_box)
        self.grid_res = LabeledSlider("Разрешение сетки (Voxel Grid)", 64, 256, DEFAULT_GRID_RESOLUTION, decimals=0, step=1)
        self.output_res = LabeledSlider("Разрешение кадра, px", 128, 2048, DEFAULT_OUTPUT_RESOLUTION, decimals=0, step=1)
        self.frames = LabeledSlider("Количество кадров", 30, 720, DEFAULT_NUM_FRAMES, decimals=0, step=1)
        grid_layout.addWidget(self.grid_res)
        grid_layout.addWidget(self.output_res)
        grid_layout.addWidget(self.frames)

        self.fill_holes = QCheckBox("Сплошная заливка (ремонт сетки)")
        self.fill_holes.setChecked(True)
        grid_layout.addWidget(self.fill_holes)
        self.preserve_internal_voids = QCheckBox(
            "Preserve internal voids (bores / threads)"
        )
        self.preserve_internal_voids.setChecked(False)
        grid_layout.addWidget(self.preserve_internal_voids)
        layout.addWidget(grid_box)

        # --- Projection engine --------------------------------------------
        backend_box = QGroupBox("Projection engine")
        backend_layout = QVBoxLayout(backend_box)
        self.projection_backend = QComboBox()
        self.projection_backend.addItem("SpinSlicer internal Radon (recommended)", "internal")
        self.projection_backend.addItem("VAMToolbox CAL (optional)", "vamtoolbox")
        self.projection_backend.addItem(
            "Auto: VAMToolbox CAL → SpinSlicer fallback (experimental)", "auto"
        )
        backend_layout.addWidget(self.projection_backend)
        self.optimizer_iterations = LabeledSlider(
            "VAM optimizer iterations", 1, 100, 8, decimals=0, step=1,
        )
        backend_layout.addWidget(self.optimizer_iterations)
        backend_hint = QLabel(
            "SpinSlicer internal Radon is the default engine. VAMToolbox is "
            "available only when explicitly selected."
        )
        backend_hint.setObjectName("hintLabel")
        backend_hint.setWordWrap(True)
        backend_layout.addWidget(backend_hint)
        layout.addWidget(backend_box)

        layout.addStretch(1)
        root.addWidget(_scroll_wrap(content))

        self.diameter.valueChanged.connect(self.diameterChanged.emit)
        self.diameter.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.fill_holes.toggled.connect(lambda _v: self.settingsChanged.emit())
        self.preserve_internal_voids.toggled.connect(lambda _v: self.settingsChanged.emit())
        self.projection_backend.currentIndexChanged.connect(lambda _v: self.settingsChanged.emit())
        self.optimizer_iterations.valueChanged.connect(lambda _v: self.settingsChanged.emit())

    def vat_diameter_mm(self) -> float:
        return self.diameter.value()

    def projection_backend_name(self) -> str:
        return str(self.projection_backend.currentData())

    def optimizer_iterations_value(self) -> int:
        return self.optimizer_iterations.value_int()


class ObjectPanel(QWidget):
    """Правая панель: точные физические трансформации выбранной модели."""

    fieldChanged = pyqtSignal(str, float)
    centerRequested = pyqtSignal()
    autoFitRequested = pyqtSignal()
    resetRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._syncing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        title = QLabel("Объект")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        info_box = QGroupBox("Модель")
        info_layout = QVBoxLayout(info_box)
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setObjectName("hintLabel")
        self.info_label = QLabel("")
        self.info_label.setObjectName("hintLabel")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.file_label)
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_box)

        # --- Bambu-style transform modes -----------------------------------
        transform_box = QGroupBox("Трансформация")
        transform_layout = QVBoxLayout(transform_box)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self.move_mode_btn = QPushButton("Перемещение")
        self.rotate_mode_btn = QPushButton("Вращение")
        self.scale_mode_btn = QPushButton("Масштаб")
        self._mode_buttons = (
            self.move_mode_btn,
            self.rotate_mode_btn,
            self.scale_mode_btn,
        )
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for index, button in enumerate(self._mode_buttons):
            button.setCheckable(True)
            button.setObjectName("transformModeButton")
            self._mode_group.addButton(button, index)
            mode_row.addWidget(button, 1)
        self.move_mode_btn.setChecked(True)
        transform_layout.addLayout(mode_row)

        self._transform_stack = QStackedWidget()
        self.pos_x, self.pos_y, self.pos_z = self._make_axis_controls(
            self._transform_stack,
            "мм",
            minimum=-100.0,
            maximum=100.0,
            decimals=2,
            step=1.0,
            suffix=" мм",
        )
        self.rot_x, self.rot_y, self.rot_z = self._make_axis_controls(
            self._transform_stack,
            "°",
            minimum=-360.0,
            maximum=360.0,
            decimals=1,
            step=15.0,
            suffix="°",
        )
        self.size_x, self.size_y, self.size_z = self._make_axis_controls(
            self._transform_stack,
            "мм",
            minimum=0.1,
            maximum=500.0,
            decimals=2,
            step=1.0,
            suffix=" мм",
        )
        transform_layout.addWidget(self._transform_stack)

        self.uniform_scale = QCheckBox("Uniform Scale (сохранять пропорции)")
        self.uniform_scale.setChecked(True)
        transform_layout.addWidget(self.uniform_scale)
        layout.addWidget(transform_box)

        # --- Быстрые действия -------------------------------------------
        actions = QHBoxLayout()
        self.center_btn = QPushButton("Центрировать")
        self.autofit_btn = QPushButton("Авто-фит под колбу")
        self.reset_btn = QPushButton("Сбросить")
        actions.addWidget(self.center_btn)
        actions.addWidget(self.autofit_btn)
        actions.addWidget(self.reset_btn)
        layout.addLayout(actions)

        layout.addStretch(1)
        root.addWidget(_scroll_wrap(content))

        self._wire_signals()
        self.set_enabled_state(False)

    @staticmethod
    def _make_axis_controls(
        stack: QStackedWidget,
        axis_suffix: str,
        *,
        minimum: float,
        maximum: float,
        decimals: int,
        step: float,
        suffix: str,
    ) -> tuple[AxisNudgeControl, AxisNudgeControl, AxisNudgeControl]:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 8, 0, 0)
        page_layout.setSpacing(6)

        controls = tuple(
            AxisNudgeControl(
                f"{axis}, {axis_suffix}",
                minimum,
                maximum,
                0.0 if minimum <= 0.0 else 1.0,
                decimals=decimals,
                step=step,
                suffix=suffix,
            )
            for axis in ("X", "Y", "Z")
        )
        for control in controls:
            page_layout.addWidget(control)
        page_layout.addStretch(1)
        stack.addWidget(page)
        return controls[0], controls[1], controls[2]

    # --- сигналы -------------------------------------------------------------
    def _wire_signals(self) -> None:
        field_map = {
            "size_x": self.size_x, "size_y": self.size_y, "size_z": self.size_z,
            "rot_x": self.rot_x, "rot_y": self.rot_y, "rot_z": self.rot_z,
            "pos_x": self.pos_x, "pos_y": self.pos_y, "pos_z": self.pos_z,
        }
        for key, w in field_map.items():
            w.valueChanged.connect(lambda v, k=key: self._emit_field(k, v))

        self.move_mode_btn.clicked.connect(lambda: self._transform_stack.setCurrentIndex(0))
        self.rotate_mode_btn.clicked.connect(lambda: self._transform_stack.setCurrentIndex(1))
        self.scale_mode_btn.clicked.connect(lambda: self._transform_stack.setCurrentIndex(2))

        self.center_btn.clicked.connect(self.centerRequested.emit)
        self.autofit_btn.clicked.connect(self.autoFitRequested.emit)
        self.reset_btn.clicked.connect(self.resetRequested.emit)

    def _emit_field(self, key: str, value: float) -> None:
        # Во время программной синхронизации (sync_from_model) сигналы
        # подавляются — иначе пришлось бы обрабатывать самим же собой
        # порождённое "эхо" на каждое обновление после Uniform Scale.
        if not self._syncing:
            self.fieldChanged.emit(key, value)

    def is_uniform(self) -> bool:
        return self.uniform_scale.isChecked()

    # --- состояние -------------------------------------------------------------
    def set_enabled_state(self, enabled: bool) -> None:
        for w in (self.size_x, self.size_y, self.size_z, self.uniform_scale,
                  self.rot_x, self.rot_y, self.rot_z,
                  self.pos_x, self.pos_y, self.pos_z,
                  self.center_btn, self.autofit_btn, self.reset_btn):
            w.setEnabled(enabled)

    def show_model_info(self, filename: str, node: ModelNode) -> None:
        self.file_label.setText(filename)
        extents = node.base_extents
        self.info_label.setText(
            f"{tr('Вершин:')} {node.vertex_count:,}\n"
            f"{tr('Граней:')} {node.face_count:,}\n"
            f"{tr('Исходные размеры:')} "
            f"{extents[0]:.2f} × {extents[1]:.2f} × {extents[2]:.2f} {tr('мм')}"
        )

    # --- синхронизация с ModelNode ---------------------------------------------
    def sync_from_model(self, node: ModelNode) -> None:
        self._syncing = True
        size = node.current_size_mm()
        self.size_x.setValue(float(size[0]))
        self.size_y.setValue(float(size[1]))
        self.size_z.setValue(float(size[2]))

        rot = node.transform.rotation_deg
        self.rot_x.setValue(float(rot[0]))
        self.rot_y.setValue(float(rot[1]))
        self.rot_z.setValue(float(rot[2]))

        pos = node.transform.translation
        self.pos_x.setValue(float(pos[0]))
        self.pos_y.setValue(float(pos[1]))
        self.pos_z.setValue(float(pos[2]))
        self._syncing = False
