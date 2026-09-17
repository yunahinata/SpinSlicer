"""
ui_panels.py
============
Виджеты настроек и управления моделью.

  ProcessSettingsPanel — настройки принтера и процесса, вынесенные в
    отдельную вкладку приложения.
  TransformToolbar — компактное управление gizmo над 3D-вьюпортом.
  ObjectPanel — прежняя расширенная панель объекта, оставленная для обратной
    совместимости с внешними импортами.

Виджеты ничего не знают про PyVista напрямую — связь с ModelNode и сценой
делает SlicerTab.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
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
from widgets import LabeledSlider


def _scroll_wrap(inner: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(inner)
    return scroll


class ProcessSettingsPanel(QWidget):
    """Настройки принтера и процесса печати для отдельной вкладки."""

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

        title = QLabel("Настройки принтера и процесса")
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
        # Threaded STL models need their bore and thread relief to survive
        # rasterization; keep the safe behavior enabled for every model.
        self.preserve_internal_voids.setChecked(True)
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


class TransformToolbar(QWidget):
    """Compact transform controls displayed above the 3D viewport."""

    modeChanged = pyqtSignal(str)
    centerRequested = pyqtSignal()
    autoFitRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("transformToolbar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        tool_label = QLabel("Инструмент:")
        tool_label.setObjectName("fieldLabel")
        layout.addWidget(tool_label)

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
            layout.addWidget(button)
        self.move_mode_btn.setChecked(True)

        self.uniform_scale = QCheckBox("Пропорционально")
        self.uniform_scale.setChecked(True)
        self.uniform_scale.setToolTip("Сохранять пропорции при масштабировании")
        layout.addWidget(self.uniform_scale)

        layout.addStretch(1)

        self.center_btn = QPushButton("Центрировать")
        self.center_btn.setToolTip("Переместить модель в центр XY")
        self.autofit_btn = QPushButton("Авто-фит под колбу")
        self.autofit_btn.setToolTip("Вписать модель в текущий размер колбы")
        layout.addWidget(self.center_btn)
        layout.addWidget(self.autofit_btn)

        self._wire_signals()
        self.set_enabled_state(False)

    def _wire_signals(self) -> None:
        for button, mode in zip(self._mode_buttons, ("move", "rotate", "scale")):
            button.clicked.connect(lambda _checked=False, m=mode: self.modeChanged.emit(m))
        self.center_btn.clicked.connect(self.centerRequested.emit)
        self.autofit_btn.clicked.connect(self.autoFitRequested.emit)

    def is_uniform(self) -> bool:
        return self.uniform_scale.isChecked()

    def set_enabled_state(self, enabled: bool) -> None:
        for widget in (*self._mode_buttons, self.uniform_scale, self.center_btn, self.autofit_btn):
            widget.setEnabled(enabled)


class ObjectPanel(QWidget):
    """Legacy expanded object panel kept for API compatibility."""

    modeChanged = pyqtSignal(str)
    centerRequested = pyqtSignal()
    autoFitRequested = pyqtSignal()
    resetRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

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

        hint = QLabel("Тяни цветные стрелки, кольца или квадратные ручки прямо на модели.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        transform_layout.addWidget(hint)

        self.position_value = QLabel("X 0.00 · Y 0.00 · Z 0.00 мм")
        self.rotation_value = QLabel("X 0.0 · Y 0.0 · Z 0.0°")
        self.size_value = QLabel("X 0.00 · Y 0.00 · Z 0.00 мм")
        for value_label in (self.position_value, self.rotation_value, self.size_value):
            value_label.setObjectName("transformValue")
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        transform_layout.addWidget(QLabel("Позиция"))
        transform_layout.addWidget(self.position_value)
        transform_layout.addWidget(QLabel("Вращение"))
        transform_layout.addWidget(self.rotation_value)
        transform_layout.addWidget(QLabel("Размер"))
        transform_layout.addWidget(self.size_value)

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

    # --- сигналы -------------------------------------------------------------
    def _wire_signals(self) -> None:
        for button, mode in zip(self._mode_buttons, ("move", "rotate", "scale")):
            button.clicked.connect(lambda _checked=False, m=mode: self.modeChanged.emit(m))

        self.center_btn.clicked.connect(self.centerRequested.emit)
        self.autofit_btn.clicked.connect(self.autoFitRequested.emit)
        self.reset_btn.clicked.connect(self.resetRequested.emit)

    def is_uniform(self) -> bool:
        return self.uniform_scale.isChecked()

    # --- состояние -------------------------------------------------------------
    def set_enabled_state(self, enabled: bool) -> None:
        for w in (*self._mode_buttons, self.uniform_scale,
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
        size = node.current_size_mm()
        rot = node.transform.rotation_deg
        pos = node.transform.translation
        self.position_value.setText(
            f"X {pos[0]:+.2f} · Y {pos[1]:+.2f} · Z {pos[2]:+.2f} мм"
        )
        self.rotation_value.setText(
            f"X {rot[0]:+.1f} · Y {rot[1]:+.1f} · Z {rot[2]:+.1f}°"
        )
        self.size_value.setText(
            f"X {size[0]:.2f} · Y {size[1]:.2f} · Z {size[2]:.2f} мм"
        )
