"""
ui_panels.py
============
Виджеты настроек и управления моделью.

  ProcessSettingsPanel — настройки принтера и процесса, вынесенные в
    отдельную вкладку приложения, включая опциональную установку VAMToolbox.
  TransformToolbar — компактное управление gizmo над 3D-вьюпортом.
  ScaleControls — компактная Fusion-style панель процентов и размеров.
  ObjectPanel — прежняя расширенная панель объекта, оставленная для обратной
    совместимости с внешними импортами.

Виджеты ничего не знают про PyVista напрямую — связь с ModelNode и сценой
делает SlicerTab.
"""
from __future__ import annotations

import subprocess
from typing import Optional

import numpy as np
from PyQt6.QtCore import QProcess, QSettings, QSignalBlocker, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from constants import (
    APP_ORG,
    APP_TITLE,
    DEFAULT_DIAMETER_MM,
    DEFAULT_GRID_RESOLUTION,
    DEFAULT_NUM_FRAMES,
    DEFAULT_OUTPUT_RESOLUTION,
)
from i18n import tr
from model_node import ModelNode
from vam_backend import (
    DEFAULT_VAM_ENV_NAME,
    build_vam_install_command,
    build_vam_probe_command,
    find_conda_executable,
    is_vamtoolbox_available,
)
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

        self._settings = QSettings(APP_ORG, APP_TITLE)
        self._vam_process: Optional[QProcess] = None
        self._vam_process_stage = ""
        self._vam_conda_executable = str(
            self._settings.value("vam/conda_executable", "") or ""
        )
        if not self._vam_conda_executable:
            self._vam_conda_executable = find_conda_executable() or ""
        self._vam_environment_name = str(
            self._settings.value("vam/environment_name", DEFAULT_VAM_ENV_NAME)
            or DEFAULT_VAM_ENV_NAME
        )
        ready_value = self._settings.value("vam/ready", False)
        self._vam_ready = str(ready_value).lower() in {"1", "true", "yes"}

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

        # --- Optional VAMToolbox runtime ----------------------------------
        vam_box = QGroupBox("VAMToolbox — альтернативный движок")
        vam_layout = QVBoxLayout(vam_box)
        vam_description = QLabel(
            "SpinSlicer Radon — встроенный быстрый и воспроизводимый движок "
            "для локальной подготовки кадров. VAMToolbox CAL — внешний "
            "оптимизатор дозы, который может лучше подавлять переэкспонирование "
            "и артефакты, но работает медленнее и требует отдельного Windows "
            "окружения Conda. VAMToolbox не входит в SpinSlicer и ставится "
            "только по запросу."
        )
        vam_description.setObjectName("hintLabel")
        vam_description.setWordWrap(True)
        vam_layout.addWidget(vam_description)

        self.vam_status = QLabel()
        self.vam_status.setObjectName("hintLabel")
        self.vam_status.setWordWrap(True)
        vam_layout.addWidget(self.vam_status)

        self.vam_install_btn = QPushButton("Скачать и установить VAMToolbox")
        self.vam_install_btn.setToolTip(
            "Создать отдельное окружение Conda и установить VAMToolbox CAL"
        )
        self.vam_install_btn.clicked.connect(self._start_vam_install)
        vam_layout.addWidget(self.vam_install_btn)

        vam_docs = QLabel(
            '<a href="https://vamtoolbox.readthedocs.io/en/latest/_docs/gettingstarted.html">'
            "Инструкция VAMToolbox и ограничения платформы</a>"
        )
        vam_docs.setOpenExternalLinks(True)
        vam_docs.setObjectName("hintLabel")
        vam_layout.addWidget(vam_docs)
        layout.addWidget(vam_box)

        layout.addStretch(1)
        content.setMaximumWidth(720)
        scroll = _scroll_wrap(content)
        scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        root.addWidget(scroll)

        self.diameter.valueChanged.connect(self.diameterChanged.emit)
        self.diameter.valueChanged.connect(lambda _v: self.settingsChanged.emit())
        self.fill_holes.toggled.connect(lambda _v: self.settingsChanged.emit())
        self.preserve_internal_voids.toggled.connect(lambda _v: self.settingsChanged.emit())
        self.projection_backend.currentIndexChanged.connect(lambda _v: self.settingsChanged.emit())
        self.optimizer_iterations.valueChanged.connect(lambda _v: self.settingsChanged.emit())

        self._refresh_vam_status()

    def vat_diameter_mm(self) -> float:
        return self.diameter.value()

    def projection_backend_name(self) -> str:
        return str(self.projection_backend.currentData())

    def optimizer_iterations_value(self) -> int:
        return self.optimizer_iterations.value_int()

    def vam_runtime_config(self) -> tuple[str, str]:
        """Return the configured isolated runtime, if it has been verified."""

        if is_vamtoolbox_available():
            return "", ""
        if self._vam_ready and self._vam_conda_executable:
            return self._vam_conda_executable, self._vam_environment_name
        return "", ""

    def shutdown(self) -> None:
        """Stop an in-progress optional dependency installation."""

        process = self._vam_process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            return
        process.terminate()
        if not process.waitForFinished(1000):
            process.kill()
            process.waitForFinished(1000)

    def _refresh_vam_status(self) -> None:
        if is_vamtoolbox_available():
            self._vam_ready = True
            self.vam_status.setText("VAMToolbox доступен в текущем Python-окружении.")
            self.vam_install_btn.setText("VAMToolbox уже доступен")
            self.vam_install_btn.setEnabled(False)
            return

        if self._vam_ready and self._vam_conda_executable:
            self.vam_status.setText(
                f"Проверенное окружение: {self._vam_environment_name}. "
                "Его можно выбрать как движок проекций."
            )
            self.vam_install_btn.setText("Проверить окружение VAMToolbox")
            self.vam_install_btn.setEnabled(False)
            QTimer.singleShot(0, self._start_vam_probe)
            return

        if self._vam_conda_executable:
            self.vam_status.setText(
                "Conda найдена. Установщик создаст отдельное окружение "
                f"«{self._vam_environment_name}» и не изменит внутренний Radon."
            )
            self.vam_install_btn.setEnabled(True)
        else:
            self.vam_status.setText(
                "Conda не найдена. Установите Miniconda или Anaconda, "
                "перезапустите SpinSlicer и повторите установку."
            )
            self.vam_install_btn.setEnabled(False)

    def _qprocess_argv(self, command: list[str]) -> tuple[str, list[str]]:
        executable = command[0]
        if executable.lower().endswith((".bat", ".cmd")):
            return "cmd.exe", ["/d", "/s", "/c", subprocess.list2cmdline(command)]
        return executable, command[1:]

    def _run_vam_process(self, command: list[str], stage: str) -> None:
        if self._vam_process is not None:
            return
        program, arguments = self._qprocess_argv(command)
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_vam_output)
        process.finished.connect(self._on_vam_process_finished)
        self._vam_process = process
        self._vam_process_stage = stage
        process.start(program, arguments)

    def _on_vam_output(self) -> None:
        process = self._vam_process
        if process is None:
            return
        raw = process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if lines:
            self.vam_status.setText(lines[-1][-240:])

    def _on_vam_process_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        process = self._vam_process
        stage = self._vam_process_stage
        self._vam_process = None
        self._vam_process_stage = ""
        if process is not None:
            process.deleteLater()

        if stage == "install":
            if exit_code != 0:
                self._vam_ready = False
                self.vam_status.setText(
                    "Установка VAMToolbox завершилась с ошибкой. "
                    "Проверьте вывод команды и каналы Conda."
                )
                self.vam_install_btn.setEnabled(bool(self._vam_conda_executable))
                return
            self.vam_status.setText("Проверяю установленный VAMToolbox...")
            self._start_vam_probe()
            return

        if stage == "probe":
            self._vam_ready = exit_code == 0
            self._settings.setValue("vam/ready", self._vam_ready)
            self._settings.setValue("vam/conda_executable", self._vam_conda_executable)
            self._settings.setValue("vam/environment_name", self._vam_environment_name)
            if self._vam_ready:
                self.vam_status.setText(
                    f"VAMToolbox установлен в окружении «{self._vam_environment_name}»."
                )
                self.vam_install_btn.setText("VAMToolbox установлен")
                self.vam_install_btn.setEnabled(False)
                self.settingsChanged.emit()
            else:
                self.vam_status.setText(
                    "Не удалось импортировать VAMToolbox из окружения. "
                    "Переустановите его или используйте внутренний Radon."
                )
                self.vam_install_btn.setEnabled(bool(self._vam_conda_executable))

    def _start_vam_install(self) -> None:
        if self._vam_process is not None:
            return
        self._vam_conda_executable = find_conda_executable() or self._vam_conda_executable
        if not self._vam_conda_executable:
            QMessageBox.information(
                self,
                tr("VAMToolbox — альтернативный движок"),
                "Установите Miniconda или Anaconda с официального сайта, "
                "добавьте Conda в PATH и перезапустите приложение.",
            )
            return

        self._vam_ready = False
        self.vam_install_btn.setEnabled(False)
        self.vam_status.setText(
            f"Устанавливаю VAMToolbox в окружение «{self._vam_environment_name}». "
            "Это может занять несколько минут..."
        )
        self._run_vam_process(
            build_vam_install_command(
                self._vam_conda_executable,
                self._vam_environment_name,
            ),
            "install",
        )

    def _start_vam_probe(self) -> None:
        if self._vam_process is not None or not self._vam_conda_executable:
            return
        self._run_vam_process(
            build_vam_probe_command(
                self._vam_conda_executable,
                self._vam_environment_name,
            ),
            "probe",
        )


class ScaleControls(QWidget):
    """Compact Fusion-style percentage/size controls for the scale gizmo."""

    scalePercentChanged = pyqtSignal(int, float)
    sizeChanged = pyqtSignal(int, float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("scaleControls")

        layout = QGridLayout(self)
        layout.setContentsMargins(10, 4, 10, 8)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)

        layout.addWidget(QLabel(""), 0, 0)
        self.axis_labels: list[QLabel] = []
        for column, (axis_name, color) in enumerate(
            (("X", "#ef6a63"), ("Y", "#62c370"), ("Z", "#5d91ef")),
            start=1,
        ):
            label = QLabel(axis_name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: {color}; font-weight: 700;")
            self.axis_labels.append(label)
            layout.addWidget(label, 0, column)

        layout.addWidget(QLabel("Масштаб"), 1, 0)
        layout.addWidget(QLabel("Размер"), 2, 0)
        self.percent_fields: list[QDoubleSpinBox] = []
        self.size_fields: list[QDoubleSpinBox] = []
        for axis in range(3):
            percent = self._make_field(0.01, 10000.0, 100.0, 2, " %")
            size = self._make_field(0.01, 100000.0, 1.0, 2, " mm")
            self.percent_fields.append(percent)
            self.size_fields.append(size)
            layout.addWidget(percent, 1, axis + 1)
            layout.addWidget(size, 2, axis + 1)
            percent.valueChanged.connect(
                lambda value, index=axis: self.scalePercentChanged.emit(index, value)
            )
            size.valueChanged.connect(
                lambda value, index=axis: self.sizeChanged.emit(index, value)
            )

        self.uniform_scale = QCheckBox("Равномерное масштабирование")
        self.uniform_scale.setChecked(True)
        self.uniform_scale.setToolTip("Сохранять соотношение сторон модели")
        layout.addWidget(self.uniform_scale, 3, 0, 1, 4)

    @staticmethod
    def _make_field(
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
        suffix: str,
    ) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        field.setRange(minimum, maximum)
        field.setDecimals(decimals)
        field.setSingleStep(1.0)
        field.setSuffix(suffix)
        field.setFixedWidth(96)
        field.setValue(value)
        return field

    def sync_from_model(self, node: ModelNode) -> None:
        """Update fields without feeding the values back into the model."""

        scale = np.asarray(node.transform.scale, dtype=np.float64)
        size = np.asarray(node.current_size_mm(), dtype=np.float64)
        for axis in range(3):
            with QSignalBlocker(self.percent_fields[axis]):
                self.percent_fields[axis].setValue(max(float(scale[axis] * 100.0), 0.01))
            with QSignalBlocker(self.size_fields[axis]):
                self.size_fields[axis].setValue(max(float(size[axis]), 0.01))

    def set_enabled_state(self, enabled: bool) -> None:
        self.setEnabled(enabled)


class TransformToolbar(QWidget):
    """Compact transform controls displayed above the 3D viewport."""

    modeChanged = pyqtSignal(str)
    centerRequested = pyqtSignal()
    autoFitRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("transformToolbar")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(6)
        layout.addLayout(tool_row)

        tool_label = QLabel("Инструмент:")
        tool_label.setObjectName("fieldLabel")
        tool_row.addWidget(tool_label)

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
            tool_row.addWidget(button)
        self.move_mode_btn.setChecked(True)

        self.uniform_scale = QCheckBox("Пропорционально")
        self.uniform_scale.setChecked(True)
        self.uniform_scale.setToolTip("Сохранять пропорции при масштабировании")
        tool_row.addWidget(self.uniform_scale)

        tool_row.addStretch(1)

        self.center_btn = QPushButton("Центрировать")
        self.center_btn.setToolTip("Переместить модель в центр XY")
        self.autofit_btn = QPushButton("Авто-фит под колбу")
        self.autofit_btn.setToolTip("Вписать модель в текущий размер колбы")
        tool_row.addWidget(self.center_btn)
        tool_row.addWidget(self.autofit_btn)

        self.scale_controls = ScaleControls(self)
        self.scale_controls.setVisible(False)
        layout.addWidget(self.scale_controls)

        self._wire_signals()
        self.set_enabled_state(False)

    def _wire_signals(self) -> None:
        for button, mode in zip(self._mode_buttons, ("move", "rotate", "scale")):
            button.clicked.connect(lambda _checked=False, m=mode: self._on_mode_clicked(m))
        self.center_btn.clicked.connect(self.centerRequested.emit)
        self.autofit_btn.clicked.connect(self.autoFitRequested.emit)
        self.scale_controls.uniform_scale.toggled.connect(self._sync_uniform_checkboxes)
        self.uniform_scale.toggled.connect(self._sync_detail_uniform_checkbox)

    def _on_mode_clicked(self, mode: str) -> None:
        self.scale_controls.setVisible(mode == "scale")
        self.modeChanged.emit(mode)

    def _sync_uniform_checkboxes(self, checked: bool) -> None:
        with QSignalBlocker(self.uniform_scale):
            self.uniform_scale.setChecked(checked)

    def _sync_detail_uniform_checkbox(self, checked: bool) -> None:
        with QSignalBlocker(self.scale_controls.uniform_scale):
            self.scale_controls.uniform_scale.setChecked(checked)

    def is_uniform(self) -> bool:
        return self.uniform_scale.isChecked()

    def sync_from_model(self, node: ModelNode) -> None:
        self.scale_controls.sync_from_model(node)

    def set_enabled_state(self, enabled: bool) -> None:
        for widget in (*self._mode_buttons, self.uniform_scale, self.center_btn, self.autofit_btn):
            widget.setEnabled(enabled)
        self.scale_controls.set_enabled_state(enabled)


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
