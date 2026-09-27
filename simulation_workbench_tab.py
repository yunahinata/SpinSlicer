"""One workspace for the light source, optical bench, resin and reconstruction.

The Slicer, projector, and simulation areas are intentionally separate top-level
tabs. This widget receives generated frames from ``SlicerTab`` and keeps all
simulation tools under one top-level tab.
"""
from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from development_paths_tab import DevelopmentPathsTab
from i18n import tr
from job_controller import JobController
from light_source_simulation import LightBudgetInput, evaluate_light_budget
from light_source_tab import LightSourceTab
from optical_tab import OpticalSimulationTab
from resin_finder_dialog import ResinFinderDialog
from resin_lab_tab import ResinLabTab
from simulator_tab import SimulatorTab
from ui_panels import ProcessSettingsPanel


class SimulationWorkbenchTab(QWidget):
    """Keep source, optical, reconstruction and resin simulations together."""

    progress = pyqtSignal(float, str)
    logMessage = pyqtSignal(str)
    # Kept as a compatibility surface for integrations that used the former
    # workbench. New frame folders arrive through ``set_output_dir``.
    outputGenerated = pyqtSignal(str)
    slicerRequested = pyqtSignal()

    _STEP_TITLES = (
        "Четыре пути", "Лазер", "Оптика", "Проекции", "Смола",
    )
    _STEP_DESCRIPTIONS = (
        "Начните здесь. Посмотрите вывод по четырём идеям; вводить параметры не нужно.",
        "Паспортный пример уже заполнен. Ручные измерения находятся в «Подробно».",
        "Можно сразу проверить лучи и геометрию с водой. Настройка размеров необязательна.",
        "Посмотрите виртуальные проекции и реконструкцию без расходного материала.",
        "Лаборатория составов нужна, когда появятся данные о конкретных компонентах.",
    )

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        job_controller: Optional[JobController] = None,
        process_panel: Optional[ProcessSettingsPanel] = None,
    ) -> None:
        super().__init__(parent)

        self._job_controller = job_controller or JobController()
        self._process_panel = (
            process_panel if process_panel is not None else ProcessSettingsPanel(self)
        )
        self._current_step = 0
        self._frames_dir: Optional[str] = None
        self._simulated_frames_dir: Optional[str] = None

        self.paths_tab = DevelopmentPathsTab(parent=self, job_controller=self._job_controller)
        self.light_source_tab = LightSourceTab(parent=self)
        self.optical_tab = OpticalSimulationTab(parent=self)
        self.simulator_tab = SimulatorTab(parent=self, job_controller=self._job_controller)
        self.resin_lab_tab = ResinLabTab(parent=self)

        self._build_ui()
        self._wire_signals()
        self._set_step(0)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(8)

        steps = QFrame()
        steps.setObjectName("workflowSteps")
        steps_layout = QVBoxLayout(steps)
        steps_layout.setContentsMargins(10, 8, 10, 8)
        steps_layout.setSpacing(6)
        title_row = QHBoxLayout()
        steps_title = QLabel("Инструменты симуляции · открывайте нужные")
        steps_title.setObjectName("fieldLabel")
        title_row.addWidget(steps_title, 1)
        self.hypotheses_button = QPushButton("Сравнить 4 гипотезы")
        self.hypotheses_button.clicked.connect(self._show_hypotheses)
        title_row.addWidget(self.hypotheses_button)
        self.resin_search_button = QPushButton("Найти смолу под лазер")
        self.resin_search_button.setToolTip(
            "Автоматически ранжировать известные материалы для лазера 450 нм и перебрать световые сценарии."
        )
        self.resin_search_button.clicked.connect(self._show_resin_search)
        title_row.addWidget(self.resin_search_button)
        steps_layout.addLayout(title_row)

        step_row = QHBoxLayout()
        step_row.setSpacing(6)
        self._step_buttons: list[QPushButton] = []
        self._step_group = QButtonGroup(self)
        self._step_group.setExclusive(True)
        for index, (title_text, description) in enumerate(
            zip(self._STEP_TITLES, self._STEP_DESCRIPTIONS),
        ):
            button = QPushButton(title_text)
            button.setCheckable(True)
            button.setObjectName("workflowStep")
            button.setToolTip(description)
            self._step_group.addButton(button, index)
            button.clicked.connect(lambda _checked=False, step=index: self._set_step(step))
            self._step_buttons.append(button)
            step_row.addWidget(button, 1)
        steps_layout.addLayout(step_row)
        self._step_description = QLabel()
        self._step_description.setObjectName("hintLabel")
        self._step_description.setWordWrap(True)
        steps_layout.addWidget(self._step_description)
        root.addWidget(steps)

        self._page_stack = QStackedWidget()
        self._page_stack.addWidget(self.paths_tab)
        self._page_stack.addWidget(self.light_source_tab)
        self._page_stack.addWidget(self.optical_tab)
        self._page_stack.addWidget(self.simulator_tab)
        self._page_stack.addWidget(self.resin_lab_tab)
        root.addWidget(self._page_stack, 1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self._page_status = QLabel()
        self._page_status.setObjectName("hintLabel")
        self._page_status.setWordWrap(True)
        footer.addWidget(self._page_status, 1)

        self._back_btn = QPushButton("← Назад")
        self._back_btn.clicked.connect(lambda: self._set_step(self._current_step - 1))
        footer.addWidget(self._back_btn)

        self._next_btn = QPushButton("К результату →")
        self._next_btn.setObjectName("generateButton")
        self._next_btn.clicked.connect(self._go_next)
        footer.addWidget(self._next_btn)
        root.addLayout(footer)

    def _wire_signals(self) -> None:
        for child in (self.optical_tab, self.simulator_tab, self.resin_lab_tab):
            child.progress.connect(self.progress.emit)
            child.logMessage.connect(self.logMessage.emit)

        self.light_source_tab.scenarioSelected.connect(self._on_source_scenario_selected)
        self.light_source_tab.inputsChanged.connect(self.paths_tab.set_laser_source)
        self.paths_tab.set_laser_source(self.light_source_tab.inputs())
        self.paths_tab.sourceRequested.connect(lambda: self._set_step(1))
        self.paths_tab.resinRequested.connect(lambda: self._set_step(4))
        self.paths_tab.slicerRequested.connect(self.slicerRequested.emit)
        self._process_panel.diameterChanged.connect(self.optical_tab.set_vat_diameter)
        self.optical_tab.simulationFramesReady.connect(self._on_simulation_frames_ready)
        self.optical_tab.set_vat_diameter(self._process_panel.vat_diameter_mm())

    def _show_hypotheses(self) -> None:
        """Open the comparison from any simulation subpage."""

        self._set_step(0)
        self.paths_tab.show_recommendations()

    def _show_resin_search(self) -> None:
        """Run the no-input material and exposure search from any simulation page."""

        ResinFinderDialog(source=self.light_source_tab.inputs(), parent=self).exec()

    def _set_step(self, step: int) -> None:
        if step < 0 or step >= self._page_stack.count():
            return
        self._current_step = step
        self._page_stack.setCurrentIndex(step)
        self._step_buttons[step].setChecked(True)
        self._step_description.setText(self._STEP_DESCRIPTIONS[step])
        self._back_btn.setEnabled(step > 0)

        if step == 0:
            self._page_status.setText(
                "Выберите идею. Числа вводить не нужно; пока данных нет, вывод остаётся предварительным."
            )
            self._next_btn.setVisible(True)
            self._next_btn.setEnabled(True)
            self._next_btn.setText("Посмотреть расчёт лазера →")
        elif step == 1:
            self._page_status.setText(
                "Поля ниже заполнены паспортным примером. Измерения можно добавить позже в «Подробно»."
            )
            self._next_btn.setVisible(True)
            self._next_btn.setEnabled(True)
            self._next_btn.setText("Проверить виртуальную оптику →")
        elif step == 2:
            self._page_status.setText(
                "Проверьте преломление и форму поля; после расчёта кадры перейдут в реконструкцию."
            )
            self._next_btn.setVisible(True)
            self._next_btn.setEnabled(self._simulated_frames_dir is not None)
            self._next_btn.setText("Посмотреть проекции →")
        elif step == 3:
            self._page_status.setText(
                "Здесь показана геометрия по симулированным кадрам; реконструкция запускается автоматически."
            )
            self._next_btn.setVisible(True)
            self._next_btn.setEnabled(True)
            self._next_btn.setText("Открыть лабораторию составов →")
        else:
            self._page_status.setText(
                "Рецептуры и измерения сохраняются при переходе между разделами симуляции."
            )
            self._next_btn.setVisible(False)

    def _go_next(self) -> None:
        if self._current_step == 0:
            self._set_step(1)
            return
        if self._current_step == 1:
            try:
                self._on_source_scenario_selected(self.light_source_tab.inputs())
            except ValueError as exc:
                QMessageBox.warning(self, "Проверьте источник", str(exc))
            return
        if self._current_step == 2 and self._simulated_frames_dir is None:
            QMessageBox.information(
                self,
                tr("Внимание"),
                "Сначала запустите оптическую симуляцию — модель строится по её кадрам.",
            )
            return
        self._set_step(self._current_step + 1)

    def _on_source_scenario_selected(self, inputs: LightBudgetInput) -> None:
        result = evaluate_light_budget(inputs)
        if inputs.measured_irradiance_mw_cm2 is not None:
            if inputs.measured_irradiance_mw_cm2 > result.source_irradiance_mw_cm2[1] * 1.05:
                QMessageBox.warning(
                    self,
                    "Проверьте измерение",
                    "Интенсивность на колбе выше предела по заданной мощности и площади. "
                    "Проверьте размеры поля, паспортную мощность и калибровку датчика.",
                )
                return
            representative_power = inputs.power_max_w
            transmission = min(
                100.0,
                max(0.0001, 100.0 * inputs.measured_irradiance_mw_cm2
                    / result.source_irradiance_mw_cm2[1]),
            )
        else:
            representative_power = inputs.power_min_w
            transmission = inputs.transmission_pct or 100.0
        self.optical_tab.apply_source_assumptions(
            wavelength_nm=inputs.wavelength_nm,
            power_w=representative_power,
            area_cm2=result.area_cm2,
            transmission_pct=transmission,
            measured=inputs.measured_irradiance_mw_cm2 is not None,
        )
        representative_intensity = (
            sum(result.plane_irradiance_mw_cm2) / 2.0
            if result.plane_irradiance_mw_cm2 is not None else None
        )
        self.resin_lab_tab.set_light_source_assumption(
            wavelength_nm=inputs.wavelength_nm,
            intensity_mw_cm2=representative_intensity,
            measured=inputs.measured_irradiance_mw_cm2 is not None,
        )
        self._set_step(2)

    def _on_simulation_frames_ready(self, path: str) -> None:
        """Switch reconstruction to the images produced by the optical bench."""

        self._simulated_frames_dir = os.path.abspath(path)
        self.simulator_tab.set_output_dir(self._simulated_frames_dir)
        self._set_step(3)
        self.simulator_tab.start_reconstruction()

    def set_model_path(self, path: str) -> None:
        """Update the simulation context when a new STL is loaded in the slicer."""

        self.set_output_dir("")

    def set_output_dir(self, path: str) -> None:
        """Receive the latest projection folder from the standalone slicer tab."""

        normalized = os.path.abspath(path) if path else ""
        self.optical_tab.set_output_dir(normalized)
        self.simulator_tab.set_output_dir(normalized)

        self._frames_dir = normalized or None
        self._simulated_frames_dir = None
        self.paths_tab.set_frames_dir(normalized)
        self._set_step(0)
