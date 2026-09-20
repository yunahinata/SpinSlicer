"""Combined optical bench and reconstruction workspace.

The Slicer, projector, and simulation areas are intentionally separate top-level
tabs. This widget is the simulation tab: it combines the optical calculation
with the inverse reconstruction while receiving generated frames from
``SlicerTab``.
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

from i18n import tr
from job_controller import JobController
from optical_tab import OpticalSimulationTab
from simulator_tab import SimulatorTab
from ui_panels import ProcessSettingsPanel


class SimulationWorkbenchTab(QWidget):
    """Combine the optical simulation and inverse reconstruction in one tab."""

    progress = pyqtSignal(float, str)
    logMessage = pyqtSignal(str)
    # Kept as a compatibility surface for integrations that used the former
    # workbench. New frame folders arrive through ``set_output_dir``.
    outputGenerated = pyqtSignal(str)

    _STEP_TITLES = ("1. Оптический стенд", "2. Результат")
    _STEP_DESCRIPTIONS = (
        "Подберите параметры колбы, квадратного аквариума и воды; сравните ход лучей и проекцию.",
        "Запустите обратную реконструкцию и проверьте, какую геометрию даст выбранный набор кадров.",
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

        self.optical_tab = OpticalSimulationTab(parent=self)
        self.simulator_tab = SimulatorTab(parent=self, job_controller=self._job_controller)

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
        steps_title = QLabel("Порядок работы")
        steps_title.setObjectName("fieldLabel")
        steps_layout.addWidget(steps_title)

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
        self._page_stack.addWidget(self.optical_tab)
        self._page_stack.addWidget(self.simulator_tab)
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
        for child in (self.optical_tab, self.simulator_tab):
            child.progress.connect(self.progress.emit)
            child.logMessage.connect(self.logMessage.emit)

        self._process_panel.diameterChanged.connect(self.optical_tab.set_vat_diameter)
        self.optical_tab.set_vat_diameter(self._process_panel.vat_diameter_mm())

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
                "Настройте преломление и при необходимости возьмите последний кадр со вкладки «Слайсер»."
            )
            self._next_btn.setVisible(True)
            self._next_btn.setEnabled(self._frames_dir is not None)
        else:
            self._page_status.setText(
                "Здесь видно, какую геометрию даст выбранный набор проекций; порог поверхности меняется быстро."
            )
            self._next_btn.setVisible(False)

    def _go_next(self) -> None:
        if self._frames_dir is None:
            QMessageBox.information(
                self,
                tr("Внимание"),
                "Сначала сгенерируйте проекции на вкладке «Слайсер» или загрузите кадр вручную.",
            )
            return
        self._set_step(self._current_step + 1)

    def set_model_path(self, path: str) -> None:
        """Update the simulation context when a new STL is loaded in the slicer."""

        self.set_output_dir("")

    def set_output_dir(self, path: str) -> None:
        """Receive the latest projection folder from the standalone slicer tab."""

        normalized = os.path.abspath(path) if path else ""
        self.optical_tab.set_output_dir(normalized)
        self.simulator_tab.set_output_dir(normalized)

        self._frames_dir = normalized or None
        self._set_step(0)
