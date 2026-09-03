"""
simulator_tab.py
================
Вкладка "Симулятор": обратная реконструкция (inverse Radon / FBP) кадров
из output_frames в приближённую 3D-геометрию — визуальный предпросмотр
того, как деталь запечётся в жидкости. Использует тот же Viewport3D,
что и вкладка "Слайсер", ради единого визуального языка приложения.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from job_controller import JobController
from reconstruction import ReconstructionResult, volume_to_mesh
from viewport import Viewport3D
from widgets import LabeledSlider
from workers import ReconstructionWorker

DEFAULT_THRESHOLD_PCT = 50.0


class SimulatorTab(QWidget):
    progress = pyqtSignal(float, str)
    logMessage = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        job_controller: Optional[JobController] = None,
    ):
        super().__init__(parent)

        self._job_controller = job_controller or JobController()
        self._output_dir: Optional[str] = None
        self._worker: Optional[ReconstructionWorker] = None
        self._result: Optional[ReconstructionResult] = None

        self._threshold_timer = QTimer(self)
        self._threshold_timer.setSingleShot(True)
        self._threshold_timer.timeout.connect(self._apply_threshold)

        self._build_ui()

    # =======================================================================
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        dir_row = QHBoxLayout()
        self.dir_label = QLabel("Папка кадров не выбрана — сначала сгенерируйте проекции на вкладке «Слайсер».")
        self.dir_label.setObjectName("hintLabel")
        dir_row.addWidget(self.dir_label, 1)

        self.browse_btn = QPushButton("Обзор папки...")
        self.browse_btn.setToolTip("Указать папку с кадрами frame_XXXX.png вручную")
        self.browse_btn.clicked.connect(self._on_browse_clicked)
        dir_row.addWidget(self.browse_btn)
        root.addLayout(dir_row)

        self.viewport = Viewport3D()
        root.addWidget(self.viewport, 1)

        buttons = QHBoxLayout()
        self.simulate_btn = QPushButton("🔬 Симулировать результат")
        self.simulate_btn.setObjectName("generateButton")
        self.simulate_btn.setToolTip("Обратная Radon-реконструкция геометрии по кадрам")
        self.simulate_btn.clicked.connect(self._on_simulate_clicked)
        buttons.addWidget(self.simulate_btn)
        root.addLayout(buttons)

        self.threshold_slider = LabeledSlider(
            "Порог визуализации (изоповерхность)", 1.0, 99.0, DEFAULT_THRESHOLD_PCT,
            decimals=0, step=1.0, suffix="%",
        )
        self.threshold_slider.setEnabled(False)
        self.threshold_slider.setToolTip("Пересчитывает только поверхность — без повторной реконструкции")
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        root.addWidget(self.threshold_slider)

        hint = QLabel(
            "Реконструкция через обратное Radon-преобразование (FBP) — визуальный "
            "предпросмотр ожидаемой геометрии, не метрологическая симуляция полимеризации."
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

    # --- общая папка вывода ---------------------------------------------------------
    def set_output_dir(self, path: str) -> None:
        if path != self._output_dir:
            self._result = None
            self.threshold_slider.setEnabled(False)
            self.viewport.clear_model()
        self._output_dir = path
        self.dir_label.setText(f"Папка кадров: {path}")

    def _on_browse_clicked(self) -> None:
        start_dir = self._output_dir or os.getcwd()
        path = QFileDialog.getExistingDirectory(self, "Выбрать папку с кадрами", start_dir)
        if path:
            self.set_output_dir(path)

    # --- реконструкция ----------------------------------------------------------------
    def _on_simulate_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self.simulate_btn.setEnabled(False)
            self.simulate_btn.setText("Остановка...")
            return

        if not self._output_dir or not os.path.isdir(self._output_dir):
            QMessageBox.warning(self, "Внимание", "Сначала укажите папку с кадрами (output_frames).")
            return

        self.viewport.clear_model()
        self.threshold_slider.setEnabled(False)
        self.progress.emit(0.0, "Запуск обратной реконструкции...")
        self.simulate_btn.setText("⏹ Отменить")

        self._worker = ReconstructionWorker(self._output_dir, self)
        self._job_controller.register(self._worker)
        self._worker.progress.connect(lambda f, m: self.progress.emit(f, m))
        self._worker.finished_ok.connect(self._on_reconstruction_done)
        self._worker.failed.connect(self._on_reconstruction_failed)
        self._worker.cancelled.connect(self._on_reconstruction_cancelled)
        self._worker.finished.connect(self._on_worker_thread_finished)
        self._worker.start()

    def _on_reconstruction_done(self, result: ReconstructionResult) -> None:
        self._result = result
        self.viewport.update_vat(result.diameter_mm)
        self.threshold_slider.setEnabled(True)
        self.progress.emit(1.0, "Реконструкция завершена.")
        self.logMessage.emit(
            f"Объём восстановлен: сетка {result.grid_res}×{result.grid_res}×{result.nz}."
        )
        self._apply_threshold()

    def _on_reconstruction_failed(self, msg: str) -> None:
        self.progress.emit(0.0, "Ошибка реконструкции.")
        QMessageBox.critical(self, "Ошибка", msg)

    def _on_reconstruction_cancelled(self) -> None:
        self.progress.emit(0.0, "Реконструкция отменена.")

    def _on_worker_thread_finished(self) -> None:
        self.simulate_btn.setEnabled(True)
        self.simulate_btn.setText("🔬 Симулировать результат")

    # --- порог визуализации (дёшево — без повторного iradon) -------------------------
    def _on_threshold_changed(self, _v: float) -> None:
        self._threshold_timer.start(150)

    def _apply_threshold(self) -> None:
        if self._result is None:
            return
        fraction = self.threshold_slider.value() / 100.0
        mesh = volume_to_mesh(self._result, fraction)

        self.viewport.clear_model()
        if mesh is None:
            self.progress.emit(1.0, "При этом пороге геометрия отсутствует — сдвиньте ползунок.")
            return

        self.viewport.set_model(mesh)
        self.viewport.update_model_transform(np.eye(4))
        self.viewport.reset_camera()

    def closeEvent(self, event) -> None:  # noqa: N802 (имя метода задано Qt)
        if self._job_controller.shutdown():
            event.accept()
        else:
            event.ignore()
