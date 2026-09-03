"""

Главное окно — тонкая оболочка над тремя вкладками:

  1. "🧊 Слайсер"         — генерация проекций (slicer_tab.SlicerTab).
  2. "🎬 Проектор (Видео)" — сборка/проигрывание/экспорт видео из кадров
                              (video_tab.ProjectorTab).
  3. "🔬 Симулятор"        — обратная реконструкция геометрии по кадрам
                              (simulator_tab.SimulatorTab).

Статус-бар, прогресс-бар и лог — ОБЩИЕ для всего приложения и живут
здесь, а не в каждой вкладке: так пользователь видит происходящее вне
зависимости от того, какая вкладка сейчас активна (например, генерация
на "Слайсере" продолжает идти, пока он смотрит "Проектор").


"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from constants import (
    ACCENT_GREEN,
    ACCENT_GREEN_HOVER,
    APP_TITLE,
    BUTTON_RADIUS,
    PANEL_RADIUS,
)
from job_controller import JobController
from simulator_tab import SimulatorTab
from slicer_tab import SlicerTab
from video_tab import ProjectorTab

EXTRA_QSS = f"""
QGroupBox {{
    border: 1px solid #33384a;
    border-radius: {PANEL_RADIUS}px;
    margin-top: 14px;
    padding-top: 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}}
QPushButton {{
    border-radius: {BUTTON_RADIUS}px;
    padding: 6px 14px;
}}
QPushButton#generateButton {{
    background-color: {ACCENT_GREEN};
    color: white;
    font-weight: 600;
}}
QPushButton#generateButton:hover {{
    background-color: {ACCENT_GREEN_HOVER};
}}
QLabel#panelTitle {{
    font-size: 16px;
    font-weight: 700;
    padding-bottom: 4px;
}}
QLabel#hintLabel {{
    color: #9a9a9a;
    font-size: 11px;
}}
QLabel#fieldLabel {{
    font-size: 12px;
}}
QLabel#videoPreview {{
    background-color: #101114;
    border: 1px solid #33384a;
    border-radius: {PANEL_RADIUS}px;
    color: #6f7480;
    font-size: 13px;
}}
QPlainTextEdit#logPanel {{
    border: 1px solid #33384a;
    border-radius: {PANEL_RADIUS}px;
    font-family: "Consolas", "Menlo", monospace;
    font-size: 11px;
}}

/* Крупные, читаемые вкладки — но без перегруза: только размер шрифта,
   отступы и лёгкое скругление верхних углов активной вкладки. */
QTabWidget::pane {{
    border: 1px solid #33384a;
    border-radius: {PANEL_RADIUS}px;
    top: -1px;
}}
QTabBar::tab {{
    font-size: 13px;
    font-weight: 600;
    padding: 10px 22px;
    margin-right: 4px;
    border-top-left-radius: {BUTTON_RADIUS}px;
    border-top-right-radius: {BUTTON_RADIUS}px;
}}
QTabBar::tab:selected {{
    background-color: {ACCENT_GREEN};
    color: white;
}}
"""


class CALSlicerMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self._job_controller = JobController()

        self._build_central()
        self._build_status_bar()
        self._wire_signals()

        self._log("Готово к работе. Загрузите STL-модель на вкладке «Слайсер», чтобы начать.")

    # =======================================================================
    # Построение интерфейса
    # =======================================================================
    def _build_central(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 10)
        root.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.slicer_tab = SlicerTab(job_controller=self._job_controller)
        self.projector_tab = ProjectorTab(job_controller=self._job_controller)
        self.simulator_tab = SimulatorTab(job_controller=self._job_controller)

        self.tabs.addTab(self.slicer_tab, "🧊 Слайсер")
        self.tabs.addTab(self.projector_tab, "🎬 Проектор (Видео)")
        self.tabs.addTab(self.simulator_tab, "🔬 Симулятор")

        self.tabs.setTabToolTip(0, "Настройка модели и генерация проекций")
        self.tabs.setTabToolTip(1, "Проигрывание и экспорт готовых кадров в MP4")
        self.tabs.setTabToolTip(2, "Обратная реконструкция геометрии по кадрам")

        root.addWidget(self.tabs, 1)

        self._log_panel = QPlainTextEdit()
        self._log_panel.setObjectName("logPanel")
        self._log_panel.setReadOnly(True)
        self._log_panel.setMaximumHeight(120)
        root.addWidget(self._log_panel)

        self.setCentralWidget(central)

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)

        self._status_label = QLabel("Готово к работе.")
        bar.addWidget(self._status_label, 1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedWidth(260)
        bar.addPermanentWidget(self._progress_bar)

    def _wire_signals(self) -> None:
        # Прогресс и лог у всех трёх вкладок стекаются в общий статус-бар/лог.
        for tab in (self.slicer_tab, self.projector_tab, self.simulator_tab):
            tab.progress.connect(self._set_progress)
            tab.logMessage.connect(self._log)

        # Как только "Слайсер" досчитал кадры — "Проектор" и "Симулятор"
        # сразу узнают, где их искать, без ручного выбора папки.
        self.slicer_tab.outputGenerated.connect(self.projector_tab.set_output_dir)
        self.slicer_tab.outputGenerated.connect(self.simulator_tab.set_output_dir)
        self.slicer_tab.outputGenerated.connect(self._on_output_generated)

    def _on_output_generated(self, out_dir: str) -> None:
        # Мягкая подсказка: переключаем пользователя на следующий логичный
        # шаг, не мешая — если он уже сам открыл другую вкладку, не трогаем.
        if self.tabs.currentIndex() == 0:
            self.tabs.setTabToolTip(1, f"Кадры готовы: {out_dir}")

    # =======================================================================
    # Статус / лог (общие для всех вкладок)
    # =======================================================================
    def _set_progress(self, frac: float, message: str) -> None:
        self._progress_bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._status_label.setText(message)
        self._log(message)

    def _log(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        self._log_panel.appendPlainText(f"[{stamp}] {message}")

    def closeEvent(self, event) -> None:  # noqa: N802 (имя метода задано Qt)
        if self._job_controller.shutdown():
            event.accept()
        else:
            self._log("Невозможно безопасно закрыть приложение: фоновая задача ещё выполняется.")
            event.ignore()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)

    try:
        import qdarktheme
        app.setStyleSheet(qdarktheme.load_stylesheet("dark") + EXTRA_QSS)
    except ImportError:
        app.setStyleSheet(EXTRA_QSS)

    window = CALSlicerMainWindow()
    window.resize(1680, 980)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
