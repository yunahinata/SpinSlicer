"""

Главное окно — тонкая оболочка над вкладками приложения:

  1. "🧊 Слайсер"         — генерация проекций (slicer_tab.SlicerTab).
  2. "🎬 Проектор (Видео)" — сборка/проигрывание/экспорт видео из кадров
                              (video_tab.ProjectorTab).
  3. "🔬 Симулятор"        — обратная реконструкция геометрии по кадрам
                              (simulator_tab.SimulatorTab).
  4. "Настройки"          — параметры принтера, процесса и VAMToolbox.

Статус-бар и лог общие для всего приложения. Прогресс-бар показывается
только на главной вкладке "Слайсер".


"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
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
    APP_ORG,
    APP_TITLE,
    BUTTON_RADIUS,
    PANEL_RADIUS,
)
from i18n import LANGUAGES, apply_translations, language, set_language, tr
from job_controller import JobController
from simulator_tab import SimulatorTab
from slicer_tab import SlicerTab
from ui_panels import ProcessSettingsPanel
from video_tab import ProjectorTab


def resource_path(relative_path: str) -> str:
    """Resolve a bundled resource both from source and a PyInstaller build."""

    bundle_root = getattr(sys, "_MEIPASS", None)
    root = Path(bundle_root) if isinstance(bundle_root, str) else Path(__file__).resolve().parent
    return str(root / relative_path)


APP_ICON_PATH = resource_path("assets/spinslicer.svg")


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
QPushButton#transformModeButton {{
    padding: 5px 6px;
}}
QPushButton#transformModeButton:checked {{
    background-color: #4f72d9;
    color: white;
    font-weight: 600;
}}
QPushButton#nudgeButton {{
    padding: 2px 8px;
    font-size: 15px;
}}
QLabel#transformValue {{
    color: #cbd2e1;
    background-color: #161c27;
    border: 1px solid #30394a;
    border-radius: 4px;
    padding: 5px 7px;
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
QFrame#workflowHint {{
    background-color: #171b25;
    border: 1px solid #33384a;
    border-radius: 8px;
}}
QLabel#workflowHintTitle {{
    color: #edf2ff;
    font-size: 12px;
    font-weight: 700;
}}
QLabel#workflowHintText {{
    color: #c0c8d8;
    font-size: 11px;
}}
QLabel#workflowHintSteps {{
    color: #8fb3f0;
    font-size: 11px;
}}
QLabel#workflowHintControls {{
    color: #8f98aa;
    font-size: 10px;
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
        self.setWindowIcon(QIcon(APP_ICON_PATH))
        self._settings = QSettings(APP_ORG, APP_TITLE)
        self._job_controller = JobController()

        self._build_central()
        self._build_status_bar()
        self._wire_signals()
        index = self.language_combo.findData(language())
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
        apply_translations(self)

        self._log(tr("Готово к работе. Загрузите STL-модель на вкладке «Слайсер», чтобы начать."))

    # =======================================================================
    # Построение интерфейса
    # =======================================================================
    def _build_central(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 10)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(APP_TITLE)
        title.setObjectName("panelTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.language_label = QLabel("Язык:")
        header.addWidget(self.language_label)
        self.language_combo = QComboBox()
        for code, label in LANGUAGES.items():
            self.language_combo.addItem(label, code)
        header.addWidget(self.language_combo)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.settings_tab = ProcessSettingsPanel()
        self.slicer_tab = SlicerTab(
            job_controller=self._job_controller,
            process_panel=self.settings_tab,
        )
        self.projector_tab = ProjectorTab(job_controller=self._job_controller)
        self.simulator_tab = SimulatorTab(job_controller=self._job_controller)

        self.tabs.addTab(self.slicer_tab, "🧊 Слайсер")
        self.tabs.addTab(self.projector_tab, "🎬 Проектор (Видео)")
        self.tabs.addTab(self.simulator_tab, "🔬 Симулятор")
        self.tabs.addTab(self.settings_tab, "Настройки")

        self.tabs.setTabToolTip(0, "Настройка модели и генерация проекций")
        self.tabs.setTabToolTip(1, "Проигрывание и экспорт готовых кадров в MP4")
        self.tabs.setTabToolTip(2, "Обратная реконструкция геометрии по кадрам")
        self.tabs.setTabToolTip(3, "Настройки принтера, процесса и альтернативного движка")

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
        # Прогресс и лог рабочих вкладок стекаются в общий статус-бар/лог.
        for tab in (self.slicer_tab, self.projector_tab, self.simulator_tab):
            tab.progress.connect(self._set_progress)
            tab.logMessage.connect(self._log)

        # Как только "Слайсер" досчитал кадры — "Проектор" и "Симулятор"
        # сразу узнают, где их искать, без ручного выбора папки.
        self.slicer_tab.outputGenerated.connect(self.projector_tab.set_output_dir)
        self.slicer_tab.outputGenerated.connect(self.simulator_tab.set_output_dir)
        self.slicer_tab.outputGenerated.connect(self._on_output_generated)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(self.tabs.currentIndex())

    def _on_tab_changed(self, index: int) -> None:
        """Show the progress indicator only on the main Slicer tab."""

        self._progress_bar.setVisible(index == self.tabs.indexOf(self.slicer_tab))

    def _on_language_changed(self, index: int) -> None:
        selected = self.language_combo.itemData(index)
        if not isinstance(selected, str) or selected == language():
            return
        set_language(selected)
        self._settings.setValue("language", selected)
        apply_translations(self)
        self._log(tr("Язык изменён") + ". " + tr("Перезапустите приложение, чтобы применить язык."))

    def _on_output_generated(self, out_dir: str) -> None:
        # Мягкая подсказка: переключаем пользователя на следующий логичный
        # шаг, не мешая — если он уже сам открыл другую вкладку, не трогаем.
        if self.tabs.currentIndex() == 0:
            projector_index = self.tabs.indexOf(self.projector_tab)
            self.tabs.setTabToolTip(projector_index, f"Кадры готовы: {out_dir}")

    # =======================================================================
    # Статус / лог (общие для всех вкладок)
    # =======================================================================
    def _set_progress(self, frac: float, message: str) -> None:
        self._progress_bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._status_label.setText(tr(message))
        self._log(message)

    def _log(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        self._log_panel.appendPlainText(f"[{stamp}] {tr(message)}")

    def closeEvent(self, event) -> None:  # noqa: N802 (имя метода задано Qt)
        self.settings_tab.shutdown()
        if self._job_controller.shutdown():
            event.accept()
        else:
            self._log("Невозможно безопасно закрыть приложение: фоновая задача ещё выполняется.")
            event.ignore()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setWindowIcon(QIcon(APP_ICON_PATH))

    settings = QSettings(APP_ORG, APP_TITLE)
    saved_language = settings.value("language", "ru")
    if isinstance(saved_language, str) and saved_language in LANGUAGES:
        set_language(saved_language)

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
