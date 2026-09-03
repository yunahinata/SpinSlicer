"""
video_tab.py
============
Вкладка "Проектор (Видео)": собирает кадры-проекции из output_frames в
последовательность numpy-массивов в памяти (через OpenCV) и проигрывает
её прямо во вкладке через QTimer, с регулировкой скорости и экспортом
в MP4 (тот же набор кадров, без повторного чтения с диска).
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from job_controller import JobController
from widgets import LabeledSlider
from workers import VideoAssembleWorker, VideoExportWorker

BASE_FPS = 24.0


class VideoPreviewWidget(QLabel):
    """Отображает кадры видео без изменения геометрии родительского окна.

    Стандартный QLabel.setPixmap(scaled_pixmap) меняет свой sizeHint на
    каждом кадре, вызывая положительную обратную связь в QLayout:
    окно начинает непрерывно растягиваться при проигрывании.
    Отрисовка через paintEvent гарантирует стабильный layout.
    """

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self._pixmap: Optional[QPixmap] = None

    def set_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        qimg = QImage(frame.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def clear_frame(self) -> None:
        self._pixmap = None
        self.update()

    def paintEvent(self, event) -> None:
        if self._pixmap is not None and not self._pixmap.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            super().paintEvent(event)


class ProjectorTab(QWidget):
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
        self._frames: List[np.ndarray] = []
        self._frame_index = 0
        self._assemble_worker: Optional[VideoAssembleWorker] = None
        self._export_worker: Optional[VideoExportWorker] = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

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

        self.preview = VideoPreviewWidget("Нажмите «Собрать и воспроизвести»,\nчтобы увидеть анимацию проекций.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setObjectName("videoPreview")
        self.preview.setMinimumHeight(420)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self.preview, 1)

        buttons = QHBoxLayout()
        self.assemble_btn = QPushButton("▶ Собрать и воспроизвести")
        self.assemble_btn.setObjectName("generateButton")
        self.assemble_btn.setToolTip("Прочитать все кадры из папки и запустить проигрывание")
        self.assemble_btn.clicked.connect(self._on_assemble_clicked)
        buttons.addWidget(self.assemble_btn)

        self.play_pause_btn = QPushButton("⏸ Пауза")
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.clicked.connect(self._on_play_pause_clicked)
        buttons.addWidget(self.play_pause_btn)

        self.save_btn = QPushButton("💾 Сохранить в MP4")
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip("Экспортировать уже собранные кадры в видеофайл")
        self.save_btn.clicked.connect(self._on_save_clicked)
        buttons.addWidget(self.save_btn)
        root.addLayout(buttons)

        self.speed_slider = LabeledSlider(
            "Скорость воспроизведения", 0.1, 4.0, 1.0, decimals=2, suffix="×",
        )
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        root.addWidget(self.speed_slider)

    # --- общая папка вывода (устанавливается из вкладки "Слайсер") -------------
    def set_output_dir(self, path: str) -> None:
        if path != self._output_dir:
            self._timer.stop()
            self._frames = []
            self._frame_index = 0
            self.play_pause_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.preview.clear_frame()
        self._output_dir = path
        self.dir_label.setText(f"Папка кадров: {path}")

    def _on_browse_clicked(self) -> None:
        start_dir = self._output_dir or os.getcwd()
        path = QFileDialog.getExistingDirectory(self, "Выбрать папку с кадрами", start_dir)
        if path:
            self.set_output_dir(path)

    # --- сборка и проигрывание ---------------------------------------------------
    def _on_assemble_clicked(self) -> None:
        if not self._output_dir or not os.path.isdir(self._output_dir):
            QMessageBox.warning(self, "Внимание", "Сначала укажите папку с кадрами (output_frames).")
            return

        self._timer.stop()
        self.assemble_btn.setEnabled(False)
        self.progress.emit(0.0, "Сборка видео из кадров...")

        self._assemble_worker = VideoAssembleWorker(self._output_dir, self)
        self._job_controller.register(self._assemble_worker)
        self._assemble_worker.progress.connect(lambda f, m: self.progress.emit(f, m))
        self._assemble_worker.finished_ok.connect(self._on_assembled)
        self._assemble_worker.failed.connect(self._on_assemble_failed)
        self._assemble_worker.cancelled.connect(
            lambda: self.progress.emit(0.0, "Сборка видео отменена.")
        )
        self._assemble_worker.finished.connect(lambda: self.assemble_btn.setEnabled(True))
        self._assemble_worker.start()

    def _on_assembled(self, frames: List[np.ndarray]) -> None:
        self._frames = frames
        self._frame_index = 0
        self.play_pause_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.play_pause_btn.setText("⏸ Пауза")
        self.progress.emit(1.0, "Видео собрано.")
        self.logMessage.emit(f"Собрано {len(frames)} кадров для проигрывания.")
        if frames:
            self.preview.set_frame(frames[0])
        self._start_playback()

    def _on_assemble_failed(self, msg: str) -> None:
        self.progress.emit(0.0, "Ошибка сборки видео.")
        QMessageBox.critical(self, "Ошибка", msg)

    def _start_playback(self) -> None:
        if not self._frames:
            return
        interval_ms = max(int(1000 / (BASE_FPS * self.speed_slider.value())), 5)
        self._timer.start(interval_ms)

    def _on_play_pause_clicked(self) -> None:
        if not self._frames:
            return
        if self._timer.isActive():
            self._timer.stop()
            self.play_pause_btn.setText("▶ Играть")
        else:
            self._start_playback()
            self.play_pause_btn.setText("⏸ Пауза")

    def _on_speed_changed(self, _v: float) -> None:
        if self._timer.isActive():
            self._start_playback()

    def _advance_frame(self) -> None:
        if not self._frames:
            return
        frame = self._frames[self._frame_index]
        self._frame_index = (self._frame_index + 1) % len(self._frames)
        self.preview.set_frame(frame)

    # --- экспорт в MP4 -----------------------------------------------------------
    def _on_save_clicked(self) -> None:
        if not self._frames:
            QMessageBox.warning(self, "Внимание", "Сначала соберите видео.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Сохранить видео", "projection_preview.mp4", "MP4 (*.mp4)")
        if not path:
            return

        fps = BASE_FPS * self.speed_slider.value()
        self.save_btn.setEnabled(False)
        self.progress.emit(0.0, "Экспорт в MP4...")

        self._export_worker = VideoExportWorker(self._frames, fps, path, self)
        self._job_controller.register(self._export_worker)
        self._export_worker.progress.connect(lambda f, m: self.progress.emit(f, m))
        self._export_worker.finished_ok.connect(self._on_export_done)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.cancelled.connect(
            lambda: self.progress.emit(0.0, "Экспорт видео отменён.")
        )
        self._export_worker.finished.connect(lambda: self.save_btn.setEnabled(True))
        self._export_worker.start()

    def _on_export_done(self, path: str) -> None:
        self.progress.emit(1.0, "Видео сохранено.")
        self.logMessage.emit(f"MP4 сохранён: {path}")
        QMessageBox.information(self, "Успех", f"Видео сохранено:\n{path}")

    def _on_export_failed(self, msg: str) -> None:
        self.progress.emit(0.0, "Ошибка экспорта.")
        QMessageBox.critical(self, "Ошибка экспорта", msg)

    def closeEvent(self, event) -> None:  # noqa: N802 (имя метода задано Qt)
        if self._job_controller.shutdown():
            event.accept()
        else:
            event.ignore()
