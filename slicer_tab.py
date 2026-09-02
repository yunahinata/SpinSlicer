"""
slicer_tab.py
=============
Вкладка "Слайсер": настройки процесса, 3D-вьюпорт с колбой на PyVista,
панель трансформации объекта и генерация проекций в фоновом QThread.

Логика идентична прежней главной версии окна — она просто перенесена
внутрь QWidget, чтобы стать одной из вкладок QTabWidget. Локальные
кнопки (загрузка/сброс/генерация/папка) остались тут же, наверху вкладки,
а не в общем тулбаре — они осмысленны только в контексте этой вкладки.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import trimesh
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QMessageBox, QPushButton, QSplitter, QVBoxLayout,
    QWidget,
)

from constants import DEFAULT_DIAMETER_MM, FILL_FRACTION
from model_node import ModelNode
from slicing_engine import ResinSettings, SliceParams
from ui_panels import ObjectPanel, ProcessSettingsPanel
from viewport import Viewport3D
from workers import GenerationWorker, LoadMeshWorker


class SlicerTab(QWidget):
    progress = pyqtSignal(float, str)
    logMessage = pyqtSignal(str)
    # Сигнализирует другим вкладкам ("Проектор", "Симулятор"), в какой папке
    # появились свежие кадры — чтобы не заставлять пользователя каждый раз
    # указывать её вручную.
    outputGenerated = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._model_node: Optional[ModelNode] = None
        self._load_worker: Optional[LoadMeshWorker] = None
        self._gen_worker: Optional[GenerationWorker] = None
        self._last_output_dir: Optional[str] = None

        self._vat_timer = QTimer(self)
        self._vat_timer.setSingleShot(True)
        self._vat_timer.timeout.connect(self._apply_vat_diameter)

        self._build_ui()
        self._wire_signals()

        self._viewport.update_vat(DEFAULT_DIAMETER_MM)
        self.progress.emit(0.0, "Готово к работе. Загрузите STL-модель, чтобы начать.")

    # =======================================================================
    # Построение интерфейса
    # =======================================================================
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        toolbar = QHBoxLayout()
        self.load_btn = QPushButton("📂 Загрузить STL")
        self.load_btn.setToolTip("Открыть STL-файл модели")
        self.load_btn.clicked.connect(self._on_load_clicked)
        toolbar.addWidget(self.load_btn)

        self.reset_btn = QPushButton("↺ Сбросить")
        self.reset_btn.setToolTip("Сбросить трансформацию и заново вписать модель в колбу")
        self.reset_btn.clicked.connect(self._on_toolbar_reset)
        toolbar.addWidget(self.reset_btn)

        self.generate_btn = QPushButton("▶ Сгенерировать проекции")
        self.generate_btn.setObjectName("generateButton")
        self.generate_btn.setToolTip("Запустить расчёт проекций в фоновом потоке")
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        toolbar.addWidget(self.generate_btn)

        self.open_dir_btn = QPushButton("📁 Открыть папку")
        self.open_dir_btn.setToolTip("Открыть последнюю папку output_frames в проводнике")
        self.open_dir_btn.clicked.connect(self._on_open_output_dir)
        toolbar.addWidget(self.open_dir_btn)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)

        self._process_panel = ProcessSettingsPanel()
        self._process_panel.setMinimumWidth(300)
        self._process_panel.setMaximumWidth(380)

        self._viewport = Viewport3D()

        self._object_panel = ObjectPanel()
        self._object_panel.setMinimumWidth(300)
        self._object_panel.setMaximumWidth(380)

        splitter.addWidget(self._process_panel)
        splitter.addWidget(self._viewport)
        splitter.addWidget(self._object_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([320, 900, 320])

        root.addWidget(splitter, 1)

    def _wire_signals(self) -> None:
        self._process_panel.diameterChanged.connect(self._on_diameter_changed)
        self._object_panel.fieldChanged.connect(self._on_object_field_changed)
        self._object_panel.centerRequested.connect(self._on_center)
        self._object_panel.autoFitRequested.connect(self._on_autofit)
        self._object_panel.resetRequested.connect(self._on_panel_reset)

    # =======================================================================
    # Колба (не зависит от модели)
    # =======================================================================
    def _on_diameter_changed(self, _value: float) -> None:
        self._vat_timer.start(120)

    def _apply_vat_diameter(self) -> None:
        self._viewport.update_vat(self._process_panel.vat_diameter_mm())

    # =======================================================================
    # Загрузка STL
    # =======================================================================
    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать STL", "", "STL файлы (*.stl)")
        if not path:
            return

        self.load_btn.setEnabled(False)
        self.progress.emit(0.1, "Загрузка модели...")

        self._load_worker = LoadMeshWorker(path, self)
        self._load_worker.loaded.connect(self._on_model_loaded)
        self._load_worker.failed.connect(self._on_model_load_failed)
        self._load_worker.finished.connect(lambda: self.load_btn.setEnabled(True))
        self._load_worker.start()

    def _on_model_loaded(self, path: str, mesh: trimesh.Trimesh) -> None:
        node = ModelNode(mesh, path)
        diameter = self._process_panel.vat_diameter_mm()
        node.fit_to_diameter(diameter, FILL_FRACTION)
        self._model_node = node

        self._viewport.set_model(node.original_mesh)
        self._viewport.update_vat(diameter)
        self._viewport.update_model_transform(node.matrix())
        self._viewport.reset_camera()

        self._object_panel.show_model_info(os.path.basename(path), node)
        self._object_panel.sync_from_model(node)
        self._object_panel.set_enabled_state(True)

        self.progress.emit(1.0, "Модель загружена.")
        self.logMessage.emit(
            f"Загружено: {os.path.basename(path)} "
            f"({node.vertex_count:,} верш., {node.face_count:,} гран.)"
        )

    def _on_model_load_failed(self, msg: str) -> None:
        self.progress.emit(0.0, "Ошибка загрузки.")
        self.logMessage.emit("ОШИБКА загрузки: " + msg.splitlines()[0])
        QMessageBox.critical(self, "Ошибка", msg)

    # =======================================================================
    # Трансформации объекта
    # =======================================================================
    def _on_object_field_changed(self, key: str, value: float) -> None:
        node = self._model_node
        if node is None:
            return

        uniform = self._object_panel.is_uniform()
        if key == "size_x":
            node.set_size_mm(x=value, uniform=uniform)
        elif key == "size_y":
            node.set_size_mm(y=value, uniform=uniform)
        elif key == "size_z":
            node.set_size_mm(z=value, uniform=uniform)
        elif key == "rot_x":
            node.set_rotation_deg(x=value)
        elif key == "rot_y":
            node.set_rotation_deg(y=value)
        elif key == "rot_z":
            node.set_rotation_deg(z=value)
        elif key == "pos_x":
            node.set_translation_mm(x=value)
        elif key == "pos_y":
            node.set_translation_mm(y=value)
        elif key == "pos_z":
            node.set_translation_mm(z=value)

        self._sync_and_redraw()

    def _on_center(self) -> None:
        if self._model_node is None:
            return
        self._model_node.center_xy()
        self._sync_and_redraw()

    def _on_autofit(self) -> None:
        if self._model_node is None:
            return
        self._model_node.fit_to_diameter(self._process_panel.vat_diameter_mm(), FILL_FRACTION)
        self._sync_and_redraw()

    def _on_panel_reset(self) -> None:
        """Сброс трансформации к чистому масштабу 1:1 (кнопка на правой панели)."""
        if self._model_node is None:
            return
        self._model_node.reset()
        self._sync_and_redraw()
        self.logMessage.emit("Трансформация объекта сброшена (масштаб 1:1).")

    def _on_toolbar_reset(self) -> None:
        """Сброс + повторный авто-фит под колбу (кнопка в шапке вкладки)."""
        if self._model_node is None:
            return
        self._model_node.reset()
        self._model_node.fit_to_diameter(self._process_panel.vat_diameter_mm(), FILL_FRACTION)
        self._sync_and_redraw()
        self.logMessage.emit("Трансформация сброшена, модель заново вписана в колбу.")

    def _sync_and_redraw(self) -> None:
        self._object_panel.sync_from_model(self._model_node)
        self._viewport.update_model_transform(self._model_node.matrix())

    # =======================================================================
    # Генерация проекций
    # =======================================================================
    def _on_generate_clicked(self) -> None:
        if self._gen_worker is not None and self._gen_worker.isRunning():
            self._gen_worker.request_cancel()
            self.generate_btn.setEnabled(False)
            self.generate_btn.setText("Остановка...")
            return

        if self._model_node is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите STL-модель.")
            return

        diameter = self._process_panel.vat_diameter_mm()
        resin = ResinSettings(
            base_exposure=self._process_panel.exposure.value(),
            intensity=self._process_panel.intensity.value(),
            threshold=self._process_panel.threshold.value(),
        )

        if self._model_node.source_path:
            base_dir = os.path.dirname(os.path.abspath(self._model_node.source_path))
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, "output_frames")

        params = SliceParams(
            diameter_mm=diameter,
            grid_res=self._process_panel.grid_res.value_int(),
            output_res=self._process_panel.output_res.value_int(),
            num_frames=self._process_panel.frames.value_int(),
            fill_holes=self._process_panel.fill_holes.isChecked(),
            resin=resin,
            output_dir=out_dir,
        )

        # Генерация всегда использует полностью трансформированную копию —
        # original_mesh внутри ModelNode остаётся нетронутым.
        mesh_to_process = self._model_node.get_transformed_mesh()

        self._gen_worker = GenerationWorker(mesh_to_process, params, self)
        self._gen_worker.progress.connect(self._on_generation_progress)
        self._gen_worker.finished_ok.connect(self._on_generation_done)
        self._gen_worker.failed.connect(self._on_generation_failed)
        self._gen_worker.cancelled.connect(self._on_generation_cancelled)
        self._gen_worker.finished.connect(self._on_generation_thread_finished)

        self.generate_btn.setText("⏹ Отменить генерацию")
        self.load_btn.setEnabled(False)
        self.logMessage.emit("Запуск генерации проекций...")
        self._gen_worker.start()

    def _on_generation_progress(self, frac: float, msg: str) -> None:
        self.progress.emit(frac, msg)

    def _on_generation_done(self, num_frames: int, out_dir: str) -> None:
        self._last_output_dir = out_dir
        self.progress.emit(1.0, "Готово!")
        self.logMessage.emit(f"Сохранено {num_frames} кадров в: {out_dir}")
        self.outputGenerated.emit(out_dir)
        QMessageBox.information(self, "Успех", f"Сохранено {num_frames} кадров в\n{out_dir}")

    def _on_generation_failed(self, msg: str) -> None:
        self.progress.emit(0.0, "Ошибка генерации.")
        self.logMessage.emit("ОШИБКА генерации: " + msg.splitlines()[0])
        QMessageBox.critical(self, "Ошибка генерации", msg)

    def _on_generation_cancelled(self) -> None:
        self.progress.emit(0.0, "Отменено пользователем.")
        self.logMessage.emit("Генерация отменена пользователем.")

    def _on_generation_thread_finished(self) -> None:
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("▶ Сгенерировать проекции")
        self.load_btn.setEnabled(True)

    # =======================================================================
    # Открыть папку вывода
    # =======================================================================
    def _on_open_output_dir(self) -> None:
        target = self._last_output_dir
        if not target or not os.path.isdir(target):
            QMessageBox.information(self, "Информация", "Папка вывода ещё не создана.")
            return
        if sys.platform == "win32":
            os.startfile(target)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{target}"')
        else:
            os.system(f'xdg-open "{target}"')

    def closeEvent(self, event) -> None:  # noqa: N802 (имя метода задано Qt)
        if self._gen_worker is not None and self._gen_worker.isRunning():
            self._gen_worker.request_cancel()
            self._gen_worker.wait(2000)
        event.accept()
