"""
workers.py
==========
UI-поток никогда не должен блокироваться загрузкой STL или расчётом
проекций. Вся тяжёлая работа выполняется в QThread, наружу летят только
Qt-сигналы — без ручных queue.Queue и after()-поллинга, как в прототипе
на tkinter.
"""
from __future__ import annotations

import threading
import traceback
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from PyQt6.QtCore import QThread, pyqtSignal

from frame_io import list_frame_paths, load_meta
from reconstruction import Reconstructor, ReconstructionCancelled, ReconstructionResult
from slicing_engine import SlicingEngine, SliceParams, SliceCancelled

# Верхняя граница стороны кадра при сборке видео в памяти — защита от
# многогигабайтного потребления RAM на длинных сериях в высоком разрешении.
# Влияет и на предпросмотр, и на экспортируемый MP4 (кадры переиспользуются).
VIDEO_PREVIEW_MAX_DIM = 900


class LoadMeshWorker(QThread):
    """Загружает STL с диска и центрирует его по bounding box."""

    loaded = pyqtSignal(str, object)   # (путь, trimesh.Trimesh)
    failed = pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self) -> None:
        try:
            loaded = trimesh.load(self.path, force="mesh")
            mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                raise ValueError("Не удалось загрузить корректную триангуляцию.")

            # Строгое геометрическое центрирование по bounding box —
            # ModelNode дальше работает только со сдвигом/масштабом поверх этого.
            mesh.apply_translation(-mesh.bounding_box.centroid)

            self.loaded.emit(self.path, mesh)
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=3)}")


class GenerationWorker(QThread):
    """Запускает SlicingEngine в отдельном потоке и транслирует прогресс в UI."""

    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(int, str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, mesh: trimesh.Trimesh, params: SliceParams, parent=None):
        super().__init__(parent)
        self.mesh = mesh
        self.params = params
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            num_frames, out_dir = SlicingEngine.run(
                self.mesh,
                self.params,
                progress_cb=lambda frac, msg: self.progress.emit(frac, msg),
                is_cancelled=self._cancel_event.is_set,
            )
            self.finished_ok.emit(num_frames, out_dir)
        except SliceCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=6)}")


class VideoAssembleWorker(QThread):
    """Читает кадры из output_frames через OpenCV и держит их в памяти
    (список numpy-массивов) — без промежуточного видеофайла на диске."""

    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(list)   # List[np.ndarray], grayscale
    failed = pyqtSignal(str)

    def __init__(self, frames_dir: str, parent=None):
        super().__init__(parent)
        self.frames_dir = frames_dir

    def run(self) -> None:
        try:
            import cv2
        except ImportError:
            self.failed.emit(
                "Не установлен OpenCV. Выполните: pip install opencv-python-headless"
            )
            return

        try:
            paths = list_frame_paths(self.frames_dir)
            if not paths:
                raise ValueError(f"В папке {self.frames_dir} нет кадров frame_*.png.")

            meta = load_meta(self.frames_dir)
            if meta is not None and meta.num_frames != len(paths):
                self.progress.emit(
                    0.0,
                    f"Внимание: найдено {len(paths)} кадров, а по метаданным должно "
                    f"быть {meta.num_frames} — похоже, генерация была прервана/отменена "
                    f"или папка содержит кадры от другого запуска. Собираю видео из того, что есть.",
                )

            frames: List[np.ndarray] = []
            target_shape: Optional[Tuple[int, int]] = None  # (h, w) по первому кадру
            report_step = max(1, len(paths) // 30)

            for i, path in enumerate(paths):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape[:2]
                max_dim = max(h, w)
                if max_dim > VIDEO_PREVIEW_MAX_DIM:
                    scale = VIDEO_PREVIEW_MAX_DIM / max_dim
                    img = cv2.resize(img, (max(int(w * scale), 1), max(int(h * scale), 1)),
                                      interpolation=cv2.INTER_AREA)

                if target_shape is None:
                    target_shape = img.shape[:2]
                elif img.shape[:2] != target_shape:
                    # Кадр другого размера — обычно "хвост" от генерации с
                    # другими параметрами (см. проверку метаданных выше).
                    # Приводим к размеру первого кадра: иначе видео дёргается,
                    # а cv2.VideoWriter при экспорте требует одинаковый размер
                    # каждого кадра и на несовпадении либо тихо портит файл,
                    # либо падает.
                    img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

                frames.append(np.ascontiguousarray(img))

                if i % report_step == 0:
                    self.progress.emit(i / len(paths), f"Загрузка кадра {i}/{len(paths)}...")

            if not frames:
                raise ValueError("Не удалось прочитать ни одного кадра.")

            self.finished_ok.emit(frames)
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=4)}")


class VideoExportWorker(QThread):
    """Кодирует уже собранные в памяти кадры в MP4 через cv2.VideoWriter."""

    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, frames: List[np.ndarray], fps: float, out_path: str, parent=None):
        super().__init__(parent)
        self.frames = frames
        self.fps = fps
        self.out_path = out_path

    def run(self) -> None:
        try:
            import cv2
        except ImportError:
            self.failed.emit(
                "Не установлен OpenCV. Выполните: pip install opencv-python-headless"
            )
            return

        try:
            if not self.frames:
                raise ValueError("Нет кадров для сохранения.")

            h, w = self.frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h), isColor=True)
            if not writer.isOpened():
                raise RuntimeError("Не удалось открыть VideoWriter — проверьте кодек/путь сохранения.")

            total = len(self.frames)
            report_step = max(1, total // 30)

            for i, frame in enumerate(self.frames):
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
                if i % report_step == 0:
                    self.progress.emit(i / total, f"Кодирование кадра {i}/{total}...")

            writer.release()
            self.finished_ok.emit(self.out_path)
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=4)}")


class ReconstructionWorker(QThread):
    """Обратная реконструкция (Radon^-1 / FBP) для вкладки "Симулятор"."""

    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(object)   # ReconstructionResult
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, frames_dir: str, parent=None):
        super().__init__(parent)
        self.frames_dir = frames_dir
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result: ReconstructionResult = Reconstructor.run(
                self.frames_dir,
                progress_cb=lambda frac, msg: self.progress.emit(frac, msg),
                is_cancelled=self._cancel_event.is_set,
            )
            self.finished_ok.emit(result)
        except ReconstructionCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=6)}")
