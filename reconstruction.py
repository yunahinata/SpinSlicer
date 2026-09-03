"""
reconstruction.py
==================
Модуль обратной реконструкции для вкладки "Симулятор".

SlicingEngine делает: mesh -> послойные маски -> Radon per Z -> кадры-проекции.
Этот модуль делает ровно обратное: кадры-проекции -> синограммы per Z ->
inverse Radon (Filtered Back-Projection) -> объёмная плотность ->
изоповерхность (marching cubes).

Важная оговорка, которую стоит держать в голове: это визуальный
предпросмотр ожидаемой геометрии по уже посчитанным кадрам, а не
метрологически точная симуляция физики полимеризации смолы (реальная
доза зависит от рассеяния света в объёме, а не только от геометрии лучей).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import trimesh
from PIL import Image
from skimage.measure import marching_cubes
from skimage.transform import iradon
from skimage.transform import resize as sk_resize

from frame_io import FrameRepository

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]

# Дефолты, если рядом с кадрами нет slice_meta.json (например, кадры
# сгенерированы вручную или старой версией софта).
FALLBACK_DIAMETER_MM = 60.0
FALLBACK_GRID_RES = 112


class ReconstructionCancelled(Exception):
    """Пользователь отменил реконструкцию из UI."""


@dataclass
class ReconstructionResult:
    volume: np.ndarray   # (grid_res, grid_res, nz), плотность, нормирована 0..1
    pitch_mm: float       # физический размер одного вокселя, мм
    diameter_mm: float
    grid_res: int
    nz: int


class Reconstructor:
    """Filtered Back-Projection по всем Z-слоям. Логика зеркальна SlicingEngine."""

    @staticmethod
    def run(
        frames_dir: str,
        progress_cb: Optional[ProgressCallback] = None,
        is_cancelled: Optional[CancelCheck] = None,
    ) -> ReconstructionResult:

        def report(frac: float, msg: str) -> None:
            if progress_cb is not None:
                progress_cb(frac, msg)

        def cancelled() -> bool:
            return is_cancelled() if is_cancelled is not None else False

        report(0.02, "Чтение кадров и метаданных...")

        frame_set = FrameRepository.validate(frames_dir)
        paths = frame_set.paths
        meta = frame_set.meta
        if meta is not None:
            diameter_mm = meta.diameter_mm
            grid_res = meta.grid_res
        else:
            diameter_mm = FALLBACK_DIAMETER_MM
            grid_res = FALLBACK_GRID_RES
            report(0.03, "slice_meta.json не найден — использую дефолтные диаметр/разрешение.")

        num_frames = frame_set.frame_count
        pitch_mm = diameter_mm / grid_res

        nz = frame_set.nz

        # Те же углы, что использовались при генерации (см. slicing_engine.py) —
        # без точного совпадения реконструкция будет искажена.
        angles = (np.linspace(0.0, 360.0, num_frames, endpoint=False) + 90.0) % 360.0

        report(0.05, "Разбор кадров обратно в синограммы...")
        sinograms = np.zeros((nz, grid_res, num_frames), dtype=np.float32)
        report_step = max(1, num_frames // 20)

        for i, path in enumerate(paths):
            if cancelled():
                raise ReconstructionCancelled()

            with Image.open(path) as image:
                img = np.array(image.convert("L"), dtype=np.float32) / 255.0
            # Обратный шаг ресайза forward-пайплайна: (target_h, output_res) -> (nz, grid_res)
            small = sk_resize(
                img, (nz, grid_res), order=1, mode="edge",
                anti_aliasing=True, preserve_range=True,
            )
            sinograms[:, :, i] = np.flipud(small)

            if i % report_step == 0:
                report(0.05 + 0.35 * (i / num_frames), f"Синограмма {i}/{num_frames}...")

        report(0.40, "Обратная реконструкция (iradon) по слоям...")
        volume = np.zeros((grid_res, grid_res, nz), dtype=np.float32)
        report_step_z = max(1, nz // 30)

        for z in range(nz):
            if cancelled():
                raise ReconstructionCancelled()

            sino_slice = sinograms[z, :, :]  # (grid_res, num_frames)
            volume[:, :, z] = iradon(sino_slice, theta=angles, circle=True, filter_name="ramp")

            if z % report_step_z == 0:
                report(0.40 + 0.55 * (z / nz), f"Реконструкция слоя {z}/{nz}...")

        vmin, vmax = float(volume.min()), float(volume.max())
        volume = (volume - vmin) / (vmax - vmin) if (vmax - vmin) > 1e-9 else np.zeros_like(volume)

        report(1.0, "Реконструкция завершена.")
        return ReconstructionResult(
            volume=volume, pitch_mm=pitch_mm, diameter_mm=diameter_mm,
            grid_res=grid_res, nz=nz,
        )


def volume_to_mesh(result: ReconstructionResult, threshold_fraction: float) -> Optional[trimesh.Trimesh]:
    """Строит изоповерхность из объёма плотности при заданном пороге (0..1).

    Быстрая операция (marching cubes) — вызывается многократно при движении
    ползунка порога, без повторного (дорогого) iradon.
    """
    level = float(np.clip(threshold_fraction, 0.01, 0.99))
    volume = result.volume
    if volume.max() < level:
        return None

    # Твёрдая область почти всегда касается границы массива (модель обычно
    # шире, чем зазор до края сетки) — marching_cubes не достраивает
    # полигон на самой границе, и без паддинга результат выходит без
    # дна/крышки. Один слой нулевых вокселей вокруг решает это.
    padded = np.pad(volume, pad_width=1, mode="constant", constant_values=0.0)

    try:
        verts, faces, _normals, _values = marching_cubes(padded, level=level, spacing=(result.pitch_mm,) * 3)
    except (ValueError, RuntimeError):
        return None

    # Центрируем по размерам ДОПОЛНЕННОГО массива — паддинг симметричен
    # (по одному слою с каждой стороны), поэтому этого достаточно, чтобы
    # результат остался в тех же координатах, что и колба во вьюпорте.
    padded_shape_mm = np.array(padded.shape) * result.pitch_mm
    verts = verts - padded_shape_mm / 2.0

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
