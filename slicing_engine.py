"""
slicing_engine.py
==================
Математическое ядро Spin Slicer.

Этот модуль НИЧЕГО не знает о PyQt6 / PyVista / customtkinter — это чистая
математика и обработка изображений. Он принимает уже готовый (трансформированный)
trimesh.Trimesh и параметры, а сообщает о прогрессе через простой callback.

Алгоритм полностью идентичен проверенному прототипу:
  • Бронебойная послойная нарезка через mesh.section в цикле — не падает
    на "грязных" STL.
  • Растеризация сечений через PIL.ImageDraw (полигоны + "рваные" линии
    сущностей — двойная защита от дыр в геометрии).
  • Опциональный ремонт сетки через binary_closing / binary_fill_holes.
  • Векторизованный расчёт проекций через skimage.transform.radon.

Логика НЕ менялась — переписан только способ сообщать о прогрессе
(callback вместо queue.Queue) и способ отмены (is_cancelled() вместо
threading.Event, доступного через self).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, binary_closing
from skimage.transform import radon, resize as sk_resize

from constants import ANTI_ALIAS_FACTOR
from frame_io import SliceMeta, save_meta

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]


@dataclass
class ResinSettings:
    """Параметры фотополимера."""
    base_exposure: float = 1.0      # базовое время засветки, с
    intensity: float = 100.0        # интенсивность источника света, %
    threshold: float = 0.0          # порог полимеризации, % от максимальной дозы


@dataclass
class SliceParams:
    """Все параметры одного запуска генерации проекций."""
    diameter_mm: float
    grid_res: int
    output_res: int
    num_frames: int
    fill_holes: bool
    resin: ResinSettings
    output_dir: str


class SliceCancelled(Exception):
    """Поднимается, когда пользователь отменил генерацию из UI."""


class SlicingEngine:
    """Бронебойный послойный слайсер + Radon-проекции. Логику не трогать."""

    @staticmethod
    def run(
        mesh: trimesh.Trimesh,
        params: SliceParams,
        progress_cb: Optional[ProgressCallback] = None,
        is_cancelled: Optional[CancelCheck] = None,
    ) -> Tuple[int, str]:
        """Считает проекции для одного (уже финально трансформированного) меша.

        Возвращает (num_frames, out_dir). Бросает SliceCancelled при отмене
        и обычные исключения при ошибках геометрии/параметров.
        """

        def report(frac: float, msg: str) -> None:
            if progress_cb is not None:
                progress_cb(frac, msg)

        def cancelled() -> bool:
            return is_cancelled() if is_cancelled is not None else False

        report(0.02, "Подготовка модели...")

        diameter_mm = params.diameter_mm
        grid_res = params.grid_res
        output_res = params.output_res
        num_frames = params.num_frames
        fill_holes = params.fill_holes
        resin = params.resin

        z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
        pitch = diameter_mm / grid_res
        z_levels = np.arange(z_min + pitch / 2.0, z_max, pitch)
        nz = len(z_levels)
        if nz == 0:
            raise ValueError("Модель слишком плоская для заданного разрешения.")

        aa_size = grid_res * ANTI_ALIAS_FACTOR
        scale_2d = aa_size / diameter_mm
        offset = aa_size / 2.0

        slices = np.zeros((grid_res, grid_res, nz), dtype=np.float32)

        report(0.05, "Оптическая нарезка сечений...")
        report_step = max(1, nz // 25)

        # --- Бронебойный послойный метод: не крашится на грязных сетках ---
        for i, z in enumerate(z_levels):
            if cancelled():
                raise SliceCancelled()

            slice_3d = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])

            if slice_3d is not None:
                matrix_2d = np.array([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -z],
                    [0.0, 0.0, 0.0, 1.0],
                ])

                try:
                    slice_2d, _ = slice_3d.to_planar(to_2D=matrix_2d)
                except Exception:
                    continue

                img = Image.new("L", (aa_size, aa_size), color=0)
                draw = ImageDraw.Draw(img)

                # 1. Заливаем честные замкнутые полигоны
                polys = getattr(slice_2d, "polygons_full", None)
                if polys is None:
                    polys = getattr(slice_2d, "polygons_closed", [])

                for poly in polys:
                    ext = np.asarray(poly.exterior.coords)
                    draw.polygon(
                        list(zip(ext[:, 0] * scale_2d + offset, ext[:, 1] * scale_2d + offset)),
                        fill=255,
                    )
                    for interior in poly.interiors:
                        ints = np.asarray(interior.coords)
                        draw.polygon(
                            list(zip(ints[:, 0] * scale_2d + offset, ints[:, 1] * scale_2d + offset)),
                            fill=0,
                        )

                # 2. Обязательно рисуем "разорванные" линии сущностей
                line_width = max(1, int(scale_2d * 0.5))
                for entity in slice_2d.entities:
                    discrete_pts = entity.discrete(slice_2d.vertices)
                    if len(discrete_pts) > 1:
                        px = discrete_pts[:, 0] * scale_2d + offset
                        py = discrete_pts[:, 1] * scale_2d + offset
                        draw.line(list(zip(px, py)), fill=255, width=line_width)

                arr = np.array(img) > 127

                # 3. Опциональный ремонт сетки (для монолитных / сломанных STL)
                if fill_holes:
                    arr = binary_closing(arr, structure=np.ones((3, 3)))
                    arr = binary_fill_holes(arr)

                # Уменьшаем с антиалиасингом для мягких краёв
                img_filled = Image.fromarray((arr * 255).astype(np.uint8))
                img_resized = img_filled.resize((grid_res, grid_res), Image.Resampling.LANCZOS)
                slices[:, :, i] = (np.array(img_resized, dtype=np.float32) / 255.0).T

            if i % report_step == 0:
                report(0.05 + 0.25 * (i / nz), f"Нарезка слоя {i}/{nz}...")

        # --- Radon-преобразование и проекции ---
        angles = (np.linspace(0.0, 360.0, num_frames, endpoint=False) + 90.0) % 360.0
        report(0.30, "Расчёт Radon-преобразования...")

        sinograms = None
        for i in range(nz):
            if cancelled():
                raise SliceCancelled()

            sino = radon(slices[:, :, i], theta=angles, circle=True)
            if sinograms is None:
                sinograms = np.zeros((nz, sino.shape[0], num_frames), dtype=np.float32)
            sinograms[i] = sino

            if i % report_step == 0:
                report(0.30 + 0.50 * (i / nz), f"Radon {i}/{nz}...")

        global_max = float(sinograms.max()) if sinograms.size else 1.0
        if global_max <= 0:
            global_max = 1.0

        max_dose = global_max * resin.intensity * resin.base_exposure
        threshold_abs = (resin.threshold / 100.0) * max_dose
        denom = max(max_dose - threshold_abs, 1e-9)

        out_dir = params.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # Чистим "хвосты" от предыдущей генерации в ту же папку. out_dir
        # всегда один и тот же для одной модели (<папка STL>/output_frames),
        # и без этой очистки при смене параметров (число кадров, разрешение)
        # между запусками — или при отмене генерации на середине — в
        # каталоге остаются вперемешку кадры от разных генераций. Проектор
        # и Симулятор читают весь frame_*.png без разбора, поэтому такой
        # "хвост" выглядит как испорченный/дёргающийся результат, хотя
        # генерация технически прошла нормально.
        for stale_path in glob.glob(os.path.join(out_dir, "frame_*.png")):
            try:
                os.remove(stale_path)
            except OSError:
                pass

        # Метаданные пишем сразу — они нужны вкладкам "Проектор"/"Симулятор"
        # даже если пользователь отменит генерацию до конца.
        save_meta(out_dir, SliceMeta(
            diameter_mm=diameter_mm, grid_res=grid_res, output_res=output_res,
            num_frames=num_frames, fill_holes=fill_holes,
            resin_base_exposure=resin.base_exposure, resin_intensity=resin.intensity,
            resin_threshold=resin.threshold,
            generated_at=datetime.now().isoformat(timespec="seconds"),
        ))

        target_h = max(int(round(output_res * nz / grid_res)), 1)

        report(0.80, "Сохранение кадров...")
        save_step = max(1, num_frames // 20)

        for i in range(num_frames):
            if cancelled():
                raise SliceCancelled()

            frame = sinograms[:, :, i]
            frame = np.flipud(frame)
            scaled = frame * resin.intensity * resin.base_exposure
            normalized = ((scaled - threshold_abs) / denom).clip(0.0, 1.0)
            output = (normalized * 255.0).astype(np.uint8)

            output_resized = sk_resize(
                output, (target_h, output_res), order=1, mode="edge",
                anti_aliasing=True, preserve_range=True,
            ).astype(np.uint8)
            Image.fromarray(output_resized, mode="L").save(
                os.path.join(out_dir, f"frame_{i:04d}.png")
            )

            if i % save_step == 0:
                report(0.80 + 0.19 * (i / num_frames), f"Сохранение кадра {i}/{num_frames}...")

        report(1.0, "Готово.")
        return num_frames, out_dir
