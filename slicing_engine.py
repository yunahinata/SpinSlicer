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

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.transform import radon
from skimage.transform import resize as sk_resize

from constants import ANTI_ALIAS_FACTOR, VAT_HEIGHT_RATIO
from frame_io import (
    FrameRepository,
    GenerationManifest,
    SliceMeta,
    save_manifest,
    save_meta,
    sha256_file,
)
from validation import (
    ValidationError,
    ensure_directory,
    preflight_mesh,
    validate_slice_parameters,
    validate_stl_path,
)

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

    def validate(self) -> None:
        """Validate values before they can determine array dimensions."""

        validate_slice_parameters(
            self.diameter_mm,
            self.grid_res,
            self.output_res,
            self.num_frames,
            self.resin,
            self.fill_holes,
        )


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
        manifest_context: Optional[dict[str, Any]] = None,
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

        params.validate()
        preflight = preflight_mesh(
            mesh,
            params,
            vat_height_mm=params.diameter_mm * VAT_HEIGHT_RATIO,
        )
        if not preflight.ok:
            raise ValidationError("\n".join(preflight.errors))
        ensure_directory(params.output_dir)

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
                    to_2d = getattr(slice_3d, "to_2D", None)
                    if callable(to_2d):
                        slice_2d, _ = to_2d(to_2D=matrix_2d)
                    else:  # trimesh < 5 compatibility
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

        sinograms = np.zeros((nz, grid_res, num_frames), dtype=np.float32)
        for i in range(nz):
            if cancelled():
                raise SliceCancelled()

            sino = radon(slices[:, :, i], theta=angles, circle=True)
            if sino.shape != (grid_res, num_frames):
                raise ValueError(
                    f"Unexpected Radon shape {sino.shape}; expected {(grid_res, num_frames)}."
                )
            sinograms[i] = sino

            if i % report_step == 0:
                report(0.30 + 0.50 * (i / nz), f"Radon {i}/{nz}...")

        global_max = float(sinograms.max()) if sinograms.size else 1.0
        if global_max <= 0:
            global_max = 1.0

        max_dose = global_max * resin.intensity * resin.base_exposure
        threshold_abs = (resin.threshold / 100.0) * max_dose
        denom = max(max_dose - threshold_abs, 1e-9)

        # Каждый запуск получает отдельную директорию. Незавершённая папка
        # не имеет complete manifest и потому не может быть автоматически
        # выбрана Projector/Simulator.
        out_dir = FrameRepository.create_run_dir(params.output_dir)

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

        # Metadata становится видимым только после того, как все PNG уже
        # записаны. Manifest публикуется после проверки полного frame-set.
        save_meta(out_dir, SliceMeta(
            diameter_mm=diameter_mm, grid_res=grid_res, output_res=output_res,
            num_frames=num_frames, fill_holes=fill_holes,
            resin_base_exposure=resin.base_exposure, resin_intensity=resin.intensity,
            resin_threshold=resin.threshold,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ))

        # Publishing the complete manifest is the commit point for a run. A
        # direct caller of SlicingEngine therefore gets the same safe output
        # contract as the GUI worker, while optional context records the STL
        # hash and the ModelNode transform when those are available.
        if cancelled():
            raise SliceCancelled()
        context = manifest_context or {}
        source_path = str(context.get("source_path", ""))
        source_hash = ""
        if source_path:
            validate_stl_path(source_path)
            source_hash = sha256_file(source_path)
        if cancelled():
            raise SliceCancelled()
        transform = context.get("transform_matrix")
        if not isinstance(transform, list):
            transform = np.eye(4, dtype=np.float64).tolist()
        frame_set = FrameRepository.validate(out_dir, require_complete=False)
        save_manifest(
            out_dir,
            GenerationManifest(
                source_sha256=source_hash,
                source_name=os.path.basename(source_path) if source_path else "",
                units=str(context.get("units", "mm")),
                transform_matrix=transform,
                slice_parameters={
                    "diameter_mm": params.diameter_mm,
                    "grid_res": params.grid_res,
                    "output_res": params.output_res,
                    "num_frames": params.num_frames,
                    "fill_holes": params.fill_holes,
                    "resin": {
                        "base_exposure": params.resin.base_exposure,
                        "intensity": params.resin.intensity,
                        "threshold": params.resin.threshold,
                    },
                },
                frame_count=frame_set.frame_count,
                frame_size=(frame_set.width, frame_set.height),
                complete=True,
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )

        report(1.0, "Готово.")
        return num_frames, out_dir
