"""Build a reconstruction-ready projection set from the optical simulation."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from frame_io import (
    FrameRepository,
    GenerationManifest,
    SliceMeta,
    save_manifest,
    save_meta,
)
from optical_simulation import OpticalScene, OpticalSimulationResult, warp_projection_frame

ProgressCallback = Callable[[float, str], None]


def _report(progress_cb: Optional[ProgressCallback], fraction: float, message: str) -> None:
    if progress_cb is not None:
        progress_cb(float(np.clip(fraction, 0.0, 1.0)), message)


def _dose_response_profile(
    result: OpticalSimulationResult,
    scene: OpticalScene,
    width: int,
) -> np.ndarray:
    """Return a normalized surface-dose response for output image columns.

    The reconstruction input is an exposure image rather than a binary cure
    mask.  Keeping the continuous ``dose / Ec`` response preserves useful
    grayscale information even when the nominal exposure is below the resin's
    critical exposure.  Optical absorption and the geometric mapping are
    already included by :func:`warp_projection_frame`.
    """

    coordinates = np.linspace(
        -result.target_extent_mm / 2.0,
        result.target_extent_mm / 2.0,
        width,
        dtype=np.float64,
    )
    valid = result.hit_mask & np.isfinite(result.target_y_mm)
    if np.count_nonzero(valid) < 2:
        return np.ones(width, dtype=np.float64)

    target_y = result.target_y_mm[valid]
    dose = result.target_dose_mj_cm2[valid]
    finite = np.isfinite(dose)
    target_y = target_y[finite]
    dose = dose[finite]
    if target_y.size < 2:
        return np.ones(width, dtype=np.float64)

    order = np.argsort(target_y)
    critical_exposure = max(float(scene.resin_critical_exposure_mj_cm2), 1e-9)
    response = np.interp(
        coordinates,
        target_y[order],
        np.clip(dose[order] / critical_exposure, 0.0, 1.0),
        left=0.0,
        right=0.0,
    )
    return np.clip(response, 0.0, 1.0)


def _simulation_root(frame_set_dir: Path) -> Path:
    """Keep simulated runs beside the source run without polluting it."""

    if frame_set_dir.name.startswith("run-"):
        return frame_set_dir.parent / "optical_simulation"
    return frame_set_dir / "optical_simulation"


def _make_meta(
    source_meta: SliceMeta | None,
    scene: OpticalScene,
    width: int,
    height: int,
    frame_count: int,
) -> SliceMeta:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if source_meta is not None:
        return replace(
            source_meta,
            num_frames=frame_count,
            generated_at=generated_at,
            complete=True,
        )

    return SliceMeta(
        diameter_mm=float(scene.vat_diameter_mm),
        grid_res=int(np.clip(width, 16, 512)),
        output_res=int(np.clip(height, 16, 4096)),
        num_frames=frame_count,
        fill_holes=True,
        resin_base_exposure=float(scene.exposure_time_s),
        resin_intensity=float(scene.source_power_w),
        resin_threshold=0.0,
        generated_at=generated_at,
        complete=True,
    )


def build_simulated_frame_set(
    source_dir: str,
    result: OpticalSimulationResult,
    scene: OpticalScene,
    *,
    source_name: str = "",
    progress_cb: Optional[ProgressCallback] = None,
    apply_resin_dose: bool = True,
) -> str:
    """Create a complete frame set whose images include the optical model.

    The source set is resolved through :class:`FrameRepository`, so a Slicer
    output root automatically selects its latest complete run.  The generated
    images retain the original frame dimensions and angle order, which lets
    the existing inverse-Radon reconstruction consume them unchanged.
    """

    _report(progress_cb, 0.01, "Проверка исходных проекций для оптической симуляции...")
    frame_set = FrameRepository.validate(source_dir)
    if frame_set.frame_count < 2:
        raise ValueError("Для построения модели нужны как минимум два исходных кадра.")

    source_directory = Path(frame_set.resolved_dir)
    simulation_root = _simulation_root(source_directory)
    output_dir = Path(FrameRepository.create_run_dir(str(simulation_root)))
    response = (
        _dose_response_profile(result, scene, frame_set.width)
        if apply_resin_dose else np.ones(frame_set.width, dtype=np.float64)
    )

    try:
        for index, path_string in enumerate(frame_set.paths):
            with Image.open(path_string) as image:
                source = np.asarray(image.convert("L"), dtype=np.float64) / 255.0

            simulated = warp_projection_frame(source, result)
            simulated = np.clip(simulated * response[None, :], 0.0, 1.0)
            encoded = np.rint(simulated * 255.0).astype(np.uint8)
            Image.fromarray(encoded, mode="L").save(output_dir / f"frame_{index:04d}.png")

            if index == 0 or index % max(1, frame_set.frame_count // 20) == 0:
                _report(
                    progress_cb,
                    0.05 + 0.78 * ((index + 1) / frame_set.frame_count),
                    f"Оптическая проекция {index + 1}/{frame_set.frame_count}...",
                )

        meta = _make_meta(
            frame_set.meta,
            scene,
            frame_set.width,
            frame_set.height,
            frame_set.frame_count,
        )
        save_meta(str(output_dir), meta)

        generated = FrameRepository.validate(str(output_dir), require_complete=False)
        angles = (np.linspace(0.0, 360.0, generated.frame_count, endpoint=False) + 90.0) % 360.0
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(
            str(output_dir),
            GenerationManifest(
                source_name=source_name or Path(frame_set.paths[-1]).name,
                units="mm",
                slice_parameters={
                    "pipeline": "optical_simulation",
                    "source_directory": str(source_directory),
                    "source_frame_count": frame_set.frame_count,
                    "vat_diameter_mm": float(scene.vat_diameter_mm),
                    "aquarium_wall_material": scene.aquarium_wall.name,
                    "aquarium_wall_thickness_mm": float(scene.aquarium_wall_thickness_mm),
                    "compensator_material": scene.water.name,
                    "vat_wall_material": scene.vat_wall.name,
                    "wavelength_nm": float(scene.wavelength_nm),
                    "source_power_w": float(scene.source_power_w),
                    "exposure_time_s": float(scene.exposure_time_s),
                    "resin_profile": scene.resin_profile_key,
                    "resin_dose_response_applied": apply_resin_dose,
                    "resin_penetration_depth_mm": (
                        float(scene.resin_penetration_depth_mm) if apply_resin_dose else None
                    ),
                    "resin_critical_exposure_mj_cm2": (
                        float(scene.resin_critical_exposure_mj_cm2) if apply_resin_dose else None
                    ),
                },
                machine_profile={"name": "optical-simulation"},
                resin_profile={
                    "name": scene.resin_profile_key,
                    "optical_only": not apply_resin_dose,
                    "penetration_depth_mm": (
                        float(scene.resin_penetration_depth_mm) if apply_resin_dose else None
                    ),
                    "critical_exposure_mj_cm2": (
                        float(scene.resin_critical_exposure_mj_cm2) if apply_resin_dose else None
                    ),
                },
                frame_schedule={
                    "angles_deg": [float(angle) for angle in angles],
                    "frame_duration_s": 1.0,
                    "frame_rate_hz": 1.0,
                },
                frame_count=generated.frame_count,
                frame_size=(generated.width, generated.height),
                complete=True,
                generated_at=generated_at,
            ),
        )
        FrameRepository.validate(str(output_dir))
    except Exception:
        # Leave an incomplete run in place.  It is intentionally ignored by
        # FrameRepository's latest-complete-run resolver and remains debuggable.
        raise

    _report(progress_cb, 1.0, "Набор симулированных проекций готов.")
    return str(output_dir)
