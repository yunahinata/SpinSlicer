"""Optional VAMToolbox projection backend.

SpinSlicer keeps its small Radon implementation as a deterministic fallback,
but exposes the same target → projection → export separation used by Tomo and
VAMToolbox.  VAMToolbox is intentionally not a hard runtime dependency: its
Windows/conda stack and licensing terms make it unsuitable for the default
pip requirements of this research prototype.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class VAMToolboxUnavailable(RuntimeError):
    """Raised when the optional VAMToolbox backend is requested but missing."""


def is_vamtoolbox_available() -> bool:
    """Return whether the optional package can be imported in this environment."""

    try:
        import vamtoolbox  # noqa: F401
    except ImportError:
        return False
    return True


def _as_array(value: Any) -> np.ndarray:
    array = getattr(value, "array", value)
    result = np.asarray(array, dtype=np.float32)
    if result.ndim != 3:
        raise ValueError(
            f"VAMToolbox returned a {result.ndim}D sinogram; expected a 3D array."
        )
    return result


def normalize_sinogram_shape(
    sinogram: np.ndarray,
    grid_res: int,
    num_frames: int,
    num_layers: int,
) -> np.ndarray:
    """Convert common VAMToolbox layouts to ``(layers, pixels, angles)``."""

    expected = (grid_res, num_frames, num_layers)
    if sinogram.shape == expected:
        return np.transpose(sinogram, (2, 0, 1)).astype(np.float32, copy=False)

    expected = (num_frames, grid_res, num_layers)
    if sinogram.shape == expected:
        return np.transpose(sinogram, (2, 1, 0)).astype(np.float32, copy=False)

    expected = (num_layers, grid_res, num_frames)
    if sinogram.shape == expected:
        return sinogram.astype(np.float32, copy=False)

    expected = (num_layers, num_frames, grid_res)
    if sinogram.shape == expected:
        return np.transpose(sinogram, (0, 2, 1)).astype(np.float32, copy=False)

    raise ValueError(
        "Unsupported VAMToolbox sinogram shape "
        f"{sinogram.shape}; expected one of the common layouts for "
        f"grid={grid_res}, angles={num_frames}, layers={num_layers}."
    )


def optimize_sinograms(
    target_volume: np.ndarray,
    angles_deg: np.ndarray,
    iterations: int = 8,
) -> np.ndarray:
    """Run VAMToolbox CAL optimization and return normalized array layout.

    The target volume uses the same convention as SpinSlicer:
    ``(x, y, z)`` with values in ``[0, 1]``.  VAMToolbox currently documents
    the parallel-ray geometry and ``CAL`` optimizer as the portable baseline;
    CUDA is disabled here so the optional path can also run on CPU.
    """

    try:
        import vamtoolbox as vam
    except ImportError as exc:
        raise VAMToolboxUnavailable(
            "VAMToolbox is not installed. Install it in a dedicated conda "
            "environment or select the internal Radon backend."
        ) from exc

    target = np.asarray(target_volume, dtype=np.float32)
    if target.ndim != 3 or target.shape[0] != target.shape[1]:
        raise ValueError("VAM target must be a square 3D voxel volume.")
    if not 1 <= int(iterations) <= 100:
        raise ValueError("VAMToolbox optimizer iterations must be between 1 and 100.")

    target_geo = vam.geometry.TargetGeometry(
        target=np.clip(target, 0.0, 1.0),
        clip_to_circle=False,
    )
    proj_geo = vam.geometry.ProjectionGeometry(
        np.asarray(angles_deg, dtype=np.float64),
        ray_type="parallel",
        CUDA=False,
    )
    options = vam.optimize.Options(
        method="CAL",
        n_iter=int(iterations),
        d_h=0.85,
        d_l=0.60,
        filter="hamming",
    )
    optimized = vam.optimize.optimize(target_geo, proj_geo, options)
    sinogram_obj = optimized[0] if isinstance(optimized, (tuple, list)) else optimized
    raw = _as_array(sinogram_obj)
    normalized = normalize_sinogram_shape(
        raw,
        grid_res=int(target.shape[0]),
        num_frames=int(len(angles_deg)),
        num_layers=int(target.shape[2]),
    )
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
