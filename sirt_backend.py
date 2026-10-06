"""Optional native SIRT projection optimizer used by the Slicer pipeline."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.sparse import coo_matrix, vstack

from constants import SIRT_OPERATOR_PEAK_BYTES_PER_PIXEL_FRAME

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]
_NativeProgressCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
)
_NativeCancelCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)

SIRT_HIGH_DOSE_THRESHOLD = 0.85


class NativeSirtUnavailable(RuntimeError):
    """Raised when the platform-specific SIRT library has not been built."""


class NativeSirtCancelled(RuntimeError):
    """Raised when the user cancels SIRT matrix construction or optimization."""


def estimate_operator_peak_bytes(grid_res: int, num_frames: int) -> int:
    """Conservatively estimate sparse projector construction and storage."""

    return (
        SIRT_OPERATOR_PEAK_BYTES_PER_PIXEL_FRAME
        * int(grid_res)
        * int(grid_res)
        * int(num_frames)
    )


def _library_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("libsirt.dll",)
    if sys.platform == "darwin":
        return ("libsirt.dylib",)
    return ("libsirt.so",)


def _library_candidates() -> list[Path]:
    names = _library_names()
    roots: list[Path] = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        roots.append(Path(bundle_root))
    roots.append(Path(__file__).resolve().parent)
    roots.append(Path(sys.executable).resolve().parent)

    candidates: list[Path] = []
    for root in roots:
        for name in names:
            candidates.extend(
                [
                    root / name,
                    root / "native" / name,
                    root / "build" / "native" / name,
                    root / "native" / "build" / name,
                ]
            )
    return candidates


def _configure_library(library: ctypes.CDLL) -> ctypes.CDLL:
    float_pointer = ctypes.POINTER(ctypes.c_float)
    int_pointer = ctypes.POINTER(ctypes.c_int32)
    bool_pointer = ctypes.POINTER(ctypes.c_bool)

    native_forward = library.forward_project
    native_forward.restype = None
    native_forward.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        float_pointer,
        int_pointer,
        int_pointer,
        float_pointer,
        float_pointer,
    ]

    native_loop = library.sirt_loop_ex
    native_loop.restype = ctypes.c_int
    native_loop.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        float_pointer,
        float_pointer,
        int_pointer,
        int_pointer,
        float_pointer,
        int_pointer,
        int_pointer,
        float_pointer,
        float_pointer,
        float_pointer,
        bool_pointer,
        float_pointer,
        _NativeProgressCallback,
        _NativeCancelCallback,
        ctypes.c_void_p,
    ]
    return library


def _load_library() -> ctypes.CDLL:
    failures: list[str] = []
    for candidate in _library_candidates():
        if not candidate.is_file():
            continue
        try:
            return _configure_library(ctypes.CDLL(str(candidate)))
        except (OSError, AttributeError) as exc:
            failures.append(f"{candidate}: {exc}")

    details = "\n".join(failures)
    if details:
        details = f"\n{details}"
    raise NativeSirtUnavailable(
        "The native SIRT library is not available for this platform. "
        "Build it with `python packaging/build_sirt.py` and restart SpinSlicer."
        f"{details}"
    )


def is_native_sirt_available() -> bool:
    """Return whether this process can load the packaged native SIRT kernel."""

    try:
        _load_library()
    except NativeSirtUnavailable:
        return False
    return True


def _pointer(array: np.ndarray, pointer_type: Any) -> Any:
    return array.ctypes.data_as(pointer_type)


def _build_ray_operator(
    grid_res: int,
    angles_deg: np.ndarray,
    progress_cb: ProgressCallback | None,
    is_cancelled: CancelCheck,
) -> tuple[Any, Any]:
    """Build a pixel-driven parallel-ray operator and its transpose.

    Pixel values are linearly distributed between the two nearest detector
    bins. At 0 and 90 degrees this reduces to exact column and row sums.
    """

    pixel_count = grid_res * grid_res
    detector_center = (grid_res - 1) * 0.5
    coordinates = np.arange(grid_res, dtype=np.float32) - detector_center
    pixel_x = np.broadcast_to(coordinates[None, :], (grid_res, grid_res)).ravel()
    pixel_y = np.broadcast_to(coordinates[:, None], (grid_res, grid_res)).ravel()
    pixel_indices = np.arange(pixel_count, dtype=np.int32)
    angle_matrices = []
    progress_step = max(1, len(angles_deg) // 30)

    for angle_index, angle_deg in enumerate(angles_deg):
        if is_cancelled():
            raise NativeSirtCancelled()

        angle = np.deg2rad(float(angle_deg))
        detector_position = (
            pixel_x * np.cos(angle) + pixel_y * np.sin(angle) + detector_center
        )
        lower_bin = np.floor(detector_position).astype(np.int32)
        fraction = (detector_position - lower_bin).astype(np.float32)
        upper_bin = lower_bin + 1

        lower_weight = 1.0 - fraction
        keep_lower = (
            (lower_bin >= 0)
            & (lower_bin < grid_res)
            & (lower_weight > 1e-7)
        )
        keep_upper = (
            (upper_bin >= 0)
            & (upper_bin < grid_res)
            & (fraction > 1e-7)
        )

        rows = np.concatenate((lower_bin[keep_lower], upper_bin[keep_upper]))
        cols = np.concatenate((pixel_indices[keep_lower], pixel_indices[keep_upper]))
        values = np.concatenate((lower_weight[keep_lower], fraction[keep_upper]))
        angle_matrix = coo_matrix(
            (values, (rows, cols)), shape=(grid_res, pixel_count), dtype=np.float32
        ).tocsr()
        angle_matrix.sum_duplicates()
        angle_matrices.append(angle_matrix)

        completed = angle_index + 1
        if progress_cb is not None and (
            completed % progress_step == 0 or completed == len(angles_deg)
        ):
            progress_cb(
                completed / len(angles_deg),
                f"Building SIRT ray matrix {completed}/{len(angles_deg)}...",
            )

    operator = vstack(angle_matrices, format="csr", dtype=np.float32)
    del angle_matrices
    operator.sort_indices()
    transpose = operator.transpose().tocsr()
    transpose.sort_indices()
    return operator, transpose


def optimize_sinograms(
    target_volume: np.ndarray,
    angles_deg: np.ndarray,
    iterations: int,
    *,
    progress_cb: ProgressCallback | None = None,
    is_cancelled: CancelCheck | None = None,
) -> np.ndarray:
    """Optimize detector images for a target volume with the native SIRT loop."""

    target = np.asarray(target_volume, dtype=np.float32)
    if target.ndim != 3 or target.shape[0] != target.shape[1]:
        raise ValueError("SIRT target must be a square 3D voxel volume.")
    if not np.isfinite(target).all():
        raise ValueError("SIRT target must contain only finite voxel values.")
    grid_res, _, num_layers = target.shape
    if not 1 <= int(iterations) <= 100:
        raise ValueError("SIRT optimizer iterations must be between 1 and 100.")

    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim != 1 or angles.size == 0 or not np.isfinite(angles).all():
        raise ValueError("SIRT projection angles must be a non-empty finite 1D array.")
    if estimate_operator_peak_bytes(grid_res, int(angles.size)) > 2 * 1024**3:
        raise ValueError(
            "The SIRT sparse projection matrix would exceed the 2 GiB memory budget. "
            "Reduce the grid resolution or number of frames."
        )

    cancelled = is_cancelled or (lambda: False)
    library = _load_library()
    num_frames = int(angles.size)
    num_rays = grid_res * num_frames

    def report_matrix_progress(fraction: float, message: str) -> None:
        if progress_cb is not None:
            progress_cb(0.30 + 0.18 * fraction, message)

    if cancelled():
        raise NativeSirtCancelled()
    operator, transpose = _build_ray_operator(
        grid_res, angles, report_matrix_progress, cancelled
    )
    if cancelled():
        raise NativeSirtCancelled()

    row_sums = np.asarray(operator.sum(axis=1), dtype=np.float32).reshape(-1)
    column_sums = np.asarray(transpose.sum(axis=1), dtype=np.float32).reshape(-1)
    row_norm = np.zeros(num_rays, dtype=np.float32)
    column_norm = np.zeros(grid_res * grid_res, dtype=np.float32)
    np.divide(1.0, row_sums, out=row_norm, where=row_sums > 0.0)
    np.divide(1.0, column_sums, out=column_norm, where=column_sums > 0.0)
    ray_hits_any = np.ascontiguousarray(row_sums > 0.0, dtype=np.bool_)

    target_flat = np.ascontiguousarray(target).reshape(-1)
    initial_flat = np.empty(num_rays * num_layers, dtype=np.float32)
    optimized_flat = np.empty_like(initial_flat)
    operator_indptr = np.ascontiguousarray(operator.indptr, dtype=np.int32)
    operator_indices = np.ascontiguousarray(operator.indices, dtype=np.int32)
    operator_data = np.ascontiguousarray(operator.data, dtype=np.float32)
    transpose_indptr = np.ascontiguousarray(transpose.indptr, dtype=np.int32)
    transpose_indices = np.ascontiguousarray(transpose.indices, dtype=np.int32)
    transpose_data = np.ascontiguousarray(transpose.data, dtype=np.float32)

    if progress_cb is not None:
        progress_cb(0.48, "Projecting target volume with the SIRT ray model...")
    float_pointer = ctypes.POINTER(ctypes.c_float)
    int_pointer = ctypes.POINTER(ctypes.c_int32)
    library.forward_project(
        num_rays,
        grid_res * grid_res,
        num_layers,
        _pointer(target_flat, float_pointer),
        _pointer(operator_indptr, int_pointer),
        _pointer(operator_indices, int_pointer),
        _pointer(operator_data, float_pointer),
        _pointer(initial_flat, float_pointer),
    )

    callback_errors: list[BaseException] = []

    @_NativeProgressCallback
    def on_progress(completed: int, total: int, _context: int) -> None:
        try:
            if progress_cb is not None:
                fraction = completed / max(total, 1)
                progress_cb(
                    0.50 + 0.28 * fraction,
                    f"Native SIRT iteration {completed}/{total}...",
                )
        except BaseException as exc:  # ctypes callbacks cannot propagate exceptions.
            callback_errors.append(exc)

    @_NativeCancelCallback
    def should_cancel(_context: int) -> int:
        try:
            return int(cancelled())
        except BaseException as exc:  # Keep Python exceptions outside native code.
            callback_errors.append(exc)
            return 1

    bool_pointer = ctypes.POINTER(ctypes.c_bool)
    status = library.sirt_loop_ex(
        num_rays,
        grid_res * grid_res,
        num_layers,
        int(iterations),
        SIRT_HIGH_DOSE_THRESHOLD,
        _pointer(initial_flat, float_pointer),
        _pointer(target_flat, float_pointer),
        _pointer(operator_indptr, int_pointer),
        _pointer(operator_indices, int_pointer),
        _pointer(operator_data, float_pointer),
        _pointer(transpose_indptr, int_pointer),
        _pointer(transpose_indices, int_pointer),
        _pointer(transpose_data, float_pointer),
        _pointer(row_norm, float_pointer),
        _pointer(column_norm, float_pointer),
        _pointer(ray_hits_any, bool_pointer),
        _pointer(optimized_flat, float_pointer),
        on_progress,
        should_cancel,
        None,
    )
    if callback_errors:
        raise RuntimeError("A Python callback failed during native SIRT optimization.") from callback_errors[0]
    if status == 1:
        raise NativeSirtCancelled()
    if status != 0:
        raise RuntimeError(f"Native SIRT returned error status {status}.")

    optimized = optimized_flat.reshape(num_frames, grid_res, num_layers)
    return np.nan_to_num(
        optimized.transpose(2, 1, 0), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)
