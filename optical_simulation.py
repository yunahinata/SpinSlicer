"""Geometrical-optics simulation for the SpinSlicer optical bench.

The printer pipeline works with projection images and inverse Radon
reconstruction.  This module adds the missing physical layer: a meridional
2-D ray trace through the projector, an optional square water compensator,
the cylindrical vat wall, and the resin.

It intentionally has no Qt/PyVista dependency.  That keeps the numerical
model testable and makes it possible to replace the 2-D cross-section with a
3-D accelerator later without changing the UI contract.

The model is a first-order laboratory simulator rather than a calibration
substitute.  It includes Snell refraction, unpolarised Fresnel transmission,
bulk absorption, finite tank aperture, and total internal reflection.  The
result is the optical mapping of a projector ray fan onto the central slice
of the resin volume; measured material indices and real lens calibration are
still required before using a physical printer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class OpticalMaterial:
    """A homogeneous optical medium used by the ray tracer.

    ``absorption_per_mm`` is a simple Beer–Lambert coefficient.  It is not a
    substitute for a wavelength-dependent resin spectrum, but it prevents a
    perfectly lossless scene from overstating the usable dose.
    """

    name: str
    refractive_index: float
    absorption_per_mm: float = 0.0

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Optical material name must not be empty.")
        if not math.isfinite(self.refractive_index) or self.refractive_index < 1.0:
            raise ValueError("Refractive index must be finite and at least 1.0.")
        if (
            not math.isfinite(self.absorption_per_mm)
            or self.absorption_per_mm < 0.0
        ):
            raise ValueError("Absorption must be a finite non-negative value.")


def _material(name: str, refractive_index: float, absorption_per_mm: float = 0.0) -> OpticalMaterial:
    return OpticalMaterial(name, refractive_index, absorption_per_mm)


@dataclass(frozen=True, slots=True)
class OpticalScene:
    """Parameters for a horizontal optical slice through the apparatus."""

    vat_diameter_mm: float = 60.0
    vat_wall_thickness_mm: float = 3.0
    vat_height_mm: float = 96.0
    optical_slice_height_mm: float = 48.0

    aquarium_inner_side_mm: float = 120.0
    aquarium_wall_thickness_mm: float = 5.0
    water_level_mm: float = 120.0
    use_water_compensator: bool = True

    projector_distance_mm: float = 250.0
    horizontal_fov_deg: float = 12.0
    target_plane_x_mm: float = 0.0
    ray_count: int = 241
    output_resolution: int = 256

    ambient: OpticalMaterial = field(
        default_factory=lambda: _material("Air", 1.000293, 0.0)
    )
    aquarium_wall: OpticalMaterial = field(
        default_factory=lambda: _material("Aquarium glass", 1.52, 0.0002)
    )
    water: OpticalMaterial = field(
        default_factory=lambda: _material("Water", 1.3330, 0.00002)
    )
    vat_wall: OpticalMaterial = field(
        default_factory=lambda: _material("Vat glass", 1.52, 0.0002)
    )
    resin: OpticalMaterial = field(
        default_factory=lambda: _material("Photopolymer", 1.49, 0.002)
    )

    @property
    def vat_outer_radius_mm(self) -> float:
        return self.vat_diameter_mm / 2.0

    @property
    def vat_inner_radius_mm(self) -> float:
        return self.vat_outer_radius_mm - self.vat_wall_thickness_mm

    @property
    def aquarium_inner_half_side_mm(self) -> float:
        return self.aquarium_inner_side_mm / 2.0

    @property
    def aquarium_outer_half_side_mm(self) -> float:
        return self.aquarium_inner_half_side_mm + self.aquarium_wall_thickness_mm

    @property
    def compensator_active(self) -> bool:
        """Whether the selected horizontal slice is actually under water."""

        return bool(
            self.use_water_compensator
            and self.water_level_mm + _EPS >= self.optical_slice_height_mm
        )

    def validate(self) -> None:
        for name, value in (
            ("vat_diameter_mm", self.vat_diameter_mm),
            ("vat_wall_thickness_mm", self.vat_wall_thickness_mm),
            ("vat_height_mm", self.vat_height_mm),
            ("optical_slice_height_mm", self.optical_slice_height_mm),
            ("aquarium_inner_side_mm", self.aquarium_inner_side_mm),
            ("aquarium_wall_thickness_mm", self.aquarium_wall_thickness_mm),
            ("water_level_mm", self.water_level_mm),
            ("projector_distance_mm", self.projector_distance_mm),
            ("horizontal_fov_deg", self.horizontal_fov_deg),
            ("target_plane_x_mm", self.target_plane_x_mm),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        if self.vat_diameter_mm <= 0.0:
            raise ValueError("Vat diameter must be positive.")
        if not 0.0 < self.vat_wall_thickness_mm < self.vat_outer_radius_mm:
            raise ValueError("Vat wall thickness must be smaller than the vat radius.")
        if not 0.0 < self.optical_slice_height_mm <= self.vat_height_mm:
            raise ValueError("Optical slice height must be inside the vat height.")
        if self.aquarium_inner_side_mm <= self.vat_diameter_mm:
            raise ValueError("Aquarium inner side must be larger than the vat diameter.")
        if self.aquarium_wall_thickness_mm <= 0.0:
            raise ValueError("Aquarium wall thickness must be positive.")
        if self.water_level_mm < 0.0:
            raise ValueError("Water level must not be negative.")
        if self.projector_distance_mm <= self.aquarium_outer_half_side_mm:
            raise ValueError("Projector must be outside the aquarium.")
        if not 0.1 <= self.horizontal_fov_deg < 89.0:
            raise ValueError("Horizontal field of view must be in [0.1, 89) degrees.")
        if not -self.vat_inner_radius_mm < self.target_plane_x_mm < self.vat_inner_radius_mm:
            raise ValueError("Target plane must pass through the resin volume.")
        if self.ray_count < 9 or self.ray_count > 4097:
            raise ValueError("Ray count must be between 9 and 4097.")
        if self.output_resolution < 16 or self.output_resolution > 4096:
            raise ValueError("Output resolution must be between 16 and 4096.")

        for material in (
            self.ambient,
            self.aquarium_wall,
            self.water,
            self.vat_wall,
            self.resin,
        ):
            material.validate()

    def warnings(self) -> tuple[str, ...]:
        """Return non-fatal setup warnings suitable for a UI status panel."""

        warnings: list[str] = []
        if self.use_water_compensator and not self.compensator_active:
            warnings.append(
                "Уровень воды ниже выбранной оптической плоскости: лучи идут без компенсации."
            )
        clearance = (
            self.aquarium_inner_side_mm - self.vat_diameter_mm
        ) / 2.0
        if clearance < 2.0 * self.vat_wall_thickness_mm:
            warnings.append(
                "Зазор между колбой и аквариумом мал: кривизна и стенки будут очень чувствительны."
            )
        if self.horizontal_fov_deg > 50.0:
            warnings.append(
                "Большое поле зрения повышает риск полного внутреннего отражения и обрезания лучей."
            )
        return tuple(warnings)

    def without_compensator(self) -> OpticalScene:
        """Return the same setup with the square water tank optically bypassed."""

        return replace(self, use_water_compensator=False)


@dataclass(frozen=True, slots=True)
class RaySegment:
    start: tuple[float, float]
    end: tuple[float, float]
    medium: str

    @property
    def length_mm(self) -> float:
        return float(np.linalg.norm(np.asarray(self.end) - np.asarray(self.start)))


@dataclass(frozen=True, slots=True)
class RayTrace:
    input_coordinate: float
    angle_deg: float
    target_y_mm: float | None
    transmission: float
    hit_resin: bool
    total_internal_reflection: bool
    clipped_by_aperture: bool
    path_length_mm: float
    segments: tuple[RaySegment, ...]


@dataclass(frozen=True, slots=True)
class OpticalSimulationResult:
    angles_deg: np.ndarray
    input_coordinates: np.ndarray
    target_y_mm: np.ndarray
    transmission: np.ndarray
    hit_mask: np.ndarray
    tir_mask: np.ndarray
    clipped_mask: np.ndarray
    input_profile: np.ndarray
    output_profile: np.ndarray
    ideal_profile: np.ndarray
    traces: tuple[RayTrace, ...]
    target_extent_mm: float
    coverage_pct: float
    mean_transmission_pct: float
    mapping_error_mean_mm: float
    mapping_error_p95_mm: float
    mapping_error_max_mm: float


def refract_direction(
    direction: Sequence[float] | np.ndarray[Any, Any],
    normal: Sequence[float] | np.ndarray[Any, Any],
    n1: float,
    n2: float,
) -> tuple[np.ndarray, bool, float]:
    """Apply Snell's law and return ``(direction, tir, transmission)``.

    ``normal`` can point either way.  It is flipped internally so that it
    opposes the incoming ray.  The returned transmission is the fraction that
    remains after the unpolarised Fresnel reflection at this interface.
    """

    d = np.asarray(direction, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    if d.shape != (2,) or n.shape != (2,):
        raise ValueError("Direction and normal must be 2-D vectors.")
    d_norm = float(np.linalg.norm(d))
    n_norm = float(np.linalg.norm(n))
    if d_norm <= _EPS or n_norm <= _EPS:
        raise ValueError("Direction and normal must be non-zero.")
    if not math.isfinite(n1) or not math.isfinite(n2) or n1 < 1.0 or n2 < 1.0:
        raise ValueError("Refractive indices must be finite and at least 1.0.")

    d = d / d_norm
    n = n / n_norm
    if float(np.dot(d, n)) > 0.0:
        n = -n
    cos_i = float(np.clip(-np.dot(n, d), 0.0, 1.0))
    eta = n1 / n2
    sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i)
    if sin2_t > 1.0 + 1e-12:
        return d, True, 0.0

    cos_t = math.sqrt(max(0.0, 1.0 - min(1.0, sin2_t)))
    transmitted = eta * d + (eta * cos_i - cos_t) * n
    transmitted /= max(float(np.linalg.norm(transmitted)), _EPS)

    r_s_den = n1 * cos_i + n2 * cos_t
    r_p_den = n1 * cos_t + n2 * cos_i
    if abs(r_s_den) <= _EPS or abs(r_p_den) <= _EPS:
        reflectance = 1.0
    else:
        r_s = (n1 * cos_i - n2 * cos_t) / r_s_den
        r_p = (n1 * cos_t - n2 * cos_i) / r_p_den
        reflectance = 0.5 * (r_s * r_s + r_p * r_p)
    return transmitted, False, float(np.clip(1.0 - reflectance, 0.0, 1.0))


def _circle_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    radius: float,
    max_x: float,
) -> tuple[float, np.ndarray] | None:
    """Return the nearest forward circle hit before the target plane."""

    b = 2.0 * float(np.dot(origin, direction))
    c = float(np.dot(origin, origin) - radius * radius)
    discriminant = b * b - 4.0 * c
    if discriminant < -1e-10:
        return None
    root = math.sqrt(max(0.0, discriminant))
    candidates = ((-b - root) / 2.0, (-b + root) / 2.0)
    for t in sorted(candidates):
        if t <= _EPS:
            continue
        point = origin + t * direction
        if point[0] <= max_x + 1e-7:
            return float(t), point
    return None


def _vertical_plane_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    x: float,
) -> tuple[float, np.ndarray] | None:
    if abs(float(direction[0])) <= _EPS:
        return None
    t = (x - float(origin[0])) / float(direction[0])
    if t <= _EPS:
        return None
    return float(t), origin + t * direction


def _target_point(origin: np.ndarray, direction: np.ndarray, target_x: float) -> np.ndarray | None:
    hit = _vertical_plane_hit(origin, direction, target_x)
    return None if hit is None else hit[1]


def _append_segment(
    segments: list[RaySegment],
    start: np.ndarray,
    end: np.ndarray,
    material: OpticalMaterial,
) -> tuple[np.ndarray, float]:
    length = float(np.linalg.norm(end - start))
    if length > _EPS:
        segments.append(
            RaySegment(
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
                material.name,
            )
        )
    absorption = math.exp(-material.absorption_per_mm * length)
    return end, float(absorption)


def _interface(
    direction: np.ndarray,
    point: np.ndarray,
    normal: np.ndarray,
    from_material: OpticalMaterial,
    to_material: OpticalMaterial,
    transmission: float,
) -> tuple[np.ndarray, bool, float]:
    new_direction, tir, interface_transmission = refract_direction(
        direction,
        normal,
        from_material.refractive_index,
        to_material.refractive_index,
    )
    del point  # kept in the signature to make interface call sites readable
    return new_direction, tir, transmission * interface_transmission


def _missed_trace(
    input_coordinate: float,
    angle_deg: float,
    *,
    transmission: float = 0.0,
    clipped: bool = False,
    tir: bool = False,
    segments: Iterable[RaySegment] = (),
) -> RayTrace:
    return RayTrace(
        input_coordinate=input_coordinate,
        angle_deg=angle_deg,
        target_y_mm=None,
        transmission=float(np.clip(transmission, 0.0, 1.0)),
        hit_resin=False,
        total_internal_reflection=tir,
        clipped_by_aperture=clipped,
        path_length_mm=float(sum(segment.length_mm for segment in segments)),
        segments=tuple(segments),
    )


def trace_ray(scene: OpticalScene, angle_rad: float) -> RayTrace:
    """Trace one projector ray through the selected optical stack."""

    scene.validate()
    if not math.isfinite(angle_rad):
        raise ValueError("Ray angle must be finite.")

    half_fov = math.radians(scene.horizontal_fov_deg) / 2.0
    input_coordinate = float(np.clip(angle_rad / max(half_fov, _EPS), -1.0, 1.0))
    angle_deg = math.degrees(angle_rad)
    direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float64)
    origin = np.array([-scene.projector_distance_mm, 0.0], dtype=np.float64)
    target_x = scene.target_plane_x_mm
    segments: list[RaySegment] = []
    transmission = 1.0
    total_path = 0.0
    medium = scene.ambient
    point = origin

    def append(end: np.ndarray, material: OpticalMaterial) -> None:
        nonlocal point, transmission, total_path
        previous = point
        point, absorption = _append_segment(segments, previous, end, material)
        total_path += float(np.linalg.norm(point - previous))
        transmission *= absorption

    def fail(*, clipped: bool = False, tir: bool = False) -> RayTrace:
        return RayTrace(
            input_coordinate=input_coordinate,
            angle_deg=angle_deg,
            target_y_mm=None,
            transmission=float(np.clip(transmission, 0.0, 1.0)),
            hit_resin=False,
            total_internal_reflection=tir,
            clipped_by_aperture=clipped,
            path_length_mm=total_path,
            segments=tuple(segments),
        )

    # The square tank is represented by its two left vertical wall surfaces.
    # A ray that misses this finite aperture hits a horizontal tank edge first
    # and is not allowed to magically enter the water from the side.
    if scene.compensator_active:
        outer_x = -scene.aquarium_outer_half_side_mm
        inner_x = -scene.aquarium_inner_half_side_mm
        outer_hit = _vertical_plane_hit(point, direction, outer_x)
        if outer_hit is None:
            return fail(clipped=True)
        _outer_t, outer_point = outer_hit
        if abs(float(outer_point[1])) > scene.aquarium_outer_half_side_mm + 1e-7:
            return fail(clipped=True)
        append(outer_point, scene.ambient)
        direction, tir, transmission = _interface(
            direction,
            point,
            np.array([-1.0, 0.0]),
            scene.ambient,
            scene.aquarium_wall,
            transmission,
        )
        if tir:
            return fail(tir=True)
        point = point + direction * 1e-7
        medium = scene.aquarium_wall

        inner_hit = _vertical_plane_hit(point, direction, inner_x)
        if inner_hit is None:
            return fail(clipped=True)
        _inner_t, inner_point = inner_hit
        if abs(float(inner_point[1])) > scene.aquarium_inner_half_side_mm + 1e-7:
            return fail(clipped=True)
        append(inner_point, scene.aquarium_wall)
        direction, tir, transmission = _interface(
            direction,
            point,
            np.array([-1.0, 0.0]),
            scene.aquarium_wall,
            scene.water,
            transmission,
        )
        if tir:
            return fail(tir=True)
        point = point + direction * 1e-7
        medium = scene.water

    outer_hit = _circle_hit(point, direction, scene.vat_outer_radius_mm, target_x)
    if outer_hit is None:
        target = _target_point(point, direction, target_x)
        if target is None:
            return fail()
        if scene.compensator_active and abs(float(target[1])) > scene.aquarium_inner_half_side_mm:
            return fail(clipped=True)
        append(target, medium)
        return fail()

    _outer_t, outer_point = outer_hit
    append(outer_point, medium)
    outer_normal = outer_point / max(scene.vat_outer_radius_mm, _EPS)
    direction, tir, transmission = _interface(
        direction,
        point,
        outer_normal,
        medium,
        scene.vat_wall,
        transmission,
    )
    if tir:
        return fail(tir=True)
    point = point + direction * 1e-7
    medium = scene.vat_wall

    inner_hit = _circle_hit(point, direction, scene.vat_inner_radius_mm, target_x)
    if inner_hit is None:
        target = _target_point(point, direction, target_x)
        if target is not None:
            append(target, medium)
        return fail()

    _inner_t, inner_point = inner_hit
    append(inner_point, scene.vat_wall)
    inner_normal = inner_point / max(scene.vat_inner_radius_mm, _EPS)
    direction, tir, transmission = _interface(
        direction,
        point,
        inner_normal,
        scene.vat_wall,
        scene.resin,
        transmission,
    )
    if tir:
        return fail(tir=True)
    point = point + direction * 1e-7
    medium = scene.resin

    target = _target_point(point, direction, target_x)
    if target is None:
        return fail()
    append(target, medium)
    return RayTrace(
        input_coordinate=input_coordinate,
        angle_deg=angle_deg,
        target_y_mm=float(target[1]),
        transmission=float(np.clip(transmission, 0.0, 1.0)),
        hit_resin=True,
        total_internal_reflection=False,
        clipped_by_aperture=False,
        path_length_mm=total_path,
        segments=tuple(segments),
    )


def _default_profile(size: int) -> np.ndarray:
    coordinates = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    bars = 0.5 + 0.5 * np.sin(7.0 * math.pi * coordinates)
    checker = 0.5 + 0.5 * np.sin(17.0 * math.pi * coordinates)
    return np.clip(0.12 + 0.68 * bars + 0.20 * checker, 0.0, 1.0)


def _accumulate_profile(
    values: np.ndarray,
    coordinates_mm: np.ndarray,
    weights: np.ndarray,
    extent_mm: float,
    output_resolution: int,
) -> np.ndarray:
    result = np.zeros(output_resolution, dtype=np.float64)
    weight_sum = np.zeros(output_resolution, dtype=np.float64)
    for value, coordinate, weight in zip(values, coordinates_mm, weights):
        if not math.isfinite(float(coordinate)) or weight <= 0.0:
            continue
        position = (float(coordinate) + extent_mm / 2.0) / extent_mm
        if position < 0.0 or position > 1.0:
            continue
        scaled = position * (output_resolution - 1)
        left = int(math.floor(scaled))
        right = min(left + 1, output_resolution - 1)
        right_weight = scaled - left
        left_weight = 1.0 - right_weight
        np.add.at(result, left, float(value) * float(weight) * left_weight)
        np.add.at(weight_sum, left, float(weight) * left_weight)
        if right != left:
            np.add.at(result, right, float(value) * float(weight) * right_weight)
            np.add.at(weight_sum, right, float(weight) * right_weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(result, weight_sum, out=np.zeros_like(result), where=weight_sum > _EPS)
    return np.clip(result, 0.0, 1.0)


def simulate_optical_projection(
    scene: OpticalScene,
    source_profile: Sequence[float] | np.ndarray | None = None,
) -> OpticalSimulationResult:
    """Simulate a projector profile through the optical scene.

    The profile is a one-dimensional horizontal slice of a real projection
    frame.  This is deliberately the same slice used in the ray diagram, so
    it can be inspected at interactive rates while tuning hardware geometry.
    """

    scene.validate()
    half_fov = math.radians(scene.horizontal_fov_deg) / 2.0
    angles = np.linspace(-half_fov, half_fov, scene.ray_count, dtype=np.float64)
    input_coordinates = np.linspace(-1.0, 1.0, scene.ray_count, dtype=np.float64)

    if source_profile is None:
        source = _default_profile(scene.ray_count)
    else:
        source = np.asarray(source_profile, dtype=np.float64).reshape(-1)
        if source.size < 2:
            raise ValueError("Source profile must contain at least two samples.")
        if not np.all(np.isfinite(source)):
            raise ValueError("Source profile must contain only finite samples.")
        source = np.clip(source, 0.0, 1.0)
    source_coordinates = np.linspace(-1.0, 1.0, source.size, dtype=np.float64)
    sampled_source = np.interp(input_coordinates, source_coordinates, source)

    traces = tuple(trace_ray(scene, float(angle)) for angle in angles)
    target_y = np.full(scene.ray_count, np.nan, dtype=np.float64)
    transmission = np.zeros(scene.ray_count, dtype=np.float64)
    hit_mask = np.zeros(scene.ray_count, dtype=bool)
    tir_mask = np.zeros(scene.ray_count, dtype=bool)
    clipped_mask = np.zeros(scene.ray_count, dtype=bool)
    for index, trace in enumerate(traces):
        transmission[index] = trace.transmission
        tir_mask[index] = trace.total_internal_reflection
        clipped_mask[index] = trace.clipped_by_aperture
        if trace.hit_resin and trace.target_y_mm is not None:
            target_y[index] = trace.target_y_mm
            hit_mask[index] = True

    extent_mm = scene.vat_inner_radius_mm * 2.0
    output_profile = _accumulate_profile(
        sampled_source,
        target_y,
        transmission * hit_mask,
        extent_mm,
        scene.output_resolution,
    )

    ideal_y = scene.projector_distance_mm * np.tan(angles)
    ideal_profile = _accumulate_profile(
        sampled_source,
        ideal_y,
        hit_mask.astype(np.float64),
        extent_mm,
        scene.output_resolution,
    )
    errors = np.abs(target_y[hit_mask] - ideal_y[hit_mask])
    if errors.size:
        mean_error = float(np.mean(errors))
        p95_error = float(np.percentile(errors, 95.0))
        max_error = float(np.max(errors))
    else:
        mean_error = p95_error = max_error = float("nan")

    valid_transmission = transmission[hit_mask]
    return OpticalSimulationResult(
        angles_deg=np.degrees(angles),
        input_coordinates=input_coordinates,
        target_y_mm=target_y,
        transmission=transmission,
        hit_mask=hit_mask,
        tir_mask=tir_mask,
        clipped_mask=clipped_mask,
        input_profile=sampled_source,
        output_profile=output_profile,
        ideal_profile=ideal_profile,
        traces=traces,
        target_extent_mm=extent_mm,
        coverage_pct=float(np.mean(hit_mask) * 100.0),
        mean_transmission_pct=float(np.mean(valid_transmission) * 100.0)
        if valid_transmission.size
        else 0.0,
        mapping_error_mean_mm=mean_error,
        mapping_error_p95_mm=p95_error,
        mapping_error_max_mm=max_error,
    )


def warp_projection_frame(
    source_frame: np.ndarray,
    result: OpticalSimulationResult,
) -> np.ndarray:
    """Warp a grayscale projection frame with a previously traced mapping.

    The ray trace is a horizontal meridional slice, so the same mapping is
    applied to every image row.  This is useful for quickly inspecting the
    projected pattern while tuning the bench.  Full 3-D angular dependence
    still requires a calibrated rotational model and is intentionally kept
    outside this fast interactive path.
    """

    frame = np.asarray(source_frame, dtype=np.float64)
    if frame.ndim != 2 or frame.shape[0] < 1 or frame.shape[1] < 2:
        raise ValueError("Source frame must be a 2-D array with at least two columns.")
    if not np.all(np.isfinite(frame)):
        raise ValueError("Source frame must contain only finite values.")
    frame = np.clip(frame, 0.0, 1.0)

    valid = result.hit_mask & np.isfinite(result.target_y_mm)
    if np.count_nonzero(valid) < 2:
        return np.zeros_like(frame)
    input_coordinates = result.input_coordinates[valid]
    target_y = result.target_y_mm[valid]
    transmission = result.transmission[valid]
    order = np.argsort(target_y)
    target_y = target_y[order]
    input_coordinates = input_coordinates[order]
    transmission = transmission[order]

    source_coordinates = np.linspace(-1.0, 1.0, frame.shape[1], dtype=np.float64)
    target_y_coordinates = np.linspace(
        -result.target_extent_mm / 2.0,
        result.target_extent_mm / 2.0,
        frame.shape[1],
        dtype=np.float64,
    )
    warped = np.zeros_like(frame)
    for row_index, row in enumerate(frame):
        ray_values = np.interp(input_coordinates, source_coordinates, row) * transmission
        warped[row_index] = np.interp(
            target_y_coordinates,
            target_y,
            ray_values,
            left=0.0,
            right=0.0,
        )
    return np.clip(warped, 0.0, 1.0)


def make_scene_with_indices(
    *,
    vat_diameter_mm: float = 60.0,
    vat_wall_thickness_mm: float = 3.0,
    aquarium_inner_side_mm: float = 120.0,
    aquarium_wall_thickness_mm: float = 5.0,
    water_level_mm: float = 120.0,
    use_water_compensator: bool = True,
    projector_distance_mm: float = 250.0,
    horizontal_fov_deg: float = 12.0,
    ray_count: int = 241,
    output_resolution: int = 256,
    aquarium_wall_ior: float = 1.52,
    water_ior: float = 1.3330,
    vat_wall_ior: float = 1.52,
    resin_ior: float = 1.49,
    water_absorption_per_mm: float = 0.00002,
    resin_absorption_per_mm: float = 0.002,
) -> OpticalScene:
    """Convenience constructor used by the Qt form and small experiments."""

    return OpticalScene(
        vat_diameter_mm=vat_diameter_mm,
        vat_wall_thickness_mm=vat_wall_thickness_mm,
        aquarium_inner_side_mm=aquarium_inner_side_mm,
        aquarium_wall_thickness_mm=aquarium_wall_thickness_mm,
        water_level_mm=water_level_mm,
        use_water_compensator=use_water_compensator,
        projector_distance_mm=projector_distance_mm,
        horizontal_fov_deg=horizontal_fov_deg,
        ray_count=ray_count,
        output_resolution=output_resolution,
        aquarium_wall=_material("Aquarium glass", aquarium_wall_ior, 0.0002),
        water=_material("Water", water_ior, water_absorption_per_mm),
        vat_wall=_material("Vat glass", vat_wall_ior, 0.0002),
        resin=_material("Photopolymer", resin_ior, resin_absorption_per_mm),
    )
