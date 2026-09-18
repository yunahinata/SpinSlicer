"""
viewport.py
===========
Аппаратно ускоренный 3D-вьюпорт на PyVista (VTK) внутри Qt-виджета.

Ключевые решения, отличающие его от прототипа на matplotlib:

  • Оси и деления полностью отключены. Внутри колбы отображается условно
    бесконечная квадратная CAD-сетка дна, а фон остаётся градиентным.
  • Колба — статичная полупрозрачная геометрия ФИКСИРОВАННОГО размера
    (задаётся только диаметром из настроек). Она никогда не пересчитывается
    из-за трансформаций модели.
  • Модель добавляется во вьюпорт РОВНО ОДИН РАЗ как pv.PolyData. Любое
    масштабирование/поворот/сдвиг применяется через actor.user_matrix —
    это GPU-трансформация без пересчёта вершин на CPU, поэтому вращение
    камеры и правка ползунков остаются плавными на 60 FPS даже на тяжёлых
    STL.
  • Для очень тяжёлых мешей (сотни тысяч+ треугольников) во вьюпорте
    показывается децимированная копия — но это влияет ТОЛЬКО на отображение.
    Генерация проекций всегда использует полный original_mesh из ModelNode.
"""
from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
import pyvista as pv
import trimesh
import trimesh.transformations as tf
import vtk
from PyQt6.QtCore import QEasingCurve, QTimer, QVariantAnimation, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from constants import (
    MAX_VIEWPORT_TRIANGLES,
    MODEL_COLOR,
    VAT_COLOR,
    VAT_HEIGHT_RATIO,
    VAT_RESOLUTION,
    VIEWPORT_BG_BOTTOM,
    VIEWPORT_BG_TOP,
)


def _trimesh_to_pyvista(mesh: trimesh.Trimesh) -> pv.PolyData:
    faces = mesh.faces
    padded = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)])
    return pv.PolyData(mesh.vertices.astype(np.float64), padded.ravel())


def _numpy_to_vtk_transform(matrix: np.ndarray) -> vtk.vtkTransform:
    vtk_matrix = vtk.vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            vtk_matrix.SetElement(row, column, float(matrix[row, column]))
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)
    return transform


def _vtk_transform_to_numpy(transform: vtk.vtkTransform) -> np.ndarray:
    vtk_matrix = transform.GetMatrix()
    return np.array(
        [[vtk_matrix.GetElement(row, column) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def _transformed_bounds(
    bounds: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the world-space AABB of a local-space bounding box."""

    local_bounds = np.asarray(bounds, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if local_bounds.shape != (2, 3) or transform.shape != (4, 4):
        raise ValueError("Expected bounds with shape (2, 3) and a 4x4 matrix.")
    if not np.all(np.isfinite(local_bounds)) or not np.all(np.isfinite(transform)):
        raise ValueError("Bounds and transform must contain only finite values.")

    minimum, maximum = local_bounds
    corners = np.array(
        [
            (x, y, z)
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    world = (transform[:3, :3] @ corners.T).T + transform[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def _compose_trs(
    scale: np.ndarray,
    angles: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Compose the transform format used by ``ModelNode``."""

    angle_values = np.asarray(angles, dtype=np.float64)
    result = tf.euler_matrix(
        float(angle_values[0]),
        float(angle_values[1]),
        float(angle_values[2]),
        axes="sxyz",
    )
    result[:3, :3] = result[:3, :3] @ np.diag(np.asarray(scale, dtype=np.float64))
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def _camera_up_vector(direction: np.ndarray) -> np.ndarray:
    """Return a camera up vector projected from the world Z axis.

    Trackball rotation can otherwise roll the camera when the pointer crosses
    the horizon. Keeping world Z as the reference makes the viewport behave
    like a CAD editor: orbiting changes azimuth/elevation without leaning the
    scene to the left or right.
    """

    view_direction = np.asarray(direction, dtype=np.float64)
    if view_direction.shape != (3,) or not np.all(np.isfinite(view_direction)):
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    length = float(np.linalg.norm(view_direction))
    if length <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    view_direction = view_direction / length

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up = world_up - np.dot(world_up, view_direction) * view_direction
    up_length = float(np.linalg.norm(up))
    if up_length <= 1e-9:
        # Looking almost straight along Z leaves no Z projection. World Y is
        # a stable fallback and still keeps the camera roll-free.
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up = fallback_up - np.dot(fallback_up, view_direction) * view_direction
        up_length = float(np.linalg.norm(up))
    if up_length <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return up / up_length


def _scale_matrix_from_box(
    candidate: np.ndarray,
    baseline: np.ndarray,
    uniform: bool,
) -> np.ndarray:
    """Convert a VTK box transform to a stable, shear-free model transform.

    ``vtkBoxRepresentation`` returns a complete transform relative to the
    original box.  Keeping its translation and rotation while extracting the
    scale avoids feeding any transient shear into ``ModelNode``.  In uniform
    mode the changed-axis ratio is applied to the complete existing scale,
    preserving proportions even after a previous non-uniform resize.
    """

    candidate_array = np.asarray(candidate, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if (
        candidate_array.shape != (4, 4)
        or baseline_array.shape != (4, 4)
        or not np.all(np.isfinite(candidate_array))
        or not np.all(np.isfinite(baseline_array))
    ):
        return baseline_array.copy()

    try:
        candidate_scale, _candidate_shear, candidate_angles, candidate_translation, _ = (
            tf.decompose_matrix(candidate_array)
        )
        baseline_scale, _baseline_shear, _baseline_angles, _baseline_translation, _ = (
            tf.decompose_matrix(baseline_array)
        )
    except (ValueError, np.linalg.LinAlgError):
        return baseline_array.copy()

    candidate_scale = np.asarray(candidate_scale, dtype=np.float64)
    baseline_scale = np.asarray(baseline_scale, dtype=np.float64)
    if (
        candidate_scale.shape != (3,)
        or baseline_scale.shape != (3,)
        or np.any(candidate_scale <= 1e-9)
        or np.any(baseline_scale <= 1e-9)
        or not np.all(np.isfinite(candidate_scale))
        or not np.all(np.isfinite(baseline_scale))
    ):
        return baseline_array.copy()

    if uniform:
        ratios = candidate_scale / baseline_scale
        if np.any(ratios <= 1e-9) or not np.all(np.isfinite(ratios)):
            return baseline_array.copy()
        changed_axis = int(np.argmax(np.abs(np.log(ratios))))
        factor = max(float(ratios[changed_axis]), 1e-4)
        scale = baseline_scale * factor
    else:
        scale = candidate_scale

    return _compose_trs(
        scale,
        np.asarray(candidate_angles, dtype=np.float64),
        np.asarray(candidate_translation, dtype=np.float64),
    )


class Viewport3D(QWidget):
    """Qt-виджет со сценой, gizmo трансформации и навигационным ViewCube."""

    transformChanged = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter: Any = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        plotter = cast(Any, self.plotter)
        plotter.set_background(VIEWPORT_BG_BOTTOM, top=VIEWPORT_BG_TOP)
        try:
            plotter.enable_anti_aliasing("msaa")
        except Exception:
            pass

        self._vat_actor: Any = None
        self._bottom_plane_actor: Any = None
        self._model_actor: Any = None
        self._affine_widget: Any = None
        self._scale_widget: Any = None
        self._scale_representation: Any = None
        self._orientation_cube: Any = None
        self._orientation_widget: Any = None
        self._affine_enabled = False
        self._transform_mode = "move"
        self._uniform_scale = True
        self._current_diameter = 0.0
        self._model_bounds: np.ndarray | None = None
        self._matrix_animation: QVariantAnimation | None = None
        self._affine_capture_pending = False
        self._pending_affine_start: np.ndarray | None = None
        self._scale_interaction_start_matrix: np.ndarray | None = None
        self._navigation_style: Any = None
        self._scale_handle_actors: list[Any] = []

        self._install_navigation_style()
        self._create_orientation_cube(plotter)

        self._reset_camera_view()

    def _create_orientation_cube(self, plotter: Any) -> None:
        """Create a fixed Fusion-style orientation cube.

        The marker is deliberately not interactive: it is a stable camera
        indicator, not another draggable/scalable object in the scene.
        """

        try:
            interactor = self._vtk_interactor(plotter)
            if interactor is None:
                raise RuntimeError("VTK interactor is unavailable")
            cube = vtk.vtkAnnotatedCubeActor()
            cube.SetXPlusFaceText("RIGHT")
            cube.SetXMinusFaceText("LEFT")
            cube.SetYPlusFaceText("FRONT")
            cube.SetYMinusFaceText("BACK")
            cube.SetZPlusFaceText("TOP")
            cube.SetZMinusFaceText("BOTTOM")
            cube.SetFaceTextScale(0.14)
            cube.SetFaceTextVisibility(True)
            cube.SetTextEdgesVisibility(True)
            cube.SetCubeVisibility(True)
            cube.GetCubeProperty().SetColor(0.82, 0.84, 0.87)
            cube.GetTextEdgesProperty().SetColor(0.16, 0.19, 0.24)
            cube.GetXPlusFaceProperty().SetColor(0.91, 0.92, 0.94)
            cube.GetXMinusFaceProperty().SetColor(0.78, 0.80, 0.84)
            cube.GetYPlusFaceProperty().SetColor(0.86, 0.88, 0.91)
            cube.GetYMinusFaceProperty().SetColor(0.73, 0.76, 0.81)
            cube.GetZPlusFaceProperty().SetColor(0.96, 0.96, 0.97)
            cube.GetZMinusFaceProperty().SetColor(0.76, 0.78, 0.82)

            orientation_widget = vtk.vtkOrientationMarkerWidget()
            orientation_widget.SetOrientationMarker(cube)
            orientation_widget.SetInteractor(plotter.iren.interactor)
            orientation_widget.SetViewport(0.79, 0.76, 0.98, 0.98)
            orientation_widget.SetEnabled(1)
            orientation_widget.SetInteractive(0)
            self._orientation_cube = cube
            self._orientation_widget = orientation_widget
        except Exception:
            # Keep the scene usable with older VTK builds lacking the marker
            # widget. The fallback is the native PyVista orientation widget.
            try:
                camera_widget = plotter.add_camera_orientation_widget(
                    animate=True,
                    n_frames=20,
                )
                self._orientation_cube = camera_widget.GetRepresentation()
                self._orientation_widget = camera_widget
            except Exception:
                self._orientation_cube = None
                self._orientation_widget = None

    # --- камера --------------------------------------------------------------
    def _install_navigation_style(self) -> None:
        """Use CAD-like mouse bindings and keep the camera horizon level."""

        if self._vtk_interactor(self.plotter) is None:
            return
        try:
            # Tinkercad-style navigation: left drag orbits, right drag pans,
            # and the middle button dollies. Modified buttons keep the same
            # intent instead of unexpectedly rolling or changing interaction.
            self.plotter.enable_custom_trackball_style(
                left="rotate",
                shift_left="rotate",
                control_left="rotate",
                middle="dolly",
                shift_middle="dolly",
                control_middle="dolly",
                right="pan",
                shift_right="pan",
                control_right="pan",
            )
            style = getattr(getattr(self.plotter, "iren", None), "style", None)
            if style is None:
                return
            style.add_observer("InteractionEvent", self._on_camera_interaction)
            style.add_observer("EndInteractionEvent", self._on_camera_interaction)
            self._navigation_style = style
            self._lock_camera_roll()
        except Exception:
            # Keep the viewport usable with older PyVista/VTK combinations.
            try:
                self.plotter.enable_trackball_style()
            except Exception:
                pass

    def _on_camera_interaction(self, *_args: object) -> None:
        self._lock_camera_roll()

    def _lock_camera_roll(self) -> None:
        camera = getattr(self.plotter, "camera", None)
        if camera is None:
            return
        try:
            position = np.asarray(camera.position, dtype=np.float64)
            focal_point = np.asarray(camera.focal_point, dtype=np.float64)
            if (
                position.shape != (3,)
                or focal_point.shape != (3,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(focal_point))
            ):
                return
            direction = focal_point - position
            up = _camera_up_vector(direction)
            camera.SetViewUp(float(up[0]), float(up[1]), float(up[2]))
        except (AttributeError, TypeError, ValueError):
            return

    def _focus_camera_on_vat(self) -> None:
        """Keep the camera target at the fixed vat origin after a reset."""

        camera = getattr(self.plotter, "camera", None)
        if camera is None:
            return
        try:
            position = np.asarray(camera.position, dtype=np.float64)
            focal_point = np.asarray(camera.focal_point, dtype=np.float64)
            if (
                position.shape != (3,)
                or focal_point.shape != (3,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(focal_point))
            ):
                return
            target = np.zeros(3, dtype=np.float64)
            camera.SetPosition(*[float(value) for value in position - focal_point])
            camera.SetFocalPoint(*[float(value) for value in target])
        except (AttributeError, TypeError, ValueError):
            return

    @staticmethod
    def _vtk_interactor(plotter: Any) -> Any | None:
        """Return VTK's event interactor, or ``None`` for off-screen renders."""

        return getattr(getattr(plotter, "iren", None), "interactor", None)

    def _camera_reset_bounds(self) -> tuple[float, float, float, float, float, float]:
        """Return bounds for framing the vat/model, excluding the large grid."""

        diameter = max(float(self._current_diameter), 20.0)
        height = diameter * VAT_HEIGHT_RATIO
        radius = diameter / 2.0
        minimum = np.array([-radius, -radius, -height / 2.0], dtype=np.float64)
        maximum = np.array([radius, radius, height / 2.0], dtype=np.float64)

        if self._model_actor is not None and self._model_bounds is not None:
            matrix = self._model_actor.user_matrix
            if matrix is None:
                matrix = np.eye(4, dtype=np.float64)
            try:
                model_min, model_max = _transformed_bounds(self._model_bounds, matrix)
            except ValueError:
                model_min, model_max = minimum, maximum
            minimum = np.minimum(minimum, model_min)
            maximum = np.maximum(maximum, model_max)

        margin = max(diameter * 0.08, 1.0)
        minimum -= margin
        maximum += margin
        return (
            float(minimum[0]), float(maximum[0]),
            float(minimum[1]), float(maximum[1]),
            float(minimum[2]), float(maximum[2]),
        )

    def _reset_camera_view(self) -> None:
        bounds = self._camera_reset_bounds()
        self.plotter.view_isometric(bounds=bounds, render=False)
        try:
            self.plotter.camera.azimuth += 25
            self.plotter.camera.elevation += 12
        except Exception:
            pass
        self._focus_camera_on_vat()
        self._lock_camera_roll()
        self.plotter.render()

    def reset_camera(self) -> None:
        self._reset_camera_view()

    # --- колба (фиксированный размер, не зависит от модели) --------------------
    def update_vat(self, diameter_mm: float) -> None:
        self._current_diameter = diameter_mm
        radius = diameter_mm / 2.0
        height = diameter_mm * VAT_HEIGHT_RATIO

        cylinder = pv.Cylinder(
            center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
            radius=radius, height=height, resolution=VAT_RESOLUTION, capping=True,
        )

        if self._vat_actor is not None:
            self.plotter.remove_actor(self._vat_actor, render=False)
        if self._bottom_plane_actor is not None:
            self.plotter.remove_actor(self._bottom_plane_actor, render=False)

        bottom_z = -height / 2.0 + max(height * 1e-4, 1e-3)
        # A large square grid gives the same spatial cue as a CAD workplane.
        # It is intentionally much larger than the vat, while camera framing
        # above uses explicit bounds so the grid does not make the model tiny.
        grid_span = max(diameter_mm * 20.0, 1000.0)
        bottom_plane = pv.Plane(
            center=(0.0, 0.0, bottom_z),
            direction=(0.0, 0.0, 1.0),
            i_size=grid_span,
            j_size=grid_span,
            i_resolution=64,
            j_resolution=64,
        )

        self._vat_actor = self.plotter.add_mesh(
            cylinder, color=VAT_COLOR, opacity=0.15, smooth_shading=True,
            specular=0.7, specular_power=20, name="vat", pickable=False, render=False,
        )
        self._bottom_plane_actor = self.plotter.add_mesh(
            bottom_plane,
            color="#334b70",
            opacity=0.24,
            show_edges=True,
            edge_color="#7190bd",
            line_width=1.0,
            name="vat-bottom",
            pickable=False,
            render=False,
        )
        self.plotter.render()

    # --- модель: геометрия добавляется один раз, дальше — только матрица -------
    def set_model(self, mesh: trimesh.Trimesh) -> None:
        """Вызывается один раз на новый загруженный файл (не на каждую правку)."""
        self._remove_transform_widgets()
        self._model_bounds = np.asarray(mesh.bounds, dtype=np.float64).copy()
        pv_mesh = _trimesh_to_pyvista(mesh)

        display_mesh = pv_mesh
        if pv_mesh.n_faces_strict > MAX_VIEWPORT_TRIANGLES:
            try:
                ratio = 1.0 - (MAX_VIEWPORT_TRIANGLES / pv_mesh.n_faces_strict)
                display_mesh = pv_mesh.decimate_pro(ratio, preserve_topology=True)
            except Exception:
                display_mesh = pv_mesh

        # Расчёт point-normals с разбиением по резким рёбрам. С
        # split_vertices=False (как было раньше) VTK усредняет нормаль
        # в вершине по ВСЕМ смежным граням независимо от угла между
        # ними — это стирает контраст именно на мелком рельефе (гравировка,
        # резьба), геометрия при этом не теряется. feature_angle=30° —
        # держим гладкую заливку на плавных поверхностях, но сохраняем
        # резкую тень там, где грань реально излом, а не кривизна.
        display_mesh = display_mesh.compute_normals(
            cell_normals=False, point_normals=True,
            split_vertices=True, feature_angle=30.0,
            consistent_normals=True, auto_orient_normals=True,
        )

        if self._model_actor is not None:
            self.plotter.remove_actor(self._model_actor, render=False)

        self._model_actor = self.plotter.add_mesh(
            display_mesh, color=MODEL_COLOR, smooth_shading=True,
            specular=0.35, specular_power=18, name="model", pickable=False, render=False,
        )
        self._create_transform_widgets(mesh)
        self.set_transform_mode(self._transform_mode)
        self.plotter.render()

    def update_model_transform(self, matrix_4x4: np.ndarray) -> None:
        """GPU-трансформация модели — без пересчёта вершин на CPU."""
        if self._model_actor is None:
            return
        self._stop_matrix_animation()
        matrix = np.asarray(matrix_4x4, dtype=np.float64).copy()
        self._set_actor_matrix(matrix)
        if self._affine_widget is not None:
            # AffineWidget3D caches the matrix between drag gestures.
            self._affine_widget._cached_matrix = matrix.copy()
        if self._scale_representation is not None:
            self._scale_representation.SetTransform(_numpy_to_vtk_transform(matrix))

    def clear_model(self) -> None:
        self._remove_transform_widgets()
        self._model_bounds = None
        if self._model_actor is not None:
            self.plotter.remove_actor(self._model_actor, render=False)
            self._model_actor = None
            self.plotter.render()

    # --- прямое управление трансформацией ------------------------------------
    def _create_transform_widgets(self, mesh: trimesh.Trimesh) -> None:
        if self._model_actor is None:
            return
        if self._vtk_interactor(self.plotter) is None:
            # Headless/off-screen rendering is used by CI and thumbnail
            # generation. It can display the model but cannot receive VTK
            # mouse events, so do not construct interactive widgets there.
            self._affine_widget = None
            self._scale_widget = None
            self._scale_representation = None
            return

        self._affine_widget = self.plotter.add_affine_transform_widget(
            self._model_actor,
            origin=(0.0, 0.0, 0.0),
            start=False,
            scale=0.18,
            line_radius=0.025,
            always_visible=True,
            axes_colors=("#ef6a63", "#62c370", "#5d91ef"),
            interact_callback=self._on_affine_interact,
            release_callback=self._on_affine_release,
        )
        self._replace_rotation_rings()
        self._update_affine_gizmo(np.eye(4, dtype=np.float64))

        # vtkBoxWidget2 + vtkBoxRepresentation дают шесть квадратных ручек
        # масштаба без устаревшего vtkBoxWidget. Поворот и перемещение самого
        # бокса отключены: в режиме «Масштаб» меняется только размер модели.
        interactor = self._vtk_interactor(self.plotter)
        if interactor is None:
            # PyVista's off-screen renderer has no VTK event interactor. The
            # affine widget can still render for smoke tests, but box handles
            # must wait for a real desktop renderer.
            self._scale_widget = None
            self._scale_representation = None
            return

        scale_representation = vtk.vtkBoxRepresentation()
        scale_representation.SetPlaceFactor(1.08)
        # trimesh stores bounds as [[xmin, ymin, zmin], [xmax, ymax, zmax]],
        # while VTK expects (xmin, xmax, ymin, ymax, zmin, zmax).
        mesh_bounds = mesh.bounds
        bounds = (
            float(mesh_bounds[0, 0]), float(mesh_bounds[1, 0]),
            float(mesh_bounds[0, 1]), float(mesh_bounds[1, 1]),
            float(mesh_bounds[0, 2]), float(mesh_bounds[1, 2]),
        )
        scale_representation.PlaceWidget(bounds)
        scale_representation.SetTransform(_numpy_to_vtk_transform(np.eye(4, dtype=np.float64)))
        scale_representation.SetHandleSize(0.030)
        scale_representation.GetHandleProperty().SetColor(0.82, 0.85, 0.90)
        scale_representation.GetHandleProperty().SetOpacity(1.0)
        scale_representation.GetSelectedHandleProperty().SetColor(1.0, 1.0, 1.0)
        scale_representation.GetSelectedHandleProperty().SetOpacity(1.0)
        scale_representation.GetFaceProperty().SetOpacity(0.035)
        scale_representation.GetSelectedFaceProperty().SetOpacity(0.14)
        scale_representation.GetOutlineProperty().SetColor(0.70, 0.76, 0.84)
        scale_representation.GetOutlineProperty().SetLineWidth(1.4)
        scale_representation.GetOutlineProperty().SetLineStipplePattern(0xF0F0)
        scale_representation.GetOutlineProperty().SetLineStippleRepeatFactor(2)
        scale_representation.GetSelectedOutlineProperty().SetColor(0.95, 0.97, 1.0)
        scale_representation.GetSelectedOutlineProperty().SetLineWidth(1.8)
        self._scale_handle_actors = self._style_scale_handles(scale_representation)

        scale_widget = vtk.vtkBoxWidget2()
        scale_widget.SetInteractor(interactor)
        scale_widget.SetCurrentRenderer(self.plotter.renderer)
        scale_widget.SetRepresentation(scale_representation)
        scale_widget.SetRotationEnabled(False)
        scale_widget.SetTranslationEnabled(False)
        scale_widget.SetScalingEnabled(True)
        scale_widget.SetMoveFacesEnabled(True)
        scale_widget.AddObserver(vtk.vtkCommand.StartInteractionEvent, self._on_scale_start)
        scale_widget.AddObserver(vtk.vtkCommand.InteractionEvent, self._on_scale_interaction)
        scale_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, self._on_scale_release)
        scale_widget.Off()
        self._scale_widget = scale_widget
        self._scale_representation = scale_representation

    @staticmethod
    def _style_scale_handles(representation: Any) -> list[Any]:
        """Color VTK's six square axis handles like a CAD scale gizmo.

        ``vtkBoxRepresentation`` exposes one shared handle property, so
        setting that property alone makes every handle the same color. The
        representation's actor collection contains the six axis handles after
        its three outline actors; give each actor its own property instead.
        """

        actors_collection = vtk.vtkPropCollection()
        representation.GetActors(actors_collection)
        actors_collection.InitTraversal()
        actors: list[Any] = []
        prop = actors_collection.GetNextProp()
        while prop is not None:
            actor = vtk.vtkActor.SafeDownCast(prop)
            if actor is not None:
                actors.append(actor)
            prop = actors_collection.GetNextProp()

        if len(actors) < 9:
            return []

        axis_colors = (
            (0.93, 0.20, 0.17),
            (0.93, 0.20, 0.17),
            (0.25, 0.79, 0.36),
            (0.25, 0.79, 0.36),
            (0.10, 0.78, 0.82),
            (0.10, 0.78, 0.82),
        )
        for actor, color in zip(actors[3:9], axis_colors):
            actor_property = vtk.vtkProperty()
            actor_property.DeepCopy(actor.GetProperty())
            actor_property.SetColor(*color)
            actor_property.SetRepresentationToSurface()
            actor_property.SetAmbient(0.25)
            actor_property.SetDiffuse(0.75)
            actor_property.SetSpecular(0.10)
            actor.SetProperty(actor_property)

        # The seventh handle is the center/translation handle. Translation is
        # disabled for this scale box, so it only adds visual noise.
        if len(actors) >= 10:
            actors[9].SetVisibility(False)
        return actors[3:]

    def _replace_rotation_rings(self) -> None:
        """Use full Tinkercad-style rings instead of PyVista's quarter arcs."""

        widget = self._affine_widget
        actor = self._model_actor
        if widget is None or actor is None:
            return

        old_circles = list(widget._circles)
        created_rings: list[Any] = []
        try:
            for old_circle in old_circles:
                self.plotter.remove_actor(old_circle, render=False)

            model_length = float(actor.GetLength())
            ring_radius = model_length * (0.18 * 1.6)
            tube_radius = max(model_length * 0.18 * 0.025, 1e-4)
            for axis, color in enumerate(("#ef6a63", "#62c370", "#5d91ef")):
                ring = pv.Circle(resolution=96)
                ring.faces = np.empty(0, dtype=np.int64)
                ring.lines = np.hstack(
                    (
                        np.array([ring.n_points + 1], dtype=np.int64),
                        np.arange(ring.n_points, dtype=np.int64),
                        np.array([0], dtype=np.int64),
                    )
                )
                if axis == 0:
                    ring.rotate_y(-90, inplace=True)
                elif axis == 1:
                    ring.rotate_x(90, inplace=True)
                ring.points *= ring_radius
                ring_actor = self.plotter.add_mesh(
                    ring.tube(
                        radius=tube_radius,
                        absolute=True,
                        radius_factor=1.0,
                    ),
                    color=color,
                    lighting=False,
                    render_lines_as_tubes=True,
                    render=False,
                )
                matrix = np.eye(4, dtype=np.float64)
                matrix[:3, 3] = np.asarray(widget.origin, dtype=np.float64)
                ring_actor.user_matrix = matrix
                ring_actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
                ring_actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0, -20000)
                created_rings.append(ring_actor)
            widget._circles = created_rings
        except Exception:
            # The stock PyVista arcs are a safe fallback for older VTK/PyVista
            # combinations where full-ring tube construction is unavailable.
            for ring_actor in created_rings:
                self.plotter.remove_actor(ring_actor, render=False)
            widget._circles = old_circles

    def _remove_transform_widgets(self) -> None:
        self._stop_matrix_animation()
        self._affine_capture_pending = False
        self._pending_affine_start = None
        if self._affine_widget is not None:
            self._affine_widget.remove()
            self._affine_widget = None
            self._affine_enabled = False
        if self._scale_widget is not None:
            self._scale_widget.Off()
            self._scale_widget.RemoveAllObservers()
            self._scale_widget = None
        self._scale_representation = None
        self._scale_handle_actors = []
        self._scale_interaction_start_matrix = None

    def set_uniform_scale(self, enabled: bool) -> None:
        self._uniform_scale = bool(enabled)

    def set_transform_mode(self, mode: str) -> None:
        """Switch between arrows, rotation rings, and square scale handles."""
        if mode not in {"move", "rotate", "scale"}:
            return
        self._transform_mode = mode

        if self._scale_widget is not None:
            if mode == "scale":
                self._scale_widget.On()
            else:
                self._scale_widget.Off()

        if self._affine_widget is None:
            return
        if not self._affine_enabled:
            self._affine_widget.enable()
            self._affine_enabled = True
        for actor in self._affine_widget._arrows:
            actor.visibility = mode == "move"
        for actor in self._affine_widget._circles:
            actor.visibility = mode == "rotate"
        if mode == "scale" and self._affine_enabled:
            self._affine_widget.disable()
            self._affine_enabled = False
        self.plotter.render()

    def _on_affine_interact(self, previous_matrix: np.ndarray) -> None:
        """Ease each small affine-widget update instead of snapping."""
        self._pending_affine_start = np.asarray(previous_matrix, dtype=np.float64).copy()
        if not self._affine_capture_pending:
            self._affine_capture_pending = True
            QTimer.singleShot(0, self._animate_pending_affine)

    def _animate_pending_affine(self) -> None:
        self._affine_capture_pending = False
        if self._model_actor is None:
            return
        target = np.asarray(self._model_actor.user_matrix, dtype=np.float64).copy()
        start = self._pending_affine_start
        self._pending_affine_start = None
        if start is None or start.shape != (4, 4) or not np.all(np.isfinite(start)):
            start = target.copy()
        # CAD tools keep the rotation handle under the cursor.  Interpolating
        # every mouse event adds visible lag, so drag gestures update directly.
        self._set_actor_matrix(target)

    def _on_affine_release(self, matrix: np.ndarray) -> None:
        self._affine_capture_pending = False
        self._pending_affine_start = None
        final_matrix = np.asarray(matrix, dtype=np.float64).copy()
        self._stop_matrix_animation()
        self._set_actor_matrix(final_matrix)
        if self._affine_widget is not None:
            self._affine_widget._cached_matrix = final_matrix.copy()
        self._emit_transform_changed(final_matrix)
        # PyVista's affine widget temporarily restores its default camera
        # style after a drag. Reapply the CAD bindings so left-drag panning
        # remains consistent after the first move/rotate operation.
        self._install_navigation_style()

    def _scale_widget_matrix(self) -> np.ndarray | None:
        if self._scale_representation is None:
            return None
        transform = vtk.vtkTransform()
        self._scale_representation.GetTransform(transform)
        return _vtk_transform_to_numpy(transform)

    def _on_scale_start(self, _widget: vtk.vtkBoxWidget2, _event: str) -> None:
        if self._model_actor is None or self._model_actor.user_matrix is None:
            return
        self._scale_interaction_start_matrix = np.asarray(
            self._model_actor.user_matrix,
            dtype=np.float64,
        ).copy()

    def _scale_matrix_for_interaction(self, matrix: np.ndarray) -> np.ndarray:
        if self._model_actor is None or self._model_actor.user_matrix is None:
            return np.asarray(matrix, dtype=np.float64).copy()
        baseline = self._scale_interaction_start_matrix
        if baseline is None:
            baseline = np.asarray(self._model_actor.user_matrix, dtype=np.float64).copy()
        return _scale_matrix_from_box(matrix, baseline, self._uniform_scale)

    def _on_scale_interaction(self, _widget: vtk.vtkBoxWidget2, _event: str) -> None:
        raw_matrix = self._scale_widget_matrix()
        if raw_matrix is None or self._model_actor is None:
            return
        if self._scale_interaction_start_matrix is None:
            self._on_scale_start(_widget, _event)
        matrix = self._scale_matrix_for_interaction(raw_matrix)
        self._set_actor_matrix(matrix)

    def _on_scale_release(self, _widget: vtk.vtkBoxWidget2, _event: str) -> None:
        raw_matrix = self._scale_widget_matrix()
        if raw_matrix is not None:
            matrix = self._scale_matrix_for_interaction(raw_matrix)
            self._stop_matrix_animation()
            self._set_actor_matrix(matrix)
            self._scale_interaction_start_matrix = None
            if self._scale_representation is not None:
                self._scale_handle_actors = self._style_scale_handles(
                    self._scale_representation,
                )
            self._emit_transform_changed(matrix)
        else:
            self._scale_interaction_start_matrix = None

    def _set_actor_matrix(self, matrix: np.ndarray) -> None:
        if self._model_actor is None:
            return
        normalized = np.asarray(matrix, dtype=np.float64).copy()
        if normalized.shape != (4, 4) or not np.all(np.isfinite(normalized)):
            return
        self._model_actor.user_matrix = normalized
        self._update_affine_gizmo(normalized)
        self.plotter.render()

    def _update_affine_gizmo(self, matrix: np.ndarray) -> None:
        """Keep move arrows and rotation rings proportional to the model."""

        if self._affine_widget is None or self._model_bounds is None:
            return
        try:
            world_min, world_max = _transformed_bounds(self._model_bounds, matrix)
        except ValueError:
            return

        base_extents = self._model_bounds[1] - self._model_bounds[0]
        base_length = float(np.linalg.norm(base_extents))
        world_length = float(np.linalg.norm(world_max - world_min))
        if base_length <= 1e-9 or world_length <= 1e-9:
            return

        origin = (world_min + world_max) / 2.0
        size_ratio = world_length / base_length
        self._affine_widget._actor_length = world_length
        self._affine_widget.origin = tuple(float(value) for value in origin)

        gizmo_matrix = np.eye(4, dtype=np.float64)
        gizmo_matrix[:3, :3] *= size_ratio
        gizmo_matrix[:3, 3] = origin
        for actor in (*self._affine_widget._arrows, *self._affine_widget._circles):
            actor.user_matrix = gizmo_matrix.copy()

    def _stop_matrix_animation(self) -> None:
        if self._matrix_animation is not None:
            self._matrix_animation.stop()
            self._matrix_animation.deleteLater()
            self._matrix_animation = None

    def _animate_actor_matrix(
        self,
        start: np.ndarray,
        target: np.ndarray,
        *,
        duration: int,
    ) -> None:
        if self._model_actor is None:
            return
        start_matrix = np.asarray(start, dtype=np.float64).copy()
        target_matrix = np.asarray(target, dtype=np.float64).copy()
        if (
            start_matrix.shape != (4, 4)
            or target_matrix.shape != (4, 4)
            or not np.all(np.isfinite(start_matrix))
            or not np.all(np.isfinite(target_matrix))
        ):
            return
        self._stop_matrix_animation()
        if np.allclose(start_matrix, target_matrix, rtol=0.0, atol=1e-8):
            self._set_actor_matrix(target_matrix)
            return

        animation = QVariantAnimation(self)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        animation.valueChanged.connect(
            lambda value: self._set_actor_matrix(
                start_matrix + (target_matrix - start_matrix) * float(value),
            )
        )

        def finish() -> None:
            self._set_actor_matrix(target_matrix)
            if self._matrix_animation is animation:
                self._matrix_animation = None
            animation.deleteLater()

        animation.finished.connect(finish)
        self._matrix_animation = animation
        self._set_actor_matrix(start_matrix)
        animation.start()

    def _emit_transform_changed(self, matrix: np.ndarray) -> None:
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return
        self.transformChanged.emit(matrix.copy())
