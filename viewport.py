"""
viewport.py
===========
Аппаратно ускоренный 3D-вьюпорт на PyVista (VTK) внутри Qt-виджета.

Ключевые решения, отличающие его от прототипа на matplotlib:

  • Оси, деления, сетка — полностью отключены. Внутри колбы отображается
    отдельная круглая плоскость дна, а фон остаётся градиентным.
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
        try:
            # Плавная орбитальная камера мышью — как в проф. слайсерах,
            # никаких "съезжающих" осей, которые были у matplotlib.
            plotter.enable_trackball_style()
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

        self._create_orientation_cube(plotter)

        self._reset_camera_view()

    def _create_orientation_cube(self, plotter: Any) -> None:
        """Create a Fusion-style interactive camera orientation cube."""
        camera_widget: Any = None
        try:
            camera_widget = plotter.add_camera_orientation_widget(
                animate=True,
                n_frames=20,
            )
            representation = camera_widget.GetRepresentation()
            representation.SetXPlusLabelText("RIGHT")
            representation.SetXMinusLabelText("LEFT")
            representation.SetYPlusLabelText("FRONT")
            representation.SetYMinusLabelText("BACK")
            representation.SetZPlusLabelText("TOP")
            representation.SetZMinusLabelText("BOTTOM")
            representation.SetSize(132, 132)
            representation.SetPadding(8, 8)
            representation.Modified()

            self._orientation_cube = representation
            self._orientation_widget = camera_widget
        except Exception:
            # Keep the scene usable with older VTK builds lacking the camera
            # widget. The fallback is visual-only, but still shows orientation.
            if camera_widget is not None:
                try:
                    camera_widget.Off()
                except Exception:
                    pass
            try:
                cube = vtk.vtkAnnotatedCubeActor()
                cube.SetXPlusFaceText("RIGHT")
                cube.SetXMinusFaceText("LEFT")
                cube.SetYPlusFaceText("FRONT")
                cube.SetYMinusFaceText("BACK")
                cube.SetZPlusFaceText("TOP")
                cube.SetZMinusFaceText("BOTTOM")
                cube.SetFaceTextScale(0.28)
                cube.SetFaceTextVisibility(True)
                cube.SetTextEdgesVisibility(True)
                cube.SetCubeVisibility(True)
                cube.GetCubeProperty().SetColor(0.12, 0.16, 0.23)
                cube.GetTextEdgesProperty().SetColor(0.82, 0.86, 0.94)
                self._orientation_cube = cube
                self._orientation_widget = plotter.add_orientation_widget(
                    cube,
                    interactive=False,
                    viewport=(0.80, 0.03, 0.98, 0.24),
                )
            except Exception:
                self._orientation_cube = None
                self._orientation_widget = None

    # --- камера --------------------------------------------------------------
    def _reset_camera_view(self) -> None:
        self.plotter.camera_position = "iso"
        try:
            self.plotter.camera.azimuth += 25
            self.plotter.camera.elevation += 12
        except Exception:
            pass
        self.plotter.render()

    def reset_camera(self) -> None:
        self.plotter.reset_camera()
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
        bottom_plane = pv.Disc(
            center=(0.0, 0.0, bottom_z),
            inner=0.0,
            outer=radius,
            normal=(0.0, 0.0, 1.0),
            r_res=1,
            c_res=VAT_RESOLUTION,
        )

        self._vat_actor = self.plotter.add_mesh(
            cylinder, color=VAT_COLOR, opacity=0.15, smooth_shading=True,
            specular=0.7, specular_power=20, name="vat", pickable=False, render=False,
        )
        self._bottom_plane_actor = self.plotter.add_mesh(
            bottom_plane,
            color="#5d91ef",
            opacity=0.30,
            show_edges=True,
            edge_color="#8fb4ff",
            line_width=1.5,
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
        self._update_affine_gizmo(np.eye(4, dtype=np.float64))

        # vtkBoxWidget2 + vtkBoxRepresentation дают шесть квадратных ручек
        # масштаба без устаревшего vtkBoxWidget. Поворот и перемещение самого
        # бокса отключены: в режиме «Масштаб» меняется только размер модели.
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
        scale_representation.SetHandleSize(0.018)
        scale_representation.GetHandleProperty().SetColor(0.95, 0.76, 0.22)
        scale_representation.GetSelectedHandleProperty().SetColor(1.0, 0.92, 0.35)
        scale_representation.GetFaceProperty().SetOpacity(0.035)
        scale_representation.GetSelectedFaceProperty().SetOpacity(0.14)
        scale_representation.GetOutlineProperty().SetColor(0.95, 0.76, 0.22)
        scale_representation.GetSelectedOutlineProperty().SetColor(1.0, 0.92, 0.35)

        scale_widget = vtk.vtkBoxWidget2()
        scale_widget.SetInteractor(self.plotter.iren.interactor)
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
        self._animate_actor_matrix(start, target, duration=80)

    def _on_affine_release(self, matrix: np.ndarray) -> None:
        self._affine_capture_pending = False
        self._pending_affine_start = None
        final_matrix = np.asarray(matrix, dtype=np.float64).copy()
        self._stop_matrix_animation()
        self._set_actor_matrix(final_matrix)
        if self._affine_widget is not None:
            self._affine_widget._cached_matrix = final_matrix.copy()
        self._emit_transform_changed(final_matrix)

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
        current = np.asarray(self._model_actor.user_matrix, dtype=np.float64).copy()
        self._animate_actor_matrix(current, matrix, duration=60)

    def _on_scale_release(self, _widget: vtk.vtkBoxWidget2, _event: str) -> None:
        raw_matrix = self._scale_widget_matrix()
        if raw_matrix is not None:
            matrix = self._scale_matrix_for_interaction(raw_matrix)
            self._stop_matrix_animation()
            self._set_actor_matrix(matrix)
            self._scale_interaction_start_matrix = None
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
