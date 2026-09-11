"""
viewport.py
===========
Аппаратно ускоренный 3D-вьюпорт на PyVista (VTK) внутри Qt-виджета.

Ключевые решения, отличающие его от прототипа на matplotlib:

  • Оси, деления, сетка — полностью отключены. Единственный "интерьер" —
    эстетичный градиентный фон.
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


class Viewport3D(QWidget):
    """Qt-виджет со сценой, gizmo трансформации и XYZ-кубом."""

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
        self._matrix_animation: QVariantAnimation | None = None
        self._affine_capture_pending = False
        self._pending_affine_start: np.ndarray | None = None

        self._create_orientation_cube(plotter)

        self._reset_camera_view()

    def _create_orientation_cube(self, plotter: Any) -> None:
        """Create a compact, clearly labelled XYZ orientation marker."""
        try:
            cube = vtk.vtkAnnotatedCubeActor()
            cube.SetXPlusFaceText("X")
            cube.SetXMinusFaceText("-X")
            cube.SetYPlusFaceText("Y")
            cube.SetYMinusFaceText("-Y")
            cube.SetZPlusFaceText("Z")
            cube.SetZMinusFaceText("-Z")
            cube.SetFaceTextScale(0.34)
            cube.SetFaceTextVisibility(True)
            cube.SetTextEdgesVisibility(True)
            cube.SetCubeVisibility(True)
            cube.GetCubeProperty().SetColor(0.12, 0.16, 0.23)
            cube.GetTextEdgesProperty().SetColor(0.82, 0.86, 0.94)

            face_colors = (
                (cube.GetXPlusFaceProperty(), (0.92, 0.28, 0.25)),
                (cube.GetXMinusFaceProperty(), (0.55, 0.12, 0.12)),
                (cube.GetYPlusFaceProperty(), (0.25, 0.72, 0.38)),
                (cube.GetYMinusFaceProperty(), (0.12, 0.42, 0.21)),
                (cube.GetZPlusFaceProperty(), (0.28, 0.48, 0.92)),
                (cube.GetZMinusFaceProperty(), (0.14, 0.26, 0.58)),
            )
            for prop, color in face_colors:
                prop.SetColor(*color)
                prop.SetOpacity(0.96)

            self._orientation_cube = cube
            self._orientation_widget = plotter.add_orientation_widget(
                cube,
                interactive=False,
                viewport=(0.82, 0.04, 0.98, 0.22),
            )
        except Exception:
            # Keep the scene usable with older VTK builds lacking the marker.
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

        self._vat_actor = self.plotter.add_mesh(
            cylinder, color=VAT_COLOR, opacity=0.15, smooth_shading=True,
            specular=0.7, specular_power=20, name="vat", pickable=False, render=False,
        )
        self.plotter.render()

    # --- модель: геометрия добавляется один раз, дальше — только матрица -------
    def set_model(self, mesh: trimesh.Trimesh) -> None:
        """Вызывается один раз на новый загруженный файл (не на каждую правку)."""
        self._remove_transform_widgets()
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

    def _on_scale_interaction(self, _widget: vtk.vtkBoxWidget2, _event: str) -> None:
        matrix = self._scale_widget_matrix()
        if matrix is None or self._model_actor is None:
            return
        current = np.asarray(self._model_actor.user_matrix, dtype=np.float64).copy()
        self._animate_actor_matrix(current, matrix, duration=60)

    def _on_scale_release(self, _widget: vtk.vtkBoxWidget2, _event: str) -> None:
        matrix = self._scale_widget_matrix()
        if matrix is not None:
            self._stop_matrix_animation()
            self._set_actor_matrix(matrix)
            self._emit_transform_changed(matrix)

    def _set_actor_matrix(self, matrix: np.ndarray) -> None:
        if self._model_actor is None:
            return
        self._model_actor.user_matrix = np.asarray(matrix, dtype=np.float64).copy()
        self.plotter.render()

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
