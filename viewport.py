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
from PyQt6.QtCore import pyqtSignal
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
        self._affine_enabled = False
        self._transform_mode = "move"
        self._uniform_scale = True
        self._current_diameter = 0.0

        try:
            # Нативный VTK-куб показывает ориентацию сцены и всегда следует
            # за камерой. Это не картинка, а интерактивный 3D-маркер.
            plotter.add_box_axes(
                interactive=False,
                viewport=(0.78, 0.03, 0.98, 0.23),
                x_color="#ef6a63",
                y_color="#62c370",
                z_color="#5d91ef",
                x_face_color="#ef6a63",
                y_face_color="#62c370",
                z_face_color="#5d91ef",
                edge_color="#d6d9e0",
                label_color="#f4f6fb",
                opacity=0.78,
            )
        except Exception:
            # Старые VTK-сборки могут не поддерживать box axes; основной
            # вьюпорт и gizmo при этом должны продолжать работать.
            pass

        self._reset_camera_view()

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
        matrix = np.asarray(matrix_4x4, dtype=np.float64).copy()
        self._model_actor.user_matrix = matrix
        if self._affine_widget is not None:
            # AffineWidget3D caches the matrix between drag gestures.
            self._affine_widget._cached_matrix = matrix.copy()
        if self._scale_widget is not None:
            self._scale_widget.SetTransform(_numpy_to_vtk_transform(matrix))
        self.plotter.render()

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
            release_callback=self._on_affine_release,
        )

        # vtkBoxWidget даёт шесть квадратных ручек масштаба. Поворот и
        # перемещение самого бокса отключены: в режиме «Масштаб» пользователь
        # меняет только размер модели по выбранной оси.
        scale_widget = vtk.vtkBoxWidget()
        scale_widget.SetInteractor(self.plotter.iren.interactor)
        scale_widget.SetCurrentRenderer(self.plotter.renderer)
        scale_widget.SetPlaceFactor(1.08)
        scale_widget.SetRotationEnabled(False)
        scale_widget.SetTranslationEnabled(False)
        bounds = tuple(float(value) for value in mesh.bounds.ravel())
        scale_widget.PlaceWidget(*bounds)
        scale_widget.SetTransform(_numpy_to_vtk_transform(np.eye(4, dtype=np.float64)))
        scale_widget.SetHandleSize(0.015)
        scale_widget.GetHandleProperty().SetColor(0.95, 0.76, 0.22)
        scale_widget.GetSelectedHandleProperty().SetColor(1.0, 0.92, 0.35)
        scale_widget.GetFaceProperty().SetOpacity(0.04)
        scale_widget.GetSelectedFaceProperty().SetOpacity(0.14)
        scale_widget.GetOutlineProperty().SetColor(0.95, 0.76, 0.22)
        scale_widget.GetSelectedOutlineProperty().SetColor(1.0, 0.92, 0.35)
        scale_widget.AddObserver(vtk.vtkCommand.InteractionEvent, self._on_scale_interaction)
        scale_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent, self._on_scale_release)
        scale_widget.Off()
        self._scale_widget = scale_widget

    def _remove_transform_widgets(self) -> None:
        if self._affine_widget is not None:
            self._affine_widget.remove()
            self._affine_widget = None
            self._affine_enabled = False
        if self._scale_widget is not None:
            self._scale_widget.Off()
            self._scale_widget.RemoveAllObservers()
            self._scale_widget = None

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

    def _on_affine_release(self, matrix: np.ndarray) -> None:
        self._emit_transform_changed(np.asarray(matrix, dtype=np.float64))

    def _scale_widget_matrix(self) -> np.ndarray | None:
        if self._scale_widget is None:
            return None
        transform = vtk.vtkTransform()
        self._scale_widget.GetTransform(transform)
        return _vtk_transform_to_numpy(transform)

    def _on_scale_interaction(self, _widget: vtk.vtkBoxWidget, _event: str) -> None:
        matrix = self._scale_widget_matrix()
        if matrix is None or self._model_actor is None:
            return
        self._model_actor.user_matrix = matrix
        self.plotter.render()

    def _on_scale_release(self, _widget: vtk.vtkBoxWidget, _event: str) -> None:
        matrix = self._scale_widget_matrix()
        if matrix is not None:
            self._emit_transform_changed(matrix)

    def _emit_transform_changed(self, matrix: np.ndarray) -> None:
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return
        self.transformChanged.emit(matrix.copy())
