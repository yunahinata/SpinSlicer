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


class Viewport3D(QWidget):
    """Qt-виджет с осями сцены: колба + активная модель."""

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
        self._current_diameter = 0.0

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
        self.plotter.render()

    def update_model_transform(self, matrix_4x4: np.ndarray) -> None:
        """GPU-трансформация модели — без пересчёта вершин на CPU."""
        if self._model_actor is None:
            return
        self._model_actor.user_matrix = matrix_4x4
        self.plotter.render()

    def clear_model(self) -> None:
        if self._model_actor is not None:
            self.plotter.remove_actor(self._model_actor, render=False)
            self._model_actor = None
            self.plotter.render()
