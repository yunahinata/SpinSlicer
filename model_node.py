"""
model_node.py
=============
ModelNode хранит:
  • original_mesh — оригинальный trimesh.Trimesh, ЦЕНТРИРОВАННЫЙ по bounding
    box при загрузке и БОЛЬШЕ НИКОГДА не изменяемый напрямую;
  • transform — отдельную матрицу трансформации (Scale, Rotation, Translation).

Трансформация применяется только в двух местах:
  1. Во вьюпорте — как actor.user_matrix (GPU-трансформация, без пересчёта
     вершин => никаких лагов при вращении/масштабировании);
  2. Перед генерацией — на КОПИИ меша, непосредственно перед передачей
     в SlicingEngine.

Масштаб всегда выражается в физических размерах (мм), а не абстрактным
множителем 0.1x–3.0x: пользователь задаёт целевой размер по оси, ModelNode
сам считает нужный коэффициент масштабирования от исходного bounding box.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
import trimesh.transformations as tf


@dataclass
class Transform:
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float64))
    rotation_deg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def reset(self) -> None:
        self.scale = np.ones(3, dtype=np.float64)
        self.rotation_deg = np.zeros(3, dtype=np.float64)
        self.translation = np.zeros(3, dtype=np.float64)

    def matrix(self) -> np.ndarray:
        """4x4 матрица: T * R * S (масштаб -> поворот -> сдвиг)."""
        s = np.eye(4)
        s[0, 0], s[1, 1], s[2, 2] = self.scale

        rx, ry, rz = np.radians(self.rotation_deg)
        r = tf.euler_matrix(rx, ry, rz, axes="sxyz")

        t = tf.translation_matrix(self.translation)
        return t @ r @ s


class ModelNode:
    """Обёртка над одной загруженной моделью в сцене."""

    def __init__(self, mesh: trimesh.Trimesh, source_path: str = ""):
        self.original_mesh = mesh
        self.source_path = source_path
        self.transform = Transform()

    # --- геометрия -------------------------------------------------------
    @property
    def base_extents(self) -> np.ndarray:
        """Размеры исходного (немасштабированного) bounding box, мм."""
        return self.original_mesh.extents.astype(np.float64)

    @property
    def vertex_count(self) -> int:
        return int(len(self.original_mesh.vertices))

    @property
    def face_count(self) -> int:
        return int(len(self.original_mesh.faces))

    # --- трансформация -----------------------------------------------------
    def matrix(self) -> np.ndarray:
        return self.transform.matrix()

    def current_size_mm(self) -> np.ndarray:
        return self.base_extents * self.transform.scale

    def set_size_mm(self, x: float | None = None, y: float | None = None,
                     z: float | None = None, uniform: bool = False) -> None:
        """Задаёт целевой физический размер по одной или нескольким осям.

        Если uniform=True — коэффициент масштаба, посчитанный для изменённой
        оси, применяется одинаково ко всем трём осям (пропорциональный
        масштаб), а не подгоняет остальные оси под точный мм-размер.
        """
        base = self.base_extents
        safe_base = np.where(base > 1e-9, base, 1.0)
        target = self.current_size_mm().copy()

        changed_axis = None
        if x is not None:
            target[0] = x
            changed_axis = 0
        if y is not None:
            target[1] = y
            changed_axis = 1
        if z is not None:
            target[2] = z
            changed_axis = 2

        new_scale = target / safe_base

        if uniform and changed_axis is not None:
            new_scale[:] = new_scale[changed_axis]

        self.transform.scale = new_scale

    def set_rotation_deg(self, x: float | None = None, y: float | None = None,
                          z: float | None = None) -> None:
        r = self.transform.rotation_deg
        if x is not None:
            r[0] = x
        if y is not None:
            r[1] = y
        if z is not None:
            r[2] = z

    def set_translation_mm(self, x: float | None = None, y: float | None = None,
                            z: float | None = None) -> None:
        t = self.transform.translation
        if x is not None:
            t[0] = x
        if y is not None:
            t[1] = y
        if z is not None:
            t[2] = z

    def fit_to_diameter(self, diameter_mm: float, fill_fraction: float) -> None:
        """Авто-масштабирование под размер колбы (по X/Y), сохраняя пропорции."""
        base = self.base_extents
        max_xy = max(base[0], base[1])
        if max_xy <= 1e-9:
            return
        s = (diameter_mm * fill_fraction) / max_xy
        self.transform.scale = np.array([s, s, s], dtype=np.float64)

    def center_xy(self) -> None:
        self.transform.translation[0] = 0.0
        self.transform.translation[1] = 0.0

    def reset(self) -> None:
        self.transform.reset()

    def get_transformed_mesh(self) -> trimesh.Trimesh:
        """Копия меша с окончательно применённой трансформацией.

        Используется ИСКЛЮЧИТЕЛЬНО перед генерацией проекций — вьюпорт
        трансформирует ту же геометрию на GPU через user_matrix, не трогая
        original_mesh.
        """
        mesh = self.original_mesh.copy()
        mesh.apply_transform(self.matrix())
        return mesh
