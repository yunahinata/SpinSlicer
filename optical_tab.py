"""Interactive optical-bench tab for the SpinSlicer desktop application."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from constants import VAT_HEIGHT_RATIO
from i18n import tr
from optical_frame_set import build_simulated_frame_set
from optical_simulation import (
    MATERIAL_PRESETS,
    RESIN_PRESETS,
    OpticalScene,
    OpticalSimulationResult,
    make_scene_with_indices,
    make_test_projection_frame,
    material_preset,
    resin_preset,
    simulate_optical_projection,
    warp_projection_frame,
)
from widgets import LabeledSlider


class OpticalPreviewWidget(QWidget):
    """Paint a compact ray diagram and the one-dimensional dose mapping."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._scene: OpticalScene | None = None
        self._result: OpticalSimulationResult | None = None
        self._baseline: OpticalSimulationResult | None = None
        self._show_rays = True
        # Keep enough room for both the ray diagram and dose graph while
        # allowing the frame-preview panel below to remain visible on compact
        # windows.  The previous 560 px minimum made the QSplitter compress
        # the two siblings into an overlapping layout.
        self.setMinimumSize(720, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(
        self,
        scene: OpticalScene,
        result: OpticalSimulationResult,
        baseline: OpticalSimulationResult,
        show_rays: bool,
    ) -> None:
        self._scene = scene
        self._result = result
        self._baseline = baseline
        self._show_rays = show_rays
        self.update()

    def clear(self) -> None:
        self._scene = None
        self._result = None
        self._baseline = None
        self.update()

    @staticmethod
    def _world_to_pixel(
        x: float,
        y: float,
        rect: QRectF,
        x_min: float,
        x_max: float,
        y_extent: float,
    ) -> QPointF:
        px = rect.left() + (x - x_min) / max(x_max - x_min, 1e-9) * rect.width()
        py = rect.center().y() - y / max(y_extent, 1e-9) * rect.height() / 2.0
        return QPointF(px, py)

    @staticmethod
    def _draw_profile(
        painter: QPainter,
        profile: np.ndarray,
        rect: QRectF,
        color: QColor,
        style: Qt.PenStyle = Qt.PenStyle.SolidLine,
    ) -> None:
        if profile.size < 2:
            return
        path = QPainterPath()
        for index, value in enumerate(profile):
            x = rect.left() + index / (profile.size - 1) * rect.width()
            y = rect.bottom() - float(np.clip(value, 0.0, 1.0)) * rect.height()
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(color, 1.8, style))
        painter.drawPath(path)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#11151d"))

        scene = self._scene
        result = self._result
        baseline = self._baseline
        if scene is None or result is None or baseline is None:
            painter.setPen(QColor("#9aa4b6"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Нажмите «Рассчитать», чтобы увидеть ход лучей и проекцию.",
            )
            painter.end()
            return

        margin = 18.0
        diagram_rect = QRectF(
            margin,
            margin,
            max(10.0, self.width() - 2.0 * margin),
            max(10.0, self.height() * 0.56 - margin),
        )
        graph_rect = QRectF(
            margin + 42.0,
            self.height() * 0.66,
            max(10.0, self.width() - 2.0 * margin - 42.0),
            max(10.0, self.height() * 0.25),
        )

        outer_half = scene.aquarium_outer_half_side_mm
        x_min = -scene.projector_distance_mm * 1.04
        x_max = max(outer_half + 18.0, scene.vat_outer_radius_mm + 18.0)
        y_extent = max(outer_half * 1.12, scene.vat_outer_radius_mm * 1.25, 10.0)

        # Section background and labels.
        painter.setPen(QPen(QColor("#343e51"), 1.0))
        painter.drawRect(diagram_rect)
        painter.setPen(QColor("#b8c3d6"))
        painter.drawText(
            QPointF(diagram_rect.left() + 10, diagram_rect.top() + 18),
            "Сечение оптического стенда",
        )

        def point(x: float, y: float) -> QPointF:
            return self._world_to_pixel(x, y, diagram_rect, x_min, x_max, y_extent)

        # Aquarium: draw the outside wall first, then the selected liquid (or air).
        tank_outer = QRectF(
            point(-outer_half, outer_half),
            point(outer_half, -outer_half),
        ).normalized()
        tank_inner_half = scene.aquarium_inner_half_side_mm
        tank_inner = QRectF(
            point(-tank_inner_half, tank_inner_half),
            point(tank_inner_half, -tank_inner_half),
        ).normalized()
        if scene.include_aquarium_walls:
            painter.setPen(QPen(QColor("#83a7c9"), 1.2))
            painter.setBrush(QBrush(QColor(55, 103, 146, 90)))
            painter.drawRect(tank_outer)
        painter.setPen(QPen(QColor("#74a9d5"), 1.0))
        liquid_color = QColor(48, 125, 185, 72) if scene.compensator_active else QColor(125, 135, 150, 45)
        painter.setBrush(QBrush(liquid_color))
        painter.drawRect(tank_inner)

        # Main cylindrical vat and the resin volume in the central section.
        center = point(0.0, 0.0)
        px_radius = abs(point(scene.vat_outer_radius_mm, 0.0).x() - center.x())
        px_inner_radius = abs(point(scene.vat_inner_radius_mm, 0.0).x() - center.x())
        painter.setPen(QPen(QColor("#d5b982"), 1.4))
        painter.setBrush(QBrush(QColor(190, 151, 86, 90)))
        painter.drawEllipse(center, px_radius, px_radius)
        painter.setPen(QPen(QColor("#e6a64d"), 1.0))
        painter.setBrush(QBrush(QColor(218, 135, 53, 64)))
        painter.drawEllipse(center, px_inner_radius, px_inner_radius)

        # The target plane is intentionally inside the resin, where the
        # central-slice mapping is read by the dose graph below.
        target_x = point(scene.target_plane_x_mm, 0.0).x()
        painter.setPen(QPen(QColor("#e8eef8"), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(target_x, diagram_rect.top()), QPointF(target_x, diagram_rect.bottom()))

        if self._show_rays:
            step = max(1, len(result.traces) // 90)
            for index in range(0, len(result.traces), step):
                trace = result.traces[index]
                if not trace.segments:
                    continue
                if trace.total_internal_reflection:
                    color = QColor(224, 92, 92, 180)
                elif trace.clipped_by_aperture:
                    color = QColor(135, 145, 160, 100)
                else:
                    color = QColor(108, 226, 187, 135)
                painter.setPen(QPen(color, 1.1))
                for segment in trace.segments:
                    painter.drawLine(
                        point(*segment.start),
                        point(*segment.end),
                    )

        # Projector and section labels.
        projector = point(-scene.projector_distance_mm, 0.0)
        painter.setPen(QPen(QColor("#e8eef8"), 1.2))
        painter.drawLine(projector + QPointF(0, -14), projector + QPointF(0, 14))
        painter.drawText(QPointF(projector.x() - 26, projector.y() - 20), "Проектор")
        painter.setPen(QColor("#8ebbe4"))
        painter.drawText(
            QPointF(point(-outer_half + 2.0, outer_half).x(), point(0.0, outer_half).y() - 6),
            f"{scene.water.name if scene.compensator_active else scene.ambient.name}; "
            f"стенка {scene.aquarium_wall.name}, {scene.aquarium_wall_thickness_mm:.2f} мм",
        )
        painter.setPen(QColor("#e6bb72"))
        painter.drawText(
            QPointF(
                point(-scene.vat_outer_radius_mm, -scene.vat_outer_radius_mm).x(),
                point(0.0, -scene.vat_outer_radius_mm).y() + 16,
            ),
            "колба / смола",
        )
        painter.setPen(QColor("#e8eef8"))
        painter.drawText(QPointF(target_x + 5, diagram_rect.top() + 38), "плоскость модели")

        # Dose graph.
        painter.setPen(QPen(QColor("#343e51"), 1.0))
        painter.drawRect(graph_rect)
        painter.setPen(QColor("#9aa4b6"))
        painter.drawText(QPointF(4.0, graph_rect.top() + 12.0), "Доза")
        painter.drawText(QPointF(graph_rect.left(), graph_rect.bottom() + 18.0), "левый край")
        painter.drawText(QPointF(graph_rect.right() - 55.0, graph_rect.bottom() + 18.0), "правый край")
        self._draw_profile(painter, result.input_profile, graph_rect, QColor("#5d91ef"))
        self._draw_profile(painter, result.output_profile, graph_rect, QColor("#6ce2bb"))
        self._draw_profile(
            painter,
            baseline.output_profile,
            graph_rect,
            QColor("#e6a64d"),
            Qt.PenStyle.DashLine,
        )
        dose_scale = max(
            scene.resin_critical_exposure_mj_cm2,
            float(np.max(result.dose_profile_mj_cm2)) if result.dose_profile_mj_cm2.size else 0.0,
        )
        self._draw_profile(
            painter,
            result.dose_profile_mj_cm2 / max(dose_scale, 1e-9),
            graph_rect,
            QColor("#d88cff"),
        )
        legend_y = graph_rect.top() - 9.0
        for label, color, x in (
            ("вход", QColor("#5d91ef"), graph_rect.left()),
            ("с жидкостью", QColor("#6ce2bb"), graph_rect.left() + 86.0),
            ("без жидкости", QColor("#e6a64d"), graph_rect.left() + 190.0),
            ("доза / Ec", QColor("#d88cff"), graph_rect.left() + 300.0),
        ):
            painter.setPen(QPen(color, 2.0))
            painter.drawLine(QPointF(x, legend_y), QPointF(x + 18.0, legend_y))
            painter.setPen(color)
            painter.drawText(QPointF(x + 24.0, legend_y + 4.0), label)
        painter.end()


class OpticalSimulationTab(QWidget):
    """Tune the projector/vat/liquid geometry and resin exposure estimate."""

    progress = pyqtSignal(float, str)
    logMessage = pyqtSignal(str)
    simulationReady = pyqtSignal(object)
    simulationFramesReady = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._output_dir: Optional[str] = None
        self._source_profile: np.ndarray | None = None
        self._source_frame: np.ndarray | None = make_test_projection_frame()
        self._water_frame: np.ndarray | None = None
        self._baseline_frame: np.ndarray | None = None
        self._source_label = "Тестовый паттерн"
        self._latest_frame_path: Optional[str] = None
        self._result: OpticalSimulationResult | None = None
        self._baseline: OpticalSimulationResult | None = None
        self._last_scene: OpticalScene | None = None

        self._simulation_timer = QTimer(self)
        self._simulation_timer.setSingleShot(True)
        self._simulation_timer.timeout.connect(self._run_simulation)

        self._build_ui()
        self.resin_profile.setCurrentIndex(self.resin_profile.findData("water_dry_run"))
        self._apply_material_preset(self.aquarium_material, self.tank_ior)
        self._apply_material_preset(
            self.compensator_material,
            self.water_ior,
            self.water_absorption,
        )
        self._apply_material_preset(self.vat_material, self.vat_ior)
        self._apply_resin_preset()
        self._wire_signals()
        self._schedule_simulation()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(10, 8, 10, 8)
        content_layout.setSpacing(10)

        title = QLabel("Оптический стенд")
        title.setObjectName("panelTitle")
        content_layout.addWidget(title)

        quick_hint = QLabel(
            "Можно начать без смолы и без настройки чисел: выбрана вода в колбе. "
            "Программа проверит ход лучей и проекции. Геометрия пока примерная; "
            "настоящую колбу и проектор можно описать позже."
        )
        quick_hint.setWordWrap(True)
        quick_hint.setObjectName("hintLabel")
        content_layout.addWidget(quick_hint)
        self.advanced_toggle = QCheckBox("Подробно: размеры, материалы и параметры света")
        content_layout.addWidget(self.advanced_toggle)

        self.vat_box = QGroupBox("Основная колба")
        vat_layout = QVBoxLayout(self.vat_box)
        self.vat_diameter = LabeledSlider("Наружный диаметр колбы, мм", 20.0, 200.0, 60.0, decimals=1)
        self.vat_wall = LabeledSlider("Толщина стенки колбы, мм", 0.2, 15.0, 3.0, decimals=2)
        self.vat_height = LabeledSlider("Высота колбы, мм", 20.0, 400.0, 60.0 * VAT_HEIGHT_RATIO, decimals=1)
        self.slice_height = LabeledSlider("Высота оптического среза, мм", 1.0, 400.0, 48.0, decimals=1)
        for widget in (self.vat_diameter, self.vat_wall, self.vat_height, self.slice_height):
            vat_layout.addWidget(widget)
        content_layout.addWidget(self.vat_box)

        self.tank_box = QGroupBox("Квадратный аквариум-компенсатор")
        tank_layout = QVBoxLayout(self.tank_box)
        self.include_aquarium = QCheckBox("Учитывать стенки внешнего аквариума")
        self.include_aquarium.setChecked(True)
        tank_layout.addWidget(self.include_aquarium)
        self.use_compensator = QCheckBox("Заполнять зазор выбранной жидкостью")
        self.use_compensator.setChecked(True)
        tank_layout.addWidget(self.use_compensator)
        self.tank_side = LabeledSlider("Внутренняя сторона аквариума, мм", 40.0, 400.0, 120.0, decimals=1)
        self.tank_wall = LabeledSlider("Толщина стенки аквариума, мм", 0.2, 25.0, 5.0, decimals=2)
        self.aquarium_material = QComboBox()
        self._fill_material_combo(self.aquarium_material, ("glass", "acrylic"))
        self.aquarium_material_label = QLabel("Материал внешней стенки")
        self.aquarium_material_label.setObjectName("fieldLabel")
        self.water_level = LabeledSlider("Уровень воды от дна, мм", 1.0, 400.0, 120.0, decimals=1)
        self.compensator_material = QComboBox()
        self._fill_material_combo(self.compensator_material, ("water", "glycerin"))
        self.compensator_material_label = QLabel("Жидкость в зазоре")
        self.compensator_material_label.setObjectName("fieldLabel")
        self.tank_ior = LabeledSlider("Показатель преломления стенки n", 1.0, 2.2, 1.524, decimals=4, step=0.001)
        self.water_ior = LabeledSlider("Показатель преломления жидкости n", 1.30, 1.60, 1.3390, decimals=4, step=0.0001)
        tank_widgets: tuple[QWidget, ...] = (
            self.tank_side,
            self.tank_wall,
            self.aquarium_material_label,
            self.aquarium_material,
            self.water_level,
            self.compensator_material_label,
            self.compensator_material,
            self.tank_ior,
            self.water_ior,
        )
        for tank_widget in tank_widgets:
            tank_layout.addWidget(tank_widget)
        content_layout.addWidget(self.tank_box)

        material_box = QGroupBox("Материалы основной колбы")
        material_layout = QVBoxLayout(material_box)
        self.vat_material = QComboBox()
        self._fill_material_combo(self.vat_material, ("glass", "acrylic"))
        vat_material_label = QLabel("Материал стенки колбы")
        vat_material_label.setObjectName("fieldLabel")
        self.vat_ior = LabeledSlider("Показатель преломления стенки n", 1.0, 2.2, 1.524, decimals=4, step=0.001)
        resin_profile_label = QLabel("Содержимое колбы")
        resin_profile_label.setObjectName("fieldLabel")
        self.resin_profile = QComboBox()
        for key, preset in RESIN_PRESETS.items():
            self.resin_profile.addItem(preset.display_name, key)
        self.resin_profile.setToolTip(
            "Без материала выберите воду. Остальные профили являются только оценками."
        )
        self.resin_ior = LabeledSlider("Показатель преломления содержимого n", 1.30, 1.80, 1.49, decimals=4, step=0.001)
        self.water_absorption = LabeledSlider("Поглощение жидкости, 1/мм", 0.0, 0.01, 0.00002, decimals=5, step=0.00001)
        self.resin_absorption = LabeledSlider("Поглощение содержимого, 1/мм", 0.0, 0.10, 0.002, decimals=4, step=0.0001)
        self.resin_penetration = LabeledSlider(
            "Глубина проникновения света Dp, мм",
            0.02,
            50.0,
            0.180,
            decimals=3,
            step=0.01,
        )
        self.resin_critical = LabeledSlider(
            "Критическая экспозиция Ec, мДж/см²",
            0.1,
            1000.0,
            70.0,
            decimals=1,
            step=1.0,
        )
        material_layout.addWidget(vat_material_label)
        material_layout.addWidget(self.vat_material)
        material_layout.addWidget(resin_profile_label)
        material_layout.addWidget(self.resin_profile)
        material_layout.addWidget(self.vat_ior)
        material_layout.addWidget(self.resin_ior)
        material_layout.addWidget(self.water_absorption)
        material_layout.addWidget(self.resin_absorption)
        material_layout.addWidget(self.resin_penetration)
        material_layout.addWidget(self.resin_critical)
        content_layout.addWidget(material_box)

        self._detailed_material_widgets = (
            vat_material_label, self.vat_material, self.vat_ior, self.resin_ior,
            self.water_absorption, self.resin_absorption, self.resin_penetration,
            self.resin_critical,
        )

        self.source_parameter_box = QGroupBox("Источник и фотополимеризация")
        source_layout = QVBoxLayout(self.source_parameter_box)
        source_preset_label = QLabel("Профиль источника")
        source_preset_label.setObjectName("fieldLabel")
        self.source_preset = QComboBox()
        self.source_preset.addItem(
            "JAP45160Z-004 → матрица (сценарий по паспорту)",
            "laser_450_3lcd",
        )
        self.source_preset.addItem("Пользовательский источник", "custom")
        source_layout.addWidget(source_preset_label)
        source_layout.addWidget(self.source_preset)
        self.source_wavelength = LabeledSlider("Длина волны, нм", 365.0, 700.0, 450.0, decimals=0, step=1.0)
        self.source_power = LabeledSlider("Оптическая мощность источника, Вт", 0.0, 50.0, 1.0, decimals=2, step=0.05)
        self.exposure_time = LabeledSlider("Время экспозиции, с", 0.01, 120.0, 1.0, decimals=2, step=0.1)
        self.exposure_area = LabeledSlider("Площадь проекции, см²", 1.0, 500.0, 36.0, decimals=1, step=1.0)
        self.source_efficiency = LabeledSlider(
            "Передача всего тракта, %",
            0.0001,
            100.0,
            100.0,
            decimals=4,
            step=0.1,
        )
        for widget in (
            self.source_wavelength,
            self.source_power,
            self.exposure_time,
            self.exposure_area,
            self.source_efficiency,
        ):
            source_layout.addWidget(widget)
        self.source_efficiency_measured = QCheckBox("Интенсивность на колбе измерена")
        self.resin_curve_measured = QCheckBox("Рабочая кривая смолы при этой длине волны измерена")
        source_layout.addWidget(self.source_efficiency_measured)
        source_layout.addWidget(self.resin_curve_measured)
        source_hint = QLabel(
            "Доза сценария: P·t·η/S. Паспорт JAP45160Z-004 даёт 1,0–1,15 Вт "
            "и 440–460 нм; 100% передачи — идеальная верхняя граница, не свойство "
            "матрицы. Числа отверждения появятся лишь после отметки измерений "
            "интенсивности и Dp/Ec именно этой смолы при выбранной длине волны."
        )
        source_hint.setObjectName("hintLabel")
        source_hint.setWordWrap(True)
        source_layout.addWidget(source_hint)
        content_layout.addWidget(self.source_parameter_box)

        self.projector_box = QGroupBox("Проектор и расчёт")
        projector_layout = QVBoxLayout(self.projector_box)
        self.projector_distance = LabeledSlider("Расстояние до центра, мм", 80.0, 1200.0, 250.0, decimals=1)
        self.horizontal_fov = LabeledSlider("Поле зрения по горизонтали, °", 1.0, 70.0, 12.0, decimals=2)
        self.ray_count = LabeledSlider("Количество лучей", 33.0, 1025.0, 241.0, decimals=0, step=8.0)
        self.output_resolution = LabeledSlider("Разрешение среза, px", 64.0, 2048.0, 256.0, decimals=0, step=16.0)
        for widget in (self.projector_distance, self.horizontal_fov, self.ray_count, self.output_resolution):
            projector_layout.addWidget(widget)
        content_layout.addWidget(self.projector_box)

        source_box = QGroupBox("Источник проекции")
        source_layout = QVBoxLayout(source_box)
        self.source_label = QLabel("Источник: тестовый оптический паттерн")
        self.source_label.setObjectName("hintLabel")
        self.source_label.setWordWrap(True)
        source_layout.addWidget(self.source_label)
        source_buttons = QHBoxLayout()
        self.load_frame_btn = QPushButton("Загрузить кадр")
        self.load_frame_btn.clicked.connect(self._on_load_frame_clicked)
        self.use_latest_btn = QPushButton("Взять последний кадр")
        self.use_latest_btn.setEnabled(False)
        self.use_latest_btn.clicked.connect(self._use_latest_frame)
        source_buttons.addWidget(self.load_frame_btn)
        source_buttons.addWidget(self.use_latest_btn)
        source_layout.addLayout(source_buttons)
        self.show_rays = QCheckBox("Показывать ход лучей")
        self.show_rays.setChecked(True)
        source_layout.addWidget(self.show_rays)
        self.auto_update = QCheckBox("Пересчитывать автоматически при изменении")
        self.auto_update.setChecked(True)
        source_layout.addWidget(self.auto_update)
        content_layout.addWidget(source_box)

        self.simulate_btn = QPushButton("🔭 Рассчитать и построить модель")
        self.simulate_btn.setObjectName("generateButton")
        self.simulate_btn.setToolTip(
            "Применить оптику к каждому кадру, учесть толщину стенок и построить модель"
        )
        self.simulate_btn.clicked.connect(self._on_simulate_clicked)
        content_layout.addWidget(self.simulate_btn)

        self.simulation_summary = QLabel("Расчёт ещё не запускался.")
        self.simulation_summary.setObjectName("hintLabel")
        self.simulation_summary.setWordWrap(True)
        content_layout.addWidget(self.simulation_summary)

        hint = QLabel(
            "Расчёт показывает центральный горизонтальный срез: преломление на стенках, "
            "выбранной жидкости и смоле, потери Френеля, геометрическую ошибку и "
            "оценку дозы/глубины отверждения. Для точной настройки перед печатью "
            "понадобятся измеренные n, толщины стенок, оптическая мощность и рабочая "
            "кривая конкретной смолы."
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        content_layout.addWidget(hint)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        scroll.setMinimumWidth(370)
        splitter.addWidget(scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.preview = OpticalPreviewWidget()
        right_layout.addWidget(self.preview, 1)

        frame_box = QGroupBox("2-D проекция — центральный срез")
        frame_layout = QHBoxLayout(frame_box)
        self.source_frame_view = self._create_frame_view(frame_layout, "Исходный кадр")
        self.water_frame_view = self._create_frame_view(frame_layout, "После жидкости")
        self.baseline_frame_view = self._create_frame_view(frame_layout, "Без жидкости, стенка есть")
        right_layout.addWidget(frame_box)
        right.setMinimumHeight(
            self.preview.minimumHeight()
            + frame_box.minimumSizeHint().height()
            + max(0, right_layout.spacing())
        )

        splitter.addWidget(right)
        splitter.setSizes([400, 1050])
        root.addWidget(splitter, 1)
        self.advanced_toggle.toggled.connect(self._set_advanced)
        self._set_advanced(False)

    def _set_advanced(self, visible: bool) -> None:
        for widget in (
            self.vat_box, self.tank_box, self.source_parameter_box, self.projector_box,
            *self._detailed_material_widgets,
        ):
            widget.setVisible(visible)

    @staticmethod
    def _fill_material_combo(combo: QComboBox, keys: tuple[str, ...]) -> None:
        for key in keys:
            preset = MATERIAL_PRESETS[key]
            combo.addItem(preset.display_name, key)

    @staticmethod
    def _combo_key(combo: QComboBox) -> str:
        value = combo.currentData()
        return str(value) if value is not None else ""

    def _apply_material_preset(
        self,
        combo: QComboBox,
        ior_control: LabeledSlider,
        absorption_control: LabeledSlider | None = None,
    ) -> None:
        preset = material_preset(self._combo_key(combo))
        ior_control.setValue(preset.refractive_index)
        if absorption_control is not None:
            absorption_control.setValue(preset.absorption_per_mm)

    def _apply_resin_preset(self) -> None:
        preset = resin_preset(self._combo_key(self.resin_profile))
        self.resin_ior.setValue(preset.refractive_index)
        self.resin_absorption.setValue(preset.absorption_per_mm)
        self.resin_penetration.setValue(preset.penetration_depth_mm)
        self.resin_critical.setValue(preset.critical_exposure_mj_cm2)

    def _apply_source_preset(self) -> None:
        if self._combo_key(self.source_preset) != "laser_450_3lcd":
            return
        self.source_wavelength.setValue(450.0)
        self.source_power.setValue(1.0)
        self.exposure_area.setValue(36.0)
        self.source_efficiency.setValue(100.0)
        self.source_efficiency_measured.setChecked(False)
        self.resin_curve_measured.setChecked(False)

    @staticmethod
    def _create_frame_view(layout: QHBoxLayout, title: str) -> QLabel:
        column = QVBoxLayout()
        caption = QLabel(title)
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption.setObjectName("hintLabel")
        column.addWidget(caption)
        image = QLabel("Нет загруженного кадра")
        image.setObjectName("videoPreview")
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image.setMinimumSize(120, 92)
        image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        column.addWidget(image, 1)
        layout.addLayout(column, 1)
        return image

    def _wire_signals(self) -> None:
        controls = (
            self.vat_diameter,
            self.vat_wall,
            self.vat_height,
            self.slice_height,
            self.tank_side,
            self.tank_wall,
            self.water_level,
            self.tank_ior,
            self.water_ior,
            self.vat_ior,
            self.resin_ior,
            self.water_absorption,
            self.resin_absorption,
            self.resin_penetration,
            self.resin_critical,
            self.projector_distance,
            self.horizontal_fov,
            self.ray_count,
            self.output_resolution,
            self.source_wavelength,
            self.source_power,
            self.exposure_time,
            self.exposure_area,
            self.source_efficiency,
        )
        for control in controls:
            control.valueChanged.connect(lambda _value: self._schedule_simulation())
        self.include_aquarium.toggled.connect(lambda _value: self._schedule_simulation())
        self.use_compensator.toggled.connect(lambda _value: self._schedule_simulation())
        self.aquarium_material.currentIndexChanged.connect(
            lambda _index: self._on_material_changed(
                self.aquarium_material,
                self.tank_ior,
            )
        )
        self.compensator_material.currentIndexChanged.connect(
            lambda _index: self._on_material_changed(
                self.compensator_material,
                self.water_ior,
                self.water_absorption,
            )
        )
        self.vat_material.currentIndexChanged.connect(
            lambda _index: self._on_material_changed(
                self.vat_material,
                self.vat_ior,
            )
        )
        self.resin_profile.currentIndexChanged.connect(lambda _index: self._on_resin_changed())
        self.source_preset.currentIndexChanged.connect(lambda _index: self._on_source_changed())
        self.source_efficiency_measured.toggled.connect(lambda _value: self._schedule_simulation())
        self.resin_curve_measured.toggled.connect(lambda _value: self._schedule_simulation())
        self.show_rays.toggled.connect(lambda _value: self._refresh_preview())

    def _on_material_changed(
        self,
        combo: QComboBox,
        ior_control: LabeledSlider,
        absorption_control: LabeledSlider | None = None,
    ) -> None:
        self._apply_material_preset(combo, ior_control, absorption_control)
        self._schedule_simulation()

    def _on_resin_changed(self) -> None:
        self._apply_resin_preset()
        optical_only = self._combo_key(self.resin_profile) == "water_dry_run"
        self.resin_penetration.setEnabled(not optical_only)
        self.resin_critical.setEnabled(not optical_only)
        self.resin_curve_measured.setEnabled(not optical_only)
        if optical_only:
            self.resin_curve_measured.setChecked(False)
        self._schedule_simulation()

    def _on_source_changed(self) -> None:
        self._apply_source_preset()
        self._schedule_simulation()

    def set_vat_diameter(self, value: float) -> None:
        if abs(self.vat_diameter.value() - float(value)) > 1e-6:
            self.vat_diameter.setValue(float(value))
            self._schedule_simulation()

    def apply_source_assumptions(
        self,
        wavelength_nm: float,
        power_w: float,
        area_cm2: float,
        transmission_pct: float,
        measured: bool,
    ) -> None:
        """Load a light-budget scenario while retaining its measurement status."""

        self.source_preset.setCurrentIndex(self.source_preset.findData("custom"))
        self.source_wavelength.setValue(wavelength_nm)
        self.source_power.setValue(power_w)
        self.exposure_area.setValue(area_cm2)
        self.source_efficiency.setValue(transmission_pct)
        self.source_efficiency_measured.setChecked(measured)
        self._schedule_simulation()

    def set_output_dir(self, path: str) -> None:
        if path != self._output_dir:
            self._source_profile = None
            self._source_frame = make_test_projection_frame()
            self._water_frame = None
            self._baseline_frame = None
            self._source_label = "Тестовый паттерн"
            self._result = None
            self._baseline = None
            self._last_scene = None
            self._latest_frame_path = None
            self.preview.clear()
            self.source_label.setText("Источник: тестовый оптический паттерн")
            self.simulation_summary.setText("Расчёт ещё не запускался.")
            self._refresh_frame_images()
        self._output_dir = path
        candidate = self._find_latest_frame(path)
        self._latest_frame_path = candidate
        if candidate:
            self.use_latest_btn.setEnabled(True)
            self.source_label.setText(
                f"Источник: тестовый паттерн. Доступен последний кадр: {os.path.basename(candidate)}"
            )
            self._load_source_frame(candidate)
        else:
            self.use_latest_btn.setEnabled(False)

    @staticmethod
    def _find_latest_frame(path: str) -> str | None:
        root = Path(path)
        if not root.is_dir():
            return None
        direct = sorted(root.glob("frame_*.png"))
        if direct:
            return str(direct[-1])
        nested = sorted(root.glob("**/frame_*.png"))
        return str(nested[-1]) if nested else None

    def _on_load_frame_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Выбрать кадр"),
            self._output_dir or "",
            tr("Изображения (*.png *.jpg *.jpeg *.bmp)"),
        )
        if path:
            self._load_source_frame(path)

    def _use_latest_frame(self) -> None:
        if self._latest_frame_path:
            self._load_source_frame(self._latest_frame_path)

    def _load_source_frame(self, path: str) -> None:
        try:
            with Image.open(path) as image:
                gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        except Exception as exc:
            QMessageBox.critical(self, tr("Ошибка"), f"Не удалось прочитать кадр:\n{exc}")
            return
        if gray.ndim != 2 or gray.shape[1] < 2:
            QMessageBox.warning(self, tr("Внимание"), "Кадр должен быть двумерным изображением.")
            return
        band = max(1, gray.shape[0] // 20)
        center = gray.shape[0] // 2
        profile = np.mean(gray[max(0, center - band):min(gray.shape[0], center + band + 1)], axis=0)
        self._source_profile = np.clip(profile, 0.0, 1.0)
        self._source_frame = np.clip(gray, 0.0, 1.0)
        self._source_label = os.path.basename(path)
        self.source_label.setText(
            f"Источник: центральная полоса кадра «{self._source_label}»"
        )
        self._schedule_simulation()

    def _schedule_simulation(self) -> None:
        if self.auto_update.isChecked():
            self._simulation_timer.start(140)

    def _on_simulate_clicked(self, _checked: bool = False) -> None:
        if not self._run_simulation(emit_status=True):
            return
        if self._output_dir is None or self._result is None or self._last_scene is None:
            self.logMessage.emit(
                "Оптический расчёт завершён, но папка исходных проекций не выбрана."
            )
            return

        try:
            simulated_dir = build_simulated_frame_set(
                self._output_dir,
                self._result,
                self._last_scene,
                source_name=self._source_label,
                progress_cb=lambda fraction, message: self.progress.emit(fraction, message),
                apply_resin_dose=(
                    self.source_efficiency_measured.isChecked()
                    and self.resin_curve_measured.isChecked()
                    and self._combo_key(self.resin_profile) != "water_dry_run"
                ),
            )
        except Exception as exc:
            self.simulation_summary.setText(
                f"Оптика рассчитана, но набор кадров для модели не создан: {exc}"
            )
            self.logMessage.emit(f"Набор оптических проекций не создан: {exc}")
            self.progress.emit(0.0, "Не удалось подготовить кадры для реконструкции.")
            return

        self.simulation_summary.setText(
            self.simulation_summary.text()
            + f"\nСимулированные кадры сохранены: {simulated_dir}\n"
            "Модель будет построена по ним автоматически."
        )
        self.logMessage.emit(f"Симулированные кадры сохранены в: {simulated_dir}")
        self.simulationFramesReady.emit(simulated_dir)

    def _scene_from_controls(self) -> OpticalScene:
        scene = make_scene_with_indices(
            vat_diameter_mm=self.vat_diameter.value(),
            vat_wall_thickness_mm=self.vat_wall.value(),
            aquarium_inner_side_mm=self.tank_side.value(),
            aquarium_wall_thickness_mm=self.tank_wall.value(),
            water_level_mm=self.water_level.value(),
            include_aquarium_walls=self.include_aquarium.isChecked(),
            use_water_compensator=self.use_compensator.isChecked(),
            projector_distance_mm=self.projector_distance.value(),
            horizontal_fov_deg=self.horizontal_fov.value(),
            ray_count=self.ray_count.value_int(),
            output_resolution=self.output_resolution.value_int(),
            aquarium_wall_material_key=self._combo_key(self.aquarium_material),
            compensator_material_key=self._combo_key(self.compensator_material),
            vat_wall_material_key=self._combo_key(self.vat_material),
            resin_profile_key=self._combo_key(self.resin_profile),
            aquarium_wall_ior=self.tank_ior.value(),
            water_ior=self.water_ior.value(),
            vat_wall_ior=self.vat_ior.value(),
            resin_ior=self.resin_ior.value(),
            water_absorption_per_mm=self.water_absorption.value(),
            resin_absorption_per_mm=self.resin_absorption.value(),
            source_name=self.source_preset.currentText(),
            wavelength_nm=self.source_wavelength.value(),
            source_power_w=self.source_power.value(),
            exposure_time_s=self.exposure_time.value(),
            exposure_area_cm2=self.exposure_area.value(),
            source_efficiency=self.source_efficiency.value() / 100.0,
            resin_penetration_depth_mm=self.resin_penetration.value(),
            resin_critical_exposure_mj_cm2=self.resin_critical.value(),
        )
        return replace(
            scene,
            vat_height_mm=self.vat_height.value(),
            optical_slice_height_mm=min(self.slice_height.value(), self.vat_height.value()),
        )

    def _run_simulation(self, emit_status: bool = False) -> bool:
        self._simulation_timer.stop()
        try:
            scene = self._scene_from_controls()
            scene.validate()
            profile = self._source_profile
            result = simulate_optical_projection(scene, profile)
            baseline = simulate_optical_projection(scene.without_compensator(), profile)
        except Exception as exc:
            self._result = None
            self._baseline = None
            self.preview.clear()
            self.simulation_summary.setText(f"Расчёт не выполнен: {exc}")
            self.logMessage.emit(f"Оптический расчёт не выполнен: {exc}")
            if emit_status:
                self.progress.emit(0.0, "Оптический расчёт не выполнен.")
            return False

        self._result = result
        self._baseline = baseline
        self._last_scene = scene
        if self._source_frame is None:
            self._source_frame = make_test_projection_frame()
        self._water_frame = warp_projection_frame(self._source_frame, result)
        self._baseline_frame = warp_projection_frame(self._source_frame, baseline)
        self.simulation_summary.setText(self._format_simulation_summary(scene, result, baseline))
        self._refresh_preview()
        self.simulationReady.emit(result)
        if emit_status:
            self.progress.emit(1.0, "Оптический расчёт завершён.")
        return True

    def _format_simulation_summary(
        self,
        scene: OpticalScene,
        result: OpticalSimulationResult,
        baseline: OpticalSimulationResult,
    ) -> str:
        def metric(value: float) -> str:
            return "—" if not np.isfinite(value) else f"{value:.2f} мм"

        compensator = "активен" if scene.compensator_active else "неактивен"
        wall_state = "включены" if scene.include_aquarium_walls else "выключены"
        lines = [
            f"Покрытие лучами: {result.coverage_pct:.1f}%; "
            f"среднее пропускание: {result.mean_transmission_pct:.1f}%.",
            f"Ошибка отображения: {metric(result.mapping_error_mean_mm)} "
            f"(P95 {metric(result.mapping_error_p95_mm)}, максимум {metric(result.mapping_error_max_mm)}).",
            f"Внешний аквариум: стенки {wall_state}, {scene.aquarium_wall.name}, "
            f"толщина {scene.aquarium_wall_thickness_mm:.2f} мм; "
            f"средний путь в стенке {result.aquarium_wall_path_mean_mm:.2f} мм.",
            f"Компенсатор: {compensator} ({scene.water.name if scene.compensator_active else scene.ambient.name}).",
            f"Источник сценария: {scene.wavelength_nm:.0f} нм, {scene.source_power_w:.2f} Вт, "
            f"{scene.exposure_time_s:.2f} с, передача {scene.source_efficiency * 100.0:.4g}%.",
        ]
        if self.source_efficiency_measured.isChecked():
            lines.append(
                f"По введённой измеренной передаче: входная экспозиция "
                f"{result.source_radiant_exposure_mj_cm2:.2f} мДж/см², "
                f"доза в плоскости {result.mean_target_dose_mj_cm2:.2f} мДж/см²."
            )
        else:
            lines.append(
                "Доза на колбе не показана: нужна измеренная передача всего тракта "
                "или интенсивность в плоскости колбы."
            )
        if scene.resin_profile_key == "water_dry_run":
            lines.append(
                "В колбе вода: рассчитаны только лучи, потери и искажения. "
                "Химическое отверждение к этому сценарию не применяется."
            )
        elif self.source_efficiency_measured.isChecked() and self.resin_curve_measured.isChecked():
            lines.append(
                f"По заданной рабочей кривой: Dp {scene.resin_penetration_depth_mm:.3f} мм, "
                f"Ec {scene.resin_critical_exposure_mj_cm2:.1f} мДж/см²; "
                f"отверждение {result.cured_fraction_pct:.1f}%, "
                f"оценочная глубина {result.max_cure_depth_mm:.3f} мм. "
                "Это оценка среза, а не объёмной CAL-печати."
            )
        else:
            lines.append(
                "Отверждение не показано: нужны интенсивность на колбе и Dp/Ec "
                "конкретной смолы при этой длине волны."
            )
        if not (
            self.source_efficiency_measured.isChecked()
            and self.resin_curve_measured.isChecked()
            and scene.resin_profile_key != "water_dry_run"
        ):
            lines.append("Кадры для реконструкции учитывают геометрию и оптические потери без химического весового множителя.")
        if np.isfinite(result.mapping_error_mean_mm) and np.isfinite(baseline.mapping_error_mean_mm):
            delta = baseline.mapping_error_mean_mm - result.mapping_error_mean_mm
            lines.append(f"Изменение средней ошибки относительно варианта без жидкости: {delta:+.2f} мм.")
        warnings = scene.warnings()
        if warnings:
            lines.append("Предупреждение: " + " ".join(warnings))
        return "\n".join(lines)

    def _refresh_preview(self) -> None:
        if self._result is None or self._baseline is None:
            return
        try:
            scene = self._scene_from_controls()
        except Exception:
            return
        self.preview.set_data(scene, self._result, self._baseline, self.show_rays.isChecked())
        self._refresh_frame_images()

    def _refresh_frame_images(self) -> None:
        self._set_frame_view(self.source_frame_view, self._source_frame)
        self._set_frame_view(self.water_frame_view, self._water_frame)
        self._set_frame_view(self.baseline_frame_view, self._baseline_frame)

    @staticmethod
    def _set_frame_view(label: QLabel, frame: np.ndarray | None) -> None:
        if frame is None or label.width() < 2 or label.height() < 2:
            label.clear()
            label.setText("Нет загруженного кадра")
            return
        image_array = np.ascontiguousarray(np.clip(frame, 0.0, 1.0) * 255.0, dtype=np.uint8)
        height, width = image_array.shape
        image = QImage(
            image_array.tobytes(),
            width,
            height,
            width,
            QImage.Format.Format_Grayscale8,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        super().resizeEvent(event)
        self._refresh_frame_images()
