"""Guided resin-formulation, screening, and comparison workspace."""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QPointF, QStandardPaths, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from resin_simulation import (
    COMPONENT_CATEGORIES,
    DEFAULT_MODEL_REGISTRY,
    Component,
    ModelParameters,
    ProcessConditions,
    Recipe,
    SimulationResult,
    calculate_recipe,
    composition_table,
    default_components,
    export_recipe,
    load_library,
    save_library,
    starter_recipes,
)

TASKS: dict[str, tuple[str, str]] = {
    "flexible": ("Гибкая деталь", "Оцените влияние гибких олигомеров и пластификаторов. Удлинение и модуль останутся предварительными оценками."),
    "rigid": ("Жёсткая деталь", "Сравните основу и реактивные разбавители по вязкости, усадке и индикаторам механики."),
    "detail": ("Высокая детализация", "Начните с тонкого слоя и низкой вязкости. Проверьте, хватает ли дозы для выбранной толщины."),
    "fast": ("Быстрое отверждение", "Подберите сочетание длины волны, интенсивности и фотоинициатора. Проверьте спектральное совпадение."),
    "experimental": ("Экспериментальный состав", "Создайте вариант и сравнивайте его только при одинаковых настройках процесса."),
}


PROPERTY_FIELDS: tuple[tuple[str, str, str, float, float, int], ...] = (
    ("density_g_ml", "Плотность", "г/мл", 1.0, 10.0, 3),
    ("viscosity_mpas", "Вязкость при 25 °C", "мПа·с", 0.01, 1_000_000, 1),
    ("viscosity_activation_kj_mol", "Энергия активации вязкости", "кДж/моль", 0.01, 300, 1),
    ("viscosity_modifier_per_wt_pct", "Поправка вязкости на 1 мас.%", "коэфф.", -5, 5, 4),
    ("refractive_index", "Показатель преломления", "n", 1, 3, 4),
    ("penetration_depth_mm", "Глубина проникновения Dp", "мм", 0.001, 100, 3),
    ("critical_exposure_mj_cm2", "Критическая экспозиция Ec", "мДж/см²", 0.001, 100_000, 2),
    ("initiator_peak_nm", "Пик поглощения инициатора", "нм", 200, 1000, 1),
    ("initiator_bandwidth_nm", "Ширина спектрального отклика", "нм", 1, 500, 1),
    ("initiator_sensitivity_per_wt_pct", "Вклад инициатора на мас.%", "коэфф.", 0, 10, 4),
    ("attenuation_per_wt_pct_mm_inv", "Доп. поглощение на мас.%", "1/мм", 0, 100, 4),
    ("shrinkage_vol_pct", "Объёмная усадка", "%", 0, 100, 2),
    ("modulus_mpa", "Модуль упругости", "МПа", 0.01, 1_000_000, 1),
    ("tensile_strength_mpa", "Прочность на растяжение", "МПа", 0.01, 100_000, 1),
    ("elongation_pct", "Относительное удлинение", "%", 0, 10_000, 1),
    ("brittleness_index", "Индекс хрупкости", "/100", 0, 100, 1),
    ("layer_bond_factor", "Фактор сцепления слоёв", "/100", 0, 100, 1),
    ("inhibition_per_wt_pct", "Ингибирующий вклад на мас.%", "коэфф.", 0, 100, 3),
)


class ComponentEditor(QDialog):
    """Edit an ingredient while keeping unknown properties genuinely blank."""

    def __init__(self, component: Component | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Компонент библиотеки")
        self.resize(600, 760)
        self._original = component
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(component.name if component else "Новый компонент")
        self.category = QComboBox()
        for key, label in COMPONENT_CATEGORIES.items():
            self.category.addItem(label, key)
        if component is None:
            self.category.setCurrentIndex(max(0, self.category.findData("additive")))
        if component:
            self.category.setCurrentIndex(max(0, self.category.findData(component.category)))
        self.source = QComboBox()
        self.source.addItem("Приближённая оценка", "estimate")
        self.source.addItem("Измерено / из паспорта материала", "measured")
        if component:
            self.source.setCurrentIndex(max(0, self.source.findData(component.source)))
        form.addRow("Название", self.name)
        form.addRow("Тип", self.category)
        form.addRow("Источник значений", self.source)
        root.addLayout(form)

        intro = QLabel("Оставьте неизвестные свойства выключенными. Подробности появятся в расчёте как недостающие данные.")
        intro.setObjectName("hintLabel")
        intro.setWordWrap(True)
        root.addWidget(intro)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        values = QFormLayout(body)
        self._values: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        for key, label, unit, low, high, decimals in PROPERTY_FIELDS:
            enabled = QCheckBox("задать")
            number = QDoubleSpinBox()
            number.setRange(low, high)
            number.setDecimals(decimals)
            number.setSingleStep(0.1 if decimals < 3 else 0.01)
            number.setEnabled(False)
            value = getattr(component, key, None) if component else None
            enabled.setChecked(value is not None)
            if value is not None:
                number.setValue(float(value))
            number.setEnabled(value is not None)
            enabled.toggled.connect(number.setEnabled)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(enabled)
            row_layout.addWidget(number, 1)
            row_layout.addWidget(QLabel(unit))
            values.addRow(label, row)
            self._values[key] = enabled, number
        scroll.setWidget(body)
        root.addWidget(scroll, 1)
        self.notes = QPlainTextEdit(component.notes if component else "")
        self.notes.setPlaceholderText("Поставщик, паспорт, условия измерения или ограничения модели")
        self.notes.setMaximumHeight(76)
        root.addWidget(self.notes)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def component(self) -> Component:
        values: dict[str, Any] = {}
        for key, (enabled, number) in self._values.items():
            values[key] = number.value() if enabled.isChecked() else None
        if self._original:
            key = self._original.key
        else:
            slug = re.sub(r"[^a-z0-9]+", "_", self.name.text().casefold()).strip("_")[:24] or "ingredient"
            key = f"custom_{slug}_{uuid.uuid4().hex[:6]}"
        item = Component(
            key=key,
            name=self.name.text().strip(),
            category=str(self.category.currentData()),
            source=str(self.source.currentData()),
            notes=self.notes.toPlainText().strip(),
            **values,
        )
        item.validate()
        return item


class DoseCurveWidget(QWidget):
    """Compact dose/depth graph drawn with the Qt painter API."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._curve: list[tuple[float, float]] = []
        self._current_depth = 0.0
        self._current_dose = 0.0
        self._layer = 0.05
        self.setMinimumHeight(210)

    def set_result(self, result: SimulationResult, recipe: Recipe) -> None:
        self._curve = result.dose_curve
        prediction = next((item for item in result.predictions if item.key == "cure_depth"), None)
        self._current_depth = _leading_number(prediction.value) if prediction and prediction.value else 0.0
        self._current_dose = recipe.process.intensity_mw_cm2 * recipe.process.exposure_s
        self._layer = recipe.process.layer_height_um / 1000
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#161c27"))
        area = self.rect().adjusted(48, 16, -16, -32)
        painter.setPen(QPen(QColor("#667085"), 1))
        painter.drawLine(area.bottomLeft(), area.bottomRight())
        painter.drawLine(area.bottomLeft(), area.topLeft())
        if len(self._curve) < 2:
            painter.setPen(QColor("#aeb7c8"))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "Укажите Dp и Ec, чтобы построить кривую")
            return
        max_dose = max(item[0] for item in self._curve)
        max_depth = max(max(item[1] for item in self._curve), self._layer * 1.5, 0.1)
        painter.setPen(QColor("#aeb7c8"))
        painter.drawText(2, area.top() + 8, "глубина, мм")
        painter.drawText(area.left(), area.bottom() + 24, "доза, мДж/см²")
        painter.setPen(QColor("#3a4558"))
        layer_y = area.bottom() - area.height() * self._layer / max_depth
        painter.drawLine(QPointF(area.left(), layer_y), QPointF(area.right(), layer_y))
        points = QPolygonF()
        for dose, depth in self._curve:
            x = area.left() + (dose / max_dose) * area.width()
            y = area.bottom() - (depth / max_depth) * area.height()
            points.append(QPointF(x, y))
        painter.setPen(QPen(QColor("#69dfb6"), 2.5))
        painter.drawPolyline(points)
        painter.setPen(QColor("#e8bd6f"))
        marker_x = area.left() + min(1.0, max(0.0, self._current_dose / max_dose)) * area.width()
        marker_y = area.bottom() - min(1.0, max(0.0, self._current_depth / max_depth)) * area.height()
        painter.setBrush(QColor("#e8bd6f"))
        painter.drawEllipse(QPointF(marker_x, marker_y), 4, 4)


class ResinLabTab(QWidget):
    """Six-step resin lab with editable component library and recipe comparison."""

    progress = pyqtSignal(float, str)
    logMessage = pyqtSignal(str)
    _STEP_NAMES = (
        "1 · Задача", "2 · Основа", "3 · Состав", "4 · Процесс", "5 · Прогноз", "6 · Сравнение",
    )
    _PROCESS_FIELDS: tuple[tuple[str, str, str, float, float, int], ...] = (
        ("wavelength_nm", "Длина волны", "нм", 200, 1000, 0),
        ("intensity_mw_cm2", "Интенсивность на поверхности смолы", "мВт/см²", 0.01, 5000, 2),
        ("exposure_s", "Экспозиция", "с", 0.001, 3600, 3),
        ("layer_height_um", "Толщина слоя", "мкм", 1, 1000, 1),
        ("temperature_c", "Температура смеси", "°C", -20, 150, 1),
        ("mix_rpm", "Скорость перемешивания", "об/мин", 0, 10000, 0),
        ("mix_minutes", "Время перемешивания", "мин", 0, 1440, 1),
        ("rest_minutes", "Выдержка / удаление пузырьков", "мин", 0, 10080, 1),
        ("wash_minutes", "Промывка", "мин", 0, 1440, 1),
        ("post_cure_minutes", "Постотверждение", "мин", 0, 1440, 1),
        ("reference_dimension_mm", "Размер детали для оценки усадки", "мм", 0.01, 100000, 1),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.components = default_components()
        self.recipes = starter_recipes()
        self.recipe = Recipe()
        self.result: SimulationResult | None = None
        self._dirty = True
        self._unit_widgets: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        self._load_path = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)) / "resin_lab.json"
        self._load_persisted()
        self._build_ui()
        self._refresh_component_combos()
        self._load_recipe_to_controls()
        self._refresh_library_list()

    def set_light_source_assumption(
        self,
        wavelength_nm: float,
        intensity_mw_cm2: float | None,
        measured: bool,
    ) -> None:
        """Transfer a source scenario without claiming an unmeasured dose is known."""

        wavelength_check, wavelength_value = self._unit_widgets["wavelength_nm"]
        wavelength_value.setValue(wavelength_nm)
        wavelength_check.setChecked(False)
        intensity_check, intensity_value = self._unit_widgets["intensity_mw_cm2"]
        if intensity_mw_cm2 is not None:
            intensity_value.setValue(intensity_mw_cm2)
        intensity_check.setChecked(measured and intensity_mw_cm2 is not None)
        self._mark_dirty()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("Лаборатория фотополимерной смолы")
        title.setObjectName("panelTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.recipe_name = QLineEdit(self.recipe.name)
        self.recipe_name.setPlaceholderText("Название рецептуры")
        self.recipe_name.setMaximumWidth(220)
        header.addWidget(QLabel("Рецепт:"))
        header.addWidget(self.recipe_name)
        self.save_button = QPushButton("Сохранить")
        self.copy_button = QPushButton("Копировать")
        self.rename_button = QPushButton("Переименовать")
        self.export_button = QPushButton("Экспорт JSON")
        for widget in (self.save_button, self.copy_button, self.rename_button, self.export_button):
            header.addWidget(widget)
        root.addLayout(header)

        steps_box = QFrame()
        steps_box.setObjectName("workflowSteps")
        steps_layout = QVBoxLayout(steps_box)
        steps_layout.setContentsMargins(9, 7, 9, 7)
        steps_layout.setSpacing(5)
        step_row = QHBoxLayout()
        self._step_buttons: list[QPushButton] = []
        for index, name in enumerate(self._STEP_NAMES):
            button = QPushButton(name)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, page=index: self._go_to(page))
            self._step_buttons.append(button)
            step_row.addWidget(button, 1)
        steps_layout.addLayout(step_row)
        self.step_description = QLabel()
        self.step_description.setObjectName("hintLabel")
        self.step_description.setWordWrap(True)
        steps_layout.addWidget(self.step_description)
        root.addWidget(steps_box)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_task_page())
        self.stack.addWidget(self._build_base_page())
        self.stack.addWidget(self._build_dose_page())
        self.stack.addWidget(self._build_process_page())
        self.stack.addWidget(self._build_results_page())
        self.stack.addWidget(self._build_compare_page())
        root.addWidget(self.stack, 1)

        footer = QHBoxLayout()
        self.footer_status = QLabel("Значения по умолчанию отмечены как предположения.")
        self.footer_status.setObjectName("hintLabel")
        self.footer_status.setWordWrap(True)
        footer.addWidget(self.footer_status, 1)
        self.back_button = QPushButton("← Назад")
        self.next_button = QPushButton("Далее →")
        self.next_button.setObjectName("generateButton")
        footer.addWidget(self.back_button)
        footer.addWidget(self.next_button)
        root.addLayout(footer)
        self.back_button.clicked.connect(lambda: self._go_to(self.stack.currentIndex() - 1))
        self.next_button.clicked.connect(self._next_step)
        self.save_button.clicked.connect(self._save_current)
        self.copy_button.clicked.connect(self._copy_current)
        self.rename_button.clicked.connect(self._rename_current)
        self.export_button.clicked.connect(self._export_current)
        self.recipe_name.textChanged.connect(self._mark_dirty)
        self._go_to(0)

    def _build_task_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(self._heading("Какие свойства важнее?", "Выбор задачи даёт стартовый рецепт и помогает интерпретировать результат."))
        row = QHBoxLayout()
        self.task_combo = QComboBox()
        for key, (name, _description) in TASKS.items():
            self.task_combo.addItem(name, key)
        row.addWidget(self.task_combo, 1)
        self.example_combo = QComboBox()
        self.example_combo.addItem("Выбрать пример рецептуры…", "")
        for name in starter_recipes():
            self.example_combo.addItem(name, name)
        row.addWidget(self.example_combo, 1)
        self.load_example_button = QPushButton("Загрузить пример")
        row.addWidget(self.load_example_button)
        layout.addLayout(row)
        self.task_description = QLabel()
        self.task_description.setObjectName("hintLabel")
        self.task_description.setWordWrap(True)
        layout.addWidget(self.task_description)
        self.task_combo.currentIndexChanged.connect(self._update_task_description)
        self.task_combo.currentIndexChanged.connect(self._mark_dirty)
        self.load_example_button.clicked.connect(self._load_selected_example)
        self._update_task_description()
        layout.addStretch(1)
        layout.addWidget(self._note("Библиотечные рецепты — учебные стартовые точки. Не используйте их как готовую спецификацию смолы."))
        return page

    def _build_base_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.addWidget(self._heading("Основа смолы", "Выберите библиотечную основу или создайте собственную. Неизвестные поля можно оставить пустыми."))
        row = QHBoxLayout()
        self.base_combo = QComboBox()
        self.base_combo.currentIndexChanged.connect(self._show_base_summary)
        self.base_combo.currentIndexChanged.connect(self._mark_dirty)
        row.addWidget(self.base_combo, 1)
        self.new_base_button = QPushButton("＋ Своя основа")
        self.edit_base_button = QPushButton("Изменить данные")
        self.new_component_button = QPushButton("＋ Компонент")
        row.addWidget(self.new_base_button)
        row.addWidget(self.edit_base_button)
        row.addWidget(self.new_component_button)
        layout.addLayout(row)
        self.base_summary = QLabel()
        self.base_summary.setObjectName("hintLabel")
        self.base_summary.setWordWrap(True)
        self.base_summary.setMinimumHeight(130)
        layout.addWidget(self.base_summary)
        layout.addWidget(self._note("Вы можете указать свои Dp/Ec по рабочей кривой материала, вязкость при известной температуре, плотность и механику. Источник каждого компонента сохраняется."))
        layout.addStretch(1)
        self.new_base_button.clicked.connect(lambda: self._edit_component(None, "base"))
        self.edit_base_button.clicked.connect(self._edit_selected_base)
        self.new_component_button.clicked.connect(lambda: self._edit_component(None, None))
        return page

    def _build_dose_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addWidget(self._heading("Состав и дозировки", "Введите массу партии и количества добавок. Основа автоматически заполняет оставшуюся массу."))
        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Итоговая масса смеси"))
        self.batch_mass = QDoubleSpinBox()
        self.batch_mass.setRange(0.01, 1_000_000)
        self.batch_mass.setDecimals(2)
        self.batch_mass.setValue(100)
        self.batch_mass.setSuffix(" г")
        batch_row.addWidget(self.batch_mass)
        batch_row.addStretch(1)
        self.base_balance_label = QLabel()
        self.base_balance_label.setObjectName("hintLabel")
        batch_row.addWidget(self.base_balance_label)
        layout.addLayout(batch_row)
        add_box = QGroupBox("Добавить компонент")
        add_row = QHBoxLayout(add_box)
        self.add_component_combo = QComboBox()
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(0.001, 1_000_000)
        self.amount_spin.setDecimals(3)
        self.amount_spin.setValue(1.5)
        self.unit_combo = QComboBox()
        self.unit_combo.addItem("г", "g")
        self.unit_combo.addItem("мас. %", "mass_pct")
        self.unit_combo.addItem("об. %", "volume_pct")
        self.add_ingredient_button = QPushButton("Добавить / изменить")
        add_row.addWidget(self.add_component_combo, 1)
        add_row.addWidget(self.amount_spin)
        add_row.addWidget(self.unit_combo)
        add_row.addWidget(self.add_ingredient_button)
        layout.addWidget(add_box)
        hint = QLabel("Объёмная доля пересчитывается через плотности всех ингредиентов. Если плотность неизвестна, поле останется пустым.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.composition_table = self._table(("Компонент", "Категория", "Масса, г", "мас. %", "об. %", "Источник"))
        layout.addWidget(self.composition_table, 1)
        actions = QHBoxLayout()
        self.remove_ingredient_button = QPushButton("Удалить выбранную добавку")
        self.component_edit_button = QPushButton("Редактировать компонент")
        actions.addWidget(self.remove_ingredient_button)
        actions.addWidget(self.component_edit_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.batch_mass.valueChanged.connect(self._refresh_composition)
        self.batch_mass.valueChanged.connect(self._mark_dirty)
        self.add_ingredient_button.clicked.connect(self._add_ingredient)
        self.remove_ingredient_button.clicked.connect(self._remove_ingredient)
        self.component_edit_button.clicked.connect(self._edit_selected_ingredient)
        self.composition_table.itemSelectionChanged.connect(self._update_selected_ingredient)
        self.unit_combo.currentIndexChanged.connect(self._update_dose_unit_hint)
        self._update_dose_unit_hint()
        return page

    def _build_process_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.addWidget(self._heading("Условия процесса", "Если точного параметра нет, оставьте предложенное значение и снимите отметку «измерено». Предположения будут перечислены в результате."))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        form = QFormLayout(body)
        for key, label, unit, low, high, decimals in self._PROCESS_FIELDS:
            check = QCheckBox("измерено")
            number = QDoubleSpinBox()
            number.setRange(low, high)
            number.setDecimals(decimals)
            number.setSingleStep(0.1 if decimals else 1)
            wrap = QWidget()
            row = QHBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(number, 1)
            row.addWidget(QLabel(unit))
            row.addWidget(check)
            form.addRow(label, wrap)
            number.valueChanged.connect(self._on_process_value_changed)
            check.toggled.connect(self._on_process_value_changed)
            self._unit_widgets[key] = (check, number)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        details = QGroupBox("Подробно · модель и коэффициенты")
        details.setCheckable(True)
        details.setChecked(False)
        detail_layout = QFormLayout(details)
        model_help = QLabel("Эти коэффициенты меняют приближённую модель. Начните с 1, меняйте после сравнения с измерениями и сохраняйте значения вместе с рецептом.")
        model_help.setObjectName("hintLabel")
        model_help.setWordWrap(True)
        detail_layout.addRow(model_help)
        self.model_combo = QComboBox()
        for plugin in DEFAULT_MODEL_REGISTRY.all():
            self.model_combo.addItem(plugin.name, plugin.key)
        self.model_combo.currentIndexChanged.connect(self._on_model_value_changed)
        detail_layout.addRow("Расчётная модель", self.model_combo)
        self._model_widgets: dict[str, QDoubleSpinBox] = {}
        model_fields = (
            ("viscosity_interaction", "Вклад идеального смешения вязкости", 0.01, 5, 1),
            ("filler_sensitivity", "Поправка вязкости добавок", 0, 5, 1),
            ("cure_depth_scale", "Масштаб Dp", 0.05, 10, 1),
            ("critical_exposure_scale", "Масштаб Ec", 0.05, 10, 1),
            ("shrinkage_scale", "Масштаб усадки", 0.05, 10, 1),
            ("mechanics_scale", "Масштаб механических оценок", 0.05, 10, 1),
            ("exposure_window_overcure_layers", "Верх окна, число слоёв", 1.01, 5, 1),
            ("temperature_reference_c", "Опорная температура вязкости", -20, 150, 1),
        )
        for key, label, low, high, step in model_fields:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(2)
            spin.setSingleStep(step)
            spin.valueChanged.connect(self._on_model_value_changed)
            detail_layout.addRow(label, spin)
            self._model_widgets[key] = spin
        outer.addWidget(details)
        self._assumption_notice = QLabel("Предположение по умолчанию: 405 нм, 5 мВт/см² на смоле, слой 50 мкм, температура 25 °C.")
        self._assumption_notice.setObjectName("hintLabel")
        self._assumption_notice.setWordWrap(True)
        outer.addWidget(self._assumption_notice)
        return page

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 10, 12, 10)
        top = QHBoxLayout()
        top.addWidget(self._heading("Прогноз и ограничения", "Нажмите на строку, чтобы увидеть источники, допущения и способ повысить точность."), 1)
        self.calculate_button = QPushButton("⟳ Рассчитать прогноз")
        self.calculate_button.setObjectName("generateButton")
        top.addWidget(self.calculate_button)
        layout.addLayout(top)
        self.result_status = QLabel("Расчёт ещё не запускался.")
        self.result_status.setObjectName("hintLabel")
        self.result_status.setWordWrap(True)
        layout.addWidget(self.result_status)
        split = QHBoxLayout()
        self.prediction_table = self._table(("Показатель", "Значение", "Диапазон модели", "Доверие", "Основные факторы"))
        self.prediction_table.setMinimumWidth(690)
        self.prediction_table.itemSelectionChanged.connect(self._show_prediction_detail)
        split.addWidget(self.prediction_table, 3)
        side = QVBoxLayout()
        curve_box = QGroupBox("Рабочая кривая Jacobs")
        curve_layout = QVBoxLayout(curve_box)
        self.curve_widget = DoseCurveWidget()
        curve_layout.addWidget(self.curve_widget)
        side.addWidget(curve_box, 1)
        self.prediction_detail = QLabel("Выберите прогноз в таблице.")
        self.prediction_detail.setObjectName("hintLabel")
        self.prediction_detail.setWordWrap(True)
        self.prediction_detail.setMinimumHeight(115)
        detail_box = QGroupBox("Подробно · объяснение")
        detail_layout = QVBoxLayout(detail_box)
        detail_layout.addWidget(self.prediction_detail)
        side.addWidget(detail_box)
        split.addLayout(side, 2)
        layout.addLayout(split, 3)
        bottom = QHBoxLayout()
        self.sensitivity_table = self._table(("Изменение", "Вязкость, %", "Глубина, %", "Усадка, %", "Индекс, п.п."))
        sensitivity_box = QGroupBox("Чувствительность · малое изменение параметра")
        sensitivity_layout = QVBoxLayout(sensitivity_box)
        sensitivity_layout.addWidget(self.sensitivity_table)
        bottom.addWidget(sensitivity_box, 3)
        self.warning_list = QListWidget()
        warning_box = QGroupBox("Предупреждения и условия")
        warning_layout = QVBoxLayout(warning_box)
        warning_layout.addWidget(self.warning_list)
        bottom.addWidget(warning_box, 2)
        layout.addLayout(bottom, 2)
        self.calculate_button.clicked.connect(self._calculate)
        return page

    def _build_compare_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addWidget(self._heading("Сравнение рецептур", "Выберите сохранённые варианты. Каждый пересчитывается на текущих условиях процесса и одной версии модели."))
        row = QHBoxLayout()
        self.library_list = QListWidget()
        self.library_list.setMaximumHeight(160)
        self.compare_button = QPushButton("Сравнить выбранные")
        self.compare_button.setObjectName("generateButton")
        self.load_recipe_button = QPushButton("Открыть в мастере")
        row.addWidget(self.library_list, 1)
        controls = QVBoxLayout()
        controls.addWidget(self.compare_button)
        controls.addWidget(self.load_recipe_button)
        controls.addStretch(1)
        row.addLayout(controls)
        layout.addLayout(row)
        self.compare_notice = QLabel("Сравнение использует одинаковые условия печати и одинаковые коэффициенты модели для всех выбранных рецептов.")
        self.compare_notice.setObjectName("hintLabel")
        self.compare_notice.setWordWrap(True)
        layout.addWidget(self.compare_notice)
        self.compare_table = self._table(("Показатель",))
        layout.addWidget(self.compare_table, 1)
        self.compare_button.clicked.connect(self._compare_selected)
        self.load_recipe_button.clicked.connect(self._open_selected_recipe)
        return page

    @staticmethod
    def _heading(title: str, description: str) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 2)
        label = QLabel(title)
        label.setObjectName("panelTitle")
        layout.addWidget(label)
        help_text = QLabel(description)
        help_text.setObjectName("hintLabel")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        return box

    @staticmethod
    def _note(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("hintLabel")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _table(headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        vertical_header = table.verticalHeader()
        horizontal_header = table.horizontalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        if horizontal_header is not None:
            horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            horizontal_header.setStretchLastSection(True)
        return table

    def _load_persisted(self) -> None:
        if not self._load_path.exists():
            return
        try:
            saved_components, saved_recipes = load_library(self._load_path)
            self.components.update(saved_components)
            self.recipes.update(saved_recipes)
        except (OSError, ValueError, TypeError) as exc:
            self.logMessage.emit(f"Библиотека рецептур не загружена: {exc}")

    def _persist(self) -> None:
        try:
            save_library(self._load_path, self.components, self.recipes)
        except OSError as exc:
            QMessageBox.warning(self, "Не удалось сохранить", str(exc))

    def _refresh_component_combos(self) -> None:
        if not hasattr(self, "base_combo"):
            return
        current_base = self.recipe.base_id
        self.base_combo.blockSignals(True)
        self.base_combo.clear()
        for key, item in self.components.items():
            if item.category == "base":
                self.base_combo.addItem(item.name, key)
        index = self.base_combo.findData(current_base)
        self.base_combo.setCurrentIndex(max(0, index))
        self.base_combo.blockSignals(False)
        current_add = self.add_component_combo.currentData() if self.add_component_combo.count() else None
        self.add_component_combo.clear()
        for key, item in self.components.items():
            if item.category != "base":
                self.add_component_combo.addItem(f"{item.name} · {COMPONENT_CATEGORIES[item.category]}", key)
        selected = self.add_component_combo.findData(current_add)
        if selected >= 0:
            self.add_component_combo.setCurrentIndex(selected)
        self._show_base_summary()
        self._refresh_composition()

    def _show_base_summary(self, *_args) -> None:
        if not hasattr(self, "base_summary") or not self.base_combo.count():
            return
        key = str(self.base_combo.currentData())
        base = self.components[key]
        props = (
            ("Плотность", base.density_g_ml, "г/мл"), ("Вязкость", base.viscosity_mpas, "мПа·с"),
            ("Dp", base.penetration_depth_mm, "мм"), ("Ec", base.critical_exposure_mj_cm2, "мДж/см²"),
            ("Усадка", base.shrinkage_vol_pct, "%"), ("Модуль", base.modulus_mpa, "МПа"),
        )
        values = " · ".join(f"{name}: {val:g} {unit}" if val is not None else f"{name}: нет данных" for name, val, unit in props)
        self.base_summary.setText(
            f"{values}\nИсточник: {'измерено / паспорт' if base.source == 'measured' else 'оценка библиотеки'}. "
            f"{base.notes}\n\nНеобязательные свойства можно задать в библиотеке компонентов. "
            "Для расчёта отверждения особенно важны Dp и Ec при той же длине волны, что и у принтера."
        )

    def _refresh_composition(self, *_args) -> None:
        if not hasattr(self, "composition_table"):
            return
        self.recipe.batch_mass_g = self.batch_mass.value()
        self.recipe.base_id = str(self.base_combo.currentData() or self.recipe.base_id) if hasattr(self, "base_combo") else self.recipe.base_id
        try:
            rows = self._composition_rows_for_controls()
            self.composition_table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                values = (
                    str(row["name"]), str(row["category"]), f"{float(row['mass_g']):.3f}",
                    f"{float(row['weight_pct']):.2f}%",
                    "—" if row["volume_pct"] is None else f"{float(row['volume_pct']):.2f}%",
                    "измерено" if self.components[str(row["key"])].source == "measured" else "оценка",
                )
                for col, value in enumerate(values):
                    self.composition_table.setItem(row_index, col, QTableWidgetItem(value))
                first_cell = self.composition_table.item(row_index, 0)
                if first_cell is not None:
                    first_cell.setData(Qt.ItemDataRole.UserRole, row["key"])
        except ValueError as exc:
            self.composition_table.setRowCount(0)
            if hasattr(self, "footer_status"):
                self.footer_status.setText(str(exc))
        base_mass = self.batch_mass.value() - sum(self.recipe.additives_g.values())
        if hasattr(self, "base_balance_label"):
            self.base_balance_label.setText(f"Основа до итога: {base_mass:.3f} г")

    def _composition_rows_for_controls(self) -> list[dict[str, Any]]:
        recipe = self._recipe_from_controls(include_process=False)
        return composition_table(recipe, self.components)

    def _recipe_from_controls(self, *, include_process: bool = True) -> Recipe:
        name = self.recipe_name.text().strip() if hasattr(self, "recipe_name") else self.recipe.name
        task = str(self.task_combo.currentData()) if hasattr(self, "task_combo") else self.recipe.task
        base_id = str(self.base_combo.currentData()) if hasattr(self, "base_combo") and self.base_combo.currentData() else self.recipe.base_id
        batch = self.batch_mass.value() if hasattr(self, "batch_mass") else self.recipe.batch_mass_g
        if include_process and hasattr(self, "_unit_widgets") and len(self._unit_widgets) == len(self._PROCESS_FIELDS):
            values: dict[str, Any] = {key: spin.value() for key, (_check, spin) in self._unit_widgets.items()}
            values["measured"] = tuple(key for key, (check, _spin) in self._unit_widgets.items() if check.isChecked())
            process = ProcessConditions(**values)
        else:
            process = self.recipe.process
        model_values = {key: spin.value() for key, spin in self._model_widgets.items()} if hasattr(self, "_model_widgets") and len(self._model_widgets) == len(ModelParameters.__dataclass_fields__) else self.recipe.model.to_dict()
        model_key = str(self.model_combo.currentData()) if hasattr(self, "model_combo") and self.model_combo.currentData() else self.recipe.model_key
        return Recipe(
            name=name or "Новая рецептура", task=task, base_id=base_id,
            batch_mass_g=batch, additives_g=dict(self.recipe.additives_g),
            process=process, model=ModelParameters(**model_values), model_key=model_key,
        )

    def _load_recipe_to_controls(self) -> None:
        self.recipe_name.setText(self.recipe.name)
        idx = self.task_combo.findData(self.recipe.task)
        if idx >= 0:
            self.task_combo.setCurrentIndex(idx)
        self._refresh_component_combos()
        self.base_combo.setCurrentIndex(max(0, self.base_combo.findData(self.recipe.base_id)))
        self.batch_mass.setValue(self.recipe.batch_mass_g)
        for key, (check, spin) in self._unit_widgets.items():
            spin.setValue(float(getattr(self.recipe.process, key)))
            check.setChecked(key in self.recipe.process.measured)
        for key, spin in self._model_widgets.items():
            spin.setValue(float(getattr(self.recipe.model, key)))
        model_index = self.model_combo.findData(self.recipe.model_key)
        if model_index >= 0:
            self.model_combo.setCurrentIndex(model_index)
        self._refresh_composition()
        self._update_task_description()
        self._mark_dirty()

    def _mark_dirty(self, *_args) -> None:
        self._dirty = True
        if hasattr(self, "result_status") and self.result is not None:
            self.result_status.setText("Параметры изменены. Пересчитайте прогноз, чтобы увидеть обновлённые значения.")

    def _on_process_value_changed(self, *_args) -> None:
        self._mark_dirty()

    def _on_model_value_changed(self, *_args) -> None:
        self._mark_dirty()

    def _go_to(self, page: int) -> None:
        if page < 0 or page >= self.stack.count():
            return
        self.stack.setCurrentIndex(page)
        for index, button in enumerate(self._step_buttons):
            button.setChecked(index == page)
        self.back_button.setEnabled(page > 0)
        self.next_button.setVisible(page < self.stack.count() - 1)
        self.next_button.setText("Рассчитать прогноз →" if page == 3 else "К сравнению →" if page == 4 else "Далее →")
        descriptions = (
            "Выберите цель и при желании загрузите пример.",
            "Укажите известные свойства основы; неизвестные оставьте пустыми.",
            "Задайте партию и добавки. Массовые и объёмные доли пересчитаются автоматически.",
            "Отметьте измеренные параметры. Остальные останутся предположениями.",
            "Прогнозы доступны только там, где хватает исходных данных.",
            "Сохранённые рецепты сравниваются на текущих одинаковых настройках процесса.",
        )
        self.step_description.setText(descriptions[page])
        if page == 4 and (self._dirty or self.result is None):
            self._calculate()
        if page == 5:
            self._refresh_library_list()
        self.footer_status.setText("Сохранено локально в библиотеке рецептур." if page == 5 else "Значения с пометкой «оценка» не являются паспортными данными.")

    def _next_step(self) -> None:
        page = self.stack.currentIndex()
        if page == 3:
            try:
                self._capture_controls()
                self.recipe.validate(self.components)
            except (ValueError, TypeError) as exc:
                QMessageBox.warning(self, "Проверьте параметры", str(exc))
                return
            self._go_to(4)
            return
        if page == 4:
            self._capture_controls()
            self._go_to(5)
            return
        if page == 2:
            self._refresh_composition()
        self._go_to(page + 1)

    def _capture_controls(self) -> None:
        self.recipe = self._recipe_from_controls()
        self.recipe.validate(self.components)

    def _update_task_description(self, *_args) -> None:
        if hasattr(self, "task_description"):
            _name, description = TASKS.get(str(self.task_combo.currentData()), TASKS["experimental"])
            self.task_description.setText(description)

    def _load_selected_example(self) -> None:
        name = self.example_combo.currentData()
        if not name:
            return
        samples = starter_recipes()
        if name in samples:
            self.recipe = Recipe.from_dict(samples[name].to_dict())
            self._load_recipe_to_controls()
            self._go_to(1)

    def _edit_component(self, component: Component | None, force_category: str | None) -> None:
        dialog = ComponentEditor(component, self)
        if force_category:
            idx = dialog.category.findData(force_category)
            if idx >= 0:
                dialog.category.setCurrentIndex(idx)
                dialog.category.setEnabled(False)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            updated = dialog.component()
            self.components[updated.key] = updated
            if updated.category == "base" and component is not None and component.category != "base":
                self.recipe.additives_g.pop(updated.key, None)
                self.recipe.base_id = updated.key
            elif updated.category == "base" and component is None:
                self.recipe.base_id = updated.key
            self._refresh_component_combos()
            if updated.category == "base":
                self.base_combo.setCurrentIndex(max(0, self.base_combo.findData(updated.key)))
            self._persist()
            self._mark_dirty()
            self.logMessage.emit(f"Компонент «{updated.name}» сохранён в библиотеку.")
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Некорректные данные компонента", str(exc))

    def _edit_selected_base(self) -> None:
        key = self.base_combo.currentData()
        if key in self.components:
            self._edit_component(self.components[str(key)], "base")

    def _edit_selected_ingredient(self) -> None:
        key = self.add_component_combo.currentData()
        if key in self.components:
            self._edit_component(self.components[str(key)], None)

    def _update_dose_unit_hint(self, *_args) -> None:
        if self.unit_combo.currentData() == "volume_pct":
            key = self.add_component_combo.currentData()
            missing = []
            if key in self.components and self.components[str(key)].density_g_ml is None:
                missing.append(self.components[str(key)].name)
            base_key = self.base_combo.currentData() if hasattr(self, "base_combo") else None
            if base_key in self.components and self.components[str(base_key)].density_g_ml is None:
                missing.append(self.components[str(base_key)].name)
            self.amount_spin.setToolTip("Нужно указать плотность: " + ", ".join(missing) if missing else "Объёмная доля вводится от полного объёма смеси.")
        else:
            self.amount_spin.setToolTip("Количество компонента в выбранных единицах.")

    def _add_ingredient(self) -> None:
        key = self.add_component_combo.currentData()
        if key not in self.components:
            return
        key = str(key)
        amount = self.amount_spin.value()
        current = dict(self.recipe.additives_g)
        try:
            mode = self.unit_combo.currentData()
            if mode == "g":
                mass = amount
            elif mode == "mass_pct":
                mass = self.batch_mass.value() * amount / 100
            else:
                mass = self._volume_pct_to_mass(key, amount, current)
            total_other = sum(value for other, value in current.items() if other != key)
            if mass <= 0 or mass + total_other >= self.batch_mass.value():
                raise ValueError("Дозировка должна быть положительной, а сумма добавок — меньше массы партии.")
            current[key] = mass
            self.recipe.additives_g = current
            self._refresh_composition()
            self._mark_dirty()
        except ValueError as exc:
            QMessageBox.warning(self, "Дозировка не добавлена", str(exc))

    def _volume_pct_to_mass(self, key: str, percent: float, current: dict[str, float]) -> float:
        if not 0 < percent < 100:
            raise ValueError("Объёмная доля должна быть между 0 и 100%.")
        base = self.components[str(self.base_combo.currentData())]
        target = self.components[key]
        if base.density_g_ml is None or target.density_g_ml is None:
            raise ValueError("Для пересчёта об.% задайте плотность основы и выбранного компонента.")
        fixed_mass = sum(value for other, value in current.items() if other != key)
        fixed_volume = 0.0
        for other, value in current.items():
            if other == key:
                continue
            density = self.components[other].density_g_ml
            if density is None:
                raise ValueError(f"Для об.% также нужна плотность компонента «{self.components[other].name}».")
            fixed_volume += value / density
        k = percent / (100 - percent)
        numerator = k * (fixed_volume + (self.batch_mass.value() - fixed_mass) / base.density_g_ml)
        denominator = 1 / target.density_g_ml + k / base.density_g_ml
        mass = numerator / denominator
        if mass + fixed_mass >= self.batch_mass.value():
            raise ValueError("Для этой объёмной доли не остаётся массы основы.")
        return mass

    def _remove_ingredient(self) -> None:
        row = self.composition_table.currentRow()
        if row < 0:
            return
        item = self.composition_table.item(row, 0)
        key = item.data(Qt.ItemDataRole.UserRole) if item else None
        if key and key != self.recipe.base_id:
            self.recipe.additives_g.pop(str(key), None)
            self._refresh_composition()
            self._mark_dirty()

    def _load_library_list(self) -> None:
        self.library_list.clear()
        for name in sorted(self.recipes, key=str.casefold):
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.library_list.addItem(item)

    def _refresh_library_list(self) -> None:
        if not hasattr(self, "library_list"):
            return
        previous: dict[str, Qt.CheckState] = {}
        for index in range(self.library_list.count()):
            previous_item = self.library_list.item(index)
            if previous_item is not None:
                previous[str(previous_item.data(Qt.ItemDataRole.UserRole))] = previous_item.checkState()
        self._load_library_list()
        for i in range(self.library_list.count()):
            library_item = self.library_list.item(i)
            if library_item is not None:
                name = str(library_item.data(Qt.ItemDataRole.UserRole))
                library_item.setCheckState(previous.get(name, Qt.CheckState.Unchecked))

    def _calculate(self) -> None:
        try:
            self._capture_controls()
            self.result = calculate_recipe(self.recipe, self.components, sensitivity=True)
            self._dirty = False
            self._populate_result()
            self.logMessage.emit(f"Расчёт рецептуры «{self.recipe.name}» завершён; оценки не являются гарантией печати.")
        except (ValueError, ArithmeticError, TypeError) as exc:
            self.result_status.setText(f"Расчёт не выполнен: {exc}")
            QMessageBox.warning(self, "Не удалось рассчитать", str(exc))

    def _populate_result(self) -> None:
        if self.result is None:
            return
        self.result_status.setText(
            f"{self.recipe.name} · {len(self.result.predictions)} показателей · "
            f"{sum(1 for item in self.result.predictions if item.value is None)} показателей скрыты из-за недостающих данных."
        )
        self.prediction_table.setRowCount(len(self.result.predictions))
        for row, prediction in enumerate(self.result.predictions):
            low_high = "—" if prediction.low is None or prediction.high is None else f"{prediction.low:.2g}…{prediction.high:.2g} {prediction.unit}"
            factors = "; ".join(prediction.drivers) if prediction.drivers else "—"
            values = (prediction.name, prediction.value or "Нет данных", low_high, prediction.confidence, factors)
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setData(Qt.ItemDataRole.UserRole, prediction.key)
                if prediction.value is None:
                    cell.setForeground(QColor("#e8bd6f"))
                    cell.setToolTip("Не выдумываем значение. Нужны данные: " + ", ".join(prediction.missing))
                else:
                    cell.setToolTip(prediction.caveat)
                self.prediction_table.setItem(row, col, cell)
        if self.result.predictions:
            self.prediction_table.selectRow(0)
        self.sensitivity_table.setRowCount(len(self.result.sensitivities))
        for row, sensitivity in enumerate(self.result.sensitivities):
            values = (
                sensitivity.name,
                _percent_change(sensitivity.viscosity_pct), _percent_change(sensitivity.cure_depth_pct),
                _percent_change(sensitivity.shrinkage_pct), _point_change(sensitivity.printability_points),
            )
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if col > 0 and sensitivity.perturbation.startswith(("модель не", "нет модели", "нет калибровки")):
                    cell.setToolTip(sensitivity.perturbation)
                self.sensitivity_table.setItem(row, col, cell)
        self.warning_list.clear()
        for warning in self.result.warnings + self.result.assumptions:
            QListWidgetItem(warning, self.warning_list)
        self.curve_widget.set_result(self.result, self.recipe)

    def _show_prediction_detail(self) -> None:
        if self.result is None:
            return
        row = self.prediction_table.currentRow()
        if row < 0:
            return
        cell = self.prediction_table.item(row, 0)
        key = cell.data(Qt.ItemDataRole.UserRole) if cell else None
        prediction = next((item for item in self.result.predictions if item.key == key), None)
        if prediction is None:
            return
        parts = [prediction.caveat or "Формула модели доступна в разделе «Подробно»."]
        if prediction.drivers:
            parts.append("Сильнее всего влияют: " + "; ".join(prediction.drivers) + ".")
        if prediction.sources:
            parts.append("Исходные данные: " + " · ".join(prediction.sources) + ".")
        if prediction.missing:
            parts.append("Не хватает: " + "; ".join(prediction.missing) + ".")
        if prediction.value is None:
            parts.append("Способ улучшить прогноз: введите измеренное свойство в библиотеке компонентов или укажите источник процесса как измеренный.")
        self.prediction_detail.setText("\n\n".join(parts))

    def _save_current(self) -> None:
        try:
            self._capture_controls()
        except ValueError as exc:
            QMessageBox.warning(self, "Рецептура не сохранена", str(exc))
            return
        self.recipes[self.recipe.name] = Recipe.from_dict(self.recipe.to_dict())
        self._persist()
        self._refresh_library_list()
        self.footer_status.setText(f"Рецептура «{self.recipe.name}» сохранена.")

    def _copy_current(self) -> None:
        self._capture_controls()
        name, accepted = QInputDialog.getText(self, "Копия рецептуры", "Название новой копии:", text=f"{self.recipe.name} — копия")
        if not accepted or not name.strip():
            return
        self.recipe = Recipe.from_dict(self.recipe.to_dict())
        self.recipe.name = name.strip()
        self._load_recipe_to_controls()
        self._save_current()

    def _rename_current(self) -> None:
        old_name = self.recipe.name
        name, accepted = QInputDialog.getText(self, "Переименовать", "Новое название:", text=old_name)
        if not accepted or not name.strip():
            return
        self.recipes.pop(old_name, None)
        self.recipe.name = name.strip()
        self.recipe_name.setText(self.recipe.name)
        self._save_current()

    def _export_current(self) -> None:
        try:
            self._capture_controls()
            path, _ = QFileDialog.getSaveFileName(self, "Экспорт рецептуры", f"{self.recipe.name}.json", "Рецептура SpinSlicer (*.json)")
            if not path:
                return
            export_recipe(path, self.recipe, self.components)
            self.footer_status.setText(f"Экспортировано: {path}")
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Экспорт не выполнен", str(exc))

    def _compare_selected(self) -> None:
        self._capture_controls()
        selected: list[str] = []
        for index in range(self.library_list.count()):
            library_item = self.library_list.item(index)
            if library_item is not None and library_item.checkState() == Qt.CheckState.Checked:
                selected.append(str(library_item.data(Qt.ItemDataRole.UserRole)))
        if self.recipe.name in self.recipes and self.recipe.name not in selected:
            selected.insert(0, self.recipe.name)
        if len(selected) < 2:
            QMessageBox.information(self, "Нужно два варианта", "Отметьте минимум две сохранённые рецептуры. Текущую можно сначала сохранить.")
            return
        variants: list[tuple[str, SimulationResult]] = []
        for name in selected:
            saved = self.recipes[name]
            common = Recipe.from_dict(saved.to_dict())
            common.process = ProcessConditions.from_dict(self.recipe.process.to_dict())
            common.model = ModelParameters.from_dict(self.recipe.model.to_dict())
            common.model_key = self.recipe.model_key
            try:
                variants.append((name, calculate_recipe(common, self.components, sensitivity=False)))
            except ValueError as exc:
                QMessageBox.warning(self, "Вариант не рассчитан", f"{name}: {exc}")
                return
        metric_keys = ["viscosity", "temperature_effect", "cure_depth", "cure_rate", "exposure_window", "shrinkage", "dimension_change", "stiffness", "flexibility", "strength", "brittleness", "layer_adhesion", "printability", "failure_risks", "mixing"]
        first_result = variants[0][1]
        names = [variants[0][0]]
        for name, _result in variants[1:]:
            names.append(name)
        self.compare_table.clear()
        self.compare_table.setColumnCount(1 + len(variants))
        self.compare_table.setHorizontalHeaderLabels(["Показатель", *names])
        self.compare_table.setRowCount(len(metric_keys))
        for row, key in enumerate(metric_keys):
            first = next((item for item in first_result.predictions if item.key == key), None)
            label = first.name if first else key
            self.compare_table.setItem(row, 0, QTableWidgetItem(label))
            for col, (_name, result) in enumerate(variants, start=1):
                item = next((prediction for prediction in result.predictions if prediction.key == key), None)
                display = item.value if item and item.value is not None else "Нет данных"
                cell = QTableWidgetItem(display)
                if item is None or item.value is None:
                    cell.setForeground(QColor("#e8bd6f"))
                    if item and item.missing:
                        cell.setToolTip("Недостаёт: " + ", ".join(item.missing))
                else:
                    cell.setToolTip(item.caveat)
                self.compare_table.setItem(row, col, cell)
        self.compare_table.resizeColumnsToContents()
        self.compare_notice.setText(
            f"Сравнение {len(variants)} вариантов при {self.recipe.process.wavelength_nm:g} нм, "
            f"{self.recipe.process.intensity_mw_cm2:g} мВт/см², экспозиции {self.recipe.process.exposure_s:g} с, "
            f"слое {self.recipe.process.layer_height_um:g} мкм и температуре {self.recipe.process.temperature_c:g} °C. "
            "Это сравнение сценариев, а не валидация модели."
        )

    def _open_selected_recipe(self) -> None:
        item = self.library_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Выберите рецепт", "Выберите вариант в списке.")
            return
        name = str(item.data(Qt.ItemDataRole.UserRole))
        if name not in self.recipes:
            return
        self.recipe = Recipe.from_dict(self.recipes[name].to_dict())
        self._load_recipe_to_controls()
        self._go_to(0)

    def _update_selected_ingredient(self) -> None:
        row = self.composition_table.currentRow()
        if row < 0:
            return
        item = self.composition_table.item(row, 0)
        key = item.data(Qt.ItemDataRole.UserRole) if item else None
        index = self.add_component_combo.findData(key)
        if index >= 0:
            self.add_component_combo.setCurrentIndex(index)
        mass = self.composition_table.item(row, 2)
        if mass and key != self.recipe.base_id:
            try:
                self.amount_spin.setValue(float(mass.text()))
            except ValueError:
                pass


def _leading_number(value: str) -> float:
    match = re.search(r"-?\d+(?:[.,]\d+)?", value)
    return float(match.group().replace(",", ".")) if match else 0.0


def _percent_change(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f}%" if abs(value) < 9999 else "сильно меняется"


def _point_change(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f}"
