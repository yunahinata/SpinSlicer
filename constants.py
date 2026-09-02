"""
constants.py
============
Единая точка правды для дефолтных значений и цветовой палитры.
Меняя цифры здесь, не нужно лазить по всему проекту.
"""

APP_TITLE = "Spin Slicer"
APP_ORG = "SpinSlicer"

# --- Дефолты слайсинга (перенесены из прототипа без изменений) -------------
DEFAULT_GRID_RESOLUTION = 112
DEFAULT_OUTPUT_RESOLUTION = 720
DEFAULT_NUM_FRAMES = 360
DEFAULT_DIAMETER_MM = 60.0
FILL_FRACTION = 0.92
ANTI_ALIAS_FACTOR = 4

# --- Вьюпорт -----------------------------------------------------------
# Колба всегда абсолютного размера: диаметр — из настроек, высота — фиксированное
# отношение к диаметру. Она НЕ пересчитывается при трансформации модели.
VAT_HEIGHT_RATIO = 1.6
VAT_RESOLUTION = 96

# Децимация полигонов — только для показа во вьюпорте (60 FPS на тяжёлых STL).
# На расчёт проекций это не влияет: генерация всегда использует исходный меш.
MAX_VIEWPORT_TRIANGLES = 400_000

# --- Палитра -------------------------------------------------------------
ACCENT_GREEN = "#2fa572"
ACCENT_GREEN_HOVER = "#247d57"
ACCENT_BLUE = "#3d8bfd"
ACCENT_BLUE_HOVER = "#2f6fd1"
DANGER_RED = "#e05c5c"

VAT_COLOR = "#6fb8cc"
MODEL_COLOR = "#00b4d8"
VIEWPORT_BG_TOP = "#232733"
VIEWPORT_BG_BOTTOM = "#121319"

PANEL_RADIUS = 12
BUTTON_RADIUS = 8
