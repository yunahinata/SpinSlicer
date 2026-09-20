"""Small runtime language layer for the desktop UI.

The original prototype was written in Russian.  Russian remains the source
text so existing log messages and saved workflows stay readable; the default
application language is now Russian and this module translates the static Qt
surface without requiring a Qt Linguist build step.
"""
from __future__ import annotations

from typing import Any

LANGUAGES = {"en": "English", "ru": "Русский"}
_language = "ru"

# Exact source-string translations.  Keys are deliberately the existing
# Russian UI strings so applying the English layer does not alter the Russian
# fallback or the public API.
_EN: dict[str, str] = {
    "Настройки процесса": "Process settings",
    "Настройки принтера и процесса": "Printer and process settings",
    "Настройки": "Settings",
    "Настройки принтера, процесса и альтернативного движка":
        "Printer, process, and alternate engine settings",
    "Печатная колба": "Print vat",
    "Диаметр колбы, мм": "Vat diameter, mm",
    "Параметры фотополимера": "Photopolymer parameters",
    "Базовое время засветки, с": "Base exposure time, s",
    "Интенсивность источника, %": "Light intensity, %",
    "Порог полимеризации, %": "Polymerization threshold, %",
    "Порог и интенсивность определяют итоговую\nконтрастность и яркость кадров.":
        "Threshold and intensity control the final\ncontrast and brightness of the frames.",
    "Разрешение и кадры": "Resolution and frames",
    "Разрешение сетки (Voxel Grid)": "Voxel grid resolution",
    "Разрешение кадра, px": "Frame resolution, px",
    "Количество кадров": "Number of frames",
    "Сплошная заливка (ремонт сетки)": "Fill holes (mesh repair)",
    "Preserve internal voids (bores / threads)": "Preserve internal voids (bores / threads)",
    "Объект": "Object",
    "Модель": "Model",
    "Файл не выбран": "No file selected",
    "Размер, мм": "Size, mm",
    "X, мм": "X, mm",
    "Y, мм": "Y, mm",
    "Z, мм": "Z, mm",
    "мм": "mm",
    "Вершин:": "Vertices:",
    "Граней:": "Faces:",
    "Исходные размеры:": "Original size:",
    "Трансформация": "Transform",
    "Инструмент:": "Tool:",
    "Перемещение": "Move",
    "Вращение": "Rotate",
    "Масштаб": "Scale",
    "Позиция": "Position",
    "Размер": "Size",
    "Тяни цветные стрелки, кольца или квадратные ручки прямо на модели.":
        "Drag the colored arrows, rings, or square handles directly on the model.",
    "Uniform Scale (сохранять пропорции)": "Uniform scale (keep proportions)",
    "Пропорционально": "Keep proportions",
    "Равномерное масштабирование": "Uniform scaling",
    "Сохранять пропорции при масштабировании": "Keep proportions while scaling",
    "Переместить модель в центр XY": "Move the model to the XY center",
    "Вписать модель в текущий размер колбы": "Fit the model to the current vat size",
    "Поворот, °": "Rotation, °",
    "Позиция, мм": "Position, mm",
    "Сдвиг X": "Offset X",
    "Сдвиг Y": "Offset Y",
    "Сдвиг Z": "Offset Z",
    "Центрировать": "Center",
    "Авто-фит под колбу": "Fit to vat",
    "Сбросить": "Reset",
    "📂 Загрузить STL": "📂 Load STL",
    "↺ Сбросить": "↺ Reset",
    "▶ Сгенерировать проекции": "▶ Generate projections",
    "📁 Открыть папку": "📁 Open output folder",
    "Настройка модели и генерация проекций": "Set up the model and generate projections",
    "Загрузка STL, настройка модели и генерация проекций":
        "Load STL, configure the model, and generate projections",
    "Проигрывание и экспорт готовых кадров в MP4": "Play and export generated frames to MP4",
    "Обратная реконструкция геометрии по кадрам": "Reconstruct geometry from projection frames",
    "Оптический расчёт и обратная реконструкция по кадрам":
        "Optical calculation and inverse reconstruction from frames",
    "VAMToolbox — альтернативный движок": "VAMToolbox — alternate engine",
    "Скачать и установить VAMToolbox": "Download and install VAMToolbox",
    "VAMToolbox уже доступен": "VAMToolbox is already available",
    "Проверить окружение VAMToolbox": "Verify VAMToolbox environment",
    "🧊 Слайсер": "🧊 Slicer",
    "🧪 Симуляция": "🧪 Simulation",
    "🎬 Проектор (Видео)": "🎬 Projector (Video)",
    "🔬 Симулятор": "🔬 Simulator",
    "🔭 Оптика": "🔭 Optics",
    "Оптика и реконструкция без расхода материала":
        "Optics and reconstruction without consuming material",
    "Симуляция без расхода материала": "Simulation without consuming material",
    "Один рабочий поток для проверки геометрии, преломления в колбе и водяном "
    "компенсаторе, а также ожидаемого результата печати.":
        "One workflow for checking geometry, refraction in the vat and water "
        "compensator, and the expected print result.",
    "Откройте рабочий процесс и загрузите STL-модель, чтобы начать.":
        "Open the workflow and load an STL model to begin.",
    "Готово к работе. Откройте рабочий процесс и загрузите STL-модель, чтобы начать.":
        "Ready. Open the workflow and load an STL model to begin.",
    "Порядок работы": "Workflow",
    "1. Оптический стенд": "1. Optical bench",
    "2. Результат": "2. Result",
    "1. Модель и кадры": "1. Model and frames",
    "2. Оптический стенд": "2. Optical bench",
    "3. Результат": "3. Result",
    "Загрузите STL, впишите модель в колбу и сгенерируйте набор проекций.":
        "Load an STL, fit the model to the vat, and generate projections.",
    "Подберите параметры колбы, квадратного аквариума и воды; сравните ход лучей и проекцию.":
        "Tune the vat, square tank, and water; compare ray paths and projection.",
    "Запустите обратную реконструкцию и проверьте, какую геометрию даст выбранный набор кадров.":
        "Run inverse reconstruction and inspect the geometry produced by the selected frames.",
    "Модель и проекции": "Model and projections",
    "Результат": "Result",
    "Проекции": "Projections",
    "не загружена": "not loaded",
    "не созданы": "not created",
    "не получены": "not available",
    "не рассчитан": "not calculated",
    "тестовый стенд": "test bench",
    "папка задана": "folder set",
    "см. вкладку «Слайсер»": "see the Slicer tab",
    "расчёт выполнен": "calculated",
    "готов к просмотру": "ready to inspect",
    "← Назад": "← Back",
    "▶ Сгенерировать кадры": "▶ Generate frames",
    "⏹ Остановить генерацию": "⏹ Stop generation",
    "Далее →": "Next →",
    "К оптике →": "To optics →",
    "К результату →": "To result →",
    "Сначала сгенерируйте проекции на вкладке «Слайсер». Здесь можно подобрать преломление в основной колбе и квадратном водяном компенсаторе, затем проверить ожидаемую геометрию результата.":
        "First generate projections on the Slicer tab. Then tune refraction in the main vat and square water compensator and inspect the expected result geometry.",
    "Настройте преломление и при необходимости возьмите последний кадр со вкладки «Слайсер».":
        "Tune refraction and, if needed, use the latest frame from the Slicer tab.",
    "Сначала сгенерируйте проекции на вкладке «Слайсер» или загрузите кадр вручную.":
        "First generate projections on the Slicer tab or load a frame manually.",
    "Здесь видно, какую геометрию даст выбранный набор проекций; порог поверхности меняется быстро.":
        "Inspect the geometry produced by the selected projections; the surface threshold changes quickly.",
    "Язык:": "Language:",
    "Готово к работе.": "Ready.",
    "Готово к работе. Загрузите STL-модель на вкладке «Слайсер», чтобы начать.":
        "Ready. Load an STL model on the Slicer tab to begin.",
    "Папка кадров не выбрана — сначала сгенерируйте проекции на вкладке «Слайсер».":
        "No frame folder selected — generate projections on the Slicer tab first.",
    "Нажмите «Собрать и воспроизвести»,\nчтобы увидеть анимацию проекций.":
        "Click “Assemble and play”\nto preview the projection animation.",
    "▶ Собрать и воспроизвести": "▶ Assemble and play",
    "Прочитать все кадры из папки и запустить проигрывание":
        "Read all frames from the folder and start playback",
    "⏸ Пауза": "⏸ Pause",
    "💾 Сохранить в MP4": "💾 Save as MP4",
    "Экспортировать уже собранные кадры в видеофайл":
        "Export assembled frames to a video file",
    "Скорость воспроизведения": "Playback speed",
    "🔬 Симулировать результат": "🔬 Simulate result",
    "Обратная Radon-реконструкция геометрии по кадрам":
        "Reconstruct geometry from frames with inverse Radon",
    "Оптический стенд": "Optical bench",
    "Основная колба": "Main vat",
    "Квадратный аквариум-компенсатор": "Square water compensator",
    "Материалы основной колбы": "Main vat materials",
    "Проектор и расчёт": "Projector and calculation",
    "Источник проекции": "Projection source",
    "Учитывать аквариум, заполненный водой": "Include the water-filled aquarium",
    "Наружный диаметр колбы, мм": "Vat outer diameter, mm",
    "Толщина стенки колбы, мм": "Vat wall thickness, mm",
    "Высота колбы, мм": "Vat height, mm",
    "Высота оптического среза, мм": "Optical slice height, mm",
    "Внутренняя сторона аквариума, мм": "Aquarium inner side, mm",
    "Толщина стенки аквариума, мм": "Aquarium wall thickness, mm",
    "Уровень воды от дна, мм": "Water level from bottom, mm",
    "Показатель преломления стенки n": "Wall refractive index n",
    "Показатель преломления воды n": "Water refractive index n",
    "Показатель преломления смолы n": "Resin refractive index n",
    "Поглощение воды, 1/мм": "Water absorption, 1/mm",
    "Поглощение смолы, 1/мм": "Resin absorption, 1/mm",
    "Расстояние до центра, мм": "Distance to center, mm",
    "Поле зрения по горизонтали, °": "Horizontal field of view, °",
    "Количество лучей": "Ray count",
    "Разрешение среза, px": "Slice resolution, px",
    "Загрузить кадр": "Load frame",
    "Взять последний кадр": "Use latest frame",
    "Показывать ход лучей": "Show ray paths",
    "Пересчитывать автоматически при изменении": "Recalculate automatically on change",
    "🔭 Рассчитать оптическую проекцию": "🔭 Calculate optical projection",
    "Расчёт преломлений в колбе и квадратном водяном компенсаторе":
        "Calculate refraction in the vat and square water compensator",
    "Выбрать кадр": "Choose a frame",
    "Изображения (*.png *.jpg *.jpeg *.bmp)": "Images (*.png *.jpg *.jpeg *.bmp)",
    "Порог визуализации (изоповерхность)": "Visualization threshold (isosurface)",
    "Пересчитывает только поверхность — без повторной реконструкции":
        "Recompute only the surface — no repeat reconstruction",
    "Реконструкция через обратное Radon-преобразование (FBP) — визуальный "
    "предпросмотр ожидаемой геометрии, не метрологическая симуляция полимеризации.":
        "Inverse Radon (FBP) reconstruction — a visual preview of the "
        "expected geometry, not a metrological cure simulation.",
    "Открыть STL-файл модели": "Open an STL model file",
    "Сбросить трансформацию и заново вписать модель в колбу":
        "Reset the transform and fit the model to the vat again",
    "Запустить расчёт проекций в фоновом потоке":
        "Start projection generation in a background thread",
    "Открыть последнюю папку output_frames в проводнике":
        "Open the latest output_frames folder in Explorer",
    "Папка вывода ещё не создана.": "The output folder has not been created yet.",
    "Сначала загрузите STL-модель.": "Load an STL model first.",
    "Выбрать STL": "Choose an STL file",
    "STL файлы (*.stl)": "STL files (*.stl)",
    "Успех": "Success",
    "Ошибка": "Error",
    "Внимание": "Warning",
    "Projection engine": "Projection engine",
    "SpinSlicer internal Radon (recommended)":
        "SpinSlicer internal Radon (recommended)",
    "Auto: VAMToolbox CAL → SpinSlicer fallback (experimental)":
        "Auto: VAMToolbox CAL → SpinSlicer fallback (experimental)",
    "SpinSlicer internal Radon is the default engine. VAMToolbox is available only when explicitly selected.":
        "SpinSlicer internal Radon is the default engine. VAMToolbox is available only when explicitly selected.",
    "Auto: VAMToolbox CAL → Radon fallback": "Auto: VAMToolbox CAL → Radon fallback",
    "Internal Radon": "Internal Radon",
    "VAMToolbox CAL (optional)": "VAMToolbox CAL (optional)",
    "VAM optimizer iterations": "VAM optimizer iterations",
    "Auto uses VAMToolbox when installed and falls back to the internal "
    "Radon projector otherwise.":
        "Auto uses VAMToolbox when installed and falls back to the internal "
        "Radon projector otherwise.",
    "Перезапустите приложение, чтобы применить язык.":
        "Restart the application to apply the selected language.",
    "Язык изменён": "Language changed",
    "Parametric threaded nut": "Parametric threaded nut",
}

_RU: dict[str, str] = {
    "Projection engine": "Движок проекций",
    "SpinSlicer internal Radon (recommended)":
        "Внутренний Radon SpinSlicer (рекомендуется)",
    "Auto: VAMToolbox CAL → SpinSlicer fallback (experimental)":
        "Авто: VAMToolbox CAL → запасной SpinSlicer (экспериментально)",
    "SpinSlicer internal Radon is the default engine. VAMToolbox is available only when explicitly selected.":
        "Внутренний Radon SpinSlicer используется по умолчанию. VAMToolbox доступен только при явном выборе.",
    "Auto: VAMToolbox CAL → Radon fallback": "Авто: VAMToolbox CAL → запасной Radon",
    "Internal Radon": "Внутренний Radon",
    "VAMToolbox CAL (optional)": "VAMToolbox CAL (опционально)",
    "VAM optimizer iterations": "Итерации оптимизатора VAM",
    "Auto uses VAMToolbox when installed and falls back to the internal "
    "Radon projector otherwise.":
        "Авто использует VAMToolbox при наличии и иначе переключается на "
        "внутренний Radon-проектор.",
    "Preserve internal voids (bores / threads)":
        "Сохранять внутренние пустоты (отверстия / резьба)",
    "Generate a parametric nut with a helical internal thread":
        "Создать параметрическую гайку с винтовой внутренней резьбой",
    "Generate a watertight nut with a helical internal thread. "
    "Dimensions are in millimetres; clearance is radial.":
        "Создать герметичный корпус гайки с винтовой внутренней резьбой. "
        "Размеры указаны в миллиметрах; зазор — радиальный.",
    "Generate a parametric nut with a helical internal thread. "
    "Dimensions are in millimetres; clearance is radial.":
        "Создайте параметрическую гайку с винтовой внутренней резьбой. "
        "Размеры указаны в миллиметрах; зазор — радиальный.",
    "Outer diameter, mm": "Наружный диаметр, мм",
    "Thread bore diameter, mm": "Диаметр резьбового отверстия, мм",
    "Nut height, mm": "Высота гайки, мм",
    "Thread pitch, mm": "Шаг резьбы, мм",
    "Thread depth (radial), mm": "Глубина резьбы (радиальная), мм",
    "Radial clearance, mm": "Радиальный зазор, мм",
    "Angular samples / turn": "Угловых отсчётов / оборот",
    "Axial samples / pitch": "Осевых отсчётов / шаг",
    "For a first print, keep at least 2–3 voxels across the thread depth "
    "and use a pitch larger than the voxel size.":
        "Для первой печати оставьте минимум 2–3 вокселя по глубине резьбы "
        "и задайте шаг больше размера вокселя.",
    "Create a threaded nut": "Создать гайку с резьбой",
    "Invalid nut parameters": "Некорректные параметры гайки",
    "Parametric threaded nut": "Параметрическая гайка с резьбой",
}

_PREFIX_EN: tuple[tuple[str, str], ...] = (
    ("Папка кадров: ", "Frames folder: "),
    ("Загружено: ", "Loaded: "),
    ("Сохранено ", "Saved "),
    ("Собрано ", "Assembled "),
    ("ОШИБКА", "ERROR"),
)


def set_language(language: str) -> None:
    """Select ``en`` or ``ru`` for subsequently created/updated UI text."""

    global _language
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    _language = language


def language() -> str:
    return _language


def tr(text: str) -> str:
    """Translate a source string while keeping Russian as the fallback."""

    if _language == "ru":
        return _RU.get(text, text)
    if text in _EN:
        return _EN[text]
    for source, translated in _PREFIX_EN:
        if text.startswith(source):
            return translated + text[len(source):]
    return text


def apply_translations(root: Any) -> None:
    """Translate static Qt widget text after a window has been built."""

    from PyQt6.QtCore import QObject
    from PyQt6.QtWidgets import QAbstractButton, QComboBox, QGroupBox, QLabel, QTabWidget

    widgets = [root, *root.findChildren(QObject)]
    for widget in widgets:
        if not hasattr(widget, "property"):
            continue

        if isinstance(widget, (QAbstractButton, QLabel)):
            source = widget.property("_spinslicer_i18n_text")
            if source is None:
                source = widget.text()
                widget.setProperty("_spinslicer_i18n_text", source)
            widget.setText(tr(str(source)))
        elif isinstance(widget, QGroupBox):
            source = widget.property("_spinslicer_i18n_title")
            if source is None:
                source = widget.title()
                widget.setProperty("_spinslicer_i18n_title", source)
            widget.setTitle(tr(str(source)))

        for prop in ("windowTitle", "toolTip", "statusTip", "whatsThis"):
            getter = getattr(widget, prop, None)
            setter = getattr(widget, f"set{prop[0].upper()}{prop[1:]}", None)
            if not callable(getter) or not callable(setter):
                continue
            property_name = f"_spinslicer_i18n_{prop}"
            source = widget.property(property_name)
            if source is None:
                source = getter()
                widget.setProperty(property_name, source)
            if source:
                setter(tr(str(source)))

        if isinstance(widget, QComboBox):
            sources = widget.property("_spinslicer_i18n_items")
            if sources is None:
                sources = [widget.itemText(i) for i in range(widget.count())]
                widget.setProperty("_spinslicer_i18n_items", sources)
            for index, source in enumerate(sources):
                widget.setItemText(index, tr(str(source)))

        if isinstance(widget, QTabWidget):
            sources = widget.property("_spinslicer_i18n_tabs")
            if sources is None:
                sources = [widget.tabText(i) for i in range(widget.count())]
                widget.setProperty("_spinslicer_i18n_tabs", sources)
            for index, source in enumerate(sources):
                widget.setTabText(index, tr(str(source)))

            tooltip_sources = widget.property("_spinslicer_i18n_tab_tooltips")
            if tooltip_sources is None:
                tooltip_sources = [widget.tabToolTip(i) for i in range(widget.count())]
                widget.setProperty("_spinslicer_i18n_tab_tooltips", tooltip_sources)
            for index, source in enumerate(tooltip_sources):
                if source:
                    widget.setTabToolTip(index, tr(str(source)))
