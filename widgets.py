"""
widgets.py
==========
Мелкие переиспользуемые Qt-виджеты. Главный — LabeledSlider: связка
QSlider + QDoubleSpinBox с двусторонней синхронизацией, аналог пары
"слайдер + Entry" из прототипа, но нативная для Qt и без ручного
парсинга строк.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class LabeledSlider(QWidget):
    """Подпись + слайдер + числовое поле, синхронизированные друг с другом."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int = 2,
        step: Optional[float] = None,
        suffix: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._decimals = decimals
        self._scale = 10 ** decimals
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.label = QLabel(label)
        self.label.setObjectName("fieldLabel")
        layout.addWidget(self.label)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(round(minimum * self._scale)))
        self.slider.setMaximum(int(round(maximum * self._scale)))
        self.slider.setValue(int(round(value * self._scale)))

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step if step is not None else max(1 / self._scale, 0.01))
        self.spin.setSuffix(suffix)
        self.spin.setValue(value)
        self.spin.setFixedWidth(88)

        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        layout.addLayout(row)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)

    # --- синхронизация -----------------------------------------------------
    def _on_slider_changed(self, raw: int) -> None:
        if self._updating:
            return
        self._updating = True
        v = raw / self._scale
        self.spin.setValue(v)
        self._updating = False
        self.valueChanged.emit(v)

    def _on_spin_changed(self, v: float) -> None:
        if self._updating:
            return
        self._updating = True
        self.slider.setValue(int(round(v * self._scale)))
        self._updating = False
        self.valueChanged.emit(v)

    # --- публичное API -----------------------------------------------------
    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v: float) -> None:
        self._updating = True
        self.spin.setValue(v)
        self.slider.setValue(int(round(v * self._scale)))
        self._updating = False

    def value_int(self) -> int:
        return int(round(self.spin.value()))
