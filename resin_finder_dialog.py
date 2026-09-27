"""Beginner-facing automatic resin search and parameter sweep dialog."""

from __future__ import annotations

from html import escape

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from light_source_simulation import LightBudgetInput
from resin_finder import (
    EXPOSURE_SWEEP_S,
    TRANSMISSION_SWEEP_PCT,
    ResinSearchResult,
    run_resin_search,
)

_STATUS_COLOR = {
    "cal_reference": QColor("#1d6b54"),
    "edge_candidate": QColor("#755d2c"),
    "spectral_candidate": QColor("#236b70"),
    "weak_match": QColor("#743f46"),
}


class ResinFinderDialog(QDialog):
    """Run a no-input evidence screen for the user's 450 nm laser path."""

    def __init__(self, source: LightBudgetInput | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source = source or LightBudgetInput()
        self._result: ResinSearchResult | None = None
        self.setWindowTitle("Автоподбор смолы для лазера и проектора")
        self.resize(1080, 820)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(10)

        title = QLabel("Автоматический поиск смолы")
        title.setObjectName("panelTitle")
        root.addWidget(title)

        explanation = QLabel(
            "Система сама просматривает каталог материалов и перебирает 30 сочетаний "
            "потерь света и выдержки. Вам не нужно угадывать свойства смолы. "
            "Если порог отверждения неизвестен, программа не объявляет печать успешной."
        )
        explanation.setObjectName("hintLabel")
        explanation.setWordWrap(True)
        root.addWidget(explanation)

        self.summary = QTextBrowser()
        self.summary.setOpenExternalLinks(True)
        self.summary.setMaximumHeight(195)
        self.summary.setObjectName("hintLabel")
        root.addWidget(self.summary)

        candidate_box = QGroupBox("Результат поиска материалов")
        candidate_layout = QVBoxLayout(candidate_box)
        self.candidates = QTableWidget(0, 5)
        self.candidates.setHorizontalHeaderLabels(
            ("Приоритет", "Материал", "Совпадение", "Что подтверждено", "Что пока неизвестно")
        )
        self.candidates.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.candidates.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidates.setWordWrap(True)
        self.candidates.setAlternatingRowColors(True)
        candidates_vertical_header = self.candidates.verticalHeader()
        if candidates_vertical_header is not None:
            candidates_vertical_header.setVisible(False)
        for column, width in enumerate((95, 245, 155, 310, 310)):
            self.candidates.setColumnWidth(column, width)
        candidates_header = self.candidates.horizontalHeader()
        if candidates_header is not None:
            candidates_header.setStretchLastSection(True)
        candidate_layout.addWidget(self.candidates)
        root.addWidget(candidate_box, 1)

        dose_box = QGroupBox("Перебор дозы: только условная равномерная засветка")
        dose_layout = QVBoxLayout(dose_box)
        dose_note = QLabel(
            f"Для расчёта взяты мощность {self._source.power_min_w:g}–"
            f"{self._source.power_max_w:g} Вт и поле {self._source.image_width_mm:g} × "
            f"{self._source.image_height_mm:g} мм. "
            "Столбцы — предполагаемые потери по тракту, строки — выдержка. Это верхняя "
            "оценка I×t, а не доза в CAL-объёме и не обещание, что смола застынет."
        )
        dose_note.setObjectName("hintLabel")
        dose_note.setWordWrap(True)
        dose_layout.addWidget(dose_note)
        self.dose_table = QTableWidget(0, 1 + len(EXPOSURE_SWEEP_S))
        self.dose_table.setHorizontalHeaderLabels(
            ("Свет дошёл", *(f"{duration:g} с" for duration in EXPOSURE_SWEEP_S))
        )
        self.dose_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        dose_vertical_header = self.dose_table.verticalHeader()
        if dose_vertical_header is not None:
            dose_vertical_header.setVisible(False)
        dose_header = self.dose_table.horizontalHeader()
        if dose_header is not None:
            dose_header.setStretchLastSection(True)
        dose_layout.addWidget(self.dose_table)
        root.addWidget(dose_box)

        missing_box = QGroupBox("Почему это пока предварительный ответ")
        missing_layout = QVBoxLayout(missing_box)
        self.missing = QLabel()
        self.missing.setWordWrap(True)
        self.missing_layout_label = self.missing
        missing_layout.addWidget(self.missing)
        root.addWidget(missing_box)

        footer = QHBoxLayout()
        self.rerun = QPushButton("Перебрать ещё раз")
        self.rerun.clicked.connect(self._run)
        footer.addWidget(self.rerun)
        footer.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        footer.addWidget(buttons)
        root.addLayout(footer)

        self._run()

    def _run(self) -> None:
        try:
            result = run_resin_search(self._source)
        except ValueError as exc:
            self.summary.setPlainText(f"Не удалось выполнить поиск: {exc}")
            return
        self._result = result
        self._render_summary(result)
        self._render_candidates(result)
        self._render_dose_sweep(result)
        self.missing.setText("\n".join(f"• {item}" for item in result.missing_for_prediction))

    def _render_summary(self, result: ResinSearchResult) -> None:
        cal = next((item for item in result.assessments if item.status == "cal_reference"), None)
        retail = next((item for item in result.assessments if item.status == "edge_candidate"), None)
        ordinary = next((item for item in result.assessments if item.candidate.key == "generic_405_resin"), None)
        pieces = [
            (
                "<b>Для CAL лучший опубликованный ответ:</b> "
                f"{escape(cal.candidate.name) if cal else 'подходящего эталона нет'}. "
                "Это отдельная исследовательская UDMA-смола под 450 нм, не обычная готовая смола."
            ),
            (
                "<b>Первый готовый кандидат для вопроса поставщику:</b> "
                f"{escape(retail.candidate.name) if retail else 'не найден'}. "
                "Он рассчитан на Daylight-систему 460 нм; 450 нм находится у края полосы лазера, "
                "а печать CAL не проверена. Второй вариант — чёрный Rigid DLFR 460 нм; "
                "его оптическое затухание через колбу тоже неизвестно."
            ),
            (
                "<b>Обычная смола 405 нм:</b> "
                f"{escape(ordinary.status_label.lower()) if ordinary else 'нет данных'}. "
                "Поиск не нашёл подтверждённую обычную смолу, которую можно уверенно рекомендовать для 450 нм."
            ),
        ]
        summary = (
            f"<p><b>Перебор завершён:</b> спектр лазера {result.laser_band_nm[0]:g}–"
            f"{result.laser_band_nm[1]:g} нм, пик {result.laser_peak_nm:g} нм; "
            f"проверено {len(result.assessments)} профиля материалов и "
            f"{result.scenario_count} световых сценариев.</p>"
            + "".join(f"<p>{piece}</p>" for piece in pieces)
            + "<p><b>Вывод:</b> смола и настройки экспозиции не могут быть найдены одним "
            "числовым перебором, пока неизвестно, сколько света проходит через конкретную "
            "смолу и какая доза делает её твёрдой при 450 нм. Программа автоматически выбрала наиболее обоснованные "
            "кандидаты и показала, какие неизвестные мешают вынести вердикт о печати.</p>"
            '<p>Источники: '
            '<a href="https://assets.pubpub.org/5emoccq0/104%20An%20Open-Sourced%2C%20'
            'Community-Driven%20Volumetric%20Additive%20Manufacturing%20Printer%20'
            'and%20Post-Processor-31760729720398.pdf">OpenCAL / CAL-состав</a> · '
            '<a href="https://photocentricgroup.com/wp-content/uploads/2026/06/'
            'TDS-Rigid-DLFR-Black-2026.pdf">паспорт Rigid DLFR</a> · '
            '<a href="https://photocentricgroup.com/wp-content/uploads/2025/08/'
            'TDS-Magna-Draft-2025.pdf">паспорт Magna Draft</a> · '
            '<a href="https://photocentricgroup.com/product/magna-draft-3d-resin/">'
            'страница Magna Draft</a> · '
            '<a href="https://photocentricgroup.com/product/rigid-dlfr-3d-resin/">'
            'страница производителя</a> · '
            '<a href="https://www.liqcreate.com/supportarticles/daylight-405nm-385nm-365nm-resin/">'
            'спектральное замечание о смолах 405 нм</a></p>'
        )
        self.summary.setHtml(summary)

    def _render_candidates(self, result: ResinSearchResult) -> None:
        self.candidates.setRowCount(len(result.assessments))
        for row, assessment in enumerate(result.assessments, start=0):
            candidate = assessment.candidate
            if candidate.key == "opencal_udma_cq_edab":
                priority = "Эталон CAL"
            elif candidate.key in {"photocentric_rigid_dlfr", "photocentric_magna_draft"}:
                priority = "Купить / проверить"
            else:
                priority = "Не первой"
            values = (
                priority,
                candidate.name,
                assessment.status_label,
                candidate.evidence + "\n" + "\n".join(candidate.known),
                "\n".join(candidate.missing) + "\n" + candidate.recommendation,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                if column == 0:
                    item.setBackground(_STATUS_COLOR.get(assessment.status, QColor("#303846")))
                self.candidates.setItem(row, column, item)
            self.candidates.setRowHeight(row, 130)
        self.candidates.resizeRowsToContents()

    def _render_dose_sweep(self, result: ResinSearchResult) -> None:
        self.dose_table.setRowCount(len(TRANSMISSION_SWEEP_PCT))
        scenario_map = {
            (item.transmission_pct, item.exposure_s): item.dose_mj_cm2
            for item in result.dose_scenarios
        }
        for row, transmission in enumerate(TRANSMISSION_SWEEP_PCT):
            self.dose_table.setItem(row, 0, QTableWidgetItem(f"{transmission:g}%"))
            for column, duration in enumerate(EXPOSURE_SWEEP_S, start=1):
                low, high = scenario_map[(transmission, duration)]
                text = f"{low:.3g}–{high:.3g}"
                item = QTableWidgetItem(text)
                item.setToolTip(
                    f"Предполагаемая полная доза {text} мДж/см²; порог материала не задан."
                )
                self.dose_table.setItem(row, column, item)
        self.dose_table.resizeColumnsToContents()
