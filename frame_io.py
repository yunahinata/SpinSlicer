"""
frame_io.py
===========
Общий слой для вкладок "Проектор" и "Симулятор": чтение списка кадров
из output_frames/ и метаданных генерации (slice_meta.json).

Зачем нужны метаданные: PNG-кадры сами по себе не хранят диаметр колбы
или разрешение вокселя (grid_res), с которым они были посчитаны. Без
этого "Симулятор" не смог бы correctно перевести пиксели обратно в
физические миллиметры при обратной Radon-реконструкции. Поэтому
SlicingEngine сохраняет slice_meta.json рядом с кадрами при генерации.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

META_FILENAME = "slice_meta.json"


@dataclass
class SliceMeta:
    diameter_mm: float
    grid_res: int
    output_res: int
    num_frames: int
    fill_holes: bool
    resin_base_exposure: float
    resin_intensity: float
    resin_threshold: float
    generated_at: str = ""


def save_meta(out_dir: str, meta: SliceMeta) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, META_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)


def load_meta(out_dir: str) -> Optional[SliceMeta]:
    path = os.path.join(out_dir, META_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return SliceMeta(**raw)
    except Exception:
        return None


def list_frame_paths(out_dir: str) -> List[str]:
    """Отсортированный список путей к frame_XXXX.png в папке."""
    return sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
