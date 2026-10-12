"""Demo presets: test-split patients (plus one clearly labelled synthetic out-of-domain case), written by
ml/train.py to ml/artifacts/presets.json using a fixed selection rule (see ml/train.py:build_presets)."""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.schemas import Preset


def load_presets(art_dir: Path) -> list[Preset]:
    return [Preset(**p) for p in json.loads((Path(art_dir) / "presets.json").read_text(encoding="utf-8"))]
