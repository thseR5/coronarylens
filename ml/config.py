"""Load the four YAML registries (single sources of truth)."""

from __future__ import annotations

import copy
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_PATH = ROOT / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"
ARTIFACTS = ROOT / "ml" / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

NAMES = ("features", "targets", "anatomy", "models")


@cache
def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load(name: str) -> dict[str, Any]:
    """Return a deep copy so callers can never mutate the cached config."""
    if name not in NAMES:
        raise KeyError(name)
    return copy.deepcopy(_load(name))


def features_cfg() -> dict[str, Any]:
    return load("features")


def targets_cfg() -> dict[str, Any]:
    return load("targets")


def models_cfg() -> dict[str, Any]:
    return load("models")


def all_configs() -> dict[str, Any]:
    return {n: load(n) for n in NAMES}


def seed() -> int:
    return int(models_cfg()["seed"])


def vessel_ids(tcfg: dict | None = None) -> list[str]:
    tcfg = tcfg or targets_cfg()
    return [v["id"] for v in tcfg["vessels"]]


def pattern_strings(tcfg: dict | None = None) -> list[str]:
    """All 2^k pattern strings ordered by code (code = int(pattern, 2))."""
    k = len(vessel_ids(tcfg))
    return [format(c, f"0{k}b") for c in range(2**k)]
