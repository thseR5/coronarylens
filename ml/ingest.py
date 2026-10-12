"""Load the Z-Alizadeh Sani xlsx and encode it into model-ready columns driven by config/features.yaml.

Encoding rules
- binary yn: Y -> 1, N -> 0 · binary 01: as is · binary sex: Male -> 1, Fmale/Female -> 0
- ordinal: raw level -> index in `levels` (code 0..k)
- categorical: one-hot, first level is the baseline (bbb -> bbb_LBBB, bbb_RBBB)
- numeric: float
Targets: cad (Cath == positive), one bit per vessel, pattern string and pattern code.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml import config
from ml.leakage import assert_clean


class InputError(ValueError):
    """Invalid model input (unknown id, forbidden column, bad value, outside plausibility range)."""


@dataclass
class Dataset:
    X: pd.DataFrame  # model columns (encoded)
    T: pd.DataFrame  # cad, LAD, LCX, RCA, pattern, pattern_code
    raw: pd.DataFrame  # original sheet


# ----------------------------------------------------------------------------- column layout
def model_columns(fcfg: dict | None = None) -> list[str]:
    fcfg = fcfg or config.features_cfg()
    cols: list[str] = []
    for f in fcfg["features"]:
        if f["type"] == "categorical":
            cols += [f"{f['id']}_{lvl}" for lvl in f["levels"][1:]]
        else:
            cols.append(f["id"])
    return cols


def column_to_feature(fcfg: dict | None = None) -> dict[str, str]:
    fcfg = fcfg or config.features_cfg()
    out: dict[str, str] = {}
    for f in fcfg["features"]:
        if f["type"] == "categorical":
            for lvl in f["levels"][1:]:
                out[f"{f['id']}_{lvl}"] = f["id"]
        else:
            out[f["id"]] = f["id"]
    return out


def feature_ids(fcfg: dict | None = None) -> list[str]:
    fcfg = fcfg or config.features_cfg()
    return [f["id"] for f in fcfg["features"]]


def feature_map(fcfg: dict | None = None) -> dict[str, dict]:
    fcfg = fcfg or config.features_cfg()
    return {f["id"]: f for f in fcfg["features"]}


# ----------------------------------------------------------------------------- value encoding
def _encode_binary(v: Any, f: dict) -> float:
    enc = str(f.get("encoding", "01"))
    if isinstance(v, str):
        s = v.strip()
        if enc == "sex":
            m = {"male": 1.0, "m": 1.0, "fmale": 0.0, "female": 0.0, "f": 0.0}
        else:
            m = {"y": 1.0, "yes": 1.0, "n": 0.0, "no": 0.0, "1": 1.0, "0": 0.0}
        if s.lower() not in m:
            raise InputError(f"{f['id']}: invalid value {v!r}")
        return m[s.lower()]
    x = float(v)
    if x not in (0.0, 1.0):
        raise InputError(f"{f['id']}: binary feature must be 0 or 1, got {v!r}")
    return x


def _encode_ordinal(v: Any, f: dict) -> float:
    levels = list(f["levels"])
    if isinstance(v, str):
        for i, lvl in enumerate(levels):
            if str(lvl).lower() == v.strip().lower():
                return float(i)
        try:
            v = float(v)
        except ValueError as e:
            raise InputError(f"{f['id']}: unknown level {v!r}; allowed {levels}") from e
    x = float(v)
    # raw-coded ordinals (levels are numbers) map value -> index; string-coded accept the code itself
    if all(isinstance(lvl, (int, float)) for lvl in levels):
        if x not in [float(lvl) for lvl in levels]:
            raise InputError(f"{f['id']}: value {v!r} not in levels {levels}")
        return float([float(lvl) for lvl in levels].index(x))
    if not x.is_integer() or not 0 <= x < len(levels):
        raise InputError(f"{f['id']}: code {v!r} not in 0..{len(levels) - 1}")
    return x


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and v.strip() == ""


def encode_value(v: Any, f: dict) -> dict[str, float]:
    """Encode one raw value of feature f into its model column(s). Raises InputError."""
    t = f["type"]
    if isinstance(v, bool):
        v = int(v)
    if t == "categorical":
        levels = [str(x) for x in f["levels"]]
        s = str(v).strip()
        match = [lvl for lvl in levels if lvl.lower() == s.lower()]
        if not match:
            raise InputError(f"{f['id']}: unknown level {v!r}; allowed {levels}")
        return {f"{f['id']}_{lvl}": float(lvl == match[0]) for lvl in levels[1:]}
    if t == "binary":
        x = _encode_binary(v, f)
    elif t == "ordinal":
        x = _encode_ordinal(v, f)
    elif t == "numeric":
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError as e:
                raise InputError(f"{f['id']}: not a number: {v!r}") from e
        x = float(v)
        if not math.isfinite(x):
            raise InputError(f"{f['id']}: not finite")
    else:  # pragma: no cover - config error
        raise ValueError(f"unknown feature type {t}")
    rng = f.get("range")
    if rng is not None and not (float(rng[0]) <= x <= float(rng[1])):
        raise InputError(f"{f['id']}: value {v!r} outside plausibility range {rng}")
    return {f["id"]: x}


def decode_row(row: pd.Series | dict, fcfg: dict | None = None) -> dict[str, Any]:
    """Model columns -> API feature dict (categorical back to level string, vhd stays a code)."""
    fcfg = fcfg or config.features_cfg()
    out: dict[str, Any] = {}
    for f in fcfg["features"]:
        if f["type"] == "categorical":
            lvl = f["levels"][0]
            for c in f["levels"][1:]:
                if float(row[f"{f['id']}_{c}"]) >= 0.5:
                    lvl = c
            out[f["id"]] = str(lvl)
        elif f["type"] == "ordinal" and all(isinstance(x, (int, float)) for x in f["levels"]):
            out[f["id"]] = int(f["levels"][round(float(row[f["id"]]))])
        elif f["type"] in ("binary", "ordinal"):
            out[f["id"]] = round(float(row[f["id"]]))
        else:
            x = float(row[f["id"]])
            out[f["id"]] = int(x) if x.is_integer() else round(x, 4)
    return out


# ----------------------------------------------------------------------------- dataset
def read_raw(path=None) -> pd.DataFrame:
    return pd.read_excel(path or config.DATA_PATH)


def data_sha256(path=None) -> str:
    with open(path or config.DATA_PATH, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def encode_frame(raw: pd.DataFrame, fcfg: dict | None = None) -> pd.DataFrame:
    fcfg = fcfg or config.features_cfg()
    assert_clean([f["column"] for f in fcfg["features"]] + [f["id"] for f in fcfg["features"]])
    cols: dict[str, list[float]] = {c: [] for c in model_columns(fcfg)}
    for f in fcfg["features"]:
        if f["column"] not in raw.columns:
            raise KeyError(f"column {f['column']!r} for feature {f['id']} missing from data")
        for v in raw[f["column"]].tolist():
            if _is_missing(v):
                raise InputError(f"{f['id']}: missing value in source data")
            for c, x in encode_value(v, f).items():
                cols[c].append(x)
    X = pd.DataFrame(cols, index=raw.index, dtype=float)
    assert_clean(X)
    return X


def encode_targets(raw: pd.DataFrame, tcfg: dict | None = None) -> pd.DataFrame:
    tcfg = tcfg or config.targets_cfg()
    c = tcfg["cad"]
    T = pd.DataFrame(index=raw.index)
    T["cad"] = (raw[c["column"]].astype(str) == c["positive"]).astype(int)
    for v in tcfg["vessels"]:
        T[v["id"]] = (raw[v["column"]].astype(str) == v["positive"]).astype(int)
    vids = [v["id"] for v in tcfg["vessels"]]
    T["pattern"] = T[vids].astype(str).agg("".join, axis=1)
    k = len(vids)
    T["pattern_code"] = sum(T[v] * 2 ** (k - 1 - i) for i, v in enumerate(vids)).astype(int)
    return T


def load_dataset(fcfg: dict | None = None, tcfg: dict | None = None, raw: pd.DataFrame | None = None) -> Dataset:
    raw = read_raw() if raw is None else raw
    return Dataset(X=encode_frame(raw, fcfg), T=encode_targets(raw, tcfg), raw=raw)


# ----------------------------------------------------------------------------- inference input
def imputation_values(X: pd.DataFrame, fcfg: dict | None = None) -> dict[str, Any]:
    """Training medians (numeric) / modes (binary, ordinal, categorical) in API value space."""
    fcfg = fcfg or config.features_cfg()
    out: dict[str, Any] = {}
    for f in fcfg["features"]:
        fid = f["id"]
        if f["type"] == "numeric":
            out[fid] = float(np.median(X[fid]))
        elif f["type"] == "categorical":
            onehot = X[[f"{fid}_{lvl}" for lvl in f["levels"][1:]]].to_numpy()
            counts = [int((onehot.sum(1) == 0).sum())] + [int(x) for x in onehot.sum(0)]
            out[fid] = str(f["levels"][int(np.argmax(counts))])
        else:
            vals, counts = np.unique(X[fid].to_numpy(), return_counts=True)
            code = int(vals[np.argmax(counts)])
            lv = f.get("levels")
            out[fid] = int(lv[code]) if lv and all(isinstance(x, (int, float)) for x in lv) else code
    return out


def training_ranges(X: pd.DataFrame, fcfg: dict | None = None) -> dict[str, list[float]]:
    fcfg = fcfg or config.features_cfg()
    return {
        f["id"]: [float(X[f["id"]].min()), float(X[f["id"]].max())]
        for f in fcfg["features"]
        if f["type"] == "numeric"
    }


def encode_input(
    features: dict[str, Any],
    imputation: dict[str, Any],
    train_ranges: dict[str, list[float]] | None = None,
    fcfg: dict | None = None,
    tcfg: dict | None = None,
) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """API feature dict -> one encoded row. Returns (X_row, imputed ids, out_of_range flags)."""
    fcfg = fcfg or config.features_cfg()
    tcfg = tcfg or config.targets_cfg()
    forbidden = {c.lower() for c in tcfg["forbidden_columns"]}
    fmap = feature_map(fcfg)
    for key in features:
        if key.strip().lower() in forbidden:
            raise InputError(f"forbidden column {key!r} (labels never reach the model)")
        if key not in fmap:
            raise InputError(f"unknown feature id {key!r}")
    row: dict[str, float] = {}
    imputed: list[str] = []
    oor: list[dict] = []
    for fid, f in fmap.items():
        v = features.get(fid)
        if _is_missing(v):
            v = imputation[fid]
            imputed.append(fid)
        enc = encode_value(v, f)
        row.update(enc)
        if train_ranges and fid in train_ranges and fid not in imputed:
            lo, hi = train_ranges[fid]
            x = enc[fid]
            if x < lo or x > hi:
                oor.append({"feature": fid, "value": x, "range": (lo, hi)})
    X = pd.DataFrame([row], columns=model_columns(fcfg), dtype=float)
    assert_clean(X, tcfg)
    return X, imputed, oor
