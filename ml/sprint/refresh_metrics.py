"""Refresh the sprint-derived parts of ml/artifacts/metrics.json from the stored nested records, using exactly the
functions ml/train.py uses (so a full retrain produces the same values). No model is refit; the test set is untouched.

Usage: uv run python -m ml.sprint.refresh_metrics
"""

from __future__ import annotations

import json

import numpy as np

from ml import config
from ml.ingest import load_dataset, model_columns
from ml.sprint import headline as SH
from ml.sprint.core import labels
from ml.train import _dump


def main() -> None:
    path = config.ARTIFACTS / "metrics.json"
    m = json.loads(path.read_text("utf-8"))
    rec = SH.load("B1")
    m["cv"]["variants"]["sprint_procedure"] = SH.variant_summary(rec)
    served = m.get("sprint", {}).get("served_on_full_training_split")
    m["sprint"] = SH.sprint_section(rec) | ({"served_on_full_training_split": served} if served else {})
    fcfg = config.features_cfg()
    tr = np.asarray(json.loads((config.ARTIFACTS / "split.json").read_text("utf-8"))["train"])
    ds = load_dataset()
    Xtr, Ttr = ds.X.to_numpy(float)[tr], ds.T.iloc[tr].reset_index(drop=True)
    m["subgroups"]["groups"] = SH.subgroups(rec, Xtr, labels(Ttr), model_columns(fcfg), 5)
    _dump(m, path)  # same rounding/format as ml/train.py


if __name__ == "__main__":
    main()
