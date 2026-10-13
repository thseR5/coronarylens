import json

from ml import config
from ml.input_confidence import LOW_FROM, REDUCED_FROM, UNCERTAIN_FROM, input_confidence, missing_signal_share

IMP = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}


def test_share_is_importance_weighted_and_bounded():
    assert missing_signal_share([], IMP) == 0.0
    assert missing_signal_share(["d"], IMP) == 0.05
    assert missing_signal_share(["a", "a"], IMP) == 0.5  # duplicates count once
    assert missing_signal_share(list(IMP), IMP) == 1.0
    assert missing_signal_share(["unknown"], IMP) == 0.0


def test_share_grows_monotonically_with_each_missing_input():
    order = ["d", "c", "b", "a"]
    shares = [missing_signal_share(order[:k], IMP) for k in range(len(order) + 1)]
    assert shares == sorted(shares)


def test_levels_and_forced_uncertainty():
    assert REDUCED_FROM < UNCERTAIN_FROM < LOW_FROM
    assert input_confidence([], IMP) == {"missing_signal_share": 0.0, "level": "full", "forces_uncertain": False}
    assert input_confidence(["c"], IMP)["level"] == "reduced"  # 0.15
    assert not input_confidence(["c"], IMP)["forces_uncertain"]
    assert input_confidence(["b"], IMP)["forces_uncertain"]  # 0.30
    assert input_confidence(["a"], IMP)["level"] == "low"  # 0.50


def test_served_importance_covers_every_feature():
    g = json.loads((config.ARTIFACTS / "global_explanations.json").read_text())
    imp = g["targets"]["CAD"]["mean_abs_shap"]
    ids = {f["id"] for f in config.features_cfg()["features"]}
    assert set(imp) == ids
    # all inputs missing -> everything comes from cohort averages
    assert input_confidence(ids, imp)["level"] == "low"


def test_status_reason_matches_reconciled_status():
    import numpy as np

    from ml.conformal import reconciled_status, status_reason

    q = {"0": 0.6, "1": 0.55}  # set: in0 if p <= 0.6, in1 if p >= 0.45
    for p in np.linspace(0, 1, 101):
        for positive in (False, True):
            s, r = reconciled_status(p, q, positive), status_reason(p, q, positive)
            assert (s != "uncertain") == (r == "single_outcome"), (p, positive, s, r)
    assert status_reason(0.5, q, True) == "both_outcomes"
    assert status_reason(0.9, q, False) == "threshold_disagrees"
    assert status_reason(0.5, {"0": 0.1, "1": 0.1}, True) == "empty_set"
