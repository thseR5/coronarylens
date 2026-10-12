import copy

import numpy as np
import pytest

from ml import config
from ml.ingest import InputError, encode_input, imputation_values, load_dataset, model_columns


def test_verified_data_facts(dataset):
    X, T, raw = dataset.X, dataset.T, dataset.raw
    assert raw.shape == (303, 59)
    assert not raw.isna().any().any()
    assert (T["cad"].sum(), T["LAD"].sum(), T["LCX"].sum(), T["RCA"].sum()) == (216, 177, 119, 114)
    assert T["pattern"].value_counts().to_dict() == {"000": 86, "100": 57, "111": 63, "110": 33, "101": 24,
                                                     "001": 17, "010": 13, "011": 10}
    anyv = T[["LAD", "LCX", "RCA"]].max(axis=1)
    assert int((anyv == T["cad"]).sum()) == 302
    assert list(np.where(anyv != T["cad"])[0]) == [93]
    assert X.shape == (303, len(model_columns()))
    assert set(np.unique(X["sex_male"])) == {0.0, 1.0}
    assert {"bbb_LBBB", "bbb_RBBB"} <= set(X.columns)
    assert X["vhd"].max() == 3


def test_pattern_code_matches_bits(dataset):
    T = dataset.T
    assert (T["pattern_code"] == 4 * T["LAD"] + 2 * T["LCX"] + T["RCA"]).all()


def test_config_extensibility(dataset):
    """Adding a feature to an in-memory copy of features.yaml is picked up by ingest and inference encoding."""
    fcfg = copy.deepcopy(config.features_cfg())
    fcfg["features"].append({"id": "dummy_score", "column": "Dummy Score", "label": "Dummy", "unit": "",
                             "type": "numeric", "group": "labs", "range": [0, 10], "ref": None, "default": 1,
                             "modifiable": False, "monotone": 0, "quick": False, "description": "test only"})
    raw = dataset.raw.copy()
    raw["Dummy Score"] = np.arange(len(raw)) % 10
    ds = load_dataset(fcfg=fcfg, raw=raw)
    assert "dummy_score" in ds.X.columns and "dummy_score" in model_columns(fcfg)
    imp = imputation_values(ds.X, fcfg)
    x, imputed, _ = encode_input({"dummy_score": 3}, imp, None, fcfg)
    assert x["dummy_score"].iloc[0] == 3 and "dummy_score" not in imputed
    with pytest.raises(InputError):
        encode_input({"dummy_score": 11}, imp, None, fcfg)


def test_input_validation_and_imputation(dataset):
    imp = imputation_values(dataset.X)
    x, imputed, _oor = encode_input({"age": 60, "vhd": "Moderate", "bbb": "LBBB", "sex_male": 1}, imp, None)
    assert x["vhd"].iloc[0] == 2 and x["bbb_LBBB"].iloc[0] == 1 and x["bbb_RBBB"].iloc[0] == 0
    assert "age" not in imputed and "ldl" in imputed and len(imputed) == len(config.features_cfg()["features"]) - 4
    assert x["ldl"].iloc[0] == imp["ldl"]
    for bad in ({"age": 150}, {"bbb": "XYZ"}, {"sex_male": 0.5}, {"vhd": 7}, {"not_a_feature": 1}, {"ldl": "abc"}):
        with pytest.raises(InputError):
            encode_input(bad, imp, None)
