import copy

import pandas as pd
import pytest

from ml import config
from ml.ingest import InputError, encode_input, imputation_values, load_dataset
from ml.leakage import LeakageError, assert_clean, proxy_scan


@pytest.mark.parametrize("col", ["LAD", "lad", "LCX", "rca", "Cath", "cath", " CATH", "cad", "Lad"])
def test_forbidden_columns_rejected(col):
    with pytest.raises(LeakageError):
        assert_clean(pd.DataFrame({"age": [1], col: [1]}))


def test_clean_frame_passes(dataset):
    assert_clean(dataset.X)


def test_training_rejects_feature_mapped_to_label(dataset):
    fcfg = copy.deepcopy(config.features_cfg())
    fcfg["features"].append({"id": "innocent", "column": "LAD", "label": "x", "unit": "", "type": "binary",
                             "encoding": "yn", "group": "labs", "range": [0, 1]})
    with pytest.raises(LeakageError):
        load_dataset(fcfg=fcfg, raw=dataset.raw)


@pytest.mark.parametrize("key", ["LAD", "LCX", "RCA", "Cath", "cath"])
def test_inference_rejects_forbidden(dataset, key):
    imp = imputation_values(dataset.X)
    with pytest.raises(InputError):
        encode_input({key: 1}, imp, None)


def test_proxy_scan_flags_label_copy(dataset):
    X = dataset.X.copy()
    y = dataset.T["cad"].to_numpy()
    assert proxy_scan(X, y) == {}
    X["copy"] = y
    assert "copy" in proxy_scan(X, y)
