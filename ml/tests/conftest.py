import json

import numpy as np
import pytest

from ml import config
from ml.ingest import load_dataset

ART = config.ARTIFACTS


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture(scope="session")
def served():
    if not (ART / "joint_model.joblib").exists():
        pytest.skip("artifacts missing: run `uv run python -m ml.train`")
    from ml.predictor import ServedModel

    return ServedModel(ART)


@pytest.fixture(scope="session")
def split():
    return json.loads((ART / "split.json").read_text())


@pytest.fixture(scope="session")
def test_X(dataset, split):
    return dataset.X.to_numpy(dtype=float)[np.array(split["test"])]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs the (quick) training pipeline twice")
