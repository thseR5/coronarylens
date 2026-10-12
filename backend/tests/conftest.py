import warnings

import pytest

from ml import config


@pytest.fixture(scope="session")
def client():
    if not (config.ARTIFACTS / "joint_model.joblib").exists():
        pytest.skip("artifacts missing: run `uv run python -m ml.train`")
    with warnings.catch_warnings():
        # third-party: starlette warns that its TestClient's httpx backend is deprecated; not our code
        warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
        from fastapi.testclient import TestClient
    from backend.app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def presets(client):
    return client.get("/api/presets").json()
