"""Running the training pipeline twice gives byte-identical metrics.json (quick settings for test speed;
the full run was also checked twice, see docs/DECISIONS.md)."""

import pytest

from ml.train import main


@pytest.mark.slow
def test_train_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    main(quick=True, out_dir=a, reports=tmp_path / "ra", n_jobs=2)
    main(quick=True, out_dir=b, reports=tmp_path / "rb", n_jobs=2)
    assert (a / "metrics.json").read_bytes() == (b / "metrics.json").read_bytes()
    for f in ["thresholds.json", "conformal.json", "presets.json", "global_explanations.json", "calibrator.json"]:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
