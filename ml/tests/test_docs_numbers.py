"""Every number in the judge-facing docs and dashboard text must be backed by a measured artifact."""

import importlib.util

from ml import config


def _checker():
    path = config.ROOT / "tools" / "check_numbers.py"
    spec = importlib.util.spec_from_file_location("check_numbers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_docs_numbers_match_artifacts(capsys):
    assert _checker().main() == 0, capsys.readouterr().out


def test_checker_catches_a_wrong_headline():
    c = _checker()
    ok = c.allowed()
    assert "0.708" in ok["LCX"] and "0.935" not in ok["CAD"] and "0.709" not in ok["LCX"]
    assert c.context("nested-CV AUC: LCX ") == "LCX" and c.context("log-loss p = ") == "p"
