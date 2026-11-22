"""Tests for the static web app: the browser model (space/model.js + model.json)
must give the same predictions as the scikit-learn model. Requires Node.js."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from features import RAW_COLUMNS, build_features

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"
DATA = ROOT / "data" / "credit_default.csv"
META = json.loads((SPACE / "metadata.json").read_text())
NODE = shutil.which("node")


def js_scores(csv_path):
    if NODE is None:
        pytest.fail("Node.js is required to test the browser model")
    out = subprocess.run([NODE, str(ROOT / "tests" / "js_predict.mjs"), str(csv_path)],
                         capture_output=True, text=True, check=True)
    return np.array(json.loads(out.stdout))


@pytest.fixture(scope="module")
def sklearn_model():
    return joblib.load(ROOT / "model" / "model.joblib")


def test_browser_model_matches_sklearn_on_all_clients(sklearn_model):
    df = pd.read_csv(DATA)
    p_py = sklearn_model.predict_proba(build_features(df))[:, 1]
    p_js = js_scores(DATA)
    assert len(p_js) == len(df)
    assert np.abs(p_py - p_js).max() < 1e-9
    t = META["threshold"]
    assert ((p_py >= t) == (p_js >= t)).all()


def test_late_payer_is_flagged_and_scores_above_clean_payer(tmp_path):
    base = dict(LIMIT_BAL=100000, SEX=2, EDUCATION=2, MARRIAGE=2, AGE=35,
                **{f"BILL_AMT{i}": 20000 for i in range(1, 7)},
                **{f"PAY_AMT{i}": 2000 for i in range(1, 7)})
    pays = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    clean = {**base, **{c: -1 for c in pays}}
    late = {**base, **dict(zip(pays, [3, 2, 2, 2, 2, 2]))}
    path = tmp_path / "two.csv"
    pd.DataFrame([clean, late], columns=RAW_COLUMNS).to_csv(path, index=False)
    p_clean, p_late = js_scores(path)
    assert p_late > p_clean
    assert p_late >= META["threshold"] > p_clean


def test_space_is_configured_as_free_static_site():
    readme = (SPACE / "README.md").read_text()
    assert re.search(r"^sdk: static$", readme, re.M)
    assert re.search(r"^app_file: index.html$", readme, re.M)
    html = (SPACE / "index.html").read_text()
    for needed in ("model.js", "model.json", "metadata.json"):
        assert needed in html and (SPACE / needed).exists()
