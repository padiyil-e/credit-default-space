"""Quality gate: the committed model must load and still meet Module 8 performance."""
import json
import re
from pathlib import Path

import joblib
import pytest
import sklearn
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from features import FEATURE_COLUMNS
from train import load_split

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"
MODEL = ROOT / "model" / "model.joblib"
META = json.loads((SPACE / "metadata.json").read_text())

# Floors set a little below the Module 8 test results (AUC 0.775, recall 0.542, F1 0.530)
MIN_AUC, MIN_RECALL, MIN_F1 = 0.76, 0.50, 0.50


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL)


@pytest.fixture(scope="module")
def test_scores(model):
    _, X_te, _, y_te = load_split()
    return y_te, model.predict_proba(X_te)[:, 1]


def test_model_expects_the_feature_pipeline(model):
    assert list(model.feature_names_in_) == FEATURE_COLUMNS
    assert META["features"] == FEATURE_COLUMNS


def test_sklearn_version_matches_pin():
    pinned = re.search(r"scikit-learn==([\d.]+)", (ROOT / "requirements.txt").read_text()).group(1)
    assert META["sklearn_version"] == pinned == sklearn.__version__


def test_probabilities_are_valid(test_scores):
    _, p = test_scores
    assert ((p >= 0) & (p <= 1)).all()


def test_performance_gate(test_scores):
    y, p = test_scores
    pred = p >= META["threshold"]
    assert roc_auc_score(y, p) >= MIN_AUC
    assert recall_score(y, pred) >= MIN_RECALL
    assert f1_score(y, pred) >= MIN_F1


def test_space_files_stay_small():
    # Plain git pushes to a Space need LFS/Xet above 10 MB; the browser also downloads model.json.
    for f in SPACE.iterdir():
        assert f.stat().st_size < 10 * 1024 * 1024, f.name
