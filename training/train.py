"""Train the tuned random forest from Module 8 and export it for the static Space.

Usage:  python training/train.py
Writes:
  model/model.joblib        scikit-learn model (source of truth, used by the tests)
  space/model.json          the same forest as plain arrays, run in the browser by model.js
  space/metadata.json       params, threshold, feature order, test metrics, versions
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from features import FEATURE_COLUMNS, TARGET, build_features  # noqa: E402

DATA = ROOT / "data" / "credit_default.csv"
MODEL_PATH = ROOT / "model" / "model.joblib"
SPACE = ROOT / "space"
SEED = 20
THRESHOLD = 0.29  # chosen in Module 8 from out-of-fold training predictions
PARAMS = dict(n_estimators=200, max_depth=None, min_samples_leaf=50,
              max_features="sqrt", class_weight=None, n_jobs=-1, random_state=SEED)


def load_split():
    df = pd.read_csv(DATA)
    X = build_features(df)
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {"threshold": threshold,
            "accuracy": round(accuracy_score(y, pred), 4),
            "precision": round(precision_score(y, pred), 4),
            "recall": round(recall_score(y, pred), 4),
            "f1": round(f1_score(y, pred), 4),
            "auc": round(roc_auc_score(y, p), 4)}


def export_forest(model) -> dict:
    """Write each tree as parallel arrays: split feature (-1 = leaf), threshold,
    left/right child, and P(default) at leaves. Thresholds keep full float64 precision."""
    trees = []
    for est in model.estimators_:
        t = est.tree_
        leaf = t.children_left == -1
        value = t.value[:, 0, :]
        p1 = value[:, 1] / value.sum(axis=1)
        trees.append({
            "f": np.where(leaf, -1, t.feature).tolist(),
            "t": np.where(leaf, 0.0, t.threshold).tolist(),
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "v": np.where(leaf, p1, 0.0).tolist(),
        })
    return {"features": FEATURE_COLUMNS, "trees": trees}


def main():
    X_tr, X_te, y_tr, y_te = load_split()
    model = RandomForestClassifier(**PARAMS)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    p = model.predict_proba(X_te)[:, 1]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    (SPACE / "model.json").write_text(json.dumps(export_forest(model), separators=(",", ":")))
    meta = {
        "model": "RandomForestClassifier (tuned, Module 8)",
        "params": {k: v for k, v in PARAMS.items() if k != "n_jobs"},
        "threshold": THRESHOLD,
        "features": FEATURE_COLUMNS,
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "test_metrics": {"at_0.50": metrics(y_te, p, 0.5),
                         f"at_{THRESHOLD}": metrics(y_te, p, THRESHOLD)},
        "fit_seconds": round(fit_s, 2),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (SPACE / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["test_metrics"], indent=2))
    print(f"fit {fit_s:.1f}s | joblib {MODEL_PATH.stat().st_size / 1e6:.2f} MB | "
          f"model.json {(SPACE / 'model.json').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
