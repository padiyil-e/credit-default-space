import pandas as pd

import app

CLEAN = ["Paid in full (-1)"] * 6
LATE = ["3 months late (3)", "2 months late (2)", "2 months late (2)",
        "2 months late (2)", "2 months late (2)", "2 months late (2)"]


def run(pays):
    return app.score(100000, "Female", "University", "Single", 35,
                     *pays, *[20000] * 6, *[2000] * 6)


def test_score_returns_label_and_verdict():
    probs, verdict = run(CLEAN)
    assert abs(probs["Default"] + probs["No default"] - 1) < 1e-9
    assert "threshold" in verdict


def test_late_payer_scores_riskier_than_clean_payer():
    p_clean = run(CLEAN)[0]["Default"]
    p_late = run(LATE)[0]["Default"]
    assert p_late > p_clean
    assert p_late >= app.THRESHOLD
    assert "High risk" in run(LATE)[1]


def test_batch_scoring(tmp_path):
    df = pd.read_csv("data/credit_default.csv", nrows=20).drop(columns="default payment next month")
    path = tmp_path / "clients.csv"
    df.to_csv(path, index=False)
    table, out = app.score_csv(str(path))
    assert len(table) == 20
    assert table["default_probability"].between(0, 1).all()
    assert pd.read_csv(out).shape[0] == 20
