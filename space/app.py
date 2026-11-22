"""Gradio app: credit card default risk (tuned random forest, Module 8)."""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from features import BILL_AMT, PAY_AMT, PAY_STATUS, RAW_COLUMNS, build_features

HERE = Path(__file__).resolve().parent
MODEL = joblib.load(HERE / "model" / "model.joblib")
META = json.loads((HERE / "model" / "metadata.json").read_text())
THRESHOLD = float(META["threshold"])

MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]  # PAY_0 = Sep ... PAY_6 = Apr
PAY_CHOICES = {
    "No consumption (-2)": -2, "Paid in full (-1)": -1, "Revolving credit (0)": 0,
    **{f"{k} month{'s' if k > 1 else ''} late ({k})": k for k in range(1, 9)},
}
SEX = {"Male": 1, "Female": 2}
EDUCATION = {"Graduate school": 1, "University": 2, "High school": 3, "Other": 4}
MARRIAGE = {"Married": 1, "Single": 2, "Other": 3}


def predict_proba(raw: pd.DataFrame) -> pd.Series:
    """Probability of default for each row of raw UCI-format inputs."""
    return pd.Series(MODEL.predict_proba(build_features(raw))[:, 1], index=raw.index)


def score(limit, sex, education, marriage, age, *rest):
    pays, bills, amts = rest[:6], rest[6:12], rest[12:18]
    row = {"LIMIT_BAL": limit, "SEX": SEX[sex], "EDUCATION": EDUCATION[education],
           "MARRIAGE": MARRIAGE[marriage], "AGE": age}
    row.update({c: PAY_CHOICES[v] for c, v in zip(PAY_STATUS, pays)})
    row.update(dict(zip(BILL_AMT, bills)))
    row.update(dict(zip(PAY_AMT, amts)))
    p = float(predict_proba(pd.DataFrame([row], columns=RAW_COLUMNS)).iloc[0])
    flagged = p >= THRESHOLD
    verdict = (f"**{'High risk: flag for review' if flagged else 'Lower risk'}**  \n"
               f"Predicted default probability {p:.1%} (decision threshold {THRESHOLD:.0%}).")
    return {"Default": p, "No default": 1 - p}, verdict


def score_csv(file):
    df = pd.read_csv(file)
    out = df.copy()
    out["default_probability"] = predict_proba(df).round(4)
    out["flagged"] = out["default_probability"] >= THRESHOLD
    path = HERE / "scored.csv"
    out.to_csv(path, index=False)
    return out.head(50), str(path)


m = META["test_metrics"][f"at_{THRESHOLD}"]
ABOUT = (f"Random forest ({META['params']['n_estimators']} trees, min leaf "
         f"{META['params']['min_samples_leaf']}) trained on 24,000 clients of the UCI Default of "
         f"Credit Card Clients data (Yeh, 2016). Held-out test set at the {THRESHOLD} threshold: "
         f"recall {m['recall']:.2f}, precision {m['precision']:.2f}, F1 {m['f1']:.2f}, AUC {m['auc']:.3f}. "
         "Course demo only, not for real lending decisions.")

with gr.Blocks(title="Credit Default Risk") as demo:
    gr.Markdown("# Credit card default risk\n" + ABOUT)
    with gr.Tab("Single client"):
        with gr.Row():
            limit = gr.Number(label="Credit limit (NT$)", value=50000, minimum=0)
            age = gr.Number(label="Age", value=35, minimum=18, maximum=100, precision=0)
        with gr.Row():
            sex = gr.Dropdown(list(SEX), value="Female", label="Sex")
            edu = gr.Dropdown(list(EDUCATION), value="University", label="Education")
            mar = gr.Dropdown(list(MARRIAGE), value="Single", label="Marital status")
        with gr.Accordion("Repayment status by month", open=True):
            with gr.Row():
                pays = [gr.Dropdown(list(PAY_CHOICES), value="Revolving credit (0)", label=mo) for mo in MONTHS]
        with gr.Accordion("Bill amounts (NT$)", open=False):
            with gr.Row():
                bills = [gr.Number(value=20000, label=mo) for mo in MONTHS]
        with gr.Accordion("Payments made (NT$)", open=False):
            with gr.Row():
                amts = [gr.Number(value=2000, label=mo) for mo in MONTHS]
        btn = gr.Button("Score client", variant="primary")
        probs = gr.Label(label="Prediction")
        verdict = gr.Markdown()
        btn.click(score, [limit, sex, edu, mar, age, *pays, *bills, *amts], [probs, verdict], api_name="score")
    with gr.Tab("Batch (CSV)"):
        gr.Markdown("Upload a CSV with the 23 UCI input columns: " + ", ".join(RAW_COLUMNS))
        f = gr.File(file_types=[".csv"], label="Clients CSV")
        table = gr.Dataframe(label="First 50 scored rows")
        dl = gr.File(label="Download scored CSV")
        f.upload(score_csv, f, [table, dl], api_name="score_csv")

if __name__ == "__main__":
    demo.launch()
