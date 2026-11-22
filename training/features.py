"""Feature pipeline for the credit default model.

Shared by training (training/train.py) and serving (app.py) so the deployed
model always sees exactly the transformations it was trained on.
Follows the Module 7/8 pipeline: unknown category codes -> "other",
three engineered features, signed log on money columns, one-hot encoding
(first level dropped), giving 29 model inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
MONEY = ["LIMIT_BAL"] + BILL_AMT + PAY_AMT
RAW_COLUMNS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_STATUS + BILL_AMT + PAY_AMT
TARGET = "default payment next month"

# Category levels after recoding; the first level of each is dropped when encoding.
SEX_LEVELS = ["male", "female"]
EDUCATION_LEVELS = ["graduate", "university", "high_school", "other"]
MARRIAGE_LEVELS = ["married", "single", "other"]

FEATURE_COLUMNS = (
    ["LIMIT_BAL", "AGE"] + PAY_STATUS + BILL_AMT + PAY_AMT
    + ["MONTHS_DELINQUENT", "MAX_DELINQUENCY", "AVG_UTILIZATION"]
    + [f"SEX_{v}" for v in SEX_LEVELS[1:]]
    + [f"EDUCATION_{v}" for v in EDUCATION_LEVELS[1:]]
    + [f"MARRIAGE_{v}" for v in MARRIAGE_LEVELS[1:]]
)


def signed_log(x):
    """sign(x) * log(1 + |x|): compresses skewed amounts and keeps negative bills."""
    return np.sign(x) * np.log1p(np.abs(x))


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw UCI columns into the 29 model inputs, in a fixed order."""
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    df = raw[RAW_COLUMNS].astype(float).copy()

    sex = df["SEX"].map({1: "male", 2: "female"}).fillna("female")
    education = df["EDUCATION"].map({1: "graduate", 2: "university", 3: "high_school"}).fillna("other")
    marriage = df["MARRIAGE"].map({1: "married", 2: "single"}).fillna("other")

    pay = df[PAY_STATUS]
    limit = df["LIMIT_BAL"].clip(lower=1)
    out = pd.DataFrame(index=df.index)
    out["LIMIT_BAL"] = df["LIMIT_BAL"]
    out["AGE"] = df["AGE"]
    for c in PAY_STATUS + BILL_AMT + PAY_AMT:
        out[c] = df[c]
    out["MONTHS_DELINQUENT"] = (pay > 0).sum(axis=1).astype(float)
    out["MAX_DELINQUENCY"] = pay.clip(lower=0).max(axis=1)
    out["AVG_UTILIZATION"] = df[BILL_AMT].div(limit, axis=0).mean(axis=1)
    for c in MONEY:
        out[c] = signed_log(out[c])
    for v in SEX_LEVELS[1:]:
        out[f"SEX_{v}"] = (sex == v).astype(float)
    for v in EDUCATION_LEVELS[1:]:
        out[f"EDUCATION_{v}"] = (education == v).astype(float)
    for v in MARRIAGE_LEVELS[1:]:
        out[f"MARRIAGE_{v}"] = (marriage == v).astype(float)
    return out[FEATURE_COLUMNS]
