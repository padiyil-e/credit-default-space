import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS, RAW_COLUMNS, build_features, signed_log


def raw_row(**overrides):
    row = {c: 0 for c in RAW_COLUMNS}
    row.update(LIMIT_BAL=50000, SEX=2, EDUCATION=2, MARRIAGE=2, AGE=35)
    row.update(overrides)
    return pd.DataFrame([row])


def test_produces_29_inputs_in_fixed_order():
    X = build_features(raw_row())
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape == (1, 29)


def test_unknown_codes_become_other():
    X = build_features(raw_row(EDUCATION=0, MARRIAGE=0))
    assert X.loc[0, "EDUCATION_other"] == 1
    assert X.loc[0, "MARRIAGE_other"] == 1
    X = build_features(raw_row(EDUCATION=6))
    assert X.loc[0, "EDUCATION_other"] == 1


def test_signed_log_keeps_sign_and_zero():
    assert signed_log(0) == 0
    assert signed_log(-100) == -signed_log(100)
    assert np.isclose(signed_log(np.e - 1), 1.0)


def test_engineered_features():
    X = build_features(raw_row(PAY_0=2, PAY_2=1, PAY_3=-1, BILL_AMT1=60000, LIMIT_BAL=60000))
    assert X.loc[0, "MONTHS_DELINQUENT"] == 2
    assert X.loc[0, "MAX_DELINQUENCY"] == 2
    assert np.isclose(X.loc[0, "AVG_UTILIZATION"], 1 / 6)


def test_missing_column_is_rejected():
    with pytest.raises(ValueError, match="Missing input columns"):
        build_features(raw_row().drop(columns="AGE"))
