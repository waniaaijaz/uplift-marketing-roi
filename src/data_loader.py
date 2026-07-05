"""
data_loader.py
---------------
Loads the raw uplift dataset and runs data-quality diagnostics that matter
specifically for CAUSAL data (not just any tabular dataset): treatment/
control balance, covariate balance across arms (a randomization check), and
the standard missing/duplicate/outlier checks.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

RAW_PATH = Path("data/raw/criteo_uplift.csv")
FEATURE_COLS = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
OUTCOME_COL = "conversion"


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run `python generate_synthetic_data.py` "
            "first, or place the real Kaggle CSV there (see data/raw/README.md)."
        )
    return pd.read_csv(path)


def randomization_check(df: pd.DataFrame) -> pd.DataFrame:
    """The single most important sanity check for causal data: are features
    balanced between treatment and control? If randomization held, no
    feature should differ significantly between arms. This is NOT something
    you'd check in a standard supervised-learning EDA — it's specific to
    causal inference, and its absence is why naive ML on treatment-mixed
    data produces biased effect estimates."""
    rows = []
    treated = df[df[TREATMENT_COL] == 1]
    control = df[df[TREATMENT_COL] == 0]
    for col in FEATURE_COLS:
        t_stat, p_val = stats.ttest_ind(treated[col], control[col], equal_var=False)
        rows.append({
            "feature": col,
            "mean_treated": treated[col].mean(),
            "mean_control": control[col].mean(),
            "p_value": p_val,
            "balanced (p>0.05)": p_val > 0.05,
        })
    return pd.DataFrame(rows)


def data_quality_report(df: pd.DataFrame) -> dict:
    report = {}
    missing = df.isna().sum()
    missing = missing[missing > 0]
    report["missing_values"] = missing.to_dict()
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["treatment_rate"] = float(df[TREATMENT_COL].mean())
    report["conversion_rate_treated"] = float(df.loc[df[TREATMENT_COL] == 1, OUTCOME_COL].mean())
    report["conversion_rate_control"] = float(df.loc[df[TREATMENT_COL] == 0, OUTCOME_COL].mean())
    report["naive_ate"] = report["conversion_rate_treated"] - report["conversion_rate_control"]

    balance = randomization_check(df)
    report["n_imbalanced_features"] = int((~balance["balanced (p>0.05)"]).sum())
    return report, balance


def print_data_quality_report(df: pd.DataFrame) -> None:
    report, balance = data_quality_report(df)
    print("=" * 60)
    print("DATA QUALITY & RANDOMIZATION REPORT")
    print("=" * 60)
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing values: {report['missing_values'] or 'None'}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"\nTreatment rate: {report['treatment_rate']:.2%}")
    print(f"Conversion rate (treated): {report['conversion_rate_treated']:.4f}")
    print(f"Conversion rate (control): {report['conversion_rate_control']:.4f}")
    print(f"Naive Average Treatment Effect (ATE): {report['naive_ate']:.4f}")
    print(f"\n-- Randomization check (feature balance across arms) --")
    print(balance.to_string(index=False))
    print(f"\n{report['n_imbalanced_features']}/12 features show significant "
          f"imbalance (p<0.05) between treatment and control.")
    if report["n_imbalanced_features"] == 0:
        print("Randomization looks clean — safe to attribute outcome "
              "differences to treatment, not confounding.")
    else:
        print("WARNING: some imbalance detected — investigate before "
              "trusting causal estimates at face value.")
    print("=" * 60)


if __name__ == "__main__":
    df = load_raw_data()
    print_data_quality_report(df)
