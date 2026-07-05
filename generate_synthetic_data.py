"""
generate_synthetic_data.py
---------------------------------
Generates a synthetic dataset with the SAME schema as the Kaggle
"Uplift Modeling, Marketing Campaign Data" dataset
(https://www.kaggle.com/datasets/arashnic/uplift-modeling), derived from the
Criteo AI Lab uplift benchmark.

Critically, this generator DESIGNS IN heterogeneous treatment effects so the
four uplift quadrants actually exist in the data and can be recovered by the
models in src/uplift_models.py:

  - Persuadables : convert ONLY if treated            -> target these
  - Sure Things  : convert regardless of treatment     -> don't waste spend
  - Lost Causes  : never convert either way            -> don't waste spend
  - Sleeping Dogs: treatment REDUCES their conversion  -> never target

This is what makes the dataset useful for demonstrating uplift modeling
rather than plain response modeling: features f2 and f5 are built to be the
true drivers of *treatment-effect heterogeneity*, not just of conversion
propensity, mirroring how real campaign data behaves.

Run:
    python generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(7)
N = 50_000

OUT_PATH = Path("data/raw/criteo_uplift.csv")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n: int = N) -> pd.DataFrame:
    # 12 anonymized continuous features, matching the real dataset's shape
    features = RNG.normal(0, 1, size=(n, 12))
    cols = {f"f{i}": features[:, i] for i in range(12)}
    df = pd.DataFrame(cols)

    # Treatment assignment: 70% treated / 30% held-out control (typical of a
    # real incrementality test, which deliberately withholds a control group)
    treatment = RNG.choice([1, 0], size=n, p=[0.70, 0.30])
    df["treatment"] = treatment

    f0, f1, f2, f3, f4, f5 = (df[f"f{i}"].values for i in range(6))

    # Baseline conversion propensity (independent of treatment) — drives
    # "Sure Things" (high baseline) vs "Lost Causes" (low baseline)
    baseline_logit = 0.6 * f0 + 0.4 * f1 - 0.5 - 0.3 * np.abs(f3)

    # TRUE individual treatment effect (this is the ground truth the uplift
    # models are trying to recover). f2 and f5 are the designed heterogeneity
    # drivers:
    #   - high f2 -> strong POSITIVE responder (Persuadable)
    #   - low  f2, high f5 -> NEGATIVE responder (Sleeping Dog)
    #   - everyone else -> near-zero incremental effect
    true_uplift_logit = 1.4 * np.clip(f2, 0, None) - 0.9 * np.clip(f5, 0, None) * (f2 < 0)

    control_logit = baseline_logit
    treated_logit = baseline_logit + true_uplift_logit

    conversion_prob_control = sigmoid(control_logit)
    conversion_prob_treated = sigmoid(treated_logit)

    realized_prob = np.where(treatment == 1, conversion_prob_treated, conversion_prob_control)
    conversion = (RNG.uniform(0, 1, n) < realized_prob).astype(int)

    # Visit is a noisier, weaker upstream signal (more people visit than convert)
    visit_prob = sigmoid(0.5 * baseline_logit + 0.3 * true_uplift_logit * treatment + 1.0)
    visit = (RNG.uniform(0, 1, n) < visit_prob).astype(int)
    visit = np.maximum(visit, conversion)  # conversion implies a visit happened

    df["visit"] = visit
    df["conversion"] = conversion

    # Store ground-truth potential outcomes ONLY for internal validation use
    # (a real dataset would never have this — it's the fundamental problem of
    # causal inference). Kept in a separate file so models never see it.
    ground_truth = pd.DataFrame({
        "true_ite": conversion_prob_treated - conversion_prob_control,
    })

    return df, ground_truth


if __name__ == "__main__":
    df, ground_truth = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    ground_truth.to_csv(OUT_PATH.parent / "ground_truth_ite.csv", index=False)

    print(f"Synthetic dataset written to {OUT_PATH} ({len(df)} rows)")
    print(f"Treatment rate: {df['treatment'].mean():.2%}")
    print(f"Conversion rate (treated):  {df.loc[df.treatment==1,'conversion'].mean():.4f}")
    print(f"Conversion rate (control):  {df.loc[df.treatment==0,'conversion'].mean():.4f}")
    print(f"Naive overall uplift (treated - control conversion rate): "
          f"{df.loc[df.treatment==1,'conversion'].mean() - df.loc[df.treatment==0,'conversion'].mean():.4f}")
