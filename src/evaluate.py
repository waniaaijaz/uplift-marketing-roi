"""
evaluate.py
-----------
Loads the trained model + test set, plots the Qini curve and uplift-by-
decile chart, and runs an expected-profit simulation across targeting
thresholds — the business-facing output of this whole project.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.uplift_metrics import (
    qini_curve, qini_coefficient, uplift_by_decile, expected_incremental_profit,
)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

# Business assumptions — tune these to your actual campaign economics
COST_PER_TREATMENT = 2.50      # e.g. cost of a promo email/discount voucher
VALUE_PER_CONVERSION = 45.0    # e.g. average order value / margin


def load_artifacts():
    model = joblib.load(MODELS_DIR / "uplift_model.joblib")
    X_test = pd.read_csv(REPORTS_DIR / "X_test.csv")
    outcomes = pd.read_csv(REPORTS_DIR / "test_outcomes.csv")
    return model, X_test, outcomes["treatment"].values, outcomes["conversion"].values


def plot_qini(y_test, t_test, uplift_scores):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = qini_curve(y_test, t_test, uplift_scores)
    qc = qini_coefficient(y_test, t_test, uplift_scores)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["fraction"], df["qini"], label="Model", color="#1565c0", linewidth=2)
    ax.plot([0, 1], [0, df["qini"].iloc[-1]], "--", color="gray", label="Random targeting")
    ax.set_xlabel("Fraction of population targeted (sorted by predicted uplift)")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.set_title(f"Qini Curve (Qini coefficient = {qc:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "qini_curve.png", dpi=150)
    plt.close(fig)


def plot_decile_chart(y_test, t_test, uplift_scores):
    decile_df = uplift_by_decile(y_test, t_test, uplift_scores)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(decile_df["decile"], decile_df["observed_uplift"], color="#2e7d32")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Decile (1 = highest predicted uplift)")
    ax.set_ylabel("Observed incremental conversion rate")
    ax.set_title("Uplift by Decile — should decrease left to right if model is well-calibrated")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "uplift_by_decile.png", dpi=150)
    plt.close(fig)
    decile_df.to_csv(REPORTS_DIR / "uplift_by_decile.csv", index=False)
    return decile_df


def plot_profit_curve(uplift_scores, n_customers):
    fractions = np.linspace(0.05, 1.0, 20)
    profits = [
        expected_incremental_profit(
            uplift_scores, n_customers, COST_PER_TREATMENT, VALUE_PER_CONVERSION, f
        )["expected_net_profit"]
        for f in fractions
    ]
    best_idx = int(np.argmax(profits))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fractions * 100, profits, color="#c62828", linewidth=2)
    ax.axvline(fractions[best_idx] * 100, linestyle="--", color="gray",
               label=f"Optimal: target top {fractions[best_idx]*100:.0f}%")
    ax.set_xlabel("% of customers targeted (sorted by predicted uplift)")
    ax.set_ylabel("Expected net profit ($)")
    ax.set_title("Expected Campaign Profit vs. Targeting Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profit_curve.png", dpi=150)
    plt.close(fig)

    return fractions[best_idx], profits[best_idx]


def run_evaluation():
    model, X_test, t_test, y_test = load_artifacts()
    uplift_scores = model.predict_uplift(X_test)

    qc = qini_coefficient(y_test, t_test, uplift_scores)
    print(f"Qini coefficient (test set): {qc:.4f}")

    plot_qini(y_test, t_test, uplift_scores)
    decile_df = plot_decile_chart(y_test, t_test, uplift_scores)
    print("\nUplift by decile:")
    print(decile_df.to_string(index=False))

    best_frac, best_profit = plot_profit_curve(uplift_scores, len(X_test))
    print(f"\nOptimal targeting fraction: top {best_frac*100:.0f}% of customers")
    print(f"Expected net profit at optimum: ${best_profit:,.2f} "
          f"(on a test set of {len(X_test)} customers — scale linearly for your full base)")

    metrics = {
        "qini_coefficient": qc,
        "optimal_targeting_fraction": float(best_frac),
        "expected_net_profit_at_optimum": float(best_profit),
        "cost_per_treatment_assumed": COST_PER_TREATMENT,
        "value_per_conversion_assumed": VALUE_PER_CONVERSION,
    }
    with open(REPORTS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved plots to {FIGURES_DIR}/")
    return metrics


if __name__ == "__main__":
    run_evaluation()
