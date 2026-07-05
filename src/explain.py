"""
explain.py
----------
Explainable AI for the UPLIFT model — this explains what drives
heterogeneity in TREATMENT EFFECT, which is a fundamentally different
question from "what drives conversion." A feature can be a strong driver of
conversion in general while being completely irrelevant to whether
targeting that person works (and vice versa) — that distinction is the
entire point of this project, and SHAP applied here proves the model
learned it.

Each meta-learner architecture needs a different SHAP strategy:
  - S-Learner: single classifier, treatment is a feature. SHAP on the
    treatment feature itself IS the model's built-in signal for uplift.
  - T-Learner: two independent classifiers. Approximate "uplift SHAP" per
    feature as SHAP(treated-model) - SHAP(control-model) on the same input.
  - X-Learner: two regressors predict treatment effect DIRECTLY, so SHAP on
    them is already an explanation of uplift, not conversion.
"""

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.uplift_models import SLearner, TLearner, XLearner

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def compute_uplift_shap(model, X: pd.DataFrame):
    """Returns (shap_values, feature_names) representing each feature's
    contribution to the PREDICTED UPLIFT, regardless of meta-learner type."""
    feature_names = list(X.columns)

    if isinstance(model, SLearner):
        X1 = X.copy(); X1["__treatment__"] = 1
        explainer = shap.TreeExplainer(model.model_)
        shap_values = explainer.shap_values(X1)
        if isinstance(shap_values, list):  # binary classifier returns [class0, class1]
            shap_values = shap_values[1]
        # Drop the synthetic treatment column's own contribution — we want
        # feature-level drivers of WHO responds, not the treatment flag itself
        shap_values = shap_values[:, :-1]
        return shap_values, feature_names

    elif isinstance(model, TLearner):
        exp_t = shap.TreeExplainer(model.model_treated_)
        exp_c = shap.TreeExplainer(model.model_control_)
        sv_t = exp_t.shap_values(X)
        sv_c = exp_c.shap_values(X)
        sv_t = sv_t[1] if isinstance(sv_t, list) else sv_t
        sv_c = sv_c[1] if isinstance(sv_c, list) else sv_c
        return sv_t - sv_c, feature_names

    elif isinstance(model, XLearner):
        exp_t = shap.TreeExplainer(model.effect_model_treated_)
        exp_c = shap.TreeExplainer(model.effect_model_control_)
        sv_t = exp_t.shap_values(X)
        sv_c = exp_c.shap_values(X)
        g = model.propensity_model_.predict_proba(X)[:, 1]
        # Same propensity-weighted combination used in predict_uplift()
        combined = (g[:, None] * sv_c) + ((1 - g[:, None]) * sv_t)
        return combined, feature_names

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def run_shap_analysis(sample_size: int = 1000):
    model = joblib.load(MODELS_DIR / "uplift_model.joblib")
    X_test = pd.read_csv(REPORTS_DIR / "X_test.csv")
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42).reset_index(drop=True)

    shap_values, feature_names = compute_uplift_shap(model, X_sample)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("What Drives Treatment-Effect Heterogeneity (Uplift), Not Just Conversion")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "uplift_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "uplift_shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    mean_abs_shap.to_csv(REPORTS_DIR / "uplift_feature_importance.csv", index=False)

    print("Top drivers of treatment-effect heterogeneity:")
    print(mean_abs_shap.head(6).to_string(index=False))
    print(f"\nSHAP plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    run_shap_analysis()
