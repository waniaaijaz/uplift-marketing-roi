"""
train.py
--------
Trains S-Learner, T-Learner, and X-Learner, compares them by Qini
coefficient on a held-out test set, and persists the winning model.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data_loader import load_raw_data, FEATURE_COLS, TREATMENT_COL, OUTCOME_COL
from src.uplift_models import MODEL_REGISTRY
from src.uplift_metrics import qini_coefficient, uplift_at_k

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def run_training_pipeline(test_size: float = 0.3, random_state: int = 42):
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_raw_data()
    X = df[FEATURE_COLS]
    treatment = df[TREATMENT_COL].values
    y = df[OUTCOME_COL].values

    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, y, test_size=test_size, stratify=treatment, random_state=random_state
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train treatment rate: {t_train.mean():.2%}, Test: {t_test.mean():.2%}")

    results = []
    fitted_models = {}

    for name, ModelClass in MODEL_REGISTRY.items():
        print(f"\nTraining {name}...")
        model = ModelClass()
        model.fit(X_train.reset_index(drop=True), t_train, y_train)
        uplift_scores = model.predict_uplift(X_test.reset_index(drop=True))

        qc = qini_coefficient(y_test, t_test, uplift_scores)
        u30 = uplift_at_k(y_test, t_test, uplift_scores, k=0.3)

        results.append({"model": name, "qini_coefficient": qc, "uplift_at_30pct": u30})
        fitted_models[name] = model
        print(f"  Qini coefficient: {qc:.4f} | Uplift@30%: {u30:.4f}")

    results_df = pd.DataFrame(results).sort_values("qini_coefficient", ascending=False).reset_index(drop=True)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\n--- Model comparison (ranked by Qini coefficient) ---")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_model, MODELS_DIR / "uplift_model.joblib")
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "qini_coefficient": float(results_df.iloc[0]["qini_coefficient"]),
            "uplift_at_30pct": float(results_df.iloc[0]["uplift_at_30pct"]),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }, f, indent=2)

    # Persist test split + all models' predictions for evaluation/explainability
    X_test.reset_index(drop=True).to_csv(REPORTS_DIR / "X_test.csv", index=False)
    pd.DataFrame({"treatment": t_test, "conversion": y_test}).to_csv(
        REPORTS_DIR / "test_outcomes.csv", index=False
    )
    joblib.dump(fitted_models, MODELS_DIR / "all_models.joblib")

    print(f"\nBest model saved to {MODELS_DIR / 'uplift_model.joblib'}")
    return best_model, results_df


if __name__ == "__main__":
    run_training_pipeline()
