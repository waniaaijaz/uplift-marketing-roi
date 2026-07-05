"""
run_pipeline.py
----------------
One-command orchestrator: data quality/randomization check -> train all
meta-learners -> evaluate the winner (Qini, deciles, profit) -> SHAP.

Usage:
    python generate_synthetic_data.py   # only if you don't have the real Kaggle CSV yet
    python run_pipeline.py
"""

from src.data_loader import load_raw_data, print_data_quality_report
from src.train import run_training_pipeline
from src.evaluate import run_evaluation
from src.explain import run_shap_analysis


def main():
    print("\n### STEP 1/4 — Data quality & randomization check ###")
    df = load_raw_data()
    print_data_quality_report(df)

    print("\n### STEP 2/4 — Training & comparing meta-learners ###")
    run_training_pipeline()

    print("\n### STEP 3/4 — Evaluation (Qini, deciles, profit simulation) ###")
    run_evaluation()

    print("\n### STEP 4/4 — SHAP explainability on treatment-effect model ###")
    run_shap_analysis()

    print("\nDone. Run `streamlit run app/streamlit_app.py` to launch the campaign simulator.")


if __name__ == "__main__":
    main()
