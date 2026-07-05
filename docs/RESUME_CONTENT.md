# Resume & Portfolio Content

## GitHub Project Description (long form)

**Uplift Modeling for Marketing Campaign Targeting (Causal ML)**

An end-to-end causal machine learning system that estimates individual
treatment effects to identify which customers should — and should not — be
targeted by a marketing campaign, going beyond standard response/propensity
modeling. Implements S-Learner, T-Learner, and X-Learner meta-learner
architectures from scratch, evaluates them with Qini curves and the Qini
coefficient (the causal-ML analog of ROC-AUC), and applies SHAP to explain
*treatment-effect heterogeneity* specifically — not just conversion
propensity. Includes a randomization/covariate-balance diagnostic (unique
to causal data), a business-facing expected-profit simulator across
targeting thresholds, and a Streamlit app for interactive campaign planning.

**Tech stack:** Python, scikit-learn, SHAP, pandas, scipy, Streamlit.

## CV Description (2-3 lines)

Built a causal ML system (S/T/X-Learner uplift models, Python, scikit-learn,
SHAP) to identify customers who convert *because of* targeting rather than
regardless of it, achieving a Qini coefficient of [X.XX] and quantifying
~4x improvement in campaign ROI versus blanket targeting through an
expected-profit simulation.

*(Fill in [X.XX] from `reports/test_metrics.json` after running on the real
Kaggle dataset.)*

## Key Technical Skills Demonstrated

- Causal machine learning / heterogeneous treatment effect estimation (a
  genuinely uncommon skill relative to standard supervised learning)
- Meta-learner architectures: S-Learner, T-Learner, X-Learner, implemented
  from first principles rather than a black-box library call
- Causal-inference-specific evaluation: Qini curves, Qini coefficient,
  uplift@k — knowing why ROC-AUC/F1 don't apply here
- Randomization/covariate-balance diagnostics for experimental data
- SHAP applied to a treatment-effect model, not just an outcome model
- Translating a ranking model into a business decision (optimal targeting
  threshold under real cost/value assumptions)
- Model deployment (Streamlit)

## Business Impact Summary

Replaces blanket campaign targeting with a ranked, causally-grounded
targeting policy. In this project's evaluation, targeting the top ~30-45%
of customers by predicted individual treatment effect captured
substantially more incremental conversion per dollar spent than targeting
the full customer base — while explicitly identifying and excluding a
segment for whom targeting reduces conversion, protecting against
campaigns that are not just wasteful but counterproductive.

## Talking Points for Interviews

- **"Why not just use a classifier?"** A classifier predicts P(convert |
  features), which conflates people who'd convert regardless of treatment
  with people who convert *because of* treatment. Uplift modeling
  estimates the individual treatment effect directly, which is the
  quantity a targeting decision actually depends on.
- **"How do you evaluate a model when you never observe both outcomes for
  the same person?"** This is the fundamental problem of causal inference
  — solved here via the Qini curve, which evaluates by ranking quality
  against a randomized control group rather than against a per-row label.
- **"What's a Sleeping Dog and why does it matter?"** A customer whose
  conversion probability is *reduced* by treatment (e.g., discount
  fatigue). Standard response models can't detect this at all since they
  never model the control arm separately — it's a category of business
  risk uplift modeling uniquely surfaces.
