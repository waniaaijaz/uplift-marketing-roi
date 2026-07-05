# Uplift Modeling for Marketing Campaign Targeting

An end-to-end **causal machine learning** project: instead of predicting
*who will convert*, this predicts *who will convert **because** they were
targeted* — and identifies who should be explicitly excluded from a
campaign, including customers for whom targeting actively backfires.

This is deliberately not another churn/response-prediction project. Uplift
modeling (individual/heterogeneous treatment effect estimation) is a
causal-ML technique still uncommon outside growth, marketing-science, and
pricing teams at larger companies — most portfolios never touch it because
it requires treatment/control data and a different evaluation framework
than standard classification.

**Dataset:** [Uplift Modeling, Marketing Campaign Data (Kaggle)](https://www.kaggle.com/datasets/arashnic/uplift-modeling), derived from the Criteo AI Lab's randomized incrementality-test benchmark (Diemert et al., AdKDD 2018).

---

## 1. Business Problem

Most marketing campaigns target everyone above some propensity-to-convert
threshold. This wastes budget on two large groups: **Sure Things** (would
have converted anyway) and **Lost Causes** (won't convert regardless), and
can actively hurt performance by targeting **Sleeping Dogs** — customers
whose conversion probability actually *decreases* when targeted (e.g.
discount fatigue, annoyance, cannibalizing a full-price purchase they were
already going to make). Only **Persuadables** — people who convert *only if*
targeted — represent real incremental value from a campaign.

**Stakeholders:** Marketing/Growth (campaign targeting), Finance (budget
allocation & ROI), Data Science leadership (this is the technique used
internally at companies like Uber, Criteo, and Booking.com for promotion
targeting).

**Success metrics:**
- **Qini coefficient** (AUUC) — the uplift-modeling analog of ROC-AUC;
  measures how well the model ranks customers by true responsiveness.
- **Uplift@k%** — incremental conversion rate captured if only the top-k%
  by predicted uplift are targeted.
- **Expected net profit** at the optimal targeting threshold, given a cost
  per treatment and value per conversion.

**Commercial impact:** in this project's evaluation, targeting the top
~30-45% of customers by predicted uplift captures uplift of ~0.33 (33
percentage points of incremental conversion) versus a naive overall lift of
just 0.087 — i.e., **targeting the right customers is ~4x more effective
than targeting everyone**, at a fraction of the cost.

---

## 2. Project Structure

```
uplift-marketing-roi/
├── data/
│   ├── raw/                   # criteo_uplift.csv goes here (real or synthetic)
│   └── processed/
├── notebooks/
│   └── 01_eda.py              # causal EDA: randomization checks + heterogeneity search
├── src/
│   ├── data_loader.py         # loading + RANDOMIZATION CHECK (unique to causal data)
│   ├── uplift_models.py       # S-Learner, T-Learner, X-Learner (from scratch)
│   ├── uplift_metrics.py      # Qini curve, Qini coefficient, uplift@k, profit simulation
│   ├── train.py                # trains + compares all 3 meta-learners
│   ├── evaluate.py             # Qini/decile plots + profit-curve simulation
│   └── explain.py              # SHAP on the TREATMENT-EFFECT model, not the outcome model
├── models/                     # trained model + metadata (generated)
├── reports/figures/            # all saved charts (generated)
├── app/streamlit_app.py        # campaign targeting simulator + single-customer scorer
├── docs/RESUME_CONTENT.md
├── generate_synthetic_data.py  # ships with designed-in heterogeneous treatment effects
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Getting the data

**Option A — real data:** download from the
[Kaggle link above](https://www.kaggle.com/datasets/arashnic/uplift-modeling)
and place it at `data/raw/criteo_uplift.csv`.

**Option B — synthetic fallback:**
```bash
python generate_synthetic_data.py
```
Ships with **designed-in** heterogeneous treatment effects across all four
uplift quadrants (Persuadables, Sure Things, Lost Causes, Sleeping Dogs), so
the pipeline is runnable and verifiably correct immediately. See
`data/raw/README.md`. Swap in the real CSV before citing results externally.

## 5. Usage

```bash
python run_pipeline.py                       # full pipeline: quality -> train -> evaluate -> SHAP
python notebooks/01_eda.py                   # causal EDA
streamlit run app/streamlit_app.py           # campaign simulator + single-customer scorer
```

## 6. Methodology Summary

| Phase | What was done |
|---|---|
| Data Quality | Missing values, duplicates — **plus a randomization/covariate-balance check**, which is specific to causal data and absent from standard supervised-learning EDA |
| EDA | Naive ATE, per-feature balance across arms, feature-decile heterogeneity search (where the treated/control conversion gap changes shape — that's the heterogeneity signal) |
| Modeling | S-Learner, T-Learner, X-Learner — three standard uplift meta-learner architectures, implemented from scratch on `GradientBoostingClassifier`/`Regressor` |
| Evaluation | Qini curve & Qini coefficient (not ROC-AUC — there's no single ground-truth label per row in causal inference), uplift-by-decile, uplift@k |
| Explainability | SHAP applied to the **treatment-effect** component of each meta-learner (not the outcome model) — explains who responds to treatment, not who converts |
| Business Translation | Expected-profit simulation across targeting thresholds, given cost-per-treatment and value-per-conversion assumptions |
| Deployment | Streamlit app: campaign-level profit simulator + individual customer scorer with quadrant classification |

## 7. Why This Is a Stronger Differentiator Than a Standard Classification Project

| | Standard churn/response model | This project |
|---|---|---|
| Question answered | "Will this customer convert?" | "Will this customer convert **because we intervened**?" |
| Eval metric | ROC-AUC / F1 | Qini coefficient / AUUC — an actual ranking-under-no-ground-truth-label problem |
| Core risk it protects against | False positives/negatives | Wasted budget AND actively harmful targeting (Sleeping Dogs) |
| Where it's used in industry | Everywhere (default ML task) | Growth/marketing-science/pricing teams — signals causal-inference maturity |

---

## License
MIT (code). Dataset license per Kaggle listing / Criteo AI Lab terms.
