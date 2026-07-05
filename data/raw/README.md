# Data Folder

**Dataset:** Uplift Modeling — Marketing Campaign Data
**Kaggle:** https://www.kaggle.com/datasets/arashnic/uplift-modeling
**Origin:** Derived from the Criteo AI Lab Uplift Prediction Dataset
(Diemert, Betlei, Renaudin, Amini — "A Large Scale Benchmark for Uplift
Modeling," AdKDD 2018), built from real randomized incrementality tests —
a randomized-trial procedure where a random slice of users is *withheld*
from ad targeting specifically so the causal effect of targeting can be
measured (not just correlated).

Columns: 12 anonymized continuous features (`f0`-`f11`), `treatment`
(1 = user was targeted, 0 = held out as control), `visit` and `conversion`
(binary outcomes).

## To use the REAL dataset
1. Download the CSV from the Kaggle link above.
2. Place it in this folder as `data/raw/criteo_uplift.csv`.
3. Everything downstream reads from here automatically.

## Why synthetic data ships with this repo
This environment has no internet access to hit Kaggle's API, so
`generate_synthetic_data.py` (project root) builds a schema-identical
dataset with realistic, *designed-in* heterogeneous treatment effects —
including all four uplift quadrants (Persuadables, Sure Things, Lost
Causes, Sleeping Dogs) — so the full causal-ML pipeline is runnable and
demonstrably correct out of the box. Swap in the real CSV before treating
any Qini/AUUC numbers as final results for a CV or interview.
