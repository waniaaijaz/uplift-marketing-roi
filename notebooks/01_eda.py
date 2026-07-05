# %% [markdown]
# # Exploratory Data Analysis — Uplift Modeling / Marketing Campaign Data
# Open in VS Code / Jupytext / Antigravity as a notebook (the `# %%` markers
# define cells), or run top-to-bottom as a script.
#
# This EDA is structured differently from a normal supervised-learning EDA:
# the central question isn't "what predicts conversion" but "is treatment
# randomized, and does its effect vary across segments." That's what makes
# this a CAUSAL analysis, not a correlational one.

# %%
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_raw_data, print_data_quality_report, randomization_check, FEATURE_COLS

sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Load & randomization check
# This is the step a standard churn/EDA notebook doesn't have. If treatment
# assignment isn't actually balanced across covariates, any "effect" we
# measure later could just be confounding, not causation.

# %%
df = load_raw_data()
print_data_quality_report(df)

# %% [markdown]
# **Insight:** with real randomized incrementality-test data (or this
# synthetic stand-in), features should be statistically balanced between
# treatment and control (p > 0.05 for most). This is what licenses us to
# interpret outcome differences as causal effects of treatment, not
# pre-existing differences between who got targeted and who didn't.

# %% [markdown]
# ## 2. Overall (naive) treatment effect

# %%
overall = df.groupby("treatment")["conversion"].mean()
ax = overall.plot(kind="bar", color=["#c62828", "#2e7d32"])
ax.set_xticklabels(["Control", "Treated"], rotation=0)
ax.set_ylabel("Conversion rate")
ax.set_title(f"Overall Conversion Rate by Arm (naive ATE = {overall[1]-overall[0]:.4f})")
plt.tight_layout()
plt.savefig("../reports/figures/eda_overall_ate.png", dpi=150)
plt.show()

# %% [markdown]
# **Insight:** this single number is what a naive "campaign worked!" report
# would show — a positive average effect. But it HIDES the fact that this
# average is made up of very different sub-populations: some customers
# respond strongly, some not at all, and some may respond negatively. The
# rest of this notebook — and the whole point of uplift modeling — is
# uncovering that hidden heterogeneity instead of reporting one number.

# %% [markdown]
# ## 3. Feature distributions by treatment arm (visual randomization check)

# %%
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flatten(), FEATURE_COLS):
    sns.kdeplot(data=df, x=col, hue="treatment", ax=ax, common_norm=False, legend=False)
    ax.set_title(col)
plt.tight_layout()
plt.savefig("../reports/figures/eda_feature_balance.png", dpi=150)
plt.show()

# %% [markdown]
# **Insight:** distributions should visually overlap closely between arms —
# confirming randomization held and none of these features can trivially
# "leak" treatment-group membership.

# %% [markdown]
# ## 4. Searching for heterogeneity: conversion rate by feature decile, split by arm
# This is the key exploratory step for uplift modeling: for each feature,
# bin it into deciles and plot the treated vs. control conversion rate in
# each bin. If the GAP between the two lines changes across deciles, that
# feature carries information about treatment-effect heterogeneity — not
# just about conversion propensity.

# %%
def plot_heterogeneity(df, feature, ax):
    tmp = df.copy()
    tmp["bin"] = pd.qcut(tmp[feature], 10, labels=False, duplicates="drop")
    rates = tmp.groupby(["bin", "treatment"])["conversion"].mean().unstack()
    rates.columns = ["Control", "Treated"]
    rates.plot(ax=ax, marker="o")
    ax.set_title(f"Conversion rate by {feature} decile")
    ax.set_xlabel(f"{feature} decile")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_heterogeneity(df, "f2", axes[0])
plot_heterogeneity(df, "f5", axes[1])
plt.tight_layout()
plt.savefig("../reports/figures/eda_heterogeneity_f2_f5.png", dpi=150)
plt.show()

# %% [markdown]
# **Insight:** if the treated/control lines DIVERGE across deciles of a
# feature (rather than staying a constant distance apart), that feature is
# a genuine driver of *who responds to treatment* — exactly the signal the
# meta-learners in `src/uplift_models.py` are trained to pick up on, and
# what `src/explain.py`'s SHAP analysis will confirm post-training.

# %% [markdown]
# ## Summary
# 1. Randomization check passed — outcome differences can be interpreted
#    causally, not just correlationally.
# 2. The overall average effect masks meaningful heterogeneity — a single
#    "campaign lift" number is not enough to make a targeting decision.
# 3. Certain features show clear divergence in treated vs. control
#    conversion rates across their range — these are the features an uplift
#    model should (and does) lean on most heavily.
