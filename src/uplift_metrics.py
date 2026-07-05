"""
uplift_metrics.py
------------------
Evaluation metrics specific to uplift modeling. These replace ROC-AUC/F1 —
standard classification metrics don't apply here because there's no single
"correct label" per row to score against (that's the fundamental problem of
causal inference: we never observe both potential outcomes for the same
person). Instead we evaluate by RANKING: if we sort customers by predicted
uplift and target the top-k%, do we actually capture more incremental
conversions than targeting randomly?

Implements:
  - qini_curve       : cumulative incremental conversions vs. random targeting
  - qini_coefficient : area between the Qini curve and the random-targeting
                        line (the uplift-modeling equivalent of ROC-AUC)
  - uplift_at_k       : incremental conversion rate if only the top-k% by
                        predicted uplift are targeted
  - expected_incremental_profit : translates uplift into $ given a cost per
                        treatment and a value per conversion
"""

import numpy as np
import pandas as pd


def qini_curve(y_true: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray) -> pd.DataFrame:
    """Builds the Qini curve: at each fraction of the population (sorted by
    predicted uplift, descending), computes the cumulative incremental
    conversions relative to what a random targeting strategy would achieve.
    """
    order = np.argsort(-uplift_score)
    y, t = y_true[order], treatment[order]
    n = len(y)

    n_treated_cum = np.cumsum(t)
    n_control_cum = np.cumsum(1 - t)
    y_treated_cum = np.cumsum(y * t)
    y_control_cum = np.cumsum(y * (1 - t))

    # Qini value at each point: incremental conversions captured so far,
    # rescaling the control group's conversions up to the treated group's
    # size so the two are directly comparable.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_control_cum > 0, n_treated_cum / n_control_cum, 0)
        qini = y_treated_cum - y_control_cum * ratio

    fraction = np.arange(1, n + 1) / n
    return pd.DataFrame({"fraction": fraction, "qini": qini})


def qini_coefficient(y_true: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray) -> float:
    """Area between the model's Qini curve and the diagonal "random
    targeting" line, normalized by population size. Higher = better ranking
    of who to target. This is the primary model-comparison metric for
    uplift modeling, analogous to ROC-AUC for classification."""
    df = qini_curve(y_true, treatment, uplift_score)
    n = len(df)
    random_line = df["qini"].iloc[-1] * df["fraction"]
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(df["qini"] - random_line, df["fraction"]))


def uplift_at_k(y_true: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray, k: float = 0.3) -> float:
    """Incremental conversion rate (treated - control) within the top-k
    fraction of the population ranked by predicted uplift. Answers the
    practical question: 'if we can only afford to target 30% of our list,
    how much extra conversion do we get from targeting the RIGHT 30%?'"""
    order = np.argsort(-uplift_score)
    n_top = int(len(y_true) * k)
    idx = order[:n_top]
    y_top, t_top = y_true[idx], treatment[idx]

    treated_rate = y_top[t_top == 1].mean() if (t_top == 1).sum() > 0 else 0
    control_rate = y_top[t_top == 0].mean() if (t_top == 0).sum() > 0 else 0
    return float(treated_rate - control_rate)


def uplift_by_decile(y_true: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray) -> pd.DataFrame:
    """Splits the population into deciles by predicted uplift and reports
    the actual observed incremental conversion rate in each — the standard
    diagnostic chart for uplift models (should be monotonically decreasing
    from decile 1 to decile 10 if the model is well-calibrated)."""
    df = pd.DataFrame({"y": y_true, "t": treatment, "score": uplift_score})
    df["decile"] = pd.qcut(df["score"].rank(method="first", ascending=False), 10, labels=range(1, 11))

    rows = []
    for d in range(1, 11):
        sub = df[df["decile"] == d]
        treated_rate = sub.loc[sub.t == 1, "y"].mean() if (sub.t == 1).sum() > 0 else np.nan
        control_rate = sub.loc[sub.t == 0, "y"].mean() if (sub.t == 0).sum() > 0 else np.nan
        rows.append({
            "decile": d, "n": len(sub),
            "conversion_rate_treated": treated_rate,
            "conversion_rate_control": control_rate,
            "observed_uplift": treated_rate - control_rate,
            "mean_predicted_uplift": sub["score"].mean(),
        })
    return pd.DataFrame(rows)


def expected_incremental_profit(uplift_score: np.ndarray, n_customers: int,
                                  cost_per_treatment: float, value_per_conversion: float,
                                  target_fraction: float) -> dict:
    """Translates a targeting policy into a dollar estimate: if we treat the
    top `target_fraction` of customers by predicted uplift, what's the
    expected incremental profit (incremental revenue minus campaign cost)?"""
    order = np.argsort(-uplift_score)
    n_target = int(n_customers * target_fraction)
    targeted_idx = order[:n_target]

    mean_uplift_targeted = uplift_score[targeted_idx].mean() if n_target > 0 else 0
    incremental_conversions = mean_uplift_targeted * n_target
    incremental_revenue = incremental_conversions * value_per_conversion
    campaign_cost = n_target * cost_per_treatment
    net_profit = incremental_revenue - campaign_cost

    return {
        "n_targeted": n_target,
        "expected_incremental_conversions": incremental_conversions,
        "expected_incremental_revenue": incremental_revenue,
        "campaign_cost": campaign_cost,
        "expected_net_profit": net_profit,
        "roi": net_profit / campaign_cost if campaign_cost > 0 else float("nan"),
    }
