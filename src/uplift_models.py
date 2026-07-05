"""
uplift_models.py
-----------------
Implements three standard "meta-learner" approaches to uplift modeling
(estimating individual/conditional treatment effects, not just outcome
propensity). These are written from scratch on top of scikit-learn base
estimators rather than imported from a black-box library, so every design
decision is visible and explainable in an interview.

Meta-learners implemented:
  - S-Learner : one model, treatment as a regular feature.
                Simple, but the treatment signal can get "washed out" by
                stronger features — often underestimates heterogeneity.
  - T-Learner : two independent models, one trained on treated-only data,
                one on control-only data. Uplift = difference in their
                predictions. Simple and interpretable, but each model only
                sees half the data, and errors in each model don't cancel
                out (they can compound in the subtraction).
  - X-Learner : addresses the T-Learner's weakness. Uses the T-Learner
                models to impute individual treatment effects for BOTH
                groups, trains a second-stage model to predict those
                imputed effects directly, then combines the two second-
                stage models using a propensity-weighted average. Generally
                the strongest of the three when treatment/control group
                sizes are imbalanced (as they are here: 70/30).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone


class SLearner:
    """Single model, treatment included as a feature. Uplift = predicted
    outcome with treatment=1 minus predicted outcome with treatment=0."""

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42
        )

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        X_aug = X.copy()
        X_aug["__treatment__"] = treatment
        self.model_ = clone(self.base_estimator)
        self.model_.fit(X_aug, y)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        X1 = X.copy(); X1["__treatment__"] = 1
        X0 = X.copy(); X0["__treatment__"] = 0
        p1 = self.model_.predict_proba(X1)[:, 1]
        p0 = self.model_.predict_proba(X0)[:, 1]
        return p1 - p0


class TLearner:
    """Two independent models: one fit on treated units, one on control
    units. Uplift = difference in their predicted conversion probability."""

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42
        )

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        treated_mask = treatment == 1
        self.model_treated_ = clone(self.base_estimator)
        self.model_control_ = clone(self.base_estimator)
        self.model_treated_.fit(X[treated_mask], y[treated_mask])
        self.model_control_.fit(X[~treated_mask], y[~treated_mask])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.model_treated_.predict_proba(X)[:, 1]
        p0 = self.model_control_.predict_proba(X)[:, 1]
        return p1 - p0


class XLearner:
    """Cross-learner: uses T-Learner base models to impute individual
    treatment effects, fits second-stage regressors on those imputed
    effects, and combines via propensity-weighting. Best default choice
    when treatment/control groups are imbalanced in size."""

    def __init__(self, outcome_estimator=None, effect_estimator=None,
                 propensity_estimator=None):
        self.outcome_estimator = outcome_estimator or GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42
        )
        self.effect_estimator = effect_estimator or GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=42
        )
        self.propensity_estimator = propensity_estimator or LogisticRegression(max_iter=1000)

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        treated_mask = treatment == 1

        # Stage 1: outcome models per arm (same as T-Learner)
        self.model_treated_ = clone(self.outcome_estimator)
        self.model_control_ = clone(self.outcome_estimator)
        self.model_treated_.fit(X[treated_mask], y[treated_mask])
        self.model_control_.fit(X[~treated_mask], y[~treated_mask])

        # Stage 2: imputed individual treatment effects
        # For treated units: observed outcome - predicted control outcome
        # For control units: predicted treated outcome - observed outcome
        d_treated = y[treated_mask] - self.model_control_.predict_proba(X[treated_mask])[:, 1]
        d_control = self.model_treated_.predict_proba(X[~treated_mask])[:, 1] - y[~treated_mask]

        self.effect_model_treated_ = clone(self.effect_estimator)
        self.effect_model_control_ = clone(self.effect_estimator)
        self.effect_model_treated_.fit(X[treated_mask], d_treated)
        self.effect_model_control_.fit(X[~treated_mask], d_control)

        # Stage 3: propensity model, used to weight the two effect estimates
        self.propensity_model_ = clone(self.propensity_estimator)
        self.propensity_model_.fit(X, treatment)

        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        tau_treated = self.effect_model_treated_.predict(X)
        tau_control = self.effect_model_control_.predict(X)
        g = self.propensity_model_.predict_proba(X)[:, 1]  # P(treatment=1 | X)
        # Propensity-weighted combination: weight each estimate by the
        # probability mass of the OPPOSITE arm (standard X-Learner formula)
        return g * tau_control + (1 - g) * tau_treated


MODEL_REGISTRY = {
    "S-Learner": SLearner,
    "T-Learner": TLearner,
    "X-Learner": XLearner,
}
