"""
Algorithm / model perturbations (PCS: Lasso, Ridge, alternate losses).

Baseline: balanced L2 logistic (sklearn default penalty).
Perturbations:
  - RidgeClassifier: same L2 spirit, different loss (least-squares on ±1)
  - L1 logistic (Lasso): sparsity-inducing algorithm swap

Yes — RidgeClassifier counts as a model/algorithm perturbation in PCS.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .shared import RANDOM_STATE


def make_logistic(C: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_ridge(alpha: float = 1.0) -> Pipeline:
    """RidgeClassifier — algorithm perturbation of the logistic baseline."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                RidgeClassifier(
                    alpha=alpha,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_lasso_logistic(C: float = 0.1) -> Pipeline:
    """L1-penalized logistic — Lasso-style algorithm perturbation."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=C,
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
