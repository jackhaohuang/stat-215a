"""
Data / resampling perturbations (PCS predictability under finite-sample noise).

Stratified k-fold on the labelled train frame. Each fold is a different
train/holdout split of the *same* population — not an algorithm change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .shared import RANDOM_STATE, fit_predict, fit_predict_by_preverbal, metrics


def stratified_kfold_metrics(
    make_clf,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Return one row of metrics per fold (holdout = that fold).

    Uses StratifiedKFold so each fold keeps ~the same ciTBI rate.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    rows = []
    y_np = y.to_numpy()
    for fold, (tr, te) in enumerate(skf.split(X, y_np), start=1):
        pred, score = fit_predict(
            make_clf(),
            X.iloc[tr],
            y.iloc[tr],
            X.iloc[te],
        )
        m = metrics(y.iloc[te], pred, score)
        rows.append(
            {
                "fold": fold,
                "n_hold": int(len(te)),
                "recall": m["recall"],
                "precision": m["precision"],
                "fnr": m["fnr"],
                "fpr": m["fpr"],
                "pr_auc": m.get("pr_auc", np.nan),
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
            }
        )
    return pd.DataFrame(rows)


def stratified_kfold_metrics_by_preverbal(
    make_clf,
    labelled: pd.DataFrame,
    feat_cols: list[str],
    y: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Stratified k-fold where each fold fits separate preverbal/verbal models.

    Fold splits are stratified on the outcome; within each fold the learner
    is the same preverbal/verbal cohort fit used elsewhere.
    """
    if "preverbal" not in labelled.columns:
        raise KeyError("stratified_kfold_metrics_by_preverbal needs 'preverbal'")

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    rows = []
    y_np = y.to_numpy()
    index = labelled.index.to_numpy()
    for fold, (tr, te) in enumerate(skf.split(labelled, y_np), start=1):
        idx_fit = pd.Index(index[tr])
        idx_hold = pd.Index(index[te])
        pred, score, y_hold = fit_predict_by_preverbal(
            make_clf, labelled, feat_cols, idx_fit, idx_hold, y
        )
        m = metrics(y_hold, pred, score)
        rows.append(
            {
                "fold": fold,
                "n_hold": int(len(idx_hold)),
                "recall": m["recall"],
                "precision": m["precision"],
                "fnr": m["fnr"],
                "fpr": m["fpr"],
                "pr_auc": m.get("pr_auc", np.nan),
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
            }
        )
    return pd.DataFrame(rows)
