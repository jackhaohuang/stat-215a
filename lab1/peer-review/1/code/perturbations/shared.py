"""Shared holdout helpers for perturbation experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

EXTRA_NONFEATURES = frozenset({"PosIntFinal", "PatNum"})
RANDOM_STATE = 215
TEST_SIZE = 0.20
OUTCOME = "PosIntFinal"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXTRA_NONFEATURES]


def load_labelled_modeling_frame(verbose: bool = False):
    import clean as cl

    df = cl.modeling_frame(verbose=verbose)
    labelled = df.dropna(subset=[OUTCOME]).copy()
    y = labelled[OUTCOME].astype(int)
    feat_cols = feature_columns(labelled)
    return labelled, y, feat_cols


def stratified_holdout(labelled: pd.DataFrame, y: pd.Series):
    idx_fit, idx_hold = train_test_split(
        labelled.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return idx_fit, idx_hold


def metrics(y_true, y_pred, y_score=None) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n_neg, n_pos = fp + tn, fn + tp
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "cm": cm,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "fpr": float(fp / n_neg) if n_neg else float("nan"),
        "fnr": float(fn / n_pos) if n_pos else float("nan"),
        "fp_per_tp": float(fp / tp) if tp else float("nan"),
    }
    if y_score is not None:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    return out


def fit_predict(clf, X_fit, y_fit, X_hold):
    clf.fit(X_fit, y_fit)
    pred = clf.predict(X_hold)
    if hasattr(clf, "predict_proba"):
        try:
            score = clf.predict_proba(X_hold)[:, 1]
        except Exception:
            score = clf.decision_function(X_hold)
    elif hasattr(clf, "decision_function"):
        score = clf.decision_function(X_hold)
    else:
        score = pred.astype(float)
    return pred, np.asarray(score, dtype=float)


def fit_predict_by_age(make_clf, labelled, feat_cols, idx_fit, idx_hold, y):
    """AgeTwoPlus-stratified fit (matches earlier best logistic setup)."""
    cols = [c for c in feat_cols if c != "AgeTwoPlus"]
    y_fit, y_hold = y.loc[idx_fit], y.loc[idx_hold]
    group_fit = labelled.loc[idx_fit, "AgeTwoPlus"]
    group_hold = labelled.loc[idx_hold, "AgeTwoPlus"]
    X_fit = labelled.loc[idx_fit, cols]
    X_hold = labelled.loc[idx_hold, cols]

    pred = np.zeros(len(idx_hold), dtype=int)
    score = np.zeros(len(idx_hold), dtype=float)
    for level in sorted(pd.unique(group_fit.dropna())):
        fit_mask = group_fit.eq(level)
        hold_mask = group_hold.eq(level)
        if not fit_mask.any() or not hold_mask.any():
            continue
        p, s = fit_predict(
            make_clf(),
            X_fit.loc[fit_mask],
            y_fit.loc[fit_mask],
            X_hold.loc[hold_mask],
        )
        pred[hold_mask.to_numpy()] = p
        score[hold_mask.to_numpy()] = s
    return pred, score, y_hold
