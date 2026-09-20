"""
Compare three baseline setups on a train-only stratified holdout:

  1. Pooled logistic regression (one model on everyone)
  2. Separate models by `preverbal` (0 vs 1)
  3. Separate models by `AgeTwoPlus` (1 = under 2, 2 = 2+)

Frozen test.csv is never used. Same holdout patients for all three so
metrics are comparable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Still strip label/id from the feature matrix after modeling_frame()
EXTRA_NONFEATURES = frozenset({"PosIntFinal", "PatNum"})

RANDOM_STATE = 215
TEST_SIZE = 0.20
OUTCOME = "PosIntFinal"


def _make_clf() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _feature_columns(df: pd.DataFrame, extra_drop: set[str] | None = None) -> list[str]:
    drop = set(EXTRA_NONFEATURES)
    if extra_drop:
        drop |= set(extra_drop)
    return [c for c in df.columns if c not in drop]


def _metrics(y_true, y_pred) -> dict:
    return {
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "cm": confusion_matrix(y_true, y_pred),
    }


def _fit_predict_pooled(X_fit, y_fit, X_hold) -> np.ndarray:
    clf = _make_clf()
    clf.fit(X_fit, y_fit)
    return clf.predict(X_hold)


def _fit_predict_by_group(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    group_fit: pd.Series,
    X_hold: pd.DataFrame,
    group_hold: pd.Series,
    feature_cols: list[str],
) -> np.ndarray:
    """Train one logistic per group level; predict holdout with the matching model."""
    pred = np.zeros(len(X_hold), dtype=int)
    levels = sorted(pd.unique(group_fit.dropna()))
    for level in levels:
        fit_mask = group_fit.eq(level)
        hold_mask = group_hold.eq(level)
        if not fit_mask.any() or not hold_mask.any():
            continue
        # drop near-constant group key from features if present
        cols = [c for c in feature_cols if c in X_fit.columns]
        clf = _make_clf()
        clf.fit(X_fit.loc[fit_mask, cols], y_fit.loc[fit_mask])
        pred[hold_mask.to_numpy()] = clf.predict(X_hold.loc[hold_mask, cols])
    return pred


def compare_cohort_baselines(
    df: pd.DataFrame | None = None,
    fig_dir: Path | None = None,
    show: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run pooled vs preverbal-cohort vs AgeTwoPlus-cohort baselines.

    If df is None, loads train and runs clean_data() first (needs preverbal).
    Writes a 1x3 confusion-matrix figure when fig_dir is set (default: clean.FIG_DIR).
    """
    import clean as cl

    if df is None:
        df = cl.modeling_frame(verbose=False)
    else:
        df = cl.modeling_frame(df, verbose=False)

    if "preverbal" not in df.columns:
        raise KeyError("compare_cohort_baselines needs 'preverbal' — run clean_data first")
    if "AgeTwoPlus" not in df.columns:
        raise KeyError("compare_cohort_baselines needs 'AgeTwoPlus'")

    labelled = df.dropna(subset=[OUTCOME]).copy()
    y = labelled[OUTCOME].astype(int)
    feat_cols = _feature_columns(labelled)

    idx_fit, idx_hold = train_test_split(
        labelled.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X = labelled[feat_cols]
    y_fit, y_hold = y.loc[idx_fit], y.loc[idx_hold]
    X_fit, X_hold = X.loc[idx_fit], X.loc[idx_hold]

    # 1) pooled
    pred_pooled = _fit_predict_pooled(X_fit, y_fit, X_hold)

    # 2) preverbal cohorts — don't use preverbal as a feature inside each arm
    cols_prev = [c for c in feat_cols if c != "preverbal"]
    pred_prev = _fit_predict_by_group(
        labelled.loc[idx_fit, cols_prev],
        y_fit,
        labelled.loc[idx_fit, "preverbal"],
        labelled.loc[idx_hold, cols_prev],
        labelled.loc[idx_hold, "preverbal"],
        cols_prev,
    )

    # 3) age-band cohorts
    cols_age = [c for c in feat_cols if c != "AgeTwoPlus"]
    pred_age = _fit_predict_by_group(
        labelled.loc[idx_fit, cols_age],
        y_fit,
        labelled.loc[idx_fit, "AgeTwoPlus"],
        labelled.loc[idx_hold, cols_age],
        labelled.loc[idx_hold, "AgeTwoPlus"],
        cols_age,
    )

    results = {
        "pooled": _metrics(y_hold, pred_pooled),
        "preverbal_cohorts": _metrics(y_hold, pred_prev),
        "age_cohorts": _metrics(y_hold, pred_age),
    }
    results["pooled"]["label"] = "Pooled"
    results["preverbal_cohorts"]["label"] = "Preverbal cohorts"
    results["age_cohorts"]["label"] = "AgeTwoPlus cohorts"

    # per-arm sizes on holdout (for the writeup)
    results["holdout_group_counts"] = {
        "preverbal": labelled.loc[idx_hold, "preverbal"].value_counts().sort_index().to_dict(),
        "AgeTwoPlus": labelled.loc[idx_hold, "AgeTwoPlus"].value_counts().sort_index().to_dict(),
    }

    if verbose:
        print(
            f"holdout n={len(y_hold)}  pos_rate={y_hold.mean():.3%}  "
            f"features(pooled)={len(feat_cols)}  random_state={RANDOM_STATE}"
        )
        print(f"{'setup':<22} {'acc':>7} {'recall':>7} {'prec':>7} {'TP':>5} {'FN':>5} {'FP':>5}")
        for key in ("pooled", "preverbal_cohorts", "age_cohorts"):
            r = results[key]
            tn, fp, fn, tp = r["cm"].ravel()
            print(
                f"{r['label']:<22} {r['accuracy']:7.3f} {r['recall']:7.3f} "
                f"{r['precision']:7.3f} {tp:5d} {fn:5d} {fp:5d}"
            )

    # figure
    out_dir = fig_dir if fig_dir is not None else cl.ensure_fig_dir()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, key in zip(axes, ("pooled", "preverbal_cohorts", "age_cohorts")):
        r = results[key]
        ConfusionMatrixDisplay(
            r["cm"], display_labels=["no ciTBI", "ciTBI"]
        ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(
            f"{r['label']}\n"
            f"acc={r['accuracy']:.3f}  rec={r['recall']:.3f}  prec={r['precision']:.3f}"
        )
    fig.suptitle(
        "Baseline logistic comparison (same train holdout; test.csv untouched)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig_path = out_dir / "06_cohort_baseline_comparison.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"wrote {fig_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    results["figure_path"] = fig_path
    results["feat_cols"] = feat_cols
    return results


if __name__ == "__main__":
    compare_cohort_baselines(show=False, verbose=True)
