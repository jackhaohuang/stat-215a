"""
Compare models on the same train-only stratified holdout:

  - balanced logistic: pooled / preverbal–verbal cohorts
  - HistGradientBoosting: pooled / preverbal–verbal cohorts
  - logistic with threshold swept to match each HGB setup's FP count
    (apples-to-apples recall at the same false-positive budget)

Frozen test.csv is never used. Writes one multi-panel confusion-matrix figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXTRA_NONFEATURES = frozenset({"PosIntFinal", "PatNum"})
RANDOM_STATE = 215
TEST_SIZE = 0.20
OUTCOME = "PosIntFinal"


def _make_logistic() -> Pipeline:
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


def _make_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        min_samples_leaf=40,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXTRA_NONFEATURES]


def _metrics(y_true, y_pred, y_score=None, threshold: float | None = None) -> dict:
    out = {
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "cm": confusion_matrix(y_true, y_pred),
        "threshold": threshold,
    }
    tn, fp, fn, tp = out["cm"].ravel()
    out["tp"], out["fp"], out["fn"], out["tn"] = int(tp), int(fp), int(fn), int(tn)
    out["fp_per_tp"] = float(fp / tp) if tp else float("nan")
    n_neg = fp + tn
    n_pos = fn + tp
    out["fpr"] = float(fp / n_neg) if n_neg else float("nan")
    out["fnr"] = float(fn / n_pos) if n_pos else float("nan")
    if y_score is not None:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out["score"] = np.asarray(y_score, dtype=float)
    return out


def _fit_predict(clf, X_fit, y_fit, X_hold) -> tuple[np.ndarray, np.ndarray]:
    clf.fit(X_fit, y_fit)
    pred = clf.predict(X_hold)
    score = clf.predict_proba(X_hold)[:, 1]
    return pred, score


def _fit_predict_by_group(
    make_clf,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    group_fit: pd.Series,
    X_hold: pd.DataFrame,
    group_hold: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros(len(X_hold), dtype=int)
    score = np.zeros(len(X_hold), dtype=float)
    for level in sorted(pd.unique(group_fit.dropna())):
        fit_mask = group_fit.eq(level)
        hold_mask = group_hold.eq(level)
        if not fit_mask.any() or not hold_mask.any():
            continue
        clf = make_clf()
        p, s = _fit_predict(
            clf,
            X_fit.loc[fit_mask],
            y_fit.loc[fit_mask],
            X_hold.loc[hold_mask],
        )
        pred[hold_mask.to_numpy()] = p
        score[hold_mask.to_numpy()] = s
    return pred, score


def _threshold_match_fp(
    y_true: np.ndarray | pd.Series,
    scores: np.ndarray,
    target_fp: int,
) -> tuple[float, np.ndarray]:
    """
    Pick a probability cutoff whose holdout FP count is as close as possible
    to target_fp (prefer not exceeding when tied).
    """
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    # Fine grid + unique score cutpoints
    grid = np.unique(
        np.concatenate(
            [np.linspace(0.0, 1.0, 1001), scores, scores + 1e-12, [0.0, 1.0]]
        )
    )
    best_thr = 0.5
    best_pred = (scores >= best_thr).astype(int)
    best_key = None  # (abs_diff, fp_overshoot, -recall)

    for thr in grid:
        pred = (scores >= thr).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        abs_diff = abs(fp - target_fp)
        overshoot = max(0, fp - target_fp)
        key = (abs_diff, overshoot, -recall)
        if best_key is None or key < best_key:
            best_key = key
            best_thr = float(thr)
            best_pred = pred

    return best_thr, best_pred


def compare_hgb_baseline(
    df: pd.DataFrame | None = None,
    fig_dir: Path | None = None,
    show: bool = False,
    verbose: bool = True,
) -> dict:
    """
    All model setups + logistic thresholds matched to HGB FP counts.

    Writes report/exploratory_figures/08_all_models_confusion.pdf
    (and refreshes 07_hgb_vs_logistic.pdf for the 2x2 HGB vs default logistic).
    """
    import clean as cl

    if df is None:
        df = cl.modeling_frame(verbose=False)
    else:
        df = cl.modeling_frame(df, verbose=False)

    for col in ("preverbal",):
        if col not in df.columns:
            raise KeyError(f"compare_hgb_baseline needs {col!r}")

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
    y_hold_np = y_hold.to_numpy()

    cols_prev = [c for c in feat_cols if c != "preverbal"]

    setups: dict[str, dict] = {}

    # --- default-threshold models ---
    pred, score = _fit_predict(_make_logistic(), X_fit, y_fit, X_hold)
    setups["log_pooled"] = {
        **_metrics(y_hold, pred, score, threshold=0.5),
        "label": "Logistic pooled\n(default thr=0.5)",
    }

    pred, score = _fit_predict_by_group(
        _make_logistic,
        labelled.loc[idx_fit, cols_prev],
        y_fit,
        labelled.loc[idx_fit, "preverbal"],
        labelled.loc[idx_hold, cols_prev],
        labelled.loc[idx_hold, "preverbal"],
    )
    setups["log_prev"] = {
        **_metrics(y_hold, pred, score, threshold=0.5),
        "label": "Logistic preverbal/verbal\n(default thr=0.5)",
    }

    pred, score = _fit_predict(_make_hgb(), X_fit, y_fit, X_hold)
    setups["hgb_pooled"] = {
        **_metrics(y_hold, pred, score, threshold=0.5),
        "label": "HGB pooled\n(default thr=0.5)",
    }

    pred, score = _fit_predict_by_group(
        _make_hgb,
        labelled.loc[idx_fit, cols_prev],
        y_fit,
        labelled.loc[idx_fit, "preverbal"],
        labelled.loc[idx_hold, cols_prev],
        labelled.loc[idx_hold, "preverbal"],
    )
    setups["hgb_prev"] = {
        **_metrics(y_hold, pred, score, threshold=0.5),
        "label": "HGB preverbal/verbal\n(default thr=0.5)",
    }

    # --- logistic thresholds matched to HGB FP budgets ---
    thr, pred = _threshold_match_fp(
        y_hold_np, setups["log_pooled"]["score"], setups["hgb_pooled"]["fp"]
    )
    setups["log_pooled_matched"] = {
        **_metrics(
            y_hold, pred, setups["log_pooled"]["score"], threshold=thr
        ),
        "label": (
            f"Logistic pooled\n(thr={thr:.3f}, match HGB pooled FP="
            f"{setups['hgb_pooled']['fp']})"
        ),
        "matched_to": "hgb_pooled",
        "target_fp": setups["hgb_pooled"]["fp"],
    }

    thr, pred = _threshold_match_fp(
        y_hold_np, setups["log_prev"]["score"], setups["hgb_prev"]["fp"]
    )
    setups["log_prev_matched"] = {
        **_metrics(y_hold, pred, setups["log_prev"]["score"], threshold=thr),
        "label": (
            f"Logistic preverbal/verbal\n(thr={thr:.3f}, match HGB prev FP="
            f"{setups['hgb_prev']['fp']})"
        ),
        "matched_to": "hgb_prev",
        "target_fp": setups["hgb_prev"]["fp"],
    }

    # Panel order for the all-models figure
    order = (
        "log_pooled",
        "log_prev",
        "hgb_pooled",
        "hgb_prev",
        "log_pooled_matched",
        "log_prev_matched",
    )

    if verbose:
        print(
            f"holdout n={len(y_hold)}  pos_rate={y_hold.mean():.3%}  "
            f"features={len(feat_cols)}  random_state={RANDOM_STATE}"
        )
        print(
            f"{'setup':<28} {'thr':>6} {'acc':>6} {'rec':>6} {'prec':>6} "
            f"{'FPR':>6} {'FNR':>6} {'FP/TP':>6} {'TP':>4} {'FN':>4} {'FP':>4}"
        )
        for key in order:
            r = setups[key]
            thr_s = f"{r['threshold']:.3f}" if r["threshold"] is not None else "—"
            print(
                f"{key:<28} {thr_s:>6} {r['accuracy']:6.3f} {r['recall']:6.3f} "
                f"{r['precision']:6.3f} {r['fpr']:6.3f} {r['fnr']:6.3f} "
                f"{r['fp_per_tp']:6.2f} {r['tp']:4d} {r['fn']:4d} {r['fp']:4d}"
            )
        print("\nApples-to-apples (same FP budget):")
        for log_key, hgb_key in (
            ("log_pooled_matched", "hgb_pooled"),
            ("log_prev_matched", "hgb_prev"),
        ):
            a, b = setups[log_key], setups[hgb_key]
            print(
                f"  {a['label'].split(chr(10))[0]} thr={a['threshold']:.3f} "
                f"FP={a['fp']} recall={a['recall']:.3f}  vs  "
                f"{b['label'].split(chr(10))[0]} FP={b['fp']} recall={b['recall']:.3f}  "
                f"(Δrecall={a['recall'] - b['recall']:+.3f})"
            )

    out_dir = Path(fig_dir) if fig_dir is not None else cl.ensure_fig_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- figure: all models ---
    n = len(order)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 3.6 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes_flat, order):
        r = setups[key]
        ConfusionMatrixDisplay(
            r["cm"], display_labels=["no ciTBI", "ciTBI"]
        ).plot(ax=ax, cmap="flare", colorbar=False, values_format="d")
        thr = r["threshold"]
        thr_bit = f"thr={thr:.3f}" if thr is not None else ""
        ax.set_title(
            f"{r['label']}\n"
            f"rec={r['recall']:.3f}  prec={r['precision']:.3f}  "
            f"FPR={r['fpr']:.3f}  FNR={r['fnr']:.3f}\n"
            f"FP/TP={r['fp_per_tp']:.1f}  {thr_bit}".strip(),
            fontsize=8.5,
        )
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(
        "All models — same train holdout (test.csv untouched)\n"
        "Bottom row: logistic threshold swept to match HGB false-positive count",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig_path = out_dir / "08_all_models_confusion.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"wrote {fig_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # keep the smaller 2x2 for the report (flare = purple→orange)
    panel_keys = ("log_pooled", "hgb_pooled", "log_prev", "hgb_prev")
    panel_titles = {
        "log_pooled": "Logistic — pooled",
        "hgb_pooled": "HistGradientBoosting — pooled",
        "log_prev": "Logistic — preverbal/verbal",
        "hgb_prev": "HistGradientBoosting — preverbal/verbal",
    }
    fig2, axes2 = plt.subplots(1, 4, figsize=(11.0, 3.2))
    for ax, key in zip(axes2.ravel(), panel_keys):
        r = setups[key]
        ConfusionMatrixDisplay(
            r["cm"], display_labels=["no", "ciTBI"]
        ).plot(ax=ax, cmap="flare", colorbar=False, values_format="d")
        ax.set_title(
            f"{panel_titles[key]}\n"
            f"rec={r['recall']:.2f}  FNR={r['fnr']:.2f}\nFPR={r['fpr']:.2f}",
            fontsize=8,
        )
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
    fig2.suptitle(
        "Logistic vs HGB (train holdout, thr=0.5)",
        fontsize=11,
        y=1.02,
    )
    fig2.tight_layout()
    fig2_path = out_dir / "09_logistic_vs_hgb_confusion.pdf"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"wrote {fig2_path}")
    plt.close(fig2)

    # drop large score arrays from return (keep metrics)
    clean_setups = {}
    for k, v in setups.items():
        clean_setups[k] = {kk: vv for kk, vv in v.items() if kk != "score"}

    return {
        **clean_setups,
        "figure_path": fig_path,
        "figure_path_hgb": fig2_path,
        "feat_cols": feat_cols,
        "order": order,
    }


if __name__ == "__main__":
    compare_hgb_baseline(show=False, verbose=True)
