"""
Final evaluation on the frozen test split.

Train on the full training modelling frame; evaluate once on test.csv.
Writes metrics JSON + a confusion-matrix figure under report/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from .compare_hgb_baseline import (
    EXTRA_NONFEATURES,
    OUTCOME,
    RANDOM_STATE,
    _feature_columns,
    _fit_predict,
    _fit_predict_by_group,
    _make_hgb,
    _make_logistic,
    _metrics,
)

CODE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_DIR.parent / "data" / "splits"
FIG_DIR = CODE_DIR.parent / "report" / "exploratory_figures"
OUT_DIR = CODE_DIR.parent / "report" / "final_eval"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


def _align_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use train feature set; add missing test cols as 0, drop extras."""
    X_train = train_df.reindex(columns=cols).copy()
    X_test = test_df.reindex(columns=cols, fill_value=0).copy()
    # reindex fill_value only fills missing columns; NaNs in existing cols stay
    for c in cols:
        if c not in test_df.columns:
            X_test[c] = 0
    X_test = X_test[cols]
    return X_train, X_test


def _load_modeling_splits(verbose: bool = False):
    import clean as cl

    train_raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_raw = pd.read_csv(TEST_PATH, low_memory=False)

    train_m = cl.modeling_frame(train_raw, verbose=verbose)
    test_m = cl.modeling_frame(test_raw, verbose=verbose)

    train_lab = train_m.dropna(subset=[OUTCOME]).copy()
    test_lab = test_m.dropna(subset=[OUTCOME]).copy()
    return train_lab, test_lab


def evaluate_final_test(
    fig_dir: Path | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Fit primary models on full train; score frozen test once.

    Models (same presets as development):
      - logistic preverbal/verbal-stratified
      - HGB preverbal/verbal-stratified
      - logistic pooled
      - HGB pooled
    """
    fig_dir = fig_dir or FIG_DIR
    out_dir = out_dir or OUT_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_lab, test_lab = _load_modeling_splits(verbose=False)
    if "preverbal" not in train_lab.columns or "preverbal" not in test_lab.columns:
        raise KeyError("evaluate_final_test needs 'preverbal' on train and test")

    y_train = train_lab[OUTCOME].astype(int)
    y_test = test_lab[OUTCOME].astype(int)

    feat_cols = _feature_columns(train_lab)
    cols_prev = [c for c in feat_cols if c != "preverbal"]

    X_train_all, X_test_all = _align_columns(train_lab, test_lab, feat_cols)
    X_train_prev, X_test_prev = _align_columns(train_lab, test_lab, cols_prev)

    setups: dict[str, dict] = {}

    pred, score = _fit_predict(_make_logistic(), X_train_all, y_train, X_test_all)
    setups["log_pooled"] = {
        **_metrics(y_test, pred, score, threshold=0.5),
        "label": "Logistic pooled",
    }

    pred, score = _fit_predict(_make_hgb(), X_train_all, y_train, X_test_all)
    setups["hgb_pooled"] = {
        **_metrics(y_test, pred, score, threshold=0.5),
        "label": "HGB pooled",
    }

    pred, score = _fit_predict_by_group(
        _make_logistic,
        X_train_prev,
        y_train,
        train_lab["preverbal"],
        X_test_prev,
        test_lab["preverbal"],
    )
    setups["log_prev"] = {
        **_metrics(y_test, pred, score, threshold=0.5),
        "label": "Logistic preverbal/verbal",
    }

    pred, score = _fit_predict_by_group(
        _make_hgb,
        X_train_prev,
        y_train,
        train_lab["preverbal"],
        X_test_prev,
        test_lab["preverbal"],
    )
    setups["hgb_prev"] = {
        **_metrics(y_test, pred, score, threshold=0.5),
        "label": "HGB preverbal/verbal",
    }

    # JSON-serializable summary (drop raw score arrays / cm objects)
    summary = {
        "n_train": int(len(train_lab)),
        "n_test": int(len(test_lab)),
        "test_pos": int(y_test.sum()),
        "test_pos_rate": float(y_test.mean()),
        "n_features": len(feat_cols),
        "random_state": RANDOM_STATE,
        "models": {},
    }
    for key, r in setups.items():
        summary["models"][key] = {
            "label": r["label"],
            "tp": r["tp"],
            "fp": r["fp"],
            "fn": r["fn"],
            "tn": r["tn"],
            "recall": r["recall"],
            "precision": r["precision"],
            "fpr": r["fpr"],
            "fnr": r["fnr"],
            "fp_per_tp": r["fp_per_tp"],
            "pr_auc": r.get("pr_auc"),
            "threshold": r.get("threshold"),
        }

    metrics_path = out_dir / "final_test_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")

    # confusion-matrix panel
    order = ["log_pooled", "log_prev", "hgb_pooled", "hgb_prev"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
    for ax, key in zip(axes.ravel(), order):
        r = setups[key]
        disp = ConfusionMatrixDisplay(confusion_matrix=r["cm"], display_labels=[0, 1])
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(
            f"{r['label']}\n"
            f"FN={r['fn']} FP={r['fp']}  FNR={r['fnr']:.3f} FPR={r['fpr']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle(
        f"Final test evaluation (train n={len(train_lab):,}, test n={len(test_lab):,})",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = fig_dir / "14_final_test_confusion.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"wrote {metrics_path}")
        print(f"wrote {fig_path}")
        print(json.dumps(summary["models"], indent=2))

    summary["metrics_path"] = str(metrics_path)
    summary["figure_path"] = str(fig_path)
    summary["setups"] = setups
    return summary


def main():
    evaluate_final_test(verbose=True)


if __name__ == "__main__":
    main()
