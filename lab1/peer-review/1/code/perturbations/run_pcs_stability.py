"""
Run PCS stability checks for the AgeTwoPlus logistic baseline:

  1) Algorithm perturbation: logistic vs RidgeClassifier (same holdout)
  2) Data perturbation: stratified 5-fold CV of the logistic

Writes ``report/exploratory_figures/10_pcs_perturbation_stability.pdf``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .algorithm import make_logistic, make_ridge
from .resampling import stratified_kfold_metrics
from .shared import (
    fit_predict_by_age,
    load_labelled_modeling_frame,
    metrics,
    stratified_holdout,
)


def run_pcs_stability(
    fig_dir: Path | None = None,
    show: bool = False,
    verbose: bool = True,
    n_splits: int = 5,
) -> dict:
    import clean as cl

    labelled, y, feat_cols = load_labelled_modeling_frame(verbose=False)
    idx_fit, idx_hold = stratified_holdout(labelled, y)

    # --- algorithm perturbation on the fixed holdout ---
    pred_log, score_log, y_hold = fit_predict_by_age(
        make_logistic, labelled, feat_cols, idx_fit, idx_hold, y
    )
    pred_ridge, score_ridge, _ = fit_predict_by_age(
        make_ridge, labelled, feat_cols, idx_fit, idx_hold, y
    )
    m_log = metrics(y_hold, pred_log, score_log)
    m_ridge = metrics(y_hold, pred_ridge, score_ridge)

    # agreement on holdout predictions
    agree = float(np.mean(pred_log == pred_ridge))

    # --- data perturbation: k-fold of pooled logistic (features w/o AgeTwoPlus key ok) ---
    cols = [c for c in feat_cols if c != "AgeTwoPlus"]
    # For k-fold we use pooled logistic on all train rows for a clean
    # resampling story (one learner, many splits).
    fold_df = stratified_kfold_metrics(
        make_logistic,
        labelled[cols],
        y,
        n_splits=n_splits,
    )

    if verbose:
        print(
            f"holdout n={len(y_hold)}  pos_rate={y_hold.mean():.3%}  "
            f"prediction agreement logistic↔ridge={agree:.3f}"
        )
        print(
            f"{'model':<18} {'rec':>6} {'prec':>6} {'FPR':>6} {'FNR':>6} "
            f"{'FP/TP':>6} {'TP':>4} {'FN':>4} {'FP':>4}"
        )
        for name, m in (("logistic L2", m_log), ("RidgeClassifier", m_ridge)):
            print(
                f"{name:<18} {m['recall']:6.3f} {m['precision']:6.3f} "
                f"{m['fpr']:6.3f} {m['fnr']:6.3f} {m['fp_per_tp']:6.2f} "
                f"{m['tp']:4d} {m['fn']:4d} {m['fp']:4d}"
            )
        print(f"\n{n_splits}-fold logistic (pooled) — recall / FNR by fold:")
        print(
            fold_df[["fold", "recall", "fnr", "fpr", "precision", "tp", "fn", "fp"]]
            .to_string(index=False)
        )
        print(
            f"recall mean±sd = {fold_df['recall'].mean():.3f}±{fold_df['recall'].std():.3f}  "
            f"FNR mean±sd = {fold_df['fnr'].mean():.3f}±{fold_df['fnr'].std():.3f}"
        )

    out_dir = Path(fig_dir) if fig_dir is not None else cl.ensure_fig_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- figure: before/after algorithm + k-fold stability ---
    fig = plt.figure(figsize=(12.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.95], hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    for ax, m, title in (
        (ax0, m_log, "Before — Logistic L2 (AgeTwoPlus)\n(default algorithm)"),
        (ax1, m_ridge, "After — RidgeClassifier (AgeTwoPlus)\n(algorithm perturbation)"),
    ):
        ConfusionMatrixDisplay(
            m["cm"], display_labels=["no ciTBI", "ciTBI"]
        ).plot(ax=ax, cmap="flare", colorbar=False, values_format="d")
        ax.set_title(
            f"{title}\n"
            f"rec={m['recall']:.3f}  FNR={m['fnr']:.3f}  "
            f"FPR={m['fpr']:.3f}  FP/TP={m['fp_per_tp']:.1f}",
            fontsize=10,
        )

    x = fold_df["fold"].to_numpy()
    w = 0.35
    ax2.bar(x - w / 2, fold_df["recall"], width=w, label="recall", color="#7b3294")
    ax2.bar(x + w / 2, fold_df["fnr"], width=w, label="FNR", color="#e66101")
    ax2.axhline(
        fold_df["recall"].mean(),
        color="#7b3294",
        ls="--",
        lw=1,
        label=f"recall mean={fold_df['recall'].mean():.3f}",
    )
    ax2.axhline(
        fold_df["fnr"].mean(),
        color="#e66101",
        ls="--",
        lw=1,
        label=f"FNR mean={fold_df['fnr'].mean():.3f}",
    )
    ax2.set_xticks(x)
    ax2.set_xlabel("stratified fold")
    ax2.set_ylabel("rate")
    ax2.set_ylim(0, 1)
    ax2.set_title(
        f"Data perturbation — stratified {n_splits}-fold CV of logistic L2 (pooled)\n"
        f"recall sd={fold_df['recall'].std():.3f}, FNR sd={fold_df['fnr'].std():.3f} "
        f"(same train pool; test.csv untouched)",
        fontsize=10,
    )
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "PCS stability: algorithm swap (Ridge) + resampling (k-fold)",
        fontsize=13,
        y=0.98,
    )
    fig_path = out_dir / "10_pcs_perturbation_stability.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"wrote {fig_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "logistic": m_log,
        "ridge": m_ridge,
        "agreement": agree,
        "kfold": fold_df,
        "figure_path": fig_path,
    }


if __name__ == "__main__":
    run_pcs_stability(show=False, verbose=True)
