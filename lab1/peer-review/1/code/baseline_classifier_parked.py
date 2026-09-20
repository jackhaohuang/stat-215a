"""
Parked from exploratory.ipynb §6 — baseline logistic regression + confusion matrix.
Not imported by the main pipeline; kept so we can drop it back into the notebook later.

Original notes
--------------
Simple logistic regression on the informative predictors, evaluated on an 80/20
holdout *inside* the train split. The frozen test.csv is still untouched.

Caveats:
- outcome is ~1.8% positive, so accuracy is a weak summary
- class_weight="balanced" trades precision for recall (many false positives)
- 91/92 are left as numeric codes here — not yet cleaned
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import clean as cl

DROP_FROM_MODEL = {
    "DeathTBI", "HospHead", "HospHeadPosCT", "Intub24Head", "Neurosurgery", "PosIntFinal",
    "EmplType", "Certification", "EDDisposition", "CTDone", "CTForm1",
    "AgeinYears",
}


def run_baseline(df: pd.DataFrame, core_cols: list[str] | None = None, show: bool = True):
    if core_cols is None:
        gates = cl.find_gates(df)
        core_cols = cl.core_columns(df, gates)

    feat_cols = [c for c in core_cols if c not in DROP_FROM_MODEL]
    labelled = df.dropna(subset=["PosIntFinal"]).copy()
    X = labelled[feat_cols]
    y = labelled["PosIntFinal"].astype(int)

    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=0.20, random_state=215, stratify=y
    )

    clf = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=215,
                ),
            ),
        ]
    )
    clf.fit(X_fit, y_fit)
    y_pred = clf.predict(X_hold)
    cm = confusion_matrix(y_hold, y_pred)

    print(f"features: {len(feat_cols)}  |  fit n={len(y_fit)}  holdout n={len(y_hold)}")
    print(f"holdout positive rate: {y_hold.mean():.3%}")
    print(f"accuracy:  {accuracy_score(y_hold, y_pred):.3f}  (misleading with imbalance)")
    print(f"recall:    {recall_score(y_hold, y_pred):.3f}")
    print(f"precision: {precision_score(y_hold, y_pred, zero_division=0):.3f}")
    print()
    print(classification_report(y_hold, y_pred, digits=3, target_names=["no ciTBI", "ciTBI"]))

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay(cm, display_labels=["no ciTBI", "ciTBI"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("Baseline logistic regression\n(confusion matrix on train holdout)")
    fig.tight_layout()
    out = cl.ensure_fig_dir() / "05_baseline_confusion_matrix.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"clf": clf, "cm": cm, "feat_cols": feat_cols}


if __name__ == "__main__":
    run_baseline(cl.load_train(), show=False)
