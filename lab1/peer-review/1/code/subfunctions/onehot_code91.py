"""
One-hot encode coded categoricals so logistic regression does not treat
codes as magnitudes (e.g. Race==90 as "ninety", LOCSeparate==2 as twice LOC).

After drop_deterministic_children, almost no 92s remain in the modeling frame;
this still covers:
  - 91 structural levels (HA_verb / Amnesia_verb / Dizzy)
  - multi-level / sentinel-style codes (Race 90, LOCSeparate 2, SFxPalp 2, …)

Pure 0/1 clinical binaries (AMS, Vomit, …) are left numeric — equivalent to
one dummy for logistic. GCS Eye/Verbal/Motor stay ordinal numeric.
"""

from __future__ import annotations

import pandas as pd

# Columns where codes are categories, not continuous quantities.
# Binary 1/2 fields (Gender, Ethnicity, AgeTwoPlus, GCSGroup) stay numeric —
# equivalent for logistic, and AgeTwoPlus is needed for cohort splits.
CATEGORICAL_CODE_COLS = (
    # structural 91
    "HA_verb",
    "Amnesia_verb",
    "Dizzy",
    # multi-level / non-magnitude codes
    "LOCSeparate",  # 0 / 1 / 2 (suspected)
    "SFxPalp",  # 0 / 1 / 2 (unclear)
    "High_impact_InjSev",  # 1 / 2 / 3
    "Race",  # 1–5, 90 (other)
)

# Backward-compatible alias used by older call sites / docs.
CATEGORICAL_91_COLS = CATEGORICAL_CODE_COLS

# Derived from SFxPalp==2; redundant once SFxPalp is one-hot.
REDUNDANT_AFTER_ONEHOT = frozenset({"SFxPalp_unclear"})

MISSING_LEVEL = "missing"


def _level_label(v) -> str:
    if v == MISSING_LEVEL or pd.isna(v):
        return MISSING_LEVEL
    try:
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
        return str(fv)
    except (TypeError, ValueError):
        return str(v)


def onehot_code91_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...] = CATEGORICAL_CODE_COLS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Replace each listed column with one-hot dummies.

    NaN -> explicit '{col}_missing'. Values become string levels so 0/1/2/90/91
    are separate categories. Original columns are dropped. Also drops
    SFxPalp_unclear if SFxPalp was encoded (same information as SFxPalp_2).
    """
    out = df.copy()
    present = [c for c in columns if c in out.columns]
    if not present:
        if verbose:
            print("onehot_code91_columns: nothing to encode")
        return out

    dummy_frames = []
    for c in present:
        s = out[c].map(_level_label)
        dummies = pd.get_dummies(s, prefix=c, dtype=int)
        dummy_frames.append(dummies)

    out = out.drop(columns=present)

    # SFxPalp_2 already carries "unclear"
    drop_extra = [
        c for c in REDUNDANT_AFTER_ONEHOT
        if c in out.columns and "SFxPalp" in present
    ]
    if drop_extra:
        out = out.drop(columns=drop_extra)

    out = pd.concat([out, *dummy_frames], axis=1)

    if verbose:
        n_new = sum(f.shape[1] for f in dummy_frames)
        extra = f"; dropped redundant {drop_extra}" if drop_extra else ""
        print(
            f"onehot_code91_columns: encoded {present} -> {n_new} dummy columns "
            f"(codes are categories, not continuous values){extra}"
        )
    return out
