"""
Dizzy blanks among preverbal children are structural, not MAR missingness.

~87% of preverbal kids have blank Dizzy vs ~8% of verbal — same clinical
meaning as code 91 (cannot self-report). We mark those blanks so they are
not median-imputed into a false symptom value.
"""

from __future__ import annotations

import pandas as pd

# Align with HA_verb / Amnesia_verb preverbal coding
STRUCTURAL_CODE = 91


def recode_dizzy_structural(
    df: pd.DataFrame,
    preverbal_col: str = "preverbal",
    dizzy_col: str = "Dizzy",
    structural_code: int = STRUCTURAL_CODE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For rows with preverbal==1 and Dizzy blank:
      - set Dizzy to `structural_code` (91)
      - set Dizzy_structural=1 (0 otherwise)

    Requires `preverbal` from handle_preverbal.add_preverbal_flag.
    """
    out = df.copy()
    if dizzy_col not in out.columns:
        raise KeyError(f"dizzy_structural needs column {dizzy_col!r}")
    if preverbal_col not in out.columns:
        raise KeyError(
            f"dizzy_structural needs {preverbal_col!r} — run handle_preverbal first"
        )

    preverbal = out[preverbal_col].eq(1)
    blank = out[dizzy_col].isna()
    structural = preverbal & blank

    out["Dizzy_structural"] = structural.astype(int)
    out.loc[structural, dizzy_col] = structural_code

    if verbose:
        n = int(structural.sum())
        still_blank = int(out[dizzy_col].isna().sum())
        print(
            f"dizzy_structural: marked {n} preverbal Dizzy blanks as code "
            f"{structural_code}; remaining Dizzy NaNs (mostly verbal): {still_blank}"
        )
    return out
