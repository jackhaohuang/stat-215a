"""
Handle code 91 as structure (preverbal), not missingness.

91 appears only on HA_verb and Amnesia_verb. We:
  - add a binary `preverbal` flag (91 on either column)
  - leave 91 in those columns as its own category (never mean-impute away)
"""

from __future__ import annotations

import pandas as pd

PREVERBAL_COLS = ("HA_verb", "Amnesia_verb")


def add_preverbal_flag(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add `preverbal` (0/1): 1 if HA_verb==91 or Amnesia_verb==91.

    Leaves 91 in place on HA_verb / Amnesia_verb so one-hot / categorical
    encoders can treat it as its own level.
    """
    out = df.copy()
    missing = [c for c in PREVERBAL_COLS if c not in out.columns]
    if missing:
        raise KeyError(f"handle_preverbal needs columns {PREVERBAL_COLS}; missing {missing}")

    preverbal = out["HA_verb"].eq(91) | out["Amnesia_verb"].eq(91)
    out["preverbal"] = preverbal.astype(int)

    if verbose:
        n = int(preverbal.sum())
        print(
            f"handle_preverbal: added preverbal flag — {n} / {len(out)} "
            f"({100 * n / len(out):.1f}%); left 91 intact on {list(PREVERBAL_COLS)}"
        )
    return out
