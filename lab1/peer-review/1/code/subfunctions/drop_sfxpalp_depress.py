"""
SFxPalpDepress is ~99% coded 92 (not asked) — useless as a numeric feature.

Drop it. The parent SFxPalp already carries the information; optionally add
SFxPalp_unclear for level 2 ("unclear"), which is what made Depress
non-deterministic in the gate test.
"""

from __future__ import annotations

import pandas as pd

DEPRESS_COL = "SFxPalpDepress"
PALP_COL = "SFxPalp"
UNCLEAR_COL = "SFxPalp_unclear"


def drop_sfxpalp_depress(
    df: pd.DataFrame,
    add_unclear: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Drop SFxPalpDepress. If add_unclear and SFxPalp is present, add
    SFxPalp_unclear = 1 where SFxPalp == 2.
    """
    out = df.copy()
    had = DEPRESS_COL in out.columns
    if had:
        n92 = int(out[DEPRESS_COL].eq(92).sum())
        out = out.drop(columns=[DEPRESS_COL])
    else:
        n92 = 0

    if add_unclear and PALP_COL in out.columns:
        out[UNCLEAR_COL] = out[PALP_COL].eq(2).astype(int)

    if verbose:
        msg = f"drop_sfxpalp_depress: dropped {DEPRESS_COL}" if had else (
            f"drop_sfxpalp_depress: {DEPRESS_COL} already absent"
        )
        if had:
            msg += f" ({n92} were coded 92)"
        if add_unclear and UNCLEAR_COL in out.columns:
            msg += f"; added {UNCLEAR_COL} for {int(out[UNCLEAR_COL].sum())} rows"
        print(msg)
    return out
