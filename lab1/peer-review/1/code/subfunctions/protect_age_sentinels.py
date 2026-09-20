"""
AgeInMonth / AgeinYears: 91 and 92 are real ages, never sentinel codes.

Any step that recodes 91/92 must call exclude_protected_age() (or pass
protect=PROTECTED_AGE into recode helpers) so these columns are skipped.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

PROTECTED_AGE = frozenset({"AgeInMonth", "AgeinYears"})
SENTINELS = (91, 92)


def exclude_protected_age(columns: Iterable[str]) -> list[str]:
    """Filter a column list so age fields are never sentinel-recoded."""
    return [c for c in columns if c not in PROTECTED_AGE]


def recode_sentinels(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    sentinels: Iterable[float | int] = SENTINELS,
    protect: frozenset[str] = PROTECTED_AGE,
    replacement=pd.NA,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Replace sentinel codes with `replacement` in selected columns, never in age.

    Default replacement is NA. Prefer not calling this on HA_verb / Amnesia_verb
    (those 91s are structural — see handle_preverbal).
    """
    out = df.copy()
    cols = list(columns) if columns is not None else list(out.columns)
    cols = [c for c in cols if c in out.columns and c not in protect]
    n = 0
    for c in cols:
        mask = out[c].isin(list(sentinels))
        n += int(mask.sum())
        out.loc[mask, c] = replacement
    if verbose:
        skipped = sorted(protect & set(df.columns))
        print(
            f"protect_age / recode_sentinels: replaced {n} sentinel cells in "
            f"{len(cols)} columns; skipped {skipped}"
        )
    return out


def protect_age_sentinels(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Cleaning step: confirm age columns are present and leave them unchanged.

    Does not rewrite data — registers the protection rule for the pipeline and
    reports how many 91/92 values in AgeInMonth are real ages (not codes).
    """
    out = df.copy()
    if verbose:
        for c in sorted(PROTECTED_AGE & set(out.columns)):
            n91 = int(out[c].eq(91).sum())
            n92 = int(out[c].eq(92).sum())
            print(
                f"protect_age_sentinels: {c} left as-is "
                f"({n91} values==91, {n92} values==92 are ages, not codes)"
            )
    return out
