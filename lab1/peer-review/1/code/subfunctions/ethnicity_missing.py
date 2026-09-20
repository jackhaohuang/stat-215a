"""
Ethnicity is ~37% blank with no shared cause with other fields — real missingness.

Do not mode-impute (that invents demographics). Keep Ethnicity and add a binary
Ethnicity_missing indicator so models can use both the observed value and the
fact of non-response.
"""

from __future__ import annotations

import pandas as pd


def add_ethnicity_missing(
    df: pd.DataFrame,
    col: str = "Ethnicity",
    indicator: str = "Ethnicity_missing",
    verbose: bool = True,
) -> pd.DataFrame:
    """Add `{indicator}` = 1 where Ethnicity is blank; leave Ethnicity NaNs as-is."""
    out = df.copy()
    if col not in out.columns:
        raise KeyError(f"add_ethnicity_missing needs column {col!r}")

    missing = out[col].isna()
    out[indicator] = missing.astype(int)

    if verbose:
        n = int(missing.sum())
        print(
            f"ethnicity_missing: {indicator}=1 for {n} / {len(out)} "
            f"({100 * n / len(out):.1f}%); left {col} NaNs un-imputed"
        )
    return out
