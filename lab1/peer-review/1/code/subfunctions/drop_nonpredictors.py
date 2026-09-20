"""
Strip columns that should not be model features.

Keeps PosIntFinal as the label and PatNum as the id. Drops:
  - site / clinician admin fields (EmplType, Certification, EDDisposition)
  - component outcome fields that leak or redefine the label
    (DeathTBI, HospHead, HospHeadPosCT, Intub24Head, Neurosurgery)

Call this when building a modeling frame; leave them in for EDA if useful.
"""

from __future__ import annotations

import pandas as pd

ADMIN_COLS = frozenset({"EmplType", "Certification", "EDDisposition"})
OUTCOME_COMPONENT_COLS = frozenset(
    {
        "DeathTBI",
        "HospHead",
        "HospHeadPosCT",
        "Intub24Head",
        "Neurosurgery",
    }
)
NONPREDICTOR_COLS = ADMIN_COLS | OUTCOME_COMPONENT_COLS


def drop_nonpredictors(
    df: pd.DataFrame,
    extra_drop: set[str] | frozenset[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Return (df without non-predictor columns, list of dropped names).

    PosIntFinal is retained (label). PatNum is retained (id).
    """
    to_drop = set(NONPREDICTOR_COLS)
    if extra_drop:
        to_drop |= set(extra_drop)
    present = [c for c in to_drop if c in df.columns]
    out = df.drop(columns=present)
    if verbose:
        print(
            f"drop_nonpredictors: removed {len(present)} columns "
            f"({df.shape[1]} -> {out.shape[1]}): {sorted(present)}"
        )
    return out, present
