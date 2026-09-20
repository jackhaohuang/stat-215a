"""
Drop columns whose 92-codes are fully determined by a parent question.

These gated children carry no information beyond the parent (SFxBas, AMS, Vomit, …).
SFxPalpDepress is kept: it is not perfectly explained by any parent (SFxPalp=2 also
gates it off), so it is not in the deterministic set — listed in KEEP_ALWAYS as a
guardrail.
"""

from __future__ import annotations

import pandas as pd

# Never drop these even if a future gate test marks them deterministic.
KEEP_ALWAYS = frozenset({"SFxPalpDepress"})


def deterministic_children(
    gates: pd.DataFrame,
    keep: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Column names to drop: deterministic gated children, minus keep."""
    keep = KEEP_ALWAYS if keep is None else frozenset(keep) | KEEP_ALWAYS
    if "deterministic" not in gates.columns or "child" not in gates.columns:
        raise ValueError("gates must have columns 'child' and 'deterministic'")
    children = gates.loc[gates["deterministic"], "child"].tolist()
    return [c for c in children if c not in keep]


def drop_deterministic_children(
    df: pd.DataFrame,
    gates: pd.DataFrame,
    keep: set[str] | frozenset[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Return (df without deterministic gated children, list of dropped names).

    Parents (SFxBas, AMS, Vomit, CTDone, …) are retained. SFxPalpDepress is retained.
    """
    to_drop = [c for c in deterministic_children(gates, keep=keep) if c in df.columns]
    out = df.drop(columns=to_drop)
    if verbose:
        kept_guard = sorted(KEEP_ALWAYS & set(df.columns))
        print(
            f"drop_deterministic_children: removed {len(to_drop)} columns "
            f"({df.shape[1]} -> {out.shape[1]})"
        )
        if kept_guard:
            print(f"  kept by exception / non-deterministic: {kept_guard}")
    return out, to_drop
