"""Load and clean the dialect survey data.

Cleaning steps:
  1. drop respondents without latitude/longitude
  2. keep only the contiguous US
  3. drop respondents who left more than 10% of the 67 questions unanswered
  4. remaining unanswered questions (coded 0) stay in as "no answer"
  5. keep unique respondents and the first copy of every duplicated answer
     vector. Later copies sit on neighboring IDs but carry other people's
     locations, so their answers do not belong to their location. Copies are
     found on the raw data, so earlier steps cannot change which one is first.
"""

from __future__ import annotations

import os
import pathlib

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from pyreadr import read_r

LAB_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = LAB_DIR / "data"
FIGS_DIR = LAB_DIR / "figs"
VALUES_DIR = LAB_DIR / "report" / "values"

# Bounding box of the contiguous United States.
LAT_RANGE = (24.0, 50.0)
LONG_RANGE = (-125.0, -66.0)
MAX_MISSING_FRAC = 0.10
SEED = 215


def load_raw() -> pd.DataFrame:
    # Return lingData.txt exactly as it is on disk
    return pd.read_csv(DATA_DIR / "lingData.txt", sep=r"\s+")

#class Questions(NamedTuple):
 #   """Question metadata from question_data.RData."""
 #   n_options: pd.Series  # number of answer options, indexed by "Q050" ...
   # text: dict[str, str]  # question text per column
   # answers: dict[str, list[str]]  # answer texts per column, option 1 first

def question_columns(df: pd.DataFrame) -> list[str]:
    # Return the question columns (Q50 ... Q121) in file order
    return [c for c in df.columns if c.startswith("Q")]


def load_questions() -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    # Return the question texts and its answer table per question number

    # The answer table has one row per option -> row i (0-based) is the answer coded as i + 1 in lingData
    raw = read_r(str(DATA_DIR / "question_data.RData"))
    quest = raw["quest.use"].copy()
    quest["qnum"] = quest["qnum"].astype(int)
    answers = {int(k.split(".")[1]): v.reset_index(drop=True)
               for k, v in raw.items() if k.startswith("ans.")}
    return quest, answers


def later_copies(raw: pd.DataFrame) -> pd.Series:
    # Flag later copies of duplicated answer vectors in the raw data
    # Copies are ranked by ID and respondents that answered nothing are never flagged
    q = question_columns(raw)
    answered = (raw[q] != 0).any(axis=1)
    by_id = raw.sort_values("ID")
    later = by_id.duplicated(subset=q, keep="first") & answered.loc[by_id.index]
    return later.reindex(raw.index)


def _filter_steps(raw: pd.DataFrame,
                  keep_copies: bool = False) -> list[tuple[str, pd.DataFrame]]:
    """Apply the cleaning steps one by one. With keep_copies=True the later copies of duplicated answer vectors stay
    in the data -> that is the sensitivity check of the main sample
    """
    q = question_columns(raw)
    later = later_copies(raw)
    steps = [("raw", raw)]
    df = raw.dropna(subset=["lat", "long"])
    steps.append(("has_location", df))
    df = df[df["lat"].between(*LAT_RANGE) & df["long"].between(*LONG_RANGE)]
    steps.append(("contiguous_us", df))
    n_missing = (df[q] == 0).sum(axis=1)
    df = df[n_missing <= MAX_MISSING_FRAC * len(q)]
    steps.append(("mostly_complete", df))
    if not keep_copies:
        df = df[~later.loc[df.index]]
        steps.append(("deduplicated", df))
    return steps


def load_clean(keep_copies: bool = False) -> pd.DataFrame:
    # Return the cleaned respondent-level data with one row per person
    return _filter_steps(load_raw(), keep_copies)[-1][1].reset_index(drop=True)


def cleaning_log() -> dict[str, int]:
    # Return the number of respondents left after each cleaning step
    return {name: len(df) for name, df in _filter_steps(load_raw())}


def one_hot(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Binary-encode the categorical answers, that returns an n x 468 0/1 matrix and the column names ("Q050_1", ...).
    The number of options per question is taken from the full raw data -> the
    width is always 468 even if some option is never chosen after cleaning.
    A 0 ("no answer") gives an all-zero block for that question.
    """
    q = question_columns(df)
    n_options = load_raw()[q].max()
    blocks, names = [], []
    for col in q:
        k = int(n_options[col])
        codes = df[col].to_numpy()
        block = (codes[:, None] == np.arange(1, k + 1)[None, :]).astype(np.uint8)
        blocks.append(block)
        names += [f"{col}_{j}" for j in range(1, k + 1)]
    return np.hstack(blocks), names


def load_states():
    # Return contiguous-US state polygons for map backgrounds
    import geopandas as gpd
    states = gpd.read_file(DATA_DIR / "shapefiles")
    states = states[(states["iso_a2"] == "US") & ~states["name"].isin(["Alaska", "Hawaii"])]
    return states


def write_values(name: str, values: dict[str, object]) -> None:
    # This function writer numbers for the report as LaTeX macros in report/values/<name>.tex.
    # -> no copy and pasting of the individual numbers
    VALUES_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"% Generated by the scripts in code/ ({name}) -- do not edit by hand."]
    for key, val in sorted(values.items()):
        if not key.isalpha():
            raise ValueError(f"macro name must be letters only: {key}")
        if isinstance(val, (int, np.integer)):
            val = f"{int(val):,}".replace(",", "{,}")
        lines.append(f"\\newcommand{{\\{key}}}{{{val}}}")
    (VALUES_DIR / f"{name}.tex").write_text("\n".join(lines) + "\n")
