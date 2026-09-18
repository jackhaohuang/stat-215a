from __future__ import annotations

import numpy as np
import pandas as pd

# Columns where the documentation defines 92 as "Not applicable": default to NaN
#
# Codes 90 ("Other") and 91 ("Pre-verbal/Non-verbal") are substantive
# answer categories elsewhere in this dataset (not missing data) and are
# intentionally left untouched by this cleaning step.
NOT_APPLICABLE_92_COLS = [
    "LocLen",
    "SeizOccur",
    "SeizLen",
    "HASeverity",
    "HAStart",
    "VomitNbr",
    "VomitStart",
    "VomitLast",
    "AMSAgitated",
    "AMSSleep",
    "AMSSlow",
    "AMSRepeat",
    "AMSOth",
    "SFxPalpDepress",
    "SFxBasHem",
    "SFxBasOto",
    "SFxBasPer",
    "SFxBasRet",
    "SFxBasRhi",
    "HemaLoc",
    "HemaSize",
    "ClavFace",
    "ClavNeck",
    "ClavFro",
    "ClavOcc",
    "ClavPar",
    "ClavTem",
    "NeuroDMotor",
    "NeuroDSensory",
    "NeuroDCranial",
    "NeuroDReflex",
    "NeuroDOth",
    "OSIExtremity",
    "OSICut",
    "OSICspine",
    "OSIFlank",
    "OSIAbdomen",
    "OSIPelvis",
    "OSIOth",
    "IndAge",
    "IndAmnesia",
    "IndAMS",
    "IndClinSFx",
    "IndHA",
    "IndHema",
    "IndLOC",
    "IndMech",
    "IndNeuroD",
    "IndRqstMD",
    "IndRqstParent",
    "IndRqstTrauma",
    "IndSeiz",
    "IndVomit",
    "IndXraySFx",
    "IndOth",
    "CTSed",
    "CTSedAgitate",
    "CTSedAge",
    "CTSedRqst",
    "CTSedOth",
    "EDCT",
    "PosCT",
    "Finding1",
    "Finding2",
    "Finding3",
    "Finding4",
    "Finding5",
    "Finding6",
    "Finding7",
    "Finding8",
    "Finding9",
    "Finding10",
    "Finding11",
    "Finding12",
    "Finding13",
    "Finding14",
    "Finding20",
    "Finding21",
    "Finding22",
    "Finding23",
]


# Columns that are nominal or ordinal with 3+ levels (not plain 0/1
# Yes/No, and not a continuous measurement), per the codebook. These get
# cast to pandas "category" dtype rather than left as raw int/float codes,
# so downstream groupby/plotting code treats them as labels, not numbers
# to average. NaN (from clean_92, or already-missing raw values) is
# preserved as a missing category.
CATEGORICAL_COLS = [
    "EmplType",
    "Certification",
    "InjuryMech",
    "High_impact_InjSev",
    "LOCSeparate",
    "LocLen",
    "SeizOccur",
    "SeizLen",
    "HASeverity",
    "HAStart",
    "VomitNbr",
    "VomitStart",
    "VomitLast",
    "HemaLoc",
    "HemaSize",
    "GCSEye",
    "GCSVerbal",
    "GCSMotor",
    "GCSGroup",
    "Gender",
    "Ethnicity",
    "Race",
    "EDDisposition",
]


def to_categorical(df: pd.DataFrame, cols=CATEGORICAL_COLS) -> pd.DataFrame:
    """Cast `cols` to pandas "category" dtype.

    Intended to run after numeric coercion and sentinel cleanup, so the
    categories are the cleaned integer codes (plus NaN for missing).
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def clean_92(df: pd.DataFrame, fill_value=np.nan) -> pd.DataFrame:
    """Replace the 92 ("Not applicable") sentinel with `fill_value`.

    Only touches columns in NOT_APPLICABLE_92_COLS, where the codebook
    defines 92 as "Not applicable". Other columns, and other sentinel
    codes (90 "Other", 91 "Pre-verbal/Non-verbal"), are left untouched
    since those are substantive answers, not missing data.

    Example: clean_92(df, 0) recodes not-applicable as 0 instead of NaN.
    """
    df = df.copy()
    for col in NOT_APPLICABLE_92_COLS:
        if col in df.columns:
            df[col] = df[col].replace(92, fill_value)
    return df


def clean_data(
    raw_df: pd.DataFrame,
    *,
    drop_missing_target: bool = False,
    not_applicable_value=np.nan,
    categorize: bool = True,
) -> pd.DataFrame:
    """Clean the raw PECARN TBI CSV export.

    `not_applicable_value` controls what the 92 ("Not applicable") sentinel
    is recoded to (default NaN). See clean_92().
    `categorize` controls whether nominal/ordinal columns (CATEGORICAL_COLS)
    are cast to pandas "category" dtype. See to_categorical().
    """
    df = raw_df.copy()

    for col in df.columns:
        if col == "PatNum":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = clean_92(df, fill_value=not_applicable_value)


    binary_like = {
        "Seiz",
        "ActNorm",
        "Vomit",
        "Dizzy",
        "Intubated",
        "Paralyzed",
        "Sedated",
        "AMS",
        "AMSAgitated",
        "AMSSleep",
        "AMSSlow",
        "AMSRepeat",
        "AMSOth",
        "SFxPalp",
        "SFxPalpDepress",
        "FontBulg",
        "SFxBas",
        "SFxBasHem",
        "SFxBasOto",
        "SFxBasPer",
        "SFxBasRet",
        "SFxBasRhi",
        "Hema",
        "Clav",
        "ClavFace",
        "ClavNeck",
        "ClavFro",
        "ClavOcc",
        "ClavPar",
        "ClavTem",
        "NeuroD",
        "NeuroDMotor",
        "NeuroDSensory",
        "NeuroDCranial",
        "NeuroDReflex",
        "NeuroDOth",
        "OSI",
        "OSIExtremity",
        "OSICut",
        "OSICspine",
        "OSIFlank",
        "OSIAbdomen",
        "OSIPelvis",
        "OSIOth",
        "Drugs",
        "CTDone",
        "EDCT",
        "PosCT",
        "Finding1",
        "Finding2",
        "Finding3",
        "Finding4",
        "Finding5",
        "Finding6",
        "Finding7",
        "Finding8",
        "Finding9",
        "Finding10",
        "Finding11",
        "Finding12",
        "Finding13",
        "Finding14",
        "Finding20",
        "Finding21",
        "Finding22",
        "Finding23",
        "DeathTBI",
        "HospHead",
        "HospHeadPosCT",
        "Intub24Head",
        "Neurosurgery",
        "PosIntFinal",
    }
    allowed = {0, 1} if not_applicable_value in (0, 1) else {0, 1, not_applicable_value}
    for col in binary_like:
        if col in df.columns:
            df[col] = df[col].where(df[col].isin(list(allowed)) | df[col].isna(), np.nan)

    if categorize:
        df = to_categorical(df)

    if drop_missing_target and "PosIntFinal" in df.columns:
        df = df.dropna(subset=["PosIntFinal"]).copy()

    return df
