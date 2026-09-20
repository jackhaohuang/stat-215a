"""
Patient-level 80/20 train/test split — run BEFORE cleaning.

- One row per PatNum in this dataset, but we still split by patient ID
  so the procedure stays correct if duplicates ever appear.
- Stratify on PosIntFinal (ciTBI / clinically important outcome).
- The 20 patients with a blank PosIntFinal are held out of both sets
  (cannot train or evaluate on an unknown label).
- Fixed random_state so the partition is reproducible.
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "TBI PUD 10-08-2013.csv"
OUT_DIR = DATA_DIR / "splits"
TRAIN_PATH = OUT_DIR / "train.csv"
TEST_PATH = OUT_DIR / "test.csv"
MANIFEST_PATH = OUT_DIR / "split_manifest.json"

TEST_SIZE = 0.20
RANDOM_STATE = 215  # course number, fixed for reproducibility
OUTCOME = "PosIntFinal"
PATIENT_ID = "PatNum"


def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)

    assert df[PATIENT_ID].is_unique, "expected one row per patient; use a group split if that changes"

    labelled = df.dropna(subset=[OUTCOME]).copy()
    unlabelled = df[df[OUTCOME].isna()].copy()

    train, test = train_test_split(
        labelled,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labelled[OUTCOME],
        shuffle=True,
    )

    # safety: no patient in both partitions
    overlap = set(train[PATIENT_ID]) & set(test[PATIENT_ID])
    assert not overlap, f"patient leakage: {overlap}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    # id lists are small enough to commit even when *.csv is gitignored
    (OUT_DIR / "train_patnums.txt").write_text(
        "\n".join(map(str, sorted(train[PATIENT_ID]))) + "\n"
    )
    (OUT_DIR / "test_patnums.txt").write_text(
        "\n".join(map(str, sorted(test[PATIENT_ID]))) + "\n"
    )
    if len(unlabelled):
        (OUT_DIR / "unlabelled_patnums.txt").write_text(
            "\n".join(map(str, sorted(unlabelled[PATIENT_ID]))) + "\n"
        )

    manifest = {
        "raw_path": str(RAW_PATH.name),
        "n_raw": int(len(df)),
        "n_unlabelled_excluded": int(len(unlabelled)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": OUTCOME,
        "patient_id": PATIENT_ID,
        "train_pos_rate": float(train[OUTCOME].mean()),
        "test_pos_rate": float(test[OUTCOME].mean()),
        "train_path": TRAIN_PATH.name,
        "test_path": TEST_PATH.name,
        "note": "Do not load or transform test.csv until final evaluation.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {TRAIN_PATH}")
    print(f"wrote {TEST_PATH}")
    print(f"wrote {MANIFEST_PATH}")
    print("TEST SET IS FROZEN — do not clean, explore, or model on it yet.")


if __name__ == "__main__":
    main()
