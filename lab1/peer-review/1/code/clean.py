"""
Exploratory diagnostics + cleaning entrypoint for PECARN TBI (train only).

Figures are written to lab1/report/exploratory_figures/.
Do not load or transform the test split here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from subfunctions.dizzy_structural import recode_dizzy_structural
from subfunctions.drop_deterministic_children import drop_deterministic_children
from subfunctions.drop_nonpredictors import drop_nonpredictors
from subfunctions.drop_sfxpalp_depress import drop_sfxpalp_depress
from subfunctions.ethnicity_missing import add_ethnicity_missing
from subfunctions.handle_preverbal import add_preverbal_flag
from subfunctions.onehot_code91 import onehot_code91_columns
from subfunctions.protect_age_sentinels import protect_age_sentinels

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
LAB_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_DIR / "data"
TRAIN_PATH = DATA_DIR / "splits" / "train.csv"
FIG_DIR = LAB_DIR / "report" / "exploratory_figures"

SENTINELS = [91, 92]
ID_COLS = {"PatNum"}
PROTECTED_AGE = {"AgeInMonth", "AgeinYears"}
OUTCOMES = [
    "PosIntFinal",
    "DeathTBI",
    "HospHead",
    "Intub24Head",
    "Neurosurgery",
    "PosCT",
    "Finding1",
]
DROP_FROM_AVAIL_HM = {
    "DeathTBI",
    "HospHead",
    "HospHeadPosCT",
    "Intub24Head",
    "Neurosurgery",
    "PosIntFinal",
    "EmplType",
    "Certification",
    "EDDisposition",
    "CTDone",
    "CTForm1",
}


def load_train(path: Path | None = None) -> pd.DataFrame:
    """Load the frozen train split. Never the test set."""
    return pd.read_csv(path or TRAIN_PATH, low_memory=False)


def ensure_fig_dir(fig_dir: Path | None = None) -> Path:
    out = fig_dir or FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save(fig: plt.Figure, name: str, fig_dir: Path | None = None) -> Path:
    out = ensure_fig_dir(fig_dir) / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    return out


# ---------------------------------------------------------------------------
# A / B — sentinel audit + gating structure
# ---------------------------------------------------------------------------
def gate_off(s: pd.Series) -> pd.Series:
    """Parent is off when answered no, preverbal, N/A, or blank."""
    return s.isin([0, 91, 92]) | s.isna()


def classify_sentinel_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (coded_cols, real_valued) — AgeInMonth lands in real_valued."""
    coded_cols, real_valued = [], []
    for c in df.columns:
        if c in ID_COLS or not df[c].isin(SENTINELS).any():
            continue
        non_sent = df.loc[~df[c].isin(SENTINELS), c].dropna()
        (coded_cols if non_sent.nunique() <= 12 else real_valued).append(c)
    return coded_cols, real_valued


def sentinel_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    coded_cols, real_valued = classify_sentinel_columns(df)
    audit = pd.DataFrame(
        {
            "pct_91": (df[coded_cols].eq(91).mean() * 100).round(1),
            "pct_92": (df[coded_cols].eq(92).mean() * 100).round(1),
            "pct_blank": (df[coded_cols].isna().mean() * 100).round(1),
            "real_levels": [
                df.loc[~df[c].isin(SENTINELS), c].nunique() for c in coded_cols
            ],
        }
    ).sort_values(["pct_92", "pct_91"], ascending=False)
    return audit, coded_cols, real_valued


def find_gates(df: pd.DataFrame, coded_cols: list[str] | None = None) -> pd.DataFrame:
    """For each 92-coded column, find whether a parent perfectly predicts the 92s."""
    if coded_cols is None:
        coded_cols, _ = classify_sentinel_columns(df)
    candidates = [c for c in df.columns if c not in ID_COLS]
    off = pd.DataFrame({p: gate_off(df[p]) for p in candidates})

    rows = []
    for c in coded_cols:
        target = df[c].eq(92)
        if not target.any():
            continue
        agree = off.eq(target, axis=0).mean().drop(c, errors="ignore")
        perfect = list(agree.index[agree == 1.0])
        rows.append(
            {
                "child": c,
                "pct_92": round(target.mean() * 100, 1),
                "parent_or_closest": ", ".join(perfect) if perfect else agree.idxmax(),
                "agreement": round(agree.max(), 4),
                "deterministic": bool(perfect),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["deterministic", "agreement"], ascending=False
    )


def core_columns(df: pd.DataFrame, gates: pd.DataFrame | None = None) -> list[str]:
    """Informative columns = all columns minus deterministic gated children."""
    if gates is None:
        gates = find_gates(df)
    redundant = set(gates.loc[gates.deterministic, "child"])
    return [c for c in df.columns if c not in ID_COLS and c not in redundant]


# ---------------------------------------------------------------------------
# cohorts
# ---------------------------------------------------------------------------
def build_cohorts(df: pd.DataFrame) -> dict:
    """
    Labelled rows + verbal / preverbal split.
    Preverbal = 91 on either HA_verb or Amnesia_verb.
    """
    df_labelled = df.dropna(subset=["PosIntFinal"]).copy()
    preverbal = df_labelled.HA_verb.eq(91) | df_labelled.Amnesia_verb.eq(91)
    return {
        "df_labelled": df_labelled,
        "preverbal": preverbal,
        "df_separated_nonverbal": df_labelled[preverbal].copy(),
        "df_separated_verbal": df_labelled[~preverbal].copy(),
        "label": preverbal.map({True: "preverbal", False: "verbal"}),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_missingness_heatmap(
    df: pd.DataFrame, fig_dir: Path | None = None, show: bool = False
) -> Path:
    """Original two-panel heatmap: % blank | % 91/92 by CTDone and age band."""
    feature_cols = [c for c in df.columns if c not in {"PatNum", "CTDone", "AgeTwoPlus"}]
    groups = {
        "No CT": df["CTDone"] == 0,
        "CT": df["CTDone"] == 1,
        "Age < 2": df["AgeTwoPlus"] == 1,
        "Age ≥ 2": df["AgeTwoPlus"] == 2,
    }
    blank_rate = pd.DataFrame(
        {name: df.loc[mask, feature_cols].isna().mean() * 100 for name, mask in groups.items()}
    )
    sentinel_rate = pd.DataFrame(
        {
            name: df.loc[mask, feature_cols].isin([91, 92]).mean() * 100
            for name, mask in groups.items()
        }
    )

    keep = (blank_rate.max(axis=1) > 0) | (sentinel_rate.max(axis=1) > 0)
    blank_rate = blank_rate.loc[keep]
    sentinel_rate = sentinel_rate.loc[keep]
    order = (
        blank_rate.mean(axis=1)
        .to_frame("blank")
        .join(sentinel_rate.mean(axis=1).rename("sentinel"))
        .sort_values(["blank", "sentinel"], ascending=False)
        .index
    )
    blank_rate = blank_rate.loc[order].T
    sentinel_rate = sentinel_rate.loc[order].T

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(12, 0.35 * len(order)), 6),
        sharex=True,
        constrained_layout=True,
    )
    heatmap_kw = dict(
        cmap="flare",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        annot_kws={"size": 5},
        linewidths=0.3,
        linecolor="white",
    )
    sns.heatmap(blank_rate, ax=axes[0], cbar_kws={"label": "% blank"}, **heatmap_kw)
    axes[0].set_title("% blank (true empty / NaN)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Group")

    sns.heatmap(sentinel_rate, ax=axes[1], cbar_kws={"label": "% 91 or 92"}, **heatmap_kw)
    axes[1].set_title("% coded 91 or 92 (preverbal / not applicable)")
    axes[1].set_xlabel("Variable")
    axes[1].set_ylabel("Group")
    axes[1].tick_params(axis="x", rotation=90)
    fig.suptitle("Missingness-like codes by group and variable (CTDone, age band)")

    path = _save(fig, "01_missingness_blank_vs_sentinel.pdf", fig_dir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_gating_tree(
    df: pd.DataFrame,
    gates: pd.DataFrame | None = None,
    fig_dir: Path | None = None,
    show: bool = False,
) -> Path:
    """Skip-logic tree: parent questions with follow-up columns hanging off them."""
    if gates is None:
        gates = find_gates(df)

    det = gates[gates.deterministic]
    groups = det.groupby("parent_or_closest")["child"].apply(list).to_dict()
    gate_pct = det.groupby("parent_or_closest")["pct_92"].first().to_dict()
    off = pd.DataFrame(
        {p: gate_off(df[p]) for p in df.columns if p not in ID_COLS}
    )

    all_children = set(det.child)
    roots = sorted(
        [p for p in groups if p not in all_children],
        key=lambda p: gate_pct[p],
        reverse=True,
    )

    pos, drawn = {}, []
    cursor = [0.0]

    def layout(node, depth):
        kids = groups.get(node)
        if not kids:
            y = cursor[0]
            cursor[0] += 1
            pos[node] = (depth, y)
            drawn.append(node)
            return y
        ys = [
            layout(k, depth + 1)
            for k in sorted(kids, key=lambda c: (c in groups, c))
        ]
        pos[node] = (depth, (min(ys) + max(ys)) / 2)
        drawn.append(node)
        return pos[node][1]

    for r in roots:
        layout(r, 0)
        cursor[0] += 1.9

    n_rows = cursor[0]
    X = {0: 0.02, 1: 0.335, 2: 0.70}
    fig_h = 0.17 * n_rows + 1.5
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    cmap = plt.get_cmap("YlOrRd")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(n_rows - 1.9, -1.9)
    ax.axis("off")

    box_art = {}
    for node in drawn:
        d, y = pos[node]
        if node in groups:
            n_on = int((~off[node]).sum())
            box_art[node] = ax.text(
                X[d],
                y,
                f"{node}\noff {gate_pct[node]:.1f}%  ·  asked in {n_on:,}",
                fontsize=8.8,
                fontweight="bold",
                va="center",
                ha="left",
                linespacing=1.5,
                bbox=dict(
                    boxstyle="round,pad=0.34",
                    fc=cmap(gate_pct[node] / 135),
                    ec="#7f2704",
                    lw=0.7,
                ),
                zorder=3,
            )
        else:
            ax.text(
                X[d], y, node, fontsize=8.6, va="center", ha="left", color="#222", zorder=3
            )

    fig.canvas.draw()
    inv = ax.transData.inverted()
    for parent, kids in groups.items():
        _, py = pos[parent]
        bb = box_art[parent].get_bbox_patch().get_window_extent()
        x0 = inv.transform((bb.x1, bb.y0))[0] + 0.008
        xm = x0 + 0.028
        for k in kids:
            cx, cy = pos[k]
            ax.add_line(Line2D([x0, xm], [py, py], color="#b0b0b0", lw=0.9, zorder=1))
            ax.add_line(Line2D([xm, xm], [py, cy], color="#b0b0b0", lw=0.9, zorder=1))
            ax.add_line(
                Line2D([xm, X[cx] - 0.008], [cy, cy], color="#b0b0b0", lw=0.9, zorder=1)
            )

    fig.text(
        0.02,
        1 - 0.28 / fig_h,
        f"The survey's skip logic as a tree: {len(roots)} root questions gate "
        f"{sum(len(v) for v in groups.values())} follow-up columns",
        fontsize=12.5,
        ha="left",
        va="center",
    )
    fig.text(
        0.02,
        1 - 0.58 / fig_h,
        "Boxed = a gating question. Plain text = a follow-up forced to 92 whenever its parent is off,\n"
        "so it carries no information the parent doesn't already have. CTSed is both, and nests a level deeper.",
        fontsize=8.4,
        ha="left",
        va="center",
        color="#444",
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.0, right=1.0, top=1 - 0.95 / fig_h, bottom=0.012)

    path = _save(fig, "02_gating_tree.pdf", fig_dir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_gated_block_content(
    df: pd.DataFrame,
    gates: pd.DataFrame | None = None,
    fig_dir: Path | None = None,
    show: bool = False,
) -> Path:
    """Stacked bar: how much of each gated block is forced 92 vs real answers."""
    if gates is None:
        gates = find_gates(df)
    det = gates[gates.deterministic]
    groups = det.groupby("parent_or_closest")["child"].apply(list).to_dict()
    gate_pct = det.groupby("parent_or_closest")["pct_92"].first().to_dict()
    order = sorted(groups, key=lambda p: gate_pct[p], reverse=True)

    comp = pd.DataFrame(
        [
            {
                "parent": f"{p}  ({len(groups[p])})",
                "gated off (92)": df[groups[p]].eq(92).sum().sum(),
                "answered 0 (sign absent)": df[groups[p]].eq(0).sum().sum(),
                "answered 1-4 (real finding)": df[groups[p]]
                .isin([1, 2, 3, 4])
                .sum()
                .sum(),
                "truly blank": df[groups[p]].isna().sum().sum(),
            }
            for p in order
        ]
    ).set_index("parent")
    pct_cells = comp.div(comp.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 5.2))
    pct_cells.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        width=0.75,
        color=["#bdbdbd", "#9ecae1", "#d1495b", "#ffe066"],
        edgecolor="white",
        lw=0.6,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_ylabel("")
    ax.set_xlabel("% of all cells across that parent's child columns", fontsize=9)
    ax.set_title(
        "Most cells in the gated columns hold no information\n"
        "grey = forced to 92 by the parent being off",
        fontsize=11.5,
        loc="left",
    )
    ax.legend(fontsize=8.2, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    path = _save(fig, "03_gated_block_information.pdf", fig_dir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_availability_heatmap(
    df: pd.DataFrame,
    gates: pd.DataFrame | None = None,
    cohorts: dict | None = None,
    fig_dir: Path | None = None,
    show: bool = False,
) -> Path:
    """
    Post-diagnostics availability: informative predictors only.
    Top = blank|91|92; bottom = true blank. Groups = verbal + age + CT.
    """
    if gates is None:
        gates = find_gates(df)
    if cohorts is None:
        cohorts = build_cohorts(df)

    core_cols = core_columns(df, gates)
    hm_cols = [c for c in core_cols if c not in DROP_FROM_AVAIL_HM]
    df_labelled = cohorts["df_labelled"]
    preverbal = cohorts["preverbal"]

    hm_groups = {
        "Preverbal": preverbal,
        "Verbal": ~preverbal,
        "Age < 2": df_labelled["AgeTwoPlus"] == 1,
        "Age ≥ 2": df_labelled["AgeTwoPlus"] == 2,
        "No CT": df_labelled["CTDone"] == 0,
        "CT": df_labelled["CTDone"] == 1,
    }

    def unavailable(series, col):
        blank = series.isna()
        if col in PROTECTED_AGE:
            return blank
        return blank | series.isin([91, 92])

    unavail_rate = pd.DataFrame(
        {
            name: pd.Series(
                {
                    c: unavailable(df_labelled.loc[mask, c], c).mean() * 100
                    for c in hm_cols
                }
            )
            for name, mask in hm_groups.items()
        }
    )
    blank_rate_now = pd.DataFrame(
        {
            name: df_labelled.loc[mask, hm_cols].isna().mean() * 100
            for name, mask in hm_groups.items()
        }
    )

    keep = (unavail_rate.max(axis=1) > 0.5) | (blank_rate_now.max(axis=1) > 0.5)
    unavail_rate = unavail_rate.loc[keep]
    blank_rate_now = blank_rate_now.loc[keep]
    order = (
        (unavail_rate["Preverbal"] - unavail_rate["Verbal"])
        .abs()
        .to_frame("gap")
        .join(unavail_rate.mean(axis=1).rename("mean"))
        .sort_values(["gap", "mean"], ascending=False)
        .index
    )
    unavail_rate = unavail_rate.loc[order].T
    blank_rate_now = blank_rate_now.loc[order].T

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(11, 0.38 * len(order)), 5.8),
        sharex=True,
        constrained_layout=True,
    )
    heatmap_kw = dict(
        cmap="flare",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        annot_kws={"size": 7},
        linewidths=0.3,
        linecolor="white",
    )
    sns.heatmap(unavail_rate, ax=axes[0], cbar_kws={"label": "% unavailable"}, **heatmap_kw)
    axes[0].set_title(
        "% unavailable = blank OR 91 (preverbal) OR 92 (N/A)  —  AgeInMonth protected"
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Group")

    sns.heatmap(blank_rate_now, ax=axes[1], cbar_kws={"label": "% blank"}, **heatmap_kw)
    axes[1].set_title(
        "% true blank (NaN only) — same columns; 91/92 no longer inflate the picture"
    )
    axes[1].set_xlabel("Informative predictor")
    axes[1].set_ylabel("Group")
    axes[1].tick_params(axis="x", rotation=90)
    fig.suptitle(
        f"Data availability after the diagnostics\n"
        f"{len(order)} informative predictors · groups = verbal split + age band + CT",
        fontsize=12,
    )

    path = _save(fig, "04_availability_after_diagnostics.pdf", fig_dir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# cleaning pipeline (train only; extend step-by-step)
# ---------------------------------------------------------------------------
def clean_data(
    df: pd.DataFrame | None = None,
    gates: pd.DataFrame | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply cleaning steps to the train split (or a provided frame).

    Current steps:
      1. Drop deterministic gated children (keep parents).
      2. Protect AgeInMonth / AgeinYears (91/92 are ages, never codes).
      3. Add `preverbal` flag; leave 91 on HA_verb / Amnesia_verb as a category.
      4. Recode preverbal Dizzy blanks to structural 91 (+ Dizzy_structural).
      5. Add Ethnicity_missing; leave Ethnicity NaNs un-imputed.
      6. Drop SFxPalpDepress (~99% 92); add SFxPalp_unclear from SFxPalp==2.

    Admin / component-outcome columns are left in for EDA. Call
    `modeling_frame()` before fitting models to strip them and one-hot
    coded categoricals (91s, Race 90, LOCSeparate/SFxPalp 2, etc.).
    """
    if df is None:
        df = load_train()
    if gates is None:
        gates = find_gates(df)

    df, _dropped = drop_deterministic_children(df, gates, verbose=verbose)
    df = protect_age_sentinels(df, verbose=verbose)
    df = add_preverbal_flag(df, verbose=verbose)
    df = recode_dizzy_structural(df, verbose=verbose)
    df = add_ethnicity_missing(df, verbose=verbose)
    df = drop_sfxpalp_depress(df, add_unclear=True, verbose=verbose)
    return df


def modeling_frame(
    df: pd.DataFrame | None = None,
    gates: pd.DataFrame | None = None,
    verbose: bool = True,
    drop_ct_decision: bool = True,
) -> pd.DataFrame:
    """
    clean_data() → drop non-predictors → one-hot coded categoricals
    (structural 91s + multi-level / sentinel-style codes like Race 90).

    drop_ct_decision: also drop CTDone / CTForm1 (clinician decision leakage
    if the scientific question is who needs a CT). Default True.
    """
    needs_clean = (
        df is None
        or "preverbal" not in df.columns
        or "Ethnicity_missing" not in df.columns
    )
    if needs_clean:
        df = clean_data(df, gates=gates, verbose=verbose)
    elif "SFxPalpDepress" in df.columns:
        df = drop_sfxpalp_depress(df, add_unclear=True, verbose=verbose)

    out, _ = drop_nonpredictors(df, verbose=verbose)
    if drop_ct_decision:
        ct_cols = [c for c in ("CTDone", "CTForm1") if c in out.columns]
        if ct_cols:
            out = out.drop(columns=ct_cols)
            if verbose:
                print(f"modeling_frame: dropped CT decision cols {ct_cols}")
    # AgeinYears redundant with AgeInMonth
    if "AgeinYears" in out.columns:
        out = out.drop(columns=["AgeinYears"])
        if verbose:
            print("modeling_frame: dropped AgeinYears (keep AgeInMonth)")
    out = onehot_code91_columns(out, verbose=verbose)
    return out


# ---------------------------------------------------------------------------
# run everything
# ---------------------------------------------------------------------------
def run_exploratory(
    df: pd.DataFrame | None = None,
    fig_dir: Path | None = None,
    show: bool = False,
) -> dict:
    """
    Reproduce the exploratory notebook diagnostics and write figures under
    report/exploratory_figures/.
    """
    if df is None:
        df = load_train()
    fig_dir = ensure_fig_dir(fig_dir)

    paths = {}
    paths["missingness"] = plot_missingness_heatmap(df, fig_dir=fig_dir, show=show)

    audit, coded_cols, real_valued = sentinel_audit(df)
    gates = find_gates(df, coded_cols)
    paths["gating_tree"] = plot_gating_tree(df, gates=gates, fig_dir=fig_dir, show=show)
    paths["gated_content"] = plot_gated_block_content(
        df, gates=gates, fig_dir=fig_dir, show=show
    )

    cohorts = build_cohorts(df)
    paths["availability"] = plot_availability_heatmap(
        df, gates=gates, cohorts=cohorts, fig_dir=fig_dir, show=show
    )

    return {
        "df": df,
        "audit": audit,
        "coded_cols": coded_cols,
        "real_valued": real_valued,
        "gates": gates,
        "core_cols": core_columns(df, gates),
        "cohorts": cohorts,
        "figure_paths": paths,
        "fig_dir": fig_dir,
    }


if __name__ == "__main__":
    result = run_exploratory(show=False)
    print(f"\nfigures in {result['fig_dir']}")
    for k, p in result["figure_paths"].items():
        print(f"  {k}: {p.name}")

    print("\n--- clean_data / modeling_frame ---")
    raw = load_train()
    cleaned = clean_data(raw, gates=result["gates"], verbose=True)
    assert "SFxBasHem" not in cleaned.columns
    assert "preverbal" in cleaned.columns
    assert "Dizzy_structural" in cleaned.columns
    assert "Ethnicity_missing" in cleaned.columns
    assert "SFxPalpDepress" not in cleaned.columns
    assert "SFxPalp_unclear" in cleaned.columns
    assert "EmplType" in cleaned.columns  # still in EDA frame
    assert cleaned["AgeInMonth"].equals(raw["AgeInMonth"])
    model_df = modeling_frame(cleaned, verbose=True)
    assert "EmplType" not in model_df.columns
    assert "DeathTBI" not in model_df.columns
    assert "CTDone" not in model_df.columns and "CTForm1" not in model_df.columns
    assert "AgeinYears" not in model_df.columns
    assert "HA_verb" not in model_df.columns  # one-hot expanded
    assert any(c.startswith("HA_verb_") for c in model_df.columns)
    assert "Race" not in model_df.columns
    assert any(c.startswith("Race_") for c in model_df.columns)
    assert "LOCSeparate" not in model_df.columns
    assert "SFxPalp_unclear" not in model_df.columns  # redundant with SFxPalp_2
    assert "PosIntFinal" in model_df.columns
    print(f"clean_data OK: {raw.shape} -> {cleaned.shape}; modeling_frame -> {model_df.shape}")
