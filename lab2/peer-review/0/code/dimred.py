"""Dimension reduction -> task 3

What this script does:
  1. one-hot encode the 67 answers (n x 468 0/1 matrix, see clean.one_hot)
  2. PCA on the centered binary matrix (our main choice) and, for comparison, on
  the centered and standardized matrix
  3. look at the individuals in PC1 vs. PC2 space, then "change the projection":
     average each PC per 1 x 1 degree grid cell and draw it on a map
  4. a second technique: PCA on location-aggregated data (the
     mean binary vector of each grid cell), which averages away individual noise
  5. show why PCA on the raw answer codes of lingData is meaningless by
     relabelling the codes at random and redoing that PCA.

Outputs:
  figs/dr_pca_blob.pdf, figs/dr_pca_maps.pdf, figs/dr_scaling.pdf
  report/values/dr.tex: (LaTeX macros, prefix "dr")
  data/pca_scores.csv: (first 10 centered-PCA scores per respondent).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clean import (DATA_DIR, FIGS_DIR, LAT_RANGE, LONG_RANGE, SEED, load_clean, load_questions, load_states, one_hot, question_columns, write_values)

N_KEEP = 10   # number of PC scores saved to data/pca_scores.csv
MIN_CELL = 10    # a grid cell needs this many respondents to be shown
N_SCATTER = 5000  # respondents shown in the individual-level scatter
RARE = 0.01  # an answer chosen by < 1% of respondents is "rare"
N_TOP = 10  # loadings looked at per PC when judging rarity
N_PERM = 20  # random relabellings in the raw-code experiment
MANY_OPTIONS = 10.  # a question with >= 10 answer options has "many"

# Okabe-Ito colours (colour-blind safe) for the 4 regions
REGION_COLORS = {"Northeast": "#0072B2", "Midwest": "#009E73", "South": "#D55E00", "West": "#CC79A7"}
CENTERED_COLOR, SCALED_COLOR = "#0072B2", "#E69F00"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})

# - -- helpers
def pca(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA of an already centered matrix

    Uses the eigendecomposition of the p x p covariance matrix. Returns the eigenvalues
    (variances of the PCs, largest first) and the loadings
    """
    cov = x.T @ x / (len(x) - 1)
    eigval, eigvec = np.linalg.eigh(cov)
    return eigval[::-1], eigvec[:, ::-1]


def orient(loadings: np.ndarray, scores: np.ndarray, ref: list[np.ndarray]) -> None:
    """Flip PC signs in place so PC k correlates positively with ref[k]. Since
    the sign of a PC is arbitrary, I fix it so that high PC1 = east and high PC2 = north,
    -> makes it easier to read
    """
    for k, r in enumerate(ref):
        if np.corrcoef(scores[:, k], r)[0, 1] < 0:
            loadings[:, k] *= -1
            scores[:, k] *= -1


def n_for(frac_var: np.ndarray, level: float) -> int:
    # Number of PCs needed to explain "level" of the total variance
    return int(np.searchsorted(np.cumsum(frac_var), level)+ 1)


def option_label(name: str, questions: clean.Questions, width: int = 28) -> str:
    # Turn a one-hot column name such as Q105_1 into its answer text
    q, j = name.split("_")
    return questions.answers[q][int(j) - 1][:width]

def answer_label(col: str, answers: dict[int, pd.DataFrame]) -> str:
    # Turn a binary column name like "Q105_1" into "soda (Q105)"
    q, j = col.split("_")
    text = answers[int(q[1:])]["ans"].iloc[int(j) - 1].strip()
    return f"{text[:45]} (Q{q[1:]})"


def grid_cells(df: pd.DataFrame) -> pd.Series:
    # Assign each respondent to a 1 x 1 degree cell
    lat0 = np.floor(df["lat"]).astype(int)
    long0 = np.floor(df["long"]).astype(int)
    return pd.Series(list(zip(lat0, long0)), index=df.index)


def cell_means(values: np.ndarray, cells: pd.Series) -> pd.DataFrame:
    # Average the columns of "values" within each grid cell.
    # Only cells with at least MIN_CELL respondents are kept -> a single person can't color a whole square
    out = pd.DataFrame(values).groupby(cells.to_numpy()).mean()
    size = cells.value_counts()
    out["n"] = size.reindex(out.index).to_numpy()
    return out[out["n"] >= MIN_CELL]


def between_cell_share(score: np.ndarray, cells: pd.Series) -> float:
    # Share of a score's variance explained by grid cell -> eta squared
    s = pd.Series(score)
    cell_mean = s.groupby(cells.to_numpy()).transform("mean")
    return float(((cell_mean - s.mean()) ** 2).sum()/ ((s - s.mean()) ** 2).sum())


def between_cell_null(score: np.ndarray, cells: pd.Series, rng: np.random.Generator) -> float:
    """Same share after shuffling people across cells.
    Small cells alone make the share a bit larger than zero -> this is the baseline to compare between_cell_share against
    """
    shuffled = pd.Series(rng.permutation(cells.to_numpy()))
    return between_cell_share(score, shuffled)


def region_of(df: pd.DataFrame) -> pd.Series:
    """Census regions from the STATE column

    Uses the region column of the Natural Earth shapefile.
    Malformed state codes (a few hundred respondents) get NaN.
    """
    states = load_states()
    lookup = dict(zip(states["postal"], states["region"]))
    return df["STATE"].map(lookup)


# - experiments
def raw_code_experiment(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """PCA on the raw answer codes before and after random relabelling.
    Answer codes are arbitrary labels: calling "soda" 3 instead of 1 changes
    nothing about the data. If PCA on the codes were meaningful, the result
    would not depend on the labelling. I relabel every question's codes
    with a random permutation (0 = no answer stays 0) and measure how well
    the new PC1 scores agree with the original ones. For the binary
    encoding, relabelling only reorders columns, so PCA is unchanged.
    """
    q = question_columns(df)
    codes= df[q].to_numpy(dtype=float)
    def pc1_scores(c: np.ndarray) -> np.ndarray:
        c = c - c.mean(axis=0)
        _, v = pca(c)
        return c @ v[:, 0]

    base = pc1_scores(codes)
    k = codes.max(axis=0).astype(int)
    corrs = []
    for _ in range(N_PERM):
        new = codes.copy()
        for j in range(len(q)):
            perm = np.concatenate([[0],rng.permutation(k[j]) + 1])
            new[:, j] = perm[codes[:, j].astype(int)]
        corrs.append(abs(np.corrcoef(base, pc1_scores(new))[0, 1]))

    var = codes.var(axis=0)
    many = k >= MANY_OPTIONS
    return {"perm_corr_median": float(np.median(corrs)),
            "perm_corr_min": float(np.min(corrs)),
            "corr_var_k": float(np.corrcoef(k, var)[0, 1]),
            "share_var_many": float(var[many].sum() / var.sum()),
            "n_many": int(many.sum())}


def top_rarity(loadings: np.ndarray, freq: np.ndarray, n_pc: int) -> np.ndarray:
    # Median frequency of the n-top largest-loading answers of each PC
    out = []
    for k in range(n_pc):
        top = np.argsort(-np.abs(loadings[:, k]))[:N_TOP]
        out.append(np.median(freq[top]))
    return np.array(out)


# - figures
def draw_map(ax, cells: pd.DataFrame, col, states, vmax: float, cmap: str):
    # Color each 1 x 1 degree cell by "col" on top of state borders
    lats = np.arange(int(LAT_RANGE[0]), int(LAT_RANGE[1]) + 1)
    longs = np.arange(int(LONG_RANGE[0]), int(LONG_RANGE[1]) + 1)
    grid = np.full((len(lats) - 1, len(longs) - 1), np.nan)
    for (la, lo), val in cells[col].items(): grid[la - lats[0], lo - longs[0]] = val
    states.plot(ax=ax, color="#f0f0f0", edgecolor="none")
    mesh = ax.pcolormesh(longs, lats, grid, cmap=cmap, vmin=-vmax, vmax=vmax, shading="flat")
    states.boundary.plot(ax=ax, color="#555555", linewidth=0.3)
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24.5, 49.5)
    ax.set_aspect(1 / np.cos(np.deg2rad(38)))
    ax.set_axis_off()
    return mesh


def fig_blob(scores, region, cell_df, cell_region, rng) -> None:
    # Individuals in PC1-PC2 space vs grid-cell averages
    fig, axes = plt.subplots(1, 2, figsize=(5.85, 2.84))
    idx = rng.choice(np.flatnonzero(region.notna()), N_SCATTER, replace=False)
    order = rng.permutation(idx)
    ax = axes[0]
    ax.scatter(scores[order, 0], scores[order, 1], s=3, alpha=0.45, lw=0, c=region.iloc[order].map(REGION_COLORS))
    ax.set_title(f"(a) {N_SCATTER:,} individual respondents")
    ax = axes[1]
    for name, color in REGION_COLORS.items():
        m = cell_region == name
        ax.scatter(cell_df.loc[m, 0], cell_df.loc[m, 1], s=9, c=color, alpha=0.85, lw=0, label=name)
    ax.set_title("(b) averages per 1°×1° cell")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               ncol=4, frameon=False, markerscale=1.8, handletextpad=0.2,
               title="census region of respondent / cell (majority)", title_fontsize=7)
    for ax in axes:
        ax.set_xlabel("PC1 score  (Midwest-like  →  Northeast-like)")
        ax.set_ylabel("PC2 score  (South-like  →  North-like)")
        ax.axhline(0, color="grey", lw=0.4)
        ax.axvline(0, color="grey", lw=0.4)
    # same limits within each panel would hide the cells -> note it instead
    axes[1].text(0.99, 0.02, "note: axes zoomed in vs. (a)",
                 transform=axes[1].transAxes, ha="right", fontsize=7, color="grey")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGS_DIR / "dr_pca_blob.pdf")
    plt.close(fig)


def fig_maps(cell_df, states, ve) -> None:
    # Maps of the grid-cell average of PC1, PC2 and PC3
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
    vmax = np.nanquantile(np.abs(cell_df[[0, 1, 2]].to_numpy()), 0.98)
    titles = ["PC1", "PC2", "PC3"]
    for k, ax in enumerate(axes):
        mesh = draw_map(ax, cell_df, k, states, vmax, "PuOr_r")
        ax.set_title(f"{titles[k]} ({100 * ve[k]:.1f}% of variance)")
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", fraction=0.06, pad=0.03, aspect=45, shrink=0.6)
    cbar.set_label("mean PC score of respondents in the 1°×1° cell")
    fig.savefig(FIGS_DIR / "dr_pca_maps.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_scaling(ve_c, ve_s, rare_c, rare_s) -> None:
    # Centered vs centered+standardized PCA
    n = len(rare_c)
    pcs = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    ax.plot(pcs, 100 * ve_c[:n], "o-", ms=3, color=CENTERED_COLOR, label="centered only")
    ax.plot(pcs, 100 * ve_s[:n], "s--", ms=3, color=SCALED_COLOR, label="centered + standardized")
    ax.set_xlabel("principal component")
    ax.set_ylabel("% of total variance")
    ax.set_title("(a) scree plot")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax = axes[1]
    ax.plot(pcs, 100 * rare_c, "o-", ms=3, color=CENTERED_COLOR, label="centered only")
    ax.plot(pcs, 100 * rare_s, "s--", ms=3, color=SCALED_COLOR, label="centered + standardized")
    ax.axhline(100 * RARE, color="grey", lw=0.6, ls=":")
    ax.text(1, 100 * RARE * 0.8, "1% (rare answer)", ha="left", va="top", fontsize=6.5, color="grey")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 1, 10, 100], ["0.1%", "1%", "10%", "100%"])
    ax.set_ylim(0.08, 100)
    ax.set_xlabel("principal component")
    ax.set_ylabel(f"popularity of the {N_TOP} top-loading\nanswers (median % choosing)")
    ax.set_title("(b) what answers is each PC built on?")
    for ax in axes: ax.set_xticks([1, 5, 10, 15, 20])
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "dr_scaling.pdf")
    plt.close(fig)


# - main
def main() -> None:
    # Run all dimension-reduction analyses and save figures + numbers
    rng = np.random.default_rng(SEED)
    df = load_clean()
    x, names = one_hot(df)
    x = x.astype(np.float64)
    names = np.array(names)
    _, answers = load_questions()
    freq = x.mean(axis=0)

    # - main PCA: centered only 
    xc = x - freq
    eig_c, load_c = pca(xc)
    ve_c = eig_c / eig_c.sum()
    scores = xc @ load_c[:, :N_KEEP]
    orient(load_c, scores, [df["long"].to_numpy(), df["lat"].to_numpy()])

    # - comparison: centered + standardized (= PCA on correlations)
    xs = xc / x.std(axis=0, ddof=1)  # no column is constant after cleaning
    eig_s, load_s = pca(xs)
    ve_s = eig_s / eig_s.sum()
    scores_s = xs @ load_s[:, :N_KEEP]
    orient(load_s, scores_s, [df["long"].to_numpy(), df["lat"].to_numpy()])
    rare_c = top_rarity(load_c, freq, 20)
    rare_s = top_rarity(load_s, freq, 20)

    # -change the projection: average per grid cell
    cells = grid_cells(df)
    cell_df = cell_means(scores[:, :3], cells)
    cell_df_s = cell_means(scores_s[:, :3], cells)
    region = region_of(df)
    cell_region = region.groupby(cells.to_numpy()).agg(
        lambda r: r.mode().iloc[0] if r.notna().any() else np.nan
    ).reindex(cell_df.index)

    # - second: PCA on location-aggregated binary vectors
    agg = cell_means(x, cells)
    xa = agg.drop(columns="n").to_numpy()
    xa = xa - xa.mean(axis=0)
    eig_a, load_a = pca(xa)
    ve_a = eig_a / eig_a.sum()
    agg_scores = xa @ load_a[:, :2]
    # absolute correlation between aggregated PC k and the cell map of individual PC j
    corr_agg = np.array([[abs(np.corrcoef(agg_scores[:, k], cell_df[j])[0, 1]) for j in range(2)] for k in range(2)])
    print("aggregated PCA, |corr| with individual PC maps:\n", corr_agg.round(2))
    for k in range(2):
        order = np.argsort(load_a[:, k])
        print(f"agg PC{k + 1}:", "; ".join(answer_label(c, answers) for c in np.concatenate([names[order[:4]], names[order[::-1][:4]]])))

    # - raw codes experiment 
    raw = raw_code_experiment(df, rng)

    # - print the interpretation of the axes
    for k in range(3):
        order = np.argsort(load_c[:, k])
        print(f"\nPC{k + 1} ({100 * ve_c[k]:.1f}%)")
        print("  negative:", "; ".join(answer_label(c, answers) for c in names[order[:6]]))
        print("  positive:", "; ".join(answer_label(c, answers) for c in names[order[::-1][:6]]))
    for k in range(3):
        c_ind = [np.corrcoef(scores[:, k], df[v])[0, 1] for v in ("long", "lat")]
        print(f"PC{k + 1}: corr(long, lat) = {np.round(c_ind, 2)}, "
              f"between-cell share = {between_cell_share(scores[:, k], cells):.3f}")
    print("raw-code experiment:", raw)

    # - figures 
    FIGS_DIR.mkdir(exist_ok=True)
    states = load_states()
    fig_blob(scores, region, cell_df, cell_region, rng)
    fig_maps(cell_df, states, ve_c)
    fig_scaling(ve_c, ve_s, rare_c, rare_s)

    # - save scores
    out = df[["ID", "lat", "long"]].copy()
    for k in range(N_KEEP):
        out[f"PC{k + 1}"] = scores[:, k]
    out.to_csv(DATA_DIR / "pca_scores.csv", index=False)

    # - numbers for the report
    def pct(v: float) -> str:
        return f"{100 * v:.1f}"

    def r2(v: float) -> str:
        return f"{v:.2f}"

    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    ones = ["One", "Two", "Three"]
    vals = {"drP": x.shape[1],
        "drVarPCOne": pct(ve_c[0]), "drVarPCTwo": pct(ve_c[1]),
        "drVarPCThree": pct(ve_c[2]),
        "drVarTopTen": pct(ve_c[:10].sum()),
        "drNPCsHalf": n_for(ve_c, 0.5), "drNPCsEighty": n_for(ve_c, 0.8),
        "drVarScaledPCOne": pct(ve_s[0]), "drVarScaledPCTwo": pct(ve_s[1]),
        "drVarScaledTopTen": pct(ve_s[:10].sum()),
        "drNPCsHalfScaled": n_for(ve_s, 0.5),
        "drNRare": int((freq < RARE).sum()),
        "drRareTopCentered": pct(np.mean([
            np.mean(freq[np.argsort(-np.abs(load_c[:, k]))[:N_TOP]] < RARE)
            for k in range(20)])),
        "drRareTopScaled": pct(np.mean([
            np.mean(freq[np.argsort(-np.abs(load_s[:, k]))[:N_TOP]] < RARE)
            for k in range(20)])),
        "drCorrPCOneLong": r2(corr(scores[:, 0], df["long"])),
        "drCorrPCTwoLat": r2(corr(scores[:, 1], df["lat"])),
        "drCorrPCThreeLat": r2(corr(scores[:, 2], df["lat"])),
        "drCorrPCThreeLong": r2(corr(scores[:, 2], df["long"])),
        "drCellCorrPCOneLong": r2(corr(cell_df[0], [c[1] for c in cell_df.index])),
        "drCellCorrPCTwoLat": r2(corr(cell_df[1], [c[0] for c in cell_df.index])),
        "drNCells": len(cell_df), "drMinCell": MIN_CELL,
        "drNScatter": N_SCATTER,
        "drAggVarPCOne": pct(ve_a[0]), "drAggVarPCTwo": pct(ve_a[1]),
        "drAggCorrPCOneIndTwo": r2(corr_agg[0, 1]),
        "drAggCorrPCTwoIndOne": r2(corr_agg[1, 0]),
        "drPermCorrMedian": r2(raw["perm_corr_median"]),
        "drPermCorrMin": r2(raw["perm_corr_min"]),
        "drCorrOptionsVar": r2(raw["corr_var_k"]),
        "drShareVarManyOptions": pct(raw["share_var_many"]),
        "drNManyOptions": raw["n_many"], "drManyOptions": MANY_OPTIONS,
        "drNPerm": N_PERM,
    }
    for k in range(3):
        # geography share: how much of the individual PC score is explained by the grid cell a person lives in
        vals[f"drCellSharePC{ones[k]}"] = pct(between_cell_share(scores[:, k], cells))
        vals[f"drCellNullPC{ones[k]}"] = pct(between_cell_null(scores[:, k], cells, rng))
        # agreement of centered vs scaled PCA
        vals[f"drScaledCorrPC{ones[k]}"] = r2(abs(corr(scores[:, k], scores_s[:, k])))
        vals[f"drScaledMapCorrPC{ones[k]}"] = r2(abs(corr(cell_df[k], cell_df_s.loc[cell_df.index, k])))
    write_values("dr", vals)
    print("\n", vals)

if __name__ == "__main__":
    main()
