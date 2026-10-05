"""EDA of two survey questions and their geography -> task 2

Two questions:
  - Q50 "What word(s) do you use to address a group of two or more people?" (you guys / y'all / you all / ...)
  - Q105 "What is your generic term for a sweetened carbonated beverage?" (soda / pop / coke / ...)

how does each one varies over the map?
Do the answers to the two questions line up into distinct geographical groups?
Does one answer helps to predict the other?
How do these two compare with the other 65 questions?

Association between two categorical variables is measured with the
bias-corrected Cramer's V, which is 0 for independence and 1
for a perfect one-to-one relation

Outputs:
  figs/eda_maps.pdf - majority answer per 1 x 1 degree grid cell
  figs/eda_joint.pdf - P(Q50 answer, Q105 answer), overall and by region
  figs/eda_many.pdf - geography vs pairwise association, top 20 questions
  report/values/eda.tex - all numbers quoted in the report
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402
from sklearn.metrics import mutual_info_score  # noqa: E402

from clean import (FIGS_DIR, SEED, load_clean, load_questions,load_states, question_columns, write_values)  # noqa: E402

# - -
# Settings

Q_A, Q_B = "Q050", "Q105"
# Answer codes kept as their own category and everything else becomes "other"
FOLD_A = {4: "you guys", 9: "y'all", 1: "you all"}
FOLD_B = {1: "soda",2: "pop", 3: "coke"}
OTHER = "other"
CELL_DEG = 1.0  # grid cell size in degrees for the maps
MIN_CELL_N = 10  # grid cells with fewer respondents are left blank
N_TOP = 20  # number of questions in the many-question figure

# Colorblind-safe Okabe-Ito colors -> "other" is a neutral light grey
GREY = "#d4d4d4"
COLORS_A = {"you guys": "#CC79A7", "y'all": "#D55E00", "you all": "#56B4E9", OTHER: GREY}
COLORS_B = {"soda": "#0072B2", "pop": "#E69F00", "coke": "#009E73",OTHER: GREY}
# The three answer pairs that win almost all grid cells -> color = drink color
PAIRS = {"soda + you guys": COLORS_B["soda"],
         "pop + you guys": COLORS_B["pop"],
         "coke + y'all": COLORS_B["coke"],
         "any other pair": GREY}

# Hand-written plot labels for questions whose answers are not words
LABEL_OVERRIDES = {
    56: "positive 'anymore' OK?",
    78: "scratch vs. scrap paper",
    86: "do you say 'cruller'?",
    118: "drive-thru liquor store",
}

# U.S. Census regions (DC counted with the South)
REGIONS = {"Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_TO_REGION = {s: r for r, states in REGIONS.items() for s in states.split()}

# Figures are drawn at their printed size and included at natural size, so the font sizes below are the printed sizes
FIG_WIDTH = 6.5
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


# - -
# Association measures

def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Bias-corrected Cramer's V and chi-square p-value of two categoricals
    Arguments:
    x: First categorical variable
    y: Second categorical variable (same index as x)

    Returns:
        Tuple (v, p) with the bias-corrected Cramer's V and the p-value of Pearson's chi-square test of independence
    """
    table = pd.crosstab(x, y).to_numpy()
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    n = table.sum()
    r, k = table.shape
    if r < 2 or k < 2: return 0.0, 1.0
    chi2, p, _, _ = chi2_contingency(table, correction=False)
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / min(k_corr - 1, r_corr - 1))), float(p)


def uncertainty_coefficient(target: pd.Series, given: pd.Series) -> float:
    """Theil's U: share of the entropy of "target" explained by "given"
    Arguments:
        target: Variable to be predicted
        given: Predictor variable
    Returns: Mutual information divided by the entropy of "target" (0 to 1)
    """
    probs = target.value_counts(normalize=True).to_numpy()
    entropy = -np.sum(probs * np.log(probs))
    return float(mutual_info_score(target, given)/ entropy)


def lookup_accuracy(df: pd.DataFrame, target: str, predictors: list[str], rng: np.random.Generator) -> float:
    """Accuracy of a "most common answer in my group" classifier

    The data are split 50/50 at random. On the training half I record the
    most common "target" answer for every combination of the predictor values.
    On the test half I predict that answer. With an empty
    predictor list this is the baseline "always guess the majority"


    Arguments:
        df: Data containing "target" and "predictors"
        target: Column to predict
        predictors: Columns defining the groups
        rng: Random generator used for the split
    Returns: Test-set accuracy.
    """
    is_train = rng.permutation(len(df)) < len(df) // 2
    train, test = df[is_train],df[~is_train]
    overall = train[target].mode()[0]
    if not predictors: return float((test[target] == overall).mean())
    rule = train.groupby(predictors)[target].agg(lambda s: s.mode()[0])
    keys = pd.MultiIndex.from_frame(test[predictors]) if len(predictors) > 1 \
        else test[predictors[0]]
    pred = rule.reindex(keys).fillna(overall).to_numpy()
    return float((pred == test[target].to_numpy()).mean())


# - -
# Helpers for labels and maps

def fold(codes: pd.Series, mapping: dict[int, str]) -> pd.Series:
    # Map answer codes to a few named categories, the rest to "other"
    return codes.map(mapping).fillna(OTHER)


def contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the contingency table of two categorical arrays (no empty rows)."""
    ia, ib = pd.factorize(a, sort=True)[0], pd.factorize(b, sort=True)[0]
    k = ib.max() + 1
    table = np.bincount(ia * k + ib, minlength=(ia.max() + 1) * k).reshape(-1, k)
    return table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]


def short_label(qnum: int, answers: dict[int, pd.DataFrame], n_words: int = 3) -> str:
    """Build a readable label like "Q105 soda / pop / coke" from the answers

    Uses the most popular answers, skipping "other", "I have no word ..."
    style answers and long sentences. Questions whose answers are not words get a hand-written label instead

    Arguments:
        qnum: Question number.
        answers: Answer tables from "load_questions".
        n_words: Number of most popular answers to show.

    Returns: Short label for plots.
    """
    if qnum in LABEL_OVERRIDES: return f"Q{qnum:03d} {LABEL_OVERRIDES[qnum]}"
    tab = answers[qnum].sort_values("per", ascending=False)
    words = [a.strip() for a in tab["ans"]]
    words = [a for a in words if a.lower() != "other" and len(a) <= 25 and not a.lower().startswith(("i ", "we ", "no,"))]
    return f"Q{qnum:03d} " + " / ".join(words[:n_words])


def grid_majority(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Most common value of "col" in every 1 x 1 degree grid cell

    Arguments:
        df: Data with "lat", "long" and "col"
        col: Categorical column to summarise

    Returns:
        Data frame with the lower-left corner of each cell, the number of
        respondents and the majority value (cells with < Min_cell_n dropped)
    """
    cells = pd.DataFrame({
        "lat0": np.floor(df["lat"] / CELL_DEG) * CELL_DEG,
        "long0": np.floor(df["long"] / CELL_DEG) * CELL_DEG,
        "value": df[col],
    })
    out = cells.groupby(["lat0", "long0"])["value"].agg(n="size", majority=lambda s: s.value_counts().index[0]).reset_index()
    return out[out["n"] >= MIN_CELL_N]


def draw_map(ax: plt.Axes, states,cells: pd.DataFrame, colors: dict[str, str], title: str) -> None:
    """Draw grid cells colored by their majority answer over state borders

    Arguments:
        ax: Axes to draw on
        states: GeoDataFrame of state polygons
        cells: Output of "grid_majority" (majority must be a key of colors)
        colors: Category -> color
        title: Panel title
    """
    for label, color in colors.items():
        sub = cells[cells["majority"] == label]
        for _, row in sub.iterrows():
            ax.add_patch(plt.Rectangle((row["long0"], row["lat0"]), CELL_DEG, CELL_DEG, facecolor=color, edgecolor="white", linewidth=0.3))
    states.boundary.plot(ax=ax, color="#333333", linewidth=0.35)
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24.5, 49.5)
    ax.set_aspect(1 /np.cos(np.deg2rad(38)))
    ax.set_axis_off()
    ax.set_title(title, loc="left")
    handles = [Patch(facecolor=c, edgecolor="none", label=lab) for lab, c in colors.items()]
    ax.legend(handles=handles, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 0.0), ncol=2, handlelength=1.2, columnspacing=1.2)


# - -
# Figures

def figure_maps(df: pd.DataFrame, states) -> None:
    #Figure 1: majority answer maps for Q050, Q105 and the answer pair
    fig = plt.figure(figsize=(5.8, 3.93))
    grid = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.05)
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[1, 1:3])]
    draw_map(axes[0], states, grid_majority(df, "A"), COLORS_A, "(a) Word for a group of people (Q050)")
    draw_map(axes[1], states, grid_majority(df, "B"), COLORS_B, "(b) Word for a soft drink (Q105)")
    pair_cells = grid_majority(df, "pair")
    pair_cells["majority"] = pair_cells["majority"].where(pair_cells["majority"].isin(PAIRS), "any other pair")
    draw_map(axes[2], states, pair_cells, PAIRS,"(c) Most common answer pair (drink + group)")
    fig.savefig(FIGS_DIR / "eda_maps.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_joint(df: pd.DataFrame) -> None:
    # Figure 2: P(Q050 answer, Q105 answer) overall and within regions
    rows = list(COLORS_B) # soda, pop, coke, other
    cols = list(COLORS_A) # you guys, y'all, you all, other
    cond = pd.crosstab(df["B"], df["A"], normalize="index").loc[rows, cols]
    counts = df["B"].value_counts().loc[rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.15, 2.52), gridspec_kw={"width_ratios": [1, 1]})

    # (a) heatmap of the conditional distribution
    ax1.imshow(cond.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = cond.iloc[i, j]
            ax1.text(j, i,f"{100 * val:.0f}%", ha="center", va="center", color="white" if val > 0.5 else "black")
    ax1.set_xticks(range(len(cols)), cols)
    ax1.set_yticks(range(len(rows)),[f"{r} (n={counts[r]:,})" for r in rows])
    ax1.set_xlabel("Word for a group of people (Q050)")
    ax1.set_ylabel("Word for a soft drink (Q105)")
    ax1.set_title("(a) Q050 answer given Q105 answer\n(each row sums to 100%)", loc="left")
    ax1.spines[:].set_visible(False)
    ax1.tick_params(length=0)

    # (b) share saying y'all, by drink word, within each region
    regions = list(REGIONS)
    drinks = ["soda", "pop", "coke"]
    width = 0.26
    x = np.arange(len(regions))
    for i, drink in enumerate(drinks):
        sub = df[df["B"] == drink]
        share = sub.groupby("region")["A"].apply(
            lambda s: (s == "y'all").mean()).reindex(regions)
        ax2.bar(x + (i - 1) * width, 100 * share, width * 0.92, color=COLORS_B[drink], label=f'says "{drink}"')
    overall = df.groupby("region")["A"].apply(
        lambda s: (s == "y'all").mean()).reindex(regions)
    ax2.scatter(x, 100 * overall, marker="_", s=350, color="black", linewidth=1.3, zorder=3, label="everyone in region")
    ax2.set_xticks(x, regions)
    ax2.tick_params(axis="x", labelsize=7)
    ax2.set_ylabel("% who say y'all")
    ax2.set_title("(b) Share saying y'all, by drink word\nwithin each census region", loc="left")
    ax2.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 100)
    handles, labs = ax2.get_legend_handles_labels()
    order = [labs.index(f'says "{d}"') for d in drinks] + \
        [labs.index("everyone in region")]
    ax2.legend([handles[i] for i in order], [labs[i] for i in order],
               frameon=False, loc="upper left", ncol=2, columnspacing=1.0,
               handlelength=1.2)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGS_DIR / "eda_joint.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_many(v_state: pd.Series, v_pair: pd.DataFrame,
                labels: dict[str, str]) -> None:
    # Figure 3: association with state and pairwise V for the top questions
    order = list(v_pair.index)
    n = len(order)
    y = np.arange(n)
    # Fixed axes positions leave room for the long question labels on the left
    fig = plt.figure(figsize=(FIG_WIDTH, 4.2))
    ax1 = fig.add_axes([0.47, 0.13, 0.12, 0.79])
    ax2 = fig.add_axes([0.61, 0.13, 0.30, 0.79], sharey=ax1)
    cax = fig.add_axes([0.925, 0.13, 0.015, 0.79])

    # (a) V with state -> my two questions in blue, the rest in grey
    highlight = [q in (Q_A, Q_B) for q in order]
    ax1.barh(y, v_state.loc[order], height=0.7, color=["#0072B2" if h else "#a6a6a6" for h in highlight])
    ax1.set_yticks(y, [labels[q] for q in order])
    for tick, h in zip(ax1.get_yticklabels(), highlight): tick.set_fontweight("bold" if h else "normal")
    ax1.set_ylim(n - 0.5, -0.5)
    ax1.set_xticks([0, 0.2, 0.4])
    ax1.tick_params(axis="y", labelsize=7)
    ax1.set_xlabel("V with state")
    ax1.set_title("(a) Geography", loc="left")
    ax1.grid(axis="x", color="#e5e5e5", linewidth=0.6)
    ax1.set_axisbelow(True)

    # (b) pairwise V heatmap, rows aligned with (a)
    mat = v_pair.to_numpy().copy()
    np.fill_diagonal(mat, np.nan)
    im = ax2.imshow(mat, cmap="Blues", vmin=0, vmax=np.nanmax(mat), aspect="auto")
    ax2.set_xticks(y, order, rotation=90)
    ax2.tick_params(axis="x", labelsize=7)
    ax2.set_ylim(n - 0.5, -0.5)
    ax2.tick_params(axis="y", labelleft=False)
    ax2.set_title("(b) Link between pairs of questions", loc="left")
    ax2.spines[:].set_visible(False)
    ax2.tick_params(length=0)
    cbar =fig.colorbar(im, cax=cax)
    cbar.set_label("Cramér's V")
    cbar.outline.set_visible(False)
    fig.savefig(FIGS_DIR / "eda_many.pdf", bbox_inches="tight")
    plt.close(fig)

"""def figure_forest(ax, assoc: pd.DataFrame) -> None:
    Draw the risk differences with their intervals as a forest plot

    Arguments:
        ax: Axes to draw on
        assoc: Output of conditional_association

    y = np.arange(len(assoc))[::-1]
    colors = [style.REGION_COLORS[r] if isinstance(r, str) else style.INK
              for r in assoc["region"]]
    est, low, high = (100 * assoc[c].to_numpy() for c in ("estimate", "low", "high"))
    for yi, e, lo, hi, c in zip(y, est, low, high, colors):
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.2)
        ax.plot(e, yi, "o", color=c, ms=4)
    labels = [f"{r} cells ({n:,})" if isinstance(r, str) else lab
              for lab, r, n in zip(assoc["label"], assoc["region"], assoc["n_coke"])]
    ax.set_yticks(y, labels)
    ax.axvline(0, color=style.INK, lw=0.6)
    ax.axhline(len(assoc) - 4.5, color=style.NEUTRAL, lw=0.6, ls=":")
    ax.set_xlim(min(0.0, low.min()) - 2, high.max() + 3)
    ax.set_ylim(-0.6, len(assoc) - 0.4)
    ax.set_xlabel("Difference in share saying y'all,\n"
                  "coke-sayers minus others (points)")
    ax.set_title("(d) Coke and y'all within places")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)"""

# - -
# Main analysis

def main() -> None:
    # Run the EDA, save the three figures and write eda.tex
    rng = np.random.default_rng(SEED)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    df_all = load_clean()
    _, answers = load_questions()
    states = load_states()

    # Pair analysis uses people who answered both questions
    df = df_all[(df_all[Q_A] > 0) & (df_all[Q_B] > 0)].copy()
    df["A"] = fold(df[Q_A],FOLD_A)
    df["B"] = fold(df[Q_B], FOLD_B)
    df["pair"] = df["B"] + " + " + df["A"]
    df["region"] = df["STATE"].map(STATE_TO_REGION)

    # - 1. The two questions against each other 
    v_ab, p_ab = cramers_v(df["A"], df["B"])
    v_ab_raw, _ = cramers_v(df[Q_A], df[Q_B])     
    cond = pd.crosstab(df["B"], df["A"], normalize="index")

    # Does V survive within regions? -> If not, the link is mostly geography
    v_within = {r: cramers_v(g["A"], g["B"])[0] for r, g in df.groupby("region")}

    # - 2. Prediction 
    acc = {name: lookup_accuracy(df, "A", preds, rng) for name, preds in [
        ("base", []), ("drink", ["B"]), ("state", ["STATE"]),
        ("statedrink", ["STATE", "B"])]}
    acc_b = {name: lookup_accuracy(df, "B", preds, rng) for name, preds in [
        ("base", []), ("group", ["A"]), ("state", ["STATE"])]}
    u_a_given_b = uncertainty_coefficient(df["A"], df["B"])
    u_b_given_a = uncertainty_coefficient(df["B"], df["A"])

    # - 3. Geography 
    v_a_state, _ = cramers_v(df["A"], df["STATE"])
    v_b_state, _ = cramers_v(df["B"], df["STATE"])
    v_a_region, _ = cramers_v(df["A"], df["region"])
    v_b_region, _ = cramers_v(df["B"], df["region"])
    v_pair_state, _ = cramers_v(df["pair"], df["STATE"])
    pair_share =df["pair"].value_counts(normalize=True)
    top3_share = pair_share[[p for p in PAIRS if p in pair_share]].sum()
    pair_cells = grid_majority(df, "pair")
    top3_cells = pair_cells["majority"].isin(PAIRS).mean()

    # - 4. Many questions 
    qcols = question_columns(df_all)
    v_state = pd.Series({q: cramers_v(df_all.loc[df_all[q] > 0, q], df_all.loc[df_all[q] > 0, "STATE"])[0]
                         for q in qcols}).sort_values(ascending=False)
    rank = {q: i + 1 for i, q in enumerate(v_state.index)}
    # the N_TOP questions most linked to state, always including my two
    must = [q for q in (Q_A, Q_B) if q not in v_state.index[:N_TOP]]
    top = [q for q in v_state.index if q not in must][:N_TOP - len(must)]
    top = sorted(top + must, key=lambda q: -v_state[q])
    v_pair = pd.DataFrame(1.0, index=top, columns=top)
    for i, qi in enumerate(top):
        for qj in top[i + 1:]:
            ok = (df_all[qi] > 0) & (df_all[qj] > 0)
            v = cramers_v(df_all.loc[ok, qi], df_all.loc[ok, qj])[0]
            v_pair.loc[qi, qj] = v_pair.loc[qj, qi] = v
    off = v_pair.to_numpy()[np.triu_indices(len(top), 1)]
    strongest = v_pair.where(np.triu(np.ones_like(v_pair, bool), 1)).stack()
    strongest = strongest.sort_values(ascending=False)
    labels = {q: short_label(int(q[1:]), answers) for q in qcols}

    # - Figures 
    figure_maps(df, states)
    figure_joint(df)
    figure_many(v_state, v_pair, labels)

    # - Numbers for the report
    def pct(x: float) -> str:
        return f"{100 * x:.0f}"

    def reg_yall(region: str, drink: str) -> float:
        sub = df[(df["region"] == region) & (df["B"] == drink)]
        return (sub["A"] == "y'all").mean()

    values = {
        "edaN": len(df),
        "edaCramerV": f"{v_ab:.2f}",
        "edaCramerVAllOptions": f"{v_ab_raw:.2f}",
        # chi2_contingency returns p = 0.0 when it underflows double precision
        "edaChiPvalue": "$< 10^{-300}$" if p_ab == 0 else f"{p_ab:.1e}",
        "edaVWithinNortheast": f"{v_within['Northeast']:.2f}",
        "edaVWithinMidwest": f"{v_within['Midwest']:.2f}",
        "edaVWithinSouth": f"{v_within['South']:.2f}",
        "edaVWithinWest": f"{v_within['West']:.2f}",
        # conditional distributions
        "edaPctYallGivenCoke": pct(cond.loc["coke", "y'all"]),
        "edaPctYallGivenPop": pct(cond.loc["pop", "y'all"]),
        "edaPctYallGivenSoda": pct(cond.loc["soda", "y'all"]),
        "edaPctGuysGivenPop": pct(cond.loc["pop", "you guys"]),
        "edaPctGuysGivenSoda": pct(cond.loc["soda", "you guys"]),
        "edaPctGuysGivenCoke": pct(cond.loc["coke", "you guys"]),
        "edaPctYallOverall": pct((df["A"] == "y'all").mean()),
        "edaPctYallSouthCoke": pct(reg_yall("South", "coke")),
        "edaPctYallSouthSoda": pct(reg_yall("South", "soda")),
        "edaPctYallSouthPop": pct(reg_yall("South", "pop")),
        "edaPctYallNortheastCoke": pct(reg_yall("Northeast", "coke")),
        "edaNNortheastCoke": int(((df["region"] == "Northeast") & (df["B"] == "coke")).sum()),
        "edaPctYallSouth": pct((df.loc[df["region"] == "South", "A"]== "y'all").mean()),
        # prediction
        "edaAccBaseline": pct(acc["base"]),
        "edaAccFromDrink": pct(acc["drink"]),
        "edaAccFromState": pct(acc["state"]),
        "edaAccFromStateDrink": pct(acc["statedrink"]),
        "edaAccDrinkBaseline": pct(acc_b["base"]),
        "edaAccDrinkFromGroup": pct(acc_b["group"]),
        "edaAccDrinkFromState": pct(acc_b["state"]),
        "edaUGroupGivenDrink": f"{u_a_given_b:.3f}",
        "edaUDrinkGivenGroup": f"{u_b_given_a:.3f}",
        # geography
        "edaVGroupState": f"{v_a_state:.2f}",
        "edaVDrinkState": f"{v_b_state:.2f}",
        "edaVGroupRegion": f"{v_a_region:.2f}",
        "edaVDrinkRegion": f"{v_b_region:.2f}",
        "edaVPairState": f"{v_pair_state:.2f}",
        # same, but using every answer option (as in the many-question figure)
        "edaVGroupStateAllOptions": f"{v_state[Q_A]:.2f}",
        "edaVDrinkStateAllOptions": f"{v_state[Q_B]:.2f}",
        "edaPctTopThreePairs":pct(top3_share),
        "edaPctCellsTopThreePairs": pct(top3_cells),
        "edaNCells": len(pair_cells),
        # many questions
        "edaRankGroupState": rank[Q_A],
        "edaRankDrinkState": rank[Q_B],
        "edaVStateMax": f"{v_state.iloc[0]:.2f}",
        "edaVStateMaxQ": v_state.index[0],
        "edaVStateMedian": f"{v_state.median():.2f}",
        "edaVStateMin": f"{v_state.iloc[-1]:.2f}",
        "edaNTop": N_TOP,
        "edaVPairMedian": f"{np.median(off):.2f}",
        "edaVPairMax": f"{strongest.iloc[0]:.2f}",
        "edaVPairMaxQa": strongest.index[0][0],
        "edaVPairMaxQb": strongest.index[0][1],
    }
    write_values("eda", values)

    # Console summary for the notes
    print(f"n = {len(df)}; V(Q050,Q105) = {v_ab:.3f} (all options "
          f"{v_ab_raw:.3f}), p = {p_ab}")
    print("within-region V:", {k: round(v, 3) for k, v in v_within.items()})
    print("P(A | B):\n", cond.round(3))
    print("accuracy A:", {k:round(v, 3) for k, v in acc.items()})
    print("accuracy B:", {k: round(v, 3) for k, v in acc_b.items()})
    print(f"U(A|B) = {u_a_given_b:.3f}, U(B|A) = {u_b_given_a:.3f}")
    print(f"V state: A {v_a_state:.3f}, B {v_b_state:.3f}, pair "
          f"{v_pair_state:.3f}; region: A {v_a_region:.3f}, B {v_b_region:.3f}")
    print("pair shares:\n", pair_share.head(6).round(3))
    print(f"top-3 pair share {top3_share:.3f}, cells won by top 3 "
          f"{top3_cells:.3f} of {len(pair_cells)}")
    print("V with state (top 20):")
    print(pd.DataFrame({"V": v_state.round(3), "label": [labels[q] for q in v_state.index]}).head(20))
    print("bottom 5:", v_state.tail(5).round(3).to_dict())
    print(f"ranks: {Q_A} {rank[Q_A]}, {Q_B} {rank[Q_B]}")
    print("strongest pairs:\n", strongest.head(8).round(3))
    print(f"median pairwise V among top {N_TOP}: {np.median(off):.3f}")


if __name__ == "__main__":
    main()
