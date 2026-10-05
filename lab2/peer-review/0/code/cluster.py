"""Cluster the dialect survey respondents (task 4).

Compare two clustering methods (k-means and Ward hierarchical clustering):

  1. k-means on individuals, using the centered 468-column binary matrix, which is the same matrix PCA works on
  2. Ward hierarchical clustering on 1 x 1 degree grid cells, with each
     cell being described by the average binary answer vector of the people in
     it -> a cleaned, normalised version of lingLocation

Outputs:
  figs/cl_choose_k.pdf: choosing k and continuum-vs-clusters diagnostics
  figs/cl_kmeans_map.pdf: share of each k-means cluster per grid cell
  figs/cl_ward_map.pdf: Ward clusters of grid cells + merge heights
  report/values/cluster.tex

The helper functions (building the matrix, fitting the reference k-means, naming clusters, drawing maps) are later used by stability.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402

from clean import (FIGS_DIR, SEED, load_clean, load_questions, load_states, one_hot, write_values) # noqa: E402

K_CHOSEN = 3  # number of clusters I settle on (see notes)
K_RANGE = range(2, 9)  # values of k I scan
N_SILHOUETTE = 5000  # people per silhouette subsample
N_SIL_REPEATS = 3  # silhouette subsamples averaged per k
MIN_CELL_COUNT = 10   # minimum people for a grid cell to be used
N_TOP_ANSWERS = 5    # answers listed per cluster in the console

# Color-blind-checked categorical colours (blue, orange, aqua), one per cluster in the fixed order South, North/West, Northeast.
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
CLUSTER_NAMES = ["South", "North/West", "Northeast"]
# One-hue sequential ramp (light to dark blue) for shares/agreement.
SEQ_CMAP = LinearSegmentedColormap.from_list("blue_ramp", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


# - - -
# Data and model helpers (also used by stability.py)

def load_matrix() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    # Return cleaned data, the centered binary matrix, and names
    df = load_clean()
    x, names = one_hot(df)
    x = x.astype(np.float32)
    return df, x - x.mean(axis=0), names


def fit_kmeans(x: np.ndarray, k: int, n_init: int = 10, seed: int = SEED) -> KMeans:
    # Fit k-means with k-means starts -> the best of n_init runs is kept
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(x)


def order_clusters(labels: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """ Relabel 3 clusters as 0 = South, 1 = North/West, 2 = Northeast

    k-means labels are arbitrary; fixing the order by geography makes the
    colours and names identical in every figure. The southernmost cluster
    (by mean latitude) is "South"; of the other two the eastern one (by mean
    longitude) is "Northeast".
    """
    k = labels.max() + 1
    lat = np.array([df["lat"].to_numpy()[labels == c].mean() for c in range(k)])
    lon = np.array([df["long"].to_numpy()[labels == c].mean() for c in range(k)])
    south = int(np.argmin(lat))
    rest = [c for c in range(k) if c != south]
    rest.sort(key=lambda c: lon[c])  # west first, then east
    order = [south] + rest
    new = np.empty(k, dtype=int)
    new[order] = np.arange(k)
    return new[labels]


def match_labels(labels: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    # Permute `labels` to agree best with `reference`
    table = np.zeros((k, k))
    np.add.at(table, (labels, reference), 1)
    rows, cols = linear_sum_assignment(-table)
    mapping = np.empty(k, dtype=int)
    mapping[rows] = cols
    return mapping[labels]


def grid_cells(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Return integer (lat, long) 1 degree cell indices for all persons
    return (np.floor(df["lat"].to_numpy()).astype(int), np.floor(df["long"].to_numpy()).astype(int))


def cell_table(df: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    # Average `values` (n*p) over 1 degree cells -> adds count per cell
    lat_i, lon_i = grid_cells(df)
    out = pd.DataFrame(values).groupby([lat_i, lon_i]).mean()
    out["count"] = pd.Series(1, index=df.index).groupby([lat_i, lon_i]).sum()
    out.index.names = ["lat_i", "lon_i"]
    return out


def draw_cells(ax, lat_i, lon_i, colors, states) -> None:
    # Draw 1 degree squares with the given facecolors over state outlines
    states.plot(ax=ax, color="#f2f2f0", edgecolor="none", zorder=0)
    for la, lo, col in zip(lat_i, lon_i, colors):
        ax.add_patch(plt.Rectangle((lo, la), 1, 1, facecolor=col, edgecolor="white", linewidth=0.3, zorder=1))
    states.boundary.plot(ax=ax, color="#555555", linewidth=0.35, zorder=2)
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24.5, 49.5)
    ax.set_aspect(1 / np.cos(np.radians(38)))   # rough equal-area look
    ax.set_axis_off()


# - - -
# Analysis steps

def scan_k(x: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    # Fit k-means for each k ->  record inertia and mean subsample silhouette
    subsamples = [rng.choice(len(x), N_SILHOUETTE, replace=False) for _ in range(N_SIL_REPEATS)]
    rows = []
    for k in K_RANGE:
        km = fit_kmeans(x, k)
        sil = np.mean([silhouette_score(x[s], km.labels_[s]) for s in subsamples])
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil, "smallest": np.bincount(km.labels_).min()})
    return pd.DataFrame(rows)


def answer_label(name: str, quest: pd.DataFrame, answers: dict) -> str:
    # Turn 'Q105_1' into a short readable label like 'Q105 soda'
    q, a = name.split("_")
    text = answers[int(q[1:])]["ans"].iloc[int(a) - 1].strip()
    return f"{q}: {text}"


def top_answers(x_raw: np.ndarray, labels: np.ndarray, names: list[str],
                quest, answers) -> dict[int, list[tuple[str, float,float]]]:
    # For all clusters, answers most over-used relative to everyone else.
    # Returns (label, share in cluster, share outside) for the answers with the largest difference in proportions
    out = {}
    for c in range(labels.max() + 1):
        inside = x_raw[labels == c].mean(axis=0)
        outside = x_raw[labels != c].mean(axis=0)
        best = np.argsort(inside - outside)[::-1][:N_TOP_ANSWERS]
        out[c] = [(answer_label(names[j], quest, answers), inside[j], outside[j]) for j in best]
    return out


def pair_projection(x: np.ndarray, labels: np.ndarray, centers: np.ndarray, a: int, b: int) -> np.ndarray:
    """Project people of clusters a and b onto the line through their centres.

    Use scaling so that centre a sits at 0 and centre b at 1
    k-means cuts at 0.5 ignoring the third cluster. Two well-separated groups
    would give an empty gap near 0.5 -> a single hump would mean the two clusters are just
    two halves of a continuum and overlapping humps are inbetween
    """
    keep = (labels == a) | (labels == b)
    d = centers[b] -centers[a]
    return (x[keep] - centers[a]) @ d / (d @ d)


def main() -> None:
    #Run both clustering methods, draw the figures and write the values
    rng = np.random.default_rng(SEED)
    df, x, names = load_matrix()
    x_raw = x + one_hot(df)[0].mean(axis=0)   # back to 0/1 for shares
    quest, answers = load_questions()
    states = load_states()

    # PCA scores
    pca = PCA(n_components=10, random_state=SEED).fit(x)
    scores = pca.transform(x)

    # - Method 1: k-means on people
    scan = scan_k(x, rng)
    print(scan.round(4))
    km = fit_kmeans(x, K_CHOSEN)
    labels = order_clusters(km.labels_, df)
    centers = np.array([x[labels == c].mean(axis=0) for c in range(K_CHOSEN)])
    sizes = np.bincount(labels)
    # k-means on the top-10 PC scores instead of all 468 columns
    km_pc = fit_kmeans(scores, K_CHOSEN)
    ari_pc = adjusted_rand_score(labels, km_pc.labels_)
    tops = top_answers(x_raw, labels, names, quest, answers)
    for c in range(K_CHOSEN):
        print(CLUSTER_NAMES[c], sizes[c], [(t, round(i, 2), round(o, 2)) for t, i, o in tops[c]])

    # Distance ratio: nearest or second-nearest centre -> 1 = on the boundary
    dist = np.sqrt(((x[:, None, :] - centers[None]) ** 2).sum(axis=2))
    dist.sort(axis=1)
    ratio = dist[:, 0] / dist[:, 1]

    # - Method 2: Ward on grid cells 
    cells_x = cell_table(df, x_raw)
    cells_x = cells_x[cells_x["count"] >= MIN_CELL_COUNT]
    feats = cells_x.drop(columns="count").to_numpy()
    tree = linkage(feats, method="ward")
    ward_sil = {k: silhouette_score(feats, fcluster(tree, k, "maxclust")) for k in range(2, 7)}
    print("Ward cell silhouettes", {k: round(v, 3) for k, v in ward_sil.items()})
    ward = fcluster(tree, K_CHOSEN, "maxclust") - 1

    # k-means shares per cell and person-level comparison
    shares = cell_table(df, np.eye(K_CHOSEN)[labels]).loc[cells_x.index]
    km_dom = shares[list(range(K_CHOSEN))].to_numpy().argmax(axis=1)
    ward = match_labels(ward, km_dom, K_CHOSEN)   # same names as k-means
    agree_cells = np.mean(ward == km_dom)
    lat_i, lon_i = grid_cells(df)
    person_cell = pd.MultiIndex.from_arrays([lat_i, lon_i])
    in_cell = person_cell.isin(cells_x.index)
    ward_of_cell = pd.Series(ward, index=cells_x.index)
    ward_person = ward_of_cell.loc[person_cell[in_cell]].to_numpy()
    ari_ward = adjusted_rand_score(labels[in_cell], ward_person)
    ari_dom = adjusted_rand_score(labels[in_cell], pd.Series(km_dom, index=cells_x.index).loc[person_cell[in_cell]].to_numpy())
    dom_share = shares[list(range(K_CHOSEN))].to_numpy().max(axis=1)

    # Correlation of PC1/PC2 with location -> does the linear view fit?
    corr_pc1_long = np.corrcoef(scores[:, 0], df["long"])[0, 1]
    corr_pc2_lat = np.corrcoef(scores[:, 1], df["lat"])[0, 1]

    # - Figure 1: choosing k and continuum diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.2))
    ax = axes[0, 0]
    ax.plot(scan["k"], scan["inertia"] / scan["inertia"].iloc[0], "o-", color="#2a78d6", lw=2, ms=5)
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("within-cluster SS\n(relative to k = 2)")
    ax.set_title("(a) Elbow plot")
    ax = axes[0, 1]
    ax.plot(scan["k"], scan["silhouette"], "o-", color="#2a78d6", lw=2, ms=5)
    ax.plot(K_CHOSEN, scan.set_index("k").loc[K_CHOSEN, "silhouette"], "o", ms=11, mfc="none", mec="#222222", mew=1.2)
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("mean silhouette width")
    ax.set_title("(b) Mean silhouette width")
    ax.set_ylim(0, None)
    ax = axes[1, 0]
    sub = rng.choice(len(x), 6000, replace=False)
    for c in range(K_CHOSEN):
        s = sub[labels[sub] == c]
        ax.scatter(scores[s, 0], scores[s, 1], s=3, alpha=0.35, lw=0, color=CLUSTER_COLORS[c], label=CLUSTER_NAMES[c], rasterized=True)
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.set_title("(c) k-means clusters on PC1 and PC2")
    ax.legend(markerscale=4, frameon=False, loc="upper right", handletextpad=0.2)
    ax = axes[1, 1]
    bins = np.linspace(-1.5, 2.5, 41)
    mids = (bins[:-1] + bins[1:]) / 2
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_colors = ["#6f6f6f", "#1c5cab", "#b3471b"]
    dip_ratio, middle_frac = [], []
    for (a, b), col in zip(pairs, pair_colors):
        proj = pair_projection(x, labels, centers, a, b)
        dens, _ = np.histogram(proj, bins=bins, density=True)
        dip_ratio.append(dens[np.argmin(np.abs(mids - 0.5))] / dens.max())
        middle_frac.append(np.mean((proj > 0.25) & (proj < 0.75)))
        ax.stairs(dens, bins, lw=1.6, color=col, label=f"{CLUSTER_NAMES[a]} (0) vs {CLUSTER_NAMES[b]} (1)")
    ax.axvline(0.5, color="#222222", lw=0.8, ls="--")
    ax.set_ylim(0, 1.75 * ax.get_ylim()[1])
    ax.set_xlabel("position between two cluster centers ")
    ax.set_ylabel("density")
    ax.set_title("(d) Pairs of clusters")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "cl_choose_k.pdf")
    plt.close(fig)

    # - Figure 2: k-means cluster shares per grid cell
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 3.7))
    idx = shares.index.to_frame().to_numpy()
    panel = "abc"
    for c, ax in enumerate(axes.flat[:K_CHOSEN]):
        vals = shares[c].to_numpy()
        draw_cells(ax, idx[:, 0], idx[:, 1], SEQ_CMAP(vals), states)
        words = ", ".join(t.split(": ")[1] for t, _, _ in tops[c][:3])
        ax.set_title(f"({panel[c]}) share in {CLUSTER_NAMES[c]} cluster "
                     f"(n = {sizes[c]:,})\ntypical: {words}", fontsize=8)
    ax = axes[1, 1]
    cols = np.array(CLUSTER_COLORS)
    draw_cells(ax, idx[:, 0], idx[:, 1], cols[km_dom], states)
    ax.set_title("(d) most common cluster per cell", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLUSTER_COLORS]
    ax.legend(handles, CLUSTER_NAMES, loc="lower left", frameon=False, fontsize=7, handlelength=1, bbox_to_anchor=(-0.02, -0.08))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.12, wspace=0.04, hspace=0.25)
    cax = fig.add_axes([0.08, 0.06, 0.38, 0.018])
    sm = plt.cm.ScalarMappable(cmap=SEQ_CMAP, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("share of the cell's respondents in the cluster (a-c)", fontsize=8)
    fig.savefig(FIGS_DIR / "cl_kmeans_map.pdf", bbox_inches="tight")
    plt.close(fig)

    # - Figure 3: Ward clusters of grid cells + dendrogram
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.27), gridspec_kw={"width_ratios": [1.9, 1]})
    draw_cells(axes[0], idx[:, 0], idx[:, 1], cols[ward], states)
    axes[0].set_title(f"(a) Ward clusters of {len(cells_x)} grid cells "
                      f"(k = {K_CHOSEN})")
    axes[0].legend(handles, CLUSTER_NAMES, loc="lower left", frameon=False, fontsize=7.5, handlelength=1)
    ax = axes[1]
    heights = tree[::-1, 2][:12]   # the 12 highest merges
    ax.plot(np.arange(1, 13), heights, "o-", color="#2a78d6", lw=2, ms=5)
    ax.axvline(K_CHOSEN - 0.5,color="#222222", lw=0.8, ls="--")
    ax.set_xlabel("merge (1 = last merge)")
    ax.set_ylabel("Ward merge height")
    ax.set_title("(b) dendrogram merge heights")
    ax.set_xticks(range(1, 13, 2))
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "cl_ward_map.pdf", bbox_inches="tight")
    plt.close(fig)

    # - Values for the report
    sil = scan.set_index("k")["silhouette"]
    word = {2: "Two", 3: "Three", 4: "Four", 5: "Five",6: "Six", 7:"Seven", 8: "Eight"}
    values = {
        "clK": K_CHOSEN,
        "clNCells": len(cells_x),
        "clMinCellCount": MIN_CELL_COUNT,
        "clNPeopleInCells": int(in_cell.sum()),
        "clNSilSub": N_SILHOUETTE,
        "clSilBest": f"{sil.max():.3f}",
        "clARIkmPC": f"{ari_pc:.2f}",
        "clARIWard": f"{ari_ward:.2f}",
        "clARIDomCell": f"{ari_dom:.2f}",
        "clCellAgreePct": f"{100 * agree_cells:.0f}",
        "clMedianDomShare": f"{100 * np.median(dom_share):.0f}",
        "clPctCellsDomAboveSeventy": f"{100 * np.mean(dom_share > 0.7):.0f}",
        "clMedianDistRatio": f"{np.median(ratio):.2f}",
        "clPctRatioAboveNinety": f"{100 * np.mean(ratio > 0.9):.0f}",
        "clPCOneVar": f"{100 * pca.explained_variance_ratio_[0]:.1f}",
        "clPCTenVar": f"{100 * pca.explained_variance_ratio_.sum():.1f}",
        "clCorrPCOneLong": f"{corr_pc1_long:.2f}",
        "clCorrPCTwoLat": f"{corr_pc2_lat:.2f}",
        "clWardSilThree": f"{ward_sil[3]:.2f}",
        "clWardSilBest": f"{max(ward_sil.values()):.2f}",
        "clWardKBest": max(ward_sil, key=ward_sil.get),
        "clDipMin": f"{min(dip_ratio):.1f}",
        "clDipMax": f"{max(dip_ratio):.1f}",
        "clPctMiddleMin": f"{100 * min(middle_frac):.0f}",
        "clPctMiddleMax": f"{100 * max(middle_frac):.0f}",
        "clInertiaDropPct": f"{100 * (1 - scan['inertia'].iloc[-1]/ scan['inertia'].iloc[0]):.1f}",
    }
    for k in K_RANGE:
        values[f"clSil{word[k]}"] = f"{sil[k]:.3f}"
    for c, key in enumerate(["South", "NorthWest", "Northeast"]):
        values[f"clSize{key}"] = int(sizes[c])
        values[f"clPct{key}"] = f"{100 * sizes[c] / len(labels):.0f}"
        for r, (text, inside, outside) in enumerate(tops[c][:3]):
            nr = ["One", "Two", "Three"][r]
            values[f"clTop{key}{nr}"] = text.split(": ")[1]
            values[f"clTop{key}{nr}In"] = f"{100 * inside:.0f}"
            values[f"clTop{key}{nr}Out"] =f"{100 * outside:.0f}"
    write_values("cluster", values)
    print(values)

if __name__ == "__main__":
    main()
