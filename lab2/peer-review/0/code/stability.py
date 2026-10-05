"""Stability of the k = 3 k-means clustering -> lab task 5

The reference partition is the k-means fit from cluster.py (k = 3, best of 10 k-means starts, seed clean.SEED)
How much does it changes when
  (a) the algorithm starts from different random points (n_init = 1)
  (b) the data are perturbed: bootstrap resamples and 50% subsamples (each re-fit is used to label all respondents, then compared with the reference by the adjusted Rand index, ARI)
  (c) the number of clusters k changes (bootstrap ARI for k = 2..6)
  (d) the later copies of duplicated answer vectors, which the main sample drops, are kept instead (compared on the respondents both share)

Outputs: figs/st_stability.pdf and report/values/stability.tex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from clean import (FIGS_DIR, SEED, load_clean, load_states, one_hot, question_columns, write_values) # noqa: E402

from cluster import (CLUSTER_NAMES, K_CHOSEN, MIN_CELL_COUNT,SEQ_CMAP, cell_table, draw_cells, fit_kmeans, load_matrix, match_labels, order_clusters) # noqa: E402

N_STARTS = 50 # single random starts
N_BOOT = 30 # bootstrap resamples
N_SUB = 30 # 50% subsamples
SUB_FRAC = 0.5
N_BOOT_PER_K = 15 # bootstrap resamples per k in the k scan
K_SCAN = range(2, 7)
N_INIT_PERTURB = 3 # starts per re-fit on perturbed data

def refit_ari(x: np.ndarray, rows: np.ndarray, k: int, reference: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
    # Fit k-means on xrows, label everyone, return ARI and matched labels
    km = fit_kmeans(x[rows], k, n_init=N_INIT_PERTURB, seed=seed)
    labels = km.predict(x)
    return (adjusted_rand_score(reference, labels), match_labels(labels, reference, k))


def main() -> None:
    # Run all stability checks, draw the figure and write the values
    rng = np.random.default_rng(SEED)
    df, x, _ =load_matrix()
    n = len(x)
    ref_fit = fit_kmeans(x, K_CHOSEN)
    reference = order_clusters(ref_fit.labels_, df)

    # (a) Different starting points: one k-means start each
    start_ari, start_obj, start_changed = [], [], []
    for i in range(N_STARTS):
        km = fit_kmeans(x, K_CHOSEN, n_init=1, seed=SEED + 1 + i)
        start_ari.append(adjusted_rand_score(reference, km.labels_))
        start_obj.append(km.inertia_ / ref_fit.inertia_)
        matched = match_labels(km.labels_, reference, K_CHOSEN)
        start_changed.append(np.mean(matched != reference))
    start_ari, start_obj = np.array(start_ari), np.array(start_obj)

    # (b) Bootstrap and subsample perturbations
    boot_ari, sub_ari, agree = [], [], np.zeros(n)
    for i in range(N_BOOT):
        rows = rng.integers(0, n, n)
        ari, lab = refit_ari(x, rows, K_CHOSEN, reference, SEED + 100 + i)
        boot_ari.append(ari)
        agree += lab == reference
    agree /= N_BOOT
    for i in range(N_SUB):
        rows = rng.choice(n, int(SUB_FRAC * n), replace=False)
        sub_ari.append(refit_ari(x, rows, K_CHOSEN, reference, SEED + 200 + i)[0])
    boot_ari, sub_ari= np.array(boot_ari), np.array(sub_ari)
    per_cluster = [agree[reference == c].mean() for c in range(K_CHOSEN)]

    # (c) Stability across k: bootstrap ARI against each k's own reference
    k_rows = []
    for k in K_SCAN:
        ref_k = fit_kmeans(x, k).labels_
        for i in range(N_BOOT_PER_K):
            rows = rng.integers(0, n, n)
            k_rows.append({"k": k, "ari": refit_ari(x, rows, k, ref_k, SEED + 300 + i)[0]})
    k_scan = pd.DataFrame(k_rows)
    k_median = k_scan.groupby("k")["ari"].median()
    print(k_scan.groupby("k")["ari"].describe().round(3))

    # (d) Keep the later copies of duplicated answer vectors and compare the fit with the reference on the respondents both samples share
    df_all = load_clean(keep_copies=True)
    q = question_columns(df_all)
    in_dup_group = df_all.duplicated(subset=q, keep=False).to_numpy()
    biggest_group = int(df_all.groupby(q).size().max())
    shared = df_all["ID"].isin(df["ID"]).to_numpy()
    assert (df_all.loc[shared, "ID"].to_numpy() == df["ID"].to_numpy()).all()
    x_all = one_hot(df_all)[0].astype(np.float32)
    km = fit_kmeans(x_all -x_all.mean(axis=0), K_CHOSEN)
    dedup_ari = adjusted_rand_score(reference, km.labels_[shared])
    n_extra = len(df_all) - len(df)

    # - Figure: ARI summaries and map of per-person agreement
    fig = plt.figure(figsize=(5.3, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.6, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    groups = [start_ari, boot_ari, sub_ari]
    ax.boxplot(groups, widths=0.5, showfliers=False,
               medianprops={"color": "#222222"},
               boxprops={"color": "#6f6f6f"}, whiskerprops={"color": "#6f6f6f"},
               capprops={"color": "#6f6f6f"})
    for j, g in enumerate(groups, start=1):
        jitter = rng.uniform(-0.12, 0.12, len(g))
        ax.scatter(j + jitter, g, s=12, color="#2a78d6", alpha=0.7, lw=0, zorder=3)
    ax.set_xticks([1, 2, 3], [f"starts\n(n = {N_STARTS})",f"bootstrap\n(n = {N_BOOT})",f"half-sample\n(n = {N_SUB})"])
    ax.set_ylabel("ARI with reference partition")
    ax.set_title("(a) Refits of k-means with k = 3")
    lo = min(g.min() for g in groups)
    ax.set_ylim(min(0.9, lo - 0.02), 1.005)

    ax = fig.add_subplot(gs[0, 1])
    for k in K_SCAN:
        vals = k_scan.loc[k_scan["k"] == k, "ari"]
        ax.scatter(np.full(len(vals), k) + rng.uniform(-0.12, 0.12, len(vals)),vals, s=12, color="#2a78d6", alpha=0.7, lw=0)
    ax.plot(k_median.index, k_median.values, "-", color="#222222", lw=1.2,label="median")
    ax.set_xticks(list(K_SCAN))
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("bootstrap ARI")
    ax.set_title("(b) Bootstrap stability by k")
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, loc="lower left")

    ax = fig.add_subplot(gs[1, :])
    # Share of people per cell who switch cluster in at least one re-fit.
    ever_switch = (agree < 1).astype(float)
    cells = cell_table(df, ever_switch[:, None])
    cells = cells[cells["count"] >= MIN_CELL_COUNT]
    idx = cells.index.to_frame().to_numpy()
    top_c = 0.3
    vals = np.clip(cells[0].to_numpy() / top_c, 0, 1)
    draw_cells(ax, idx[:, 0], idx[:, 1], SEQ_CMAP(vals), states=load_states())
    ax.set_title("(c) Share of people per cell who switch cluster "
                 "in a bootstrap refit")
    sm = plt.cm.ScalarMappable(cmap=SEQ_CMAP, norm=plt.Normalize(0, top_c))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, extend="max")
    cbar.set_label("share of people who ever switch")
    fig.savefig(FIGS_DIR / "st_stability.pdf", bbox_inches="tight")
    plt.close(fig)

    # - Values for the report 
    word = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}
    values = {
        "stNStarts": N_STARTS,
        "stNBoot": N_BOOT,
        "stNSub": N_SUB,
        "stSubPct": f"{100 * SUB_FRAC:.0f}",
        "stNBootPerK": N_BOOT_PER_K,
        "stStartARIMin": f"{start_ari.min():.3f}",
        "stStartARIMedian": f"{np.median(start_ari):.3f}",
        "stStartObjMaxPct": f"{100 * (start_obj.max() - 1):.3f}",
        "stStartPctSame": f"{100 * np.mean(start_ari > 0.99):.0f}",
        "stBootARIMin": f"{boot_ari.min():.3f}",
        "stBootARIMedian": f"{np.median(boot_ari):.3f}",
        "stSubARIMin": f"{sub_ari.min():.3f}",
        "stSubARIMedian": f"{np.median(sub_ari):.3f}",
        "stPctPeopleStable": f"{100 * np.mean(agree >= 0.9):.0f}",
        "stPctPeopleUnstable": f"{100 * np.mean(agree < 0.5):.1f}",
        "stPctEverSwitch": f"{100 * np.mean(agree < 1):.0f}",
        "stStartPctChangedMax": f"{100 * max(start_changed):.1f}",
        "stNDupRows": n_extra,
        "stPctInDupGroup": f"{100 * in_dup_group.mean():.0f}",
        "stBiggestDupGroup": biggest_group,
        "stDedupARI": f"{dedup_ari:.2f}",
    }
    for c, key in enumerate(["South", "NorthWest", "Northeast"]): values[f"stAgree{key}"] = f"{100 * per_cluster[c]:.0f}"
    for k in K_SCAN: values[f"stBootARIk{word[k]}"] = f"{k_median[k]:.2f}"
    write_values("stability", values)
    print(values)
    print(dict(zip(CLUSTER_NAMES, np.round(per_cluster, 3))))


if __name__ == "__main__":
    main()
