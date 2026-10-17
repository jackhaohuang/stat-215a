"""Similarity between two cluster-membership vectors.

Implements the Jaccard coefficient described in Ben-Hur et al. (2002) for
comparing two k-means partitions of the same set of points. The naive way
to compute this coefficient looks at all q(q-1)/2 pairs of points and is
O(q^2) in both time and memory (it conceptually builds a q x q
co-membership matrix). similarity() instead builds the much smaller
k1 x k2 contingency table between the two label vectors (k1, k2 are the
number of clusters in each partition, typically <= 10) and derives the
Jaccard coefficient from that table using the standard pair-counting
identities. This makes the algorithm O(q + k1*k2), i.e. linear in q since
k1, k2 are small constants, and it never stores anything larger than the
k1 x k2 table in memory. The core loop is compiled with Numba so that the
function is fast enough to be called thousands of times inside Algorithm
1 (see algorithm1.py).

regular_table() and numpy_table() are reference implementations (a naive
O(q^2) pairwise loop, and an uncompiled-but-vectorized version of the same
O(q) algorithm) used only as timing baselines, and generate_timing_csv()
benchmarks all three; see that function's docstring.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _jaccard_from_labels(labels1, labels2, k1, k2):
    """Numba-compiled core: build the contingency table and reduce it.

    Args:
        labels1: int64 array of cluster labels (0..k1-1) from clustering 1.
        labels2: int64 array of cluster labels (0..k2-1) from clustering 2,
            same length and point order as labels1.
        k1: number of clusters in clustering 1.
        k2: number of clusters in clustering 2.

    Returns:
        The Jaccard similarity between the two clusterings, as a float.
    """
    # contingency_table[a, b] = number of points with label a in
    # clustering 1 and label b in clustering 2.
    contingency_table = np.zeros((k1, k2), dtype=np.int64)
    n_points = labels1.shape[0]
    for i in range(n_points):
        contingency_table[labels1[i], labels2[i]] += 1

    # n_11 = number of point pairs placed in the same cluster by *both*
    # clusterings = sum over contingency table cells of C(count, 2).
    n_11 = 0
    row_sums = np.zeros(k1, dtype=np.int64)
    col_sums = np.zeros(k2, dtype=np.int64)
    for a in range(k1):
        for b in range(k2):
            count = contingency_table[a, b]
            n_11 += count * (count - 1) // 2
            row_sums[a] += count
            col_sums[b] += count

    # n_10 = pairs placed together by clustering 1 but not clustering 2,
    # n_01 = pairs placed together by clustering 2 but not clustering 1.
    # sum C(row_sum, 2) counts every pair together in clustering 1
    # (= n_11 + n_10), and likewise for the columns and clustering 2.
    pairs_together_1 = 0
    for a in range(k1):
        pairs_together_1 += row_sums[a] * (row_sums[a] - 1) // 2
    pairs_together_2 = 0
    for b in range(k2):
        pairs_together_2 += col_sums[b] * (col_sums[b] - 1) // 2
    n_10 = pairs_together_1 - n_11
    n_01 = pairs_together_2 - n_11

    denominator = n_11 + n_10 + n_01
    if denominator == 0:
        # Both clusterings put every point in its own singleton cluster
        # (no pair is ever together), which we define as perfect
        # agreement rather than dividing by zero.
        return 1.0
    return n_11 / denominator


def similarity(a, b):
    """Jaccard similarity between two cluster-membership vectors.

    Args:
        a: 1D array-like of cluster labels, one per point.
        b: 1D array-like of cluster labels for the *same*
            points in the *same* order, from a second clustering.

    Returns:
        A float in [0, 1]: the fraction of point pairs placed in the
        same cluster by both clusterings, among pairs placed in the same
        cluster by at least one of the two.

    Note:
        Labels are assumed to be small non-negative integers (as
        produced by scikit-learn's KMeans, 0..k-1), not necessarily
        contiguous. Re-mapping arbitrary labels (e.g. via np.unique)
        would add a sort that costs more than the similarity
        computation itself, so this is intentionally not done here.
    """
    labels1 = np.asarray(a).astype(np.int64)
    labels2 = np.asarray(b).astype(np.int64)

    k1 = int(labels1.max()) + 1 if labels1.size else 0
    k2 = int(labels2.max()) + 1 if labels2.size else 0

    return _jaccard_from_labels(labels1, labels2, k1, k2)


def regular_table(a, b):
    """Naive O(q^2) Jaccard similarity: loop over all point pairs directly.

    No contingency table, no numba -- this is the textbook definition,
    used only as a timing baseline for similarity() (see
    generate_timing_csv()).
    """
    labels1 = np.asarray(a)
    labels2 = np.asarray(b)

    q = labels1.shape[0]
    n_11 = n_10 = n_01 = 0
    for i in range(q):
        for j in range(i + 1, q):
            same1 = labels1[i] == labels1[j]
            same2 = labels2[i] == labels2[j]
            if same1 and same2:
                n_11 += 1
            elif same1:
                n_10 += 1
            elif same2:
                n_01 += 1

    denominator = n_11 + n_10 + n_01
    return n_11 / denominator if denominator else 1.0


def numpy_table(a, b):
    """Same O(q + k^2) contingency-table algorithm as similarity(), but as
    vectorized, uncompiled numpy calls instead of a compiled loop.

    Isolates how much of similarity()'s speed comes from the O(q)
    algorithm itself versus from numba's compilation; a timing baseline
    only (see generate_timing_csv()).
    """
    labels1 = np.asarray(a)
    labels2 = np.asarray(b)
    k1 = int(labels1.max()) + 1
    k2 = int(labels2.max()) + 1

    flat_idx = labels1.astype(np.int64) * k2 + labels2
    contingency = np.bincount(flat_idx, minlength=k1 * k2).reshape(k1, k2)
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)

    def n_choose_2(x):
        return x * (x - 1) // 2

    n_11 = n_choose_2(contingency).sum()
    denominator = n_choose_2(row_sums).sum() + n_choose_2(col_sums).sum() - n_11
    return n_11 / denominator if denominator else 1.0


def generate_timing_csv(
    output_path="../results/similarity_timing.csv",
    n_lingbinary=45152,
    k=10,
    n_repeats=20,
    naive_max_q=5000,
    m_values=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
):
    """Benchmark similarity(), numpy_table(), and regular_table() at the q
    values that actually arise from lingBinary.Rdata for a range of
    subsample fractions m, and write a long-format CSV to output_path
    (used for the runtime comparison figure in the report notebook).

    Only needs to be run once (or again if the implementations change);
    otherwise the notebook just reads the already-saved CSV.
    """
    import time

    import pandas as pd

    def time_call(fn, labels1, labels2, repeats):
        fn(labels1, labels2)  # warm up / compile
        start = time.perf_counter()
        for _ in range(repeats):
            fn(labels1, labels2)
        return (time.perf_counter() - start) / repeats

    rng = np.random.default_rng(0)
    rows = []
    # the actual number of points common to two independent m-fraction
    # subsamples of n is approximately m^2 * n.
    for m in m_values:
        q = int(round(m**2 * n_lingbinary))
        labels1 = rng.integers(0, k, q)
        labels2 = rng.integers(0, k, q)

        rows.append({
            "m": m, "q": q, "implementation": "numba contingency jaccard",
            "seconds": time_call(similarity, labels1, labels2, n_repeats),
        })
        rows.append({
            "m": m, "q": q, "implementation": "numpy contingency jaccard",
            "seconds": time_call(numpy_table, labels1, labels2, n_repeats),
        })
        # the naive O(q^2) python loop is only feasible for the smaller
        # q values, since it is >10,000x slower than the other two.
        if q <= naive_max_q:
            rows.append({
                "m": m, "q": q, "implementation": "regular jaccard",
                "seconds": time_call(regular_table, labels1, labels2, 2),
            })

    timing = pd.DataFrame(rows)
    timing.to_csv(output_path, index=False)
    return timing
