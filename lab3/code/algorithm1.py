"""Algorithm 1 (Ben-Hur et al., 2002): k-means stability via subsampling.

For each k = 2, ..., k_max, draws B pairs of random m-fraction subsamples
of the data, runs k-means on each half, and records the Jaccard similarity
(similarity.py) between the two clusterings restricted to the points
common to both subsamples. Writes a long-format CSV with one row per
(k, b) replicate, used to reproduce the stability plot (Figure 3 in
Ben-Hur et al.) in the report notebook.

Deliberately a standalone script rather than code inside lab3.ipynb: it is
the computationally expensive part of the lab and is meant to be
submitted to the SCF cluster with sbatch (see run_algorithm1.sbatch).

Parallelization: k-means on k=10 clusters takes roughly 2x as long as on
k=2, so parallelizing only over k would leave cores idle while the
slowest k was still running. Instead every (k, b) replicate is flattened
into one list of kmax_values * B independent tasks and handed to a single
joblib.Parallel call, keeping all cores busy regardless of which k a
replicate belongs to.
"""

import argparse
import os
import time

# Keep BLAS single-threaded inside each joblib worker, otherwise
# numpy/scikit-learn would each spawn their own thread pool and
# oversubscribe the machine once joblib is running one worker per core.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import pyreadr
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from similarity import similarity


def load_lingbinary(path):
    """Load the binary question columns of lingBinary.Rdata as a float32 array."""
    df = pyreadr.read_r(path)["lingBinary"]
    q_cols = [c for c in df.columns if c.startswith("Q")]
    return df[q_cols].to_numpy(dtype=np.float32)


def one_replicate(X, k, m, seed):
    """Run one (sub1, sub2, cluster, compare) replicate of Algorithm 1.

    Args:
        X: (n, p) data matrix.
        k: number of k-means clusters.
        m: subsample fraction, applied independently to each subsample.
        seed: seed controlling both subsamples and both k-means fits, so
            the whole replicate is reproducible.

    Returns:
        (k, similarity) tuple, where similarity is the Jaccard
        similarity between the two clusterings on the points common to
        both subsamples.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    size = int(round(m * n))
    idx1 = rng.choice(n, size=size, replace=False)
    idx2 = rng.choice(n, size=size, replace=False)

    labels1 = KMeans(n_clusters=k, random_state=seed).fit(X[idx1]).labels_
    labels2 = KMeans(n_clusters=k, random_state=seed + 1).fit(X[idx2]).labels_

    # Points common to both subsamples, with their positions within each
    # subsample, so we can look up cluster labels without ever forming an
    # (n choose 2) co-membership object.
    _, pos1, pos2 = np.intersect1d(idx1, idx2, assume_unique=True, return_indices=True)
    return k, similarity(labels1[pos1], labels2[pos2])


def run_algorithm1(X, k_max, B, m, n_jobs, base_seed=0):
    """Run Algorithm 1 for k = 2..k_max, B replicates each, in parallel.

    Returns:
        A long-format DataFrame with columns "k" and "similarity", one
        row per (k, b) replicate.
    """
    tasks = [
        (k, b)
        for k in range(2, k_max + 1)
        for b in range(B)
    ]
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(one_replicate)(X, k, m, base_seed + 1000 * k + b)
        for k, b in tasks
    )
    return pd.DataFrame(results, columns=["k", "similarity"])


def generate_m_pilot_csv(
    data="../data/lingBinary.Rdata",
    output="../results/m_pilot.csv",
    k_max=10,
    B_pilot=10,
    m_values=(0.6, 0.7, 0.8),
    base_seed=0,
):
    """Pilot-time one_replicate() across m in {0.6, 0.7, 0.8} and k = 2..k_max.

    Runs B_pilot replicates per (m, k) cell, single-threaded and serially
    (not through joblib), so the measured per-call cost is directly
    comparable to what one joblib worker sees in the real run. Records
    both the wall-clock cost and the resulting Jaccard similarity for
    each replicate, so the pilot can speak to runtime (extrapolated
    full-job cost) and to reliability (whether the similarity scores
    Algorithm 1 depends on are sensitive to m) at once. Used to choose m
    for the full run -- see "Algorithm 1 implementation" in lab3.ipynb.

    Writes a long-format CSV with columns m, k, b, seconds, similarity.
    """
    X = load_lingbinary(data)
    rows = []
    for m in m_values:
        for k in range(2, k_max + 1):
            for b in range(B_pilot):
                seed = base_seed + 1000 * k + b
                start = time.time()
                _, sim = one_replicate(X, k, m, seed)
                elapsed = time.time() - start
                rows.append((m, k, b, elapsed, sim))
    df = pd.DataFrame(rows, columns=["m", "k", "b", "seconds", "similarity"])
    df.to_csv(output, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="../data/lingBinary.Rdata")
    parser.add_argument("--output", default="../results/S.csv")
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--m", type=float, default=0.8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    X = load_lingbinary(args.data)
    print(f"X shape: {X.shape}")

    start = time.time()
    S = run_algorithm1(
        X, k_max=args.k_max, B=args.B, m=args.m, n_jobs=args.n_jobs, base_seed=args.seed
    )
    elapsed = time.time() - start
    print(f"Algorithm 1 finished in {elapsed:.1f}s for k=2..{args.k_max}, B={args.B}, m={args.m}")

    S.to_csv(args.output, index=False)
    print(f"Saved {len(S)} rows to {args.output}")


if __name__ == "__main__":
    main()
