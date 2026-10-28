"""Paired comparison of two forecasting methods over scored (cohort, snapshot) pairs.

Declared procedure (paper v3, round 3 of the reviews):
  statistic   d = log(|relative error of a| + floor) - log(|relative error of b| + floor); error ratio = exp(mean d)
  primary     exact cluster sign-flip test: D_c = sum of d over the pairs of cohort c; all 2^C sign vectors
  robustness  cohort-cluster bootstrap interval (the pre-specified one); two-way (cohort x snapshot) pigeonhole
              bootstrap; wild cluster bootstrap-t with centred residuals; leave-one-snapshot-out
  secondary   ratio of mean absolute relative errors; floors 1e-4 and 1e-3; 5% trimming of d

A cohort is scored at several snapshots, so pairs of one cohort are not independent: cohorts are the clusters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Comparison",
    "cluster_bootstrap",
    "compare",
    "leave_one_out",
    "log_ratio",
    "sign_flip_p",
    "two_way_bootstrap",
    "wild_cluster_p",
]

Array = NDArray[np.float64]
FLOOR: Final = 1e-6  # pre-specified
MAX_EXACT: Final = 20  # 2^20 sign vectors are enumerated; above that a fixed random sample is used
N_BOOT: Final = 2000


@dataclass(frozen=True, slots=True)
class Comparison:
    pairs: int
    clusters: int
    snapshots: int
    error_ratio: float
    ci95_cluster_bootstrap: tuple[float, float]  # of the mean log ratio
    sign_flip_p: float
    clusters_where_a_is_better: int
    snapshots_where_a_is_better: int
    ci95_two_way_bootstrap: tuple[float, float]
    wild_cluster_p: float
    leave_one_snapshot_out_ratio: tuple[float, float]
    leave_one_snapshot_out_max_p: float
    mae_ratio: float
    mae_sign_flip_p: float
    ratio_floor_1e4: float
    sign_flip_p_floor_1e4: float
    ratio_floor_1e3: float
    sign_flip_p_floor_1e3: float
    ratio_trimmed_5pct: float
    sign_flip_p_trimmed_5pct: float

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__slots__}


def log_ratio(error_a: Array, error_b: Array, floor: float = FLOOR) -> Array:
    """Paired difference of log absolute relative errors."""
    return np.asarray(
        np.log(np.abs(error_a) + floor) - np.log(np.abs(error_b) + floor), dtype=np.float64
    )


def _sums(d: Array, clusters: NDArray[np.int64]) -> Array:
    return np.asarray(
        np.bincount(np.unique(clusters, return_inverse=True)[1], weights=d), dtype=np.float64
    )


def _signs(c: int, seed: int) -> NDArray[np.int8]:
    """All 2^c sign vectors, or 2^MAX_EXACT random ones (with the identity) when c is larger."""
    if c <= MAX_EXACT:
        bits = (np.arange(2**c)[:, None] >> np.arange(c)) & 1
        return np.asarray(1 - 2 * bits, dtype=np.int8)
    random = np.random.default_rng(seed).integers(0, 2, size=(2**MAX_EXACT, c))
    random[0] = 0
    return np.asarray(1 - 2 * random, dtype=np.int8)


def sign_flip_p(d: Array, clusters: NDArray[np.int64], seed: int = 0) -> float:
    """Two-sided randomization p value of mean d = 0 when the sign of each cluster sum is exchangeable."""
    sums = _sums(d, clusters)
    flipped = np.abs(_signs(len(sums), seed) @ sums)
    return float(np.mean(flipped >= abs(float(sums.sum())) - 1e-12))


def cluster_bootstrap(
    d: Array, clusters: NDArray[np.int64], n_boot: int = N_BOOT, seed: int = 0
) -> tuple[float, float]:
    """95% percentile interval of mean d, resampling clusters (the code of the pre-specified runs)."""
    keys = np.unique(clusters)
    index = {k: np.where(clusters == k)[0] for k in keys}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate([index[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        boot[i] = d[pick].mean()
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def two_way_bootstrap(
    d: Array,
    cohorts: NDArray[np.int64],
    snapshots: NDArray[np.int64],
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> tuple[float, float]:
    """Pigeonhole bootstrap (Owen 2007): cohorts and snapshots are resampled independently; a pair's weight is the
    product of the two counts. It allows dependence within a cohort and within a snapshot."""
    c_index = np.unique(cohorts, return_inverse=True)[1]
    s_index = np.unique(snapshots, return_inverse=True)[1]
    n_c, n_s = int(c_index.max()) + 1, int(s_index.max()) + 1
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        weight = (
            np.bincount(rng.integers(0, n_c, n_c), minlength=n_c)[c_index]
            * np.bincount(rng.integers(0, n_s, n_s), minlength=n_s)[s_index]
        )
        boot[i] = float(np.sum(weight * d) / weight.sum()) if weight.sum() > 0 else np.nan
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return float(low), float(high)


def wild_cluster_p(d: Array, clusters: NDArray[np.int64], seed: int = 0) -> float:
    """Wild cluster bootstrap-t (Cameron, Gelbach & Miller 2008) of mean d = 0, Rademacher weights on centred
    residuals. With the null imposed, the statistic's denominator does not change under sign flips and the test is
    the sign-flip test; this version does not impose the null, so it is a different check."""
    index = np.unique(clusters, return_inverse=True)[1]
    c, n = int(index.max()) + 1, len(d)
    if c < 2:
        return 1.0
    counts = np.bincount(index, minlength=c).astype(float)
    sums = np.bincount(index, weights=d, minlength=c)
    residual = sums - float(d.mean()) * counts  # cluster sums of the centred values
    scale = c / (c - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = sums.sum() / np.sqrt(np.sum(residual**2) * scale)
        # bootstrap sample = mean + w_c x centred value: its deviation from the mean has cluster sums `star`
        star = _signs(c, seed).astype(float) * residual
        centred = star - counts * (star.sum(axis=1, keepdims=True) / n)
        t_star = star.sum(axis=1) / np.sqrt(np.sum(centred**2, axis=1) * scale)
    t_star = np.nan_to_num(t_star, nan=0.0)
    return float(np.mean(np.abs(t_star) >= abs(float(observed)) - 1e-12))


def leave_one_out(d: Array, groups: NDArray[np.int64]) -> tuple[float, float]:
    """Smallest and largest mean of d when one group (snapshot) is left out."""
    means = [float(d[groups != g].mean()) for g in np.unique(groups)]
    return min(means), max(means)


def compare(
    error_a: Array,
    error_b: Array,
    cohorts: NDArray[np.int64],
    snapshots: NDArray[np.int64],
    floor: float = FLOOR,
) -> Comparison:
    """All statistics of the declared procedure for method a against method b. Errors are relative errors
    (estimate - truth) / truth of the same scored pairs."""
    a, b = (
        np.abs(np.asarray(error_a, dtype=np.float64)),
        np.abs(np.asarray(error_b, dtype=np.float64)),
    )
    d = log_ratio(a, b, floor)
    by_cohort, by_snapshot = _sums(d, cohorts), _sums(d, snapshots)
    snapshot_means = np.array(
        [[a[snapshots == s].mean(), b[snapshots == s].mean()] for s in np.unique(snapshots)]
    )
    loo = leave_one_out(d, snapshots)
    loo_p = max(
        sign_flip_p(d[snapshots != s], cohorts[snapshots != s]) for s in np.unique(snapshots)
    )
    keep = np.abs(d - np.median(d)) <= np.quantile(np.abs(d - np.median(d)), 0.95)
    floors = {f: log_ratio(a, b, f) for f in (1e-4, 1e-3)}
    low, high = cluster_bootstrap(d, cohorts)
    return Comparison(
        pairs=len(d),
        clusters=len(by_cohort),
        snapshots=len(by_snapshot),
        error_ratio=float(np.exp(d.mean())),
        ci95_cluster_bootstrap=(low, high),
        sign_flip_p=sign_flip_p(d, cohorts),
        clusters_where_a_is_better=int((by_cohort < 0).sum()),
        snapshots_where_a_is_better=int((snapshot_means[:, 0] < snapshot_means[:, 1]).sum()),
        ci95_two_way_bootstrap=two_way_bootstrap(d, cohorts, snapshots),
        wild_cluster_p=wild_cluster_p(d, cohorts),
        leave_one_snapshot_out_ratio=(float(np.exp(loo[0])), float(np.exp(loo[1]))),
        leave_one_snapshot_out_max_p=float(loo_p),
        mae_ratio=float(a.mean() / b.mean()),
        mae_sign_flip_p=sign_flip_p(a - b, cohorts),
        ratio_floor_1e4=float(np.exp(floors[1e-4].mean())),
        sign_flip_p_floor_1e4=sign_flip_p(floors[1e-4], cohorts),
        ratio_floor_1e3=float(np.exp(floors[1e-3].mean())),
        sign_flip_p_floor_1e3=sign_flip_p(floors[1e-3], cohorts),
        ratio_trimmed_5pct=float(np.exp(d[keep].mean())),
        sign_flip_p_trimmed_5pct=sign_flip_p(d[keep], cohorts[keep]),
    )
