import numpy as np
from numba import njit, prange
from typing import Iterable, Tuple, Optional

# ---------- Numba-friendly binary searches ----------

@njit(inline='always')
def _lower_bound(a: np.ndarray, x: float) -> int:
    """First index i where a[i] >= x."""
    lo = 0
    hi = a.size
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo

@njit(inline='always')
def _upper_bound_inclusive(a: np.ndarray, x: float) -> int:
    """First index i where a[i] > x (i.e., last <= x is i-1)."""
    lo = 0
    hi = a.size
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo

@njit(inline='always')
def _upper_bound_exclusive(a: np.ndarray, x: float) -> int:
    """First index i where a[i] >= x (i.e., last < x is i-1)."""
    # same as lower_bound; kept separate for clarity
    return _lower_bound(a, x)

# ---------- Angle helpers ----------

@njit(inline='always')
def _wrap_abs_diff(a: float, b: float) -> float:
    d = (a - b + np.pi) % (2*np.pi) - np.pi
    return abs(d)

# ---------- Core kernels ----------

@njit
def _radius_query_once(th_sorted, th_ext, ids_sorted, ids_ext,
                       theta_q: float, delta: float, inclusive: bool) -> np.ndarray:
    """
    Return a view of ids_sorted / ids_ext for θ within ±delta of θq (wrap-aware).
    Uses custom lower/upper bounds to avoid Numba's searchsorted(side=...) issues.
    """
    if delta < 0.0 or th_sorted.size == 0:
        return ids_sorted[0:0]  # empty view

    two_pi = 2.0 * np.pi
    θq = theta_q % two_pi
    lo = θq - delta
    hi = θq + delta

    # choose upper bound function based on inclusivity
    # (right index is exclusive)
    if inclusive:
        ub = _upper_bound_inclusive
    else:
        ub = _upper_bound_exclusive

    if lo >= 0.0 and hi < two_pi:
        i0 = _lower_bound(th_sorted, lo)
        i1 = ub(th_sorted, hi)
        return ids_sorted[i0:i1]
    elif hi >= two_pi:
        # high-side wrap: search [lo, hi] in th_ext
        i0 = _lower_bound(th_ext, lo)
        i1 = ub(th_ext, hi)
        return ids_ext[i0:i1]
    else:
        # low-side wrap (lo < 0): search [lo+2π, hi+2π] in th_ext
        i0 = _lower_bound(th_ext, lo + two_pi)
        i1 = ub(th_ext, hi + two_pi)
        return ids_ext[i0:i1]
    

@njit
def _knn_query_once(th_sorted, ids_sorted, theta_q: float, k: int):
    """k-NN on the circle (returns fixed-size arrays of length k)."""
    n = th_sorted.size
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.float64)
    k = min(max(1, k), n)

    θq = theta_q % (2*np.pi)
    # insertion point == lower_bound
    i = _lower_bound(th_sorted, θq)
    L = i - 1
    R = i

    ids_out = np.empty(k, np.int64)
    dists_out = np.empty(k, np.float64)
    t = 0
    while t < k and (L >= 0 or R < n):
        dL = _wrap_abs_diff(th_sorted[L], θq) if L >= 0 else 1e300
        dR = _wrap_abs_diff(th_sorted[R], θq) if R < n else 1e300
        if dL <= dR:
            ids_out[t] = ids_sorted[L]
            dists_out[t] = dL
            L -= 1
        else:
            ids_out[t] = ids_sorted[R]
            dists_out[t] = dR
            R += 1
        t += 1
    return ids_out, dists_out

@njit(parallel=True)
def _batch_radius_pack(th_sorted, th_ext, ids_sorted, ids_ext,
                       queries: np.ndarray, delta: float, inclusive: bool):
    """
    Batch radius queries with fixed delta.
    Returns (flat_ids, offsets) where:
      - flat_ids is concatenation of all result ID slices,
      - offsets has length len(queries)+1, with offsets[i]:start of query i.
    """
    Q = queries.size
    lengths = np.empty(Q, np.int64)

    # First pass: compute result sizes
    for qi in prange(Q):
        sl = _radius_query_once(th_sorted, th_ext, ids_sorted, ids_ext,
                                queries[qi], delta, inclusive)
        lengths[qi] = sl.size

    # Prefix sum
    offsets = np.empty(Q + 1, np.int64)
    offsets[0] = 0
    for i in range(Q):
        offsets[i+1] = offsets[i] + lengths[i]

    flat_ids = np.empty(offsets[-1], np.int64)

    # Second pass: copy results
    for qi in prange(Q):
        sl = _radius_query_once(th_sorted, th_ext, ids_sorted, ids_ext,
                                queries[qi], delta, inclusive)
        start = offsets[qi]
        flat_ids[start:start+sl.size] = sl

    return flat_ids, offsets

# ---------- Public class ----------

class CircularAngleIndexNumba:
    """
    θ-only index with Numba-accelerated queries.
    - radius_query(): Numba
    - knn_query(): Numba
    - batch_radius_query(): Numba (flat_ids, offsets)
    """

    def __init__(self, theta: Iterable[float], ids: Optional[Iterable[int]] = None):
        th = (np.asarray(theta, np.float64) % (2*np.pi))
        order = np.argsort(th)
        self.th_sorted = th[order].copy()
        self.n = self.th_sorted.size

        if ids is None:
            self.ids_sorted = order.astype(np.int64)
        else:
            ids_arr = np.asarray(ids, dtype=np.int64)
            self.ids_sorted = ids_arr[order]

        # Extended arrays for wrap-around queries
        self._th_ext = np.concatenate([self.th_sorted, self.th_sorted + 2*np.pi])
        self._ids_ext = np.concatenate([self.ids_sorted, self.ids_sorted])

    # ------- single queries -------
    def radius_query(self, theta_q: float, delta: float, *, inclusive: bool = True) -> np.ndarray:
        return _radius_query_once(self.th_sorted, self._th_ext, self.ids_sorted, self._ids_ext,
                                  float(theta_q), float(delta), inclusive).copy()

    def knn_query(self, theta_q: float, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        ids, dists = _knn_query_once(self.th_sorted, self.ids_sorted, float(theta_q), int(k))
        return ids.copy(), dists.copy()

    # ------- batch (fixed delta) -------
    def batch_radius_query(self, thetas_q: Iterable[float], delta: float, *, inclusive: bool = True):
        qs = np.asarray(thetas_q, np.float64)
        return _batch_radius_pack(self.th_sorted, self._th_ext, self.ids_sorted, self._ids_ext,
                                  qs, float(delta), inclusive)


"""
# Build
theta = np.random.uniform(0, 2*np.pi, size=200_000)
edge_ids = np.arange(theta.size, dtype=np.int64)
idx = CircularAngleIndexNumba(theta, ids=edge_ids)

# Single radius query (wrap-aware)
θq, δ = 6.25, 0.05
ids = idx.radius_query(θq, δ)           # np.ndarray of IDs

# k-NN on the circle
ids_k, dists_k = idx.knn_query(θq, k=8) # nearest 8 headings

# Batch radius queries with a fixed delta (fastest path)
Q = np.random.uniform(0, 2*np.pi, size=10_000)
flat_ids, offsets = idx.batch_radius_query(Q, delta=0.02)
# To iterate per-query results:
for i in range(Q.size):
    res_i = flat_ids[offsets[i]:offsets[i+1]]


    

th = [0.01, 0.03, 0.20, 6.20, 6.25, 6.27]    # still [0, 2π)
ids = [  0,   1,   2,   3,   4,   5]
idx = CircularAngleIndexNumba(th, ids=ids)

θq, δ = 6.25, 0.05
ids = idx.radius_query(θq, δ)           # np.ndarray of IDs



import sys
sys.path.append('./src')
import numpy as np
from edge_bundle import EdgeBundle

edge_bundle_file_location = 'edge_bundles/eb_unicycle_kinodynamic_TI_edges_100000.npz'
data = np.load(edge_bundle_file_location)
kd_TI_eb_unicycle = EdgeBundle(data, fix_num_edges=100000, use_all_edges=False)

from kd_tree_unicycle import CircularAngleIndexNumba
edge_ids = np.arange(kd_TI_eb_unicycle.num_edges, dtype=np.int64)
thetas = kd_TI_eb_unicycle.start_states[:, 2]  # heading angle θ
idx_TI_eb_unicycle = CircularAngleIndexNumba(thetas, ids=edge_ids)

θq, δ = 6.25, 0.05
ids = idx_TI_eb_unicycle.radius_query(θq, δ)           # np.ndarray of IDs


import time
θq, δ = 6.25, 0.05
start_t = time.perf_counter()
ids = idx_TI_eb_unicycle.radius_query(θq, δ)           # np.ndarray of IDs
end_t = time.perf_counter()
print(f"Time taken for radius query: {end_t - start_t:.6f} seconds")
print(f"radius_query took {(end_t - start_t)*1e6:.3f} µs")


import time
θq, δ = 1.2, 0.1

num_tries = 10000
θq_array = np.random.uniform(0.0, 2.0*np.pi, size=num_tries)
start = time.perf_counter()
for i in range(num_tries):
    idx_TI_eb_unicycle.radius_query(θq_array[i], δ)
end = time.perf_counter()

print(f"Average time: {(end - start)/num_tries*1e6:.2f} µs per call")


"""