import numpy as np
from scipy.spatial import cKDTree

class VxyzTree:
    """
    KD-tree index over velocities [vx, vy, vz].

    Embedding:
        X = [vx/sx, vy/sy, vz/sz]

    Radius query with r=delta returns all points satisfying:
        (Δvx/sx)^2 + (Δvy/sy)^2 + (Δvz/sz)^2 <= delta^2
    """

    def __init__(self, vx, vy, vz, ids=None, scales=(1.0, 1.0, 1.0)):
        vx = np.asarray(vx, dtype=np.float64)
        vy = np.asarray(vy, dtype=np.float64)
        vz = np.asarray(vz, dtype=np.float64)
        assert vx.shape == vy.shape == vz.shape
        n = vx.size

        if ids is None:
            ids = np.arange(n, dtype=np.int64)
        else:
            ids = np.asarray(ids, dtype=np.int64)
            assert ids.shape == vx.shape

        sx, sy, sz = map(float, scales)
        if sx <= 0 or sy <= 0 or sz <= 0:
            raise ValueError("All scales must be > 0.")

        self.ids = ids
        self.vx_scale = sx
        self.vy_scale = sy
        self.vz_scale = sz

        X = np.empty((n, 3), dtype=np.float64)
        X[:, 0] = vx / sx
        X[:, 1] = vy / sy
        X[:, 2] = vz / sz

        self._X = X
        self._vx = vx
        self._vy = vy
        self._vz = vz
        self.tree = cKDTree(X)

        self._query_buf = np.empty(3, dtype=np.float64)
        self._empty_idx = np.empty(0, dtype=np.int64)

        # Preallocate a big buffer to hold radius query results
        # self._candidate_indices_buffer = np.empty(n, dtype=np.int64)

    def _embed_query(self, query):
        x = self._query_buf
        x[0] = float(query[0]) / self.vx_scale
        x[1] = float(query[1]) / self.vy_scale
        x[2] = float(query[2]) / self.vz_scale
        return x

    def radius_query(self, query, delta):
        """Return IDs within radius delta in the scaled velocity space."""
        q = self._embed_query(query)
        cand_idx = self.tree.query_ball_point(q, r=delta)

        # if not cand:
        #     return self.ids[0:0]
        # return self.ids[np.asarray(cand, dtype=np.int64)]

        # return self.ids[np.asarray(cand_idx, dtype=np.int64)]
        return np.asarray(cand_idx, dtype=np.int64)
        #Note - this only returns the indices in the kd-tree, not the original IDs
        #However, in our usage, the indices in the kd-tree correspond to the original IDs.
        #So it works for now. But be careful if you use this class elsewhere.


    def knn_query(self, query, k=1):
        """Return k nearest IDs and their distances (in scaled space)."""
        q = self._embed_query(query)
        dist, idx = self.tree.query(q, k=int(k))
        idx = np.atleast_1d(idx)
        dist = np.atleast_1d(dist)
        return self.ids[idx], dist



"""

import sys
sys.path.append('./src')
import numpy as np
import time
from edge_bundle import EdgeBundle
from kd_tree_quadcopter6d import VxyzTree   # adjust import path if needed


N = 100_000
max_speed = 0.5 # m/s
# Example velocity ranges (adjust if your system differs)
vx = np.random.uniform(-max_speed, max_speed, size=N)
vy = np.random.uniform(-max_speed, max_speed, size=N)
vz = np.random.uniform(-max_speed, max_speed, size=N)
edge_ids = np.arange(N, dtype=np.int64)

kd_tree_quadcopter6d = VxyzTree(vx, vy, vz,ids=edge_ids,
                        scales=(1.0, 1.0, 1.0)   # pure Euclidean in m/s
                        )

# Single query sanity check
vxq, vyq, vzq, δ = 0.5, -0.5, 0.5, 0.1
ids = kd_tree_quadcopter6d.radius_query([vxq, vyq, vzq], δ)
print("Query:", vxq, vyq, vzq, "delta =", δ)
print("Found", len(ids), "neighbors\n")

for i in range(min(len(ids), 10)):  # print only a few
    idx = ids[i]
    print(f"ID: {idx}, "
        f"vx: {vx[idx]:.4f}, "
        f"vy: {vy[idx]:.4f}, "
        f"vz: {vz[idx]:.4f}"
    )

# Single-call timing
t0 = time.perf_counter()
near_ids = kd_tree_quadcopter6d.radius_query([vxq, vyq, vzq], δ)
t1 = time.perf_counter()
print(
    f"\nSingle radius_query: {(t1 - t0)*1e6:.2f} µs, "
    f"found {near_ids.size} neighbors")

# Average timing over many calls
δ = 0.1
num_tries = 10_000
vxq_array = np.random.uniform(-max_speed, max_speed, size=num_tries)
vyq_array = np.random.uniform(-max_speed, max_speed, size=num_tries)
vzq_array = np.random.uniform(-max_speed, max_speed, size=num_tries)
start = time.perf_counter()
for i in range(num_tries):
    q = np.array([vxq_array[i], vyq_array[i], vzq_array[i]])
    kd_tree_quadcopter6d.radius_query(q,δ)
end = time.perf_counter()

print(f"Average time: {(end - start)/num_tries*1e6:.2f} µs per call")



"""