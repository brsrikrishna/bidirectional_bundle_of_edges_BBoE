import numpy as np
from scipy.spatial import cKDTree


class VOmegaTheta1Tree:
    """
    KD-tree index for car-with-trailer non-position state: [v, omega, theta1].

    The edge bundle is translation-invariant in (x, y) and rotation-invariant
    in theta (edges start at theta=0, translated by rotating).
    The remaining state dimensions (v, omega, theta1) determine edge compatibility.

    Feature embedding:
        x0 = v      / v_scale
        x1 = omega  / omega_scale
        x2 = theta1 / theta1_scale

    theta1 is angular and lives in [-pi, pi]. To make wrap-around neighbors
    findable by Euclidean cKDTree.query_ball_point, each bundle row is
    triplicated at theta1, theta1 - 2*pi, theta1 + 2*pi (analogous to the
    unicycle's CircularAngleIndexNumba). Returned indices map back to the
    original row id via `_id_map`.
    """

    def __init__(self, v, omega, theta1, ids=None,
                 v_scale=1.0, omega_scale=1.0, theta1_scale=1.0):
        v = np.asarray(v, dtype=np.float64)
        omega = np.asarray(omega, dtype=np.float64)
        theta1 = np.asarray(theta1, dtype=np.float64)
        assert v.shape == omega.shape == theta1.shape
        n = v.shape[0]

        if ids is None:
            ids = np.arange(n, dtype=np.int64)
        else:
            ids = np.asarray(ids, dtype=np.int64)
            assert ids.shape == v.shape

        self.ids = ids
        self.v_scale = float(v_scale)
        self.omega_scale = float(omega_scale)
        self.theta1_scale = float(theta1_scale)

        # Triplicate each row at theta1 + {-2pi, 0, +2pi} so wrap-around
        # angular neighbours are reachable by ordinary Euclidean radius query.
        two_pi = 2.0 * np.pi
        offsets = np.array([-two_pi, 0.0, two_pi], dtype=np.float64)
        X = np.empty((n * 3, 3), dtype=np.float64)
        for k, off in enumerate(offsets):
            sl = slice(k * n, (k + 1) * n)
            X[sl, 0] = v / self.v_scale
            X[sl, 1] = omega / self.omega_scale
            X[sl, 2] = (theta1 + off) / self.theta1_scale
        # Map padded-row index back to canonical row id.
        self._id_map = np.tile(ids, 3)
        self._X = X
        self.tree = cKDTree(X)

        self._query_buf = np.empty(3, dtype=np.float64)

    def _embed_query(self, query):
        # Wrap query theta1 to [-pi, pi] so it sits inside the central tile
        # of the triplicated tree.
        two_pi = 2.0 * np.pi
        theta1 = ((float(query[2]) + np.pi) % two_pi) - np.pi
        x = self._query_buf
        x[0] = float(query[0]) / self.v_scale
        x[1] = float(query[1]) / self.omega_scale
        x[2] = theta1 / self.theta1_scale
        return x

    def radius_query(self, query, delta):
        q = self._embed_query(query)
        cand_idx = self.tree.query_ball_point(q, r=delta)
        if not cand_idx:
            return np.empty(0, dtype=np.int64)
        # Map padded indices back to canonical row ids and de-duplicate (a
        # query near the wrap seam can hit two padded copies of the same row).
        canonical = self._id_map[np.asarray(cand_idx, dtype=np.int64)]
        return np.unique(canonical)
