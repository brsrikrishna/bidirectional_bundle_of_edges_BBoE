import numpy as np
from scipy.spatial import cKDTree


class VlVrTree:
    """
    KD-tree index for treaded vehicle tread-velocity state [v_l, v_r].

    Feature embedding:
        x0 = v_l / vl_scale
        x1 = v_r / vr_scale

    Euclidean distance in this 2D feature space corresponds to:
        d^2 = (dv_l / vl_scale)^2 + (dv_r / vr_scale)^2
    """

    def __init__(self, vl, vr, ids=None, vl_scale=1.0, vr_scale=1.0):
        vl = np.asarray(vl, dtype=np.float64)
        vr = np.asarray(vr, dtype=np.float64)
        assert vl.shape == vr.shape
        n = vl.shape[0]

        if ids is None:
            ids = np.arange(n, dtype=np.int64)
        else:
            ids = np.asarray(ids, dtype=np.int64)
            assert ids.shape == vl.shape

        self.ids = ids
        self.vl = vl
        self.vr = vr
        self.vl_scale = float(vl_scale)
        self.vr_scale = float(vr_scale)

        X = np.empty((n, 2), dtype=np.float64)
        X[:, 0] = vl / self.vl_scale
        X[:, 1] = vr / self.vr_scale
        self._X = X
        self.tree = cKDTree(X)

        self._query_buf = np.empty(2, dtype=np.float64)

    def _embed_query(self, query):
        x = self._query_buf
        x[0] = float(query[0]) / self.vl_scale
        x[1] = float(query[1]) / self.vr_scale
        return x

    def radius_query(self, query, delta):
        q = self._embed_query(query)
        cand_idx = self.tree.query_ball_point(q, r=delta)
        return np.asarray(cand_idx, dtype=np.int64)
