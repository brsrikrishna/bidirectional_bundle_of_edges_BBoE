"""
Db-RRT-C — Bidirectional Db-RRT (idbrrt.pdf Section IV.D.a).

  Ortiz-Haro et al., "iDb-RRT: Sampling-based Kinodynamic Motion Planning
  with Motion Primitives and Trajectory Optimization", arXiv:2403.10745.

From Sec. IV.D.a:
  "Inspired by RRT-Connect, we present a bidirectional variant of Db-RRT,
   where we grow two trees, one from the start (using standard motion
   primitives) and one from the goal (using reversed motion primitives),
   and attempt to connect them. The expansion step in a backward search
   mirrors that of a forward search but requires reversing the order of
   states and controls in the motion primitives beforehand. The two trees
   are connected if two of their states are within the discontinuity
   bound."

Implementation:
  - Forward tree from start uses standard expansion (Db-RRT Alg 2/3/4 with
    forward dynamics, i.e. `agent.get_next_state(..., +duration)`).
  - Reverse tree from goal uses backward dynamics
    `agent.get_next_state(..., -duration)` — same mechanism BBoE uses for
    its rev tree. The bundle's action+duration is the same primitive;
    only the integration direction flips.
  - Each iteration: sample x_rand, attempt to extend BOTH trees toward
    x_rand. After each successful addition, search the OTHER tree's
    nearest node; if within δ_disc, the trees are connected.
  - Path reconstruction: walk forward tree from connection point
    F_c back to start; then walk reverse tree from connection point R_c
    up to its root (the goal). Because each reverse-tree edge stores the
    forward (action, duration) that — when applied to the rev-node's
    state — reaches the rev-node's parent (i.e. closer to goal), walking
    rev-tree UP from R_c gives the forward-time chain R_c → … → goal.
  - δ_disc semantics: in iDb-RRT this is the discontinuity bound that TO
    later repairs. We don't run TO; the returned path has up to one δ_disc
    seam at the F_c/R_c boundary. Cost is the sum of forward + reverse
    edge costs and ignores the seam.
"""
import networkx as nx
import numpy as np
import time
from utils import find_roundoff_decimal_digits, get_nearest_index
from numba.typed import List
from numba import types
from rrt import DynamicMatrix, TreeNode


class DbRRTCTreeNode(TreeNode):
    def __init__(self, sid, state, parent_id, parent_action,
                 parent_action_duration, path_from_parent,
                 time_so_far, cost):
        super().__init__(sid, state, parent_id, parent_action,
                         parent_action_duration, path_from_parent,
                         time_so_far, cost)
        self.edge_bundle_indices = None
        self.edge_bundle_mask = None


class DbRRTC:
    """Bidirectional Db-RRT (idbrrt.pdf §IV.D.a)."""

    def __init__(self, *, start, goal, goal_radius, env, agent,
                 sampling_time_step=1.0,
                 minimum_time_step=0.1,
                 max_iter=10000,
                 planning_time=10.0,
                 isvalid_function,
                 cost_function,
                 reached_goal_function,
                 random_point_function,
                 edge_bundle,
                 eb_kd_tree,
                 get_eb_kd_tree_query,
                 kd_tree_delta_radius=np.pi,
                 translate_function,
                 sort_edges_function,
                 max_num_edges_per_node=1000,
                 delta_disc=1.0,
                 goal_bias=0.05,
                 max_step_size=None,
                 udf_seed=77,
                 debug_flag=False,
                 print_logs=False,
                 dynamic_obstacles=None):

        self.start = np.array(start)
        self.goal = np.array(goal)
        self.goal_radius = goal_radius
        self.goal_check = self.goal
        self.env = env
        self.agent = agent
        self.max_sample_T = sampling_time_step
        self.minimum_time_step = minimum_time_step
        self.max_iter = max_iter
        self.planning_time = planning_time
        self.isvalid = isvalid_function
        self.cost = cost_function
        self.reached_goal = reached_goal_function
        self.get_random_point = random_point_function
        self.delta_disc = float(delta_disc)
        self.goal_bias = float(goal_bias)
        # max_step_size = None disables clamping (use x_rand as-is).
        self.max_step_size = (float(max_step_size)
                              if max_step_size is not None else float('inf'))

        self.edge_bundle = edge_bundle
        self.eb_kd_tree = eb_kd_tree
        self.get_eb_kd_tree_query = get_eb_kd_tree_query
        self.kd_tree_delta_radius = kd_tree_delta_radius
        self.translate_function = translate_function
        self.sort_edges = sort_edges_function
        self.max_num_edges_per_node = max_num_edges_per_node
        self.distance_array = np.zeros(
            (self.max_num_edges_per_node,), dtype=np.float64)

        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.debug_flag = debug_flag
        self.print_logs = print_logs

        self.static_circular_obstacles = self.env.static_circular_obstacles
        self.static_rectangular_obstacles = self.env.static_rectangular_obstacles
        self.dynamic_agent_obstacles = (
            List.empty_list(types.Array(types.float64, 2, "C"))
            if dynamic_obstacles is None else dynamic_obstacles)

        if hasattr(self.agent, 'distance_metric_state_size'):
            self.distance_metric_state_size = self.agent.distance_metric_state_size
        else:
            self.distance_metric_state_size = 2

        self._metric_scales = getattr(self.agent, 'metric_state_scales', None)
        self.roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)
        self.node_class = DbRRTCTreeNode

        # Trees, node matrices and "last added" counters per tree.
        self.tree_for = nx.DiGraph()
        self.tree_rev = nx.DiGraph()
        self._node_matrix_for = DynamicMatrix(
            initial_capacity=1024, dim=self.distance_metric_state_size)
        self._node_matrix_rev = DynamicMatrix(
            initial_capacity=1024, dim=self.distance_metric_state_size)
        self.last_added_node_id_for = -1
        self.last_added_node_id_rev = -1

        self.path_found = False
        self.path_cost = float('inf')
        self.path_time = 0.0
        self.connection_for_id = None
        self.connection_rev_id = None

    # ------------------------------------------------------------------ #
    #   Helpers                                                           #
    # ------------------------------------------------------------------ #
    def _metric_embed(self, state):
        if hasattr(self.agent, 'metric_embed_state'):
            return self.agent.metric_embed_state(state)
        s = state[:self.distance_metric_state_size]
        if self._metric_scales is not None:
            return s * self._metric_scales
        return s

    def reset_trees(self):
        self.tree_for = nx.DiGraph()
        self.tree_rev = nx.DiGraph()
        self._node_matrix_for = DynamicMatrix(
            initial_capacity=1024, dim=self.distance_metric_state_size)
        self._node_matrix_rev = DynamicMatrix(
            initial_capacity=1024, dim=self.distance_metric_state_size)
        self.last_added_node_id_for = -1
        self.last_added_node_id_rev = -1
        self.path_found = False
        self.path_cost = float('inf')
        self.path_time = 0.0
        self.connection_for_id = None
        self.connection_rev_id = None

    def _add_to_tree(self, is_forward, state, parent_id, parent_action,
                     parent_action_duration, path_from_parent,
                     time_elapsed, cost):
        if is_forward:
            new_id = self.last_added_node_id_for + 1
            self.last_added_node_id_for = new_id
            tree = self.tree_for
            mat = self._node_matrix_for
        else:
            new_id = self.last_added_node_id_rev + 1
            self.last_added_node_id_rev = new_id
            tree = self.tree_rev
            mat = self._node_matrix_rev
        node = self.node_class(
            new_id, state, parent_id, parent_action,
            parent_action_duration, path_from_parent,
            round(time_elapsed, self.roundoff_digits), cost)
        tree.add_node(new_id, value=node)
        mat.append(self._metric_embed(state), new_id)
        return new_id

    def sample_random_point(self):
        return self.get_random_point(
            self.env, self.static_circular_obstacles,
            self.static_rectangular_obstacles, self.rng)

    def _nearest_in_tree(self, query_state, is_forward):
        mat = self._node_matrix_for if is_forward else self._node_matrix_rev
        n = mat.count
        if n == 0:
            return None, None
        idx = get_nearest_index(mat.matrix, n, self._metric_embed(query_state))
        nid = int(mat.ids[idx])
        tree = self.tree_for if is_forward else self.tree_rev
        return nid, tree.nodes[nid]['value']

    def _nearest_distance(self, state, is_forward):
        mat = self._node_matrix_for if is_forward else self._node_matrix_rev
        n = mat.count
        if n == 0:
            return float('inf'), -1
        emb = self._metric_embed(state)
        diffs = mat.matrix[:n] - emb
        dists = np.linalg.norm(diffs, axis=1)
        best = int(dists.argmin())
        return float(dists[best]), int(mat.ids[best])

    # ------------------------------------------------------------------ #
    #   Propagation (forward or reversed)                                 #
    # ------------------------------------------------------------------ #
    def _propagate_edge(self, parent_node, edge_bundle_index, is_forward):
        eb = self.edge_bundle
        action = eb.actions[edge_bundle_index]
        duration = abs(eb.timesteps[edge_bundle_index])
        eval_time = duration if is_forward else -duration
        num_steps = max(1, round(duration / self.minimum_time_step))

        new_state, path = self.agent.get_next_state(
            parent_node.state, action, eval_time, num_steps=num_steps)

        valid = self.isvalid(
            path, self.agent.radius, self.env.size,
            self.static_circular_obstacles, self.static_rectangular_obstacles,
            self.dynamic_agent_obstacles, self.env.obstacle_buffer,
            self.env.boundary_buffer, parent_node.time_elapsed, duration,
            self.minimum_time_step)
        if not valid:
            return None
        return (new_state, path, action, duration)

    def _ensure_edge_cache(self, parent_node):
        if parent_node.edge_bundle_indices is None:
            query = self.get_eb_kd_tree_query(parent_node.state)
            edge_ids = self.eb_kd_tree.radius_query(
                query, self.kd_tree_delta_radius)
            l = min(len(edge_ids), self.max_num_edges_per_node)
            parent_node.edge_bundle_indices = edge_ids[:l]
            parent_node.edge_bundle_mask = np.full((l,), False, dtype=bool)

    def _expand_randomized(self, parent_node, is_forward):
        """Db-RRT Alg 4 — try untried primitives in random order, return
        first collision-free result. Used for both forward and reverse
        trees in this Db-RRT-C variant: it preserves the exploration
        diversity that's load-bearing in cluttered / narrow-passage envs
        (focused-everywhere regressed CWT and quadrotor empirically)."""
        self._ensure_edge_cache(parent_node)
        curr_indices = parent_node.edge_bundle_indices
        curr_mask = parent_node.edge_bundle_mask
        untried = np.where(~curr_mask)[0]
        if len(untried) == 0:
            return None
        order = self.rng.permutation(untried)
        for x in order:
            edge_bundle_index = curr_indices[x]
            curr_mask[int(x)] = True
            result = self._propagate_edge(
                parent_node, edge_bundle_index, is_forward)
            if result is not None:
                return result
        return None

    def _expand_focused(self, parent_node, x_target, is_forward):
        """Focused expansion — pick the untried primitive whose post-prop
        endpoint is closest to x_target.

        Forward tree: relies on the bundle's stored final_state under SE(2)
        translation invariance — `agent.sort_kd_tree_edges` returns
        untried primitives sorted by d_X to x_target, and we try them in
        sort order, returning the first collision-free one. This is the
        same fast path Db-RRT (forward) uses for goal-biased samples.

        Reverse tree: the bundle stores FORWARD final_states. With
        backward integration the actual endpoint differs in direction
        (we ran into this with BBoE's rev tree). So we propagate ALL
        untried primitives and pick the post-prop endpoint with min d_X
        to x_target. More expensive per iteration but mathematically
        correct under our integration semantics.
        """
        self._ensure_edge_cache(parent_node)
        eb = self.edge_bundle
        curr_indices = parent_node.edge_bundle_indices
        curr_mask = parent_node.edge_bundle_mask

        if is_forward:
            sorted_indices, num_valid = self.sort_edges(
                parent_node.state, x_target,
                eb.start_states, eb.final_states,
                curr_indices, curr_mask, self.distance_array)
            for idx in range(num_valid):
                x = sorted_indices[idx]
                edge_bundle_index = curr_indices[x]
                curr_mask[x] = True
                result = self._propagate_edge(
                    parent_node, edge_bundle_index, True)
                if result is not None:
                    return result
            return None

        # rev tree — propagate-first / pick-best-by-post-prop-d_X
        untried = np.where(~curr_mask)[0]
        if len(untried) == 0:
            return None
        x_target_embed = self._metric_embed(x_target)
        best_result = None
        best_dist = float('inf')
        for x in untried:
            edge_bundle_index = curr_indices[x]
            curr_mask[int(x)] = True
            result = self._propagate_edge(
                parent_node, edge_bundle_index, False)
            if result is None:
                continue
            new_state = result[0]
            d = float(np.linalg.norm(
                self._metric_embed(new_state) - x_target_embed))
            if d < best_dist:
                best_dist = d
                best_result = result
        return best_result

    def _pad_to_state(self, x):
        """Zero-pad an embed-space point to full state-vector shape so
        downstream code (clamping, sort_edges, _metric_embed) can operate
        on a uniform shape. If x is already full-state-shaped, returns it
        unchanged."""
        if x.shape == self.start.shape:
            return x
        full = np.zeros_like(self.start, dtype=np.float64)
        full[:len(x)] = x
        return full

    def _sample_target(self, a_is_forward):
        """Returns a x_rand vector in full state shape.

        With probability `goal_bias`, x_rand is taken from a random node
        of the OTHER tree (iDb-RRT reference's bidirectional bias —
        Sec. IV.D.a + the `dbrrtConnect` GitHub source). Otherwise from
        the agent's uniform-random sampler (which may return an embed-
        space point; we pad to full state for uniform handling)."""
        if self.rng.uniform(0, 1) < self.goal_bias:
            other_mat = (self._node_matrix_rev if a_is_forward
                         else self._node_matrix_for)
            other_tree = (self.tree_rev if a_is_forward else self.tree_for)
            n = other_mat.count
            if n > 0:
                idx = int(self.rng.integers(0, n))
                nid = int(other_mat.ids[idx])
                return other_tree.nodes[nid]['value'].state.astype(np.float64)
        return self._pad_to_state(self.sample_random_point())

    def _clamp_to_step(self, near_state, x_rand):
        """Linear interpolation toward x_rand from near_state in full
        state space, clipped at max_step_size in the d_X metric.
        Mirrors the reference's `robot->interpolate(...)`."""
        if not np.isfinite(self.max_step_size):
            return x_rand
        d_embed = float(np.linalg.norm(
            self._metric_embed(x_rand) - self._metric_embed(near_state)))
        if d_embed <= self.max_step_size:
            return x_rand
        frac = self.max_step_size / d_embed
        return near_state + frac * (x_rand - near_state)

    # ------------------------------------------------------------------ #
    #   Connection check                                                  #
    # ------------------------------------------------------------------ #
    def _try_connect(self, new_state, added_is_forward):
        """After adding `new_state` to one tree, check if the OTHER tree
        has a node within delta_disc. Returns the matching node id if so,
        else -1."""
        dist, nid = self._nearest_distance(
            new_state, is_forward=(not added_is_forward))
        if nid >= 0 and dist <= self.delta_disc:
            return nid
        return -1

    # ------------------------------------------------------------------ #
    #   Main loop                                                         #
    # ------------------------------------------------------------------ #
    def plan_path(self):
        self.reset_trees()
        # Root the trees
        self._add_to_tree(True, self.start, -1, None, None, None, 0.0, 0.0)
        self._add_to_tree(False, self.goal, -1, None, None, None, 0.0, 0.0)

        # Early-out: start already within δ of goal?
        d0, _ = self._nearest_distance(self.start, is_forward=False)
        if d0 <= self.delta_disc:
            self.path_found = True
            self.connection_for_id = 0
            self.connection_rev_id = 0
            self.path_cost = 0.0
            self.path_time = 0.0
            return

        curr_iter = 0
        start_time = time.time()

        # RRT-Connect-style: extend tree_a toward x_rand; after each
        # successful extension, check the OTHER tree for proximity.
        # Alternate roles so neither tree starves.
        a_is_forward = True

        while curr_iter <= self.max_iter:
            # iDb-RRT reference: with prob `goal_bias` sample from the
            # OTHER tree's nodes; otherwise uniform-random.
            x_rand = self._sample_target(a_is_forward)

            # Pick parent in tree A (the one we're extending this iter)
            _pid, parent_node = self._nearest_in_tree(
                x_rand, is_forward=a_is_forward)
            if parent_node is None:
                curr_iter += 1
                continue

            # Randomized expansion (Db-RRT Alg 4). Step-clamping and
            # focused-everywhere were tried and regressed CWT/quadrotor —
            # see comment in _expand_randomized.
            ext = self._expand_randomized(parent_node,
                                          is_forward=a_is_forward)
            if ext is not None:
                new_state, path, action, duration = ext
                # Cost = sum of edge costs along whichever tree (forward or
                # backward time is symmetric — duration is +ve, isvalid was
                # checked, path is a sequence of states).
                edge_cost = self.cost(
                    self.env, self.agent, parent_node.state,
                    action, duration, path)
                total_time = parent_node.time_elapsed + duration
                total_cost = parent_node.cost_so_far + edge_cost

                # Min-spacing rule per tree (Alg 2 line 14)
                nearest_same, _ = self._nearest_distance(
                    new_state, is_forward=a_is_forward)
                if nearest_same > self.delta_disc:
                    pid_in_tree_a = (self.last_added_node_id_for
                                     if a_is_forward
                                     else self.last_added_node_id_rev)
                    new_id = self._add_to_tree(
                        a_is_forward, new_state,
                        parent_id=parent_node.id,
                        parent_action=action,
                        parent_action_duration=duration,
                        path_from_parent=path,
                        time_elapsed=total_time,
                        cost=total_cost)

                    # Connection check against the OTHER tree
                    other_id = self._try_connect(new_state, a_is_forward)
                    if other_id >= 0:
                        if a_is_forward:
                            self.connection_for_id = new_id
                            self.connection_rev_id = other_id
                        else:
                            self.connection_for_id = other_id
                            self.connection_rev_id = new_id
                        self.path_found = True
                        self.path_cost = (
                            self.tree_for.nodes[self.connection_for_id]['value'].cost_so_far +
                            self.tree_rev.nodes[self.connection_rev_id]['value'].cost_so_far)
                        self.path_time = (
                            self.tree_for.nodes[self.connection_for_id]['value'].time_elapsed +
                            self.tree_rev.nodes[self.connection_rev_id]['value'].time_elapsed)
                        break

            # Swap tree roles for next iteration (RRT-Connect style)
            a_is_forward = not a_is_forward

            if time.time() - start_time >= self.planning_time:
                break
            curr_iter += 1

        total_run = time.time() - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)
        if self.print_logs or self.debug_flag:
            msg = "Db-RRT-C planning"
            if hasattr(self.agent, "id"):
                msg += f" agent {self.agent.id}"
            print(f"{msg}: {total_run:.2f}s, iters={curr_iter}, "
                  f"found={self.path_found}, cost={self.path_cost}")

    # ------------------------------------------------------------------ #
    #   Path extraction                                                   #
    # ------------------------------------------------------------------ #
    def _trace_back(self, tree, leaf_id):
        """Walk parents from leaf_id to root in `tree`. Returns (ids,
        states, controls, durations) where ids[0] = root, ids[-1] = leaf.
        controls[i] / durations[i] is the (action, duration) on edge
        (ids[i], ids[i+1])."""
        state_length = self.agent.state_length
        control_length = self.agent.action_length
        max_n = len(tree.nodes)
        ids = np.empty(max_n, dtype=np.int32)
        states = np.empty((max_n, state_length), dtype=np.float64)
        ctrls = np.empty((max_n, control_length), dtype=np.float64)
        durs = np.empty(max_n, dtype=np.float64)
        n = 0
        nid = leaf_id
        while nid != -1:
            node = tree.nodes[nid]['value']
            ids[n] = node.id
            states[n] = node.state
            ctrls[n] = (node.parent_action if node.parent_action is not None
                        else np.zeros(control_length))
            durs[n] = (node.parent_action_duration
                       if node.parent_action_duration is not None else 0.0)
            n += 1
            nid = node.parent_id
        return (ids[:n][::-1], states[:n][::-1],
                ctrls[:n][::-1], durs[:n][::-1])

    def get_path(self):
        """Bidirectional path: start → connection_for_id (forward tree),
        connection_rev_id → goal (reverse tree, traversed from connection
        up to root).

        For the reverse half: each rev-tree edge stores (action, duration)
        such that forward-applying from a child reaches its parent. So
        walking rev-tree from connection_rev_id UP to root (goal) yields
        the forward-time chain.
        """
        if not self.path_found:
            return (np.empty(0, dtype=np.int16),
                    np.empty((0, self.agent.state_length), dtype=np.float64),
                    np.empty((0, self.agent.action_length), dtype=np.float64),
                    np.empty(0, dtype=np.float64))

        # Forward half: ids/states ordered root→leaf, ctrls/durs are the
        # edges between consecutive states. Drop the last ctrl/dur (that's
        # the edge leading INTO leaf, which we've already included).
        ids_f, states_f, ctrls_f, durs_f = self._trace_back(
            self.tree_for, self.connection_for_id)
        # Reverse-tree traversal: walk from connection_rev_id UP to root.
        # _trace_back returns root→leaf order which is goal→connection.
        # We want connection→goal forward, so reverse it.
        _, states_r, ctrls_r, durs_r = self._trace_back(
            self.tree_rev, self.connection_rev_id)
        states_r = states_r[::-1]
        ctrls_r = ctrls_r[::-1]
        durs_r = durs_r[::-1]

        # ctrls_f[i] is the action that takes states_f[i-1] → states_f[i].
        # We want all forward-applied edges. Forward half contributes
        # ctrls_f[1:] (skip the dummy 0 at index 0). For rev half: rev tree
        # was built with parent_action recorded on each child. Walking up
        # from connection_rev_id back to goal: the action stored on each
        # rev_node (its `parent_action`) is what forward-takes that rev_node
        # to its parent. After reversing the order to be
        # connection→goal, ctrls_r[i] is the action from states_r[i] to
        # states_r[i+1]. So just append ctrls_r[:-1].
        states = np.concatenate([states_f, states_r[1:]], axis=0)
        controls = np.concatenate([ctrls_f[1:], ctrls_r[:-1]], axis=0)
        durations = np.concatenate([durs_f[1:], durs_r[:-1]], axis=0)
        ids = np.arange(len(states), dtype=np.int32)
        return ids, states, controls, durations
