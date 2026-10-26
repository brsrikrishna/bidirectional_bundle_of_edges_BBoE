"""
Db-RRT — RRT with motion primitives and bounded discontinuity.

Implements paper Algorithm 2 (Db-RRT) from:
  Ortiz-Haro et al., "iDb-RRT: Sampling-based Kinodynamic Motion Planning
  with Motion Primitives and Trajectory Optimization", arXiv:2403.10745, 2024.

This module is the SEARCH component only. The paper's full iDb-RRT
algorithm (Alg 1) wraps Db-RRT in an outer iterative loop and adds a
trajectory-optimization step (Sec IV.E, Feasibility-driven DDP). We
implement neither — this is plain Db-RRT.

What we cover:
  - Alg 2: the Db-RRT main loop (sample, nearest, expand, min-spacing
    check, goal check).
  - Alg 3: focused expansion — sort neighbour primitives by translated
    end-state distance to x_target, try in order, return first
    collision-free. Used when x_rand is the goal (goal-biased samples).
  - Alg 4: randomized expansion — try neighbour primitives in random
    order, return first collision-free. Used when x_rand is a random
    sample.
  - Min-spacing rule (Alg 2 line 14): only add x_new if no existing
    tree node lies within `delta_disc`.

What's different from the paper's `Apply`:
  Paper's expand step (Algs 3-4 line 4) checks the primitive's STORED
  trajectory in collision and uses the primitive's stored end-state
  `m.x_f` as the new node, accepting up to a δ-bounded discontinuity at
  the seam between live state x_o and primitive start m.x_s. We
  re-integrate dynamics online from x_o using the primitive's
  (action, duration) — equivalent to paper Db-RRT with δ_disc → 0 (no
  seam discontinuity by construction). For systems with full state-
  translation invariance (our unicycle), this is mathematically identical
  to paper-faithful translated `Apply`. For non-invariant systems
  (treaded, CWT, quadrotor) it produces a different
  trajectory than the paper would; the δ-disc framework that motivates
  iDb-RRT's TO step is therefore not exercised on those systems.

Why we don't implement TO + the outer loop:
  Trajectory optimization (Alg 1 line 6) requires a per-system DDP
  solver respecting dynamics + obstacles + control bounds — substantial
  engineering work, deferred. Without TO the outer loop's
  optimization-fail branch never fires, and the search-fail branch's
  primitive-increase step is moot in our setup because we always run
  with the full precomputed library M_L (no subsetting).
"""
import networkx as nx
import numpy as np
import time
from utils import find_roundoff_decimal_digits, get_nearest_index
from numba.typed import List
from numba import types
from rrt import DynamicMatrix, TreeNode


class DbRRTTreeNode(TreeNode):
    def __init__(self, sid, state, parent_id, parent_action,
                 parent_action_duration, path_from_parent,
                 time_so_far, cost):
        super().__init__(sid, state, parent_id, parent_action,
                         parent_action_duration, path_from_parent,
                         time_so_far, cost)
        self.edge_bundle_indices = None
        self.edge_bundle_mask = None


class DbRRT:
    """Forward Db-RRT planner with motion primitives and a min-spacing rule.

    Implements paper Alg 2 (Db-RRT). See module docstring for what's
    included and what's deliberately not (Alg 1 outer loop, TO step,
    primitive subsetting).
    """

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
                 udf_seed=77,
                 debug_flag=False,
                 print_logs=False,
                 dynamic_obstacles=None):

        self.start = np.array(start)
        self.goal = np.array(goal)
        self.goal_radius = goal_radius
        self.goal_state_length = len(goal)
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
        self.goal_bias = float(goal_bias)
        self.delta_disc = float(delta_disc)

        self.edge_bundle = edge_bundle
        self.eb_kd_tree = eb_kd_tree
        self.get_eb_kd_tree_query = get_eb_kd_tree_query
        self.kd_tree_delta_radius = kd_tree_delta_radius
        self.translate_function = translate_function
        self.sort_edges = sort_edges_function
        self.max_num_edges_per_node = max_num_edges_per_node
        self.distance_array = np.zeros(
            (self.max_num_edges_per_node,), dtype=np.float64)

        self.tree = nx.DiGraph()
        self.goal_node_id = None
        self.path_found = False
        self.path_time = 0.0
        self.path_cost = float('inf')
        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.debug_flag = debug_flag
        self.print_logs = print_logs

        self.last_added_node_id = -1

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
        self._node_matrix = DynamicMatrix(
            initial_capacity=1024, dim=self.distance_metric_state_size)

        self.roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)
        self.node_class = DbRRTTreeNode

        self.threshold = 3.0

    # -------------------------------------------------------------- #
    #   Helpers                                                        #
    # -------------------------------------------------------------- #
    def _metric_embed(self, state):
        if hasattr(self.agent, 'metric_embed_state'):
            return self.agent.metric_embed_state(state)
        s = state[:self.distance_metric_state_size]
        if self._metric_scales is not None:
            return s * self._metric_scales
        return s

    def reset_tree(self):
        self.tree = nx.DiGraph()
        self._node_matrix = DynamicMatrix(
            initial_capacity=1024, dim=self.distance_metric_state_size)
        self.goal_node_id = None
        self.path_found = False
        self.path_cost = float('inf')
        self.path_time = 0.0
        self.last_added_node_id = -1

    def add_node(self, state, parent_id, parent_action,
                 parent_action_duration, path_from_parent,
                 time_elapsed, cost):
        new_id = self.last_added_node_id + 1
        node = self.node_class(
            new_id, state, parent_id, parent_action,
            parent_action_duration, path_from_parent,
            round(time_elapsed, self.roundoff_digits), cost)
        self.tree.add_node(new_id, value=node)
        self.last_added_node_id = new_id
        self._node_matrix.append(self._metric_embed(state), new_id)
        return new_id

    def sample_random_point(self):
        return self.get_random_point(self.env, self.static_circular_obstacles,
                                     self.static_rectangular_obstacles, self.rng)

    def get_nearest_node(self, query_point):
        states = self._node_matrix.get_valid_matrix()
        n = self._node_matrix.count
        if n == 0:
            return None, None
        idx = get_nearest_index(states, n, self._metric_embed(query_point))
        nid = self._node_matrix.ids[idx]
        return nid, self.tree.nodes[nid]['value']

    def nearest_distance(self, state):
        """Min distance from `state` to any existing tree node (paper's d_M)."""
        states = self._node_matrix.get_valid_matrix()
        n = self._node_matrix.count
        if n == 0:
            return float('inf')
        emb = self._metric_embed(state)
        diffs = states[:n] - emb
        dists = np.linalg.norm(diffs, axis=1)
        return float(dists.min())

    # -------------------------------------------------------------- #
    #   Edge-bundle propagation                                       #
    # -------------------------------------------------------------- #
    def _propagate_edge(self, parent_node, edge_bundle_index):
        """Re-propagate a bundle edge from parent_node.state. Returns
        (new_state, path, action, duration) on success, None on collision."""
        eb = self.edge_bundle
        action = eb.actions[edge_bundle_index]
        duration = abs(eb.timesteps[edge_bundle_index])
        num_steps = max(1, round(duration / self.minimum_time_step))

        new_state, path = self.agent.get_next_state(
            parent_node.state, action, duration, num_steps=num_steps)

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
            edge_ids = self.eb_kd_tree.radius_query(query, self.kd_tree_delta_radius)
            l = min(len(edge_ids), self.max_num_edges_per_node)
            parent_node.edge_bundle_indices = edge_ids[:l]
            parent_node.edge_bundle_mask = np.full((l,), False, dtype=bool)

    # -------------------------------------------------------------- #
    #   Algorithm 3 — Focused expansion                                #
    # -------------------------------------------------------------- #
    def _expand_focused(self, parent_node, x_target):
        """Sort neighbour primitives by distance-to-target, try in order."""
        self._ensure_edge_cache(parent_node)
        eb = self.edge_bundle
        curr_indices = parent_node.edge_bundle_indices
        curr_mask = parent_node.edge_bundle_mask

        sorted_indices, num_valid = self.sort_edges(
            parent_node.state, x_target,
            eb.start_states, eb.final_states,
            curr_indices, curr_mask, self.distance_array)

        for idx in range(num_valid):
            x = sorted_indices[idx]
            edge_bundle_index = curr_indices[x]
            curr_mask[x] = True
            result = self._propagate_edge(parent_node, edge_bundle_index)
            if result is not None:
                return result
        return None

    # -------------------------------------------------------------- #
    #   Algorithm 4 — Randomized expansion                             #
    # -------------------------------------------------------------- #
    def _expand_randomized(self, parent_node):
        """Try neighbour primitives in random order, return first
        collision-free result."""
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
            result = self._propagate_edge(parent_node, edge_bundle_index)
            if result is not None:
                return result
        return None

    # -------------------------------------------------------------- #
    #   Algorithm 2 — Db-RRT main loop                                 #
    # -------------------------------------------------------------- #
    def plan_path(self):
        self.reset_tree()
        self.add_node(self.start, -1, None, None, None, 0.0, 0.0)

        curr_iter = 0
        start_time = time.time()

        while curr_iter <= self.max_iter:
            # 1) Sample a target
            if self.rng.uniform(0, 1) < self.goal_bias:
                x_rand = self.goal
                use_focused = True
            else:
                x_rand = self.sample_random_point()
                use_focused = False

            # 2) Nearest existing node
            parent_id, parent_node = self.get_nearest_node(x_rand)
            if parent_node is None:
                curr_iter += 1
                continue

            # 3) Expand: focused for goal targets, randomized for samples
            if use_focused:
                ext = self._expand_focused(parent_node, x_rand)
            else:
                ext = self._expand_randomized(parent_node)

            if ext is not None:
                new_state, path, action, duration = ext

                # 4a) Goal check at the propagated endpoint
                reached, goal_distance = self.reached_goal(
                    new_state, self.goal_check, self.goal_radius, self.agent)

                if reached:
                    edge_cost = self.cost(
                        self.env, self.agent, parent_node.state,
                        action, duration, path)
                    total_time = parent_node.time_elapsed + duration
                    total_cost = parent_node.cost_so_far + edge_cost
                    new_id = self.add_node(
                        new_state, parent_id, action, duration,
                        path, total_time, total_cost)
                    self.path_found = True
                    self.goal_node_id = new_id
                    self.path_cost = total_cost
                    self.path_time = total_time
                    break

                # 4b) Walk-along-edge check (mirrors RRT/SST/GBRRT/BBoE).
                # When the propagated endpoint is close to goal but
                # didn't quite reach it, scan the trajectory's
                # intermediate waypoints — accepting an earlier waypoint
                # if it falls inside the goal ball. Truncates the edge
                # there. Gated by `goal_distance < self.threshold` to
                # avoid scanning every edge.
                if goal_distance < self.threshold:
                    intermediate_time = parent_node.time_elapsed
                    for index, intermediate_state in enumerate(path):
                        intermediate_time += self.minimum_time_step
                        goal_flag, _ = self.reached_goal(
                            intermediate_state, self.goal_check,
                            self.goal_radius, self.agent)
                        if goal_flag:
                            modified_edge_time = (
                                intermediate_time - parent_node.time_elapsed)
                            new_path = path[:index + 1]
                            edge_cost = self.cost(
                                self.env, self.agent, parent_node.state,
                                action, modified_edge_time, new_path)
                            total_cost = parent_node.cost_so_far + edge_cost
                            new_id = self.add_node(
                                intermediate_state, parent_id, action,
                                modified_edge_time, new_path,
                                intermediate_time, total_cost)
                            self.path_found = True
                            self.goal_node_id = new_id
                            self.path_cost = total_cost
                            self.path_time = intermediate_time
                            break
                    if self.path_found:
                        break  # exit the outer planning loop

                # 5) Min-spacing check (Alg 2 line 14): only add far-enough nodes
                edge_cost = self.cost(
                    self.env, self.agent, parent_node.state,
                    action, duration, path)
                total_time = parent_node.time_elapsed + duration
                total_cost = parent_node.cost_so_far + edge_cost
                if self.nearest_distance(new_state) > self.delta_disc:
                    self.add_node(
                        new_state, parent_id, action, duration,
                        path, total_time, total_cost)

            if time.time() - start_time >= self.planning_time:
                break
            curr_iter += 1

        total_run = time.time() - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)
        if self.print_logs or self.debug_flag:
            msg = "Db-RRT planning"
            if hasattr(self.agent, "id"):
                msg += f" agent {self.agent.id}"
            print(f"{msg}: {total_run:.2f}s, iters={curr_iter}, "
                  f"found={self.path_found}, cost={self.path_cost}")

    # -------------------------------------------------------------- #
    #   Path extraction                                                #
    # -------------------------------------------------------------- #
    def get_path_to_node_id(self, goal_node_id):
        state_length = self.agent.state_length
        control_length = self.agent.action_length

        max_nodes = len(self.tree.nodes)
        ids = np.empty(max_nodes, dtype=np.int32)
        states = np.empty((max_nodes, state_length), dtype=np.float64)
        controls = np.empty((max_nodes, control_length), dtype=np.float64)
        timesteps = np.empty(max_nodes, dtype=np.float64)
        node_id = goal_node_id
        path_length = 0
        while node_id != -1:
            node = self.tree.nodes[node_id]['value']
            ids[path_length] = node.id
            states[path_length] = node.state
            controls[path_length] = (node.parent_action
                                      if node.parent_action is not None
                                      else np.zeros(control_length))
            timesteps[path_length] = (node.parent_action_duration
                                       if node.parent_action_duration is not None
                                       else 0.0)
            path_length += 1
            node_id = node.parent_id
        ids = ids[:path_length][::-1]
        states = states[:path_length][::-1]
        controls = controls[:path_length-1][::-1]
        timesteps = timesteps[:path_length-1][::-1]
        return ids, states, controls, timesteps

    def get_path(self):
        if not self.path_found:
            return (np.empty(0, dtype=np.int16),
                    np.empty((0, self.agent.state_length), dtype=np.float64),
                    np.empty((0, self.agent.action_length), dtype=np.float64),
                    np.empty(0, dtype=np.float64))
        return self.get_path_to_node_id(self.goal_node_id)
