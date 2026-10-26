import networkx as nx
import numpy as np
import time
from utils import find_roundoff_decimal_digits, get_nearest_index
from numba.typed import List
from numba import types
from rrt import DynamicMatrix, TreeNode
from gbrrt import PriorityQueue


class BBoETreeNode:
    def __init__(self, sid, state, parent_id, parent_action, parent_action_duration,
                 path_from_parent, time_so_far, cost):
        self.id = sid
        self.state = state
        self.parent_id = parent_id
        self.parent_action = parent_action
        self.parent_action_duration = parent_action_duration
        self.path_from_parent = path_from_parent
        self.time_elapsed = time_so_far
        self.cost_so_far = cost
        self.edge_bundle_indices = None
        self.edge_bundle_mask = None


class BBoE:
    """
    Bidirectional Bundle of Edges (BBoE) planner.

    Uses precomputed edge bundles with a biased/random edge selection strategy
    in a bidirectional RRT framework (GBRRT-style).

    Reference: BBoE: Leveraging Bundle of Edges for Kinodynamic
               Bidirectional Motion Planning (Raghu & Roncone)
    """

    def __init__(self, *, start, goal, goal_radius, env, agent,
                 sampling_time_step=1.0,
                 minimum_time_step=0.1,
                 max_iter=1000,
                 planning_time=10.0,
                 isvalid_function,
                 cost_function,
                 reached_goal_function,
                 random_point_function,
                 edge_bundle,
                 eb_kd_tree,
                 get_eb_kd_tree_query,
                 kd_tree_delta_radius=0.5,
                 translate_function,
                 sort_edges_function,
                 exploitation_ratio=0.8,
                 heuristic_radius=50.0,
                 gamma=50.0,
                 delta_reach=0.0,
                 k_bias=0.85,
                 num_skip_edges=10,
                 goal_bias=0.0,
                 udf_seed=77,
                 debug_flag=False,
                 print_logs=False,
                 dynamic_obstacles=None,
                 ):

        self.start = np.array(start)
        self.goal = np.array(goal)
        self.goal_radius = goal_radius
        self.goal_state_length = len(goal)
        # Goal check passes the full goal vector; each agent's reached_goal
        # implements its own paper-faithful d_X (treaded uses x,y + weighted vL,vR).
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
        self.exploitation_ratio = exploitation_ratio
        self.heuristic_radius = heuristic_radius
        self.gamma = gamma
        # Fix B: rev nodes within delta_reach of any forward node are
        # excluded from future exploit targeting — forces forward progress
        # up the reverse-tree chain toward goal.
        self.delta_reach = float(delta_reach)
        self.consumed_rev_ids = set()
        self.k_bias = k_bias
        self.num_skip_edges = num_skip_edges  # N from paper
        # goal_bias: prob each iteration that the forward tree targets the
        # global goal directly, and the reverse tree targets the global start.
        # Mirrors the standard goal-bias trick in unidirectional RRT.
        self.goal_bias = float(goal_bias)

        self.edge_bundle = edge_bundle
        self.eb_kd_tree = eb_kd_tree
        self.get_eb_kd_tree_query = get_eb_kd_tree_query
        self.kd_tree_delta_radius = kd_tree_delta_radius
        self.translate_function = translate_function
        self.sort_edges = sort_edges_function

        self.tree_for = nx.DiGraph()
        self.tree_rev = nx.DiGraph()
        self.goal_node_id = None
        self.path_found = False
        self.path_time = 0.0
        self.path_cost = float('inf')
        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.debug_flag = debug_flag
        self.print_logs = print_logs

        self.last_added_node_id_for = -1
        self.last_added_node_id_rev = -1
        self.threshold = 3.0

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

        self._node_matrix_for = DynamicMatrix(initial_capacity=1024, dim=self.distance_metric_state_size)
        self._node_matrix_rev = DynamicMatrix(initial_capacity=1024, dim=self.distance_metric_state_size)
        self.q_priority = PriorityQueue()

        self.roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)
        self.node_class = BBoETreeNode

        # Pre-allocate distance array for edge sorting
        self.distance_array = np.zeros(self.edge_bundle.num_edges, dtype=np.float64)

    def _metric_embed(self, state):
        """Project a full state into paper's d_X space (also used as d_M for
        tree NN). If the agent defines `metric_embed_state`, use it —
        otherwise fall back to slice + optional scaling."""
        if hasattr(self.agent, 'metric_embed_state'):
            return self.agent.metric_embed_state(state)
        s = state[:self.distance_metric_state_size]
        if self._metric_scales is not None:
            return s * self._metric_scales
        return s

    # ------------------------------------------------------------------ #
    #  Helpers (shared with GBRRT)                                        #
    # ------------------------------------------------------------------ #

    def get_rk(self):
        v_rev = self._node_matrix_rev.count
        if v_rev <= 1:
            return self.heuristic_radius
        d = len(self.start)
        val = self.gamma * ((np.log(v_rev) / v_rev) ** (1.0 / (d + 1.0)))
        return min(val, self.heuristic_radius)

    def reset_tree(self):
        self.tree_for = nx.DiGraph()
        self.tree_rev = nx.DiGraph()
        self._node_matrix_for = DynamicMatrix(initial_capacity=1024, dim=self.distance_metric_state_size)
        self._node_matrix_rev = DynamicMatrix(initial_capacity=1024, dim=self.distance_metric_state_size)
        self.q_priority = PriorityQueue()
        self.goal_node_id = None
        self.path_found = False
        self.path_cost = float('inf')
        self.path_time = 0.0
        self.last_added_node_id_for = -1
        self.last_added_node_id_rev = -1
        self.consumed_rev_ids = set()

    def _mark_consumed_rev_near(self, for_state):
        """Fix B: mark rev nodes within delta_reach of this forward state as
        consumed — excluded from all future exploit targeting."""
        states_rev = self._node_matrix_rev.get_valid_matrix()
        if states_rev.shape[0] == 0:
            return
        for_embed = self._metric_embed(for_state)
        dists = np.linalg.norm(states_rev - for_embed, axis=1)
        within = np.where(dists <= self.delta_reach)[0]
        for i in within:
            self.consumed_rev_ids.add(int(self._node_matrix_rev.ids[i]))

    def add_node_for(self, state, parent_node_id, parent_action, parent_action_duration,
                     path_from_parent, time_elapsed, cost):
        new_node_id = self.last_added_node_id_for + 1
        new_node = self.node_class(new_node_id, state, parent_node_id, parent_action,
                                   parent_action_duration, path_from_parent,
                                   round(time_elapsed, self.roundoff_digits), cost)
        self.tree_for.add_node(new_node_id, value=new_node)
        self.last_added_node_id_for = new_node_id
        self._node_matrix_for.append(self._metric_embed(state), new_node_id)
        return new_node_id

    def add_node_rev(self, state, parent_node_id, parent_action, parent_action_duration,
                     path_from_parent, time_elapsed, cost):
        new_node_id = self.last_added_node_id_rev + 1
        new_node = self.node_class(new_node_id, state, parent_node_id, parent_action,
                                   parent_action_duration, path_from_parent,
                                   round(time_elapsed, self.roundoff_digits), cost)
        self.tree_rev.add_node(new_node_id, value=new_node)
        self.last_added_node_id_rev = new_node_id
        self._node_matrix_rev.append(self._metric_embed(state), new_node_id)
        return new_node_id

    def sample_random_point(self):
        return self.get_random_point(self.env, self.static_circular_obstacles,
                                     self.static_rectangular_obstacles, self.rng)

    def get_nearest_node_for(self, random_point):
        states = self._node_matrix_for.get_valid_matrix()
        num_entries = self._node_matrix_for.count
        if num_entries == 0:
            return None, None
        nearest_index = get_nearest_index(states, num_entries,
                                          self._metric_embed(random_point))
        nearest_node_id = self._node_matrix_for.ids[nearest_index]
        return nearest_node_id, self.tree_for.nodes[nearest_node_id]['value']

    def get_nearest_node_rev(self, random_point):
        states = self._node_matrix_rev.get_valid_matrix()
        num_entries = self._node_matrix_rev.count
        if num_entries == 0:
            return None, None
        nearest_index = get_nearest_index(states, num_entries,
                                          self._metric_embed(random_point))
        nearest_node_id = self._node_matrix_rev.ids[nearest_index]
        return nearest_node_id, self.tree_rev.nodes[nearest_node_id]['value']

    def update_priority_queue(self, rev_node, rev_node_id):
        """Algorithm 1, line 14: updatePriorityQueue — find nearest forward node to new reverse node."""
        states_for = self._node_matrix_for.get_valid_matrix()
        num_entries = self._node_matrix_for.count
        if num_entries == 0:
            return
        rev_pos = self._metric_embed(rev_node.state)
        nearest_index = get_nearest_index(states_for, num_entries, rev_pos)
        nearest_for_id = self._node_matrix_for.ids[nearest_index]
        dist = np.linalg.norm(states_for[nearest_index] - rev_pos)
        if dist <= self.get_rk():
            score = dist + rev_node.cost_so_far
            self.q_priority.add_or_update(nearest_for_id, score)

    def insert_priority_queue(self, for_node, for_node_id):
        """Algorithm 1, line 38: InsertToPriorityQueue — find nearest reverse node to new forward node."""
        states_rev = self._node_matrix_rev.get_valid_matrix()
        num_entries = self._node_matrix_rev.count
        if num_entries == 0:
            return
        for_pos = self._metric_embed(for_node.state)
        nearest_index = get_nearest_index(states_rev, num_entries, for_pos)
        nearest_rev_id = self._node_matrix_rev.ids[nearest_index]
        dist = np.linalg.norm(states_rev[nearest_index] - for_pos)
        if dist <= self.get_rk():
            rev_node = self.tree_rev.nodes[nearest_rev_id]['value']
            score = dist + rev_node.cost_so_far
            self.q_priority.add_or_update(for_node_id, score)

    # ------------------------------------------------------------------ #
    #  Edge Bundle Propagation                                             #
    # ------------------------------------------------------------------ #

    def _propagate_edge(self, edge_index, parent_state, parent_time_elapsed,
                        is_forward=True):
        """
        Pull (action, duration) from the stored edge and re-propagate online
        from the parent state (matches BBoE Alg. 3 line 11
        `SIMULATECONTROL(v, u_near, t)`).

        is_forward semantics (paper / GBRRT-style):
          * is_forward=True  → forward integration with `+duration`. Used by
            the forward tree's exploit / fast-explore / random-explore /
            goal-bias paths.
          * is_forward=False → backward integration with `-duration`. Used by
            the reverse tree so that rev nodes are states whose forward-
            integrated dynamics (with the same action over `duration`) reach
            the parent rev node — i.e., the rev tree contains states
            *backward-reachable* from the goal. The cost-so-far stored on a
            rev node then represents the true forward-time cost from that
            state to the goal, making the exploit heuristic semantically
            correct.

        Cost is computed against `duration` (always positive) and the path
        recorded for the rev tree is the backward-time trajectory; BBoE only
        ever splices forward-tree edges into the returned path, so rev-tree
        path geometry is heuristic-only.
        """
        eb = self.edge_bundle
        action = eb.actions[edge_index]
        duration = abs(eb.timesteps[edge_index])
        eval_time = duration if is_forward else -duration

        num_record_steps = max(1, round(duration / self.minimum_time_step))
        new_state, path = self.agent.get_next_state(
            parent_state, action, eval_time, num_steps=num_record_steps)

        # Collision check on full path
        valid = self.isvalid(
            path, self.agent.radius, self.env.size,
            self.static_circular_obstacles, self.static_rectangular_obstacles,
            self.dynamic_agent_obstacles, self.env.obstacle_buffer,
            self.env.boundary_buffer, parent_time_elapsed, duration,
            self.minimum_time_step)

        if not valid:
            return None

        new_state = path[-1].copy()
        return (new_state, path, action, duration)

    def best_prop_using_edge_bundle(self, parent_node, x_des, k_bias_value,
                                    is_forward=True):
        """
        Algorithm 2: BestPropUsingEdgeBundle.

        1. Gather nearby edges (β_near) via KD-tree query on heading angle.
        2. Sort by translated distance to x_des (β_near_sorted), excluding
           edges already tried from this node (per-node edge masking).
        3. Build β_biased (every Nth edge) and β_random (M random edges).
        4. With probability k_bias_value choose β_biased, else β_random.
        5. Propagate edges sequentially; return first collision-free result.

        Pass `is_forward=False` from the reverse tree so the propagation
        integrates dynamics with `-duration` — see `_propagate_edge`.

        REV-TREE BUG NOTE (is_forward=False): the agent's `sort_edges` keys
        on the bundle's *forward* final_state translated to the parent. For
        backward integration the actual new state is the predecessor of the
        parent, which lies in the OPPOSITE direction of the forward final
        offset. Sorting by forward-final-distance therefore picks edges that
        send the rev tree away from x_des — visually the rev tree clumps
        near the goal because every expansion lands in roughly the same
        wrong-direction region (verified empirically on unicycle: 200+ rev
        nodes within 7 m of the goal in obstacle-free env).
        Fix: for is_forward=False, propagate the chosen subset *first*, then
        pick the post-propagation state with the smallest d_X to x_des.
        """
        eb = self.edge_bundle
        x_near_state = parent_node.state
        parent_time_elapsed = parent_node.time_elapsed

        # Initialize per-node edge cache on first visit
        if parent_node.edge_bundle_indices is None:
            query = self.get_eb_kd_tree_query(x_near_state)
            edge_ids = self.eb_kd_tree.radius_query(query, self.kd_tree_delta_radius)
            if len(edge_ids) == 0:
                return None
            parent_node.edge_bundle_indices = edge_ids
            parent_node.edge_bundle_mask = np.full(len(edge_ids), False, dtype=bool)

        curr_edge_indices = parent_node.edge_bundle_indices
        curr_edge_mask = parent_node.edge_bundle_mask
        num_edges = len(curr_edge_indices)

        if is_forward:
            # FORWARD path — bundle's forward final_state matches the actual
            # post-propagation new state under SE(2) translation invariance,
            # so sort_edges' approximation is correct. Existing behavior.

            # Ensure distance array is large enough
            if len(self.distance_array) < num_edges:
                self.distance_array = np.zeros(num_edges, dtype=np.float64)

            # Line 2: β_near_sorted ← sort by paper d_X between the translated
            # edge endpoint and x_des. Agent-specific sort_edges receives the
            # full x_des (5-D for treaded) and applies d_X internally.
            sorted_indices, num_valid = self.sort_edges(
                x_near_state, x_des, eb.start_states, eb.final_states,
                curr_edge_indices, curr_edge_mask, self.distance_array)

            if num_valid == 0:
                return None

            # Line 3: β_biased ← every Nth edge {e_0, e_N, e_2N, ..., e_MN}
            N = self.num_skip_edges
            beta_biased = list(range(0, num_valid, N))

            # Line 4: β_random ← M random edges from β_near_sorted
            M = len(beta_biased)
            if M >= num_valid:
                beta_random = list(range(num_valid))
            else:
                beta_random = self.rng.choice(num_valid, size=M, replace=False).tolist()

            # Lines 5-10: Choose β_final based on k_bias probability
            c_rand = self.rng.uniform(0, 1)
            if c_rand <= k_bias_value:
                beta_final = beta_biased
            else:
                beta_final = beta_random

            # Lines 11-15: Propagate edges sequentially until collision-free
            for pos in beta_final:
                x = sorted_indices[pos]
                edge_bundle_index = curr_edge_indices[x]
                result = self._propagate_edge(edge_bundle_index, x_near_state,
                                              parent_time_elapsed,
                                              is_forward=True)
                # Mark edge as tried regardless of success or failure
                curr_edge_mask[x] = True
                if result is not None:
                    return result

            return None

        # REV path (is_forward=False) — propagate first, then pick best by
        # actual post-prop d_X. Skip sort_edges since its forward-final key
        # is the wrong direction for backward integration.
        untried_positions = np.where(~curr_edge_mask)[0]
        if len(untried_positions) == 0:
            return None

        # Subset size mirrors the forward path's β_biased size
        # (≈ num_untried / num_skip_edges) so per-iter compute stays bounded.
        # Bug guard: when num_skip_edges <= 1, the original formula would set
        # M = num_untried (all of them), exhausting the parent node in a
        # single call (rev path has no early-exit). Use total_cached // 5 in
        # that case so the node stays "alive" for future extensions.
        # Preserves the original behavior for any num_skip_edges >= 2.
        if self.num_skip_edges <= 1:
            M = max(1, len(curr_edge_indices) // 5)
        else:
            M = max(1, len(untried_positions) // self.num_skip_edges)
        if M >= len(untried_positions):
            sampled = untried_positions
        else:
            sampled = self.rng.choice(untried_positions, size=M, replace=False)

        x_des_embed = self._metric_embed(x_des)
        best_result = None
        best_dist = float('inf')
        for x in sampled:
            edge_bundle_index = curr_edge_indices[x]
            result = self._propagate_edge(edge_bundle_index, x_near_state,
                                          parent_time_elapsed,
                                          is_forward=False)
            curr_edge_mask[x] = True
            if result is None:
                continue
            new_state = result[0]
            d = float(np.linalg.norm(
                self._metric_embed(new_state) - x_des_embed))
            if d < best_dist:
                best_dist = d
                best_result = result
        return best_result

    def random_prop_using_edge_bundle(self, parent_node):
        """
        Algorithm 3: RandomPropUsingEdgeBundle.
        Pick a single random neighboring edge and propagate it.
        """
        # Initialize per-node edge cache on first visit
        if parent_node.edge_bundle_indices is None:
            query = self.get_eb_kd_tree_query(parent_node.state)
            edge_ids = self.eb_kd_tree.radius_query(query, self.kd_tree_delta_radius)
            if len(edge_ids) == 0:
                return None
            parent_node.edge_bundle_indices = edge_ids
            parent_node.edge_bundle_mask = np.full(len(edge_ids), False, dtype=bool)

        curr_edge_indices = parent_node.edge_bundle_indices
        curr_edge_mask = parent_node.edge_bundle_mask

        # Find untried edges
        untried = [i for i in range(len(curr_edge_indices)) if not curr_edge_mask[i]]
        if len(untried) == 0:
            return None

        pick = self.rng.integers(0, len(untried))
        x = untried[pick]
        edge_bundle_index = curr_edge_indices[x]
        curr_edge_mask[x] = True
        return self._propagate_edge(edge_bundle_index, parent_node.state,
                                    parent_node.time_elapsed)

    # ------------------------------------------------------------------ #
    #  Tree Extension                                                      #
    # ------------------------------------------------------------------ #

    def extend_tree_rev(self, parent_node_id, parent_node, random_point):
        """
        Algorithm 1, lines 7-14: Reverse tree expansion.
        Uses BestPropUsingEdgeBundle with k_exploration=1 (always β_biased)
        AND backward time integration so rev nodes are states whose forward
        dynamics reach the parent — i.e., backward-reachable from goal
        (paper / GBRRT semantics).
        """
        result = self.best_prop_using_edge_bundle(
            parent_node, random_point, 1.0, is_forward=False)

        if result is None:
            return

        new_state, path, action, duration = result
        edge_cost = self.cost(self.env, self.agent, parent_node.state,
                              action, duration, path)
        total_time = parent_node.time_elapsed + duration
        total_cost = parent_node.cost_so_far + edge_cost

        new_node_id = self.add_node_rev(new_state, parent_node_id, action,
                                        duration, path, total_time, total_cost)
        self.update_priority_queue(
            self.tree_rev.nodes[new_node_id]['value'], new_node_id)

    def _add_forward_node(self, parent_node_id, parent_node,
                          new_state, path, action, duration):
        """
        Algorithm 1, lines 32-38: Add a new node to the forward tree.
        Checks goal at the final state and along intermediate waypoints.
        Inserts into priority queue if not a goal node.
        """
        edge_cost = self.cost(self.env, self.agent, parent_node.state,
                              action, duration, path)
        total_elapsed_time = parent_node.time_elapsed + duration
        total_cost = parent_node.cost_so_far + edge_cost

        # Check goal at final state
        reached_goal_flag, goal_distance = self.reached_goal(
            new_state, self.goal_check, self.goal_radius, self.agent)

        if reached_goal_flag:
            new_node_id = self.add_node_for(
                new_state, parent_node_id, action, duration,
                path, total_elapsed_time, total_cost)
            self.path_found = True
            self.goal_node_id = new_node_id
            self.path_cost = total_cost
            self.path_time = total_elapsed_time
            return

        # Check intermediate states if close to goal
        if goal_distance < self.threshold:
            te = parent_node.time_elapsed
            for index, intermediate_state in enumerate(path):
                te += self.minimum_time_step
                goal_flag, _ = self.reached_goal(
                    intermediate_state, self.goal_check, self.goal_radius,
                    self.agent)
                if goal_flag:
                    modified_edge_time = te - parent_node.time_elapsed
                    new_path = path[:index + 1]
                    ec = self.cost(self.env, self.agent, parent_node.state,
                                   action, modified_edge_time, new_path)
                    tc = parent_node.cost_so_far + ec
                    new_node_id = self.add_node_for(
                        intermediate_state, parent_node_id, action,
                        modified_edge_time, new_path, te, tc)
                    self.path_found = True
                    self.goal_node_id = new_node_id
                    self.path_cost = tc
                    self.path_time = te
                    return

        # No goal reached — add node and insert into priority queue
        new_node_id = self.add_node_for(
            new_state, parent_node_id, action, duration,
            path, total_elapsed_time, total_cost)
        self.insert_priority_queue(
            self.tree_for.nodes[new_node_id]['value'], new_node_id)

    # ------------------------------------------------------------------ #
    #  Main Planning Loop (Algorithm 1)                                    #
    # ------------------------------------------------------------------ #

    def plan_path(self):
        self.reset_tree()

        # Line 1-2: Initialize trees
        self.add_node_for(self.start, -1, None, None, None, 0.0, 0.0)
        self.add_node_rev(self.goal, -1, None, None, None, 0.0, 0.0)

        curr_num_steps = 0
        start_time = time.time()

        while curr_num_steps <= self.max_iter:

            # --- Reverse Tree Expansion (lines 7-14) ---
            # goal_bias: with prob `goal_bias` aim the reverse tree at the
            # global start state. Short-circuit when goal_bias == 0 to avoid
            # consuming rng draws and shifting the planner's random sequence.
            if self.goal_bias > 0.0 and self.rng.uniform(0, 1) < self.goal_bias:
                rand_rev = self.start
            else:
                rand_rev = self.sample_random_point()
            rev_nearest_id, rev_nearest_node = self.get_nearest_node_rev(rand_rev)
            self.extend_tree_rev(rev_nearest_id, rev_nearest_node, rand_rev)

            # --- Forward Tree Expansion (lines 15-38) ---
            c_rand = self.rng.uniform(0, 1)
            e_for_result = None
            for_parent_id = None
            for_parent_node = None

            # goal_bias: with prob `goal_bias` aim the forward tree at the
            # global goal directly (skip exploit/explore). Falls through to
            # the normal logic if no valid edge is found. Short-circuit
            # when goal_bias == 0 to preserve the original rng sequence.
            if self.goal_bias > 0.0 and self.rng.uniform(0, 1) < self.goal_bias:
                for_nearest_id, for_nearest_node = self.get_nearest_node_for(self.goal)
                if for_nearest_node is not None:
                    e_for_result = self.best_prop_using_edge_bundle(
                        for_nearest_node, self.goal, self.k_bias)
                    if e_for_result is not None:
                        for_parent_id = for_nearest_id
                        for_parent_node = for_nearest_node

            if e_for_result is None and c_rand < self.exploitation_ratio:
                # Exploitation (lines 19-23)
                if not self.q_priority.is_empty():
                    for_pop_id, _ = self.q_priority.pop()
                    if for_pop_id is not None:
                        for_pop_node = self.tree_for.nodes[for_pop_id]['value']

                        # Line 21: V_rev_near within r_k of x_pop (paper d_X)
                        # States are already d_X-embedded via _metric_embed.
                        states_rev = self._node_matrix_rev.get_valid_matrix()
                        for_pos = self._metric_embed(for_pop_node.state)
                        diffs = states_rev - for_pos
                        dists = np.linalg.norm(diffs, axis=1)
                        valid_indices = np.where(dists <= self.get_rk())[0]

                        # Line 22: x_best = argmin g(x_pop) + d_X + h(x)
                        best_target_score = float('inf')
                        target_point = None
                        # Skip the consumed-rev filter when delta_reach == 0
                        # (no rev nodes ever get marked) — saves a per-iter
                        # set-membership check across all valid_indices.
                        _check_consumed = self.delta_reach > 0.0
                        for i in valid_indices:
                            rev_node_id = int(self._node_matrix_rev.ids[i])
                            # Fix B: skip already-reached rev nodes to force
                            # forward progress up the reverse-tree chain.
                            if _check_consumed and rev_node_id in self.consumed_rev_ids:
                                continue
                            rev_node = self.tree_rev.nodes[rev_node_id]['value']
                            score = (for_pop_node.cost_so_far + dists[i]
                                     + rev_node.cost_so_far)
                            if score < best_target_score:
                                best_target_score = score
                                target_point = rev_node.state

                        # Line 23: BestPropUsingEdgeBundle with k_bias
                        if target_point is not None:
                            e_for_result = self.best_prop_using_edge_bundle(
                                for_pop_node, target_point, self.k_bias)
                            if e_for_result is not None:
                                for_parent_id = for_pop_id
                                for_parent_node = for_pop_node
                                # Fix B: mark rev nodes within delta_reach of
                                # the new forward node as consumed. Skip the
                                # O(|rev|) scan entirely when delta_reach == 0.
                                if self.delta_reach > 0.0:
                                    new_state, _path, _action, _dur = e_for_result
                                    self._mark_consumed_rev_near(new_state)

                # Exploration fallback (lines 24-27)
                if e_for_result is None:
                    rand_for = self.sample_random_point()
                    for_nearest_id, for_nearest_node = self.get_nearest_node_for(rand_for)
                    e_for_result = self.best_prop_using_edge_bundle(
                        for_nearest_node, rand_for, 1.0)
                    if e_for_result is not None:
                        for_parent_id = for_nearest_id
                        for_parent_node = for_nearest_node

            # Random exploration (lines 28-31)
            if e_for_result is None:
                rand_for = self.sample_random_point()
                for_nearest_id, for_nearest_node = self.get_nearest_node_for(rand_for)
                e_for_result = self.random_prop_using_edge_bundle(for_nearest_node)
                if e_for_result is not None:
                    for_parent_id = for_nearest_id
                    for_parent_node = for_nearest_node

            # Lines 32-38: Add forward node, check goal, update priority queue
            if e_for_result is not None:
                self._add_forward_node(for_parent_id, for_parent_node,
                                       *e_for_result)

            if self.path_found or (time.time() - start_time >= self.planning_time):
                break
            curr_num_steps += 1

        total_time = time.time() - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)

        if self.print_logs or self.debug_flag:
            planning_time_msg = "Total Planning Time"
            if hasattr(self.agent, "id"):
                planning_time_msg += " for agent " + str(self.agent.id)
            planning_time_msg += " after " + str(curr_num_steps) + " iterations"
            print(planning_time_msg + ": ", total_time)

    # ------------------------------------------------------------------ #
    #  Path Extraction                                                     #
    # ------------------------------------------------------------------ #

    def get_path_to_node_id(self, goal_node_id):
        state_length = self.agent.state_length
        control_length = self.agent.action_length

        max_nodes = len(self.tree_for.nodes)
        path_rrt_node_ids = np.empty(max_nodes, dtype=np.int32)
        path_states = np.empty((max_nodes, state_length), dtype=np.float64)
        path_controls = np.empty((max_nodes, control_length), dtype=np.float64)
        path_timesteps = np.empty(max_nodes, dtype=np.float64)
        node_id = goal_node_id
        path_length = 0

        while node_id != -1:
            rrt_node = self.tree_for.nodes[node_id]['value']
            path_rrt_node_ids[path_length] = rrt_node.id
            path_states[path_length] = rrt_node.state
            path_controls[path_length] = (rrt_node.parent_action
                                          if rrt_node.parent_action is not None
                                          else np.zeros(control_length))
            path_timesteps[path_length] = (rrt_node.parent_action_duration
                                           if rrt_node.parent_action_duration is not None
                                           else 0.0)
            path_length += 1
            node_id = rrt_node.parent_id

        ids = path_rrt_node_ids[:path_length][::-1]
        states = path_states[:path_length][::-1]
        controls = path_controls[:path_length - 1][::-1]
        timesteps = path_timesteps[:path_length - 1][::-1]
        return ids, states, controls, timesteps

    def get_path(self):
        if not self.path_found:
            print("Path can't be found because goal hasn't been reached!")
            return (np.empty(0, dtype=np.int16),
                    np.empty((0, self.agent.state_length), dtype=np.float64),
                    np.empty((0, self.agent.action_length), dtype=np.float64),
                    np.empty(0, dtype=np.float64))
        else:
            return self.get_path_to_node_id(self.goal_node_id)
