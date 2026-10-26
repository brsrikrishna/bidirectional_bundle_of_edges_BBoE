"""
BoE — "Bundle of Edges" planner from Shome & Kavraki (ICRA 2021).

Implements Algorithm 3 (RetracePath) of:
  R. Shome and L. E. Kavraki, "Asymptotically Optimal Kinodynamic Planning
  Using Bundles of Edges", ICRA 2021.

This is a faithful port of the paper's runtime planner. It assumes the
edge bundle has already been precomputed offline (Algorithms 1-2 in the
paper); we reuse the same `EdgeBundle` and KD-tree infrastructure that
BBoE uses. The bundle's KD-tree returns indices of all edges whose start
state lies within radius θ of the query state.

Algorithm summary (Alg 3):
    Q ← {x_0}
    while Q ≠ ∅:
        v ← SELECT(Q)            # A*-like f-value priority + greedy override
        if v ∈ Q_goal: update best path
        for v_near ∈ RadialNN(v, θ):     # all bundle edges within θ of v
            (u, t) ← out-edge of v_near
            e_new ← SimulateControl(v, u, t)
            if collision-free: Q ← Q ∪ {child(v, e_new)}

Key implementation choices (all called out in Sec V of the paper):
  - SELECT: f-value priority queue (g + h, where h is Euclidean to goal).
  - Greedy override: if a newly generated child has lower h than its parent,
    it is queued at top priority for the next iteration.
  - Goal-biased selection: with probability `goal_select_prob`, override the
    queue pop with the tree node nearest to the goal (Sec V "An analog to
    goal biasing selects the nearest unexplored node from the goal for a
    fraction of iterations").
  - SST-lite witness pruning (paper Sec V):
        * Each child state queries the nearest witness in d_X.
        * If no witness is closer than `min_spacing` (= witness radius δ_S),
          a new witness is created and the child becomes its representative
          (active in the main A* queue).
        * If a witness exists within δ_S, costs are compared:
              child_g < rep.cost  →  child dominates rep; old rep is
                                     deactivated, the witness's rep is
                                     swapped to the child, and the child
                                     enters the main A* queue.
              else                 →  child is dominated; it is added to the
                                     tree (so the parent pointer is preserved)
                                     but marked inactive and parked in a
                                     deferred side heap instead of the main
                                     queue.
  - Deferred re-injection (paper Sec V): when the main A* heap empties (no
    more active nodes to expand), the deferred side heap is drained back
    into the main heap, every drained candidate is reactivated, and the
    search resumes. This recovers from witness over-pruning on tight
    problems where the active rep failed to reach the goal but a dominated
    candidate in the same witness might.

Note: this implementation breaks on first goal hit (NOT the paper's
asymptotic-optimal RetracePath which keeps refining). The witness pruning
+ deferred re-injection still help by improving search-tree quality and
robustness on tight problems, even without the AO outer loop.
"""
import networkx as nx
import numpy as np
import time
import heapq
from utils import find_roundoff_decimal_digits, get_nearest_index
from numba.typed import List
from numba import types, njit
from rrt import DynamicMatrix, TreeNode


@njit(cache=True)
def _nearest_witness_brute_njit(witness_states, n, query):
    """JIT-compiled brute-force nearest-witness scan. Same answer as a
    numpy `np.linalg.norm(witness_states[:n] - query, axis=1).argmin()`
    but ~5-10x faster because the AOT-compiled inner loop avoids
    numpy's per-call Python overhead (which is the dominant cost when
    n is in the hundreds-to-thousands range and dims are small)."""
    best_idx = -1
    best_d2 = 1e30
    dim = query.shape[0]
    for i in range(n):
        d2 = 0.0
        for k in range(dim):
            diff = witness_states[i, k] - query[k]
            d2 += diff * diff
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i
    return best_idx, np.sqrt(best_d2)


class BoETreeNode(TreeNode):
    def __init__(self, sid, state, parent_id, parent_action,
                 parent_action_duration, path_from_parent,
                 time_so_far, cost):
        super().__init__(sid, state, parent_id, parent_action,
                         parent_action_duration, path_from_parent,
                         time_so_far, cost)


class _FPQueue:
    """Min-heap on f-value, with stable insertion-order tie-break.
    Supports a head-injection slot for greedy override."""
    def __init__(self):
        self.heap = []
        self.counter = 0
        self.head = None  # greedy override slot

    def push(self, node_id, f):
        heapq.heappush(self.heap, (f, self.counter, node_id))
        self.counter += 1

    def push_head(self, node_id):
        # Greedy override — bypass the heap on the next pop.
        self.head = node_id

    def pop(self):
        if self.head is not None:
            nid = self.head
            self.head = None
            return nid
        if not self.heap:
            return None
        return heapq.heappop(self.heap)[2]

    def is_empty(self):
        return self.head is None and not self.heap


class BoE:
    """RetracePath / Bundle-of-Edges planner (Shome & Kavraki, ICRA 2021)."""

    def __init__(self, *, start, goal, goal_radius, env, agent,
                 sampling_time_step=1.0,
                 minimum_time_step=0.1,
                 max_iter=100000,
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
                 sort_edges_function=None,
                 max_num_edges_per_node=1000,
                 goal_select_prob=0.05,
                 min_spacing=0.0,
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

        self.edge_bundle = edge_bundle
        self.eb_kd_tree = eb_kd_tree
        self.get_eb_kd_tree_query = get_eb_kd_tree_query
        self.theta = float(kd_tree_delta_radius)
        self.translate_function = translate_function
        self.sort_edges_function = sort_edges_function    # unused in BoE; kept for interface parity
        self.max_num_edges_per_node = max_num_edges_per_node

        self.goal_select_prob = float(goal_select_prob)
        self.min_spacing = float(min_spacing)

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
        # Distance below which the walk-along-edge goal check fires
        # (mirrors RRT/SST/GBRRT/BBoE). When an edge's endpoint is within
        # `self.threshold` of the goal but didn't reach the goal ball,
        # we walk the edge's waypoints and accept any intermediate
        # waypoint that does fall inside the ball.
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
        self._node_matrix = DynamicMatrix(
            initial_capacity=1024, dim=self.distance_metric_state_size)

        self.roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)
        self.node_class = BoETreeNode

    # ---------------------------------------------------------------- #
    #   Helpers                                                         #
    # ---------------------------------------------------------------- #
    def _metric_embed(self, state):
        if hasattr(self.agent, 'metric_embed_state'):
            return self.agent.metric_embed_state(state)
        s = state[:self.distance_metric_state_size]
        if self._metric_scales is not None:
            return s * self._metric_scales
        return s

    def _heuristic(self, state):
        """Euclidean distance from `state` to goal in the agent's metric
        space — the standard A* heuristic for Lipschitz-smooth dynamics."""
        return float(np.linalg.norm(
            self._metric_embed(state) - self._metric_embed(self.goal)))

    def reset_tree(self):
        self.tree = nx.DiGraph()
        self._node_matrix = DynamicMatrix(
            initial_capacity=1024, dim=self.distance_metric_state_size)
        self.goal_node_id = None
        self.path_found = False
        self.path_cost = float('inf')
        self.path_time = 0.0
        self.last_added_node_id = -1
        self._expanded = set()   # node ids already popped/expanded

        # SST-lite witness pruning state.
        # `_witness_matrix` stores embedded witness states; `ids` slot is
        # repurposed to hold the current representative node id for each
        # witness. `_active` tracks per-node "is currently a witness rep
        # / re-injected from deferred queue" — main-heap pop loop skips
        # inactive nodes. `_deferred_heap` is the side queue for dominated
        # candidates; drained into the main heap when the latter empties.
        self._witness_matrix = DynamicMatrix(
            initial_capacity=1024, dim=self.distance_metric_state_size)
        self._active = {}
        self._deferred_heap = []
        self._deferred_counter = 0

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

    def _nearest_tree_node_id(self, query_state):
        states = self._node_matrix.get_valid_matrix()
        n = self._node_matrix.count
        if n == 0:
            return None
        idx = get_nearest_index(states, n, self._metric_embed(query_state))
        return int(self._node_matrix.ids[idx])

    def _nearest_unexpanded_to_goal(self):
        """Goal-biased selection: nearest tree node to the goal that has not
        yet been expanded AND is currently active (paper Sec V: "nearest
        unexplored node from goal"). Inactive (witness-dominated) nodes are
        skipped — they would be popped-and-skipped from the main heap
        anyway, so picking them here would waste an iteration."""
        states = self._node_matrix.get_valid_matrix()
        n = self._node_matrix.count
        if n == 0:
            return None
        goal_emb = self._metric_embed(self.goal)
        diffs = states[:n] - goal_emb
        dists = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dists)
        for i in order:
            nid = int(self._node_matrix.ids[i])
            if nid in self._expanded:
                continue
            if not self._active.get(nid, True):
                continue
            return nid
        return None

    def _nearest_tree_distance(self, query_state):
        """Distance from query_state to its nearest tree node (paper's NearestDistance)."""
        states = self._node_matrix.get_valid_matrix()
        n = self._node_matrix.count
        if n == 0:
            return float('inf')
        emb = self._metric_embed(query_state)
        diffs = states[:n] - emb
        return float(np.linalg.norm(diffs, axis=1).min())

    # ---------------------------------------------------------------- #
    #   SST-lite witness pruning helpers (paper Sec V)                  #
    # ---------------------------------------------------------------- #
    def _add_witness(self, state, rep_id):
        """Record `state` as a new witness whose current representative is
        the tree node with id `rep_id`. The rep_id slot is later updated
        in place via `self._witness_matrix.ids[witness_idx]` when a
        cheaper candidate dominates the current rep."""
        self._witness_matrix.append(self._metric_embed(state), rep_id)

    def _nearest_witness_index(self, state):
        """Return (witness_index, distance) of the nearest witness to
        `state` under paper d_X, or (-1, inf) if no witnesses exist.

        Implementation note: dispatches to a numba-jitted inner loop so
        the per-query cost is ~10x lower than a pure-Python /
        numpy-broadcasting equivalent. The answer is identical to a
        plain brute-force scan; this is purely a speed optimisation.
        The paper does not mandate any particular data structure for
        witness lookup."""
        n = self._witness_matrix.count
        if n == 0:
            return -1, float('inf')
        # Use the underlying buffer directly (zero-copy). _nearest_witness_brute_njit
        # only reads the first `n` rows.
        emb = np.ascontiguousarray(self._metric_embed(state), dtype=np.float64)
        idx, dist = _nearest_witness_brute_njit(
            self._witness_matrix.matrix, n, emb)
        return int(idx), float(dist)

    def _drain_deferred_into_main(self, Q):
        """Re-inject every still-pending dominated candidate from the
        deferred side heap into the main A* queue and reactivate it.

        Called when the main heap empties so the search can recover
        from over-aggressive witness pruning on tight problems. Returns
        the number of candidates actually re-injected (0 if all
        deferred candidates have already been expanded)."""
        if not self._deferred_heap:
            return 0
        drained = 0
        while self._deferred_heap:
            f, _, node_id = heapq.heappop(self._deferred_heap)
            if node_id in self._expanded:
                continue
            # Reactivate the node — it can now be popped and expanded.
            self._active[node_id] = True
            Q.push(node_id, f)
            drained += 1
        return drained

    # ---------------------------------------------------------------- #
    #   Edge propagation (re-propagates bundle's stored action+duration  #
    #   from the live tree node, matching paper Alg 3 line 12).         #
    # ---------------------------------------------------------------- #
    def _propagate_edge(self, parent_state, parent_time_elapsed,
                        edge_bundle_index):
        eb = self.edge_bundle
        action = eb.actions[edge_bundle_index]
        duration = abs(eb.timesteps[edge_bundle_index])
        num_steps = max(1, round(duration / self.minimum_time_step))

        new_state, path = self.agent.get_next_state(
            parent_state, action, duration, num_steps=num_steps)

        valid = self.isvalid(
            path, self.agent.radius, self.env.size,
            self.static_circular_obstacles, self.static_rectangular_obstacles,
            self.dynamic_agent_obstacles, self.env.obstacle_buffer,
            self.env.boundary_buffer, parent_time_elapsed, duration,
            self.minimum_time_step)
        if not valid:
            return None
        return (new_state, path, action, duration)

    # ---------------------------------------------------------------- #
    #   Algorithm 3 — RetracePath main loop                              #
    # ---------------------------------------------------------------- #
    def plan_path(self):
        self.reset_tree()
        root_id = self.add_node(self.start, -1, None, None, None, 0.0, 0.0)

        # Root is the first witness's representative.
        self._active[root_id] = True
        if self.min_spacing > 0.0:
            self._add_witness(self.start, root_id)

        Q = _FPQueue()
        Q.push(root_id, self._heuristic(self.start))   # g=0 + h(start)

        curr_iter = 0
        start_time = time.time()

        # Use the witness mechanism only if the user enabled it via
        # min_spacing > 0; otherwise fall back to no pruning. This
        # preserves the behavior of `min_spacing=0.0` users.
        use_witness = self.min_spacing > 0.0

        while curr_iter <= self.max_iter:
            # ---- Selection (A* + greedy + goal-biased) ----
            v_id = None
            if self.rng.uniform(0, 1) < self.goal_select_prob:
                # Goal-biased: nearest unexpanded ACTIVE tree node to goal.
                v_id = self._nearest_unexpanded_to_goal()

            if v_id is None:
                v_id = Q.pop()
                # Skip already-expanded OR inactive (witness-dominated) nodes.
                while v_id is not None and (
                        v_id in self._expanded
                        or not self._active.get(v_id, True)):
                    v_id = Q.pop()
                if v_id is None:
                    # Main heap exhausted → drain deferred candidates back
                    # into it (paper Sec V deferred re-injection). If the
                    # drain produces nothing fresh either, we are done.
                    drained = self._drain_deferred_into_main(Q)
                    if drained == 0:
                        break
                    v_id = Q.pop()
                    while v_id is not None and (
                            v_id in self._expanded
                            or not self._active.get(v_id, True)):
                        v_id = Q.pop()
                    if v_id is None:
                        break

            self._expanded.add(v_id)
            v = self.tree.nodes[v_id]['value']

            # ---- Goal check (A5.1: still terminates on first solution) ----
            reached, _ = self.reached_goal(
                v.state, self.goal_check, self.goal_radius, self.agent)
            if reached:
                if v.cost_so_far < self.path_cost:
                    self.path_found = True
                    self.goal_node_id = v_id
                    self.path_cost = v.cost_so_far
                    self.path_time = v.time_elapsed
                    # Paper's Alg 3 continues the loop to find better paths
                    # (asymptotic optimality), but our budget-bounded usage
                    # returns the first solution.
                    break

            # ---- Expand: all bundle edges within θ of v ----
            query = self.get_eb_kd_tree_query(v.state)
            edge_ids = self.eb_kd_tree.radius_query(query, self.theta)
            n_edges = min(len(edge_ids), self.max_num_edges_per_node)
            edge_ids = edge_ids[:n_edges]

            best_child_id = None
            best_child_h = float('inf')
            v_h = self._heuristic(v.state)

            walked_to_goal = False  # set if walk-along-edge fires below
            for edge_idx in edge_ids:
                result = self._propagate_edge(
                    v.state, v.time_elapsed, edge_idx)
                if result is None:
                    continue
                new_state, path, action, duration = result

                # Walk-along-edge goal check (mirrors RRT/SST/GBRRT/BBoE/
                # Db-RRT). The endpoint goal check is normally
                # done at pop time in BoE's A* style, but we add an
                # extension-time *intermediate-waypoint* check for
                # symmetry with the other planners: when the edge passes
                # through the goal ball mid-traverse, accept that
                # waypoint immediately and short-circuit. Gated by
                # endpoint distance < self.threshold to keep it cheap.
                _, endpoint_goal_distance = self.reached_goal(
                    new_state, self.goal_check, self.goal_radius, self.agent)
                if endpoint_goal_distance < self.threshold:
                    te = v.time_elapsed
                    for wi, intermediate_state in enumerate(path[:-1]):
                        te += self.minimum_time_step
                        gflag, _ = self.reached_goal(
                            intermediate_state, self.goal_check,
                            self.goal_radius, self.agent)
                        if gflag:
                            modified_edge_time = te - v.time_elapsed
                            new_path = path[:wi + 1]
                            ec = self.cost(self.env, self.agent, v.state,
                                           action, modified_edge_time, new_path)
                            tc = v.cost_so_far + ec
                            child_id = self.add_node(
                                intermediate_state, v_id, action,
                                modified_edge_time, new_path, te, tc)
                            self._active[child_id] = True
                            if tc < self.path_cost:
                                self.path_found = True
                                self.goal_node_id = child_id
                                self.path_cost = tc
                                self.path_time = te
                                walked_to_goal = True
                            break
                    if walked_to_goal:
                        break  # exit edge loop; outer loop will exit on next iter

                edge_cost = self.cost(self.env, self.agent, v.state,
                                      action, duration, path)
                child_g = v.cost_so_far + edge_cost
                child_time = v.time_elapsed + duration

                child_h = self._heuristic(new_state)

                # ---- SST-lite witness pruning + deferred queue ----
                if use_witness:
                    w_idx, w_dist = self._nearest_witness_index(new_state)
                    if w_idx == -1 or w_dist > self.min_spacing:
                        # No nearby witness → register a new one with this
                        # child as its rep, and queue normally.
                        child_id = self.add_node(
                            new_state, v_id, action, duration,
                            path, child_time, child_g)
                        self._active[child_id] = True
                        self._add_witness(new_state, child_id)
                        Q.push(child_id, child_g + child_h)

                        if child_h < best_child_h:
                            best_child_h = child_h
                            best_child_id = child_id
                    else:
                        # An existing witness covers this region → compare
                        # this candidate's cost-so-far against the rep.
                        rep_id = int(self._witness_matrix.ids[w_idx])
                        rep_node = self.tree.nodes[rep_id]['value']
                        if child_g < rep_node.cost_so_far:
                            # Child dominates the rep. Add it, deactivate
                            # the old rep, swap the witness's rep pointer.
                            child_id = self.add_node(
                                new_state, v_id, action, duration,
                                path, child_time, child_g)
                            self._active[rep_id] = False
                            self._witness_matrix.ids[w_idx] = child_id
                            self._active[child_id] = True
                            Q.push(child_id, child_g + child_h)

                            if child_h < best_child_h:
                                best_child_h = child_h
                                best_child_id = child_id
                        else:
                            # Child is dominated. Add it to the tree (so the
                            # parent pointer is preserved for path tracing
                            # if it later gets re-injected and reaches goal),
                            # but mark inactive and park it in the deferred
                            # heap instead of the main A* queue.
                            child_id = self.add_node(
                                new_state, v_id, action, duration,
                                path, child_time, child_g)
                            self._active[child_id] = False
                            self._deferred_counter += 1
                            heapq.heappush(
                                self._deferred_heap,
                                (child_g + child_h,
                                 self._deferred_counter,
                                 child_id))
                            # Do NOT track inactive child for greedy override.
                else:
                    # No witness pruning (min_spacing == 0): every child is
                    # a fresh active node.
                    child_id = self.add_node(
                        new_state, v_id, action, duration,
                        path, child_time, child_g)
                    self._active[child_id] = True
                    Q.push(child_id, child_g + child_h)

                    if child_h < best_child_h:
                        best_child_h = child_h
                        best_child_id = child_id

            # ---- Greedy override: if best child improves heuristic over v,
            # queue it as the next pop (paper Sec V).
            if best_child_id is not None and best_child_h < v_h:
                Q.push_head(best_child_id)

            # If walk-along-edge accepted an intermediate goal during this
            # expansion, exit the planning loop now (A5.1: first-solution
            # termination). Without this, the next iteration would try to
            # pop the goal node and goal-detect there — same answer, just
            # slower.
            if walked_to_goal:
                break

            if time.time() - start_time >= self.planning_time:
                break
            curr_iter += 1

        total_run = time.time() - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)
        if self.print_logs or self.debug_flag:
            msg = "BoE planning"
            if hasattr(self.agent, "id"):
                msg += f" agent {self.agent.id}"
            print(f"{msg}: {total_run:.2f}s, iters={curr_iter}, "
                  f"found={self.path_found}, cost={self.path_cost}")

    # ---------------------------------------------------------------- #
    #   Path extraction                                                  #
    # ---------------------------------------------------------------- #
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
