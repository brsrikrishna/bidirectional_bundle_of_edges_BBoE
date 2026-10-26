import networkx as nx
import numpy as np
import time
import heapq
from utils import euclidean_distance_numba_with_l, \
    get_dtype_from_input,find_roundoff_decimal_digits,\
    get_nearest_index, compact_matrix
from numba.typed import List
from numba import types
from rrt import DynamicMatrix, TreeNode

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.REMOVED = '<removed-node>'
        self.counter = 0
        self.popped = set()  # Nodes that have been popped are never reinserted

    def add_or_update(self, node_id, priority):
        if node_id in self.popped:
            return

        if node_id in self.entry_finder:
            # Only update if the new priority is better (lower)
            if priority < self.entry_finder[node_id][0]:
                self.remove(node_id)
            else:
                return

        entry = [priority, self.counter, node_id]
        self.entry_finder[node_id] = entry
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def remove(self, node_id):
        if node_id in self.entry_finder:
            entry = self.entry_finder.pop(node_id)
            entry[-1] = self.REMOVED

    def pop(self):
        while self.heap:
            priority, count, node_id = heapq.heappop(self.heap)
            if node_id is not self.REMOVED:
                del self.entry_finder[node_id]
                self.popped.add(node_id)
                return node_id, priority
        return None, None

    def is_empty(self):
        return len(self.entry_finder) == 0

class GBRRTTreeNode(TreeNode):
    def __init__(self, sid, state, parent_id, parent_action, parent_action_duration,
                    path_from_parent, time_so_far, cost):
        super().__init__(sid, state, parent_id, parent_action, parent_action_duration,
                        path_from_parent, time_so_far, cost)
        self.edge_bundle_indices = None
        self.edge_bundle_mask = None


class GBRRT:
    def __init__(self, * , start, goal, goal_radius, env, agent,
                    use_fixed_sampling_time=True, 
                    sampling_time_step=1.0,
                    minimum_time_step=0.1, 
                    max_iter=1000, 
                    planning_time=10.0,
                    num_extension_trials=1,
                    isvalid_function, 
                    cost_function, 
                    reached_goal_function, 
                    random_point_function, 
                    udf_seed = 77,
                    debug_flag = False,
                    print_logs = False, 
                    dynamic_obstacles=None,
                    exploitation_ratio=0.8,
                    heuristic_radius=50.0,
                    gamma=50.0,
                    delta_reach=0.0,
                    edge_bundle=None,
                    eb_kd_tree=None,
                    get_eb_kd_tree_query=None,
                    kd_tree_delta_radius=0.5,
                    translate_function=None,
                    max_num_edges_per_node=3000
                    ):

        self.start = np.array(start)
        self.goal = np.array(goal)
        self.goal_radius = goal_radius
        self.goal_state_length = len(goal)
        # Goal check passes the full goal vector; each agent's reached_goal
        # implements its own paper-faithful d_X (which may use a subset of
        # dims with custom weights — e.g. treaded d_X uses x,y and weighted vL,vR).
        self.goal_check = self.goal
        self.env = env
        self.agent = agent
        self.use_fixed_sampling_time = use_fixed_sampling_time
        self.max_sample_T = sampling_time_step
        self.minimum_time_step = minimum_time_step
        self.max_iter = max_iter
        self.planning_time = planning_time
        self.num_extension_trials = num_extension_trials
        self.isvalid = isvalid_function
        self.cost = cost_function
        self.reached_goal = reached_goal_function
        self.get_random_point = random_point_function
        self.exploitation_ratio = exploitation_ratio
        self.heuristic_radius = heuristic_radius
        self.gamma = gamma
        # Fix B: after the forward tree reaches within delta_reach of a rev
        # node, that rev node is excluded from all future exploit targeting.
        # Forces the forward tree to advance up the reverse-tree chain toward
        # goal instead of oscillating around the same rev node.
        self.delta_reach = float(delta_reach)
        self.consumed_rev_ids = set()

        self.edge_bundle = edge_bundle
        self.eb_kd_tree = eb_kd_tree
        self.get_eb_kd_tree_query = get_eb_kd_tree_query
        self.kd_tree_delta_radius = kd_tree_delta_radius
        self.translate_function = translate_function
        self.max_num_edges_per_node = max_num_edges_per_node

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
        self.node_class = GBRRTTreeNode

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

    def get_rk(self):
        v_rev = self._node_matrix_rev.count
        if v_rev <= 1:
            return self.heuristic_radius
        d = len(self.start)  # state-space dimension per paper's r_k formula
        val = self.gamma * ((np.log(v_rev) / v_rev) ** (1.0 / (d + 1.0)))
        return min(val, self.heuristic_radius)

    def get_random_time(self):
        return round(self.rng.uniform(self.minimum_time_step, self.max_sample_T), self.roundoff_digits)

    def get_fixed_time(self):
        return self.max_sample_T
    
    def get_time(self):
        if self.use_fixed_sampling_time:
            return self.get_fixed_time()
        else:
            return self.get_random_time()

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
        consumed — they will be excluded from all future exploit targeting."""
        states_rev = self._node_matrix_rev.get_valid_matrix()
        if states_rev.shape[0] == 0:
            return
        for_embed = self._metric_embed(for_state)
        dists = np.linalg.norm(states_rev - for_embed, axis=1)
        within = np.where(dists <= self.delta_reach)[0]
        for i in within:
            self.consumed_rev_ids.add(int(self._node_matrix_rev.ids[i]))

    def add_node_for(self, state, parent_node_id, parent_action, parent_action_duration, path_from_parent, time_elapsed, cost):
        new_node_id = self.last_added_node_id_for + 1
        new_node = self.node_class(new_node_id, state, parent_node_id, parent_action, parent_action_duration,
                                 path_from_parent, round(time_elapsed, self.roundoff_digits), cost)
        self.tree_for.add_node(new_node_id, value=new_node)
        self.last_added_node_id_for = new_node_id
        self._node_matrix_for.append(self._metric_embed(state), new_node_id)
        return new_node_id

    def add_node_rev(self, state, parent_node_id, parent_action, parent_action_duration, path_from_parent, time_elapsed, cost):
        new_node_id = self.last_added_node_id_rev + 1
        new_node = self.node_class(new_node_id, state, parent_node_id, parent_action, parent_action_duration,
                                 path_from_parent, round(time_elapsed, self.roundoff_digits), cost)
        self.tree_rev.add_node(new_node_id, value=new_node)
        self.last_added_node_id_rev = new_node_id
        self._node_matrix_rev.append(self._metric_embed(state), new_node_id)
        return new_node_id

    def sample_random_point(self):
        return self.get_random_point(self.env, self.static_circular_obstacles, self.static_rectangular_obstacles, self.rng)

    def get_nearest_node_for(self, random_point):
        states = self._node_matrix_for.get_valid_matrix()
        num_entries = self._node_matrix_for.count
        if num_entries == 0:
            return None, None
        nearest_index = get_nearest_index(states, num_entries, self._metric_embed(random_point))
        nearest_node_id = self._node_matrix_for.ids[nearest_index]
        return nearest_node_id, self.tree_for.nodes[nearest_node_id]['value']

    def get_nearest_node_rev(self, random_point):
        states = self._node_matrix_rev.get_valid_matrix()
        num_entries = self._node_matrix_rev.count
        if num_entries == 0:
            return None, None
        nearest_index = get_nearest_index(states, num_entries, self._metric_embed(random_point))
        nearest_node_id = self._node_matrix_rev.ids[nearest_index]
        return nearest_node_id, self.tree_rev.nodes[nearest_node_id]['value']

    def update_priority_queue(self, rev_node, rev_node_id):
        # Algorithm 3: find single nearest forward node to new reverse node
        states_for = self._node_matrix_for.get_valid_matrix()
        num_entries = self._node_matrix_for.count
        if num_entries == 0: return

        rev_pos = self._metric_embed(rev_node.state)
        nearest_index = get_nearest_index(states_for, num_entries, rev_pos)
        nearest_for_id = self._node_matrix_for.ids[nearest_index]

        dist = np.linalg.norm(states_for[nearest_index] - rev_pos)
        if dist <= self.get_rk():
            score = dist + rev_node.cost_so_far  # d_X(x_closest, x_rev) + h(x_rev)
            self.q_priority.add_or_update(nearest_for_id, score)

    def insert_priority_queue(self, for_node, for_node_id):
        # Algorithm 2: find single nearest reverse node to new forward node
        states_rev = self._node_matrix_rev.get_valid_matrix()
        num_entries = self._node_matrix_rev.count
        if num_entries == 0: return

        for_pos = self._metric_embed(for_node.state)
        nearest_index = get_nearest_index(states_rev, num_entries, for_pos)
        nearest_rev_id = self._node_matrix_rev.ids[nearest_index]

        dist = np.linalg.norm(states_rev[nearest_index] - for_pos)
        if dist <= self.get_rk():
            rev_node = self.tree_rev.nodes[nearest_rev_id]['value']
            score = dist + rev_node.cost_so_far  # d_X(x_for, x_closest) + h(x_closest)
            self.q_priority.add_or_update(for_node_id, score)

    def _select_best_extension_candidate(self, parent_node, random_point, is_forward=True, num_trials=None):
        best_candidate = None
        best_score = float('inf')

        trials = num_trials if num_trials is not None else self.num_extension_trials

        # Embed random_point once under paper's d_X; we score every MC
        # candidate by d_X(new_state, random_point) = ||embed(new) - embed(rand)||.
        rand_embed = self._metric_embed(random_point)

        # Per paper Alg. 9, the reverse tree also extends via BestInputProp.
        # Both trees pull (action, duration) from the shared library and
        # re-propagate online from the parent (forward time for the forward
        # tree, reverse time for the reverse tree).
        use_maneuver_lib = self.edge_bundle is not None

        if use_maneuver_lib:
            # Per-node edge caching: query KD-tree only on first visit
            if parent_node.edge_bundle_indices is None:
                query = self.get_eb_kd_tree_query(parent_node.state)
                edge_ids = self.eb_kd_tree.radius_query(query, self.kd_tree_delta_radius)
                l = min(len(edge_ids), self.max_num_edges_per_node)
                parent_node.edge_bundle_indices = edge_ids[:l]
                parent_node.edge_bundle_mask = np.full((l,), False, dtype=bool)

            curr_edge_indices = parent_node.edge_bundle_indices
            curr_edge_mask = parent_node.edge_bundle_mask

            # Find untried edges
            untried = np.where(~curr_edge_mask)[0]
            if len(untried) > 0:
                num_samples = min(trials, len(untried))
                random_slots = self.rng.choice(len(untried), size=num_samples, replace=False)

                for slot in random_slots:
                    mask_index = untried[slot]
                    edge_bundle_index = curr_edge_indices[mask_index]
                    random_action = self.edge_bundle.actions[edge_bundle_index]
                    random_time = self.edge_bundle.timesteps[edge_bundle_index]

                    num_record_steps = max(1, round(abs(random_time) / self.minimum_time_step))
                    eval_time = random_time if is_forward else -random_time
                    new_state, path_to_new_state = self.agent.get_next_state(
                        parent_node.state, random_action, eval_time, num_steps=num_record_steps
                    )

                    accept_new_node = self.isvalid(
                        path_to_new_state, self.agent.radius, self.env.size,
                        self.static_circular_obstacles, self.static_rectangular_obstacles,
                        self.dynamic_agent_obstacles, self.env.obstacle_buffer, self.env.boundary_buffer,
                        parent_node.time_elapsed, abs(random_time), self.minimum_time_step
                    )

                    # Mark edge as tried regardless of result
                    curr_edge_mask[mask_index] = True

                    if not accept_new_node:
                        continue

                    score = float(np.linalg.norm(self._metric_embed(new_state) - rand_embed))

                    if score < best_score:
                        best_score = score
                        best_candidate = (new_state, path_to_new_state, random_action, abs(random_time))

                return best_candidate

        # Fallback to pure random online integration if no maneuver library or not forward
        for _ in range(trials):
            random_action = self.agent.get_random_action(self.rng)
            random_time = self.get_time()

            num_record_steps = max(1, round(abs(random_time) / self.minimum_time_step))
            eval_time = random_time if is_forward else -random_time

            new_state, path_to_new_state = self.agent.get_next_state(
                parent_node.state, random_action, eval_time, num_steps=num_record_steps
            )

            if is_forward:
                accept_new_node = self.isvalid(
                    path_to_new_state, self.agent.radius, self.env.size,
                    self.static_circular_obstacles, self.static_rectangular_obstacles,
                    self.dynamic_agent_obstacles, self.env.obstacle_buffer, self.env.boundary_buffer,
                    parent_node.time_elapsed, random_time, self.minimum_time_step
                )
            else:
                 accept_new_node = self.isvalid(
                    path_to_new_state, self.agent.radius, self.env.size,
                    self.static_circular_obstacles, self.static_rectangular_obstacles,
                    self.dynamic_agent_obstacles, self.env.obstacle_buffer, self.env.boundary_buffer,
                    parent_node.time_elapsed, abs(random_time), self.minimum_time_step
                )

            if not accept_new_node:
                continue

            score = float(np.linalg.norm(self._metric_embed(new_state) - rand_embed))

            if score < best_score:
                best_score = score
                best_candidate = (new_state, path_to_new_state, random_action, abs(random_time))

        return best_candidate

    def extend_tree_rev(self, parent_node_id, parent_node, random_point):
        # Algorithm 9: RevSrchFastExplore uses BestInputProp (N_B trials)
        best_candidate = self._select_best_extension_candidate(parent_node, random_point, is_forward=False, num_trials=self.num_extension_trials)
        if best_candidate is None: return
        new_state, path_to_new_state, random_action, random_time = best_candidate

        # Remember reverse tree nodes trace backward from goal. 
        # parent_action here means "action that takes new_state -> parent_node"
        # Since it's symmetric, cost_so_far is cost to reach goal from new_state
        edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action, random_time, path_to_new_state)
        total_time = parent_node.time_elapsed + random_time
        total_cost = parent_node.cost_so_far + edge_cost

        new_node_id = self.add_node_rev(new_state, parent_node_id, random_action, random_time, path_to_new_state, total_time, total_cost)
        self.update_priority_queue(self.tree_rev.nodes[new_node_id]['value'], new_node_id)


    def extend_tree_for(self, parent_node_id, parent_node, target_point, num_trials=None):
        if num_trials is None:
            num_trials = self.num_extension_trials
        best_candidate = self._select_best_extension_candidate(parent_node, target_point, is_forward=True, num_trials=num_trials)
        if best_candidate is None: return False
        new_state, path_to_new_state, random_action, random_time = best_candidate

        reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal_check, self.goal_radius, self.agent)

        if reached_goal_flag:
            edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action, random_time, path_to_new_state)
            total_elapsed_time = parent_node.time_elapsed + random_time
            total_cost = parent_node.cost_so_far + edge_cost
            new_node_id = self.add_node_for(new_state, parent_node_id, random_action, random_time, path_to_new_state, total_elapsed_time, total_cost)
            
            self.path_found = True
            self.goal_node_id = new_node_id
            self.path_cost = total_cost
            self.path_time = total_elapsed_time
            return True
        else:
            if goal_distance < self.threshold:
                total_elapsed_time = parent_node.time_elapsed
                for (index, intermediate_state) in enumerate(path_to_new_state):
                    total_elapsed_time += self.minimum_time_step
                    goal_flag, d = self.reached_goal(intermediate_state, self.goal_check, self.goal_radius, self.agent)
                    if goal_flag:
                        modified_edge_time = total_elapsed_time - parent_node.time_elapsed
                        new_path_to_new_state = path_to_new_state[:index+1]
                        edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action, modified_edge_time, new_path_to_new_state)
                        total_cost = parent_node.cost_so_far + edge_cost
                        new_node_id = self.add_node_for(intermediate_state, parent_node_id, random_action, modified_edge_time, new_path_to_new_state, total_elapsed_time, total_cost)
                        self.path_found = True
                        self.goal_node_id = new_node_id
                        self.path_cost = total_cost
                        self.path_time = total_elapsed_time
                        return True

            edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action, random_time, path_to_new_state)
            total_elapsed_time = parent_node.time_elapsed + random_time
            total_cost = parent_node.cost_so_far + edge_cost

            new_node_id = self.add_node_for(new_state, parent_node_id, random_action, random_time, path_to_new_state, total_elapsed_time, total_cost)
            self.insert_priority_queue(self.tree_for.nodes[new_node_id]['value'], new_node_id)
            return True

    def plan_path(self):
        self.reset_tree()
        self.add_node_for(self.start, -1, None, None, None, 0.0, 0.0)
        
        # Add a node representing the goal to reverse tree
        self.add_node_rev(self.goal, -1, None, None, None, 0.0, 0.0)

        curr_num_steps = 0
        start_time = time.time()
        
        while curr_num_steps <= self.max_iter:
            
            # Step 1: Reverse Tree Expansion
            rand_rev = self.sample_random_point()
            rev_nearest_id, rev_nearest_node = self.get_nearest_node_rev(rand_rev)
            self.extend_tree_rev(rev_nearest_id, rev_nearest_node, rand_rev)

            # Step 2: Forward Tree Expansion (Algorithm 1, lines 12-32)
            #
            # Paper Alg. 1 spec:
            #   if c_rand < q : ForSrchExploit (Alg 4); if εfor = NULL, fall
            #                   back to ForSrchFastExplore (Alg 5).
            #   if (c_rand >= q) OR (εfor = NULL after the above):
            #                   ForSrchRandomExplore (Alg 7).
            # The random-explore arm is BOTH the c_rand >= q branch and the
            # last-resort fallback when exploit + fast-explore both fail —
            # see the paper text on p. 3641 ("εfor = NULL implies both
            # exploitation and fast exploration were unsuccessful").
            c_rand = self.rng.uniform(0, 1)
            e_for_success = False

            if c_rand < self.exploitation_ratio:
                # Exploitation attempt (Algorithm 4: ForSrchExploit)
                if not self.q_priority.is_empty():
                    for_pop_id, _ = self.q_priority.pop()
                    if for_pop_id is not None:
                        for_pop_node = self.tree_for.nodes[for_pop_id]['value']

                        # Find best reverse node target: argmin g(x_pop) + d_X + h(x)
                        # States are already d_X-embedded via _metric_embed.
                        states_rev = self._node_matrix_rev.get_valid_matrix()
                        for_pos = self._metric_embed(for_pop_node.state)

                        diffs = states_rev - for_pos
                        dists = np.linalg.norm(diffs, axis=1)
                        valid_indices = np.where(dists <= self.get_rk())[0]

                        best_target_score = float('inf')
                        target_point = None

                        for i in valid_indices:
                            rev_node_id = int(self._node_matrix_rev.ids[i])
                            # Fix B: skip already-reached rev nodes so the
                            # exploit advances up the reverse-tree chain.
                            if rev_node_id in self.consumed_rev_ids:
                                continue
                            rev_node = self.tree_rev.nodes[rev_node_id]['value']
                            score = for_pop_node.cost_so_far + dists[i] + rev_node.cost_so_far
                            if score < best_target_score:
                                best_target_score = score
                                target_point = rev_node.state

                        if target_point is not None:
                            e_for_success = self.extend_tree_for(
                                for_pop_id, for_pop_node, target_point,
                                num_trials=self.num_extension_trials)
                            if e_for_success:
                                # Fix B: mark all rev nodes within delta_reach
                                # of the new forward node as consumed.
                                new_for_id = self.last_added_node_id_for
                                new_for_state = self.tree_for.nodes[new_for_id]['value'].state
                                self._mark_consumed_rev_near(new_for_state)

                # Fallback to fast exploration (Algorithm 5: ForSrchFastExplore)
                # Per paper Alg. 1 line 6: only inside the c_rand < q branch.
                if not e_for_success:
                    rand_for = self.sample_random_point()
                    for_nearest_id, for_nearest_node = self.get_nearest_node_for(rand_for)
                    e_for_success = self.extend_tree_for(
                        for_nearest_id, for_nearest_node, rand_for,
                        num_trials=self.num_extension_trials)

            # Paper Alg. 1 line 28: random-explore when c_rand >= q OR
            # εfor = NULL (i.e., both exploit and fast-explore failed).
            if c_rand >= self.exploitation_ratio or not e_for_success:
                rand_for = self.sample_random_point()
                for_nearest_id, for_nearest_node = self.get_nearest_node_for(rand_for)
                self.extend_tree_for(
                    for_nearest_id, for_nearest_node, rand_for,
                    num_trials=1)


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
            path_controls[path_length] = rrt_node.parent_action if rrt_node.parent_action is not None else np.zeros(control_length)
            path_timesteps[path_length] = rrt_node.parent_action_duration if rrt_node.parent_action_duration is not None else 0.0
            path_length+=1
            node_id = rrt_node.parent_id

        ids = path_rrt_node_ids[:path_length][::-1]
        states = path_states[:path_length][::-1]
        controls = path_controls[:path_length-1][::-1]
        timesteps = path_timesteps[:path_length-1][::-1]
        return ids, states, controls, timesteps

    def get_path(self):
        if(self.path_found == False):
            print("Path can't be found because goal hasn't been reached!")        
            return np.empty(0,dtype=np.int16), \
                np.empty((0,self.agent.state_length), dtype=np.float64), \
                np.empty((0,self.agent.action_length), dtype=np.float64), \
                np.empty(0, dtype=np.float64)
        else:           
            return self.get_path_to_node_id(self.goal_node_id)
