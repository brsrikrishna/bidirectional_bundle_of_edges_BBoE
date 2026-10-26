import numpy as np
import time
from utils import get_dtype_from_input,find_roundoff_decimal_digits, get_nearest_index, \
    get_nearest_index, compact_matrix, check_dynamic_collisions_to_end, \
    euclidean_distance_numba_with_l
from numba.typed import List
from numba import types, njit
from rrt import TreeNode, RRT  # Assuming TreeNode and RRT are defined in rrt.py
from typing import Optional, Tuple

# -----------------------------
# Metrics & utilities (Numba)
# -----------------------------
@njit
def l2(a: np.ndarray, b: np.ndarray) -> float:
    s = 0.0
    for i in range(a.shape[0]):
        d = a[i] - b[i]
        s += d * d
    return np.sqrt(s)

@njit
def l2_pos_scaled(a, b, pos_dims, scales):
    """Weighted Euclidean over the first pos_dims coordinates.
    scales is an ndarray of shape (pos_dims,); pass 1s for unweighted."""
    s = 0.0
    for i in range(pos_dims):
        d = (a[i] - b[i]) * scales[i]
        s += d * d
    return np.sqrt(s)

@njit
def get_best_active_near_index(states, costs, active, n_nodes, sample,
                               best_near_radius, pos_dims, scales):
    best = -1
    best_cost = 1e30
    for i in range(n_nodes):
        if active[i] == 1:
            if l2_pos_scaled(states[i], sample, pos_dims, scales) < best_near_radius:
                c = costs[i]
                if c < best_cost:
                    best = i
                    best_cost = c
    return best

@njit
def get_nearest_active_index(states, active, n_nodes, sample, pos_dims, scales):
    best = -1
    best_d = 1e30
    for i in range(n_nodes):
        if active[i] == 1:
            d = l2_pos_scaled(states[i], sample, pos_dims, scales)
            if d < best_d:
                best = i
                best_d = d
    return best

@njit
def get_best_or_nearest_active_index(states, costs, active, n_nodes, sample,
                                     best_near_radius, pos_dims, scales):
    """
    Return index of best-cost active node within best_near_radius of sample.
    If none, return nearest active node.
    """
    best = -1
    best_cost = 1e30
    nearest = -1
    nearest_d = 1e30

    for i in range(n_nodes):
        if active[i] == 1:
            d = l2_pos_scaled(states[i], sample, pos_dims, scales)
            if d < nearest_d:
                nearest = i
                nearest_d = d
            if d < best_near_radius:
                c = costs[i]
                if c < best_cost:
                    best = i
                    best_cost = c

    if best == -1:
        return nearest
    return best

@njit
def get_nearest_witness_index(w_states, n_w, x, pos_dims, scales):
    """
    Nearest witness by weighted distance; returns (index, distance).
    """
    if n_w == 0:
        return -1, 1e30
    best = 0
    best_d = l2_pos_scaled(w_states[0], x, pos_dims, scales)
    for i in range(1, n_w):
        d = l2_pos_scaled(w_states[i], x, pos_dims, scales)
        if d < best_d:
            best = i
            best_d = d
    return best, best_d


class TreeNodeMatrix:
    def __init__(self, initial_capacity, state_dim, action_dim, max_sub_path_length):
        self.id = np.full(initial_capacity, -1, dtype=np.int32)
        self.state = np.zeros((initial_capacity, state_dim), dtype=np.float64)
        self.parent = np.full(initial_capacity, -1, dtype=np.int32)
        self.parent_action = np.zeros((initial_capacity, action_dim), dtype=np.float64)
        self.action_duration = np.full(initial_capacity, -1, dtype=np.float32)
        self.path_from_parent = np.zeros((initial_capacity, max_sub_path_length, state_dim), dtype=np.float64)
        self.sub_path_length = np.full(initial_capacity, -1, dtype=np.int32)
        self.time_elapsed = np.full(initial_capacity, -1, dtype=np.float32)
        self.cost = np.full(initial_capacity, -1, dtype=np.float32)  # time cost
        self.count = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_sub_path_length = max_sub_path_length

    def append(self,node_id,state,parent,action,action_duration,path_from_parent,time_elapsed,cost):
        if self.count >= self.state.shape[0]:
            self._grow()
        curr_index = self.count
        self.id[curr_index] = node_id
        self.state[curr_index] = state
        self.parent[curr_index] = parent
        self.parent_action[curr_index] = action
        self.action_duration[curr_index] = action_duration
        len_path_from_parent = path_from_parent.shape[0]
        self.path_from_parent[curr_index][:len_path_from_parent] = path_from_parent
        self.sub_path_length[curr_index] = len_path_from_parent
        self.time_elapsed[curr_index] = time_elapsed
        self.cost[curr_index] = cost
        self.count += 1
        return curr_index

    def _grow(self):
        old_cap = self.state.shape[0]
        new_cap = 2 * old_cap

        new_id = np.full(new_cap, -1, dtype=np.int32)
        new_state = np.zeros((new_cap, self.state_dim), dtype=np.float64)
        new_parent = np.full(new_cap, -1, dtype=np.int32)
        new_parent_action = np.zeros((new_cap, self.action_dim), dtype=np.float64)
        new_action_duration = np.full(new_cap, -1, dtype=np.float32)
        new_path_from_parent = np.zeros((new_cap, self.max_sub_path_length, self.state_dim),
                                             dtype=np.float64)
        new_sub_path_length = np.full(new_cap, -1, dtype=np.int32)
        new_time_elapsed = np.full(new_cap, -1, dtype=np.float32)
        new_cost = np.full(new_cap, -1, dtype=np.float32)

        new_id[:old_cap] = self.id
        new_state[:old_cap] = self.state
        new_parent[:old_cap] = self.parent
        new_parent_action[:old_cap] = self.parent_action
        new_action_duration[:old_cap] = self.action_duration
        new_path_from_parent[:old_cap] = self.path_from_parent
        new_sub_path_length[:old_cap] = self.sub_path_length
        new_time_elapsed[:old_cap] = self.time_elapsed
        new_cost[:old_cap] = self.cost

        self.id = new_id
        self.state = new_state
        self.parent = new_parent
        self.parent_action = new_parent_action
        self.action_duration = new_action_duration
        self.path_from_parent = new_path_from_parent
        self.sub_path_length = new_sub_path_length
        self.time_elapsed = new_time_elapsed
        self.cost = new_cost

    def get_valid_states(self):
        return self.state[:self.count]

    def get_valid_ids(self):
        return self.id[:self.count]


class SSTNodeMatrix(TreeNodeMatrix):
    def __init__(self, initial_capacity, state_dim, action_dim, max_sub_path_length,
                 embedded_dim=None):
        super().__init__(initial_capacity, state_dim, action_dim, max_sub_path_length)
        self.active = np.full(initial_capacity, -1, dtype=np.uint8)
        self.num_children = np.zeros(initial_capacity, dtype=np.int32)
        self.embedded_dim = embedded_dim if embedded_dim is not None else state_dim
        # Parallel cache of d_X-embedded states (for paper-faithful distance
        # computations in BestNear/NN/witness queries).
        self.embedded_state = np.zeros((initial_capacity, self.embedded_dim),
                                       dtype=np.float64)

    def append(self,node_id,state,parent,action,action_duration,path_from_parent,
               time_elapsed,cost, embedded_state=None):
        if self.count >= self.state.shape[0]:
            self._grow()
        curr_index = self.count
        self.id[curr_index] = node_id
        self.state[curr_index] = state
        if embedded_state is not None:
            self.embedded_state[curr_index] = embedded_state
        else:
            # Fallback: use first embedded_dim dims (identity embedding)
            self.embedded_state[curr_index] = state[:self.embedded_dim]
        self.parent[curr_index] = parent
        self.parent_action[curr_index] = action
        self.action_duration[curr_index] = action_duration
        len_path_from_parent = path_from_parent.shape[0]
        self.path_from_parent[curr_index][:len_path_from_parent] = path_from_parent
        self.sub_path_length[curr_index] = len_path_from_parent
        self.time_elapsed[curr_index] = time_elapsed
        self.cost[curr_index] = cost
        self.active[curr_index] = 0  # Mark as inactive
        self.num_children[curr_index] = 0
        self.count += 1
        return curr_index

    def _grow(self):
        old_cap = self.state.shape[0]
        new_cap = 2 * old_cap

        new_id = np.full(new_cap, -1, dtype=np.int32)
        new_state = np.zeros((new_cap, self.state_dim), dtype=np.float64)
        new_embedded_state = np.zeros((new_cap, self.embedded_dim), dtype=np.float64)
        new_parent = np.full(new_cap, -1, dtype=np.int32)
        new_parent_action = np.zeros((new_cap, self.action_dim), dtype=np.float64)
        new_action_duration = np.full(new_cap, -1, dtype=np.float32)
        new_path_from_parent = np.zeros((new_cap, self.max_sub_path_length, self.state_dim),
                                             dtype=np.float64)
        new_sub_path_length = np.full(new_cap, -1, dtype=np.int32)
        new_time_elapsed = np.full(new_cap, -1, dtype=np.float32)
        new_cost = np.full(new_cap, -1, dtype=np.float32)
        new_active = np.full(new_cap, -1, dtype=np.uint8)
        new_num_children = np.zeros(new_cap, dtype=np.int32)

        new_id[:old_cap] = self.id
        new_state[:old_cap] = self.state
        new_embedded_state[:old_cap] = self.embedded_state
        new_parent[:old_cap] = self.parent
        new_parent_action[:old_cap] = self.parent_action
        new_action_duration[:old_cap] = self.action_duration
        new_path_from_parent[:old_cap] = self.path_from_parent
        new_sub_path_length[:old_cap] = self.sub_path_length
        new_time_elapsed[:old_cap] = self.time_elapsed
        new_cost[:old_cap] = self.cost
        new_active[:old_cap] = self.active
        new_num_children[:old_cap] = self.num_children

        self.id = new_id
        self.state = new_state
        self.embedded_state = new_embedded_state
        self.parent = new_parent
        self.parent_action = new_parent_action
        self.action_duration = new_action_duration
        self.path_from_parent = new_path_from_parent
        self.sub_path_length = new_sub_path_length
        self.time_elapsed = new_time_elapsed
        self.cost = new_cost
        self.active = new_active
        self.num_children = new_num_children

    def get_valid_states(self):
        return self.state[:self.count]

    def get_valid_ids(self):
        return self.id[:self.count]


class SSTWitnessMatrix:
    def __init__(self, initial_capacity, state_dim, embedded_dim=None):
        self.id = np.full(initial_capacity, -1, dtype=np.int32)
        self.state = np.zeros((initial_capacity, state_dim), dtype=np.float64)
        self.rep_index = np.full(initial_capacity, -1, dtype=np.int32)
        self.count = 0
        self.state_dim = state_dim
        self.embedded_dim = embedded_dim if embedded_dim is not None else state_dim
        self.embedded_state = np.zeros((initial_capacity, self.embedded_dim),
                                       dtype=np.float64)

    def append(self,node_id,state,rep_index, embedded_state=None):
        if self.count >= self.state.shape[0]:
            self._grow()
        self.id[self.count] = node_id
        self.state[self.count] = state
        if embedded_state is not None:
            self.embedded_state[self.count] = embedded_state
        else:
            self.embedded_state[self.count] = state[:self.embedded_dim]
        self.rep_index[self.count] = rep_index
        self.count += 1

    def _grow(self):
        old_cap = self.state.shape[0]
        new_cap = 2 * old_cap

        new_id = np.full(new_cap, -1, dtype=np.int32)
        new_state = np.zeros((new_cap, self.state_dim), dtype=np.float64)
        new_embedded_state = np.zeros((new_cap, self.embedded_dim), dtype=np.float64)
        new_rep_idx = np.full(new_cap, -1, dtype=np.int32)

        new_id[:old_cap] = self.id
        new_state[:old_cap] = self.state
        new_embedded_state[:old_cap] = self.embedded_state
        new_rep_idx[:old_cap] = self.rep_index

        self.id = new_id
        self.state = new_state
        self.embedded_state = new_embedded_state
        self.rep_index = new_rep_idx

    def get_valid_ids(self):
        return self.id[:self.count]

    def get_valid_states(self):
        return self.state[:self.count]
    
    def get_valid_representatives(self):
        return self.rep_index[:self.count]


class SST(RRT):
    def __init__(self, * , start, goal, goal_radius, env, agent,
                    use_fixed_sampling_time=True,
                    sampling_time_step=1.0,
                    minimum_time_step=0.1,
                    max_iter=1000,
                    planning_time=10.0,
                    isvalid_function,
                    cost_function,
                    reached_goal_function,
                    random_point_function,
                    udf_seed = 77,
                    goal_bias = 0.0,
                    best_near_radius=2.0,
                    prune_radius = 0.5,
                    debug_flag = False,
                    print_logs = False,
                    dynamic_obstacles=None,
                    edge_bundle=None,
                    eb_kd_tree=None,
                    get_eb_kd_tree_query=None,
                    kd_tree_delta_radius=0.5,
                    translate_function=None,
                    num_extension_trials=1
                    ):


        self.start = np.array(start)
        self.goal = np.array(goal)
        self.goal_radius = goal_radius
        self.goal_state_length = len(goal)
        self.env = env
        self.agent = agent
        self.use_fixed_sampling_time = use_fixed_sampling_time #Bool to use fixed sampling time or not
        self.max_sample_T = sampling_time_step #Max time to apply the control for in propogating each edge
        self.minimum_time_step = minimum_time_step #Smallest time duration between consecutive agent positions
        self.max_iter = max_iter #Max number of iterations for RRT
        self.planning_time = planning_time #Time to plan the path in seconds
        self.isvalid = isvalid_function #Function to check if the path is valid
        self.cost = cost_function #Function to calculate the cost of a edge
        self.reached_goal = reached_goal_function
        self.get_random_point = random_point_function
        self.best_near_radius = best_near_radius
        self.prune_radius = prune_radius
        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.goal_bias = goal_bias
        self.debug_flag = debug_flag
        self.print_logs = print_logs

        # Maneuver library support
        self.edge_bundle = edge_bundle
        self.eb_kd_tree = eb_kd_tree
        self.get_eb_kd_tree_query = get_eb_kd_tree_query
        self.kd_tree_delta_radius = kd_tree_delta_radius
        self.translate_function = translate_function
        self.num_extension_trials = num_extension_trials

        self.goal_node_id = None
        self.path_found = False
        self.path_time = 0.0
        self.path_cost = float('inf')
        self.last_added_node_id = -1
        self.last_added_witness_id = -1
        self.threshold = 3.0

        # Preprocess the static obstacles for fast collision checking
        # This is done only once at the start of the planning
        self.static_circular_obstacles = self.env.static_circular_obstacles
        self.static_rectangular_obstacles = self.env.static_rectangular_obstacles
        # Dynamic obstacles are not preprocessed, they are updated in real-time
        self.dynamic_agent_obstacles = (
            List.empty_list(types.Array(types.float64, 2, "C"))
            if dynamic_obstacles is None else dynamic_obstacles)

        #Set the variable for distance metric state size that will be used for distance calculations
        if hasattr(self.agent, 'distance_metric_state_size'):
            self.distance_metric_state_size = self.agent.distance_metric_state_size
        else:
            self.distance_metric_state_size = 2

        # Optional per-dimension scaling for the nearest-neighbor / witness metric
        # (used when an agent provides `metric_state_scales`). Default: unweighted Euclidean.
        _agent_scales = getattr(self.agent, 'metric_state_scales', None)
        if _agent_scales is not None:
            self._metric_scales = np.asarray(_agent_scales, dtype=np.float64)[:self.distance_metric_state_size]
        else:
            self._metric_scales = np.ones(self.distance_metric_state_size, dtype=np.float64)

        #Set number of roundoff digits for time sampling
        self.roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)

        # Use Matrices for preallocated fast storage and for Numba optimization
        state_length = len(self.start)
        agent_action = agent.get_random_action(agent.rng)
        self.dummy_root_action = np.full(np.array(agent_action).shape, -1, dtype=np.float64)
        action_length = len(agent_action)
        max_sub_path_length = int(round(self.max_sample_T/self.minimum_time_step,
                                        self.roundoff_digits))
        self.state_length = state_length
        self.action_length = action_length
        self.max_sub_path_length = max_sub_path_length
        # Determine embedded dim by probing the agent's metric_embed_state
        # (falls back to distance_metric_state_size for agents without the hook).
        self._embedded_dim = len(self._metric_embed(self.start))
        self._ones_scales = np.ones(self._embedded_dim, dtype=np.float64)
        self._node_matrix = SSTNodeMatrix(initial_capacity=1024, state_dim=state_length,
                            action_dim=action_length, max_sub_path_length=max_sub_path_length,
                            embedded_dim=self._embedded_dim)
        self.type_node_matrix = type(self._node_matrix)
        self._witness_matrix = SSTWitnessMatrix(initial_capacity=1024, state_dim=state_length,
                                                embedded_dim=self._embedded_dim)
        self.type_witness_matrix = type(self._witness_matrix)


    def add_sst_node(self, state, parent_index, parent_action, parent_action_duration,
                    path_from_parent, time_elapsed, cost):
        # Add a new node to the SST
        new_node_id = self.last_added_node_id + 1
        embedded = self._metric_embed(state)  # paper d_X embedding
        added_matrix_index = self._node_matrix.append(new_node_id, state, parent_index,
                                    parent_action, parent_action_duration, path_from_parent,
                                    round(time_elapsed, self.roundoff_digits), cost,
                                    embedded_state=embedded)
        self.last_added_node_id = new_node_id

        # Track parent-child relationship for cascade pruning
        if parent_index >= 0:
            self._node_matrix.num_children[parent_index] += 1

        return added_matrix_index


    def add_sst_witness(self,state: np.ndarray,rep_index: int) -> int:
        # Add a new witness to the SST
        new_witness_id = self.last_added_witness_id + 1
        embedded = self._metric_embed(state)  # paper d_X embedding
        added_matrix_index = self._witness_matrix.append(new_witness_id, state[:],
                                    rep_index, embedded_state=embedded)
        self.last_added_witness_id = new_witness_id

        return added_matrix_index


    def _cascade_prune(self, node_index):
        """
        Cascade-prune childless inactive nodes up the tree.
        Per the original SST paper (Li, Littlefield, Bekris 2016),
        when a node becomes inactive, if it has no children it is
        removed and its parent's child count is decremented. This
        repeats up the tree.
        """
        sst_nodes = self._node_matrix
        while node_index >= 0:
            # Active nodes or nodes with children cannot be pruned
            if sst_nodes.active[node_index] == 1:
                break
            if sst_nodes.num_children[node_index] > 0:
                break
            # This node is inactive with no children — prune it
            parent = sst_nodes.parent[node_index]
            if parent >= 0:
                sst_nodes.num_children[parent] -= 1
            # Move up to check parent
            node_index = parent

    def _best_input_prop(self, parent_state, target_point, parent_time_elapsed=0.0):
        """
        BestInputProp: Generate num_extension_trials candidate trajectories,
        select the best valid one closest to target_point by d_X.
        Uses maneuver library when available, otherwise online integration.
        Returns (new_state, path_to_new_state, random_action, random_time) or None.
        """
        best_candidate = None
        best_score = float('inf')
        trials = self.num_extension_trials

        # Embed target_point under paper's d_X once; score each MC candidate
        # by d_X(new_state, target_point).
        target_embed = self._metric_embed(target_point)

        # Use maneuver library: pull (action, duration) from stored edges and
        # re-propagate online from the parent.
        use_maneuver_lib = self.edge_bundle is not None

        if use_maneuver_lib:
            query = self.get_eb_kd_tree_query(parent_state)
            edge_ids = self.eb_kd_tree.radius_query(query, self.kd_tree_delta_radius)
            if len(edge_ids) > 0:
                l = min(len(edge_ids), 3000)
                valid_edge_indices = edge_ids[:l]

                num_samples = min(trials, len(valid_edge_indices))
                random_slots = self.rng.choice(len(valid_edge_indices), size=num_samples, replace=False)

                for slot in random_slots:
                    edge_bundle_index = valid_edge_indices[slot]
                    random_action = self.edge_bundle.actions[edge_bundle_index]
                    random_time = self.edge_bundle.timesteps[edge_bundle_index]

                    num_record_steps = max(1, round(abs(random_time) / self.minimum_time_step))
                    new_state, path_to_new_state = self.agent.get_next_state(
                        parent_state, random_action, random_time,
                        num_steps=num_record_steps
                    )

                    accept_new_node = self.isvalid(
                        path_to_new_state, self.agent.radius, self.env.size,
                        self.static_circular_obstacles, self.static_rectangular_obstacles,
                        self.dynamic_agent_obstacles, self.env.obstacle_buffer, self.env.boundary_buffer,
                        parent_time_elapsed, parent_time_elapsed + abs(random_time), self.minimum_time_step
                    )
                    if not accept_new_node:
                        continue

                    score = float(np.linalg.norm(self._metric_embed(new_state) - target_embed))
                    if score < best_score:
                        best_score = score
                        best_candidate = (new_state, path_to_new_state, random_action, abs(random_time))

                return best_candidate

        # Fallback: online integration with random actions
        for _ in range(trials):
            random_action = self.agent.get_random_action(self.rng)
            random_time = self.get_time()
            num_record_steps = round(random_time / self.minimum_time_step)
            new_state, path_to_new_state = self.agent.get_next_state(
                parent_state, random_action, random_time, num_steps=num_record_steps)

            accept_new_node = self.isvalid(
                path_to_new_state, self.agent.radius, self.env.size,
                self.static_circular_obstacles, self.static_rectangular_obstacles,
                self.dynamic_agent_obstacles, self.env.obstacle_buffer, self.env.boundary_buffer,
                parent_time_elapsed, parent_time_elapsed + random_time, self.minimum_time_step
            )
            if not accept_new_node:
                continue

            score = float(np.linalg.norm(self._metric_embed(new_state) - target_embed))
            if score < best_score:
                best_score = score
                best_candidate = (new_state, path_to_new_state, random_action, random_time)

        return best_candidate

    def extend_tree(self, parent_node_index, random_point):
        """
        Propagate from parent using BestInputProp.

        Returns (goal_node_index, None) if goal is reached (node added to tree),
        or (-1, candidate_tuple) for a valid non-goal candidate (NOT yet added),
        or (-1, None) if all trials are invalid.

        candidate_tuple = (new_state, path_to_new_state, random_action,
                           random_time, total_elapsed_time, total_cost)
        """

        #Get values for the parent_node
        sst_nodes = self._node_matrix
        parent_state = sst_nodes.state[parent_node_index]
        parent_time_elapsed = sst_nodes.time_elapsed[parent_node_index]
        parent_cost = sst_nodes.cost[parent_node_index]

        result = self._best_input_prop(parent_state, random_point, parent_time_elapsed)
        if result is None:
            if self.debug_flag:
                print("~~~~~~~~~~All extension trials invalid~~~~~~~~~~")
            return -1, None
        new_state, path_to_new_state, random_action, random_time = result

        reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal,
                                                        self.goal_radius, self.agent)
        if reached_goal_flag:
            if check_dynamic_collisions_to_end(new_state, self.agent.radius,
                                               self.dynamic_agent_obstacles,
                                               self.env.obstacle_buffer,
                                               parent_time_elapsed + random_time,
                                               self.minimum_time_step):
                if self.debug_flag:
                    print("Goal state will collide with high-priority agent. Trying again!")
                return -1, None
            edge_cost = self.cost(self.env, self.agent, parent_state, random_action,
                        random_time, path_to_new_state)
            total_elapsed_time = parent_time_elapsed + random_time
            total_cost = parent_cost + edge_cost

            new_node_index = self.add_sst_node(new_state, parent_node_index, random_action, random_time,
                                            path_to_new_state, total_elapsed_time, total_cost)
            self.path_found = True
            if self.debug_flag:
                print("Goal Reached! Path found for ",self.agent.id)
            self.goal_node_id = new_node_index
            self.path_cost = total_cost
            self.path_time = total_elapsed_time
            return new_node_index, None
        else:
            if goal_distance < self.threshold:
                total_elapsed_time = parent_time_elapsed
                for (index, intermediate_state) in enumerate(path_to_new_state):
                    total_elapsed_time += self.minimum_time_step
                    goal_flag, d = self.reached_goal(intermediate_state, self.goal,
                                                     self.goal_radius, self.agent)
                    if goal_flag:
                        if check_dynamic_collisions_to_end(intermediate_state, self.agent.radius,
                                                self.dynamic_agent_obstacles,
                                                self.env.obstacle_buffer,
                                                total_elapsed_time,
                                                self.minimum_time_step):
                                if self.print_logs:
                                    print("Goal state will collide with high-priority agent. Trying again!")
                                return -1, None
                        modified_edge_time = total_elapsed_time - parent_time_elapsed
                        new_path_to_new_state = path_to_new_state[:index+1]
                        edge_cost = self.cost(self.env, self.agent, parent_state, random_action,
                                            modified_edge_time, new_path_to_new_state)
                        total_cost = parent_cost + edge_cost
                        new_node_index = self.add_sst_node(intermediate_state, parent_node_index, random_action,
                                                        modified_edge_time, new_path_to_new_state,
                                                        total_elapsed_time, total_cost)
                        self.path_found = True
                        if self.debug_flag:
                            print("Goal Reached! Path found for ",self.agent.id)
                        self.goal_node_id = new_node_index
                        self.path_cost = total_cost
                        self.path_time = total_elapsed_time
                        return new_node_index, None

            # Non-goal candidate: compute cost but do NOT add to tree yet
            edge_cost = self.cost(self.env, self.agent, parent_state, random_action,
                                    random_time, path_to_new_state)
            total_elapsed_time = parent_time_elapsed + random_time
            total_cost = parent_cost + edge_cost

            return -1, (new_state, path_to_new_state, random_action,
                        random_time, total_elapsed_time, total_cost,
                        parent_node_index)


    def get_path_to_node_id(self, goal_node_id):

        # Return node ids, agent state and the executed control from start to goal 

        node_matrix = self._node_matrix
        state_length = node_matrix.state_dim
        action_length = node_matrix.action_dim
        max_nodes = node_matrix.count

        path_node_ids = np.empty(max_nodes, dtype=np.int32)
        path_states = np.empty((max_nodes, state_length), dtype=np.float64)
        path_controls = np.empty((max_nodes, action_length), dtype=np.float64)
        path_timesteps = np.empty(max_nodes, dtype=np.float64)
        node_index = goal_node_id
        path_length = 0

        while node_index != -1:
            # print("Node ID: ", node_index)
            path_node_ids[path_length] = node_matrix.id[node_index]
            path_states[path_length] = node_matrix.state[node_index]
            path_controls[path_length] = node_matrix.parent_action[node_index]
            path_timesteps[path_length] = node_matrix.action_duration[node_index]
            path_length += 1
            node_index = node_matrix.parent[node_index]

        """
        Note: 
        1) Slicing a numpy array doesn't cause new array allocation. It just returns a view
        of the original array.
        2) Reversing the numpy array using [::-1] doesn't cause new array allocation. 
        It also just returns a view of the original array.
        3) Slicing and reversing the numpy array doesn't cause any allocation!!
        """    

        ids = path_node_ids[:path_length][::-1]
        states = path_states[:path_length][::-1]
        controls = path_controls[:path_length-1][::-1]
        timesteps = path_timesteps[:path_length-1][::-1]
        return ids, states, controls, timesteps


    def get_high_resolution_path_numpy_array(self):
        """
        Return a high-resolution path from start to goal using SST matrices.
        """
        if not self.path_found:
            print("Path can't be found because goal hasn't been reached!")
            return np.empty((0, self.agent.state_length), dtype=np.float64)

        total_path_time = self.path_time
        min_time_step = self.minimum_time_step
        path_length = int(round(total_path_time / min_time_step, self.roundoff_digits)) + 1
        path_states = np.empty((path_length, self.agent.state_length), dtype=np.float64)

        node_index = self.goal_node_id
        curr_index = path_length - 1

        node_matrix = self._node_matrix

        while node_index != -1:  # Repeat until you reach the root node
            path_to_node_len = node_matrix.sub_path_length[node_index]
            start_index = curr_index - path_to_node_len + 1
            path_states[start_index:curr_index + 1] = node_matrix.path_from_parent[node_index, :path_to_node_len]
            curr_index -= path_to_node_len
            node_index = node_matrix.parent[node_index]

        # Fill remaining states with start state
        path_states[:curr_index + 1] = self.start

        return path_states


    def get_tree_structure(self):
        return (self._node_matrix, self._witness_matrix)


    def reset_tree(self, some_existing_tree=None):
        if some_existing_tree == None:
            self._node_matrix = self.type_node_matrix(initial_capacity=1024,
                            state_dim=self.state_length,action_dim=self.action_length,
                            max_sub_path_length=self.max_sub_path_length,
                            embedded_dim=self._embedded_dim)
            self._witness_matrix = self.type_witness_matrix(initial_capacity=1024,
                            state_dim=self.state_length,
                            embedded_dim=self._embedded_dim)
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            self.last_added_node_id = -1
            self.last_added_witness_id = -1
            self._node_matrix.count = 0
            self._witness_matrix.count = 0
        else:
            self._node_matrix = some_existing_tree[0]
            self._witness_matrix = some_existing_tree[1]
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            self.last_added_node_id = self._node_matrix.count - 1
            self.last_added_witness_id = self._witness_matrix.count - 1

    def plan_path(self):
        """
        Plan a path using SST algorithm from scratch.
        """

        self.path_found = False
        self.goal_node_id = None
        self.last_added_node_id = -1
        self.last_added_witness_id = -1
        self.reset_tree()
        first_node_state = self.start
        state_length = len(first_node_state)
        empty_path_from_parent = np.empty((0, state_length), dtype=np.float64)
        new_sst_node_index = self.add_sst_node(first_node_state, -1, self.dummy_root_action, 
                                               0.0, empty_path_from_parent, 0.0, 0.0)
        self._node_matrix.active[new_sst_node_index] = 1
        self.add_sst_witness(first_node_state, new_sst_node_index)

        curr_num_steps = 0
        sst_nodes = self._node_matrix
        sst_witnesses = self._witness_matrix

        start_time = time.time()
        
        while curr_num_steps<=self.max_iter:
            if self.debug_flag:
                print("*************************************")
                print("Iteration in SST algorithm: ", curr_num_steps)

            # Sample a random point in the environment
            random_point = self.sample_random_point()
            random_point_embed = self._metric_embed(random_point)  # paper d_X

            #Find the nearest active node in the tree within best_near_radius
            #distance from the random point (under paper d_X).
            nearest_node_index = get_best_or_nearest_active_index(
                sst_nodes.embedded_state, sst_nodes.cost,
                sst_nodes.active, sst_nodes.count, random_point_embed,
                self.best_near_radius, self._embedded_dim,
                self._ones_scales)

            if nearest_node_index==-1:
                # No active node close to x_rand; skip this iteration (SST behavior)
                nearest_node_index = get_nearest_active_index(
                    sst_nodes.embedded_state, sst_nodes.active, sst_nodes.count,
                    random_point_embed, self._embedded_dim,
                    self._ones_scales)

            # Propagate from nearest node
            goal_index, candidate = self.extend_tree(nearest_node_index, random_point)

            if goal_index >= 0:
                # Goal was reached; node already added in extend_tree
                break

            if candidate is None:
                # All extension trials invalid; skip
                if self.path_found or (time.time() - start_time >= self.planning_time):
                    break
                curr_num_steps += 1
                continue

            # Unpack candidate (node NOT yet added to tree)
            (new_state, path_to_new_state, random_action,
             random_time, total_elapsed_time, total_cost,
             parent_idx) = candidate

            # Find nearest witness (under paper d_X)
            new_state_embed = self._metric_embed(new_state)
            nearest_witness_index, nearest_witness_distance = get_nearest_witness_index(
                sst_witnesses.embedded_state, sst_witnesses.count, new_state_embed,
                self._embedded_dim, self._ones_scales)

            if self.debug_flag:
                print("Nearest Witness Index: ", nearest_witness_index)
                print("Nearest Witness Distance: ", nearest_witness_distance)

            if nearest_witness_distance > self.prune_radius:
                # New witness region — add node and witness
                new_node_index = self.add_sst_node(new_state, parent_idx, random_action,
                                    random_time, path_to_new_state,
                                    total_elapsed_time, total_cost)
                sst_nodes.active[new_node_index] = 1
                self.add_sst_witness(new_state, new_node_index)
                if self.debug_flag:
                    print("New Node Added to the SST Tree: ", new_node_index)
            else:
                # Existing witness — check dominance
                curr_rep_index = sst_witnesses.rep_index[nearest_witness_index]
                if total_cost < sst_nodes.cost[curr_rep_index]:
                    # New node dominates — add it and deactivate old rep
                    new_node_index = self.add_sst_node(new_state, parent_idx, random_action,
                                        random_time, path_to_new_state,
                                        total_elapsed_time, total_cost)
                    sst_nodes.active[curr_rep_index] = 0
                    sst_witnesses.rep_index[nearest_witness_index] = new_node_index
                    sst_nodes.active[new_node_index] = 1
                    # Cascade-prune the old representative if it's now childless
                    self._cascade_prune(curr_rep_index)
                    if self.debug_flag:
                        print("New Node Added to the SST Tree: ", new_node_index)
                # else: new node is dominated — discard it entirely

            if self.path_found or (time.time() - start_time >= self.planning_time):
                # If the goal is reached or planning time exceeded, break the loop
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

