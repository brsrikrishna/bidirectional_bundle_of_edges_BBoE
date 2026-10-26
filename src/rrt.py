import networkx as nx
import numpy as np
import time
from utils import euclidean_distance_numba_with_l, \
    get_dtype_from_input,find_roundoff_decimal_digits,\
    get_nearest_index, compact_matrix
from numba.typed import List
from numba import types

"""
Numpy Matrix that stores the RRT nodes and is used for fast computation of nearest node.
It doubles dynamically when the capacity is reached.
"""
class DynamicMatrix:
    def __init__(self, initial_capacity, dim):
        self.matrix = np.zeros((initial_capacity, dim), dtype=np.float64)
        self.ids = np.full(initial_capacity, -1, dtype=np.int32)  # Store node ids
        # self.active = np.zeros(initial_capacity, dtype=np.bool_)  # Track active entries
        self.count = 0
        self.dim = dim

    def append(self, row, node_id):
        if self.count >= self.matrix.shape[0]:
            self._grow()
        self.matrix[self.count] = row
        self.ids[self.count] = node_id
        # self.active[self.count] = True
        self.count += 1

    def _grow(self):
        old_cap = self.matrix.shape[0]
        new_cap = 2 * old_cap
        new_matrix = np.zeros((new_cap, self.dim), dtype=np.float64)
        new_matrix[:old_cap] = self.matrix
        self.matrix = new_matrix

        new_ids = np.full(new_cap, -1, dtype=np.int32)
        new_ids[:old_cap] = self.ids
        self.ids = new_ids

        # new_active = np.zeros(new_cap, dtype=np.bool_)
        # new_active[:old_cap] = self.active
        # self.active = new_active

    def get_valid_matrix(self):
        return self.matrix[:self.count]

    def get_valid_ids(self):
        return self.ids[:self.count]

    # def get_active_entries(self):
    #     return self.active[:self.count]



class TreeNode:
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


"""
while True:
    sample a new point
    find the nearest node in the tree to the sample point
    extend the tree towards the sample point
    if the new node is in collision, continue
    if the new node is not in collision, add the new node to the tree
    if the maximum number of iterations is reached, break
    if the new node is within the goal radius, a path is found, break. 
        
"""

class RRT:
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
                    goal_bias = 0.0,
                    debug_flag = False,
                    print_logs = False, 
                    dynamic_obstacles=None,
                    edge_bundle=None,
                    eb_kd_tree=None,
                    get_eb_kd_tree_query=None,
                    kd_tree_delta_radius=0.5,
                    translate_function=None
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
        self.num_extension_trials = num_extension_trials #Number of trials to extend the RRT tree
        self.isvalid = isvalid_function #Function to check if the path is valid
        self.cost = cost_function #Function to calculate the cost of a edge
        self.reached_goal = reached_goal_function
        self.get_random_point = random_point_function
        self.tree = nx.DiGraph()
        self.goal_node_id = None
        self.path_found = False
        self.path_time = 0.0
        self.path_cost = float('inf')
        self.rng_seed = udf_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.goal_bias = goal_bias
        self.debug_flag = debug_flag
        self.print_logs = print_logs
        self.last_added_node_id = -1
        self.threshold = 3.0

        # Edge bundle support for maneuver library based propagation
        self.edge_bundle = edge_bundle
        self.eb_kd_tree = eb_kd_tree
        self.get_eb_kd_tree_query = get_eb_kd_tree_query
        self.kd_tree_delta_radius = kd_tree_delta_radius
        self.translate_function = translate_function

        # Preprocess the static obstacles for fast collision checking
        # This is done only once at the start of the RRT planning
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

        # Optional per-dimension scaling for the nearest-neighbor metric
        # (used when an agent provides `metric_state_scales`). When None, unscaled Euclidean is used.
        self._metric_scales = getattr(self.agent, 'metric_state_scales', None)

        # Use DynamicMatrix for preallocated fast storage
        self._node_matrix = DynamicMatrix(initial_capacity=1024, dim=self.distance_metric_state_size)

        #Set number of roundoff digits for time sampling
        self.roundoff_digits = find_roundoff_decimal_digits(self.minimum_time_step)

        #Class for RRT tree nodes 
        self.node_class = TreeNode

    def _metric_embed(self, state):
        """Project a full state into the metric space used for tree NN, priority
        queue, goal checks, and MC scoring. If the agent provides
        `metric_embed_state` (e.g. treaded's paper d_X with weighted velocities),
        use it directly. Otherwise slice to distance_metric_state_size + scales."""
        if hasattr(self.agent, 'metric_embed_state'):
            return self.agent.metric_embed_state(state)
        s = state[:self.distance_metric_state_size]
        if self._metric_scales is not None:
            return s * self._metric_scales
        return s

    def get_random_time(self):
        return round(self.rng.uniform(self.minimum_time_step, self.max_sample_T), self.roundoff_digits)

    def get_fixed_time(self):
        return self.max_sample_T
    
    def get_time(self):
        if self.use_fixed_sampling_time:
            return self.get_fixed_time()
        else:
            return self.get_random_time()

    def num_rrt_nodes(self):
        return len(self.tree.nodes)
    
    def get_tree_structure(self):
        return (self.tree, self._node_matrix)

    def reset_tree(self, some_existing_tree=None):
        if some_existing_tree == None:
            self.tree = nx.DiGraph()
            self._node_matrix = DynamicMatrix(initial_capacity=1024, dim=self.distance_metric_state_size)
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            self.last_added_node_id = -1
            self._node_matrix.count = 0
        else:
            self.tree = some_existing_tree[0]
            self._node_matrix = some_existing_tree[1]
            self.goal_node_id = None
            self.path_found = False
            self.path_cost = float('inf')
            self.path_time = 0.0
            last_added_node_id = next(reversed(self.tree._node))
            self.last_added_node_id = last_added_node_id
            # self._node_matrix.count = last_added_node_id + 1

    def add_rrt_node(self, state, parent_node_id, parent_action, parent_action_duration, 
                        path_from_parent, time_elapsed, cost):
        # new_node_id = len(self.tree.nodes)
        new_node_id = self.last_added_node_id + 1
        new_node = self.node_class(new_node_id, state, parent_node_id, parent_action, parent_action_duration,
                                 path_from_parent, round(time_elapsed, self.roundoff_digits), cost)
        self.tree.add_node(new_node_id, value=new_node)
        self.last_added_node_id = new_node_id

        # Only store the relevant metric dimensions (e.g., x, y) with optional per-dim scaling
        self._node_matrix.append(self._metric_embed(state), new_node_id)

        return new_node_id
     
    def sample_random_point(self):
        r = self.rng.uniform(0, 1)
        if r < self.goal_bias:
            random_point = self.goal
        else:
            # random_point = self.get_random_point(self.env, self.agent, self.rng)
            random_point = self.get_random_point(self.env, self.static_circular_obstacles,
                                                 self.static_rectangular_obstacles, self.rng)
        if self.debug_flag:
            print("Sampled random point: ", random_point)
        return random_point

    def get_nearest_node(self, random_point):
        states = self._node_matrix.get_valid_matrix()
        # active = self._node_matrix.get_active_entries()
        num_entries = self._node_matrix.count
        #Each entry in states is of size self.distance_metric_state_size
        #which by construction should be the same as the size of random_point
        nearest_index = get_nearest_index(states, num_entries,
                                          self._metric_embed(random_point))
        # nearest_index = get_active_nearest_index(states, active, num_entries, random_point)
        nearest_node_id = self._node_matrix.ids[nearest_index]
        nearest_node = self.tree.nodes[nearest_node_id]['value']

        if self.debug_flag:
            print("Nearest node id: ", nearest_node_id)
            print("Nearest node state: ", nearest_node.state[:self.distance_metric_state_size])
        return nearest_node_id, nearest_node

    def find_descendants(self, node_id):
        """
        Find all the descendants of the node with id node_id in the tree.
        Since the tree is a directed graph, the descendants are all the nodes that are 
        reachable from node_id.
        Additionally, our RRT tree is only a set of nodes, there are no edges.
        So, we need to iterate and find all the nodes that are reachable from node_id.
        Good thing: Only need to look at the nodes with id > node_id because the nodes are 
        added sequentially with the number of nodes added before it. 

        If there are 100 nodes in the tree, then the node with id 0 is the root node 
        and all the other nodes are its descendants.
        The indices of the nodes go from 0 to 99.
        So, if we want to find the descendants of node_id 19, we need to look at all 
        the nodes with id > 19.
        """

        max_node_id = next(reversed(self.tree._node))
        descendants = {node_id}
        # active = self._node_matrix.get_active_entries()

        # for i in range(node_id+1, max_node_id+1):
        #     if active[i]: # Or if(i in self.tree.nodes):
        #         parent_id = self.tree.nodes[i]['value'].parent_id
        #         if parent_id in descendants:
        #             #This means the node is a descendant of the node_id.
        #             descendants.add(i)
        #         #If the node is not a descendant of the node_id, then we can skip it.

        for i in range(node_id+1, max_node_id+1):
            parent_id = self.tree.nodes[i]['value'].parent_id
            if parent_id in descendants:
                #This means the node is a descendant of the node_id.
                descendants.add(i)
            #If the node is not a descendant of the node_id, then we can skip it.


        return descendants

    def remove_descendants(self, descendants):
        """
        Remove all the descendants of the node with id node_id from the tree.
        """
        self.tree.remove_nodes_from(descendants)

        #Adjust the node_matrix to remove the descendants
        # valid_ids_set = np.zeros(self.last_added_node_id + 1, dtype=np.int8)
        # for node_id in self.tree.nodes:
        #     valid_ids_set[node_id] = 1 
        # self._node_matrix.count = compact_matrix(self._node_matrix.matrix, self._node_matrix.ids,
        #                                          valid_ids_set, self._node_matrix.count)
        
        #Adjust the node_matrix to remove the descendants
        # for node_id in descendants:
        #     self._node_matrix.active[node_id] = False            

    def compute_children_counts(self):
        """
        Post-process the current RRT tree and compute the number of children
        for each node (based on the parent_id field of TreeNode).

        Returns
        -------
        children_counts : dict
            Mapping node_id -> number of children.
        """
        # Initialize count for all existing nodes
        children_counts = {node_id: 0 for node_id in self.tree.nodes}

        # For each node, increment the count of its parent (if parent exists)
        for node_id in self.tree.nodes:
            node = self.tree.nodes[node_id]['value']
            parent_id = node.parent_id
            if parent_id in children_counts:
                children_counts[parent_id] += 1

        return children_counts

    def _select_best_extension_candidate(self, parent_node, random_point):
        """
        Try self.num_extension_trials rollouts from parent_node and
        return the best valid one.

        Returns
        -------
        None
            If all trials are invalid.
        (new_state, path_to_new_state, random_action, random_time)
            Best valid candidate according to a simple score.
        """

        best_candidate = None
        best_score = np.inf

        # Embed random_point once under paper's d_X; score each MC candidate
        # by d_X(new_state, random_point).
        rand_embed = self._metric_embed(random_point)

        # Use maneuver library: pull (action, duration) from stored edges and
        # re-propagate online from the parent (matches GBRRT Alg. 8 / BBoE
        # Alg. 3 line 11 `SIMULATECONTROL(v, u_near, t)`).
        use_edge_bundle = self.edge_bundle is not None

        if use_edge_bundle:
            query = self.get_eb_kd_tree_query(parent_node.state)
            edge_ids = self.eb_kd_tree.radius_query(query, self.kd_tree_delta_radius)
            if len(edge_ids) > 0:
                l = min(len(edge_ids), 3000)
                valid_edge_indices = edge_ids[:l]

                num_samples = min(self.num_extension_trials, len(valid_edge_indices))
                random_slots = self.rng.choice(len(valid_edge_indices), size=num_samples, replace=False)

                for slot in random_slots:
                    edge_bundle_index = valid_edge_indices[slot]
                    random_action = self.edge_bundle.actions[edge_bundle_index]
                    random_time = self.edge_bundle.timesteps[edge_bundle_index]

                    num_record_steps = max(1, round(abs(random_time) / self.minimum_time_step))
                    new_state, path_to_new_state = self.agent.get_next_state(
                        parent_node.state, random_action, random_time,
                        num_steps=num_record_steps
                    )

                    accept_new_node = self.isvalid(
                        path_to_new_state, self.agent.radius, self.env.size,
                        self.static_circular_obstacles, self.static_rectangular_obstacles,
                        self.dynamic_agent_obstacles, self.env.obstacle_buffer, self.env.boundary_buffer,
                        parent_node.time_elapsed, random_time, self.minimum_time_step
                    )

                    if not accept_new_node:
                        continue

                    score = float(np.linalg.norm(self._metric_embed(new_state) - rand_embed))
                    if score < best_score:
                        best_score = score
                        best_candidate = (new_state, path_to_new_state, random_action, abs(random_time))

                return best_candidate

        # Fallback: online integration with random actions
        for _ in range(self.num_extension_trials):
            random_action = self.agent.get_random_action(self.rng)
            random_time = self.get_time()
            num_record_steps = round(random_time / self.minimum_time_step)

            new_state, path_to_new_state = self.agent.get_next_state(
                parent_node.state,
                random_action,
                random_time,
                num_steps=num_record_steps
            )

            accept_new_node = self.isvalid(
                path_to_new_state,
                self.agent.radius,
                self.env.size,
                self.static_circular_obstacles,
                self.static_rectangular_obstacles,
                self.dynamic_agent_obstacles,
                self.env.obstacle_buffer,
                self.env.boundary_buffer,
                parent_node.time_elapsed,
                random_time,
                self.minimum_time_step
            )

            if not accept_new_node:
                continue

            # Score: d_X distance to the sampled point (classic RRT heuristic)
            score = float(np.linalg.norm(self._metric_embed(new_state) - rand_embed))

            if score < best_score:
                best_score = score
                best_candidate = (
                    new_state,
                    path_to_new_state,
                    random_action,
                    random_time
                )

        return best_candidate

    def extend_tree(self, parent_node_id, parent_node, random_point):
        """
            * Sample a random action and time duration.
            * Extend the tree towards the random point.

            Extend by trying multiple random control+duration 
            samples and taking the best valid one.
        """

        best_candidate = self._select_best_extension_candidate(
                        parent_node,random_point)
        
        if best_candidate is None:
            if self.debug_flag:
                print("~~~~~~~~~~All extension trials invalid~~~~~~~~~~")
                print("~~~~~~~~~~Sampled New RRT Node is invalid. Trying again!~~~~~~~~~~")
            return
        else:
            new_state, path_to_new_state, random_action, random_time = best_candidate

            reached_goal_flag, goal_distance = self.reached_goal(new_state, self.goal, 
                                                            self.goal_radius, self.agent)
            if reached_goal_flag:
                edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action, 
                            random_time, path_to_new_state)
                total_elapsed_time = parent_node.time_elapsed + random_time
                total_cost = parent_node.cost_so_far + edge_cost

                new_node_id = self.add_rrt_node(new_state, parent_node_id, random_action, random_time,
                                                path_to_new_state, total_elapsed_time, total_cost)
                self.path_found = True
                if self.debug_flag:
                    print("Goal Reached! Path found for ",self.agent.id)
                self.goal_node_id = new_node_id
                self.path_cost = total_cost
                self.path_time = total_elapsed_time
                return
            else:
                if goal_distance < self.threshold:
                    total_elapsed_time = parent_node.time_elapsed
                    for (index, intermediate_state) in enumerate(path_to_new_state):
                        total_elapsed_time += self.minimum_time_step
                        goal_flag, d = self.reached_goal(intermediate_state, self.goal, 
                                                         self.goal_radius, self.agent)
                        if goal_flag:
                            modified_edge_time = total_elapsed_time - parent_node.time_elapsed
                            new_path_to_new_state = path_to_new_state[:index+1]
                            edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action,
                                                modified_edge_time, new_path_to_new_state)
                            total_cost = parent_node.cost_so_far + edge_cost
                            new_node_id = self.add_rrt_node(intermediate_state, parent_node_id, random_action, 
                                                            modified_edge_time, new_path_to_new_state, 
                                                            total_elapsed_time, total_cost)
                            self.path_found = True
                            if self.debug_flag:
                                print("Goal Reached! Path found for ",self.agent.id)
                            self.goal_node_id = new_node_id
                            self.path_cost = total_cost
                            self.path_time = total_elapsed_time
                            return
                
                edge_cost = self.cost(self.env, self.agent, parent_node.state, random_action, 
                                        random_time, path_to_new_state)
                total_elapsed_time = parent_node.time_elapsed + random_time
                total_cost = parent_node.cost_so_far + edge_cost

                new_node_id = self.add_rrt_node(new_state, parent_node_id, random_action, random_time,
                                                path_to_new_state, total_elapsed_time, total_cost)
                if self.debug_flag:
                    print("New Node Added to the RRT Tree: ", new_node_id)
            return

    def get_path_to_node_id(self, goal_node_id):

        # Return RRT node ids, agent state and the executed control from start to goal 

        # get state types
        state_length = self.agent.state_length
        control_length = self.agent.action_length

        max_nodes = len(self.tree.nodes)
        path_rrt_node_ids = np.empty(max_nodes, dtype=np.int32)
        path_states = np.empty((max_nodes, state_length), dtype=np.float64)
        path_controls = np.empty((max_nodes, control_length), dtype=np.float64)
        path_timesteps = np.empty(max_nodes, dtype=np.float64)
        node_id = goal_node_id
        path_length = 0            

        while node_id != -1:
            # print("Node ID: ", node_id)
            rrt_node = self.tree.nodes[node_id]['value']
            path_rrt_node_ids[path_length] = rrt_node.id
            path_states[path_length] = rrt_node.state
            path_controls[path_length] = rrt_node.parent_action
            path_timesteps[path_length] = rrt_node.parent_action_duration
            path_length+=1
            node_id = rrt_node.parent_id

        """
        Note: 
        1) Slicing a numpy array doesn't cause new array allocation. It just returns a view
        of the original array.
        2) Reversing the numpy array using [::-1] doesn't cause new array allocation. 
        It also just returns a view of the original array.
        3) Slicing and reversing the numpy array doesn't cause any allocation!!
        """    

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

    def get_high_resolution_path(self):
            
        # Return a path with a smaller resolution than the one returned by get_path
        # state_data_type = get_dtype_from_input(self.start)
        # path_dict: Dict[float,state_data_type] = {}
        path_dict = {}
        max_nodes = len(self.tree.nodes)
        path_rrt_node_ids = np.empty(max_nodes, dtype=np.int32)

        if(self.path_found == False):
            print("Path can't be found because goal hasn't been reached!")
            return path_dict
        else:
            node_id = self.goal_node_id
            path_length = 0
            path_time = 0.0            

            while node_id != -1:
                # print("Node ID: ", node_id)
                rrt_node = self.tree.nodes[node_id]['value']
                path_rrt_node_ids[path_length] = rrt_node.id
                path_length+=1
                node_id = rrt_node.parent_id
            
            ids = path_rrt_node_ids[:path_length][::-1]
            start_id = ids[0]
            path_dict[path_time] = self.tree.nodes[start_id]['value'].state

            for node_id in ids[1:]:
                rrt_node = self.tree.nodes[node_id]['value']
                path_from_parent = rrt_node.path_from_parent
                for point in path_from_parent:
                    path_time = round(path_time + self.minimum_time_step,self.roundoff_digits)
                    path_dict[path_time] = point

            return path_dict

    def get_high_resolution_path_numpy_array(self):
            
        # Return a path with a smaller resolution than the one returned by get_path
 
        if(self.path_found == False):
            print("Path can't be found for agent {} because goal " \
            "hasn't been reached!".format(self.agent.id))
            return np.empty((0,self.agent.state_length),dtype=np.float64)
        else:
            total_path_time = self.path_time
            min_time_step = self.minimum_time_step
            path_length = int( round(total_path_time/min_time_step,self.roundoff_digits)) + 1
            path_states = np.empty((path_length,self.agent.state_length), dtype=np.float64)

            node_id = self.goal_node_id
            curr_index = path_length - 1

            while node_id != 0: # Repeat until you have reached the start node
                # print("Node ID: ", node_id)
                rrt_node = self.tree.nodes[node_id]['value']
                path_to_node = rrt_node.path_from_parent
                len_path_to_node = len(path_to_node)
                start_index = curr_index - len_path_to_node + 1
                path_states[start_index:curr_index+1] = path_to_node
                curr_index -= len_path_to_node
                node_id = rrt_node.parent_id

            # Fill the remaining states with the start state
            path_states[:curr_index+1] = self.start
            return path_states

    def plan_path(self):
        """
        Plan a path using RRT algorithm from scratch.
        """

        self.path_found = False
        self.goal_node_id = None
        self.last_added_node_id = -1
        self.reset_tree()
        first_node_state = self.start 
        self.add_rrt_node(first_node_state, -1, None, None, None, 0.0, 0.0)
    
        curr_num_steps = 0
        start_time = time.time()
        
        while curr_num_steps<=self.max_iter:
            if self.debug_flag:
                print("*************************************")
                print("Iteration: ", curr_num_steps)
            
            random_point = self.sample_random_point()
            nearest_node_id, nearest_node = self.get_nearest_node(random_point)
            self.extend_tree(nearest_node_id, nearest_node, random_point)

            if self.path_found or (time.time() - start_time >= self.planning_time):
                break
            curr_num_steps+=1


        end_time = time.time()
        total_time = end_time - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)

        if self.print_logs or self.debug_flag:
            planning_time_msg = "Total Planning Time"
            if hasattr(self.agent, "id"):
                planning_time_msg += " for agent " + str(self.agent.id)
            planning_time_msg += " after " + str(curr_num_steps) + " iterations" 
            print(planning_time_msg + ": ", total_time)

    def replan_path(self):
        """
        This function is called when planning for a path was attempted before but failed.
        It doesn't regrow the tree from scratch, but expands the tree built during the 
        previous iteration of planning.
        """

        self.path_found = False
        self.goal_node_id = None
        self.path_time = 0.0
        self.path_cost = float('inf')

        if self.debug_flag or self.print_logs:
            print("Replanning the path...")
    
        curr_num_steps = 0
        start_time = time.time()

        while curr_num_steps<=self.max_iter:
            if self.debug_flag:
                print("*************************************")
                print("Iteration: ", curr_num_steps)
            
            random_point = self.sample_random_point()
            nearest_node_id, nearest_node = self.get_nearest_node(random_point)
            self.extend_tree(nearest_node_id, nearest_node, random_point)

            if self.path_found or (time.time() - start_time >= self.planning_time):
                break
            curr_num_steps+=1


        end_time = time.time()
        total_time = end_time - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)

        if self.print_logs or self.debug_flag:
            # Print the total planning time
            # This is useful for debugging and logging purposes
            planning_time_msg = "Total Planning Time"
            if hasattr(self.agent, "id"):
                planning_time_msg += " for agent " + str(self.agent.id)
            planning_time_msg += " after " + str(curr_num_steps) + " iterations" 
            print(planning_time_msg + ": ", total_time)
        
