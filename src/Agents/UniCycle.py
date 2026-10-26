import math
import numpy as np
import random
from numba import njit

from utils import clamp, wrap_between_0_and_2pi, euclidean_distance_numba, \
    euclidean_distance_satisfaction_numba, euclidean_distance_numba_with_l, \
    euclidean_distance_satisfaction_numba_with_l, \
    validate_random_point_numba, is_new_node_valid_numba, is_state_valid_numba, \
    get_distance_covered_numba, check_dynamic_collisions_to_end
    

@njit
def unicycle_equation_of_motion_numba(state, control):
    x, y, theta = state
    v, omega = control
    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = omega
    return np.array([x_dot, y_dot, theta_dot], dtype=np.float64)
    # return (x_dot, y_dot, theta_dot)

@njit
def unicycle_move_vehicle_numba(state, v, omega, dt):
    x, y, theta = state
    new_x = x + v * np.cos(theta) * dt
    new_y = y + v * np.sin(theta) * dt
    new_theta = wrap_between_0_and_2pi(theta + omega * dt)
    return np.array([new_x, new_y, new_theta], dtype=np.float64)
    # return (new_x, new_y, new_theta)

@njit
def unicycle_get_next_state_numba(state, control, dt, num_steps):
    v = control[0]
    omega = control[1]
    path = np.empty((num_steps, 3), dtype=np.float64)
    curr_state = state
    exec_dt = dt / num_steps

    for i in range(num_steps):
        new_state = unicycle_move_vehicle_numba(curr_state, v, omega, exec_dt)
        path[i, 0] = new_state[0]
        path[i, 1] = new_state[1]
        path[i, 2] = new_state[2]
        curr_state = new_state

    return curr_state, path

@njit
def unicycle_point_translate_function_numba(base_point, edge_end_point):
    theta = base_point[2]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    new_x = edge_end_point[0] * cos_theta - edge_end_point[1] * sin_theta
    new_y = edge_end_point[0] * sin_theta + edge_end_point[1] * cos_theta

    return (base_point[0] + new_x, base_point[1] + new_y, theta + edge_end_point[2])

@njit
def unicycle_batch_translate_trajectory_numba(base_point, start_point, traj):
    """
    Batch SE(2) translation of a whole trajectory from the offline edge frame
    to the current pose. Semantics match `unicycle_point_translate_function_kd_tree_numba`
    applied waypoint-by-waypoint. Writes into a freshly allocated (L, 3) array.
    """
    x_c = base_point[0]
    y_c = base_point[1]
    theta_c = base_point[2]
    theta_e = start_point[2]
    delta_theta = theta_c - theta_e
    cos_d = np.cos(delta_theta)
    sin_d = np.sin(delta_theta)

    L = traj.shape[0]
    out = np.empty((L, 3), dtype=np.float64)
    for i in range(L):
        x_f = traj[i, 0]
        y_f = traj[i, 1]
        theta_f = traj[i, 2]
        out[i, 0] = x_c + x_f * cos_d - y_f * sin_d
        out[i, 1] = y_c + x_f * sin_d + y_f * cos_d
        out[i, 2] = theta_c + (theta_f - theta_e)
    return out


@njit
def unicycle_point_translate_function_kd_tree_numba(base_point, start_point, edge_end_point):
    """
    base_point:      (x_c, y_c, theta_c)
    start_point:     (0, 0, theta_e)        for this offline edge
    edge_end_point:  (x_f, y_f, theta_f)    offline end state from [0,0,theta_e]

    Returns new final state (x_f', y_f', theta_f') applying same edge from base_point.
    """

    # Note: Although the following simpler translation below is incorrect, it
    # is a reasonble heuristic and generates just as good results in practice.
    # I know this because I tested both methods in the trial script - HG.
    # return (base_point[0] + edge_end_point[0],
    #         base_point[1] + edge_end_point[1],
    #         base_point[2] + (edge_end_point[2] - start_point[2]))


    # Unpack
    x_c = base_point[0]
    y_c = base_point[1]
    theta_c = base_point[2]

    theta_e = start_point[2]      # offline edge start heading
    x_f = edge_end_point[0]       # offline end displacement in world frame
    y_f = edge_end_point[1]
    theta_f = edge_end_point[2]

    # 1. Heading difference to align edge's start with current heading
    delta_theta_heading = theta_c - theta_e

    cos_d = np.cos(delta_theta_heading)
    sin_d = np.sin(delta_theta_heading)

    # 2. Rotate offline displacement by the alignment angle
    dx_world = x_f * cos_d - y_f * sin_d
    dy_world = x_f * sin_d + y_f * cos_d

    # 3. Translate from current pose
    new_x = x_c + dx_world
    new_y = y_c + dy_world

    # 4. Heading: add the offline heading change (theta_f - theta_e)
    dtheta_edge = theta_f - theta_e
    # new_theta = wrap_between_0_and_2pi(theta_c + dtheta_edge)
    new_theta = theta_c + dtheta_edge

    return new_x, new_y, new_theta

@njit
def unicycle_sort_edges_numba(closest_tree_point, random_point, final_states, distance_array):
    n = final_states.shape[0]
    for i in range(n):
        potential_new_point = unicycle_point_translate_function_numba(closest_tree_point, 
                                                                final_states[i])
        dist = euclidean_distance_numba(potential_new_point, random_point)
        distance_array[i] = dist

    sorted_indices = np.argsort(distance_array)
    return sorted_indices

#Need to modify this function later for two things.
#1) You don't really need to add distance as 1e10 for already explored edges.
#2) Need to stop creating and returning a new array when you call argsort. 
@njit
def unicycle_sort_kd_tree_edges_numba(closest_tree_point, random_point,
                                start_states, final_states, curr_edge_indices,
                                curr_edge_mask, distance_array):
    """Sort library edges by paper d_X (Table I, unicycle): sqrt(dx^2 + dy^2).
    Only (x, y) is compared; theta is ignored per paper spec."""
    n = curr_edge_indices.shape[0]
    num_valid_edges = 0
    for i in range(n):
        edge_idx = curr_edge_indices[i]
        if curr_edge_mask[i] == True:
            #That edge has been explored before
            distance_array[i] = 1e10
        else:
            potential_new_point = unicycle_point_translate_function_kd_tree_numba(closest_tree_point,
                                    start_states[edge_idx], final_states[edge_idx])
            dx = potential_new_point[0] - random_point[0]
            dy = potential_new_point[1] - random_point[1]
            distance_array[i] = math.sqrt(dx * dx + dy * dy)
            num_valid_edges += 1

    sorted_indices = np.argsort(distance_array[:n])
    return sorted_indices[:num_valid_edges], num_valid_edges


@njit
def dbcbs_unicycle_cost(dt, beta, v, omega):
    return dt + beta * (v*v + omega*omega)


class UniCycle:
    def __init__(self, * , agent_id=1, max_speed, max_omega, radius, rng_seed=11):
        self.id = agent_id
        self.state_length = 3
        self.max_speed = max_speed
        self.max_omega = max_omega
        self.radius = radius
        self.rng = np.random.default_rng(rng_seed)
        self.state_datatype = np.dtype([('x', 'f8'), ('y', 'f8'), ('theta', 'f8')])
        self.state_length = 3
        self.distance_metric_state_size = 2
        self.action_length = 2

    def equation_of_motion(self, state, control):
        return unicycle_equation_of_motion_numba(state, control)

    def move_vehicle(self, state, v, omega, dt):
        return unicycle_move_vehicle_numba(state, v, omega, dt)

    def get_next_state(self, state, control, dt, num_steps=10):
        return unicycle_get_next_state_numba(state, control, dt, num_steps)

    def get_random_action(self,udf_rng):
        v = udf_rng.uniform(-self.max_speed, self.max_speed)
        omega = udf_rng.uniform(-self.max_omega, self.max_omega)
        return (v, omega)
        
    def get_distance(self, state1, state2):
        # d = euclidean_distance((state1[0], state1[1]), (state2[0], state2[1]))
        return euclidean_distance_numba_with_l(state1, state2, self.distance_metric_state_size)

    def check_collision(self, base_agent_state, point):
        d = self.get_distance(base_agent_state,point)
        if d<=self.radius*2:
            return True
        else:
            return False

    """
    UTILITY FUNCS
    """
    @staticmethod
    def is_state_valid(state, agent_radius, env_size,
                         circ_obs, rect_obs, dyn_obs,
                         obstacle_buffer, boundary_buffer, t):
        
        return is_state_valid_numba(state, agent_radius, env_size,
                                    circ_obs, rect_obs, dyn_obs,
                                    obstacle_buffer, boundary_buffer, t)
    
  
    @staticmethod
    def is_new_node_valid(path_to_new_state, agent_radius, env_size,
                            circ_obs, rect_obs, dyn_obs,
                            obstacle_buffer, boundary_buffer,
                            start_time, time_duration,
                            dt_per_step=0.1):
        return is_new_node_valid_numba(path_to_new_state, agent_radius, env_size,
                                        circ_obs, rect_obs, dyn_obs,
                                        obstacle_buffer, boundary_buffer,
                                        start_time, time_duration,
                                        dt_per_step)
    

    @staticmethod
    def get_cost(env, agent, parent_state, a, t, path):        
        v, omega = a
        return get_distance_covered_numba(v, t)
        # return dbcbs_unicycle_cost(t, 1.0, v, omega)
        # return t
    
    @staticmethod
    def get_random_point(env, circular_obstacles, rectangular_obstacles, rng):
        """
        Gets a random point for a UniCycle agent in an environment
        """ 
        p = rng.uniform(env.env_start, env.size)

        while not validate_random_point_numba(p, circular_obstacles,
                                rectangular_obstacles, env.obstacle_buffer):
            p = rng.uniform(env.env_start, env.size)

        return np.array(p, dtype=np.float64)
    
    @staticmethod
    def agent_reached_goal(state, goal, goal_radius, agent):
        """
        Per GBRRT paper Table I, unicycle d_X = sqrt(dx^2 + dy^2) — theta
        is NOT in d_X. Always compute the goal distance over (x, y) only,
        regardless of whether the caller passes a 2-D or 3-D goal vector.
        GBRRT/BBoE pass a 3-D goal so they can seed a reverse tree with a
        full state, but the goal-reach check stays paper-faithful (and
        matches what RRT/SST/Db-RRT/BoE compute when given a 2-D goal).
        """
        return euclidean_distance_satisfaction_numba_with_l(state, goal, 2, goal_radius)


    @staticmethod
    def point_translate_function(base_point, edge_end_point):
        """
        Translates the agent's state by a given 2D vector.
        """
        return unicycle_point_translate_function_numba(base_point, edge_end_point)

    @staticmethod
    def kd_tree_point_translate_function(base_point, edge_start_point, edge_end_point):
        """
        Translates the agent's state by a given 2D vector.
        """
        return unicycle_point_translate_function_kd_tree_numba(base_point,
                            edge_start_point, edge_end_point)

    @staticmethod
    def batch_translate_trajectory(base_point, edge_start_point, trajectory):
        """
        Vectorized SE(2) translation of a whole trajectory (L, 3) array.
        Equivalent to calling kd_tree_point_translate_function waypoint-by-waypoint.
        """
        return unicycle_batch_translate_trajectory_numba(
            base_point, edge_start_point, trajectory)


    @staticmethod
    def sort_edges(closest_tree_point, random_point, final_states, distance_array):
        """
        Sorts edges based on their distance from a base point.
        """
        return unicycle_sort_edges_numba(closest_tree_point, random_point, 
                                         final_states, distance_array)

    @staticmethod
    def sort_kd_tree_edges(closest_tree_point, random_point, start_states,
            final_states, curr_edge_indices, curr_edge_mask, distance_array):
        """
        Sorts edges based on their distance from a base point.
        """
        return unicycle_sort_kd_tree_edges_numba(closest_tree_point, random_point, 
        start_states, final_states, curr_edge_indices, curr_edge_mask, distance_array)

    def get_eb_kd_tree_query(self, state):
        """
        Gets the query point for the KD-Tree based on the agent's state.
        """
        # return np.array([state[2]], dtype=np.float64)
        return state[2]

"""
# Example instantiation, usage of the UniCycle Agent

import sys
sys.path.append('./src')
from Agents import *

# instantiate 
u = UniCycle(agent_id = 1, 
                 max_speed = 2.0,
                 max_omega= np.pi/2,
                 radius = 1.0,
                 rng_seed= 77)

# move from a start state to new state
# 's' using a control input
state = (0,0,0)
control = (1,np.pi/6)
dt = 0.5
s, path = u.get_next_state(state, control, dt)

i = 100
control = eb.d.actions[i]
dt = eb.d.timesteps[i]
s, path = u.get_next_state(state, control, dt)

control = eb_unicycle.actions[i]
dt = eb_unicycle.timesteps[i]
s, path = u.get_next_state(state, control, dt)


"""     