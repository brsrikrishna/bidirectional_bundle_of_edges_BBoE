import math
import numpy as np
from numba import njit

from utils import clamp, wrap_between_0_and_2pi, euclidean_distance_numba, \
    euclidean_distance_satisfaction_numba, euclidean_distance_numba_with_l, \
    validate_random_point_numba, is_new_node_valid_numba, is_state_valid_numba


"""
Car with Trailer (second-order car towing a passive trailer)
As described in Nayak & Otte, IEEE T-RO 2022 (GBRRT paper).

State:   S = (x, y, theta, v, omega, theta1)
  - x, y:     car position
  - theta:    car heading
  - v:        linear velocity
  - omega:    steering angle
  - theta1:   trailer heading

Control: U = (linear_acceleration, angular_acceleration)

Dynamics:
  x_dot      = v * cos(theta) * cos(omega)
  y_dot      = v * sin(theta) * cos(omega)
  theta_dot  = v * sin(omega)
  v_dot      = linear_acceleration
  omega_dot  = angular_acceleration
  theta1_dot = v * sin(theta - theta1)
"""


@njit
def cwt_eom_numba(state, control, max_speed, max_omega):
    x, y, theta, v, omega, theta1 = state
    v = clamp(v, -max_speed, max_speed)
    omega = clamp(omega, -max_omega, max_omega)
    linear_acceleration, angular_acceleration = control

    x_dot = v * np.cos(theta) * np.cos(omega)
    y_dot = v * np.sin(theta) * np.cos(omega)
    theta_dot = v * np.sin(omega)
    v_dot = linear_acceleration
    omega_dot = angular_acceleration
    theta1_dot = v * np.sin(theta - theta1)

    return np.array([x_dot, y_dot, theta_dot, v_dot, omega_dot, theta1_dot],
                    dtype=np.float64)


@njit
def cwt_move_vehicle_numba(state, control, dt, max_speed, max_omega):
    k1 = cwt_eom_numba(state, control, max_speed, max_omega)
    k2 = cwt_eom_numba(state + 0.5 * dt * k1, control, max_speed, max_omega)
    k3 = cwt_eom_numba(state + 0.5 * dt * k2, control, max_speed, max_omega)
    k4 = cwt_eom_numba(state + dt * k3, control, max_speed, max_omega)

    next_state = np.empty_like(state)
    tmp = dt / 6.0
    for i in range(state.shape[0]):
        next_state[i] = state[i] + tmp * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

    next_state[3] = clamp(next_state[3], -max_speed, max_speed)
    next_state[4] = clamp(next_state[4], -max_omega, max_omega)

    # Wrap trailer heading theta1 to [-pi, pi] so the bundle KD-tree key stays
    # in the bundle's sampled range. Without this wrap, theta1 accumulates and
    # the KD-tree query silently drops every matching edge.
    two_pi = 2.0 * np.pi
    next_state[5] = ((next_state[5] + np.pi) % two_pi) - np.pi

    return next_state


@njit
def cwt_get_next_state_numba(state, control, dt, num_steps,
                              max_speed, max_omega, state_length):
    path = np.empty((num_steps, state_length), dtype=np.float64)
    curr_state = state.copy()
    exec_dt = dt / num_steps

    for i in range(num_steps):
        new_state = cwt_move_vehicle_numba(curr_state, control, exec_dt,
                                            max_speed, max_omega)
        path[i] = new_state
        curr_state = new_state

    return curr_state, path


@njit
def cwt_point_translate_function_kd_tree_numba(base_point,
                                                edge_start_point,
                                                edge_end_point):
    """
    Apply a precomputed edge to a new base state.
    base_point:       (x_c, y_c, theta_c, v_c, omega_c, theta1_c)
    edge_start_point: (0,   0,   0,       v_s, omega_s, theta1_s)
    edge_end_point:   (dx,  dy,  dtheta,  v_e, omega_e, theta1_e)

    Rotate (dx, dy) by theta_c, add dtheta to theta_c.
    Returns: (x_new, y_new, theta_new, v_e, omega_e, theta1_new)
    """
    theta_c = base_point[2]
    cos_theta = np.cos(theta_c)
    sin_theta = np.sin(theta_c)

    new_x = edge_end_point[0] * cos_theta - edge_end_point[1] * sin_theta
    new_y = edge_end_point[0] * sin_theta + edge_end_point[1] * cos_theta

    return (base_point[0] + new_x,
            base_point[1] + new_y,
            base_point[2] + edge_end_point[2],
            edge_end_point[3],
            edge_end_point[4],
            base_point[5] + edge_end_point[5])  # trailer heading also rotates


@njit
def cwt_dx_dist_after_edge(base_point, edge_end_point, random_point):
    """Paper d_X for car-with-trailer (Table I, p. 3647):
        d_X = sqrt((x1-x2)^2 + (y1-y2)^2 + (0.25)^2 * (v1-v2)^2)
    3-D in (x, y, v) with 0.25 weight on velocity. theta/omega/theta1 not in d_X.
    random_point is 5+ dims (x, y, theta, v, ...); random_point[3] = v target."""
    theta_c = base_point[2]
    cos_theta = np.cos(theta_c)
    sin_theta = np.sin(theta_c)
    new_x = base_point[0] + edge_end_point[0] * cos_theta - edge_end_point[1] * sin_theta
    new_y = base_point[1] + edge_end_point[0] * sin_theta + edge_end_point[1] * cos_theta
    new_v = edge_end_point[3]  # v after edge (kd_tree_point_translate_function replaces v)
    dxp = new_x - random_point[0]
    dyp = new_y - random_point[1]
    dv = 0.25 * (new_v - random_point[3])
    return math.sqrt(dxp * dxp + dyp * dyp + dv * dv)


@njit
def cwt_sort_kd_tree_edges_numba(closest_tree_point, random_point,
                                  start_states, final_states,
                                  curr_edge_indices, curr_edge_mask,
                                  distance_array):
    n = curr_edge_indices.shape[0]
    num_valid_edges = 0

    for i in range(n):
        if curr_edge_mask[i]:
            distance_array[i] = 1e10
        else:
            edge_idx = curr_edge_indices[i]
            dist = cwt_dx_dist_after_edge(closest_tree_point,
                                           final_states[edge_idx],
                                           random_point)
            distance_array[i] = dist
            num_valid_edges += 1

    sorted_indices = np.argsort(distance_array[:n])
    return sorted_indices[:num_valid_edges], num_valid_edges


class CarWithTrailer:
    """
    Car with passive trailer (second-order nonholonomic).

    State:   [x, y, theta, v, omega, theta1]
    Control: [linear_acceleration, angular_acceleration]
    """

    def __init__(self, *,
                 agent_id=-1,
                 max_speed=2.0,
                 max_linear_acceleration=1.0,
                 max_omega=np.pi / 3,
                 max_angular_acceleration=0.5,
                 radius=0.3,
                 rng_seed=42):
        self.id = agent_id
        self.max_speed = float(max_speed)
        self.max_linear_acceleration = float(max_linear_acceleration)
        self.max_omega = float(max_omega)
        self.max_angular_acceleration = float(max_angular_acceleration)
        self.radius = float(radius)
        self.rng = np.random.default_rng(int(rng_seed))

        self.state_length = 6
        self.action_length = 2
        # Paper-faithful d_X (Table I, p. 3647):
        #   d_X = sqrt((x1-x2)^2 + (y1-y2)^2 + 0.25^2 * (v1-v2)^2)
        # The embedding (x, y, 0.25*v) makes Euclidean norm of the diff equal
        # d_X. Used for tree NN, priority queue, exploit, MC scoring, goal
        # check, via the metric_embed_state hook. theta, omega, theta1 are
        # not in d_X per paper.
        self.distance_metric_state_size = 3

    def metric_embed_state(self, state):
        """Paper d_X embedding: (x, y, 0.25*v). Euclidean norm of the diff
        between two embedded states equals paper's d_X. Consumed by
        RRT/SST/GBRRT/BBoE as the paper's distance function d_X."""
        return np.array([state[0], state[1], 0.25 * state[3]], dtype=np.float64)

    def equation_of_motion(self, state, control):
        return cwt_eom_numba(state, control, self.max_speed, self.max_omega)

    def move_vehicle(self, state, control, dt):
        return cwt_move_vehicle_numba(state, control, dt,
                                       self.max_speed, self.max_omega)

    def get_next_state(self, state, control, dt, num_steps=10):
        return cwt_get_next_state_numba(state, control, dt, num_steps,
                                         self.max_speed, self.max_omega,
                                         self.state_length)

    def get_distance(self, state1, state2):
        return euclidean_distance_numba_with_l(state1, state2,
                                               self.distance_metric_state_size)

    def get_random_action(self, rng):
        a = rng.uniform(-self.max_linear_acceleration,
                        self.max_linear_acceleration)
        omega_a = rng.uniform(-self.max_angular_acceleration,
                              self.max_angular_acceleration)
        return (a, omega_a)

    def check_collision(self, base_agent_state, point):
        d = self.get_distance(base_agent_state, point)
        return d <= self.radius * 2.0

    # ---- static utility functions ----

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
                          start_time, time_duration, dt_per_step=0.1):
        return is_new_node_valid_numba(path_to_new_state, agent_radius, env_size,
                                       circ_obs, rect_obs, dyn_obs,
                                       obstacle_buffer, boundary_buffer,
                                       start_time, time_duration, dt_per_step)

    @staticmethod
    def get_cost(env, agent, parent_state, control, t, path):
        return t

    @staticmethod
    def get_random_point(env, circular_obstacles, rectangular_obstacles, rng):
        """Returns a 6-D target (x, y, 0, 0, 0, 0): random position with zero
        theta, velocity, omega, trailer heading. Zero-padding makes d_X to
        this point equal sqrt(dx^2+dy^2+0.25^2*v^2), steering the tree
        toward positions where the car is also near rest."""
        p = rng.uniform(env.env_start, env.size)
        while not validate_random_point_numba(p, circular_obstacles,
                                               rectangular_obstacles,
                                               env.obstacle_buffer):
            p = rng.uniform(env.env_start, env.size)
        return np.array([p[0], p[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def agent_reached_goal(state, goal, goal_radius, agent):
        """Paper-faithful d_X for car-with-trailer (Table I, p. 3647):
            d_X = sqrt((x1-x2)^2 + (y1-y2)^2 + 0.25^2 * (v1-v2)^2)
        Only x, y, v are checked — theta, omega, theta1 are not in d_X.
        Zero terminal velocity is required (paper convention)."""
        dx = state[0] - goal[0]
        dy = state[1] - goal[1]
        dv = state[3] - goal[3]
        d = np.sqrt(dx * dx + dy * dy + 0.0625 * dv * dv)
        return d <= goal_radius, d

    @staticmethod
    def kd_tree_point_translate_function(base_point, edge_start_point,
                                          edge_end_point):
        return cwt_point_translate_function_kd_tree_numba(
            base_point, edge_start_point, edge_end_point)

    @staticmethod
    def sort_kd_tree_edges(closest_tree_point, random_point, start_states,
                           final_states, curr_edge_indices, curr_edge_mask,
                           distance_array):
        return cwt_sort_kd_tree_edges_numba(closest_tree_point, random_point,
                                             start_states, final_states,
                                             curr_edge_indices, curr_edge_mask,
                                             distance_array)

    def get_eb_kd_tree_query(self, state):
        # Defensive theta1 wrap to [-pi, pi] so even unwrapped states from
        # external callers land in the bundle's sampled range.
        two_pi = 2.0 * np.pi
        theta1_wrapped = ((state[5] + np.pi) % two_pi) - np.pi
        return (state[3], state[4], theta1_wrapped)  # v, omega, theta1
