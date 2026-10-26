import math
import numpy as np
from numba import njit

from utils import clamp, wrap_between_0_and_2pi, euclidean_distance_numba, \
    euclidean_distance_satisfaction_numba, euclidean_distance_numba_with_l, \
    validate_random_point_numba, is_new_node_valid_numba, is_state_valid_numba, \
    get_distance_covered_numba


"""
Treaded Vehicle (differential-drive with second-order dynamics)
As described in Nayak & Otte, IEEE T-RO 2022 (GBRRT paper).

State:   S = (x, y, theta, v_l, v_r)
Control: U = (a_l, a_r)

Dynamics:
    x_dot     = (v_l + v_r) / 2 * cos(theta)
    y_dot     = (v_l + v_r) / 2 * sin(theta)
    theta_dot = (v_r - v_l) / L
    v_l_dot   = a_l
    v_r_dot   = a_r
"""


@njit
def treaded_eom_numba(state, control, max_speed, track_width):
    x, y, theta, vl, vr = state
    al, ar = control

    vl = clamp(vl, -max_speed, max_speed)
    vr = clamp(vr, -max_speed, max_speed)

    v_avg = (vl + vr) / 2.0
    x_dot = v_avg * np.cos(theta)
    y_dot = v_avg * np.sin(theta)
    theta_dot = (vr - vl) / track_width
    vl_dot = al
    vr_dot = ar

    return np.array([x_dot, y_dot, theta_dot, vl_dot, vr_dot], dtype=np.float64)


@njit
def treaded_move_vehicle_numba(state, control, dt, max_speed, track_width):
    k1 = treaded_eom_numba(state, control, max_speed, track_width)
    k2 = treaded_eom_numba(state + 0.5 * dt * k1, control, max_speed, track_width)
    k3 = treaded_eom_numba(state + 0.5 * dt * k2, control, max_speed, track_width)
    k4 = treaded_eom_numba(state + dt * k3, control, max_speed, track_width)

    next_state = np.empty_like(state)
    tmp = dt / 6.0
    for i in range(state.shape[0]):
        next_state[i] = state[i] + tmp * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

    # Clamp tread velocities
    next_state[3] = clamp(next_state[3], -max_speed, max_speed)
    next_state[4] = clamp(next_state[4], -max_speed, max_speed)

    return next_state


@njit
def treaded_get_next_state_numba(state, control, dt, num_steps, max_speed,
                                 track_width, state_length):
    path = np.empty((num_steps, state_length), dtype=np.float64)
    curr_state = state.copy()
    exec_dt = dt / num_steps

    for i in range(num_steps):
        new_state = treaded_move_vehicle_numba(curr_state, control, exec_dt,
                                               max_speed, track_width)
        path[i] = new_state
        curr_state = new_state

    return curr_state, path


@njit
def treaded_get_distance_covered_numba(start_state, path):
    if path.shape[0] == 0:
        return 0.0

    dx = path[0, 0] - start_state[0]
    dy = path[0, 1] - start_state[1]
    total_distance = np.sqrt(dx * dx + dy * dy)
    for i in range(1, path.shape[0]):
        dx = path[i, 0] - path[i - 1, 0]
        dy = path[i, 1] - path[i - 1, 1]
        total_distance += np.sqrt(dx * dx + dy * dy)
    return total_distance


@njit
def treaded_point_translate_function_kd_tree_numba(base_point,
                                                    edge_start_point,
                                                    edge_end_point):
    """
    Apply a precomputed edge to a new base state.
    base_point:       (x_c, y_c, theta_c, vl_c, vr_c)
    edge_start_point: (0,   0,   0,       vl_s, vr_s)
    edge_end_point:   (dx,  dy,  dtheta,  vl_e, vr_e)

    Returns: (x_c + R(theta_c)*[dx,dy], theta_c + dtheta, vl_e, vr_e)
    """
    theta_c = base_point[2]
    cos_theta = np.cos(theta_c)
    sin_theta = np.sin(theta_c)

    new_x = edge_end_point[0] * cos_theta - edge_end_point[1] * sin_theta
    new_y = edge_end_point[0] * sin_theta + edge_end_point[1] * cos_theta

    return (base_point[0] + new_x, base_point[1] + new_y,
            base_point[2] + edge_end_point[2],
            edge_end_point[3],
            edge_end_point[4])


@njit
def treaded_dx_dist_after_edge(base_point, edge_end_point, random_point):
    """Paper d_X between the translated edge endpoint and random_point:
        d_X = sqrt(dx^2 + dy^2 + 0.25^2 * (dv_L^2 + dv_R^2))
    edge_end_point: (dx, dy, dtheta, v_L_end, v_R_end)
    random_point:   (x, y, theta, v_L, v_R)  — 5-D from TreadedVehicle.get_random_point
    """
    theta_c = base_point[2]
    cos_theta = np.cos(theta_c)
    sin_theta = np.sin(theta_c)
    new_x = base_point[0] + edge_end_point[0] * cos_theta - edge_end_point[1] * sin_theta
    new_y = base_point[1] + edge_end_point[0] * sin_theta + edge_end_point[1] * cos_theta
    # Velocities after applying the edge are the edge's endpoint velocities
    # (they replace base's velocities — see kd_tree_point_translate_function).
    new_vL = edge_end_point[3]
    new_vR = edge_end_point[4]
    dxp = new_x - random_point[0]
    dyp = new_y - random_point[1]
    dvL = 0.25 * (new_vL - random_point[3])
    dvR = 0.25 * (new_vR - random_point[4])
    return math.sqrt(dxp * dxp + dyp * dyp + dvL * dvL + dvR * dvR)


@njit
def treaded_sort_kd_tree_edges_numba(closest_tree_point, random_point,
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
            dist = treaded_dx_dist_after_edge(closest_tree_point,
                                               final_states[edge_idx],
                                               random_point)
            distance_array[i] = dist
            num_valid_edges += 1

    sorted_indices = np.argsort(distance_array[:n])
    return sorted_indices[:num_valid_edges], num_valid_edges


class TreadedVehicle:
    """
    Treaded vehicle (differential-drive, second-order dynamics).

    State:   [x, y, theta, v_l, v_r]
    Control: [a_l, a_r]
    """

    def __init__(self, *,
                 agent_id=1,
                 max_speed=5.0,
                 max_acceleration=10.0,
                 track_width=1.0,
                 radius=0.4,
                 rng_seed=77):
        self.id = agent_id
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self.track_width = float(track_width)
        self.radius = float(radius)
        self.rng = np.random.default_rng(int(rng_seed))

        self.state_length = 5
        self.action_length = 2
        # Paper-faithful d_X (Table I, p. 3647):
        #   d_X = sqrt(dx^2 + dy^2 + 0.25^2 * (dv_L^2 + dv_R^2))
        # The embedding (x, y, 0.25*v_L, 0.25*v_R) makes Euclidean norm of
        # the diff equal d_X. Used everywhere distance is computed in the
        # planners (tree NN, priority queue, exploit, BestInputProp scoring,
        # goal check), via the metric_embed_state hook.
        self.distance_metric_state_size = 4

    def metric_embed_state(self, state):
        """Paper d_X embedding: (x, y, 0.25*v_L, 0.25*v_R). theta is not
        included per Table I. Consumed by RRT/SST/GBRRT/BBoE as the paper's
        distance function d_X (also used as d_M since this embedding is a
        proper weighted Euclidean metric)."""
        return np.array([state[0], state[1],
                         0.25 * state[3], 0.25 * state[4]],
                        dtype=np.float64)

    def equation_of_motion(self, state, control):
        return treaded_eom_numba(state, control, self.max_speed, self.track_width)

    def move_vehicle(self, state, control, dt):
        return treaded_move_vehicle_numba(state, control, dt,
                                          self.max_speed, self.track_width)

    def get_next_state(self, state, control, dt, num_steps=10):
        return treaded_get_next_state_numba(state, control, dt, num_steps,
                                            self.max_speed, self.track_width,
                                            self.state_length)

    def get_distance(self, state1, state2):
        return euclidean_distance_numba_with_l(state1, state2,
                                               self.distance_metric_state_size)

    def get_random_action(self, rng):
        al = rng.uniform(-self.max_acceleration, self.max_acceleration)
        ar = rng.uniform(-self.max_acceleration, self.max_acceleration)
        return (al, ar)

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
        """Returns a 5-D target (x, y, 0, 0, 0): random position in free space
        with zero theta and zero tread velocities. The zero-padding makes d_X
        to this point equal sqrt(dx^2+dy^2+0.25^2*(v_L^2+v_R^2)), which
        steers the tree toward positions where the vehicle is also near rest."""
        p = rng.uniform(env.env_start, env.size)
        while not validate_random_point_numba(p, circular_obstacles,
                                               rectangular_obstacles,
                                               env.obstacle_buffer):
            p = rng.uniform(env.env_start, env.size)
        return np.array([p[0], p[1], 0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def agent_reached_goal(state, goal, goal_radius, agent):
        """
        Paper-faithful d_X for treaded (Nayak & Otte 2022, Table I, p. 3647):
            d_X = sqrt((x1-x2)^2 + (y1-y2)^2
                      + 0.25^2 * ((v_L1-v_L2)^2 + (v_R1-v_R2)^2))
        Note: theta is not included in d_X. Velocities have 0.25^2 = 0.0625
        weight (lower than position) per paper to emphasise position.
        """
        dx = state[0] - goal[0]
        dy = state[1] - goal[1]
        dvL = state[3] - goal[3]
        dvR = state[4] - goal[4]
        d = np.sqrt(dx * dx + dy * dy
                    + 0.0625 * (dvL * dvL + dvR * dvR))
        return d <= goal_radius, d

    @staticmethod
    def kd_tree_point_translate_function(base_point, edge_start_point,
                                          edge_end_point):
        return treaded_point_translate_function_kd_tree_numba(
            base_point, edge_start_point, edge_end_point)

    @staticmethod
    def sort_kd_tree_edges(closest_tree_point, random_point, start_states,
                           final_states, curr_edge_indices, curr_edge_mask,
                           distance_array):
        return treaded_sort_kd_tree_edges_numba(closest_tree_point, random_point,
                                                start_states, final_states,
                                                curr_edge_indices, curr_edge_mask,
                                                distance_array)

    def get_eb_kd_tree_query(self, state):
        return (state[3], state[4])  # v_l, v_r
