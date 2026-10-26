import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]   # repository root
import time
sys.path.append(str(ROOT / 'src'))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from Environments import SquareEnvironment, CircularObstacle2D, RectangleObstacle2D
from mapf_env_square_agent_unicycle import get_unicycle_agent
from edge_bundle import EdgeBundle
from kd_tree_unicycle import CircularAngleIndexNumba
from unicycle_discrete_actions import apply_discrete_random_action
from bboe import BBoE

# Paper Environment: 100x100, Unicycle (3D)
circular_obstacles = [
    CircularObstacle2D(4.7280, 71.1084, 4.8503),
    CircularObstacle2D(95.1437, 44.7364, 5.3812),
    CircularObstacle2D(26.2557, 72.9904, 4.4192),
    CircularObstacle2D(64.0323, 94.2136, 4.9226),
    CircularObstacle2D(46.6180, 31.3614, 4.0535),
    CircularObstacle2D(84.7776, 40.3829, 5.7209),
    CircularObstacle2D(46.8831, 50.4821, 5.3116),
    CircularObstacle2D(38.5017, 25.2092, 5.7481),
    CircularObstacle2D(12.2435, 32.3249, 5.0544),
    CircularObstacle2D(57.3401, 25.7696, 4.3951),
    CircularObstacle2D(95.1684, 40.6942, 5.5478),
    CircularObstacle2D(48.1801, 79.6717, 4.5356),
    CircularObstacle2D(72.8617, 48.0098, 4.9581),
    CircularObstacle2D(9.2336, 85.2890, 5.8196),
    CircularObstacle2D(44.5743, 37.4870, 4.1500),
    CircularObstacle2D(97.2016, 37.0088, 5.1752),
    CircularObstacle2D(24.9650, 38.3652, 5.3928),
    CircularObstacle2D(4.4601, 12.1331, 4.8014),
    CircularObstacle2D(83.1934, 31.0452, 5.5106),
    CircularObstacle2D(71.1277, 34.4686, 5.5136),
    CircularObstacle2D(88.1508, 67.0991, 4.0413),
    CircularObstacle2D(10.0939, 48.5954, 4.3324),
    CircularObstacle2D(90.9893, 30.2436, 5.3024),
    CircularObstacle2D(7.8265, 94.8158, 4.1126),
    CircularObstacle2D(69.3942, 57.1883, 4.9337),
    CircularObstacle2D(96.2347, 48.3807, 5.7634),
    CircularObstacle2D(43.3506, 28.1864, 4.5507),
    CircularObstacle2D(57.9631, 84.1230, 5.1935),
    CircularObstacle2D(89.1343, 74.6027, 4.1076),
    CircularObstacle2D(37.1139, 8.7152, 4.4701),
    CircularObstacle2D(13.1943, 3.6775, 5.9519),
    CircularObstacle2D(17.3332, 73.6080, 4.5046),
    CircularObstacle2D(90.3713, 31.5880, 5.0149),
]
rectangular_obstacles = [
    RectangleObstacle2D(16.7098, 16.9611, 10.8686, 8.5024),
    RectangleObstacle2D(21.0424, 51.6855, 10.5168, 9.3864),
    RectangleObstacle2D(87.5924, 97.3590, 8.8218, 9.4436),
    RectangleObstacle2D(69.2475, 20.9249, 10.7889, 8.6901),
    RectangleObstacle2D(6.7638, 47.5327, 8.5484, 7.7492),
    RectangleObstacle2D(88.0585, 8.4206, 9.8547, 10.5486),
    RectangleObstacle2D(2.5603, 76.4629, 9.7399, 7.4066),
    RectangleObstacle2D(17.5367, 6.1873, 9.7904, 8.9090),
    RectangleObstacle2D(52.2582, 89.0204, 8.1154, 7.9922),
    RectangleObstacle2D(54.8860, 52.4636, 10.1351, 7.7044),
    RectangleObstacle2D(74.8421, 11.7063, 10.2229, 7.1359),
    RectangleObstacle2D(41.6773, 18.5897, 8.8510, 8.9693),
    RectangleObstacle2D(40.2649, 96.7344, 8.0322, 7.8306),
    RectangleObstacle2D(44.0761, 41.8925, 9.0145, 10.7516),
    RectangleObstacle2D(96.9435, 86.3312, 9.5088, 10.4911),
    RectangleObstacle2D(20.7034, 45.4841, 7.3415, 10.8680),
    RectangleObstacle2D(94.2697, 65.9418, 8.1842, 7.6518),
    RectangleObstacle2D(94.8467, 36.5604, 10.8719, 8.0919),
    RectangleObstacle2D(50.7900, 24.2166, 9.2079, 8.3341),
    RectangleObstacle2D(66.2407, 59.1539, 10.2580, 10.6480),
    RectangleObstacle2D(22.6719, 3.4642, 7.4709, 8.0117),
    RectangleObstacle2D(43.9868, 70.0384, 9.0220, 8.6638),
    RectangleObstacle2D(9.3382, 75.3920, 7.3146, 9.2339),
    RectangleObstacle2D(12.7850, 87.0119, 7.6349, 9.7555),
    RectangleObstacle2D(86.7186, 22.1514, 10.3965, 10.6846),
]

env = SquareEnvironment(100, 100, circular_obstacles + rectangular_obstacles)
agent = get_unicycle_agent(1)
apply_discrete_random_action(agent)

start = np.array([30.0, 90.0, 0.0])
goal = np.array([79.0, 5.0, 0.0])
goal_radius = 2.0
s = 42

# Load discrete maneuver library
edge_bundle_file_location = str(ROOT / 'edge_bundles/eb_unicycle_discrete_190.npz')
data = np.load(edge_bundle_file_location)
eb = EdgeBundle(data, use_all_edges=True)
edge_ids = np.arange(eb.num_edges, dtype=np.int64)
thetas = eb.start_states[:, 2]
kd_tree_eb = CircularAngleIndexNumba(thetas, ids=edge_ids)


class BBoEViz(BBoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []

    def plan_path(self):
        self.reset_tree()

        # Initialize trees
        self.add_node_for(self.start, -1, None, None, None, 0.0, 0.0)
        self.add_node_rev(self.goal, -1, None, None, None, 0.0, 0.0)

        curr_num_steps = 0
        start_time = time.time()

        while curr_num_steps <= self.max_iter:
            step_record = {}
            step_record['curr_num'] = curr_num_steps
            step_record['rk'] = self.get_rk()

            # 1. Reverse Tree Expansion
            rand_rev = self.sample_random_point()
            rev_nearest_id, rev_nearest_node = self.get_nearest_node_rev(rand_rev)

            prev_rev_id = self.last_added_node_id_rev
            self.extend_tree_rev(rev_nearest_id, rev_nearest_node, rand_rev)

            step_record['rand_rev'] = rand_rev[:2].copy()
            step_record['rev_edges'] = []
            for i in range(prev_rev_id + 1, self.last_added_node_id_rev + 1):
                node = self.tree_rev.nodes[i]['value']
                if node.parent_id != -1:
                    p = self.tree_rev.nodes[node.parent_id]['value']
                    step_record['rev_edges'].append(
                        (p.state[:2].copy(), node.state[:2].copy()))

            # 2. Forward Tree Expansion
            prev_for_id = self.last_added_node_id_for
            c_rand = self.rng.uniform(0, 1)
            e_for_result = None
            for_parent_id = None
            for_parent_node = None
            step_record['type'] = 'random'

            if c_rand < self.exploitation_ratio:
                # Exploitation
                if not self.q_priority.is_empty():
                    for_pop_id, _ = self.q_priority.pop()
                    if for_pop_id is not None:
                        for_pop_node = self.tree_for.nodes[for_pop_id]['value']
                        states_rev = self._node_matrix_rev.get_valid_matrix()
                        for_pos = self._metric_embed(for_pop_node.state)
                        diffs = states_rev - for_pos
                        dists = np.linalg.norm(diffs, axis=1)
                        valid_indices = np.where(dists <= self.get_rk())[0]

                        best_target_score = float('inf')
                        target_point = None
                        for i in valid_indices:
                            rev_node_id = self._node_matrix_rev.ids[i]
                            rev_node = self.tree_rev.nodes[rev_node_id]['value']
                            score = (for_pop_node.cost_so_far + dists[i]
                                     + rev_node.cost_so_far)
                            if score < best_target_score:
                                best_target_score = score
                                target_point = rev_node.state

                        if target_point is not None:
                            e_for_result = self.best_prop_using_edge_bundle(
                                for_pop_node, target_point, self.k_bias)
                            if e_for_result is not None:
                                for_parent_id = for_pop_id
                                for_parent_node = for_pop_node
                                step_record['type'] = 'exploit'
                                step_record['for_pop'] = for_pos[:2].copy()
                                step_record['target'] = target_point[:2].copy()

                # Exploration fallback
                if e_for_result is None:
                    rand_for = self.sample_random_point()
                    for_nearest_id, for_nearest_node = self.get_nearest_node_for(rand_for)
                    e_for_result = self.best_prop_using_edge_bundle(
                        for_nearest_node, rand_for, 1.0)
                    if e_for_result is not None:
                        for_parent_id = for_nearest_id
                        for_parent_node = for_nearest_node
                        if step_record['type'] != 'exploit':
                            step_record['type'] = 'explore'
                            step_record['rand_for'] = rand_for[:2].copy()

            # Random exploration
            if e_for_result is None:
                rand_for = self.sample_random_point()
                for_nearest_id, for_nearest_node = self.get_nearest_node_for(rand_for)
                e_for_result = self.random_prop_using_edge_bundle(
                    for_nearest_node)
                if e_for_result is not None:
                    for_parent_id = for_nearest_id
                    for_parent_node = for_nearest_node
                step_record['type'] = 'random'
                step_record['rand_for'] = rand_for[:2].copy()

            # Add forward node
            if e_for_result is not None:
                self._add_forward_node(for_parent_id, for_parent_node,
                                       *e_for_result)

            # Record new forward edges
            step_record['for_edges'] = []
            for i in range(prev_for_id + 1, self.last_added_node_id_for + 1):
                node = self.tree_for.nodes[i]['value']
                if node.parent_id != -1:
                    p = self.tree_for.nodes[node.parent_id]['value']
                    step_record['for_edges'].append(
                        (p.state[:2].copy(), node.state[:2].copy()))

            self.history.append(step_record)

            if self.path_found or (time.time() - start_time >= self.planning_time):
                break
            curr_num_steps += 1

        total_time = time.time() - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)
        print(f"BBoE Planning Time: {total_time:.2f}s | "
              f"Path Found: {self.path_found} | Path Cost: {self.path_cost:.2f}")


bboe = BBoEViz(
    start=start, goal=goal, goal_radius=goal_radius,
    env=env, agent=agent,
    sampling_time_step=1.0, minimum_time_step=0.1,
    max_iter=10000, planning_time=60.0,
    isvalid_function=agent.is_new_node_valid,
    cost_function=agent.get_cost,
    random_point_function=agent.get_random_point,
    reached_goal_function=agent.agent_reached_goal,
    edge_bundle=eb,
    eb_kd_tree=kd_tree_eb,
    get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
    kd_tree_delta_radius=np.pi,
    translate_function=agent.kd_tree_point_translate_function,
    sort_edges_function=agent.sort_kd_tree_edges,
    exploitation_ratio=0.8,
    heuristic_radius=7.0,
    gamma=14.0,
    k_bias=0.85,
    num_skip_edges=5,
    udf_seed=s, print_logs=True
)

bboe.plan_path()

print(f"Recorded {len(bboe.history)} frames for animation.")

# --- Build the final path for highlighting ---
final_path_edges = []
if bboe.path_found:
    ids, states, controls, timesteps = bboe.get_path_to_node_id(bboe.goal_node_id)
    for i in range(len(states) - 1):
        final_path_edges.append(
            (states[i][:2].copy(), states[i + 1][:2].copy()))

# --- Animation ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect('equal')
ax.set_title("BBoE Iteration: 0 | r_k: 0.0")

# Draw environment
for ob in env.static_circular_obstacles:
    circle = patches.Circle((ob[0], ob[1]), ob[2],
                             facecolor='gray', edgecolor='black')
    ax.add_patch(circle)
for ob in env.static_rectangular_obstacles:
    rect = patches.Rectangle((ob[0], ob[2]), ob[1] - ob[0], ob[3] - ob[2],
                              facecolor='gray', edgecolor='black')
    ax.add_patch(rect)

ax.plot(start[0], start[1], 'ro', markersize=8, label="Start")
ax.plot(goal[0], goal[1], 'go', markersize=8, label="Goal")
ax.add_patch(patches.Circle((goal[0], goal[1]), goal_radius,
             fill=False, edgecolor='g', linestyle='--'))

scat_rev_rand = ax.scatter([], [], c='orange', marker='x', label='Reverse Sample')
scat_for_rand = ax.scatter([], [], c='cyan', marker='x', label='Forward Sample')
scat_target = ax.scatter([], [], c='purple', marker='*', s=150,
                         label='Heuristic Target')

rk_circle = patches.Circle((0, 0), radius=0, fill=False, edgecolor='purple',
                            linestyle=':', linewidth=2, alpha=0)
ax.add_patch(rk_circle)

ax.legend(loc='lower left')

path_drawn = [False]


def update(frame):
    step = bboe.history[frame]
    ax.set_title(f"BBoE Iteration: {step['curr_num']} | "
                 f"r_k: {step['rk']:.2f} | {step['type']}")

    # Reverse tree edges (red)
    for e in step['rev_edges']:
        ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]],
                color='red', alpha=0.6, linewidth=0.6)

    # Forward tree edges (blue for exploit, green otherwise)
    for e in step['for_edges']:
        color = 'blue' if step['type'] == 'exploit' else 'green'
        ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]],
                color=color, alpha=0.6, linewidth=0.6)

    # Reverse sample marker
    scat_rev_rand.set_offsets(step['rand_rev'].reshape(1, 2))

    # Exploitation markers
    if step['type'] == 'exploit':
        scat_for_rand.set_offsets(np.empty((0, 2)))
        scat_target.set_offsets(step['target'].reshape(1, 2))
        rk_circle.set_alpha(1)
        rk_circle.center = (step['for_pop'][0], step['for_pop'][1])
        rk_circle.set_radius(step['rk'])
    else:
        if 'rand_for' in step:
            scat_for_rand.set_offsets(step['rand_for'].reshape(1, 2))
        else:
            scat_for_rand.set_offsets(np.empty((0, 2)))
        scat_target.set_offsets(np.empty((0, 2)))
        rk_circle.set_alpha(0)

    # Draw final path on last frame
    if frame == len(bboe.history) - 1 and not path_drawn[0]:
        for e in final_path_edges:
            ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]],
                    color='red', linewidth=2.5, alpha=0.9)
        path_drawn[0] = True

    return scat_rev_rand, scat_for_rand, scat_target, rk_circle


print("Rendering GIF...")
ani = FuncAnimation(fig, update, frames=len(bboe.history),
                    blit=False, repeat=False)
ani.save(str(ROOT / 'media/bboe_execution.gif'), writer='pillow', fps=15)
print("Finished rendering GIF to media/bboe_execution.gif")
