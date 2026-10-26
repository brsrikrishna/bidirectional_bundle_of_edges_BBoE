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
from mapf_env_square_agent_treaded import get_treaded_agent
from edge_bundle import EdgeBundle
from kd_tree_treaded import VlVrTree
from bboe import BBoE

# Source obstacles + env from compare_treaded.py so the gif always
# reflects the latest benchmark layout (currently the v3 25-obstacle env).
_src_path = str(ROOT / 'trial_scripts/compare_treaded.py')
_src = open(_src_path).read()
_setup = _src.split('seeds = list(range(50))')[0]
_ns = {'__file__': _src_path}
exec(_setup, _ns)
circular_obstacles = _ns['circular_obstacles']
rectangular_obstacles = _ns['rectangular_obstacles']
env = _ns['env']
agent = _ns['agent']

start = np.array([7.0, 2.0, 0.0, 0.0, 0.0])
goal = np.array([22.5, 23.5, 0.0, 0.0, 0.0])  # 5-D paper-faithful
goal_radius = 0.75
s = 5191  # seed 5191 succeeded in 0.85s during benchmark

# Edge bundle + KD-tree
edge_bundle_file = str(ROOT / 'edge_bundles/eb_treaded_kinodynamic_TI_edges_100000.npz')
data = np.load(edge_bundle_file)
eb = EdgeBundle(data, fix_num_edges=50000, use_all_edges=False, rng_seed=42)
edge_ids = np.arange(eb.num_edges, dtype=np.int64)
vl = eb.start_states[:, 3]
vr = eb.start_states[:, 4]
kd_tree_eb = VlVrTree(vl, vr, ids=edge_ids,
                       vl_scale=agent.max_speed, vr_scale=agent.max_speed)


class BBoEViz(BBoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []

    def plan_path(self):
        self.reset_tree()
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
                if not self.q_priority.is_empty():
                    for_pop_id, _ = self.q_priority.pop()
                    if for_pop_id is not None:
                        for_pop_node = self.tree_for.nodes[for_pop_id]['value']
                        # States are already d_X-embedded via _metric_embed.
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

            if e_for_result is None:
                rand_for = self.sample_random_point()
                for_nearest_id, for_nearest_node = self.get_nearest_node_for(rand_for)
                e_for_result = self.random_prop_using_edge_bundle(for_nearest_node)
                if e_for_result is not None:
                    for_parent_id = for_nearest_id
                    for_parent_node = for_nearest_node
                step_record['type'] = 'random'

            if e_for_result is not None:
                self._add_forward_node(for_parent_id, for_parent_node,
                                       *e_for_result)

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
    max_iter=100000, planning_time=60.0,
    isvalid_function=agent.is_new_node_valid,
    cost_function=agent.get_cost,
    random_point_function=agent.get_random_point,
    reached_goal_function=agent.agent_reached_goal,
    edge_bundle=eb,
    eb_kd_tree=kd_tree_eb,
    get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
    kd_tree_delta_radius=0.1,
    translate_function=agent.kd_tree_point_translate_function,
    sort_edges_function=agent.sort_kd_tree_edges,
    exploitation_ratio=0.7,
    heuristic_radius=8.0,
    gamma=7.0,
    delta_reach=1.0,  # Fix B: mark reached rev nodes as consumed
    k_bias=0.85,
    num_skip_edges=5,
    udf_seed=s, print_logs=True
)

bboe.plan_path()
print(f"Recorded {len(bboe.history)} frames for animation.")

# --- Build final path ---
final_path_edges = []
if bboe.path_found:
    ids, states, controls, timesteps = bboe.get_path_to_node_id(bboe.goal_node_id)
    for i in range(len(states) - 1):
        final_path_edges.append(
            (states[i][:2].copy(), states[i + 1][:2].copy()))

# --- Pre-collect edges by frame ---
all_for_edges_by_frame = [step['for_edges'] for step in bboe.history]
all_rev_edges_by_frame = [step['rev_edges'] for step in bboe.history]

# --- Animation ---
fig, ax = plt.subplots(figsize=(7, 7), dpi=72)
ax.set_xlim(0, 25)
ax.set_ylim(0, 25)
ax.set_aspect('equal')
ax.set_title("BBoE (Treaded Vehicle) Iteration: 0")

# Draw obstacles
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

# Legend
ax.plot([], [], color='blue', linewidth=1.5, label='Forward Tree')
ax.plot([], [], color='red', linewidth=1.5, label='Reverse Tree')
ax.legend(loc='lower left')

path_drawn = [False]
last_drawn_frame = [-1]


def update(frame):
    step = bboe.history[frame]
    ax.set_title(f"BBoE (Treaded Vehicle) Iteration: {step['curr_num']} | "
                 f"r_k: {step['rk']:.2f} | {step['type']}")

    for f in range(last_drawn_frame[0] + 1, frame + 1):
        if f < len(all_for_edges_by_frame):
            for e in all_for_edges_by_frame[f]:
                ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]],
                        color='blue', alpha=0.6, linewidth=1.0)
            for e in all_rev_edges_by_frame[f]:
                ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]],
                        color='red', alpha=0.6, linewidth=1.0)
    last_drawn_frame[0] = frame

    if frame == len(bboe.history) - 1 and not path_drawn[0]:
        for e in final_path_edges:
            ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]],
                    color='lime', linewidth=3, alpha=0.9)
        path_drawn[0] = True


# Subsample frames
total_frames = len(bboe.history)
if total_frames > 60:
    step = total_frames // 60
    frame_indices = list(range(0, total_frames - 1, step))
    if frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)
else:
    frame_indices = list(range(total_frames))

print(f"Rendering GIF with {len(frame_indices)} frames "
      f"(subsampled from {total_frames})...")
ani = FuncAnimation(fig, update, frames=frame_indices,
                    blit=False, repeat=False)
ani.save(str(ROOT / 'media/bboe_treaded_execution.gif'), writer='pillow', fps=15)
print("Finished rendering GIF to media/bboe_treaded_execution.gif")
