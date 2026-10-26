import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]   # repository root
import time
sys.path.append(str(ROOT / 'src'))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

from Environments import CuboidEnvironment, CuboidObstacle3D
from mapf_env_cuboid_agent_quadcopter6d import get_quadcopter_agent
from edge_bundle import EdgeBundle
from kd_tree_quadcopter6d import VxyzTree
from bboe import BBoE

# Environment: 40x40x10
ENV_Y = 40.0
ENV_Z = 10.0

def wall_with_hole(x_start, x_width, y_hole_start, y_hole_end,
                   z_hole_start, z_hole_end):
    obs = []
    if y_hole_start > 0:
        cy = y_hole_start / 2
        obs.append(CuboidObstacle3D(
            x_start + x_width / 2, cy, ENV_Z / 2,
            x_width, y_hole_start, ENV_Z))
    if y_hole_end < ENV_Y:
        w = ENV_Y - y_hole_end
        cy = y_hole_end + w / 2
        obs.append(CuboidObstacle3D(
            x_start + x_width / 2, cy, ENV_Z / 2,
            x_width, w, ENV_Z))
    if z_hole_start > 0:
        obs.append(CuboidObstacle3D(
            x_start + x_width / 2,
            (y_hole_start + y_hole_end) / 2,
            z_hole_start / 2,
            x_width, y_hole_end - y_hole_start, z_hole_start))
    if z_hole_end < ENV_Z:
        h = ENV_Z - z_hole_end
        obs.append(CuboidObstacle3D(
            x_start + x_width / 2,
            (y_hole_start + y_hole_end) / 2,
            z_hole_end + h / 2,
            x_width, y_hole_end - y_hole_start, h))
    return obs

obstacles = []
# Narrow-bottleneck quadrotor env (matches compare_quadrotor_narrow.py):
# 2 walls (width 4) with NARROW 4x3 holes (vs original 8x4).
obstacles += wall_with_hole(12, 4, 10, 14, 3.5, 6.5)    # W1: 4x3 hole, low-Y
obstacles += wall_with_hole(24, 4, 26, 30, 5.5, 8.5)    # W2: 4x3 hole, high-Y

env = CuboidEnvironment(length=40.0, breadth=40.0, height=10.0, obs=obstacles)
agent = get_quadcopter_agent(1)

start = np.array([4.0, 12.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64)
goal_6d = np.array([36.0, 28.0, 7.0, 0.0, 0.0, 0.0], dtype=np.float64)
goal_radius = 2.0
s = 42

# Load edge bundle
edge_bundle_file = str(ROOT / 'edge_bundles/eb_quadcopter6d_kinodynamic_TI_edges_200000.npz')
data = np.load(edge_bundle_file)
eb = EdgeBundle(data, fix_num_edges=100000, use_all_edges=False, rng_seed=42)
edge_ids = np.arange(eb.num_edges, dtype=np.int64)
vx = eb.start_states[:, 3]
vy = eb.start_states[:, 4]
vz = eb.start_states[:, 5]
kd_tree_eb = VxyzTree(vx, vy, vz, ids=edge_ids, scales=(1.0, 1.0, 1.0))


# ============================================================
# Helper: draw a cuboid in 3D
# ============================================================
def draw_cuboid(ax, cx, cy, cz, lx, ly, lz, color='gray', alpha=0.3):
    """Draw a cuboid centered at (cx,cy,cz) with dimensions (lx,ly,lz)."""
    x0, x1 = cx - lx/2, cx + lx/2
    y0, y1 = cy - ly/2, cy + ly/2
    z0, z1 = cz - lz/2, cz + lz/2

    # 6 faces, each a list of 4 corners
    verts = [
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],  # bottom
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],  # top
        [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],  # front
        [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],  # back
        [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],  # left
        [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],  # right
    ]
    ax.add_collection3d(Poly3DCollection(
        verts, facecolors=color, edgecolors='black',
        linewidths=0.3, alpha=alpha))


# ============================================================
# BBoEViz: records 3D edges per iteration
# ============================================================
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
                        (p.state[:3].copy(), node.state[:3].copy()))

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
                        (p.state[:3].copy(), node.state[:3].copy()))

            self.history.append(step_record)

            if self.path_found or (time.time() - start_time >= self.planning_time):
                break
            curr_num_steps += 1

        total_time = time.time() - start_time
        self.path_time = round(self.path_time, self.roundoff_digits)
        print(f"BBoE Planning Time: {total_time:.2f}s | "
              f"Path Found: {self.path_found} | Path Cost: {self.path_cost:.2f}")


# ============================================================
# Run planner
# ============================================================
bboe = BBoEViz(
    start=start, goal=goal_6d, goal_radius=goal_radius,
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
    kd_tree_delta_radius=0.05,  # TUNED: tighter v-match (~40 candidates/call)
    translate_function=agent.kd_tree_point_translate_function,
    sort_edges_function=agent.sort_kd_tree_edges,
    exploitation_ratio=0.95,    # TUNED: lean harder on bundle
    heuristic_radius=6.0,       # TUNED: wider fwd-rev connection
    gamma=6.0,
    k_bias=0.85,
    num_skip_edges=1,           # TUNED: evaluate every sorted edge
    udf_seed=s, print_logs=True
)

bboe.plan_path()
print(f"Recorded {len(bboe.history)} frames for animation.")

# --- Build final path (3D) ---
final_path_xyz = []
if bboe.path_found:
    ids, states, controls, timesteps = bboe.get_path_to_node_id(bboe.goal_node_id)
    final_path_xyz = states[:, :3].copy()

# --- Pre-collect all edges by frame ---
all_for_edges_by_frame = [step['for_edges'] for step in bboe.history]
all_rev_edges_by_frame = [step['rev_edges'] for step in bboe.history]

# ============================================================
# 3D Animation
# ============================================================
fig = plt.figure(figsize=(7, 5), dpi=72)
ax = fig.add_subplot(111, projection='3d')

# Draw obstacles
# Wall 1 segments
obs_data = [
    # (cx, cy, cz, lx, ly, lz)
    # Wall 1: X=12-16
    (14, 4, 5, 4, 8, 10),       # left segment Y=0-8
    (14, 28, 5, 4, 24, 10),     # right segment Y=16-40
    (14, 12, 1.5, 4, 8, 3),     # bottom segment (hole Y=8-16, Z=0-3)
    (14, 12, 8.5, 4, 8, 3),     # top segment (hole Y=8-16, Z=7-10)
    # Wall 2: X=24-28
    (26, 12, 5, 4, 24, 10),     # left segment Y=0-24
    (26, 36, 5, 4, 8, 10),      # right segment Y=32-40
    (26, 28, 2.5, 4, 8, 5),     # bottom segment (hole Y=24-32, Z=0-5)
    (26, 28, 9.5, 4, 8, 1),     # top segment (hole Y=24-32, Z=9-10)
]

for cx, cy, cz, lx, ly, lz in obs_data:
    draw_cuboid(ax, cx, cy, cz, lx, ly, lz, color='gray', alpha=0.15)

# Start and goal
ax.scatter(*start[:3], c='red', s=80, zorder=5, label='Start')
ax.scatter(*goal_6d[:3], c='green', s=80, zorder=5, label='Goal')

ax.set_xlim(0, 40)
ax.set_ylim(0, 40)
ax.set_zlim(0, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("BBoE Iteration: 0")
ax.view_init(elev=25, azim=-60)

# Legend entries
ax.plot([], [], [], color='blue', linewidth=1.5, label='Forward Tree')
ax.plot([], [], [], color='red', linewidth=1.5, label='Reverse Tree')
ax.legend(loc='upper left')

path_drawn = [False]
last_drawn_frame = [-1]


def update(frame):
    step = bboe.history[frame]
    ax.set_title(f"BBoE Iteration: {step['curr_num']} | "
                 f"r_k: {step['rk']:.2f} | {step['type']}")

    # Draw all edges from frames we haven't drawn yet
    for f in range(last_drawn_frame[0] + 1, frame + 1):
        if f < len(all_for_edges_by_frame):
            for e in all_for_edges_by_frame[f]:
                ax.plot([e[0][0], e[1][0]],
                        [e[0][1], e[1][1]],
                        [e[0][2], e[1][2]],
                        color='blue', alpha=0.6, linewidth=1.0)
            for e in all_rev_edges_by_frame[f]:
                ax.plot([e[0][0], e[1][0]],
                        [e[0][1], e[1][1]],
                        [e[0][2], e[1][2]],
                        color='red', alpha=0.6, linewidth=1.0)
    last_drawn_frame[0] = frame

    # Draw final path on last frame
    if frame == len(bboe.history) - 1 and not path_drawn[0]:
        if len(final_path_xyz) > 1:
            ax.plot(final_path_xyz[:, 0], final_path_xyz[:, 1],
                    final_path_xyz[:, 2],
                    color='lime', linewidth=3, alpha=0.9, label='Solution')
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

print(f"Rendering GIF with {len(frame_indices)} frames (subsampled from {total_frames})...")
ani = FuncAnimation(fig, update, frames=frame_indices,
                    blit=False, repeat=False)
ani.save(str(ROOT / 'media/bboe_quadrotor_execution.gif'), writer='pillow', fps=15)
print("Finished rendering GIF to media/bboe_quadrotor_execution.gif")
