"""
Per-planner planning-process GIF on a random Unicycle environment of the
chosen difficulty (Easy / Medium / Hard / Very Hard). Mirrors the recipe
used in compare_difficulty_curated.py so the env matches the benchmark.

Tree edges are drawn progressively in 2D (x,y), forward in blue and reverse
in red (BBoE / GBRRT). The solution path is highlighted in lime on the
final frame.

Usage:
    python3 trial_scripts/create_difficulty_planner_gif.py \
        --planner=bboe --difficulty="Very Hard"
    python3 trial_scripts/create_difficulty_planner_gif.py \
        --planner=rrt --difficulty=Hard --seed=42 --frames=80
"""
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]   # repository root
import time
import argparse
import os
sys.path.append(str(ROOT / 'src'))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from Environments import SquareEnvironment, CircularObstacle2D, RectangleObstacle2D
from mapf_env_square_agent_unicycle import get_unicycle_agent
from edge_bundle import EdgeBundle
from kd_tree_unicycle import CircularAngleIndexNumba
from rrt import RRT
from sst import SST
from gbrrt import GBRRT
from bboe import BBoE


# ============================================================
# CLI
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument('--planner', default='bboe',
                choices=['rrt', 'sst', 'gbrrt', 'bboe'])
ap.add_argument('--difficulty', default='Very Hard',
                choices=['Easy', 'Medium', 'Hard', 'Very Hard'])
ap.add_argument('--trial', type=int, default=0,
                help="Curated trial index 0..49 (matches compare_difficulty_curated.py)")
ap.add_argument('--budget', type=float, default=5.0,
                help="Planning time budget (s). Defaults to benchmark value.")
ap.add_argument('--out_dir', default='media')
ap.add_argument('--frames', type=int, default=80,
                help='Total animation frames (subsamples tree growth)')
ap.add_argument('--fps', type=int, default=20)
args = ap.parse_args()


# ============================================================
# Env generation (must mirror compare_difficulty_curated.py)
# ============================================================
ENV_SIZE = 100
start = np.array([30.0, 90.0, 0.0])
goal_2d = np.array([79.0, 5.0])
goal_3d = np.array([79.0, 5.0, 0.0])
goal_radius = 2.0

DIFFICULTIES = {
    "Easy": {
        "n_circ": 4, "n_rect": 3,
        "circ_r_range": (3, 5), "rect_w_range": (5, 8),
    },
    "Medium": {
        "n_circ": 22, "n_rect": 18,
        "circ_r_range": (4, 7), "rect_w_range": (7, 12),
    },
    "Hard": {
        "n_circ": 33, "n_rect": 25,
        "circ_r_range": (4, 6), "rect_w_range": (7, 11),
    },
    "Very Hard": {
        "n_circ": 36, "n_rect": 27,
        "circ_r_range": (4, 6), "rect_w_range": (7, 10),
    },
}


def point_free_of_obstacles(px, py, radius, circ_obs, rect_obs):
    for ob in circ_obs:
        dx, dy = px - ob.x, py - ob.y
        if dx*dx + dy*dy < (radius + ob.r + 1.0)**2:
            return False
    for ob in rect_obs:
        half_w = ob.w / 2.0 + radius + 1.0
        half_h = ob.h / 2.0 + radius + 1.0
        if abs(px - ob.x) < half_w and abs(py - ob.y) < half_h:
            return False
    return True


def generate_random_environment(difficulty_cfg, rng):
    n_circ = difficulty_cfg["n_circ"]
    n_rect = difficulty_cfg["n_rect"]
    cr_min, cr_max = difficulty_cfg["circ_r_range"]
    rw_min, rw_max = difficulty_cfg["rect_w_range"]

    margin = -3.0  # negative — obstacles can extend past walls (no free corridor)
    max_attempts = 500

    circ_obs, rect_obs = [], []

    for _ in range(n_circ):
        for _ in range(max_attempts):
            r = rng.uniform(cr_min, cr_max)
            cx = rng.uniform(margin + r, ENV_SIZE - margin - r)
            cy = rng.uniform(margin + r, ENV_SIZE - margin - r)
            dx_s = cx - start[0]; dy_s = cy - start[1]
            dx_g = cx - goal_2d[0]; dy_g = cy - goal_2d[1]
            if dx_s*dx_s + dy_s*dy_s < (r + goal_radius + 2.0)**2:
                continue
            if dx_g*dx_g + dy_g*dy_g < (r + goal_radius + 2.0)**2:
                continue
            circ_obs.append(CircularObstacle2D(cx, cy, r))
            break

    for _ in range(n_rect):
        for _ in range(max_attempts):
            w = rng.uniform(rw_min, rw_max)
            h = rng.uniform(rw_min, rw_max)
            cx = rng.uniform(margin + w/2, ENV_SIZE - margin - w/2)
            cy = rng.uniform(margin + h/2, ENV_SIZE - margin - h/2)
            if not point_free_of_obstacles(start[0], start[1], goal_radius,
                                           [], [type('R', (), {'x': cx, 'y': cy, 'w': w, 'h': h})()]):
                continue
            if not point_free_of_obstacles(goal_2d[0], goal_2d[1], goal_radius,
                                           [], [type('R', (), {'x': cx, 'y': cy, 'w': w, 'h': h})()]):
                continue
            rect_obs.append(RectangleObstacle2D(cx, cy, w, h))
            break

    return SquareEnvironment(ENV_SIZE, ENV_SIZE, circ_obs + rect_obs)


# Match compare_difficulty_curated.py exactly: env_seed from CURATED_ENV_SEEDS,
# udf_seed from TRIAL_SEEDS at the chosen trial index.
CURATED_ENV_SEEDS = {
    "Easy":      [15795, 860, 76820, 54886, 6265, 82386, 37194, 87498, 44131, 60263],
    "Medium":    [16795, 1861, 77820, 55886, 7265, 83386, 38194, 88499, 45131, 61265],
    "Hard":      [17797, 2860, 78820, 56886, 8265, 84387, 39196, 89498, 46131, 62263],
    "Very Hard": [18795, 3862, 79820, 57886, 9266, 85390, 40194, 90499, 47131, 63263],
}
np.random.seed(42)
TRIAL_SEEDS = np.random.randint(0, 100000, size=50)  # match compare_difficulty_curated.py

trial_idx = max(0, min(49, args.trial))
# compare_difficulty_curated.py cycles through 10 envs across 50 trials
env_seed = CURATED_ENV_SEEDS[args.difficulty][trial_idx % 10]
trial_seed = int(TRIAL_SEEDS[trial_idx])
env_rng = np.random.default_rng(env_seed)
env = generate_random_environment(DIFFICULTIES[args.difficulty], env_rng)


# ============================================================
# Agent + edge bundle (shared)
# ============================================================
agent = get_unicycle_agent(1)
T_max = 1.0

eb_path = str(ROOT / 'edge_bundles/eb_unicycle_discrete_190.npz')
data = np.load(eb_path)
eb = EdgeBundle(data, use_all_edges=True)
edge_ids = np.arange(eb.num_edges, dtype=np.int64)
thetas = eb.start_states[:, 2]
kd_tree_eb = CircularAngleIndexNumba(thetas, ids=edge_ids)

v_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
w_vals = [i * np.pi / 18.0 for i in range(10)]
discrete_actions = []
for v in v_vals:
    for w in w_vals:
        discrete_actions.append((v, w))
        if v != 0: discrete_actions.append((-v, w))
        if w != 0: discrete_actions.append((v, -w))
        if v != 0 and w != 0: discrete_actions.append((-v, -w))
discrete_actions = list(set(discrete_actions))


def get_discrete_random_action(rng):
    return np.array(discrete_actions[rng.integers(0, len(discrete_actions))])


agent.get_random_action = get_discrete_random_action


# ============================================================
# Planner factories — match compare_difficulty_curated.py exactly
# ============================================================
MAX_ITER = 10000

def mk_rrt():
    return RRT(
        start=start, goal=goal_2d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=args.budget,
        num_extension_trials=40, goal_bias=0.05,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb, eb_kd_tree=kd_tree_eb,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        udf_seed=trial_seed, print_logs=False)

def mk_sst():
    return SST(
        start=start, goal=goal_2d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=args.budget,
        num_extension_trials=40, goal_bias=0.05,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb, eb_kd_tree=kd_tree_eb,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        udf_seed=trial_seed, print_logs=False,
        best_near_radius=5.0)

def mk_gbrrt():
    return GBRRT(
        start=start, goal=goal_3d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=args.budget,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        num_extension_trials=40,
        heuristic_radius=7.0, gamma=14.0, exploitation_ratio=0.8,
        edge_bundle=eb, eb_kd_tree=kd_tree_eb,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        udf_seed=trial_seed, print_logs=False)

def mk_bboe():
    return BBoE(
        start=start, goal=goal_3d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=args.budget,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb, eb_kd_tree=kd_tree_eb,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        exploitation_ratio=0.8,
        heuristic_radius=7.0, gamma=14.0,
        k_bias=0.85, num_skip_edges=5,
        udf_seed=trial_seed, print_logs=False)


factories = {
    'rrt': mk_rrt, 'sst': mk_sst, 'gbrrt': mk_gbrrt, 'bboe': mk_bboe,
}
print(f"[{args.planner.upper()}] difficulty={args.difficulty} "
      f"seed={trial_seed} budget={args.budget}s "
      f"obstacles={len(env.obstacles)}")

# Warm numba JIT by running a throwaway instance for 0.3s. JIT cache is
# process-global so the second instance uses pre-compiled code while
# starting with a fresh RNG (so it matches the benchmark exactly).
print("Warming JIT...")
warm = factories[args.planner]()
warm.planning_time = 0.3
warm.plan_path()
del warm

planner = factories[args.planner]()
t0 = time.time()
planner.plan_path()
plan_dt = time.time() - t0
print(f"  done in {plan_dt:.2f}s — found={planner.path_found} "
      f"cost={planner.path_cost if planner.path_found else 'inf'}")


# ============================================================
# Tree extraction in node-id (≈ insertion) order
# ============================================================
is_bidir = args.planner in ('bboe', 'gbrrt')
is_sst = args.planner == 'sst'


def extract_tree_edges_nx(tree):
    edges = []
    for nid in sorted(tree.nodes.keys()):
        node = tree.nodes[nid]['value']
        if node.parent_id == -1:
            continue
        parent_state = tree.nodes[node.parent_id]['value'].state
        edges.append((parent_state[:2].copy(), node.state[:2].copy()))
    return edges


def extract_tree_edges_sst(nm):
    edges = []
    for i in range(nm.count):
        pidx = int(nm.parent[i])
        if pidx < 0:
            continue
        edges.append((nm.state[pidx][:2].copy(),
                      nm.state[i][:2].copy()))
    return edges


if is_bidir:
    for_edges = extract_tree_edges_nx(planner.tree_for)
    rev_edges = extract_tree_edges_nx(planner.tree_rev)
elif is_sst:
    for_edges = extract_tree_edges_sst(planner._node_matrix)
    rev_edges = []
else:
    for_edges = extract_tree_edges_nx(planner.tree)
    rev_edges = []

# Solution path
sol_edges = []
if planner.path_found:
    _, states, _, _ = planner.get_path_to_node_id(planner.goal_node_id)
    states = np.asarray(states)
    for i in range(len(states) - 1):
        sol_edges.append((states[i][:2].copy(), states[i + 1][:2].copy()))


# ============================================================
# Render animation
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=90)
ax.set_xlim(0, ENV_SIZE)
ax.set_ylim(0, ENV_SIZE)
ax.set_aspect('equal')

# Draw obstacles
for ob in env.static_circular_obstacles:
    ax.add_patch(patches.Circle((ob[0], ob[1]), ob[2],
                                 facecolor='gray', edgecolor='black',
                                 linewidth=0.5))
for ob in env.static_rectangular_obstacles:
    x0, x1, y0, y1 = ob[0], ob[1], ob[2], ob[3]
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    facecolor='gray', edgecolor='black',
                                    linewidth=0.5))

ax.plot(start[0], start[1], 'o', markersize=10,
        markerfacecolor='#555555', markeredgecolor='black', label='Start')
ax.plot(goal_2d[0], goal_2d[1], 'o', markersize=10,
        markerfacecolor='lime', markeredgecolor='black', label='Goal')
ax.add_patch(patches.Circle((goal_2d[0], goal_2d[1]), goal_radius,
                             fill=False, edgecolor='lime', linestyle='--'))

# Legend proxies
ax.plot([], [], color='steelblue', lw=1.2, label='Forward tree')
if is_bidir:
    ax.plot([], [], color='tomato', lw=1.2, label='Reverse tree')
ax.plot([], [], color='lime', lw=2.5, label='Solution')
ax.legend(loc='upper right', fontsize=8)


# Frame plan: pace by the bigger tree
total_for = len(for_edges)
total_rev = len(rev_edges)
master_total = max(total_for, total_rev, 1)


def subsample(n_items, n_frames):
    if n_items <= n_frames:
        return list(range(n_items))
    return [int(i * n_items / n_frames) for i in range(n_frames)] + [n_items - 1]


frame_idxs = subsample(master_total, args.frames)
frames = []
for idx in frame_idxs:
    fwd_n = int(min(total_for, (idx + 1) * total_for / master_total))
    rev_n = int(min(total_rev, (idx + 1) * total_rev / master_total)) if is_bidir else 0
    frames.append((idx, fwd_n, rev_n))

drawn_for = [0]
drawn_rev = [0]
sol_drawn = [False]


def update(frame_tuple):
    idx, fwd_n, rev_n = frame_tuple
    for k in range(drawn_for[0], fwd_n):
        p1, p2 = for_edges[k]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='steelblue', alpha=0.55, linewidth=0.6)
    drawn_for[0] = fwd_n

    for k in range(drawn_rev[0], rev_n):
        p1, p2 = rev_edges[k]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='tomato', alpha=0.55, linewidth=0.6)
    drawn_rev[0] = rev_n

    status = "SOLVED" if planner.path_found else "TIMEOUT"
    ax.set_title(f"{args.planner.upper()}  |  {args.difficulty}  |  "
                 f"seed {trial_seed}  |  fw {fwd_n} rv {rev_n}  |  "
                 f"{status} {plan_dt:.2f}s", fontsize=10)

    if frame_tuple is frames[-1] and not sol_drawn[0] and len(sol_edges) > 0:
        for p1, p2 in sol_edges:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color='lime', linewidth=2.5, alpha=0.95, zorder=8)
        sol_drawn[0] = True


ani = FuncAnimation(fig, update, frames=frames, blit=False, repeat=False)
os.makedirs(args.out_dir, exist_ok=True)
diff_label = args.difficulty.lower().replace(' ', '_')
out_path = f"{args.out_dir}/{args.planner}_unicycle_{diff_label}.gif"
ani.save(out_path, writer='pillow', fps=args.fps)
print(f"Saved {out_path}")
