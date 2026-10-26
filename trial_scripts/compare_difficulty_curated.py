"""
Run all 5 planners on user-curated unicycle environments.

Env seeds are taken from a curated list (one approved env per
(difficulty, trial_idx)). Planner RNG (`udf_seed`) is still tied to the
trial slot's reference seed.
"""
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]   # repository root
import time
sys.path.append(str(ROOT / 'src'))
import numpy as np

from Environments import SquareEnvironment, CircularObstacle2D, RectangleObstacle2D
from mapf_env_square_agent_unicycle import get_unicycle_agent
from rrt import RRT
from sst import SST
from gbrrt import GBRRT
from bboe import BBoE
from db_rrt import DbRRT
from db_rrt_c import DbRRTC
from BoE import BoE
from edge_bundle import EdgeBundle
from kd_tree_unicycle import CircularAngleIndexNumba

# ---- CLI overrides ----
radius_override = None
planners_filter = None
for a in sys.argv:
    if a.startswith("--radius="):
        radius_override = float(a.split("=", 1)[1])
    elif a.startswith("--planners="):
        planners_filter = set(s.strip().lower()
                              for s in a.split("=", 1)[1].split(","))

# ============================================================
# Curated env seeds (10 per difficulty)
# ============================================================
CURATED_ENV_SEEDS = {
    "Easy":      [15795, 860, 76820, 54886, 6265, 82386, 37194, 87498, 44131, 60263],
    "Medium":    [16795, 1861, 77820, 55886, 7265, 83386, 38194, 88499, 45131, 61265],
    "Hard":      [17797, 2860, 78820, 56886, 8265, 84387, 39196, 89498, 46131, 62263],
    "Very Hard": [18795, 3862, 79820, 57886, 9266, 85390, 40194, 90499, 47131, 63263],
}

# Reference trial seeds (used as udf_seed — planner RNG)
np.random.seed(42)
TRIAL_SEEDS = np.random.randint(0, 100000, size=50)


# ============================================================
# Agent + Edge Bundles
#   * RRT, SST, GBRRT use the original 190-edge fixed-duration library.
#   * BBoE uses the new 1000-edge random-duration library.
# ============================================================
agent = get_unicycle_agent(1)
if radius_override is not None:
    agent.radius = radius_override
    print(f"Overriding unicycle radius = {agent.radius}")
T_max = 1.0

# 190-edge fixed-duration library (used by all EB-aware planners)
data_190 = np.load(str(ROOT / 'edge_bundles/eb_unicycle_discrete_190.npz'))
eb_190 = EdgeBundle(data_190, use_all_edges=True)
ids_190 = np.arange(eb_190.num_edges, dtype=np.int64)
kd_tree_190 = CircularAngleIndexNumba(eb_190.start_states[:, 2], ids=ids_190)

v_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
w_vals = [i * np.pi / 18.0 for i in range(10)]
actions = []
for v in v_vals:
    for w in w_vals:
        actions.append((v, w))
        if v != 0: actions.append((-v, w))
        if w != 0: actions.append((v, -w))
        if v != 0 and w != 0: actions.append((-v, -w))
discrete_actions = list(set(actions))


def get_discrete_random_action(rng):
    return np.array(discrete_actions[rng.integers(0, len(discrete_actions))])


agent.get_random_action = get_discrete_random_action


# ============================================================
# Fixed start/goal
# ============================================================
start = np.array([30.0, 90.0, 0.0])
goal_2d = np.array([79.0, 5.0])
goal_3d = np.array([79.0, 5.0, 0.0])
goal_radius = 2.0


# ============================================================
# Difficulty configs
# ============================================================
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

ENV_SIZE = 100


def point_free_of_obstacles(px, py, radius, circ_obs, rect_obs):
    for ob in circ_obs:
        dx = px - ob.x; dy = py - ob.y
        if dx*dx + dy*dy < (radius + ob.r + 1.0)**2:
            return False
    for ob in rect_obs:
        if abs(px - ob.x) < ob.w/2.0 + radius + 1.0 and \
           abs(py - ob.y) < ob.h/2.0 + radius + 1.0:
            return False
    return True


def generate_random_environment(difficulty_cfg, rng):
    n_circ = difficulty_cfg["n_circ"]
    n_rect = difficulty_cfg["n_rect"]
    cr_min, cr_max = difficulty_cfg["circ_r_range"]
    rw_min, rw_max = difficulty_cfg["rect_w_range"]

    margin = -3.0  # obstacles can extend past walls
    max_attempts = 500

    circ_obs, rect_obs = [], []

    for _ in range(n_circ):
        for _ in range(max_attempts):
            r = rng.uniform(cr_min, cr_max)
            cx = rng.uniform(margin + r, ENV_SIZE - margin - r)
            cy = rng.uniform(margin + r, ENV_SIZE - margin - r)
            if (cx-start[0])**2 + (cy-start[1])**2 < (r+goal_radius+2.0)**2: continue
            if (cx-goal_2d[0])**2 + (cy-goal_2d[1])**2 < (r+goal_radius+2.0)**2: continue
            circ_obs.append(CircularObstacle2D(cx, cy, r))
            break

    for _ in range(n_rect):
        for _ in range(max_attempts):
            w = rng.uniform(rw_min, rw_max)
            h = rng.uniform(rw_min, rw_max)
            cx = rng.uniform(margin + w/2, ENV_SIZE - margin - w/2)
            cy = rng.uniform(margin + h/2, ENV_SIZE - margin - h/2)
            R = type('R', (), {'x': cx, 'y': cy, 'w': w, 'h': h})()
            if not point_free_of_obstacles(start[0], start[1], goal_radius, [], [R]): continue
            if not point_free_of_obstacles(goal_2d[0], goal_2d[1], goal_radius, [], [R]): continue
            rect_obs.append(RectangleObstacle2D(cx, cy, w, h))
            break

    return SquareEnvironment(ENV_SIZE, ENV_SIZE, circ_obs + rect_obs)


# ============================================================
# Planner runners
# ============================================================
PLANNING_TIME = 5.0
MAX_ITER = 10000


# ---- RRT/SST/GBRRT each ship as two variants ----
# (MC) pure Monte Carlo propagation: no maneuver library access. The
#      190-edge bundle has timesteps fixed at 1.0 s and actions drawn from
#      the discrete primitive set, so the MC variant uses
#      `use_fixed_sampling_time=True` (always T_max=1.0 s) AND the agent's
#      already-overridden discrete-action sampler — same generating
#      distribution as the library, just sampled freshly each iteration.
# (EB) maneuver library: pulls (action, t) directly from `eb_190`.

def run_rrt_mc(env, seed):
    p = RRT(start=start, goal=goal_2d, goal_radius=goal_radius,
        env=env, agent=agent,
        use_fixed_sampling_time=True,   # bundle is fixed t=1.0s
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        num_extension_trials=40, goal_bias=0.05,
        isvalid_function=agent.is_new_node_valid, cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        udf_seed=seed, print_logs=False)
    t0 = time.time(); p.plan_path(); dt = time.time() - t0
    return p.path_found, dt, p.path_cost


def run_rrt_eb(env, seed):
    p = RRT(start=start, goal=goal_2d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        num_extension_trials=40, goal_bias=0.05,
        isvalid_function=agent.is_new_node_valid, cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb_190, eb_kd_tree=kd_tree_190,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        udf_seed=seed, print_logs=False)
    t0 = time.time(); p.plan_path(); dt = time.time() - t0
    return p.path_found, dt, p.path_cost


def run_sst_mc(env, seed):
    p = SST(start=start, goal=goal_2d, goal_radius=goal_radius,
        env=env, agent=agent,
        use_fixed_sampling_time=True,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        num_extension_trials=40, goal_bias=0.05,
        isvalid_function=agent.is_new_node_valid, cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        udf_seed=seed, print_logs=False, best_near_radius=5.0)
    t0 = time.time(); p.plan_path(); dt = time.time() - t0
    return p.path_found, dt, p.path_cost


def run_sst_eb(env, seed):
    p = SST(start=start, goal=goal_2d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        num_extension_trials=40, goal_bias=0.05,
        isvalid_function=agent.is_new_node_valid, cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb_190, eb_kd_tree=kd_tree_190,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        udf_seed=seed, print_logs=False, best_near_radius=5.0)
    t0 = time.time(); p.plan_path(); dt = time.time() - t0
    return p.path_found, dt, p.path_cost


def run_gbrrt_mc(env, seed):
    p = GBRRT(start=start, goal=goal_3d, goal_radius=goal_radius,
        env=env, agent=agent,
        use_fixed_sampling_time=True,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid, cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        num_extension_trials=40,
        heuristic_radius=7.0, gamma=14.0, exploitation_ratio=0.8,
        udf_seed=seed, print_logs=False)
    t0 = time.time(); p.plan_path(); dt = time.time() - t0
    return p.path_found, dt, p.path_cost


def run_gbrrt_eb(env, seed):
    p = GBRRT(start=start, goal=goal_3d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid, cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        num_extension_trials=40,
        heuristic_radius=7.0, gamma=14.0, exploitation_ratio=0.8,
        edge_bundle=eb_190, eb_kd_tree=kd_tree_190,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        udf_seed=seed, print_logs=False)
    t0 = time.time(); p.plan_path(); dt = time.time() - t0
    return p.path_found, dt, p.path_cost


def run_dbrrt(env, seed):
    p = DbRRT(start=start, goal=goal_2d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid, cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb_190, eb_kd_tree=kd_tree_190,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        max_num_edges_per_node=1000,
        delta_disc=1.0, goal_bias=0.05,
        udf_seed=seed, print_logs=False)
    t0 = time.time(); p.plan_path(); dt = time.time() - t0
    return p.path_found, dt, p.path_cost


def run_dbrrtc(env, seed):
    p = DbRRTC(start=start, goal=goal_3d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid, cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb_190, eb_kd_tree=kd_tree_190,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        max_num_edges_per_node=1000,
        delta_disc=1.0, goal_bias=0.05,
        udf_seed=seed, print_logs=False)
    t0 = time.time(); p.plan_path(); dt = time.time() - t0
    return p.path_found, dt, p.path_cost


def run_bboe(env, seed):
    p = BBoE(start=start, goal=goal_3d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid, cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb_190, eb_kd_tree=kd_tree_190,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        exploitation_ratio=0.8, heuristic_radius=7.0, gamma=14.0,
        k_bias=0.85, num_skip_edges=5,
        udf_seed=seed, print_logs=False)
    t0 = time.time(); p.plan_path(); dt = time.time() - t0
    return p.path_found, dt, p.path_cost


def run_boe(env, seed):
    p = BoE(start=start, goal=goal_2d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=100000, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid, cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb_190, eb_kd_tree=kd_tree_190,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        max_num_edges_per_node=190,
        goal_select_prob=0.05, min_spacing=0.5,
        udf_seed=seed, print_logs=False)
    t0 = time.time(); p.plan_path(); dt = time.time() - t0
    return p.path_found, dt, p.path_cost


_ALL_PLANNERS = [
    ("RRT-MC",   run_rrt_mc),
    ("RRT-EB",   run_rrt_eb),
    ("SST-MC",   run_sst_mc),
    ("SST-EB",   run_sst_eb),
    ("GBRRT-MC", run_gbrrt_mc),
    ("GBRRT-EB", run_gbrrt_eb),
    ("BBoE",     run_bboe),
    ("Db-RRT-F",   run_dbrrt),
    ("Db-RRT-C",   run_dbrrtc),
    ("BoE",      run_boe),
]
if planners_filter is not None:
    _kept = [(n, f) for (n, f) in _ALL_PLANNERS if n.lower() in planners_filter]
    if _kept:
        _ALL_PLANNERS = _kept
    print(f"Filtered planners: {[n for n,_ in _ALL_PLANNERS]}")
PLANNER_NAMES = [n for n, _ in _ALL_PLANNERS]
PLANNER_FUNCS = [f for _, f in _ALL_PLANNERS]


# ============================================================
# Main loop — curated envs
# ============================================================
all_results = {}

for diff_name, diff_cfg in DIFFICULTIES.items():
    print(f"\n{'='*70}")
    print(f"  DIFFICULTY: {diff_name}")
    print(f"  Circular: {diff_cfg['n_circ']}  |  Rectangular: {diff_cfg['n_rect']}")
    print(f"{'='*70}")

    stats = {p: {"planning_time": [], "path_cost": [], "success": 0}
             for p in PLANNER_NAMES}

    env_seeds = CURATED_ENV_SEEDS[diff_name]

    for trial_idx in range(50):
        env_seed = env_seeds[trial_idx % 10]   # cycle through 10 envs
        udf_seed = int(TRIAL_SEEDS[trial_idx])

        env_rng = np.random.default_rng(env_seed)
        env = generate_random_environment(diff_cfg, env_rng)

        n_obs = len(env.obstacles)
        print(f"\n  Trial {trial_idx+1}/50 | env_seed={env_seed} | "
              f"udf_seed={udf_seed} | Obstacles: {n_obs}")

        for p_name, p_func in zip(PLANNER_NAMES, PLANNER_FUNCS):
            found, elapsed, cost = p_func(env, udf_seed)
            if found:
                stats[p_name]["success"] += 1
                stats[p_name]["planning_time"].append(elapsed)
                stats[p_name]["path_cost"].append(cost)
            status = "✅" if found else "❌"
            print(f"    {p_name:>14s}: {status} | {elapsed:.2f}s | "
                  f"Cost: {cost if found else 'inf':<8}", flush=True)

    all_results[diff_name] = stats


# ============================================================
# Summary
# ============================================================
print(f"\n\n{'='*90}")
print(f"{'SUMMARY (CURATED ENVS)':^90s}")
print(f"{'='*90}")
header = f"{'Planner':>14s}"
for d in DIFFICULTIES:
    header += f" | {d:^20s}"
print(header)
print("-" * 90)
for p in PLANNER_NAMES:
    row = f"{p:>14s}"
    for d in DIFFICULTIES:
        s = all_results[d][p]
        rate = s["success"] / 50 * 100
        row += f" | {rate:5.0f}% ({s['success']:>2d}/50)        "
    print(row)
print()
print(f"{'':>14s}", end="")
for d in DIFFICULTIES:
    print(f" | {'Avg Time / Avg Cost':^20s}", end="")
print()
print("-" * 90)
for p in PLANNER_NAMES:
    row = f"{p:>14s}"
    for d in DIFFICULTIES:
        s = all_results[d][p]
        if s["success"]:
            t = np.mean(s["planning_time"])
            c = np.mean(s["path_cost"])
            row += f" | {t:5.2f}s / {c:7.1f}    "
        else:
            row += f" | {'  --   /    --':^20s}"
    print(row)
print(f"\n{'='*90}")
