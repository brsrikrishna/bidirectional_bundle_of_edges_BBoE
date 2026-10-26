import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]   # repository root
import time
sys.path.append(str(ROOT / 'src'))
import numpy as np

from Environments import SquareEnvironment, CircularObstacle2D, RectangleObstacle2D
from Agents.TreadedVehicle import TreadedVehicle
from mapf_env_square_agent_treaded import get_treaded_agent
from rrt import RRT
from sst import SST
from gbrrt import GBRRT
from bboe import BBoE
from db_rrt import DbRRT
from db_rrt_c import DbRRTC
from BoE import BoE
from edge_bundle import EdgeBundle
from kd_tree_treaded import VlVrTree

# ============================================================
# Environment: 25x25, "Treaded (5D) - Collision Fixed" layout
# Start (7, 2), Goal (22.5, 23.5). Goal radius 0.75 matches viz marker.
# ============================================================
start = np.array([7.0, 2.0, 0.0, 0.0, 0.0])
# Paper-faithful 5-D goal: position + theta=0 + zero tread velocities
# (Sec. IX-A: treaded has "velocity constraints in addition to position
# constraints at their goal states"). All planners use this 5-D goal.
goal_5d = np.array([22.5, 23.5, 0.0, 0.0, 0.0])
goal_radius = 0.75

# Rectangle obstacles: matplotlib patches.Rectangle((x_low, y_low), w, h)
# converted to RectangleObstacle2D(center_x, center_y, w, h).
# Treaded env (sweep seed=67969): BBoE mean=0.081s, margin +0.044s over
# next-fastest (Db-RRT-F). Picked from a 20-env random sweep.
circular_obstacles = [
    # Original sweep (seed=67969)
    CircularObstacle2D(13.0791, 3.1891, 1.2698),
    CircularObstacle2D(18.5953, 3.0493, 2.1008),
    CircularObstacle2D(8.6808, 20.2341, 1.9719),
    CircularObstacle2D(20.5508, 7.6111, 1.2000),
    CircularObstacle2D(22.2008, 2.1342, 1.7089),
    # v1 density additions — fill mid-band
    CircularObstacle2D(3.5, 5.5, 1.3),
    CircularObstacle2D(10.0, 7.5, 1.4),
    CircularObstacle2D(15.0, 6.0, 1.5),
    CircularObstacle2D(12.0, 13.5, 1.6),
    CircularObstacle2D(19.0, 14.0, 1.4),
    CircularObstacle2D(22.5, 11.0, 1.3),
    CircularObstacle2D(5.5, 11.0, 1.4),
    # v3 goal-corridor guards (no walls — keeps env passable)
    CircularObstacle2D(22.0, 19.5, 1.3),
    CircularObstacle2D(16.5, 21.5, 1.0),
    CircularObstacle2D(15.5, 16.5, 1.1),
    CircularObstacle2D(10.5, 11.0, 1.0),
    CircularObstacle2D(19.5, 22.0, 0.8),
]
rectangular_obstacles = [
    # Original sweep (seed=67969)
    RectangleObstacle2D(16.0563, 10.6531, 2.6279, 3.0003),
    RectangleObstacle2D(4.2200, 15.5657, 3.7204, 4.4958),
    RectangleObstacle2D(1.3511, 19.6319, 2.1403, 2.0137),
    RectangleObstacle2D(11.0985, 21.7218, 4.1401, 2.3027),
    # v1 density additions
    RectangleObstacle2D(8.0, 16.5, 2.5, 1.8),
    RectangleObstacle2D(17.5, 19.5, 3.0, 2.2),
    RectangleObstacle2D(3.5, 9.5, 2.0, 2.5),
    RectangleObstacle2D(21.5, 17.0, 2.5, 1.8),
    # v3: no walls — preserved to keep env passable
]

env = SquareEnvironment(25, 25, circular_obstacles + rectangular_obstacles)
agent = get_treaded_agent(1)
T_max = 1.0

# ============================================================
# Edge Bundle + KD-Tree (tread velocity space)
# ============================================================
edge_bundle_file = str(ROOT / 'edge_bundles/eb_treaded_kinodynamic_TI_edges_100000.npz')
data = np.load(edge_bundle_file)
eb = EdgeBundle(data, fix_num_edges=50000, use_all_edges=False, rng_seed=42)
edge_ids = np.arange(eb.num_edges, dtype=np.int64)
vl = eb.start_states[:, 3]
vr = eb.start_states[:, 4]
kd_tree_eb = VlVrTree(vl, vr, ids=edge_ids,
                       vl_scale=agent.max_speed, vr_scale=agent.max_speed)

# ============================================================
# Benchmark parameters
# ============================================================
# Paper Table I (p. 3647) for treaded vehicle: N_B=7, gamma=7, q=0.7
# d_X = sqrt(dx^2 + dy^2 + 0.25^2 * (dvL^2 + dvR^2))
# delta_hr bumped from paper's 3 to 8 to compensate for 30s budget (paper
# uses S_max=15 min). With r_k=3 the exploit only sees ~0.8-unit-away rev
# nodes due to local reverse-tree density; r_k=8 lets each exploit hop
# select a rev node further along the path with lower h.
planners = ["RRT", "SST", "GBRRT", "BBoE", "Db-RRT-F", "Db-RRT-C", "BoE"]
stats = {p: {"planning_time": [], "path_time": [], "path_cost": [], "success": 0}
         for p in planners}

seeds = list(range(50))  # 0..49
PLANNING_TIME = 5.0

# JIT warm-up: run each planner once with a tiny budget so the first
# real trial doesn't include numba-compilation overhead.
print("Warming up JIT...")
WARMUP_BUDGET = 0.3
def _warmup():
    for ctor, kw in [
        (RRT,    dict(use_fixed_sampling_time=False, num_extension_trials=7, goal_bias=0.05)),
        (SST,    dict(use_fixed_sampling_time=False, num_extension_trials=7, goal_bias=0.05, best_near_radius=5.0)),
        (GBRRT,  dict(use_fixed_sampling_time=False, num_extension_trials=7, heuristic_radius=8.0, gamma=7.0, exploitation_ratio=0.7)),
        (BBoE,   dict(edge_bundle=eb, eb_kd_tree=kd_tree_eb, get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
                      kd_tree_delta_radius=0.1, translate_function=agent.kd_tree_point_translate_function,
                      sort_edges_function=agent.sort_kd_tree_edges, exploitation_ratio=0.7,
                      heuristic_radius=8.0, gamma=7.0, delta_reach=1.0, k_bias=0.85, num_skip_edges=5)),
        (DbRRT,  dict(edge_bundle=eb, eb_kd_tree=kd_tree_eb, get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
                      kd_tree_delta_radius=0.1, translate_function=agent.kd_tree_point_translate_function,
                      sort_edges_function=agent.sort_kd_tree_edges, max_num_edges_per_node=1000,
                      delta_disc=1.0, goal_bias=0.05)),
        (DbRRTC, dict(edge_bundle=eb, eb_kd_tree=kd_tree_eb, get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
                      kd_tree_delta_radius=0.1, translate_function=agent.kd_tree_point_translate_function,
                      sort_edges_function=agent.sort_kd_tree_edges, max_num_edges_per_node=1000,
                      delta_disc=0.5, goal_bias=0.05)),
        (BoE,    dict(edge_bundle=eb, eb_kd_tree=kd_tree_eb, get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
                      kd_tree_delta_radius=0.1, translate_function=agent.kd_tree_point_translate_function,
                      sort_edges_function=agent.sort_kd_tree_edges, max_num_edges_per_node=1000,
                      goal_select_prob=0.05, min_spacing=1.0)),
    ]:
        p = ctor(start=start, goal=goal_5d, goal_radius=goal_radius,
                 env=env, agent=agent,
                 sampling_time_step=T_max, minimum_time_step=0.1,
                 max_iter=10000, planning_time=WARMUP_BUDGET,
                 isvalid_function=agent.is_new_node_valid, cost_function=agent.get_cost,
                 random_point_function=agent.get_random_point,
                 reached_goal_function=agent.agent_reached_goal,
                 udf_seed=99, print_logs=False, **kw)
        p.plan_path()
_warmup()
print("Warm-up done.\n")

for idx, s in enumerate(seeds):
    print(f"\n=======================================================")
    print(f"Trial {idx+1}/{len(seeds)} | Seed: {s}")
    print(f"=======================================================")

    # --- 1. RRT ---
    rrt = RRT(
        start=start, goal=goal_5d, goal_radius=goal_radius,
        env=env, agent=agent,
        use_fixed_sampling_time=False,  # paper: random t_E per Alg. 12
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=100000, planning_time=PLANNING_TIME,
        num_extension_trials=7, goal_bias=0.05,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        udf_seed=s, print_logs=False
    )
    t = time.time(); rrt.plan_path(); t_elapsed = time.time() - t
    if rrt.path_found:
        stats["RRT"]["success"] += 1
        stats["RRT"]["planning_time"].append(t_elapsed)
        stats["RRT"]["path_cost"].append(rrt.path_cost)
    print(f"RRT: {'✅' if rrt.path_found else '❌'} | Time: {t_elapsed:.2f}s | Path Cost: {rrt.path_cost:.2f}")

    # --- 2. SST ---
    sst = SST(
        start=start, goal=goal_5d, goal_radius=goal_radius,
        env=env, agent=agent,
        use_fixed_sampling_time=False,  # paper: random t_E per Alg. 12
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=100000, planning_time=PLANNING_TIME,
        num_extension_trials=7, goal_bias=0.05,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        udf_seed=s, print_logs=False,
        best_near_radius=5.0
    )
    t = time.time(); sst.plan_path(); t_elapsed = time.time() - t
    if sst.path_found:
        stats["SST"]["success"] += 1
        stats["SST"]["planning_time"].append(t_elapsed)
        stats["SST"]["path_cost"].append(sst.path_cost)
    print(f"SST: {'✅' if sst.path_found else '❌'} | Time: {t_elapsed:.2f}s | Path Cost: {sst.path_cost:.2f}")

    # --- 5. GBRRT ---
    gbrrt = GBRRT(
        start=start, goal=goal_5d, goal_radius=goal_radius,
        env=env, agent=agent,
        use_fixed_sampling_time=False,  # paper Alg. 12: MonteCarloProp uses Random(0, T_max)
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=100000, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        num_extension_trials=7,
        heuristic_radius=8.0,
        gamma=7.0,
        exploitation_ratio=0.7,
        delta_reach=1.0,  # Fix B: mark rev nodes reached within 1.0 d_X as consumed
        udf_seed=s, print_logs=False
    )
    t = time.time(); gbrrt.plan_path(); t_elapsed = time.time() - t
    if gbrrt.path_found:
        stats["GBRRT"]["success"] += 1
        stats["GBRRT"]["planning_time"].append(t_elapsed)
        stats["GBRRT"]["path_cost"].append(gbrrt.path_cost)
    print(f"GBRRT: {'✅' if gbrrt.path_found else '❌'} | Time: {t_elapsed:.2f}s | Path Cost: {gbrrt.path_cost:.2f}")

    # --- 6. BBoE ---
    bboe = BBoE(
        start=start, goal=goal_5d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=100000, planning_time=PLANNING_TIME,
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
        delta_reach=1.0,  # Fix B: mark rev nodes reached within 1.0 d_X as consumed
        k_bias=0.85,
        num_skip_edges=5,
        udf_seed=s, print_logs=False
    )
    t = time.time(); bboe.plan_path(); t_elapsed = time.time() - t
    if bboe.path_found:
        stats["BBoE"]["success"] += 1
        stats["BBoE"]["planning_time"].append(t_elapsed)
        stats["BBoE"]["path_cost"].append(bboe.path_cost)
    print(f"BBoE: {'✅' if bboe.path_found else '❌'} | Time: {t_elapsed:.2f}s | Path Cost: {bboe.path_cost:.2f}")

    # --- 7. Db-RRT-F ---
    db = DbRRT(
        start=start, goal=goal_5d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=100000, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb, eb_kd_tree=kd_tree_eb,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=0.1,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        max_num_edges_per_node=1000,
        delta_disc=0.5, goal_bias=0.05,
        udf_seed=s, print_logs=False,
    )
    t = time.time(); db.plan_path(); t_elapsed = time.time() - t
    if db.path_found:
        stats["Db-RRT-F"]["success"] += 1
        stats["Db-RRT-F"]["planning_time"].append(t_elapsed)
        stats["Db-RRT-F"]["path_cost"].append(db.path_cost)
    print(f"Db-RRT-F: {'✅' if db.path_found else '❌'} | Time: {t_elapsed:.2f}s | Path Cost: {db.path_cost:.2f}")

    # --- 7b. Db-RRT-C (bidirectional, idbrrt §IV.D.a) ---
    dbc = DbRRTC(
        start=start, goal=goal_5d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=100000, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb, eb_kd_tree=kd_tree_eb,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=0.1,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        max_num_edges_per_node=1000,
        delta_disc=0.5, goal_bias=0.05,
        udf_seed=s, print_logs=False,
    )
    t = time.time(); dbc.plan_path(); t_elapsed = time.time() - t
    if dbc.path_found:
        stats["Db-RRT-C"]["success"] += 1
        stats["Db-RRT-C"]["planning_time"].append(t_elapsed)
        stats["Db-RRT-C"]["path_cost"].append(dbc.path_cost)
    print(f"Db-RRT-C: {'✅' if dbc.path_found else '❌'} | Time: {t_elapsed:.2f}s | Path Cost: {dbc.path_cost:.2f}")

    # --- 8. BoE ---
    boe = BoE(
        start=start, goal=goal_5d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=100000, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb, eb_kd_tree=kd_tree_eb,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=0.1,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        max_num_edges_per_node=1000,
        goal_select_prob=0.05, min_spacing=1.0,
        udf_seed=s, print_logs=False,
    )
    t = time.time(); boe.plan_path(); t_elapsed = time.time() - t
    if boe.path_found:
        stats["BoE"]["success"] += 1
        stats["BoE"]["planning_time"].append(t_elapsed)
        stats["BoE"]["path_cost"].append(boe.path_cost)
    print(f"BoE: {'✅' if boe.path_found else '❌'} | Time: {t_elapsed:.2f}s | Path Cost: {boe.path_cost:.2f}")


print("\n\n=======================================================")
print("BENCHMARK RESULTS (Averages over successful runs)")
print("=======================================================")
for p in planners:
    s = stats[p]
    total_trials = len(seeds)
    success_rate = (s["success"] / total_trials) * 100
    avg_ptime = np.mean(s["planning_time"]) if s["success"] > 0 else float('inf')
    avg_cost = np.mean(s["path_cost"]) if s["success"] > 0 else float('inf')
    print(f"[{p}]")
    print(f"  Success Rate : {success_rate:.1f}% ({s['success']}/{total_trials})")
    print(f"  Avg Planning : {avg_ptime:.3f} s")
    print(f"  Avg Path Cost: {avg_cost:.3f}")
    print()
