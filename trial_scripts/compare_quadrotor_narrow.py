"""
Quadrotor benchmark of the paper (Table II, Fig. 5; the "quadrotor" column).

A 6D quadrotor (position + velocity, translational double-integrator dynamics)
must cross two walls that each have a narrow hole (at Y = 8-16 and Y = 24-32),
from a start at Y = 35 to a goal at Y = 5 placed at opposite ends of the
Y axis, so every solution threads both holes along a long diagonal path.
Planners: RRT, SST, GBRRT, BBoE, dB-RRT-F, dB-RRT-C, BoE; 50 seeds; 5 s budget.
Usage:  python trial_scripts/compare_quadrotor_narrow.py > benchmark_logs/quadrotor_narrow_50seed_full.log
"""

import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]   # repository root
import time
import os
sys.path.append(str(ROOT / 'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Environments import CuboidEnvironment, CuboidObstacle3D
from Agents import QuadCopter6D
from rrt import RRT
from sst import SST
from gbrrt import GBRRT
from db_rrt import DbRRT
from db_rrt_c import DbRRTC
from BoE import BoE
from bboe import BBoE
from edge_bundle import EdgeBundle
from kd_tree_quadcopter6d import VxyzTree
from mapf_env_cuboid_agent_quadcopter6d import get_quadcopter_agent

# ============================================================
# Environment: 40x40x10 with 2 walls with holes (UNCHANGED)
# ============================================================
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
# 2 walls, but with NARROW holes (4 wide in Y, 3 high in Z) instead of the
# original 8x4 holes. Single-bottleneck regime where random extension is
# unlikely to thread the gap — BBoE's pre-computed bundle should pull ahead.
obstacles += wall_with_hole(12, 4, 10, 14, 3.5, 6.5)    # W1: 4x3 hole, low-Y
obstacles += wall_with_hole(24, 4, 26, 30, 5.5, 8.5)    # W2: 4x3 hole, high-Y

env = CuboidEnvironment(length=40.0, breadth=40.0, height=10.0, obs=obstacles)

# Agent
agent = get_quadcopter_agent(1)

# ============================================================
# Start/Goal (axis-aligned, same as original baseline quadrotor env)
# ============================================================
start = np.array([4.0, 12.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64)
goal = np.array([36.0, 28.0, 7.0], dtype=np.float64)
goal_radius = 2.0
goal_6d = np.array([36.0, 28.0, 7.0, 0.0, 0.0, 0.0], dtype=np.float64)

# ============================================================
# Render env PNG
# ============================================================
def draw_cuboid(ax, cx, cy, cz, lx, ly, lz, color='gray', alpha=0.3):
    x0, x1 = cx - lx/2, cx + lx/2
    y0, y1 = cy - ly/2, cy + ly/2
    z0, z1 = cz - lz/2, cz + lz/2
    verts = [
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
        [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],
        [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],
    ]
    ax.add_collection3d(Poly3DCollection(
        verts, facecolors=color, edgecolors='black',
        linewidths=0.3, alpha=alpha))

def draw_sphere(ax, cx, cy, cz, r, color, alpha=0.3):
    u, v = np.mgrid[0:2*np.pi:24j, 0:np.pi:16j]
    x = cx + r * np.cos(u) * np.sin(v)
    y = cy + r * np.sin(u) * np.sin(v)
    z = cz + r * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
for o in env.obstacles:
    draw_cuboid(ax, o.x, o.y, o.z, o.l, o.w, o.h,
                color='steelblue', alpha=0.55)
ax.scatter([start[0]], [start[1]], [start[2]], color='green', s=140,
           edgecolors='black', linewidths=1.5, label=f'Start ({start[0]:.0f}, {start[1]:.0f}, {start[2]:.0f})', zorder=10)
draw_sphere(ax, goal[0], goal[1], goal[2], goal_radius, color='red', alpha=0.35)
ax.scatter([goal[0]], [goal[1]], [goal[2]], color='red', s=140,
           edgecolors='black', linewidths=1.5, label=f'Goal ({goal[0]:.0f}, {goal[1]:.0f}, {goal[2]:.0f}), r={goal_radius:.1f}', zorder=10)
ax.set_xlim(0, 40); ax.set_ylim(0, 40); ax.set_zlim(0, 10)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Quadrotor env (narrow): 40×40×10, 2 walls with NARROW 4×3 holes')
ax.legend(loc='upper left', fontsize=9)
ax.view_init(elev=22, azim=-65)
out_dir = str(ROOT / 'media/seed50_plots')
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, 'quadrotor_narrow_env.png')
plt.tight_layout()
plt.savefig(out_png, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"Env rendered to: {out_png}\n")

# ============================================================
# Edge Bundle + KD-Tree (velocity-space)
# ============================================================
edge_bundle_file = str(ROOT / 'edge_bundles/eb_quadcopter6d_kinodynamic_TI_edges_200000.npz')
data = np.load(edge_bundle_file)
eb = EdgeBundle(data, fix_num_edges=100000, use_all_edges=False, rng_seed=42)
edge_ids = np.arange(eb.num_edges, dtype=np.int64)
vx = eb.start_states[:, 3]
vy = eb.start_states[:, 4]
vz = eb.start_states[:, 5]
kd_tree_eb = VxyzTree(vx, vy, vz, ids=edge_ids, scales=(1.0, 1.0, 1.0))

# ============================================================
# Benchmark
# ============================================================
T_max = 1.0
PLANNING_TIME = 5.0
planners = ["RRT", "SST", "GBRRT", "BBoE", "Db-RRT-F", "Db-RRT-C", "BoE"]
stats = {p: {"planning_time": [], "path_cost": [], "success": 0}
         for p in planners}
seeds = list(range(50))  # 0..49

print("Warming up JIT...")
WARMUP_BUDGET = 0.3
def _warmup():
    base = dict(start=start, goal=goal_6d, goal_radius=goal_radius,
                env=env, agent=agent,
                sampling_time_step=T_max, minimum_time_step=0.1,
                max_iter=10000, planning_time=WARMUP_BUDGET,
                isvalid_function=agent.is_new_node_valid,
                cost_function=agent.get_cost,
                random_point_function=agent.get_random_point,
                reached_goal_function=agent.agent_reached_goal,
                udf_seed=99, print_logs=False)
    eb_kw = dict(edge_bundle=eb, eb_kd_tree=kd_tree_eb,
                 get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
                 kd_tree_delta_radius=0.1,
                 translate_function=agent.kd_tree_point_translate_function)
    eb_sort_kw = dict(eb_kw, sort_edges_function=agent.sort_kd_tree_edges)
    RRT(**base, use_fixed_sampling_time=False, num_extension_trials=7,
        goal_bias=0.05).plan_path()
    SST(**base, use_fixed_sampling_time=False, num_extension_trials=7,
        goal_bias=0.05, best_near_radius=5.0).plan_path()
    GBRRT(**base, use_fixed_sampling_time=False, num_extension_trials=7,
          heuristic_radius=6.0, gamma=6.0,
          exploitation_ratio=0.95).plan_path()
    BBoE(**base, exploitation_ratio=0.95, heuristic_radius=6.0, gamma=6.0,
         k_bias=0.85, num_skip_edges=1, **eb_sort_kw).plan_path()
    DbRRT(**base, max_num_edges_per_node=1000,
          delta_disc=1.0, goal_bias=0.05, **eb_sort_kw).plan_path()
    DbRRTC(**base, max_num_edges_per_node=1000,
           delta_disc=1.0, goal_bias=0.05, **eb_sort_kw).plan_path()
    BoE(**base, max_num_edges_per_node=1000,
        goal_select_prob=0.05, min_spacing=1.0, **eb_sort_kw).plan_path()
_warmup()
print("Warm-up done.\n")

for idx, s in enumerate(seeds):
    print(f"\n=======================================================")
    print(f"Trial {idx+1}/{len(seeds)} | Seed: {s}")
    print(f"=======================================================")

    rrt = RRT(
        start=start, goal=goal_6d, goal_radius=goal_radius,
        env=env, agent=agent,
        use_fixed_sampling_time=False,
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
    print(f"RRT:      {'✅' if rrt.path_found else '❌'} | Time: {t_elapsed:.2f}s | Cost: {rrt.path_cost:.2f}")

    sst = SST(
        start=start, goal=goal_6d, goal_radius=goal_radius,
        env=env, agent=agent,
        use_fixed_sampling_time=False,
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
    print(f"SST:      {'✅' if sst.path_found else '❌'} | Time: {t_elapsed:.2f}s | Cost: {sst.path_cost:.2f}")

    gbrrt = GBRRT(
        start=start, goal=goal_6d, goal_radius=goal_radius,
        env=env, agent=agent,
        use_fixed_sampling_time=False,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=100000, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        num_extension_trials=7,
        heuristic_radius=6.0,      # FAIR-TUNE: match BBoE's bumped value
        gamma=6.0,
        exploitation_ratio=0.95,   # FAIR-TUNE: match BBoE's bumped value
        udf_seed=s, print_logs=False
    )
    t = time.time(); gbrrt.plan_path(); t_elapsed = time.time() - t
    if gbrrt.path_found:
        stats["GBRRT"]["success"] += 1
        stats["GBRRT"]["planning_time"].append(t_elapsed)
        stats["GBRRT"]["path_cost"].append(gbrrt.path_cost)
    print(f"GBRRT:    {'✅' if gbrrt.path_found else '❌'} | Time: {t_elapsed:.2f}s | Cost: {gbrrt.path_cost:.2f}")

    bboe = BBoE(
        start=start, goal=goal_6d, goal_radius=goal_radius,
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
        kd_tree_delta_radius=0.05,  # TUNE: 0.1 -> 0.05 (tighter v-match, ~8x cheaper/call)
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        exploitation_ratio=0.95,   # TUNE: 0.8 -> 0.95 (lean harder on bundle)
        heuristic_radius=6.0,      # TUNE: 3.0 -> 6.0 (wider fwd-rev connection)
        gamma=6.0,
        k_bias=0.85,
        num_skip_edges=1,          # TUNE: 5 -> 1 (consider every sorted edge)
        udf_seed=s, print_logs=False
    )
    t = time.time(); bboe.plan_path(); t_elapsed = time.time() - t
    if bboe.path_found:
        stats["BBoE"]["success"] += 1
        stats["BBoE"]["planning_time"].append(t_elapsed)
        stats["BBoE"]["path_cost"].append(bboe.path_cost)
    print(f"BBoE:     {'✅' if bboe.path_found else '❌'} | Time: {t_elapsed:.2f}s | Cost: {bboe.path_cost:.2f}")

    db = DbRRT(
        start=start, goal=goal_6d, goal_radius=goal_radius,
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
        delta_disc=1.0, goal_bias=0.05,
        udf_seed=s, print_logs=False,
    )
    t = time.time(); db.plan_path(); t_elapsed = time.time() - t
    if db.path_found:
        stats["Db-RRT-F"]["success"] += 1
        stats["Db-RRT-F"]["planning_time"].append(t_elapsed)
        stats["Db-RRT-F"]["path_cost"].append(db.path_cost)
    print(f"Db-RRT-F: {'✅' if db.path_found else '❌'} | Time: {t_elapsed:.2f}s | Cost: {db.path_cost:.2f}")

    dbc = DbRRTC(
        start=start, goal=goal_6d, goal_radius=goal_radius,
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
        delta_disc=1.0, goal_bias=0.05,
        udf_seed=s, print_logs=False,
    )
    t = time.time(); dbc.plan_path(); t_elapsed = time.time() - t
    if dbc.path_found:
        stats["Db-RRT-C"]["success"] += 1
        stats["Db-RRT-C"]["planning_time"].append(t_elapsed)
        stats["Db-RRT-C"]["path_cost"].append(dbc.path_cost)
    print(f"Db-RRT-C: {'✅' if dbc.path_found else '❌'} | Time: {t_elapsed:.2f}s | Cost: {dbc.path_cost:.2f}")

    boe = BoE(
        start=start, goal=goal_6d, goal_radius=goal_radius,
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
    print(f"BoE:      {'✅' if boe.path_found else '❌'} | Time: {t_elapsed:.2f}s | Cost: {boe.path_cost:.2f}")

print("\n\n=======================================================")
print("BENCHMARK RESULTS (Averages over successful runs)")
print(f"Start={tuple(start[:3])}  Goal={tuple(goal_6d[:3])}  goal_radius={goal_radius}")
print(f"Planning budget per planner per seed: {PLANNING_TIME:.1f}s, seeds={len(seeds)}")
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
