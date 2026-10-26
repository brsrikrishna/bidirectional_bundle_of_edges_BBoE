"""Adversarial unicycle environment for the RA-L rebuttal (R2: failure cases).

Obstacle inflation (SquareEnvironment's 1.25 m buffer) is DISABLED here: with it, every inside corner
becomes a 1.45 m fillet that gives a 1 m library edge room to turn at the slot junctions.

A full-width barrier with an offset "keyhole": a vertical slot through the top
plate, a thin chamber, and a second vertical slot through the bottom plate
shifted laterally by D.  Passing requires a lateral shift of D inside a
chamber whose free band (after the environment's 1.25 m obstacle buffer and
the 0.2 m agent radius) is only BAND tall.  The 190-edge GBRRT maneuver
library has no pivot-in-place (v=0 absent), every edge lasts exactly 1 s
(>= 0.72 m of travel) and its minimum turning radius is 0.64 m, so no edge
sequence can make the shift.  A continuous-control unicycle can (near-zero v
with max omega), which we verify with a hand-built control sequence through
the same collision checker, and with a continuous-control Monte-Carlo RRT.

Outputs: benchmark_logs/adversarial_keyhole.log, media/adversarial_keyhole.png
"""
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]   # repository root
import time
sys.path.append(str(ROOT / 'src'))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from Environments import SquareEnvironment, RectangleObstacle2D
from mapf_env_square_agent_unicycle import get_unicycle_agent
from edge_bundle import EdgeBundle
from kd_tree_unicycle import CircularAngleIndexNumba
from unicycle_discrete_actions import apply_discrete_random_action
from bboe import BBoE
from rrt import RRT
from numba.typed import List
from numba import types
DYN = List.empty_list(types.Array(types.float64, 2, 'C'))

# ---------------- geometry ----------------
ENV = 100.0
start = np.array([30.0, 90.0, 0.0]); goal = np.array([79.0, 5.0, 0.0]); goal_radius = 2.0
R_EFF = 0.2                         # agent radius; obstacle buffer disabled for this env (obs_buffers=False) so inside corners stay sharp
BAND = 0.12                         # free centreline band: 1 m library edges must stay within atan(0.12/1)=6.8 deg of the axis
W = 2 * R_EFF + BAND                # slot width / chamber height = 3.3
PLATE = 3.0                         # plate thickness (heading tolerance atan(BAND/PLATE) = 7.6 deg)
D = 4.0                             # lateral jog between the two slots
YC = 50.0                           # chamber centre y
X1 = 46.0                           # left edge of slot 1 ; slot 2 is [X1+D, X1+D+W]
y_top_c = YC + W / 2 + PLATE / 2; y_bot_c = YC - W / 2 - PLATE / 2

def rect(x0, x1, y0, y1):  # RectangleObstacle2D takes centre + size
    return RectangleObstacle2D((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
obstacles = [
    rect(-5, X1, YC + W/2, YC + W/2 + PLATE),                 # top plate, left of slot 1
    rect(X1 + W, ENV + 5, YC + W/2, YC + W/2 + PLATE),        # top plate, right of slot 1
    rect(-5, X1 + D, YC - W/2 - PLATE, YC - W/2),             # bottom plate, left of slot 2
    rect(X1 + D + W, ENV + 5, YC - W/2 - PLATE, YC - W/2),    # bottom plate, right of slot 2
    rect(-5, X1, YC - W/2, YC + W/2),                         # chamber end, left
    rect(X1 + D + W, ENV + 5, YC - W/2, YC + W/2),            # chamber end, right
]
env = SquareEnvironment(ENV, ENV, obstacles, obs_buffers=False)
agent = get_unicycle_agent(1)            # used for BBoE (patched to discrete actions below)
agent_cont = get_unicycle_agent(2)       # continuous-control agent for the reference RRT
T_max = 1.0
data = np.load(str(ROOT / 'edge_bundles/eb_unicycle_discrete_190.npz'))
eb = EdgeBundle(data, use_all_edges=True)
kd = CircularAngleIndexNumba(eb.start_states[:, 2], ids=np.arange(eb.num_edges, dtype=np.int64))
apply_discrete_random_action(agent)

# ---------------- analytic feasibility check (continuous unicycle) ----------------
def simulate(controls, x0):
    """controls: list of (v, omega, duration). Returns full path (all recorded steps) and validity."""
    x = x0.copy(); pts = []; ok = True
    for v, w, dt in controls:
        n = max(1, int(round(dt / 0.1)))
        xn, path = agent_cont.get_next_state(x, np.array([v, w]), dt, num_steps=n)
        valid = agent_cont.is_new_node_valid(path, agent_cont.radius, env.size,
                                             env.static_circular_obstacles, env.static_rectangular_obstacles,
                                             DYN,
                                             env.obstacle_buffer, env.boundary_buffer, 0.0, dt, 0.1)
        ok = ok and bool(valid); pts.append(path); x = xn
    return np.vstack(pts), ok

xs1 = X1 + W / 2; xs2 = X1 + D + W / 2   # slot centre lines
# start (30,90,0): turn to face slot-1 column, drive to above slot 1, face down, descend, pivot, shift, pivot, descend, go to goal
def go(x, y, heading_deg, x_t, y_t):
    """rotate in place (v~0) to face target, drive straight. returns controls and new pose"""
    h = np.deg2rad(heading_deg); tgt = np.arctan2(y_t - y, x_t - x)
    dh = (tgt - h + np.pi) % (2*np.pi) - np.pi
    c = []
    if abs(dh) > 1e-6:
        c.append((0.01, np.sign(dh) * np.pi/2, abs(dh) / (np.pi/2)))   # pivot at v=0.01 (radius 6 mm)
    dist = np.hypot(x_t - x, y_t - y)
    c.append((5.0, 0.0, dist / 5.0))
    return c, np.rad2deg(tgt)
# (build explicitly to keep poses exact)
pose = np.array([30.0, 90.0, 0.0]); ref = []
for (xt, yt) in [(xs1, 90.0), (xs1, YC), (xs2, YC), (xs2, YC - W/2 - PLATE - 3), (goal[0], goal[1])]:
    c, hd = go(pose[0], pose[1], np.rad2deg(pose[2]), xt, yt)
    ref += c; pose = np.array([xt, yt, np.deg2rad(hd)])
ref_path, ref_ok = simulate(ref, start)
print(f"Continuous-control reference trajectory valid: {ref_ok}  (segments={len(ref)}, samples={len(ref_path)})")

# ---------------- planners ----------------
def make_bboe(seed, budget):
    return BBoE(start=start, goal=goal, goal_radius=goal_radius, env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1, max_iter=100000, planning_time=budget,
        isvalid_function=agent.is_new_node_valid, cost_function=agent.get_cost,
        random_point_function=agent.get_random_point, reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb, eb_kd_tree=kd, get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi, translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        exploitation_ratio=0.8, heuristic_radius=7.0, gamma=14.0, k_bias=0.85, num_skip_edges=5,
        udf_seed=seed, print_logs=False)

def make_rrt_cont(seed, budget):
    return RRT(start=start, goal=goal[:2], goal_radius=goal_radius, env=env, agent=agent_cont,
        use_fixed_sampling_time=False, sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=1000000, planning_time=budget, num_extension_trials=40, goal_bias=0.05,
        isvalid_function=agent_cont.is_new_node_valid, cost_function=agent_cont.get_cost,
        random_point_function=agent_cont.get_random_point, reached_goal_function=agent_cont.agent_reached_goal,
        udf_seed=seed, print_logs=False)

def run(label, ctor, seeds, budget):
    succ = 0; times = []; last = None
    for s in seeds:
        p = ctor(s, budget); t = time.time(); p.plan_path(); dt = time.time() - t
        if p.path_found: succ += 1; times.append(dt)
        last = p
    msg = f"{label:34s} budget={budget:>5.1f}s  success={succ}/{len(seeds)}" + (f"  mean_time={np.mean(times):.2f}s" if times else "")
    print(msg, flush=True); return msg, last

seeds = list(np.random.default_rng(12345).integers(0, 100000, 50))
lines = [f"Adversarial keyhole: BAND={BAND} m, slot/chamber W={W} m, plate={PLATE} m, jog D={D} m, R_eff={R_EFF} m",
         f"Continuous-control reference trajectory valid: {ref_ok}"]
# warm-up JIT
make_bboe(1, 0.3).plan_path(); make_rrt_cont(1, 0.3).plan_path()
m, bboe_last = run("BBoE (190-edge library)", make_bboe, seeds, 5.0); lines.append(m)
m, _ = run("BBoE (190-edge library)", make_bboe, seeds[:10], 30.0); lines.append(m)
m, rrt_last = run("RRT-MC continuous controls", make_rrt_cont, seeds, 5.0); lines.append(m)
open(str(ROOT / 'benchmark_logs/adversarial_keyhole.log'), 'w').write("\n".join(lines) + "\n")

# ---------------- figure ----------------
fig, ax = plt.subplots(figsize=(5, 5))
for o in obstacles:
    ax.add_patch(Rectangle((o.x - o.w/2, o.y - o.h/2), o.w, o.h, color='0.35'))
# BBoE forward tree (blue) from last run
for nid in bboe_last.tree_for.nodes:
    n = bboe_last.tree_for.nodes[nid]['value']
    if n.path_from_parent is not None: ax.plot(n.path_from_parent[:, 0], n.path_from_parent[:, 1], color='steelblue', lw=0.4, alpha=0.6)
for nid in bboe_last.tree_rev.nodes:
    n = bboe_last.tree_rev.nodes[nid]['value']
    if n.path_from_parent is not None: ax.plot(n.path_from_parent[:, 0], n.path_from_parent[:, 1], color='tomato', lw=0.4, alpha=0.6)
ax.plot(ref_path[:, 0], ref_path[:, 1], color='lime', lw=1.5, label='continuous-control reference')
ax.plot(*start[:2], 'o', color='#555', ms=6, label='start'); ax.plot(*goal[:2], 'o', color='lime', mec='k', ms=7, label='goal')
ax.set_xlim(0, ENV); ax.set_ylim(0, ENV); ax.set_aspect('equal'); ax.legend(loc='lower left', fontsize=7)
ax.set_title(f"Keyhole: BBoE trees after 5 s (blue fwd / red rev)", fontsize=9)
axins = fig.add_axes([0.62, 0.55, 0.3, 0.3]); axins.set_xlim(X1 - 2, X1 + D + W + 2); axins.set_ylim(YC - W/2 - PLATE - 1, YC + W/2 + PLATE + 1)
for o in obstacles: axins.add_patch(Rectangle((o.x - o.w/2, o.y - o.h/2), o.w, o.h, color='0.35'))
axins.plot(ref_path[:, 0], ref_path[:, 1], color='lime', lw=1.2); axins.set_aspect('equal'); axins.set_xticks([]); axins.set_yticks([])
fig.savefig(str(ROOT / 'media/adversarial_keyhole.png'), dpi=200, bbox_inches='tight')
print("saved figure + log")
