"""BBoE ablations on the curated unicycle benchmark (4 difficulties × 50 seeds).

Ablations
=========
1. num_skip_edges ∈ {1 (none), 2, 5, 8, 10}
2. k_bias ∈ {0.75, 0.85, 0.95}
3. UnidirectionalBBoE — forward tree only, 5 % goal bias, biased/random skipping

Outputs (under media/seed50_plots/):
    ablation_skip_edges_table.txt
    ablation_kbias_time.png
    ablation_kbias_success.png
    ablation_kbias_cost.png
    ablation_unidirectional_table.txt
"""
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]   # repository root
import time
from pathlib import Path

sys.path.append(str(ROOT / 'src'))
import numpy as np
import matplotlib.pyplot as plt

from Environments import SquareEnvironment, CircularObstacle2D, RectangleObstacle2D
from mapf_env_square_agent_unicycle import get_unicycle_agent
from edge_bundle import EdgeBundle
from kd_tree_unicycle import CircularAngleIndexNumba
from unicycle_discrete_actions import apply_discrete_random_action
from bboe import BBoE

OUT_DIR = ROOT / "media/seed50_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLANNING_TIME = 5.0
MAX_ITER = 10000

# === Mirror compare_difficulty_curated.py exactly ===
agent = get_unicycle_agent(1)
T_max = 1.0
data_190 = np.load(str(ROOT / 'edge_bundles/eb_unicycle_discrete_190.npz'))
eb_190 = EdgeBundle(data_190, use_all_edges=True)
ids_190 = np.arange(eb_190.num_edges, dtype=np.int64)
kd_tree_190 = CircularAngleIndexNumba(eb_190.start_states[:, 2], ids=ids_190)
apply_discrete_random_action(agent)

start = np.array([30.0, 90.0, 0.0])
goal_2d = np.array([79.0, 5.0])
goal_3d = np.array([79.0, 5.0, 0.0])
goal_radius = 2.0

DIFFICULTIES = {
    "Easy": {"n_circ": 4, "n_rect": 3, "circ_r_range": (3, 5), "rect_w_range": (5, 8)},
    "Medium": {"n_circ": 22, "n_rect": 18, "circ_r_range": (4, 7), "rect_w_range": (7, 12)},
    "Hard": {"n_circ": 33, "n_rect": 25, "circ_r_range": (4, 6), "rect_w_range": (7, 11)},
    "Very Hard": {"n_circ": 36, "n_rect": 27, "circ_r_range": (4, 6), "rect_w_range": (7, 10)},
}
CURATED_ENV_SEEDS = {
    "Easy":      [15795, 860, 76820, 54886, 6265, 82386, 37194, 87498, 44131, 60263],
    "Medium":    [16795, 1861, 77820, 55886, 7265, 83386, 38194, 88499, 45131, 61265],
    "Hard":      [17797, 2860, 78820, 56886, 8265, 84387, 39196, 89498, 46131, 62263],
    "Very Hard": [18795, 3862, 79820, 57886, 9266, 85390, 40194, 90499, 47131, 63263],
}
ENV_SIZE = 100

np.random.seed(42)
TRIAL_SEEDS = np.random.randint(0, 100000, size=50)
N_TRIALS = 50


def point_free(px, py, radius, circ_obs, rect_obs):
    for ob in circ_obs:
        if (px - ob.x)**2 + (py - ob.y)**2 < (radius + ob.r + 1.0)**2:
            return False
    for ob in rect_obs:
        if abs(px - ob.x) < ob.w/2 + radius + 1.0 and \
           abs(py - ob.y) < ob.h/2 + radius + 1.0:
            return False
    return True


def gen_env(cfg, rng):
    cr_min, cr_max = cfg["circ_r_range"]
    rw_min, rw_max = cfg["rect_w_range"]
    margin = -3.0
    circ_obs, rect_obs = [], []
    for _ in range(cfg["n_circ"]):
        for _ in range(500):
            r = rng.uniform(cr_min, cr_max)
            cx = rng.uniform(margin + r, ENV_SIZE - margin - r)
            cy = rng.uniform(margin + r, ENV_SIZE - margin - r)
            if (cx-start[0])**2 + (cy-start[1])**2 < (r+goal_radius+2.0)**2: continue
            if (cx-goal_2d[0])**2 + (cy-goal_2d[1])**2 < (r+goal_radius+2.0)**2: continue
            circ_obs.append(CircularObstacle2D(cx, cy, r))
            break
    for _ in range(cfg["n_rect"]):
        for _ in range(500):
            w = rng.uniform(rw_min, rw_max)
            h = rng.uniform(rw_min, rw_max)
            cx = rng.uniform(margin + w/2, ENV_SIZE - margin - w/2)
            cy = rng.uniform(margin + h/2, ENV_SIZE - margin - h/2)
            R = type('R', (), {'x': cx, 'y': cy, 'w': w, 'h': h})()
            if not point_free(start[0], start[1], goal_radius, [], [R]): continue
            if not point_free(goal_2d[0], goal_2d[1], goal_radius, [], [R]): continue
            rect_obs.append(RectangleObstacle2D(cx, cy, w, h))
            break
    return SquareEnvironment(ENV_SIZE, ENV_SIZE, circ_obs + rect_obs)


def make_bboe(env, seed, *, num_skip_edges=5, k_bias=0.85,
              exploitation_ratio=0.8):
    return BBoE(
        start=start, goal=goal_3d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb_190, eb_kd_tree=kd_tree_190,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        exploitation_ratio=exploitation_ratio,
        heuristic_radius=7.0, gamma=14.0,
        k_bias=k_bias, num_skip_edges=num_skip_edges,
        udf_seed=seed, print_logs=False)


def make_unidirectional_bboe(env, seed, *, num_skip_edges=5, k_bias=0.85):
    """Forward-only BBoE: no reverse tree, NO goal bias (per ablation request),
    same biased/random skipping strategy from Algorithm 2."""
    p = BBoE(
        start=start, goal=goal_3d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_max, minimum_time_step=0.1,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb_190, eb_kd_tree=kd_tree_190,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        exploitation_ratio=0.0,    # never exploit (no rev tree)
        heuristic_radius=7.0, gamma=14.0,
        k_bias=k_bias, num_skip_edges=num_skip_edges,
        udf_seed=seed, print_logs=False)
    return p


def run_unidirectional(p):
    """Forward-only plan_path. Mirrors BBoE's add/extend pattern but skips
    the reverse tree. Pure random target sampling — no goal bias."""
    p.reset_tree()
    p.add_node_for(p.start, -1, None, None, None, 0.0, 0.0)

    curr_iter = 0
    start_time = time.time()
    while curr_iter <= p.max_iter:
        x_des = p.sample_random_point()

        # Find nearest forward node; extend with biased/random skipping
        nearest_id, nearest_node = p.get_nearest_node_for(x_des)
        result = p.best_prop_using_edge_bundle(nearest_node, x_des, p.k_bias)
        if result is None:
            result = p.random_prop_using_edge_bundle(nearest_node)
        if result is not None:
            p._add_forward_node(nearest_id, nearest_node, *result)

        if p.path_found or (time.time() - start_time) >= p.planning_time:
            break
        curr_iter += 1


def run_one(planner_factory, env, seed, custom_runner=None):
    """Run one trial; return (success, time, cost)."""
    p = planner_factory(env, seed)
    t0 = time.time()
    if custom_runner is not None:
        custom_runner(p)
    else:
        p.plan_path()
    dt = time.time() - t0
    return p.path_found, dt, p.path_cost


def aggregate(trials):
    """trials = list of (success, time, cost). Return
    (n_succ, mean_t, min_t, max_t, mean_c, min_c, max_c).

    Failed trials are excluded from time/cost stats (they only count toward
    success-rate denominator). Min/max give the observed range across the
    successful trials. NaN when no successes; equal min == max when only
    one success."""
    n_succ = sum(1 for ok, _, _ in trials if ok)
    times = [t for ok, t, _ in trials if ok]
    costs = [c for ok, _, c in trials if ok and np.isfinite(c)]
    if times:
        mean_t = float(np.mean(times))
        min_t = float(np.min(times))
        max_t = float(np.max(times))
    else:
        mean_t = min_t = max_t = float('nan')
    if costs:
        mean_c = float(np.mean(costs))
        min_c = float(np.min(costs))
        max_c = float(np.max(costs))
    else:
        mean_c = min_c = max_c = float('nan')
    return n_succ, mean_t, min_t, max_t, mean_c, min_c, max_c


def run_for_difficulty_and_planner(diff_name, planner_factory,
                                   custom_runner=None):
    """Run 50 trials for the given (difficulty, planner_factory)."""
    cfg = DIFFICULTIES[diff_name]
    env_seeds = CURATED_ENV_SEEDS[diff_name]
    trials = []
    for trial_idx in range(N_TRIALS):
        env_seed = env_seeds[trial_idx % 10]
        udf_seed = int(TRIAL_SEEDS[trial_idx])
        env = gen_env(cfg, np.random.default_rng(env_seed))
        ok, dt, cost = run_one(planner_factory, env, udf_seed, custom_runner)
        trials.append((ok, dt, cost))
    return aggregate(trials)


# =========================================================
# Ablation 1 — num_skip_edges
# =========================================================
def ablate_skip_edges():
    print("\n=== Ablation 1 — num_skip_edges ===")
    skip_values = [1, 2, 5, 8, 10]   # 1 == no skip (every edge)
    # results: (skip, diff) -> (n_succ, mean_t, min_t, max_t, mean_c, min_c, max_c)
    results = {}
    for skip in skip_values:
        for diff_name in ("Easy", "Medium", "Hard", "Very Hard"):
            t0 = time.time()
            tup = run_for_difficulty_and_planner(
                diff_name,
                lambda env, seed, sk=skip: make_bboe(env, seed,
                                                     num_skip_edges=sk))
            elapsed = time.time() - t0
            results[(skip, diff_name)] = tup
            n_succ, mean_t, min_t, max_t, mean_c, min_c, max_c = tup
            print(f"  skip={skip:>2d}  {diff_name:<10s}  "
                  f"{n_succ:>2d}/{N_TRIALS}  t={mean_t:6.3f} [{min_t:.3f},{max_t:.3f}]s  "
                  f"c={mean_c:7.2f} [{min_c:.2f},{max_c:.2f}]  [took {elapsed:.1f}s]",
                  flush=True)
    return skip_values, results


def write_skip_table(skip_values, results, out_path):
    diffs = ["Easy", "Medium", "Hard", "Very Hard"]
    lines = []
    lines.append("Ablation 1 — BBoE num_skip_edges (50 seeds, 5 s budget)")
    lines.append("Time / cost format: mean [min, max] over successful trials only.")
    lines.append("")
    lines.append("Success rate (n/50):")
    head = "  skip_edges | " + " | ".join(f"{d:^20s}" for d in diffs)
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for sk in skip_values:
        cells = [f"{results[(sk, d)][0]:>3d}/{N_TRIALS}" for d in diffs]
        label = f"{sk:>10d}" if sk != 1 else "  1 (none)"
        lines.append(f"  {label} | " + " | ".join(f"{c:^20s}" for c in cells))
    lines.append("")
    lines.append("Planning time (s) — mean [min, max] on successes:")
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for sk in skip_values:
        cells = [f"{results[(sk, d)][1]:.2f} [{results[(sk, d)][2]:.2f},{results[(sk, d)][3]:.2f}]"
                 for d in diffs]
        label = f"{sk:>10d}" if sk != 1 else "  1 (none)"
        lines.append(f"  {label} | " + " | ".join(f"{c:^20s}" for c in cells))
    lines.append("")
    lines.append("Path cost — mean [min, max] on successes:")
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for sk in skip_values:
        cells = [f"{results[(sk, d)][4]:.1f} [{results[(sk, d)][5]:.1f},{results[(sk, d)][6]:.1f}]"
                 for d in diffs]
        label = f"{sk:>10d}" if sk != 1 else "  1 (none)"
        lines.append(f"  {label} | " + " | ".join(f"{c:^20s}" for c in cells))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


# =========================================================
# Ablation 2 — k_bias
# =========================================================
def ablate_kbias():
    print("\n=== Ablation 2 — k_bias ===")
    kbias_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    # results: (kbias, diff) -> (n_succ, mean_t, min_t, max_t, mean_c, min_c, max_c)
    results = {}
    for kb in kbias_values:
        for diff_name in ("Easy", "Medium", "Hard", "Very Hard"):
            t0 = time.time()
            tup = run_for_difficulty_and_planner(
                diff_name,
                lambda env, seed, k=kb: make_bboe(env, seed, k_bias=k))
            elapsed = time.time() - t0
            results[(kb, diff_name)] = tup
            n_succ, mean_t, min_t, max_t, mean_c, min_c, max_c = tup
            print(f"  k_bias={kb:.2f}  {diff_name:<10s}  "
                  f"{n_succ:>2d}/{N_TRIALS}  t={mean_t:6.3f} [{min_t:.3f},{max_t:.3f}]s  "
                  f"c={mean_c:7.2f} [{min_c:.2f},{max_c:.2f}]  [{elapsed:.1f}s]",
                  flush=True)
    return kbias_values, results


def plot_kbias(kbias_values, results, out_dir):
    """Time + cost: bar=mean, whiskers=[min, max] across successes.
    Success rate: bar only (range across binary trials is degenerate)."""
    diffs = ["Easy", "Medium", "Hard", "Very Hard"]
    diff_x = np.arange(len(diffs))
    n = len(kbias_values)
    slot = 0.8
    width = slot / n
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]
    offsets = (np.arange(n) - (n - 1) / 2) * width

    def _bars(ys_for, lows_for, highs_for, ylabel, title, fname, ylim=None):
        """ys_for: kb -> [mean per diff]. lows_for/highs_for: kb -> [whisker
        length per diff]. Both None ⇒ no whiskers."""
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, kb in enumerate(kbias_values):
            ys = ys_for(kb)
            yerr_pair = None
            if lows_for is not None and highs_for is not None:
                lo = [0.0 if (e is None or not np.isfinite(e)) else max(0.0, e)
                      for e in lows_for(kb)]
                hi = [0.0 if (e is None or not np.isfinite(e)) else max(0.0, e)
                      for e in highs_for(kb)]
                yerr_pair = [lo, hi]
            ax.bar(diff_x + offsets[i], ys, width, yerr=yerr_pair, capsize=3,
                   label=f"k_bias={kb:.2f}", color=colors[i],
                   edgecolor="black",
                   error_kw=dict(ecolor="black", lw=0.8, capthick=0.8))
        ax.set_xticks(diff_x); ax.set_xticklabels(diffs)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=120)
        plt.close(fig)

    # Indices into result tuple: (n_succ, mean_t, min_t, max_t, mean_c, min_c, max_c)
    _bars(
        ys_for=lambda kb: [results[(kb, d)][1] for d in diffs],
        lows_for=lambda kb: [results[(kb, d)][1] - results[(kb, d)][2] for d in diffs],
        highs_for=lambda kb: [results[(kb, d)][3] - results[(kb, d)][1] for d in diffs],
        ylabel="Planning time (s) — mean / [min, max]",
        title="BBoE k_bias ablation — Time vs difficulty (50 seeds, whiskers = min/max)",
        fname="ablation_kbias_time.png")
    # Success rate: no whiskers (range across binary trials is degenerate)
    _bars(
        ys_for=lambda kb: [results[(kb, d)][0] / N_TRIALS * 100 for d in diffs],
        lows_for=None, highs_for=None,
        ylabel="Success rate (%)",
        title="BBoE k_bias ablation — Success rate vs difficulty (50 seeds)",
        fname="ablation_kbias_success.png", ylim=(0, 105))
    _bars(
        ys_for=lambda kb: [results[(kb, d)][4] for d in diffs],
        lows_for=lambda kb: [results[(kb, d)][4] - results[(kb, d)][5] for d in diffs],
        highs_for=lambda kb: [results[(kb, d)][6] - results[(kb, d)][4] for d in diffs],
        ylabel="Path cost — mean / [min, max]",
        title="BBoE k_bias ablation — Path cost vs difficulty (50 seeds, whiskers = min/max)",
        fname="ablation_kbias_cost.png")
    print(f"Wrote ablation_kbias_{{time,success,cost}}.png to {out_dir}")


def write_kbias_table(kbias_values, results, out_path):
    diffs = ["Easy", "Medium", "Hard", "Very Hard"]
    lines = [
        "Ablation 2 — BBoE k_bias (50 seeds, 5 s budget)",
        "k_bias = P(pick from biased β_biased) vs (1 - k_bias) = P(pick random β_random)",
        "Time / cost format: mean [min, max] over successful trials only.",
        "",
        "Success rate (n/50):",
    ]
    head = "  k_bias    | " + " | ".join(f"{d:^20s}" for d in diffs)
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for kb in kbias_values:
        cells = [f"{results[(kb, d)][0]:>3d}/{N_TRIALS}" for d in diffs]
        lines.append(f"  {kb:>6.2f}    | " + " | ".join(f"{c:^20s}" for c in cells))
    lines.append("")
    lines.append("Planning time (s) — mean [min, max]:")
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for kb in kbias_values:
        cells = [f"{results[(kb, d)][1]:.2f} [{results[(kb, d)][2]:.2f},{results[(kb, d)][3]:.2f}]"
                 for d in diffs]
        lines.append(f"  {kb:>6.2f}    | " + " | ".join(f"{c:^20s}" for c in cells))
    lines.append("")
    lines.append("Path cost — mean [min, max]:")
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for kb in kbias_values:
        cells = [f"{results[(kb, d)][4]:.1f} [{results[(kb, d)][5]:.1f},{results[(kb, d)][6]:.1f}]"
                 for d in diffs]
        lines.append(f"  {kb:>6.2f}    | " + " | ".join(f"{c:^20s}" for c in cells))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


# =========================================================
# Ablation 3 — UnidirectionalBBoE
# =========================================================
def ablate_unidirectional():
    print("\n=== Ablation 3 — Unidirectional BBoE ===")
    # results: diff -> (n_succ, mean_t, min_t, max_t, mean_c, min_c, max_c)
    results = {}
    for diff_name in ("Easy", "Medium", "Hard", "Very Hard"):
        t0 = time.time()
        tup = run_for_difficulty_and_planner(
            diff_name,
            make_unidirectional_bboe,
            custom_runner=run_unidirectional)
        elapsed = time.time() - t0
        results[diff_name] = tup
        n_succ, mean_t, min_t, max_t, mean_c, min_c, max_c = tup
        print(f"  {diff_name:<10s}  {n_succ:>2d}/{N_TRIALS}  "
              f"t={mean_t:6.3f} [{min_t:.3f},{max_t:.3f}]s  "
              f"c={mean_c:7.2f} [{min_c:.2f},{max_c:.2f}]  "
              f"[{elapsed:.1f}s]", flush=True)
    return results


def write_uni_table(results, out_path):
    lines = []
    lines.append("Ablation 3 — Unidirectional BBoE (50 seeds, 5 s budget)")
    lines.append("Forward tree only, NO goal bias, biased/random skipping retained")
    lines.append("Time / cost format: mean [min, max] over successful trials only.")
    lines.append("")
    head = (f"  {'Difficulty':<11s} | {'Success':<10s} | "
            f"{'Time (s)':<26s} | {'Cost':<26s}")
    lines.append(head)
    lines.append("  " + "-" * (len(head) - 2))
    for d in ("Easy", "Medium", "Hard", "Very Hard"):
        n_succ, mean_t, min_t, max_t, mean_c, min_c, max_c = results[d]
        lines.append(
            f"  {d:<11s} | {n_succ:>2d}/{N_TRIALS}      | "
            f"{f'{mean_t:.3f} [{min_t:.3f},{max_t:.3f}]':<26s} | "
            f"{f'{mean_c:.2f} [{min_c:.2f},{max_c:.2f}]':<26s}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="BBoE ablations on the curated unicycle tiers.")
    ap.add_argument("--ablations", default="unidirectional",
                    choices=["unidirectional", "skip", "kbias", "all"],
                    help="which ablation(s) to run; the paper reports 'unidirectional' "
                         "(Table III(c)) from this script. The N and k_bias sweeps of "
                         "Table III(a,b) come from run_ablations_unicycle_hardcoded.py.")
    which = ap.parse_args().ablations
    if which in ("skip", "all"):
        skip_values, skip_results = ablate_skip_edges()
        write_skip_table(skip_values, skip_results,
                         OUT_DIR / "ablation_skip_edges_table.txt")
    if which in ("kbias", "all"):
        kbias_values, kbias_results = ablate_kbias()
        plot_kbias(kbias_values, kbias_results, OUT_DIR)
        write_kbias_table(kbias_values, kbias_results,
                          OUT_DIR / "ablation_kbias_table.txt")
    if which in ("unidirectional", "all"):
        uni_results = ablate_unidirectional()
        write_uni_table(uni_results, OUT_DIR / "ablation_unidirectional_table.txt")
    print("\nAll ablations complete.")


if __name__ == "__main__":
    main()
