"""BBoE ablations on the Unicycle system in the HARDCODED compare_unicycle
environment (59 obstacles, 100×100 map), using a 1000-edge random-duration
edge bundle (vs the 190-edge bundle used by the original compare).

Runs two ablations across 50 seeds, 5 s planning budget:
  1. num_skip_edges  ∈ {1, 5, 20, 100}
  2. k_bias          ∈ {0.0, 0.25, 0.5, 0.75, 1.0}

The hardcoded compare_unicycle env is fixed (not difficulty-stratified), so
each ablation is a 1-D sweep.

Outputs (media/seed50_plots/):
  unicycle_hardcoded_ablation_skip_edges_table.txt
  unicycle_hardcoded_ablation_kbias_{time,success,cost}.png
  unicycle_hardcoded_ablation_kbias_table.txt
"""
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]   # repository root
import time
sys.path.append(str(ROOT / 'src'))
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Environments import SquareEnvironment, CircularObstacle2D, RectangleObstacle2D
from mapf_env_square_agent_unicycle import get_unicycle_agent
from bboe import BBoE
from edge_bundle import EdgeBundle
from kd_tree_unicycle import CircularAngleIndexNumba

OUT_DIR = ROOT / "media/seed50_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 50
PLANNING_TIME = 5.0
MAX_ITER = 100_000
T_MAX = 1.0
MIN_DT = 0.1


# ============================================================
# Unicycle env (hardcoded compare_unicycle.py — sourced via exec to keep
# obstacle coords in one place)
# ============================================================
_src_path = str(ROOT / 'trial_scripts/compare_unicycle.py')
_src = open(_src_path).read()
# Run only the env+agent setup, stop before the seed loop / planner runs.
_setup = _src.split('seeds = list(range(50))')[0]
_ns = {'__file__': _src_path}
exec(_setup, _ns)
env = _ns['env']
agent = _ns['agent']
start = _ns['start']
goal_3d = _ns['goal']            # 3-D unicycle goal
goal_radius = _ns['goal_radius']

# 1000-edge random-duration bundle (NEW for these ablations) — overrides
# the 190-edge bundle that compare_unicycle's setup loaded.
data = np.load(str(ROOT / 'edge_bundles/eb_unicycle_random_duration_1000.npz'))
eb = EdgeBundle(data, use_all_edges=True)
edge_ids = np.arange(eb.num_edges, dtype=np.int64)
kd_tree_eb = CircularAngleIndexNumba(eb.start_states[:, 2], ids=edge_ids)


TRIAL_SEEDS = list(range(N_TRIALS))


# ============================================================
# Helpers
# ============================================================
def aggregate(trials):
    n_succ = sum(1 for ok, _, _ in trials if ok)
    times = [t for ok, t, _ in trials if ok]
    costs = [c for ok, _, c in trials if ok and np.isfinite(c)]
    if times:
        mean_t = float(np.mean(times)); min_t = float(np.min(times)); max_t = float(np.max(times))
    else:
        mean_t = min_t = max_t = float('nan')
    if costs:
        mean_c = float(np.mean(costs)); min_c = float(np.min(costs)); max_c = float(np.max(costs))
    else:
        mean_c = min_c = max_c = float('nan')
    return n_succ, mean_t, min_t, max_t, mean_c, min_c, max_c


def run_one(planner_factory, seed, custom_runner=None):
    p = planner_factory(seed)
    t0 = time.time()
    if custom_runner is not None:
        custom_runner(p)
    else:
        p.plan_path()
    return p.path_found, time.time() - t0, p.path_cost


def run_50(planner_factory, custom_runner=None):
    trials = []
    for s in TRIAL_SEEDS:
        ok, dt, c = run_one(planner_factory, s, custom_runner)
        trials.append((ok, dt, c))
    return aggregate(trials)


# ============================================================
# BBoE factory (matches compare_unicycle.py BBoE block exactly)
# ============================================================
def make_bboe(seed, *, num_skip_edges=5, k_bias=0.85,
              exploitation_ratio=0.8, heuristic_radius=7.0, gamma=14.0):
    return BBoE(
        start=start, goal=goal_3d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_MAX, minimum_time_step=MIN_DT,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb, eb_kd_tree=kd_tree_eb,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        exploitation_ratio=exploitation_ratio,
        heuristic_radius=heuristic_radius, gamma=gamma,
        k_bias=k_bias, num_skip_edges=num_skip_edges,
        udf_seed=seed, print_logs=False)


def make_unidirectional_bboe(seed, *, num_skip_edges=5, k_bias=0.85):
    """Forward tree only, NO goal bias."""
    return BBoE(
        start=start, goal=goal_3d, goal_radius=goal_radius,
        env=env, agent=agent,
        sampling_time_step=T_MAX, minimum_time_step=MIN_DT,
        max_iter=MAX_ITER, planning_time=PLANNING_TIME,
        isvalid_function=agent.is_new_node_valid,
        cost_function=agent.get_cost,
        random_point_function=agent.get_random_point,
        reached_goal_function=agent.agent_reached_goal,
        edge_bundle=eb, eb_kd_tree=kd_tree_eb,
        get_eb_kd_tree_query=agent.get_eb_kd_tree_query,
        kd_tree_delta_radius=np.pi,
        translate_function=agent.kd_tree_point_translate_function,
        sort_edges_function=agent.sort_kd_tree_edges,
        exploitation_ratio=0.0,
        heuristic_radius=7.0, gamma=14.0,
        k_bias=k_bias, num_skip_edges=num_skip_edges,
        udf_seed=seed, print_logs=False)


def run_unidirectional(p):
    p.reset_tree()
    p.add_node_for(p.start, -1, None, None, None, 0.0, 0.0)
    curr_iter = 0
    t0 = time.time()
    while curr_iter <= p.max_iter:
        x_des = p.sample_random_point()
        nearest_id, nearest_node = p.get_nearest_node_for(x_des)
        result = p.best_prop_using_edge_bundle(nearest_node, x_des, p.k_bias)
        if result is None:
            result = p.random_prop_using_edge_bundle(nearest_node)
        if result is not None:
            p._add_forward_node(nearest_id, nearest_node, *result)
        if p.path_found or (time.time() - t0) >= p.planning_time:
            break
        curr_iter += 1


# ============================================================
def warmup():
    print("Warming up JIT (one short BBoE run)...")
    p = make_bboe(99); p.planning_time = 0.5
    p.plan_path()


# ============================================================
def ablate_skip_edges():
    print("\n=== Ablation 1 — num_skip_edges (Unicycle hardcoded, 1000-eb) ===")
    skip_values = [1, 5, 20, 100]
    results = {}
    for skip in skip_values:
        t0 = time.time()
        tup = run_50(lambda s, sk=skip: make_bboe(s, num_skip_edges=sk))
        elapsed = time.time() - t0
        results[skip] = tup
        n, mt, mn_t, mx_t, mc, mn_c, mx_c = tup
        print(f"  skip={skip:>2d}  {n:>2d}/{N_TRIALS}  "
              f"t={mt:6.3f} [{mn_t:.3f},{mx_t:.3f}]s  "
              f"c={mc:7.2f} [{mn_c:.2f},{mx_c:.2f}]  [{elapsed:.1f}s]",
              flush=True)
    return skip_values, results


def write_skip_table(skip_values, results, out_path):
    lines = [
        "Unicycle hardcoded Ablation 1 — BBoE num_skip_edges (50 seeds, 5 s budget, 1000-edge bundle)",
        "Time / cost format: mean [min, max] over successful trials only.",
        "",
        f"{'skip_edges':>11s} | {'Success':>9s} | "
        f"{'Time (s)':<22s} | {'Cost':<22s}",
        "-" * 80,
    ]
    for sk in skip_values:
        n, mt, mn_t, mx_t, mc, mn_c, mx_c = results[sk]
        lines.append(
            f"{sk:>11d} | {n:>3d}/{N_TRIALS}    | "
            f"{f'{mt:.3f} [{mn_t:.3f},{mx_t:.3f}]':<22s} | "
            f"{f'{mc:.2f} [{mn_c:.2f},{mx_c:.2f}]':<22s}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


# ============================================================
def ablate_kbias():
    print("\n=== Ablation 2 — k_bias (Unicycle hardcoded, 1000-eb) ===")
    kbias_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    for kb in kbias_values:
        t0 = time.time()
        tup = run_50(lambda s, k=kb: make_bboe(s, k_bias=k))
        elapsed = time.time() - t0
        results[kb] = tup
        n, mt, mn_t, mx_t, mc, mn_c, mx_c = tup
        print(f"  k_bias={kb:.2f}  {n:>2d}/{N_TRIALS}  "
              f"t={mt:6.3f} [{mn_t:.3f},{mx_t:.3f}]s  "
              f"c={mc:7.2f} [{mn_c:.2f},{mx_c:.2f}]  [{elapsed:.1f}s]",
              flush=True)
    return kbias_values, results


def plot_kbias(kbias_values, results, out_dir):
    n = len(kbias_values)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]
    xs = np.arange(n)
    width = 0.7

    def _bar(values, lows, highs, ylabel, title, fname, ylim=None):
        fig, ax = plt.subplots(figsize=(7, 5))
        yerr_pair = None
        if lows is not None and highs is not None:
            lo = [0.0 if (e is None or not np.isfinite(e)) else max(0.0, e) for e in lows]
            hi = [0.0 if (e is None or not np.isfinite(e)) else max(0.0, e) for e in highs]
            yerr_pair = [lo, hi]
        bars = ax.bar(xs, values, width, yerr=yerr_pair, capsize=4,
                      color=colors, edgecolor="black", alpha=0.85,
                      error_kw=dict(ecolor="black", lw=1.0, capthick=1.0))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{kb:.2f}" for kb in kbias_values])
        ax.set_xlabel("k_bias")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, values):
            if not np.isfinite(v):
                continue
            ax.annotate(
                f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="center", fontsize=9, zorder=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.85))
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=120)
        plt.close(fig)

    times = [results[k][1] for k in kbias_values]
    t_lo = [results[k][1] - results[k][2] for k in kbias_values]
    t_hi = [results[k][3] - results[k][1] for k in kbias_values]
    rates = [results[k][0] / N_TRIALS * 100 for k in kbias_values]
    costs = [results[k][4] for k in kbias_values]
    c_lo = [results[k][4] - results[k][5] for k in kbias_values]
    c_hi = [results[k][6] - results[k][4] for k in kbias_values]

    _bar(times, t_lo, t_hi,
         "Planning time (s) — mean / [min, max]",
         "BBoE k_bias ablation (Unicycle hardcoded, 1000-eb) — Time (50 seeds, whiskers = min/max)",
         "unicycle_hardcoded_ablation_kbias_time.png")
    _bar(rates, None, None,
         "Success rate (%)",
         "BBoE k_bias ablation (Unicycle hardcoded, 1000-eb) — Success rate (50 seeds)",
         "unicycle_hardcoded_ablation_kbias_success.png", ylim=(0, 105))
    _bar(costs, c_lo, c_hi,
         "Path cost — mean / [min, max]",
         "BBoE k_bias ablation (Unicycle hardcoded, 1000-eb) — Path cost (50 seeds, whiskers = min/max)",
         "unicycle_hardcoded_ablation_kbias_cost.png")
    print(f"Wrote unicycle_hardcoded_ablation_kbias_{{time,success,cost}}.png to {out_dir}")


def write_kbias_table(kbias_values, results, out_path):
    lines = [
        "Unicycle hardcoded Ablation 2 — BBoE k_bias (50 seeds, 5 s budget, 1000-edge bundle)",
        "k_bias = P(pick from biased β_biased) vs (1 - k_bias) = P(pick random β_random)",
        "Time / cost format: mean [min, max] over successful trials only.",
        "",
        f"{'k_bias':>8s} | {'Success':>9s} | "
        f"{'Time (s)':<22s} | {'Cost':<22s}",
        "-" * 78,
    ]
    for kb in kbias_values:
        n, mt, mn_t, mx_t, mc, mn_c, mx_c = results[kb]
        lines.append(
            f"{kb:>8.2f} | {n:>3d}/{N_TRIALS}    | "
            f"{f'{mt:.3f} [{mn_t:.3f},{mx_t:.3f}]':<22s} | "
            f"{f'{mc:.2f} [{mn_c:.2f},{mx_c:.2f}]':<22s}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


# ============================================================
def ablate_unidirectional():
    print("\n=== Ablation 3 — Unidirectional BBoE, no goal bias (Unicycle hardcoded, 1000-eb) ===")
    t0 = time.time()
    tup = run_50(make_unidirectional_bboe, custom_runner=run_unidirectional)
    elapsed = time.time() - t0
    n, mt, mn_t, mx_t, mc, mn_c, mx_c = tup
    print(f"  Unidirectional   {n:>2d}/{N_TRIALS}  "
          f"t={mt:6.3f} [{mn_t:.3f},{mx_t:.3f}]s  "
          f"c={mc:7.2f} [{mn_c:.2f},{mx_c:.2f}]  [{elapsed:.1f}s]",
          flush=True)
    return tup


def write_uni_table(result, out_path):
    n, mt, mn_t, mx_t, mc, mn_c, mx_c = result
    lines = [
        "Unicycle hardcoded Ablation 3 — Unidirectional BBoE (50 seeds, 5 s budget, 1000-edge bundle)",
        "Forward tree only, NO goal bias, biased/random skipping retained.",
        "Time / cost format: mean [min, max] over successful trials only.",
        "",
        f"{'Config':>22s} | {'Success':>9s} | "
        f"{'Time (s)':<22s} | {'Cost':<22s}",
        "-" * 86,
        f"{'Unidirectional (no GB)':>22s} | {n:>3d}/{N_TRIALS}    | "
        f"{f'{mt:.3f} [{mn_t:.3f},{mx_t:.3f}]':<22s} | "
        f"{f'{mc:.2f} [{mn_c:.2f},{mx_c:.2f}]':<22s}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


# ============================================================
def main():
    warmup()
    skip_values, skip_results = ablate_skip_edges()
    write_skip_table(skip_values, skip_results,
                     OUT_DIR / "unicycle_hardcoded_ablation_skip_edges_table.txt")
    kbias_values, kbias_results = ablate_kbias()
    plot_kbias(kbias_values, kbias_results, OUT_DIR)
    write_kbias_table(kbias_values, kbias_results,
                      OUT_DIR / "unicycle_hardcoded_ablation_kbias_table.txt")
    print("\nUnicycle hardcoded ablations (skip-edges + k_bias) complete.")


if __name__ == "__main__":
    main()
