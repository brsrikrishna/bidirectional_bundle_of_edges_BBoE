"""Parse 50-seed compare-script logs and emit 24 PNG plots.

Inputs (logs, persistent in repo):
  benchmark_logs/{unicycle, treaded, cwt, quadrotor, curated}.log

Outputs (PNGs):
  media/seed50_plots/{system}_{metric}.png  for hardcoded envs
  media/seed50_plots/curated_{difficulty}_{metric}.png  for curated

To refresh logs: rerun the relevant compare_*.py and pipe stdout into
benchmark_logs/<system>.log. Style-only changes (colors, labels) only
require rerunning this parser — no replanning needed.
"""
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]   # repository root
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "benchmark_logs"
OUT_DIR = REPO_ROOT / "media" / "seed50_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Match per-trial lines like:
#   "RRT-MC: ✅ | Time: 0.27s | Path Cost: 159.00"
#   "RRT: ❌ | Time: 5.00s | Path Cost: inf"
# (used by compare_unicycle / treaded / cwt / quadrotor)
# NOTE: the cost segment must be required, not optional. Wrapping it in
# `(?:...)?` lets the engine accept a match without ever trying to capture
# cost (because `.*?` is happy to consume zero chars and the optional group
# can no-op), yielding cost=None for every line. All logs include cost,
# so make it mandatory.
TRIAL_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z\-]*[A-Za-z]):\s*([✅❌]).*?Time:\s*([0-9.]+)s.*?(?:Path\s+)?Cost:\s*([0-9.]+|inf|nan)",
    re.MULTILINE,
)
# Match curated-script per-trial lines:
#   "            RRT-MC: ✅ | 0.07s | Cost: 122.0   "
CURATED_TRIAL_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z\-]*[A-Za-z]):\s*([✅❌])\s*\|\s*([0-9.]+)s\s*\|\s*Cost:\s*([0-9.]+|inf|nan)",
    re.MULTILINE,
)
# Curated difficulty header
DIFF_HEADER_RE = re.compile(r"DIFFICULTY:\s+([A-Za-z ]+?)\n")


def parse_hardcoded_log(log_path):
    """Return dict planner -> list of (success: bool, time: float, cost: float|None)."""
    text = log_path.read_text()
    results = {}
    for m in TRIAL_RE.finditer(text):
        planner, ok, t, cost = m.group(1), m.group(2), m.group(3), m.group(4)
        success = (ok == "✅")
        t = float(t)
        if cost in (None, "inf", "nan"):
            cost_v = None
        else:
            try:
                cost_v = float(cost)
                if not np.isfinite(cost_v):
                    cost_v = None
            except ValueError:
                cost_v = None
        results.setdefault(planner, []).append((success, t, cost_v))
    return results


def parse_curated_log(log_path):
    """Return dict difficulty -> dict planner -> list of (success, time, cost)."""
    text = log_path.read_text()
    out = {}
    # Split text into sections by difficulty header
    parts = re.split(r"DIFFICULTY:\s+([A-Za-z ]+?)\n", text)
    # parts[0] = preamble, then alternating [diff, body, diff, body, ...]
    for i in range(1, len(parts), 2):
        diff = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        # Stop at the SUMMARY block
        body = body.split("SUMMARY")[0]
        results = {}
        for m in CURATED_TRIAL_RE.finditer(body):
            planner, ok, t, cost = m.group(1), m.group(2), m.group(3), m.group(4)
            success = (ok == "✅")
            t = float(t)
            if cost in ("inf", "nan"):
                cost_v = None
            else:
                try:
                    cost_v = float(cost)
                    if not np.isfinite(cost_v):
                        cost_v = None
                except ValueError:
                    cost_v = None
            results.setdefault(planner, []).append((success, t, cost_v))
        if results:
            out[diff] = results
    return out


def planner_order(results, prefer=None):
    """Return planners in a stable display order: prefer list first, then alpha."""
    prefer = prefer or []
    keys = list(results.keys())
    seen = set()
    out = []
    for p in prefer:
        if p in keys and p not in seen:
            out.append(p); seen.add(p)
    for p in sorted(keys):
        if p not in seen:
            out.append(p); seen.add(p)
    return out


# Display order — RRT/SST/BoE/GBRRT/Db-RRT-F/Db-RRT-C/BBoE (BBoE rightmost)
HARDCODED_ORDER = [
    "RRT", "RRT-MC", "RRT-EB",
    "SST", "SST-MC", "SST-EB",
    "BoE",
    "GBRRT", "GBRRT-MC", "GBRRT-EB",
    "Db-RRT-F",
    "Db-RRT-C",
    "BBoE",
]
CURATED_ORDER = HARDCODED_ORDER  # same


def plot_metric(planners, values, title, ylabel, out_path,
                color="#1f77b4", show_value=True, yerr_lower=None,
                yerr_upper=None):
    """Bar plot — one bar per planner.

    `yerr_lower` / `yerr_upper` are the lengths of the lower / upper whiskers
    (i.e. distance from bar top down/up). Pass both None for no whiskers.
    Caller computes `yerr_lower = mean - min` and `yerr_upper = max - mean`
    so whiskers represent the observed range across trials.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(planners) * 0.9), 5))
    xs = np.arange(len(planners))
    yerr_pair = None
    if yerr_lower is not None and yerr_upper is not None:
        lo = [0.0 if (e is None or not np.isfinite(e)) else max(0.0, e)
              for e in yerr_lower]
        hi = [0.0 if (e is None or not np.isfinite(e)) else max(0.0, e)
              for e in yerr_upper]
        yerr_pair = [lo, hi]
    bars = ax.bar(xs, values, yerr=yerr_pair, capsize=4,
                  color=color, edgecolor="black", alpha=0.85,
                  error_kw=dict(ecolor="black", lw=1.0, capthick=1.0))
    ax.set_xticks(xs)
    ax.set_xticklabels(planners, rotation=30, ha="right", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    if show_value:
        for i, (b, v) in enumerate(zip(bars, values)):
            if not np.isfinite(v):
                continue
            # Label sits exactly on the bar top (the mean). Render with a
            # white background so the upper-range whisker doesn't visually
            # obscure the text. zorder above bars + whiskers.
            ax.annotate(
                f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="center", fontsize=8, zorder=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.85))
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def _stats_for_planner(trials):
    """Return per-planner stats over only-successful trials.

    Returns: (mean_t, min_t, max_t, succ_rate_pct, mean_c, min_c, max_c).
    Failed trials contribute to the success-rate denominator only — they are
    not included in time/cost mean/min/max.
    """
    n = len(trials)
    succ_trials = [(t, c) for ok, t, c in trials if ok]
    p_succ = (len(succ_trials) / n) if n else 0.0
    rate = p_succ * 100.0
    if succ_trials:
        time_arr = np.array([t for t, _ in succ_trials])
        mean_t = float(time_arr.mean())
        min_t = float(time_arr.min())
        max_t = float(time_arr.max())
        valid = [c for _, c in succ_trials if c is not None]
        if valid:
            cost_arr = np.array(valid)
            mean_c = float(cost_arr.mean())
            min_c = float(cost_arr.min())
            max_c = float(cost_arr.max())
        else:
            mean_c = min_c = max_c = float("nan")
    else:
        mean_t = min_t = max_t = float("nan")
        mean_c = min_c = max_c = float("nan")
    return mean_t, min_t, max_t, rate, mean_c, min_c, max_c


# Per-planner bar colors. BBoE = green, GBRRT-family + Db-RRT = yellow,
# everything else (RRT*, SST*, BoE) = red.
_GREEN = "#2ca02c"
_YELLOW = "#e6b800"   # darker than pure yellow so white-bg labels stay readable
_RED = "#d62728"


def _planner_colors(planners):
    out = []
    for name in planners:
        if name == "BBoE":
            out.append(_GREEN)
        elif name in ("GBRRT", "GBRRT-MC", "GBRRT-EB",
                      "Db-RRT", "Db-RRT-F", "Db-RRT-C"):
            out.append(_YELLOW)
        else:
            out.append(_RED)
    return out


def emit_plots_for_system(system_label, results, out_prefix):
    """Emit time, success-rate, and cost plots for a single hardcoded env.

    Time + cost: bar = mean across successes, whiskers = [min, max] range.
    Success rate: bar only — range across binary trials is degenerate (always
    [0%, 100%]) so whiskers are omitted.
    Bar colors: BBoE green, GBRRT-* / Db-RRT yellow, others red.
    """
    planners = planner_order(results, prefer=HARDCODED_ORDER)
    colors = _planner_colors(planners)
    times, t_lows, t_highs = [], [], []
    succ_rates = []
    costs, c_lows, c_highs = [], [], []
    for p in planners:
        mt, mn_t, mx_t, rate, mc, mn_c, mx_c = _stats_for_planner(results[p])
        times.append(mt); t_lows.append(mt - mn_t); t_highs.append(mx_t - mt)
        succ_rates.append(rate)
        costs.append(mc); c_lows.append(mc - mn_c); c_highs.append(mx_c - mc)

    plot_metric(planners, times,
                f"Initial solution time vs Planner — {system_label} "
                f"(50 seeds, whiskers = min/max)",
                "Mean planning time (s) on successes",
                OUT_DIR / f"{out_prefix}_time.png", color=colors,
                yerr_lower=t_lows, yerr_upper=t_highs)
    plot_metric(planners, succ_rates,
                f"Success rate vs Planner — {system_label} (50 seeds)",
                "Success rate (%)",
                OUT_DIR / f"{out_prefix}_success.png", color=colors)
    plot_metric(planners, costs,
                f"Solution cost vs Planner — {system_label} "
                f"(50 seeds, whiskers = min/max)",
                "Mean path cost on successes",
                OUT_DIR / f"{out_prefix}_cost.png", color=colors,
                yerr_lower=c_lows, yerr_upper=c_highs)
    print(f"  wrote {out_prefix}_time/success/cost.png")


def main():
    # ---- 4 hardcoded envs ----
    HARDCODED = {
        "Unicycle": ("unicycle.log", "unicycle"),
        "Treaded": ("treaded.log", "treaded"),
        "CWT": ("cwt.log", "cwt"),
        "Quadrotor (Narrow)": ("quadrotor_narrow_50seed_full.log", "quadrotor_narrow"),
    }
    for label, (log_name, prefix) in HARDCODED.items():
        log_path = LOG_DIR / log_name
        if not log_path.exists():
            print(f"SKIP {label}: log not found")
            continue
        results = parse_hardcoded_log(log_path)
        if not results:
            print(f"SKIP {label}: no parseable trials in log")
            continue
        n = max(len(v) for v in results.values())
        print(f"\n=== {label} ({n} trials per planner) ===")
        emit_plots_for_system(label, results, prefix)

    # ---- Curated unicycle: 4 difficulties × 3 metrics ----
    curated_log = LOG_DIR / "curated.log"
    if curated_log.exists():
        print("\n=== Curated unicycle ===")
        diffs = parse_curated_log(curated_log)
        for diff_name, results in diffs.items():
            n = max(len(v) for v in results.values()) if results else 0
            print(f"  {diff_name}: {n} trials per planner")
            label = f"Unicycle Curated — {diff_name}"
            slug = diff_name.lower().replace(" ", "_")
            emit_plots_for_system(label, results, f"curated_{slug}")
    else:
        print("SKIP curated: log not found")

    print(f"\nAll plots written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
