"""Summarize a benchmark log: per-planner success count, mean solution time and mean cost.

Works for the fixed-environment logs (unicycle, treaded, cwt, quadrotor) and the curated
unicycle log (per difficulty tier). With --compare, prints a second log side by side and
the differences, which is the quickest way to check a rerun against the shipped logs.

  python tools/summarize_log.py benchmark_logs/unicycle.log
  python tools/summarize_log.py my_rerun.log --compare benchmark_logs/unicycle.log
"""
import argparse
import re
from collections import OrderedDict, defaultdict

LINE = re.compile(r"^\s*([A-Za-z0-9\-\+\(\) ]+?):\s*(✅|❌)\s*\|\s*(?:Time:\s*)?([0-9.]+)s\s*\|\s*(?:Path )?Cost:\s*([0-9.]+|inf|nan)")
TIER = re.compile(r"DIFFICULTY:\s*(.+)")


def parse(path):
    """Return {tier: {planner: [(ok, time, cost), ...]}} ('' tier for fixed-environment logs)."""
    out = OrderedDict()
    tier = ""
    for line in open(path, encoding="utf-8", errors="replace"):
        m = TIER.search(line)
        if m:
            tier = m.group(1).strip()
            continue
        m = LINE.match(line)
        if not m:
            continue
        name, ok, t, c = m.group(1).strip(), m.group(2) == "✅", float(m.group(3)), float(m.group(4))
        out.setdefault(tier, OrderedDict()).setdefault(name, []).append((ok, t, c))
    return out


def stats(trials):
    n = len(trials)
    succ = [x for x in trials if x[0]]
    mt = sum(x[1] for x in succ) / len(succ) if succ else float("nan")
    mc = sum(x[2] for x in succ) / len(succ) if succ else float("nan")
    return len(succ), n, mt, mc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--compare", help="reference log to compare against (e.g. the shipped one)")
    a = ap.parse_args()
    A = parse(a.log)
    B = parse(a.compare) if a.compare else None
    for tier, planners in A.items():
        if tier:
            print(f"\n== {tier} ==")
        hdr = f"{'planner':12s} {'success':>9s} {'mean t[s]':>10s} {'mean cost':>10s}"
        if B:
            hdr += f" | {'ref succ':>9s} {'ref t[s]':>9s} {'ref cost':>9s} | {'Δsucc':>6s} {'Δt[s]':>7s}"
        print(hdr)
        for name, trials in planners.items():
            s, n, mt, mc = stats(trials)
            row = f"{name:12s} {s:>5d}/{n:<3d} {mt:>10.2f} {mc:>10.2f}"
            if B and name in B.get(tier, {}):
                rs, rn, rmt, rmc = stats(B[tier][name])
                row += f" | {rs:>5d}/{rn:<3d} {rmt:>9.2f} {rmc:>9.2f} | {s - rs:>+6d} {mt - rmt:>+7.2f}"
            print(row)


if __name__ == "__main__":
    main()
