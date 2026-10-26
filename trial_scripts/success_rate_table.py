"""Consolidated success-rate table across all 8 environments/systems.

Reuses the log parsers from plot_seed50_results.py so the numbers match the
success-rate plots exactly. Emits a markdown table (stdout + .md) and a .csv.

Environments (columns): the 4 curated unicycle difficulty tiers
(Easy/Medium/Hard/Very Hard) plus the 4 hardcoded systems
(Unicycle, CWT, Treaded, Quadrotor — narrow variant).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_seed50_results as P  # noqa: E402

LOG_DIR = P.LOG_DIR
OUT_DIR = P.OUT_DIR

# (display name, kind, source) — kind 'curated' keys into the curated log,
# 'hard' names a hardcoded log file.
ENVIRONMENTS = [
    ("Easy",      "curated", "Easy"),
    ("Medium",    "curated", "Medium"),
    ("Hard",      "curated", "Hard"),
    ("Very Hard", "curated", "Very Hard"),
    ("Unicycle",  "hard",    "unicycle.log"),
    ("CWT",       "hard",    "cwt.log"),
    ("Treaded",   "hard",    "treaded.log"),
    ("Quadrotor", "hard",    "quadrotor_narrow_50seed_full.log"),
]


def success_rate(trials):
    n = len(trials)
    if not n:
        return None
    return 100.0 * sum(1 for ok, _, _ in trials if ok) / n


def fmt(r):
    return "—" if r is None else f"{r:.0f}"


def main():
    curated = P.parse_curated_log(LOG_DIR / "curated.log")
    table = {}            # env display -> {planner: rate}
    planners_seen = set()
    for disp, kind, src in ENVIRONMENTS:
        results = (curated.get(src, {}) if kind == "curated"
                   else P.parse_hardcoded_log(LOG_DIR / src))
        table[disp] = {pl: success_rate(tr) for pl, tr in results.items()}
        planners_seen.update(results.keys())

    planners = P.planner_order({p: None for p in planners_seen},
                               prefer=P.HARDCODED_ORDER)
    cols = [d for d, _, _ in ENVIRONMENTS]

    # markdown
    md = []
    md.append("| Planner | " + " | ".join(cols) + " |")
    md.append("|" + "---|" * (len(cols) + 1))
    for pl in planners:
        cells = [fmt(table[d].get(pl)) for d in cols]
        md.append(f"| {pl} | " + " | ".join(cells) + " |")
    md_text = "\n".join(md)

    # csv
    csv = ["Planner," + ",".join(cols)]
    for pl in planners:
        csv.append(pl + "," + ",".join(fmt(table[d].get(pl)) for d in cols))
    csv_text = "\n".join(csv)

    (OUT_DIR / "success_rate_table.md").write_text(
        "# Success rate (%) by planner and environment — 50 seeds\n\n"
        "Columns 1-4 are curated unicycle difficulty tiers; columns 5-8 are the "
        "hardcoded systems (Quadrotor = narrow variant).\n\n" + md_text + "\n")
    (OUT_DIR / "success_rate_table.csv").write_text(csv_text + "\n")

    print("Success rate (%) — 50 seeds\n")
    print(md_text)
    print(f"\nWrote {OUT_DIR/'success_rate_table.md'}")
    print(f"Wrote {OUT_DIR/'success_rate_table.csv'}")


if __name__ == "__main__":
    main()
