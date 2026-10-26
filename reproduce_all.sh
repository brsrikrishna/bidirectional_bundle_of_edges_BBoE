#!/usr/bin/env bash
# Regenerates every table and figure of the paper from scratch (about 1.5 hours on the reference laptop).
# Run the benchmarks one at a time: the 5 s planning budget is wall-clock time, so concurrent jobs
# shift the success counts of planners that finish close to the budget.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
mkdir -p benchmark_logs
$PY trial_scripts/compare_unicycle.py              > benchmark_logs/unicycle.log
$PY trial_scripts/compare_treaded.py               > benchmark_logs/treaded.log
$PY trial_scripts/compare_car_with_trailer.py      > benchmark_logs/cwt.log
$PY trial_scripts/compare_quadrotor_narrow.py      > benchmark_logs/quadrotor_narrow_50seed_full.log
$PY trial_scripts/compare_difficulty_curated.py    > benchmark_logs/curated.log
$PY trial_scripts/run_ablations_unicycle_hardcoded.py          # Table III(a), (b)
$PY trial_scripts/run_ablations.py --ablations unidirectional  # Table III(c)
$PY trial_scripts/plot_seed50_results.py                       # Figs. 5 and 6
$PY trial_scripts/success_rate_table.py                        # Table II
$PY trial_scripts/render_bboe_stills.py && $PY trial_scripts/create_bboe_montage.py   # Fig. 3
echo "Done. Logs in benchmark_logs/, plots and tables in media/seed50_plots/."
