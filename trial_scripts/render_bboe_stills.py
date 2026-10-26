"""
Re-render the final (path-found) frame of each BBoE planning GIF as a clean,
text-free still PNG, using one unified colour palette across all systems.

Why re-render instead of cropping the GIF frame: the titles and legends are
rasterised *into* the frames (legend overlaps the plot), so they cannot be
cropped away. Each create_bboe_*_gif.py script splits cleanly into a
"planning setup" part (which builds the planner, tree edges and solution path)
and an "animation" part. We exec only the setup part, then redraw the final
state ourselves with no title / no legend / axes off.

Planning is deterministic for the 4 hardcoded systems (fixed seed, breaks on
path_found) and uses a fixed env + seed for the difficulty tiers, so the redraw
matches the GIFs.

Output: media/montage_panels/{key}.png
"""
import os
import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]   # repository root
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.append(str(ROOT / 'src'))

OUT_DIR = str(ROOT / 'media/montage_panels')
os.makedirs(OUT_DIR, exist_ok=True)

# ---- unified palette (adopted from create_difficulty_planner_gif.py) ----
C_OBST_FACE = 'gray'
C_OBST_EDGE = 'black'
C_FORWARD   = 'steelblue'
C_REVERSE   = 'tomato'
C_SOLUTION  = 'lime'
START_FACE  = '#555555'
GOAL_FACE   = 'lime'

TREE_LW, TREE_ALPHA = 0.7, 0.6
SOL_LW = 2.6
MARKER_MS = 12


def run_prefix(script_path, split_text, argv=None):
    """Exec a GIF script up to (but not including) split_text; return its namespace."""
    with open(script_path) as fh:
        src = fh.read()
    assert split_text in src, f"split marker not found in {script_path!r}"
    prefix = src.split(split_text)[0]
    ns = {'__name__': '__main__', '__file__': os.path.abspath(script_path)}
    old_argv = sys.argv
    if argv is not None:
        sys.argv = argv
    try:
        exec(compile(prefix, script_path, 'exec'), ns)
    finally:
        sys.argv = old_argv
    return ns


def _lc(ax, edges, color, lw, alpha, z=1):
    if not edges:
        return
    segs = [[(e[0][0], e[0][1]), (e[1][0], e[1][1])] for e in edges]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw,
                                     alpha=alpha, zorder=z))


def render_2d(out, circ, rect, for_e, rev_e, sol_e,
              start_xy, goal_xy, goal_r, xlim, ylim):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    for ob in circ:
        ax.add_patch(patches.Circle((ob[0], ob[1]), ob[2],
                     facecolor=C_OBST_FACE, edgecolor=C_OBST_EDGE, linewidth=0.6))
    for ob in rect:                       # rect = (x0, x1, y0, y1)
        ax.add_patch(patches.Rectangle((ob[0], ob[2]), ob[1] - ob[0], ob[3] - ob[2],
                     facecolor=C_OBST_FACE, edgecolor=C_OBST_EDGE, linewidth=0.6))
    _lc(ax, rev_e, C_REVERSE, TREE_LW, TREE_ALPHA, z=2)
    _lc(ax, for_e, C_FORWARD, TREE_LW, TREE_ALPHA, z=3)
    _lc(ax, sol_e, C_SOLUTION, SOL_LW, 0.95, z=8)
    ax.plot(start_xy[0], start_xy[1], marker='o', ms=MARKER_MS, mfc=START_FACE,
            mec='black', ls='None', zorder=9)
    ax.plot(goal_xy[0], goal_xy[1], marker='o', ms=MARKER_MS, mfc=GOAL_FACE,
            mec='black', ls='None', zorder=9)
    if goal_r:
        ax.add_patch(patches.Circle((goal_xy[0], goal_xy[1]), goal_r, fill=False,
                     edgecolor=GOAL_FACE, linestyle='--', zorder=7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"  saved {out}")


def render_3d(out, draw_cuboid, obs_data, for_e, rev_e, sol_xyz,
              start_xyz, goal_xyz):
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    for cx, cy, cz, lx, ly, lz in obs_data:
        draw_cuboid(ax, cx, cy, cz, lx, ly, lz, color=C_OBST_FACE, alpha=0.15)

    def lc3(edges, color, lw, alpha):
        if not edges:
            return
        segs = [[tuple(e[0]), tuple(e[1])] for e in edges]
        ax.add_collection3d(Line3DCollection(segs, colors=color,
                                             linewidths=lw, alpha=alpha))
    lc3(rev_e, C_REVERSE, 0.8, TREE_ALPHA)
    lc3(for_e, C_FORWARD, 0.8, TREE_ALPHA)
    if len(sol_xyz) > 1:
        ax.plot(sol_xyz[:, 0], sol_xyz[:, 1], sol_xyz[:, 2],
                color=C_SOLUTION, lw=3, alpha=0.95)
    ax.scatter(*start_xyz, c=START_FACE, edgecolors='black', s=80, zorder=5)
    ax.scatter(*goal_xyz, c=GOAL_FACE, edgecolors='black', s=80, zorder=5)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 10)
    ax.view_init(elev=25, azim=-60)
    ax.set_axis_off()                      # remove panes/ticks/labels -> tight, text-free
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"  saved {out}")


# ---- quadrotor obstacles live inside the animation block; copy them here ----
QUAD_OBS = [
    (14, 4, 5, 4, 8, 10), (14, 28, 5, 4, 24, 10),
    (14, 12, 1.5, 4, 8, 3), (14, 12, 8.5, 4, 8, 3),
    (26, 12, 5, 4, 24, 10), (26, 36, 5, 4, 8, 10),
    (26, 28, 2.5, 4, 8, 5), (26, 28, 9.5, 4, 8, 1),
]

DIFFS = [('unicycle_easy', 'Easy'), ('unicycle_medium', 'Medium'),
         ('unicycle_hard', 'Hard'), ('unicycle_very_hard', 'Very Hard')]

DIFF_SPLIT = 'fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=90)'
HARD_SPLIT = '# --- Animation ---'
QUAD_SPLIT = 'fig = plt.figure(figsize=(7, 5), dpi=72)'


def flat(history, key):
    return [e for step in history for e in step[key]]


def do_difficulty(key, difficulty):
    print(f"[{key}] re-running difficulty={difficulty} ...")
    ns = run_prefix(str(ROOT / 'trial_scripts/create_difficulty_planner_gif.py'), DIFF_SPLIT,
                    argv=['create_difficulty_planner_gif.py',
                          '--planner=bboe', f'--difficulty={difficulty}'])
    env = ns['env']
    size = ns['ENV_SIZE']
    render_2d(f"{OUT_DIR}/{key}.png",
              env.static_circular_obstacles, env.static_rectangular_obstacles,
              ns['for_edges'], ns['rev_edges'], ns['sol_edges'],
              np.asarray(ns['start'])[:2], np.asarray(ns['goal_2d'])[:2],
              ns['goal_radius'], (0, size), (0, size))


def do_hardcoded_2d(key, script, xlim, ylim):
    print(f"[{key}] re-running {script} ...")
    ns = run_prefix(str(ROOT / f'trial_scripts/{script}'), HARD_SPLIT)
    env = ns['env']
    render_2d(f"{OUT_DIR}/{key}.png",
              env.static_circular_obstacles, env.static_rectangular_obstacles,
              flat(ns['bboe'].history, 'for_edges'),
              flat(ns['bboe'].history, 'rev_edges'),
              ns['final_path_edges'],
              np.asarray(ns['start'])[:2], np.asarray(ns['goal'])[:2],
              ns['goal_radius'], xlim, ylim)


def do_quadrotor(key, script):
    print(f"[{key}] re-running {script} ...")
    ns = run_prefix(str(ROOT / f'trial_scripts/{script}'), QUAD_SPLIT)
    goal = ns.get('goal_6d', ns.get('goal'))
    render_3d(f"{OUT_DIR}/{key}.png",
              ns['draw_cuboid'], QUAD_OBS,
              flat(ns['bboe'].history, 'for_edges'),
              flat(ns['bboe'].history, 'rev_edges'),
              np.asarray(ns['final_path_xyz']),
              np.asarray(ns['start'])[:3], np.asarray(goal)[:3])


def main():
    for key, diff in DIFFS:
        do_difficulty(key, diff)
    do_hardcoded_2d('unicycle', 'create_bboe_gif.py', (0, 100), (0, 100))
    do_hardcoded_2d('cwt', 'create_bboe_car_with_trailer_gif.py', (0, 20), (0, 20))
    do_hardcoded_2d('treaded', 'create_bboe_treaded_gif.py', (0, 25), (0, 25))
    do_quadrotor('quadrotor', 'create_bboe_quadrotor_gif.py')
    print("All panels rendered.")


if __name__ == "__main__":
    main()
