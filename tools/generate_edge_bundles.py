"""Regenerate the edge bundles used in the paper.

Every bundle in ``edge_bundles/`` was produced by this recipe with the seed
listed below, so the shipped files can be reproduced from scratch:

  name                          edges   seed  notes
  unicycle_discrete_190           190     -   GBRRT maneuver library (deterministic, see generate_discrete_eb.py)
  unicycle_random_duration_1000  1000    42   same discrete actions, random 0.1-1.0 s durations (ablations)
  treaded                      100000     7   start states uniform in wheel speeds (10% at the bounds)
  cwt                          100000     7   uniform in speed, steering angle (10% at bounds) and trailer angle (5%)
  quadrotor                    200000     7   uniform in velocity (10% at the bounds)

The treaded, unicycle and quadrotor bundles regenerate bit-for-bit. The shipped
CWT bundle predates the trailer-angle wrap now applied in the dynamics, so a
regenerated CWT bundle differs from the shipped file only by multiples of 2*pi
in the trailer angle of about 9% of the stored trajectories (same physical
states, identical positions, headings and velocities). Use ``--verify`` to
compare a regenerated bundle against the shipped file.

Examples
  python tools/generate_edge_bundles.py --bundle treaded --verify          # ~1 min, compares to the shipped file
  python tools/generate_edge_bundles.py --bundle quadrotor --num-edges 1000 --out /tmp/q.npz
  python tools/generate_edge_bundles.py --all                               # rewrites every bundle in edge_bundles/
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "tools"))

from edge_bundle import GenerateEdgeBundle                      # noqa: E402
from Environments import SquareEnvironment, CuboidEnvironment  # noqa: E402

BUNDLES = {
    "unicycle_discrete_190": "eb_unicycle_discrete_190.npz",
    "unicycle_random_duration_1000": "eb_unicycle_random_duration_1000.npz",
    "treaded": "eb_treaded_kinodynamic_TI_edges_100000.npz",
    "cwt": "eb_car_with_trailer_kinodynamic_TI_edges_100000.npz",
    "quadrotor": "eb_quadcopter6d_kinodynamic_TI_edges_200000.npz",
}
DEFAULT_EDGES = {"unicycle_random_duration_1000": 1000, "treaded": 100000, "cwt": 100000, "quadrotor": 200000}
DEFAULT_SEED = {"unicycle_random_duration_1000": 42, "treaded": 7, "cwt": 7, "quadrotor": 7}


def _sample_with_extremes(rng, low, high, p_extreme):
    if rng.random() < p_extreme:
        return low if rng.random() < 0.5 else high
    return rng.uniform(low, high)


def gen_kinodynamic(name, num_edges, seed):
    """Translation-invariant bundles: pose at the origin, kinodynamic components sampled."""
    if name == "treaded":
        from mapf_env_square_agent_treaded import get_treaded_agent
        env, agent = SquareEnvironment(100, 100, []), get_treaded_agent(1)

        def start(env, agent, rng):
            s = agent.max_speed
            return np.array([0.0, 0.0, 0.0, _sample_with_extremes(rng, -s, s, 0.1),
                             _sample_with_extremes(rng, -s, s, 0.1)], dtype=np.float64)
    elif name == "cwt":
        from mapf_env_square_agent_car_with_trailer import get_car_with_trailer_agent
        env, agent = SquareEnvironment(1000, 1000, [], obs_buffers=False), get_car_with_trailer_agent(1)

        def start(env, agent, rng):
            return np.array([0.0, 0.0, 0.0,
                             _sample_with_extremes(rng, -agent.max_speed, agent.max_speed, 0.1),
                             _sample_with_extremes(rng, -agent.max_omega, agent.max_omega, 0.1),
                             _sample_with_extremes(rng, -np.pi, np.pi, 0.05)], dtype=np.float64)
    elif name == "quadrotor":
        from mapf_env_cuboid_agent_quadcopter6d import get_quadcopter_agent
        env, agent = CuboidEnvironment(40, 40, 40, []), get_quadcopter_agent(1)

        def start(env, agent, rng):
            s = agent.max_speed
            return np.array([0.0, 0.0, 0.0] + [_sample_with_extremes(rng, -s, s, 0.1) for _ in range(3)])
    else:
        raise ValueError(name)
    gen = GenerateEdgeBundle(env=env, agent=agent, num_edges=num_edges, get_start_state=start,
                             minimum_time_step=0.1, max_sample_T=1.0, rng_seed=seed)
    return gen.generate_edge_bundle()


def gen_unicycle_random_duration(num_edges, seed):
    """Discrete (v, omega) actions of the GBRRT library applied for random durations of 0.1-1.0 s."""
    from mapf_env_square_agent_unicycle import get_unicycle_agent
    v_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    w_vals = [i * np.pi / 18.0 for i in range(10)]
    acts = []
    for v in v_vals:
        for w in w_vals:
            acts.append((v, w))
            if v != 0:
                acts.append((-v, w))
            if w != 0:
                acts.append((v, -w))
            if v != 0 and w != 0:
                acts.append((-v, -w))
    actions_grid = sorted(set(acts))
    min_dt, max_steps = 0.1, 10
    agent = get_unicycle_agent(1)
    rng = np.random.default_rng(seed)
    start_states = np.zeros((num_edges, 3))
    final_states = np.empty((num_edges, 3))
    actions = np.empty((num_edges, 2))
    timesteps = np.empty(num_edges)
    trajectories = np.full((num_edges, max_steps, 3), np.nan)
    lengths = np.empty(num_edges, dtype=np.int32)
    for i in range(num_edges):
        action = np.array(actions_grid[rng.integers(0, len(actions_grid))])
        n_steps = int(rng.integers(1, max_steps + 1))
        duration = n_steps * min_dt
        nxt, path = agent.get_next_state(start_states[i], action, duration, num_steps=n_steps)
        actions[i], timesteps[i], final_states[i] = action, duration, nxt
        trajectories[i, :n_steps], lengths[i] = path, n_steps
    return dict(start_states=start_states, final_states=final_states, actions=actions,
                timesteps=timesteps, trajectories=trajectories, trajectory_lengths=lengths)


def generate(name, num_edges=None, seed=None):
    if name == "unicycle_discrete_190":
        import generate_discrete_eb
        return generate_discrete_eb.generate_discrete_edges(save=False)
    num_edges = num_edges or DEFAULT_EDGES[name]
    seed = DEFAULT_SEED[name] if seed is None else seed
    if name == "unicycle_random_duration_1000":
        return gen_unicycle_random_duration(num_edges, seed)
    return gen_kinodynamic(name, num_edges, seed)


def verify(name, eb):
    shipped = np.load(ROOT / "edge_bundles" / BUNDLES[name])
    n = len(eb["timesteps"])
    report = {}
    for k in eb:
        if k not in shipped.files:
            continue
        a, b = np.asarray(eb[k])[:n], np.asarray(shipped[k])[:n]
        if a.ndim == 3:
            b = b[:, :a.shape[1]]
        report[k] = bool(np.allclose(a, b, equal_nan=True))
    if name == "cwt" and not report.get("trajectories", True):
        a, b = eb["trajectories"][:n], shipped["trajectories"][:n, :eb["trajectories"].shape[1]]
        d = np.abs(a - b)
        d[np.isnan(d)] = 0
        d[..., 5] = np.minimum(d[..., 5], np.abs(d[..., 5] - 2 * np.pi))   # trailer angle modulo 2*pi
        report["trajectories (mod 2*pi in trailer angle)"] = bool(d.max() < 1e-9)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", choices=list(BUNDLES), help="which bundle to generate")
    ap.add_argument("--all", action="store_true", help="generate every bundle with its default size and seed")
    ap.add_argument("--num-edges", type=int, help="override the number of edges (default: as in the paper)")
    ap.add_argument("--seed", type=int, help="override the RNG seed (default: as in the paper)")
    ap.add_argument("--out", help="output .npz (default: the shipped path in edge_bundles/)")
    ap.add_argument("--verify", action="store_true", help="compare the result with the shipped bundle instead of saving")
    args = ap.parse_args()
    names = list(BUNDLES) if args.all else ([args.bundle] if args.bundle else ap.error("give --bundle or --all"))
    for name in names:
        eb = generate(name, args.num_edges, args.seed)
        if args.verify:
            print(f"{name}: compared first {len(eb['timesteps'])} edges with the shipped file ->", verify(name, eb))
        else:
            out = Path(args.out) if (args.out and not args.all) else ROOT / "edge_bundles" / BUNDLES[name]
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, **eb)
            print(f"{name}: wrote {out} ({len(eb['timesteps'])} edges)")


if __name__ == "__main__":
    main()
