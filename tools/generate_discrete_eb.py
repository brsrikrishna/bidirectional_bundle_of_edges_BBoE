import sys
import numpy as np
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))
from mapf_env_square_agent_unicycle import get_unicycle_agent
from utils import wrap_between_0_and_2pi

def trapezoidal_velocity_profile(v_peak, w_peak, T, num_steps):
    """
    Generate a trapezoidal velocity profile for a unicycle maneuver.

    The velocity ramps from 0 to peak over [0, T/4],
    holds at peak over [T/4, 3T/4],
    and ramps back to 0 over [3T/4, T].

    This matches the GBRRT paper (Nayak & Otte, 2022) where maneuvers
    are designed with zero terminal linear and angular velocities using
    a time-varying nonlinear controller (ref [38]: Carona et al.).
    """
    dt = T / num_steps
    t_ramp = T / 4.0

    trajectory = np.zeros((num_steps, 3), dtype=np.float64)
    x, y, theta = 0.0, 0.0, 0.0

    for i in range(num_steps):
        t = (i + 0.5) * dt  # midpoint of timestep for better integration

        # Trapezoidal profile
        if t < t_ramp:
            # Ramp up
            scale = t / t_ramp
        elif t < T - t_ramp:
            # Hold
            scale = 1.0
        else:
            # Ramp down
            scale = (T - t) / t_ramp

        v = v_peak * scale
        w = w_peak * scale

        # Euler integration
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt

        trajectory[i, 0] = x
        trajectory[i, 1] = y
        trajectory[i, 2] = wrap_between_0_and_2pi(theta)

    final_state = trajectory[-1].copy()
    return final_state, trajectory


def generate_discrete_edges(save=True):
    agent = get_unicycle_agent(1)

    # Paper Table I & Section VIII-A-1:
    # Linear velocity: 1 to 5 m/s in 1 m/s increments
    # Angular velocity: 0 to pi/2 rad/s in pi/18 rad/s increments
    # "in all four quadrants" -> positive/negative combinations of v and w
    v_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    w_vals = [i * np.pi / 18.0 for i in range(10)]

    actions = []
    for v in v_vals:
        for w in w_vals:
            actions.append((v, w))
            if v != 0: actions.append((-v, w))
            if w != 0: actions.append((v, -w))
            if v != 0 and w != 0: actions.append((-v, -w))

    discrete_actions = sorted(set(actions))

    N = len(discrete_actions)
    T = 1.0
    num_steps = 10  # matches minimum_time_step=0.1

    print(f"Generating {N} maneuvers with trapezoidal velocity profiles (zero terminal velocities)...")

    states = np.zeros((N, 3), dtype=np.float64)       # all start at origin
    out_final_states = np.zeros((N, 3), dtype=np.float64)
    out_actions = np.zeros((N, 2), dtype=np.float64)   # peak (v, w) for reference
    out_timesteps = np.full(N, T, dtype=np.float64)
    out_trajectories = np.zeros((N, num_steps, 3), dtype=np.float64)
    out_lengths = np.full(N, num_steps, dtype=np.int32)

    for i, (v_peak, w_peak) in enumerate(discrete_actions):
        final_state, trajectory = trapezoidal_velocity_profile(v_peak, w_peak, T, num_steps)
        out_final_states[i] = final_state
        out_actions[i] = np.array([v_peak, w_peak])
        out_trajectories[i] = trajectory

    filename = str(ROOT / 'edge_bundles/eb_unicycle_discrete_190.npz')
    bundle = dict(start_states=states,
        final_states=out_final_states,
        actions=out_actions,
        timesteps=out_timesteps,
        trajectories=out_trajectories,
        trajectory_lengths=out_lengths
    )
    if save:
        np.savez(filename, **bundle)
    print(f"Saved {N} maneuvers to {filename}")

    # Print some stats
    displacements = np.sqrt(out_final_states[:, 0]**2 + out_final_states[:, 1]**2)
    print(f"  Max displacement: {displacements.max():.2f}")
    print(f"  Min displacement: {displacements.min():.2f}")
    print(f"  Mean displacement: {displacements.mean():.2f}")
    return bundle


if __name__ == "__main__":
    generate_discrete_edges()
