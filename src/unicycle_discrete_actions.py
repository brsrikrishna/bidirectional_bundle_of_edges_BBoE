"""
Discrete unicycle action primitives used by the unicycle benchmark and GIF
scripts. Matches the ~400-primitive set the edge bundle was generated from
so that `agent.get_random_action` draws from the same distribution as the
maneuver library.
"""
import numpy as np


def _build_discrete_actions():
    v_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    w_vals = [i * np.pi / 18.0 for i in range(10)]
    actions = []
    for v in v_vals:
        for w in w_vals:
            actions.append((v, w))
            if v != 0:
                actions.append((-v, w))
            if w != 0:
                actions.append((v, -w))
            if v != 0 and w != 0:
                actions.append((-v, -w))
    return list(set(actions))


DISCRETE_ACTIONS = _build_discrete_actions()


def apply_discrete_random_action(agent):
    """
    Patch `agent.get_random_action` so it draws from DISCRETE_ACTIONS. Call
    this once after constructing the unicycle agent.
    """
    def get_discrete_random_action(rng):
        idx = rng.integers(0, len(DISCRETE_ACTIONS))
        return np.array(DISCRETE_ACTIONS[idx])
    agent.get_random_action = get_discrete_random_action
    return agent
