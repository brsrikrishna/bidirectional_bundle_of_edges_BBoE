import numpy as np

from Environments import SquareEnvironment, CircularObstacle2D
from Agents import UniCycle
from edge_bundle import EdgeBundle
from kd_tree_unicycle import CircularAngleIndexNumba


def get_unicycle_agent(agent_id):
    agent = UniCycle(
                agent_id = agent_id,
                max_speed = 5.0,
                max_omega = np.pi / 2, # ~1.57
                radius = 0.2,
                rng_seed= 77 + agent_id
                )
    return agent
