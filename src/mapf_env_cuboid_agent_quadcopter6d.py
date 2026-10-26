import numpy as np

from Agents import QuadCopter6D


def get_quadcopter_agent(agent_id):
    return QuadCopter6D(
        agent_id=agent_id,
        max_speed=0.5,
        max_acceleration=2.0,
        radius=0.25,
        rng_seed=agent_id + 77,
    )
