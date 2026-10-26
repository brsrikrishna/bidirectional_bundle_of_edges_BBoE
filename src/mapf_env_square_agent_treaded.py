import numpy as np

from Agents.TreadedVehicle import TreadedVehicle


def get_treaded_agent(agent_id):
    agent = TreadedVehicle(
        agent_id=agent_id,
        max_speed=5.0,
        max_acceleration=10.0,
        track_width=1.0,
        radius=0.4,
        rng_seed=77 + agent_id,
    )
    return agent
