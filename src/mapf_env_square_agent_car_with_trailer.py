import numpy as np

from Agents.CarWithTrailer import CarWithTrailer


def get_car_with_trailer_agent(agent_id):
    agent = CarWithTrailer(
        agent_id=agent_id,
        max_speed=4.0,              # 2x: was 2.0
        max_linear_acceleration=4.0, # 4x: was 1.0
        max_omega=np.pi / 3,         # unchanged (steering-angle limit)
        max_angular_acceleration=2.0, # 4x: was 0.5
        radius=0.4,
        rng_seed=77 + agent_id,
    )
    return agent
