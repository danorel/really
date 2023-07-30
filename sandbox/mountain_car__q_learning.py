import os.path

import numpy as np
import numpy.typing as npt

from random import random

from environments.mountain_car import env


def continious2discrete(
    continious_from: float, continious_to: float, discrete_chunks: int
):
    discrete_space = []
    chunk_margin = abs(continious_to - continious_from) / discrete_chunks
    chunk_from = continious_from
    for _ in range(discrete_chunks):
        chunk_to = chunk_from + chunk_margin
        discrete_space.append((chunk_from, chunk_to))
        chunk_from = chunk_to
    return discrete_space


def search4discrete(discrete_space: list, continious_observation: float):
    lo, hi = 0, len(discrete_space) - 1
    while lo <= hi:
        mid = lo + round((hi - lo) / 2)
        observation_limit_from, observation_limit_to = discrete_space[mid]
        if continious_observation > observation_limit_to:
            lo = mid + 1
        elif continious_observation < observation_limit_from:
            hi = mid - 1
        else:
            return mid
    return -1


def get_discrete_state(
    continious_states: npt.NDArray,
    discrete_spaces: list[list],
    discrete_multipliers=[10, 1],
) -> int:
    discrete_state = 0
    discrete_values = np.array([])

    for discrete_space, continious_observation in zip(
        discrete_spaces, continious_states
    ):
        discrete_values = np.append(
            discrete_values, search4discrete(
                discrete_space, continious_observation)
        )

    for discrete_value, discrete_multiplier in zip(
        discrete_values, discrete_multipliers
    ):
        discrete_state += discrete_value * discrete_multiplier

    return int(discrete_state)


if __name__ == "__main__":
    pickle = 'mountain_car__q_learning.npy'

    episodes = 30

    steps = 1500
    alpha = 0.1
    gamma = 0.95

    eps = 0.5
    eps_decay = 0.01
    eps_limit = 0.01

    position_dimensions = 25
    velocity_dimensions = 10

    discrete_observation_space = position_dimensions * velocity_dimensions
    discrete_action_space = 3

    observation_position_space = continious2discrete(
        continious_from=-1.2, continious_to=0.6, discrete_chunks=position_dimensions
    )
    observation_velocity_space = continious2discrete(
        continious_from=-0.03, continious_to=0.03, discrete_chunks=velocity_dimensions
    )

    if os.path.exists(pickle):
        with open(pickle, 'rb') as f:
            q_table = np.load(f)
    else:
        q_table = np.zeros([discrete_observation_space, discrete_action_space])

    initial_state = env.reset()

    for episode in range(episodes):
        old_state = initial_state[0]

        for step in range(steps):
            old_discrete_state = get_discrete_state(
                old_state,
                discrete_spaces=[
                    observation_position_space,
                    observation_velocity_space,
                ],
            )

            if step % 100 == 0:
                print(q_table)

            if random() > eps:
                action = np.argmax(q_table[old_discrete_state])
            else:
                action = env.action_space.sample()

            new_state, reward, terminated, truncated, info = env.step(action)

            new_discrete_state = get_discrete_state(
                new_state,
                discrete_spaces=[
                    observation_position_space,
                    observation_velocity_space,
                ],
            )

            old_value = q_table[old_discrete_state, action]
            new_max = np.max(q_table[new_discrete_state])
            new_value = (1 - alpha) * old_value + \
                alpha * (reward + gamma * new_max)
            q_table[old_discrete_state, action] = new_value

            if terminated:
                print(f"Flag is captured in episode {episode}!")

            old_state = new_state

        eps = max(eps_limit, eps - eps_decay)
        print(f"Epsilon after decay: {eps}")

    env.close()

    with open(pickle, 'wb') as f:
        np.save(f, q_table)
