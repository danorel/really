import pathlib
import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt

from random import random

from environments.mountain_car import env
from utils.continious2discrete import continious2discrete, search4discrete


def get_discrete_state(
    continious_states: npt.NDArray,
    discrete_spaces: list[list],
    discrete_multipliers=[10, 1],
) -> int:
    discrete_index = 0
    discrete_observations = np.array([])

    for discrete_space, continious_observation in zip(
        discrete_spaces, continious_states
    ):
        discrete_observation = search4discrete(
            discrete_space, continious_observation)
        discrete_observations = np.append(
            discrete_observations, discrete_observation
        )

    for discrete_observation, discrete_multiplier in zip(
        discrete_observations, discrete_multipliers
    ):
        discrete_index += discrete_observation * discrete_multiplier

    return int(discrete_index)


if __name__ == "__main__":
    IMAGE_ASSET_DIR = "images"
    MODEL_ASSET_DIR = "models"

    pathlib.Path(IMAGE_ASSET_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(MODEL_ASSET_DIR).mkdir(parents=True, exist_ok=True)

    IMAGE_FILENAME = 'loss-reward'
    MODEL_FILENAME = 'mountain_car__q_learning'

    episodes = 30
    steps = 1500

    eps = 0.5
    eps_decay = 0.01
    eps_limit = 0.01

    alpha = 0.1
    gamma = 0.95

    discrete_position_chunks = 25
    discrete_velocity_chunks = 10

    discrete_position_space = continious2discrete(
        continious_from=-1.2, continious_to=0.6, discrete_chunks=discrete_position_chunks)
    discrete_velocity_space = continious2discrete(
        continious_from=-0.03, continious_to=0.03, discrete_chunks=discrete_velocity_chunks)

    discrete_observation_dimensions = discrete_position_chunks * discrete_velocity_chunks
    discrete_action_dimensions = 3

    version = f"observations={discrete_observation_dimensions}"

    image_path = pathlib.Path(
        IMAGE_ASSET_DIR + "/" + IMAGE_FILENAME + "__" + version + ".png")
    model_path = pathlib.Path(
        MODEL_ASSET_DIR + "/" + MODEL_FILENAME + "__" + version)

    if model_path.exists():
        with model_path.open(mode='rb') as f:
            q_table = np.load(f)
    else:
        q_table = np.zeros(
            [discrete_observation_dimensions, discrete_action_dimensions])

    reward_history = []

    successes = 0

    for episode in range(episodes):
        print(f"Starting episode {episode}...")

        episode_reward = 0

        old_state, _ = env.reset()

        for step in range(steps):
            old_discrete_state = get_discrete_state(
                old_state,
                discrete_spaces=[
                    discrete_position_space,
                    discrete_velocity_space,
                ],
            )

            if random() > eps:
                action = np.argmax(q_table[old_discrete_state])
            else:
                action = env.action_space.sample()

            new_state, reward, terminated, truncated, info = env.step(action)

            new_discrete_state = get_discrete_state(
                new_state,
                discrete_spaces=[
                    discrete_position_space,
                    discrete_velocity_space,
                ],
            )

            old_value = q_table[old_discrete_state, action]
            new_max = np.max(q_table[new_discrete_state])
            new_value = (1 - alpha) * old_value + \
                alpha * (reward + gamma * new_max)
            q_table[old_discrete_state, action] = new_value

            episode_reward += reward

            if terminated:
                successes += 1
                eps = max(eps_limit, eps - eps_decay)
                print(f"Flag is captured in episode {episode}!")
                print(f"Epsilon after decay: {eps}")
                break
            else:
                old_state = new_state

        reward_history.append(episode_reward)

    env.close()

    print(f"Success rate: {successes} out of {episodes}")

    with model_path.open(mode='wb') as f:
        np.save(f, q_table)

    plt.plot(reward_history, label="Reward history")
    plt.legend()
    plt.savefig(image_path)
