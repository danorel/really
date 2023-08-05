import pathlib
import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt

from random import random

from environments.mountain_car import env
from utils.continious2discrete import continious2discrete, search4discrete


EPISODES = 200
STEPS = 1500

EPS_MAX = 0.2
EPS_MIN = 0.01

ALPHA = 0.1
GAMMA = 0.95
SPEED_REWARD_FACTOR = 300

DISCRETE_POSITION_CHUNKS = 250
DISCRETE_VELOCITY_CHUNKS = 100

DISCRETE_OBSERVATION_DIMENSIONS = DISCRETE_POSITION_CHUNKS * DISCRETE_VELOCITY_CHUNKS
DISCRETE_ACTION_DIMENSIONS = 3

IMAGE_ASSET_DIR = "images"
MODEL_ASSET_DIR = "models"

pathlib.Path(IMAGE_ASSET_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(MODEL_ASSET_DIR).mkdir(parents=True, exist_ok=True)

IMAGE_FILENAME = 'mountain_car__q_learning__loss-reward'
MODEL_FILENAME = 'mountain_car__q_learning'
MODEL_VERSION = f"position={DISCRETE_POSITION_CHUNKS}xvelocity={DISCRETE_VELOCITY_CHUNKS}"

IMAGE_PATH = pathlib.Path(IMAGE_ASSET_DIR + "/" +
                          IMAGE_FILENAME + "__" + MODEL_VERSION + ".png")
MODEL_PATH = pathlib.Path(MODEL_ASSET_DIR + "/" +
                          MODEL_FILENAME + "__" + MODEL_VERSION)


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
    discrete_position_space = continious2discrete(
        continious_from=-1.2, continious_to=0.6, discrete_chunks=DISCRETE_POSITION_CHUNKS)
    discrete_velocity_space = continious2discrete(
        continious_from=-0.03, continious_to=0.03, discrete_chunks=DISCRETE_VELOCITY_CHUNKS)

    if MODEL_PATH.exists():
        with MODEL_PATH.open(mode='rb') as f:
            q_table = np.load(f)
    else:
        q_table = np.zeros(
            [DISCRETE_OBSERVATION_DIMENSIONS, DISCRETE_ACTION_DIMENSIONS])

    reward_history = []

    successes = 0

    eps, eps_decay = (
        EPS_MAX,
        (EPS_MAX - EPS_MIN) / EPISODES
    )

    for episode in range(EPISODES):
        print(f"Starting episode {episode}. Epsilon = {eps}")

        episode_reward = 0

        old_state, _ = env.reset()

        for step in range(STEPS):
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

            _, new_speed = new_state
            _, old_speed = old_state

            reward += SPEED_REWARD_FACTOR * \
                (GAMMA * abs(new_speed) - abs(old_speed))

            new_discrete_state = get_discrete_state(
                new_state,
                discrete_spaces=[
                    discrete_position_space,
                    discrete_velocity_space,
                ],
            )

            old_value = q_table[old_discrete_state, action]
            new_max = np.max(q_table[new_discrete_state])
            new_value = (1 - ALPHA) * old_value + \
                ALPHA * (reward + GAMMA * new_max)
            q_table[old_discrete_state, action] = new_value

            episode_reward += reward

            if terminated:
                successes += 1
                print(f"Flag is captured in episode {episode}!")
                break
            else:
                old_state = new_state

        reward_history.append(episode_reward)

        eps -= eps_decay

    env.close()

    print(f"Success rate: {successes} out of {EPISODES}")

    with MODEL_PATH.open(mode='wb') as f:
        np.save(f, q_table)

    plt.plot(reward_history, label="Reward history")
    plt.legend()
    plt.savefig(IMAGE_PATH)
