import argparse
import typing
import pathlib
import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt

from random import random

from environments.mountain_car import env
from utils.continious2discrete import continious2discrete, search4discrete

DISCRETE_ACTION_DIMENSIONS = 3

IMAGE_ASSET_DIR = "images"
MODEL_ASSET_DIR = "models"

pathlib.Path(IMAGE_ASSET_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(MODEL_ASSET_DIR).mkdir(parents=True, exist_ok=True)

IMAGE_FILENAME = 'mountain_car__q_learning__loss-reward'
MODEL_FILENAME = 'mountain_car__q_learning'


def make_continious2discrete_converter(discrete_spaces: list[list], discrete_multipliers=[10, 1]):
    def continious2discrete_state_converter(continious_states: npt.NDArray) -> int:
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
    return continious2discrete_state_converter


def run(q_table: npt.NDArray,
        continious2discrete_state_converter: typing.Callable[[npt.NDArray], int],
        episodes: int,
        steps: int,
        eps_max: float,
        eps_min: float,
        alpha: float,
        gamma: float,
        speed_reward_factor: float
        ):
    reward_history = []

    successes = 0

    eps, eps_decay = (
        eps_max,
        (eps_max - eps_min) / episodes
    )

    for episode in range(episodes + 1):
        print(f"Starting episode {episode}. Epsilon = {eps}")

        episode_reward = 0

        old_continious_state, _ = env.reset()

        for _ in range(steps):
            old_discrete_state = continious2discrete_state_converter(
                old_continious_state)

            if random() > eps:
                action = np.argmax(q_table[old_discrete_state])
            else:
                action = env.action_space.sample()

            new_continious_state, reward, terminated, _, _ = env.step(action)

            _, new_continious_speed = new_continious_state
            _, old_continious_speed = old_continious_state

            reward += speed_reward_factor * \
                (gamma * abs(new_continious_speed) - abs(old_continious_speed))

            new_discrete_state = continious2discrete_state_converter(
                new_continious_state)

            old_value = q_table[old_discrete_state, action]
            new_max = np.max(q_table[new_discrete_state])
            new_value = (1 - alpha) * old_value + \
                alpha * (reward + gamma * new_max)
            q_table[old_discrete_state, action] = new_value

            episode_reward += reward

            if terminated:
                successes += 1
                print(f"Flag is captured in episode {episode}!")
                break
            else:
                old_continious_state = new_continious_state

        reward_history.append(episode_reward)

        eps -= eps_decay

    env.close()

    report = dict(reward_history=reward_history)

    print(f"Success rate: {successes} out of {episodes}")

    return successes, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Running Q-Learning algorithm on MountainCar problem.')

    parser.add_argument('--alpha', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='discount rate (default: 0.95)')
    parser.add_argument('--epsilon_min', type=float, default=0.01,
                        help='lowest possible value of epsilon (epsilon-greedy strategy) (default: 0.01)')
    parser.add_argument('--epsilon_max', type=float, default=0.3,
                        help='highest possible value of epsilon (epsilon-greedy strategy) (default: 0.3)')

    parser.add_argument('--episodes', type=int, default=100,
                        help='training episodes (default: 100)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='training steps per episode (default: 1000)')

    parser.add_argument('--speed_reward_factor', type=float,
                        default=300, help='reward trick (default: 300)')
    parser.add_argument('--discrete_position_chunks', type=int, default=100,
                        help='number of discrete chunks of position parameter (default: 100)')
    parser.add_argument('--discrete_velocity_chunks', type=int, default=25,
                        help='number of discrete chunks of velocity parameter (default: 25)')

    args = parser.parse_args()

    DISCRETE_POSITION_CHUNKS = args.discrete_position_chunks
    DISCRETE_VELOCITY_CHUNKS = args.discrete_velocity_chunks

    DISCRETE_OBSERVATION_DIMENSIONS = DISCRETE_POSITION_CHUNKS * DISCRETE_VELOCITY_CHUNKS

    MODEL_VERSION = f"position={DISCRETE_POSITION_CHUNKS}xvelocity={DISCRETE_VELOCITY_CHUNKS}"

    IMAGE_PATH = pathlib.Path(IMAGE_ASSET_DIR + "/" +
                              IMAGE_FILENAME + "__" + MODEL_VERSION + ".png")
    MODEL_PATH = pathlib.Path(MODEL_ASSET_DIR + "/" +
                              MODEL_FILENAME + "__" + MODEL_VERSION)

    discrete_position_space = continious2discrete(
        continious_from=-1.2, continious_to=0.6, discrete_chunks=DISCRETE_POSITION_CHUNKS)
    discrete_velocity_space = continious2discrete(
        continious_from=-0.03, continious_to=0.03, discrete_chunks=DISCRETE_VELOCITY_CHUNKS)

    continious2discrete_state_converter = make_continious2discrete_converter(discrete_spaces=[
        discrete_position_space,
        discrete_velocity_space,
    ])

    if MODEL_PATH.exists():
        with MODEL_PATH.open(mode='rb') as f:
            q_table = np.load(f)
    else:
        q_table = np.zeros(
            [DISCRETE_OBSERVATION_DIMENSIONS, DISCRETE_ACTION_DIMENSIONS])

    successes, report = run(q_table,
                            continious2discrete_state_converter,
                            episodes=args.episodes,
                            steps=args.steps,
                            eps_max=args.epsilon_max,
                            eps_min=args.epsilon_min,
                            alpha=args.alpha,
                            gamma=args.gamma,
                            speed_reward_factor=args.speed_reward_factor)

    with MODEL_PATH.open(mode='wb') as f:
        np.save(f, q_table)

    plt.plot(report.get('reward_history'), label="Reward history")
    plt.legend()
    plt.savefig(IMAGE_PATH)
