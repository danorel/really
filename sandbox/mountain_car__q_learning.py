import argparse
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

IMAGE_FILENAME = "mountain_car__q_learning__loss-reward"
MODEL_FILENAME = "mountain_car__q_learning"


def make_continious2discrete_converter(
    discrete_spaces: list[list], discrete_multipliers=[10, 1]
):
    def continious2discreteobservation_converter(
        continious_observations: npt.NDArray,
    ) -> int:
        discrete_index = 0
        discrete_observations = np.array([])
        for discrete_space, continious_observation in zip(
            discrete_spaces, continious_observations
        ):
            discrete_observation = search4discrete(
                discrete_space, continious_observation
            )
            discrete_observations = np.append(
                discrete_observations, discrete_observation
            )
        for discrete_observation, discrete_multiplier in zip(
            discrete_observations, discrete_multipliers
        ):
            discrete_index += discrete_observation * discrete_multiplier
        return int(discrete_index)

    return continious2discreteobservation_converter


def make_fit(continious2discreteobservation_converter, hyperparameters: dict):
    alpha, gamma, reward_speed_coefficient = (
        hyperparameters.get("alpha"),
        hyperparameters.get("gamma"),
        hyperparameters.get("reward_speed_coefficient"),
    )
    assert alpha is not None, "learning rate was not defined"
    assert gamma is not None, "decay rate was not defined"
    assert (
        reward_speed_coefficient is not None
    ), "reward speed coefficient was not defined"

    def fit(sample, q_table):
        old_continious_observation, new_continious_observation, action, reward = sample

        _, new_continious_speed = new_continious_observation
        _, old_continious_speed = old_continious_observation

        reward += reward_speed_coefficient * (
            gamma * abs(new_continious_speed) - abs(old_continious_speed)
        )

        old_discrete_observation, new_discrete_observation = (
            continious2discreteobservation_converter(
                old_continious_observation),
            continious2discreteobservation_converter(
                new_continious_observation),
        )

        old_v = q_table[old_discrete_observation, action]
        new_max_v = np.max(q_table[new_discrete_observation])
        new_v = (1 - alpha) * old_v + alpha * (reward + gamma * new_max_v)
        q_table[old_discrete_observation, action] = new_v

        return reward

    return fit


def explore_or_exploit(
    continious2discreteobservation_converter, old_continious_observation, q_table, eps
):
    if random() > eps:
        old_discrete_observation = continious2discreteobservation_converter(
            old_continious_observation
        )
        action = np.argmax(q_table[old_discrete_observation])
    else:
        action = env.action_space.sample()
    return action


def train(
    q_table: npt.NDArray,
    continious2discreteobservation_converter,
    fit,
    episodes: int,
    steps: int,
    eps_max: float,
    eps_min: float,
    strict: bool = False,
    verbose: bool = False
):
    reward_history = []

    successes = 0

    eps, eps_decay = (eps_max, (eps_max - eps_min) / episodes)

    for episode in range(episodes):
        if verbose:
            print(f"Starting episode {episode + 1}. Epsilon = {eps}")

        episode_reward = 0

        old_continious_observation, _ = env.reset()

        for _ in range(steps):
            action = explore_or_exploit(
                continious2discreteobservation_converter,
                old_continious_observation,
                q_table,
                eps,
            )

            new_continious_observation, reward, terminated, truncated, _ = env.step(
                action
            )

            reward = fit(
                (
                    old_continious_observation,
                    new_continious_observation,
                    action,
                    reward,
                ),
                q_table,
            )

            episode_reward += reward

            if terminated:
                successes += 1
                if verbose:
                    print(f"Flag is captured in episode {episode}!")
                break

            if strict and truncated:
                if verbose:
                    print(f"Timeout in episode {episode}!")
                break

            old_continious_observation = new_continious_observation

        reward_history.append(episode_reward)

        eps -= eps_decay

    env.close()

    report = dict(reward_history=reward_history)

    if verbose:
        print(f"Success rate: {successes} out of {episodes}")

    return successes, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running Q-Learning algorithm on MountainCar problem."
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="enable additional logging during training (default: False)",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--strict",
        type=bool,
        default=False,
        help="enable strict limitation for training agent, 200 steps to achieve the goal (default: False)",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--alpha", type=float, default=0.1, help="learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.95, help="discount rate (default: 0.95)"
    )
    parser.add_argument(
        "--epsilon_min",
        type=float,
        default=0.01,
        help="lowest possible value of epsilon (epsilon-greedy strategy) (default: 0.01)",
    )
    parser.add_argument(
        "--epsilon_max",
        type=float,
        default=0.3,
        help="highest possible value of epsilon (epsilon-greedy strategy) (default: 0.3)",
    )

    parser.add_argument(
        "--episodes", type=int, default=100, help="training episodes (default: 100)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="training steps per episode (default: 1000)",
    )

    parser.add_argument(
        "--reward_speed_coefficient",
        type=float,
        default=300,
        help="reward trick (default: 300)",
    )
    parser.add_argument(
        "--discrete_position_chunks",
        type=int,
        default=100,
        help="number of discrete chunks of position parameter (default: 100)",
    )
    parser.add_argument(
        "--discrete_velocity_chunks",
        type=int,
        default=25,
        help="number of discrete chunks of velocity parameter (default: 25)",
    )

    args = parser.parse_args()

    DISCRETE_POSITION_CHUNKS = args.discrete_position_chunks
    DISCRETE_VELOCITY_CHUNKS = args.discrete_velocity_chunks

    DISCRETE_OBSERVATION_DIMENSIONS = (
        DISCRETE_POSITION_CHUNKS * DISCRETE_VELOCITY_CHUNKS
    )

    MODEL_VERSION = (
        f"position={DISCRETE_POSITION_CHUNKS}xvelocity={DISCRETE_VELOCITY_CHUNKS}"
    )

    IMAGE_PATH = pathlib.Path(
        IMAGE_ASSET_DIR + "/" + IMAGE_FILENAME + "__" + MODEL_VERSION + ".png"
    )
    MODEL_PATH = pathlib.Path(
        MODEL_ASSET_DIR + "/" + MODEL_FILENAME + "__" + MODEL_VERSION
    )

    discrete_position_space = continious2discrete(
        continious_from=-1.2,
        continious_to=0.6,
        discrete_chunks=DISCRETE_POSITION_CHUNKS,
    )
    discrete_velocity_space = continious2discrete(
        continious_from=-0.03,
        continious_to=0.03,
        discrete_chunks=DISCRETE_VELOCITY_CHUNKS,
    )

    continious2discreteobservation_converter = make_continious2discrete_converter(
        discrete_spaces=[
            discrete_position_space,
            discrete_velocity_space,
        ]
    )

    if MODEL_PATH.exists():
        with MODEL_PATH.open(mode="rb") as f:
            q_table = np.load(f)
    else:
        q_table = np.zeros(
            [DISCRETE_OBSERVATION_DIMENSIONS, DISCRETE_ACTION_DIMENSIONS]
        )

    fit = make_fit(
        continious2discreteobservation_converter,
        hyperparameters=dict(
            alpha=args.alpha,
            gamma=args.gamma,
            reward_speed_coefficient=args.reward_speed_coefficient,
        ),
    )

    successes, report = train(
        q_table,
        continious2discreteobservation_converter,
        fit,
        episodes=args.episodes,
        steps=args.steps,
        eps_max=args.epsilon_max,
        eps_min=args.epsilon_min,
        strict=args.strict,
        verbose=args.verbose,
    )

    with MODEL_PATH.open(mode="wb") as f:
        np.save(f, q_table)

    plt.plot(report.get("reward_history"), label="Reward history")
    plt.legend()
    plt.savefig(IMAGE_PATH)
