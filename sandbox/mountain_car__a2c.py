import argparse
import os
import operator
import gym
import numpy as np
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions.categorical import Categorical

from environments.mountain_car import env, OBSERVATION_DIMENSIONS, ACTION_DIMENSIONS
from sandbox.runner import AbstractRunner


DIR = "reports/mountain-car/a2c"
report = pathlib.Path(DIR)
report.mkdir(parents=True, exist_ok=True)

loss_reward_file, model_file = (
    report / 'loss-reward',
    report / 'model'
)


class A2C(nn.Module):
    def __init__(self, obs_space: int, hid_space: int, act_space: int):
        super(A2C).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_space, hid_space),
            nn.ReLU(),
            nn.Linear(hid_space, act_space),
            nn.Softmax()
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_space, hid_space),
            nn.ReLU(),
            nn.Linear(hid_space, 1)
        )

    def forward(self, obs):
        probs = self.actor(obs)
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)


class A2CRunner(AbstractRunner):
    def optimize(self):
        pass

    def learn(self):
        agent = A2C(obs_space=OBSERVATION_DIMENSIONS,
                    hid_space=args.hidden_dimensions,
                    act_space=ACTION_DIMENSIONS)
        agent.device(self.args.device)

        optimizer = optim.Adam(lr=args.lr)

        envs = gym.VectorEnv(args.N)

        for _ in range(1, args.total_timesteps // (args.N * args.M)):
            self._reinforce(envs,
                            agent,
                            optimizer,
                            args.M,
                            args.gamma)

    def _reinforce(self,
                   envs: gym.vector.VectorEnv,
                   agent: A2C,
                   optimizer: optim.Adam,
                   M: int,
                   gamma: float):
        # Setup initial state for environment
        observations = envs.reset()

        # Gain experiences from environment via actor policy
        mean_action_log_probabilities, mean_rewards = np.array(
            []), np.array([])
        for _ in range(M):
            action_returns = [agent.forward(observation)
                              for observation in observations]
            actions, actions_log_probabilities = (
                map(operator.itemgetter(0), action_returns),
                map(operator.itemgetter(1), action_returns)
            )
            observations, rewards, terminations, truncations, infos = envs.step(
                actions)
            mean_action_log_probabilities = np.append(
                mean_action_log_probabilities, np.mean(actions_log_probabilities))
            mean_rewards = np.append(mean_rewards, np.mean(rewards))

        # Compute expected returns / cumulative rewards
        T = len(mean_rewards)
        expected_returns = np.array([0] * T)
        expected_returns[T - 1] = mean_rewards[T - 1]
        cumulated_gamma = gamma
        for t in range(T - 2, -1, -1):
            expected_returns[t] = mean_rewards[t] * \
                cumulated_gamma + expected_returns[t + 1]
            cumulated_gamma *= gamma
        eps = np.finfo(np.float32).eps
        expected_returns = (
            expected_returns - expected_returns.mean()) / (expected_returns.std() + eps)

        # Compute loss
        loss = np.array([])
        for expected_return, action_log_probability in zip(expected_returns, mean_action_log_probabilities):
            loss = np.append(-action_log_probability * expected_return)
        loss = torch.from_numpy(loss).sum()

        # Optimize policy parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running A2C algorithm on MountainCar problem."
    )
    parser.add_argument(
        '-o',
        '--optimize',
        action='store_false'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_false'
    )
    parser.add_argument(
        '-d',
        '--device',
        default='cpu'
    )
    parser.add_argument(
        '-t',
        '--timestamps',
        type=int,
        default=5000,
        help="Number of timestamps during learning phase"
    )
    parser.add_argument(
        '-N',
        '--N',
        type=int,
        default=os.cpu_count(),
        help="Number of agents simultaneously learning",
    )
    parser.add_argument(
        '-M',
        '--M',
        type=int,
        default=500,
        help="Number of steps agent does during learning per iteration",
    )
    parser.add_argument(
        '-hd',
        '--hidden_dimensions',
        type=int,
        default=500,
        help="Number of hidden dimensions of shared A2C hidden layer architecture"
    )
    parser.add_argument(
        '-g',
        '--gamma',
        type=float,
        default=0.99,
        help="Reward discount factor"
    )
    parser.add_argument(
        '-l',
        '--lr',
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    args = parser.parse_args()
    runner = A2CRunner(args)
