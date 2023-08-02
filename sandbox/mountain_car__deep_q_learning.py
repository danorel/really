import torch
import pathlib
import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt

from random import random
from torch.autograd import Variable

from environments.mountain_car import env
from utils.continious2discrete import continious2discrete, search4discrete


def get_discrete_state(
    continious_states: npt.NDArray,
    discrete_spaces: list[list],
    discrete_multipliers=[10, 1],
) -> Variable:
    discrete_observations = np.array([])
    for discrete_space, continious_observation in zip(
        discrete_spaces, continious_states
    ):
        discrete_observation = search4discrete(
            discrete_space, continious_observation)
        discrete_observations = np.append(
            discrete_observations, discrete_observation
        )

    discrete_index = 0
    for discrete_observation, discrete_multiplier in zip(
        discrete_observations, discrete_multipliers
    ):
        discrete_index += discrete_observation * discrete_multiplier
    discrete_index = int(discrete_index)

    discrete_shape = (np.array([len(discrete_space)
                                for discrete_space in discrete_spaces]).prod(),)
    discrete_array = np.zeros(discrete_shape)

    discrete_array[discrete_index] = 1

    return Variable(torch.from_numpy(discrete_array).type(torch.FloatTensor))


class DeepQModule(torch.nn.Module):
    def __init__(self, observation_dimensions, action_dimensions, hidden_dimensions):
        super().__init__()
        self.nonlinear = torch.nn.Sequential(
            torch.nn.Linear(observation_dimensions, hidden_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimensions, hidden_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimensions, action_dimensions),
        )
        self.classify = torch.nn.Softmax()

    def forward(self, observation):
        logits = self.nonlinear(observation)
        action = self.classify(logits)
        return action


if __name__ == "__main__":
    IMAGE_ASSET_DIR = "images"
    MODEL_ASSET_DIR = "models"

    pathlib.Path(IMAGE_ASSET_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(MODEL_ASSET_DIR).mkdir(parents=True, exist_ok=True)

    IMAGE_FILENAME = 'mountain_car__q_learning__loss-reward'
    MODEL_FILENAME = 'mountain_car__q_learning'

    episodes = 30
    steps = 1500

    eps = 0.1
    eps_decay = 0.01
    eps_limit = 0.01

    alpha = 0.0001
    gamma = 0.99

    hidden_dimensions = 32

    discrete_position_chunks = 50
    discrete_velocity_chunks = 25

    discrete_position_space = continious2discrete(
        continious_from=-1.2, continious_to=0.6, discrete_chunks=discrete_position_chunks)
    discrete_velocity_space = continious2discrete(
        continious_from=-0.07, continious_to=0.07, discrete_chunks=discrete_velocity_chunks)

    discrete_observation_dimensions = discrete_position_chunks * discrete_velocity_chunks
    discrete_action_dimensions = 3

    version = f"observations={discrete_observation_dimensions}xhidden={hidden_dimensions}"

    image_path = pathlib.Path(
        IMAGE_ASSET_DIR + "/" + IMAGE_FILENAME + "__" + version + ".png")
    model_path = pathlib.Path(
        MODEL_ASSET_DIR + "/" + MODEL_FILENAME + "__" + version)

    model = DeepQModule(discrete_observation_dimensions,
                        discrete_action_dimensions,
                        hidden_dimensions)

    if model_path.exists():
        model.load_state_dict(torch.load(model_path))

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    reward_history = []
    loss_history = []

    successes = 0

    for episode in range(episodes):
        print(f"Starting episode {episode}...")

        episode_reward = 0
        episode_loss = 0

        old_state, _ = env.reset()

        for step in range(steps):
            old_discrete_state = get_discrete_state(
                old_state,
                discrete_spaces=[
                    discrete_position_space,
                    discrete_velocity_space,
                ],
            )

            old_q = model.forward(old_discrete_state)

            if random() > eps:
                _, action = torch.max(old_q, -1)
                action = action.item()
            else:
                action = env.action_space.sample()

            new_state, reward, terminated, truncated, info = env.step(action)

            reward = reward + 300 * \
                (gamma * abs(new_state[1]) - abs(old_state[1]))

            new_discrete_state = get_discrete_state(
                new_state,
                discrete_spaces=[
                    discrete_position_space,
                    discrete_velocity_space,
                ],
            )

            new_q = model.forward(new_discrete_state)
            new_max_q, _ = torch.max(new_q, -1)

            target_q = old_q.clone()
            target_q = Variable(target_q.data)
            target_q[action] = reward + torch.mul(new_max_q.detach(), gamma)

            loss = loss_fn(old_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            for param in model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            episode_reward += reward
            episode_loss += loss.item()

            if terminated:
                successes += 1
                eps = max(eps_limit, eps - eps_decay)
                print(f"Flag is captured in episode {episode}!")
                print(f"Epsilon after decay: {eps}")
                break
            else:
                old_state = new_state

        reward_history.append(episode_reward)
        loss_history.append(episode_loss)

    env.close()

    print(f"Success rate: {successes} out of {episodes}")

    torch.save(model.state_dict(), model_path)

    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.plot(reward_history, label="Reward history")
    plt.plot(loss_history, label="Loss history")
    plt.legend()
    plt.savefig(image_path)
