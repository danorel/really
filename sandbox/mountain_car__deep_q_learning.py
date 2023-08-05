import copy
import torch
import pathlib
import matplotlib.pyplot as plt

from random import random

from environments.mountain_car import env


EPISODES = 30
STEPS = 1500

C = 10

EPS_MAX = 0.01
EPS_MIN = 0.0

ALPHA = 0.0001
GAMMA = 0.99
SPEED_REWARD_FACTOR = 300

OBSERVATION_DIMENSIONS = 2
ACTION_DIMENSIONS = 3
HIDDEN_DIMENSIONS = 32

DEVICE = torch.device("cpu")

IMAGE_ASSET_DIR = "images"
MODEL_ASSET_DIR = "models"

pathlib.Path(IMAGE_ASSET_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(MODEL_ASSET_DIR).mkdir(parents=True, exist_ok=True)

IMAGE_FILENAME = 'mountain_car__deep_q_learning__loss-reward'
MODEL_FILENAME = 'mountain_car__deep_q_learning'
MODEL_VERSION = f"hidden={HIDDEN_DIMENSIONS}"

IMAGE_PATH = pathlib.Path(IMAGE_ASSET_DIR + "/" +
                          IMAGE_FILENAME + "__" + MODEL_VERSION + ".png")
MODEL_PATH = pathlib.Path(MODEL_ASSET_DIR + "/" +
                          MODEL_FILENAME + "__" + MODEL_VERSION)


class DeepQModule(torch.nn.Module):
    def __init__(self, observation_dimensions, action_dimensions, hidden_dimensions):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(observation_dimensions, hidden_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimensions, action_dimensions),
        )

    def forward(self, observation):
        return self.network(observation)


class ReplayMemory():
    def __init__(self):
        pass


def create_model():
    model = DeepQModule(OBSERVATION_DIMENSIONS,
                        ACTION_DIMENSIONS,
                        HIDDEN_DIMENSIONS)

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH))

    target_model = copy.deepcopy(model)

    model.to(DEVICE)
    target_model.to(DEVICE)

    return model, target_model


if __name__ == "__main__":
    model, target_model = create_model()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)

    reward_history = []
    loss_history = []

    successes = 0

    eps, eps_decay = (
        EPS_MAX,
        (EPS_MAX - EPS_MIN) / EPISODES
    )

    for episode in range(EPISODES):
        print(f"Starting episode {episode}. Epsilon = {eps}")

        episode_reward = 0
        episode_loss = 0

        old_state, _ = env.reset()

        for step in range(STEPS):
            old_tensor = torch.tensor(old_state).to(DEVICE)
            old_q = model.forward(old_tensor)

            if random() > eps:
                _, old_max_index = torch.max(old_q, -1)
                action = old_max_index.item()
            else:
                action = env.action_space.sample()

            new_state, reward, terminated, truncated, info = env.step(action)

            _, new_speed = new_state
            _, old_speed = old_state

            reward += SPEED_REWARD_FACTOR * \
                (GAMMA * abs(new_speed) - abs(old_speed))

            new_tensor = torch.tensor(new_state).to(DEVICE)

            new_q = model.forward(new_tensor)
            new_max_q, _ = torch.max(new_q, -1)

            target_q = target_model.forward(new_tensor)
            target_q[action] = reward + torch.mul(new_max_q.detach(), GAMMA)

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
                print(f"Flag is captured in episode {episode}!")
                break
            else:
                old_state = new_state

            if step % C == 0:
                target_model = copy.deepcopy(model)

        reward_history.append(episode_reward)
        loss_history.append(episode_loss)

        eps -= eps_decay

    env.close()

    print(f"Success rate: {successes} out of {EPISODES}")

    torch.save(model.state_dict(), MODEL_PATH)

    plt.xlabel("Episode")
    plt.plot(reward_history, label="Reward history")
    plt.plot(loss_history, label="Loss history")
    plt.legend()
    plt.savefig(IMAGE_PATH)
