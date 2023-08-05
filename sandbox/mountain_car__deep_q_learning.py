import argparse
import copy
import torch
import pathlib
import matplotlib.pyplot as plt

from random import random

from environments.mountain_car import env


EPS_MAX = 0.01
EPS_MIN = 0.0

OBSERVATION_DIMENSIONS = 2
ACTION_DIMENSIONS = 3

DEVICE = torch.device("cpu")

IMAGE_ASSET_DIR = "images"
MODEL_ASSET_DIR = "models"

pathlib.Path(IMAGE_ASSET_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(MODEL_ASSET_DIR).mkdir(parents=True, exist_ok=True)

IMAGE_FILENAME = 'mountain_car__deep_q_learning__loss-reward'
MODEL_FILENAME = 'mountain_car__deep_q_learning'


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


def run(model,
        target_model,
        optimizer,
        loss_fn,
        episodes: int,
        steps: int,
        eps_max: float,
        eps_min: float,
        c: float,
        gamma: float,
        speed_reward_factor: float
        ):
    reward_history = []
    loss_history = []

    successes = 0

    eps, eps_decay = (
        eps_max,
        (eps_max - eps_min) / episodes
    )

    for episode in range(episodes + 1):
        print(f"Starting episode {episode}. Epsilon = {eps}")

        episode_reward = 0
        episode_loss = 0

        old_state, _ = env.reset()

        for step in range(steps):
            old_tensor = torch.tensor(old_state).to(DEVICE)
            old_q = model.forward(old_tensor)

            if random() > eps:
                _, old_max_index = torch.max(old_q, -1)
                action = old_max_index.item()
            else:
                action = env.action_space.sample()

            new_state, reward, terminated, _, _ = env.step(action)

            _, new_speed = new_state
            _, old_speed = old_state

            reward += speed_reward_factor * \
                (gamma * abs(new_speed) - abs(old_speed))

            new_tensor = torch.tensor(new_state).to(DEVICE)

            new_q = model.forward(new_tensor)
            new_max_q, _ = torch.max(new_q, -1)

            target_q = target_model.forward(new_tensor)
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
                print(f"Flag is captured in episode {episode}!")
                break
            else:
                old_state = new_state

            if step % c == 0:
                target_model = copy.deepcopy(model)

        reward_history.append(episode_reward)
        loss_history.append(episode_loss)

        eps -= eps_decay

    env.close()

    report = dict(reward_history=reward_history,
                  loss_history=loss_history)

    print(f"Success rate: {successes} out of {episodes}")

    return successes, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Running Deep Q-Learning algorithm on MountainCar problem.')

    parser.add_argument('--alpha', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--c', type=int, default=10,
                        help='amount of iterations after which target Q-Learning model clones original Q-Learning weights (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount rate (default: 0.99)')
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
    parser.add_argument('--hidden_dimensions', type=int, default=32,
                        help='number of nodes in hidden dimension layer (default: 32)')

    args = parser.parse_args()

    HIDDEN_DIMENSIONS = args.hidden_dimensions

    MODEL_VERSION = f"hidden_dimensions={HIDDEN_DIMENSIONS}"

    IMAGE_PATH = pathlib.Path(IMAGE_ASSET_DIR + "/" +
                              IMAGE_FILENAME + "__" + MODEL_VERSION + ".png")
    MODEL_PATH = pathlib.Path(MODEL_ASSET_DIR + "/" +
                              MODEL_FILENAME + "__" + MODEL_VERSION)

    model = DeepQModule(OBSERVATION_DIMENSIONS,
                        ACTION_DIMENSIONS,
                        HIDDEN_DIMENSIONS)

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH))

    target_model = copy.deepcopy(model)

    model.to(DEVICE)
    target_model.to(DEVICE)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.alpha)

    successes, report = run(model,
                            target_model,
                            optimizer,
                            loss_fn,
                            episodes=args.episodes,
                            steps=args.steps,
                            eps_min=args.epsilon_min,
                            eps_max=args.epsilon_max,
                            c=args.c,
                            gamma=args.gamma,
                            speed_reward_factor=args.speed_reward_factor)

    torch.save(model.state_dict(), MODEL_PATH)

    plt.xlabel("Episode")
    plt.plot(report.get('reward_history'), label="Reward history")
    plt.plot(report.get('loss_history'), label="Loss history")
    plt.legend()
    plt.savefig(IMAGE_PATH)
