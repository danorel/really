import argparse
import copy
import random
import torch
import pathlib
import matplotlib.pyplot as plt

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

IMAGE_FILENAME = "mountain_car__deep_q_learning__loss-reward"
MODEL_FILENAME = "mountain_car__deep_q_learning"


class DeepQModule(torch.nn.Module):
    def __init__(self, observation_dimensions, action_dimensions, hidden_dimensions):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(observation_dimensions, hidden_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimensions, hidden_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimensions, action_dimensions),
        )

    def forward(self, observation):
        return self.network(observation)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory: list[tuple] = []
        self.capacity = capacity

    def push(self, experience: tuple):
        if len(self.memory) > self.capacity:
            self.memory.pop()
            self.memory.insert(0, experience)
        else:
            self.memory.append(experience)

    def sample(self, batch_size: int):
        return list(zip(*random.sample(self.memory, batch_size)))

    def __len__(self):
        return len(self.memory)


def make_fit(optimizer, loss_fn, hyperparameters: dict):
    gamma, reward_speed_coefficient = (
        hyperparameters.get("gamma"),
        hyperparameters.get("reward_speed_coefficient"),
    )
    assert gamma is not None, "decay rate was not defined"
    assert (
        reward_speed_coefficient is not None
    ), "reward speed coefficient was not defined"

    def fit(batch: list[tuple], model, target_model):
        old_observation, action, reward, new_observation, terminated = batch

        old_observation, new_observation, action, reward, terminated = (
            torch.tensor(old_observation).to(DEVICE).float(),
            torch.tensor(new_observation).to(DEVICE).float(),
            torch.tensor(action).to(DEVICE),
            torch.tensor(reward).to(DEVICE).float(),
            torch.tensor(terminated).to(DEVICE),
        )

        new_speed_tensor = new_observation[:, 1]
        old_speed_tensor = old_observation[:, 1]

        reward += reward_speed_coefficient * (
            gamma * abs(new_speed_tensor) - abs(old_speed_tensor)
        )

        old_actions = model.forward(old_observation)
        old_actions = old_actions.gather(1, action.unsqueeze(1))

        target_actions = torch.zeros(reward.size()[0]).float().to(DEVICE)
        with torch.no_grad():
            target_actions = target_model(new_observation).max(1)[0].view(-1)
            target_actions[terminated] = 0
        target_actions = reward + target_actions * gamma
        target_actions = target_actions.unsqueeze(1)

        loss = loss_fn(old_actions, target_actions)

        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        return reward.sum(), loss.item()

    return fit


def explore_or_exploit(old_observation, model, eps):
    if random.random() > eps:
        old_action = model(
            torch.tensor(old_observation).to(DEVICE).float().unsqueeze(0)
        )
        action = old_action[0].max(0)[1].view(1, 1).item()
    else:
        action = env.action_space.sample()
    return action


def train(
    model,
    target_model,
    replay_memory,
    fit,
    episodes: int,
    steps: int,
    eps_max: float,
    eps_min: float,
    c: float,
    batch_size: int,
    strict: bool = False,
    verbose: bool = False,
):
    reward_history = []
    loss_history = []

    successes = 0

    eps, eps_decay = (eps_max, (eps_max - eps_min) / episodes)

    for episode in range(episodes):
        if verbose:
            print(f"Starting episode {episode + 1}. Epsilon = {eps}")

        episode_reward = 0
        episode_loss = 0

        old_observation, _ = env.reset()

        for step in range(steps):
            action = explore_or_exploit(old_observation, model, eps)
            new_observation, reward, terminated, truncated, _ = env.step(
                action)
            replay_memory.push(
                (old_observation, action, reward, new_observation, terminated)
            )

            if terminated:
                successes += 1
                if verbose:
                    print(f"Flag is captured in episode {episode}!")
                break

            if strict and truncated:
                if verbose:
                    print(f"Timeout in episode {episode}!")
                break

            if step > batch_size:
                reward, loss = fit(
                    replay_memory.sample(batch_size), model, target_model
                )
                episode_reward += reward
                episode_loss += loss

            old_observation = new_observation

            if step % c == 0:
                target_model = copy.deepcopy(model)

        reward_history.append(episode_reward)
        loss_history.append(episode_loss)

        eps -= eps_decay

    env.close()

    report = dict(reward_history=reward_history, loss_history=loss_history)

    if verbose:
        print(f"Success rate: {successes} out of {episodes}")

    return successes, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running Deep Q-Learning algorithm on MountainCar problem."
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
        "--alpha", type=float, default=0.0001, help="learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--c",
        type=int,
        default=128,
        help="amount of iterations after which target Q-Learning model clones original Q-Learning weights (default: 128)",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=64000,
        help="capacity of experience memory (default: 64000)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount rate (default: 0.99)"
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
        default=10000,
        help="training steps per episode (default: 10000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="training batch size for tuning (default: 128)",
    )

    parser.add_argument(
        "--reward_speed_coefficient",
        type=float,
        default=300,
        help="reward trick (default: 300)",
    )
    parser.add_argument(
        "--hidden_dimensions",
        type=int,
        default=32,
        help="number of nodes in hidden dimension layer (default: 32)",
    )

    args = parser.parse_args()

    HIDDEN_DIMENSIONS = args.hidden_dimensions

    MODEL_VERSION = f"hidden_dimensions={HIDDEN_DIMENSIONS}"

    IMAGE_PATH = pathlib.Path(
        IMAGE_ASSET_DIR + "/" + IMAGE_FILENAME + "__" + MODEL_VERSION + ".png"
    )
    MODEL_PATH = pathlib.Path(
        MODEL_ASSET_DIR + "/" + MODEL_FILENAME + "__" + MODEL_VERSION
    )

    model = DeepQModule(OBSERVATION_DIMENSIONS,
                        ACTION_DIMENSIONS, HIDDEN_DIMENSIONS)

    if MODEL_PATH.exists():
        model_state_dict = torch.load(MODEL_PATH)
        model.load_state_dict(model_state_dict)

    target_model = copy.deepcopy(model)

    model.to(DEVICE)
    target_model.to(DEVICE)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.alpha)

    replay_memory = ReplayMemory(capacity=args.capacity)
    fit = make_fit(
        optimizer,
        loss_fn,
        hyperparameters=dict(
            gamma=args.gamma, reward_speed_coefficient=args.reward_speed_coefficient
        ),
    )

    successes, report = train(
        model,
        target_model,
        replay_memory,
        fit,
        episodes=args.episodes,
        steps=args.steps,
        eps_min=args.epsilon_min,
        eps_max=args.epsilon_max,
        c=args.c,
        batch_size=args.batch_size,
        strict=args.strict,
        verbose=args.verbose,
    )

    torch.save(model.state_dict(), MODEL_PATH)

    plt.xlabel("Episode")
    plt.plot(report.get("reward_history"), label="Reward history")
    plt.plot(report.get("loss_history"), label="Loss history")
    plt.legend()
    plt.savefig(IMAGE_PATH)
