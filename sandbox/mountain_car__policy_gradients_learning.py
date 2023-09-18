import argparse
import gym
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
import pathlib

from gym import Env

from environments.mountain_car import env

OBSERVATION_DIMENSIONS = 2
ACTION_DIMENSIONS = 3
HIDDEN_DIMENSIONS = 16

DEVICE = torch.device("cpu")

IMAGE_ASSET_DIR = "images"
MODEL_ASSET_DIR = "models"

pathlib.Path(IMAGE_ASSET_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(MODEL_ASSET_DIR).mkdir(parents=True, exist_ok=True)

IMAGE_FILENAME = "mountain_car__policy_gradients_learning__loss-reward"
MODEL_FILENAME = "mountain_car__policy_gradients_learning"


class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        logits = self.fc(state)
        action = F.softmax(logits)
        return action

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        action_probabilities = self.forward(state)
        action_vector = distributions.Categorical(action_probabilities)
        action = action_vector.sample()
        return action.item(), action_vector.log_prob(action)


def reinforce(env: Env,
              policy: Policy,
              optimizer: optim.Adam,
              episodes: int,
              steps: int,
              gamma: float,
              print_every: int = 10,
              strict: bool = False,
              verbose: bool = False,
              trial=None):
    is_optimization = trial is not None

    scores = []

    for episode in range(1, episodes + 1):
        state = env.reset()[0]

        epoch_rewards, log_probabilities = [], []

        for step in range(1, steps + 1):
            action, log_prob = policy.act(state)
            state, reward, terminated, truncated, info = env.step(action)
            epoch_rewards.append(reward)
            log_probabilities.append(log_prob)

            if terminated:
                if verbose:
                    print(f"Flag is captured in episode {episode}!")
                break

            if strict and truncated:
                if verbose:
                    print(f"Timeout in episode {episode}!")
                break

        score = sum(epoch_rewards)
        scores.append(score)

        T = len(epoch_rewards)

        returns, cummulated_gamma = [0] * T, 1
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                returns[t] = epoch_rewards[t]
            else:
                returns[t] = epoch_rewards[t] * \
                    cummulated_gamma + returns[t + 1]
            cummulated_gamma *= gamma
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        returns = returns.flip((0,))

        loss = []
        for cumulated_epoch_reward, log_probability in zip(returns, log_probabilities):
            loss.append(-log_probability * cumulated_epoch_reward)
        loss = torch.cat(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_optimization:
            trial.report(score, episode)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if episode % print_every == 0:
            print(f"Episode = {episode}, Loss = {loss}, Reward = {score}")

    return scores


def optimize(args):
    default_kwargs = {
        "state_size": OBSERVATION_DIMENSIONS,
        "action_size": ACTION_DIMENSIONS
    }

    def sample_kwargs(trial):
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        gamma = 1 - trial.suggest_loguniform("gamma", 1e-5, 1e-1)
        hidden_size = 2 ** trial.suggest_int("hidden_size", 3, 5)

        return {
            "lr": lr,
            "gamma": gamma,
            "hidden_size": hidden_size
        }

    def objective(trial):
        kwargs = default_kwargs.copy()
        kwargs.update(sample_kwargs(trial))

        # Generate the model.
        policy = Policy(state_size=kwargs['state_size'],
                        action_size=kwargs['action_size'],
                        hidden_size=kwargs['hidden_size'])

        optimizer = optim.Adam(policy.parameters(), lr=kwargs['lr'])

        optimization_env = gym.make("MountainCar-v0", render_mode="rgb_array")

        scores = reinforce(optimization_env,
                           policy,
                           optimizer,
                           episodes=args.episodes,
                           steps=args.steps,
                           gamma=kwargs['gamma'],
                           trial=trial)

        score = np.mean(scores)

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    pruned_trials = [t for t in study.trials if t.state ==
                     optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state ==
                       optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def train(args):
    policy = Policy(state_size=OBSERVATION_DIMENSIONS,
                    action_size=ACTION_DIMENSIONS,
                    hidden_size=args.hidden_size)

    optimizer = optim.Adam(policy.parameters(), lr=args.alpha)

    scores = reinforce(env,
                       policy,
                       optimizer,
                       episodes=args.episodes,
                       steps=args.steps,
                       gamma=args.gamma,
                       strict=args.strict,
                       verbose=args.verbose)

    print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running Policy Gradients algorithm on MountainCar problem."
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
        '--optimize',
        type=bool,
        default=False,
        help="search optimal hyperparameters (default: False)",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount rate (default: 0.998)"
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="training episodes (default: 1000)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="training steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=16,
        help="number of neurons in the hidden layer (default: 16)"
    )

    args = parser.parse_args()

    if args.optimize:
        optimize(args)
    else:
        train(args)
