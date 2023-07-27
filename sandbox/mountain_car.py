import time

from environments.mountain_car import env
from lib.q_learning import QLearningTable

if __name__ == "__main__":
    steps = 1500
    obs = env.reset()

    model = QLearningTable(actions=env.action_space,
                           states=env.observation_space)

    for step in range(steps):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        time.sleep(0.01)

        if terminated:
            env.reset()

    env.close()
