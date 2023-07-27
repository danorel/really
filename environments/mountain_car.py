import gym
import time

env = gym.make('MountainCar-v0', render_mode="human")

obs_space = env.observation_space
action_space = env.action_space

print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

if __name__ == "__main__":
    steps = 1500
    obs = env.reset()

    for step in range(steps):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        time.sleep(0.01)

        if terminated:
            env.reset()

    env.close()
