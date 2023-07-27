import gym
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0', render_mode="rgb_array")

obs_space = env.observation_space
action_space = env.action_space

print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

obs_origin = env.reset()
print("The initial observation is {}".format(obs_origin))

env_screen = env.render()
plt.imsave("origin.png", env_screen)

random_action = env.action_space.sample()
print("Random action is {}".format(random_action))

env_screen = env.render()
plt.imsave("random-action.png", env_screen)

new_obs, reward, terminated, truncated, info = env.step(random_action)
print("The new observation is {}".format(new_obs))

env.close()
