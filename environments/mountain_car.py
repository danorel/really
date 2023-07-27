import gym

env = gym.make('MountainCar-v0', render_mode="human")

obs_space = env.observation_space
action_space = env.action_space

if __name__ == "__main__":
    print("The observation space: {}".format(obs_space))
    print("The action space: {}".format(action_space))
