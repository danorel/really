import gym

from gym.utils.play import PlayPlot, play


def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    return [
        rew,
    ]


plotter = PlayPlot(callback, 30 * 5, ["reward"])

env = gym.make("MountainCar-v0", render_mode="rgb_array")

keys_to_action = {("a"): 0, ("s"): 1, ("d"): 2}

if __name__ == "__main__":
    env.reset()
    env.render()
    print("The observation space: {}".format(env.observation_space))
    print("The action space: {}".format(env.action_space))
    play(env, callback=plotter.callback, keys_to_action=keys_to_action)
